"""Analyst UI over saved results only (read-only: no training, no writes). Run: streamlit run app/streamlit_app.py"""
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))  # for threat_map.py next to this file
from itgnn import ui_data as U
from itgnn.neo4j_ui import auto_source, browser_link, load_queries
from itgnn.ui_graph import build_payload
from itgnn.ui_theme import theme_for

RESULTS = os.getenv("ITGNN_RESULTS", "results")
NAV = ["Overview", "Threat map", "Alerts", "Investigate", "What-if", "Rigor and limits"]

st.set_page_config(page_title="Insider threat analyst view", layout="wide")
need = ["ui_scores.parquet", "ui_weeks.parquet", "ui_edges.parquet", "ui_metrics.json"]
if [f for f in need if not (Path(RESULTS) / f).exists()]:
    st.error(f"Missing results in `{RESULTS}/`: {[f for f in need if not (Path(RESULTS) / f).exists()]}. Create them with:")
    st.code("python -m itgnn.ui_export all")
    st.stop()


@st.cache_data
def _scores(path):
    return U.load_scores(path)


@st.cache_data
def _weeks(path):
    return pd.read_parquet(Path(path) / "ui_weeks.parquet")


@st.cache_data
def _metrics(path):
    return U.load_metrics(path)


@st.cache_resource
def _source(path):
    return auto_source(path)


@st.cache_data(show_spinner="Building the threat map...")
def _payload(path, backend, eval_view):
    """Graph payload for the map. Labels (insider flags) enter ONLY when the evaluation view is on."""
    src, _ = _source(path)
    insiders = None
    if eval_view:
        ev = U.load_eval(path)
        insiders = set(ev.loc[ev["scenario"] > 0, "user"])
    return build_payload(_scores(path), _weeks(path), src.all_edges(), insiders)


scores, weeks, metrics = _scores(RESULTS), _weeks(RESULTS), _metrics(RESULTS)
source, backend_note = _source(RESULTS)
mode = getattr(st.context.theme, "type", None) or "dark"
PAL = theme_for(mode)

# ---- navigation and shared selection -----------------------------------------------------------
if "sel_user" not in st.session_state:
    st.session_state.sel_user, st.session_state.sel_week = None, None


def goto(page, user=None, week=None):
    """Callback: switch page (and optionally the selected user-week). Runs before the next rerun, so the nav widget picks it up."""
    st.session_state.nav = page
    if user is not None:
        st.session_state.sel_user, st.session_state.sel_week = user, pd.Timestamp(week)
    st.session_state.map_focus = {"user": user, "week": str(pd.Timestamp(week).date())} if (page == "Threat map" and user is not None) else None


def _map_open():
    """Trigger from the map's details panel: open Investigate on that user-week."""
    ev = next((st.session_state[k] for k in st.session_state if str(k).startswith("threat_map_")), None)
    v = getattr(ev, "open", None) if ev is not None else None
    if v:
        goto("Investigate", v["user"], v["week"])


st.title("Insider threat analyst view")
nav = st.segmented_control("Navigation", NAV, default="Overview", key="nav", label_visibility="collapsed")
page = nav or st.session_state.get("_page", "Overview")
st.session_state["_page"] = page

with st.sidebar:
    st.header("Controls")
    q = st.radio("Alert budget (share of scored weeks)", U.BUDGETS, index=1, format_func=lambda x: f"{x}%")
    cw = sorted(scores["cal_week"].unique())
    start = scores.groupby("cal_week")["week_start"].min()
    wk = st.selectbox("Calendar week", ["All weeks (global ranking)"] + cw,
                      format_func=lambda x: x if isinstance(x, str) else f"week {int(x)} ({start[x]:%Y-%m-%d})")
    wk = None if isinstance(wk, str) else wk
    eval_view = st.checkbox("Evaluation view (uses ground-truth labels)", value=False)
    if eval_view:
        st.error("EVALUATION VIEW: ground-truth labels and attack windows are shown. Analysts never see these.")
    st.caption(f"Neighbourhood backend: {backend_note}")

ev = U.load_eval(RESULTS) if (eval_view and page in ("Alerts", "Investigate")) else None  # labels are loaded only here and in the map payload, only on request


def page_overview():
    st.subheader("Overview")
    st.info("The Overview (headline results, live alert budget) and the What-if scorer arrive in the next build stage. "
            "This stage delivers the interactive Threat map; Alerts, Investigate and Rigor are the earlier versions, unchanged.")


def page_map():
    st.subheader("Threat map")
    st.caption("Flagged user-weeks (the top 2% of all scored weeks) and what they touched. Week view: one calendar week at a time. "
               "Campaign view: flagged users linked through websites, PCs and email shared by at least two of them. "
               "Risk colours follow the alert budgets; named-site categories are labelled.")
    payload, stats = _payload(RESULTS, source.name, bool(eval_view))
    import threat_map
    threat_map.render(payload, PAL, on_open=_map_open, focus=st.session_state.get("map_focus"))
    c = payload["campaign"]
    st.caption(f"{stats['user_weeks_drawn']:,} flagged user-weeks, {stats['connectors']:,} websites/PCs/contacts, {stats['edges']:,} edges in total; the week view draws only the selected week. "
               f"Campaign view: {len(c['users'])} users, {c['modularity']} modularity ({len(c['clusters'])} communities of at least 3): weak structure means the clusters are suggestive, not proof.")
    qs = load_queries()
    su, sw = st.session_state.sel_user, st.session_state.sel_week
    links = []
    if su is not None:
        links.append(("Neighbourhood of the selected user-week", browser_link(qs["neighbourhood"]["cypher"], key=U.week_key(su, sw))))
    links += [("All visitors of wikileaks.org", browser_link(qs["visitors"]["cypher"], domain="wikileaks.org")), ("Campaign query (shared websites/PCs)", browser_link(qs["campaign"]["cypher"]))]
    st.markdown("Open in Neo4j Browser (log in with the password from `.env`): " + " · ".join(f"[{t}]({u})" for t, u in links))


def page_alerts():
    alerts = U.rank_alerts(scores, q, wk)
    scope = "all weeks" if wk is None else f"calendar week {int(wk)}"
    st.subheader(f"Top {q}% of weeks ({scope}): {len(alerts):,} alerts, {alerts['user'].nunique()} distinct users")
    if wk is None:
        st.caption("Ranked globally over every scored user-week, which matches the reported metrics. A global top-q% is easier than a real weekly queue: pick a calendar week in the sidebar to see one.")
    if not len(alerts):
        st.info("No scored weeks for this selection."); return
    t = U.alert_table(alerts)
    if ev is not None:
        t = t.merge(ev[["user", "week_start", "label", "scenario"]].rename(columns={"label": "attack week (truth)", "scenario": "scenario (truth)"}), on=["user", "week_start"], how="left")
    st.dataframe(t.rename(columns={"score": "score (W2+E+H)", "score_sd": "sd over 5 repeats", "score_generic": "score (generic features)"}),
                 hide_index=True, width="stretch", column_config={"week_start": st.column_config.DateColumn("week", format="YYYY-MM-DD")})
    st.caption("The generic score uses no named-site (category-domain) features, so it shows the evidence that does not rely on the near-label markers.")
    i = st.selectbox("Open an alert", range(len(alerts)), format_func=lambda i: f"{alerts.loc[i, 'user']}  {alerts.loc[i, 'week_start']:%Y-%m-%d}  score {alerts.loc[i, 'score']:.3f}")
    u, w = alerts.loc[i, "user"], alerts.loc[i, "week_start"]
    st.session_state.sel_user, st.session_state.sel_week = u, w
    a, b, _ = st.columns([1, 1, 4])
    a.button("Open in Investigate", on_click=goto, args=("Investigate", u, w))
    b.button("Open in Threat map", on_click=goto, args=("Threat map", u, w))


def page_investigate():
    users = sorted(scores["user"].unique())
    su = st.session_state.sel_user
    user = st.selectbox("User", users, index=users.index(su) if su in users else 0)
    tl = U.user_timeline(scores, user)
    weeks_l = list(tl["week_start"])
    sw = st.session_state.sel_week
    pick = st.selectbox("Week", weeks_l, index=weeks_l.index(sw) if (su == user and sw in weeks_l) else len(weeks_l) - 1, format_func=lambda d: f"{d:%Y-%m-%d}")
    row = scores[(scores["user"] == user) & (scores["week_start"] == pick)].iloc[0]
    cuts = U.budget_cutoffs(scores)
    c1, c2, c3 = st.columns(3)
    c1.metric("Score (W2+E+H)", f"{row['score']:.3f}", f"sd {row['score_sd']:.3f} over 5 repeats", delta_color="off")
    c2.metric("Score (generic features)", f"{row['score_generic']:.3f}")
    c3.metric("Within the top", next((f"{b}%" for b in U.BUDGETS if row["score"] >= cuts[b]), "more than 2% of weeks (not alerted)"))
    long = pd.concat([tl.assign(series="W2+E+H score")[["week_start", "score", "series"]], tl.assign(series="generic-feature score", score=tl["score_generic"])[["week_start", "score", "series"]]])
    colour = alt.Color("series:N", scale=alt.Scale(domain=["W2+E+H score", "generic-feature score"], range=[PAL["accent"], PAL["muted"]]), legend=alt.Legend(title=None, orient="top"))
    base = alt.Chart(long).encode(x=alt.X("week_start:T", title="week"), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="score (out-of-fold, mean of 5 repeats)"))
    layers = [base.mark_line(strokeWidth=2).encode(color=colour), base.mark_point(size=28, filled=True).encode(color=colour)]
    cut = pd.DataFrame({"cutoff": list(cuts.values()), "budget": [f"top {b}%" for b in cuts]})
    layers.append(alt.Chart(cut).mark_rule(strokeDash=[4, 4], color=PAL["muted"]).encode(y="cutoff:Q", tooltip=["budget", "cutoff"]))
    layers.append(alt.Chart(pd.DataFrame({"week_start": [pick], "score": [row["score"]]})).mark_point(size=160, color=PAL["text"], filled=False, strokeWidth=2).encode(x="week_start:T", y="score:Q"))
    if ev is not None:
        e = ev[ev["user"] == user]
        if e["window_start"].notna().any():
            layers.insert(0, alt.Chart(pd.DataFrame({"s": [e["window_start"].iloc[0]], "e": [e["window_end"].iloc[0]]})).mark_rect(color=PAL["risk"][3], opacity=0.15).encode(x="s:T", x2="e:T"))
            st.error(f"EVALUATION VIEW: shaded band = true attack window {e['window_start'].iloc[0]:%Y-%m-%d} to {e['window_end'].iloc[0]:%Y-%m-%d} (scenario {int(e['scenario'].iloc[0])}). Not available to analysts.")
        else:
            st.error("EVALUATION VIEW: this user is not a listed insider.")
    st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
    st.caption("Dashed lines: score cut-offs of the four global alert budgets. Weeks before a user's fifth active week are not scored.")
    g = pd.DataFrame({"group": list(U.GROUP_COLS.values()), "contribution (log-odds)": [float(row[c]) for c in U.GROUP_COLS]})
    st.subheader(f"Why this week scored as it did ({pick:%Y-%m-%d})")
    st.altair_chart(alt.Chart(g).mark_bar().encode(y=alt.Y("group:N", sort="-x", title=None), x=alt.X("contribution (log-odds):Q"),
                                                   color=alt.condition(alt.datum["contribution (log-odds)"] > 0, alt.value(PAL["positive"]), alt.value(PAL["negative"]))).properties(height=260), width="stretch")
    st.caption(f"Positive bars push the score up. Baseline (bias) {float(row['bias']):+.2f} log-odds. XGBoost TreeSHAP contributions grouped into feature groups; not causal.")
    st.button("Show this user in the Threat map", on_click=goto, args=("Threat map", user, pick))


def page_whatif():
    st.subheader("What-if scorer")
    st.info("The what-if scorer (sliders, presets and a sensitivity sweep, computed live from a final model) arrives in the next build stage.")


def page_rigor():
    T = U.metrics_tables(metrics)
    st.subheader("What these alert lists achieve")
    st.dataframe(T["lists"].reset_index(), hide_index=True, width="stretch")
    st.caption("A global top-q% of all weeks is easier than a real weekly queue. Both lists use the out-of-fold scores shown in this app; insiders caught are out of 70.")
    st.subheader("PR-AUC, 25 folds (5x5 user-split CV), XGBoost")
    st.dataframe(T["pr_auc"], width="stretch")
    st.dataframe(T["paired"], width="stretch")
    st.caption("W2 = weekly counts and history; +E+H = email and web features; generic = without the named-site (category-domain) features; calendar = a negative control (calendar week as a feature).")
    st.subheader("Alert budget, 25 folds")
    st.dataframe(T["budget"], hide_index=True, width="stretch")
    st.subheader("Time-based evaluation (train on weeks <= T, test later)")
    st.dataframe(T["time_split"], hide_index=True, width="stretch")
    st.subheader("Unseen users and future weeks")
    st.dataframe(T["double_holdout"], hide_index=True, width="stretch")
    st.subheader("Limitations")
    for line in U.limitations(metrics):
        st.markdown(f"- {line}")


{"Overview": page_overview, "Threat map": page_map, "Alerts": page_alerts, "Investigate": page_investigate, "What-if": page_whatif, "Rigor and limits": page_rigor}[page]()
