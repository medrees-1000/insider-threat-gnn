"""Analyst UI over saved results only (read-only: no training, no writes). Run: streamlit run app/streamlit_app.py"""
import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))  # for threat_map.py next to this file
from itgnn import ui_data as U
from itgnn import whatif as W
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
def _scores_p(path):
    return U.with_percentiles(U.load_scores(path))


@st.cache_data
def _weeks(path):
    return pd.read_parquet(Path(path) / "ui_weeks.parquet")


@st.cache_data
def _metrics(path):
    return U.load_metrics(path)


@st.cache_data
def _json(path, name):
    p = Path(path) / name
    return json.load(open(p)) if p.exists() else None


@st.cache_data
def _final_rows(path):
    return pd.read_parquet(Path(path) / "final_model_rows.parquet")


@st.cache_resource
def _models(path):
    return W.load_models(path)


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

st.markdown(f"""<style>
.st-key-wi_row [data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
.st-key-wi_row [data-testid="stColumn"] {{ align-self: stretch; }}
.st-key-wi_row [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ height: 100%; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-wi_rating) {{ position: sticky; top: 3.75rem; z-index: 30; background: {PAL['page']}; padding-bottom: 0.4rem; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
</style>""", unsafe_allow_html=True)

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
    curve, facts, m = _json(RESULTS, "ui_budget_curve.json"), _json(RESULTS, "ui_rigor_facts.json"), metrics
    if curve is None or facts is None:
        st.error("The Overview needs `ui_budget_curve.json` and `ui_rigor_facts.json`. Create them with `python -m itgnn.ui_export curve` and `python -m itgnn.ui_export facts`."); return
    st.markdown("### Finding insiders from everyday activity")
    st.write("Each week, every person's logons, USB use, file copies, email and web browsing are scored for how unusual and risky they look. "
             "The riskiest few weeks become alerts. All results below come from people the model had never seen when it scored them. The data is synthetic.")
    g, wq, a25 = m["ui_global_lists"]["0.25"], m["ui_weekly_queue_lists"]["0.25"], m["alert_budget_25fold"]["W2+E+H"]["0.25"]
    pr, ts = m["pr_auc_25fold"], {r["T"]: r for r in m["time_split"] if r["set"] == "W2+E+H"}
    base_rate = m["attack_weeks"] / m["n_scored_weeks"]
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True, height=360):
        st.caption("Insiders caught at the tightest budget")
        st.markdown(f"## {g['insiders caught']} of {curve['n_insiders']}")
        st.write(f"with **{g['normal users flagged']}** of {curve['n_normal_users']} normal users flagged; {g['alerts']} alerts, {g['week precision']:.0%} of them real attack weeks.")
        st.caption(f"Across 25 train/test splits: {a25['caught'][0]:.1f} ± {a25['caught'][1]:.1f} caught, {a25['normal flagged'][0]:.1f} ± {a25['normal flagged'][1]:.1f} normal users flagged. "
                   f"This is a global top-0.25% list over the whole period, which is easier than a real weekly queue (a weekly queue of the same size catches {wq['insiders caught']} and flags {wq['normal users flagged']}).")
    with c2.container(border=True, height=360):
        st.caption("How well attack weeks rank above ordinary weeks")
        st.markdown(f"## {pr['W2+E+H']['pr_auc_mean']:.2f} vs {pr['W2']['pr_auc_mean']:.2f}")
        st.write("final feature set versus the first baseline (weekly counts and history only).")
        st.caption(f"1.0 would be a perfect ranking; guessing would score about {base_rate:.3f}. Spread over 25 splits: ± {pr['W2+E+H']['pr_auc_std']:.2f}.")
    with c3.container(border=True, height=360):
        st.caption("Does it hold up on weeks it has not seen?")
        st.markdown(f"## {ts[39]['R']:.0%} to {ts[49]['R']:.0%}")
        st.write("of its accuracy is kept when trained only on earlier weeks and tested on later ones, once enough history exists.")
        st.caption(f"At the earliest cutoff, with few attacks in the training period, it keeps {ts[29]['R']:.0%}. The comparison is the same weeks scored in the main evaluation.")
    st.subheader("Try an alert budget")
    grid = curve["budgets"]
    q0 = st.select_slider("Share of all weeks that may be flagged as alerts", options=grid, value=0.5, format_func=lambda x: f"{x}%")
    i = grid.index(q0)
    sets = {"Final model": "W2+E+H", "Without named-site features": "W2+E+H generic", "First baseline": "W2"}
    k1, k2, k3 = st.columns(3)
    f = curve["sets"]["W2+E+H"]
    k1.metric("Real insiders caught (final model)", f"{f['caught'][i]:.1f} of {curve['n_insiders']}", f"± {f['caught_sd'][i]:.1f} over 5 repeats", delta_color="off")
    k2.metric("Normal users flagged by mistake", f"{f['normal_flagged'][i]:.1f} of {curve['n_normal_users']}", f"± {f['normal_flagged_sd'][i]:.1f}", delta_color="off")
    k3.metric("Weeks flagged", f"{f['weeks_flagged'][i]:,.0f}", f"of {curve['n_scored_weeks']:,} scored weeks", delta_color="off")
    rows = [{"model": name, "budget": b, "normal users flagged": curve["sets"][key]["normal_flagged"][j], "insiders caught": curve["sets"][key]["caught"][j], "selected": b == q0}
            for name, key in sets.items() for j, b in enumerate(grid)]
    df = pd.DataFrame(rows)
    colour = alt.Color("model:N", scale=alt.Scale(domain=list(sets), range=[PAL["accent"], PAL["text2"], PAL["muted"]]), legend=alt.Legend(title=None, orient="top", labelLimit=300))
    dash = alt.StrokeDash("model:N", scale=alt.Scale(domain=list(sets), range=[[1, 0], [1, 0], [5, 4]]), legend=None)
    base = alt.Chart(df).encode(x=alt.X("normal users flagged:Q", title="Normal users flagged by mistake (out of %d)" % curve["n_normal_users"]),
                                y=alt.Y("insiders caught:Q", scale=alt.Scale(domain=[0, curve["n_insiders"]]), title="Real insiders caught (out of %d)" % curve["n_insiders"]))
    chart = base.mark_line(strokeWidth=2).encode(color=colour, strokeDash=dash) + base.mark_circle(size=34).encode(color=colour, tooltip=["model", "budget", "insiders caught", "normal users flagged"]) \
        + base.transform_filter(alt.datum.selected).mark_point(size=190, filled=False, strokeWidth=2.5).encode(color=colour)
    st.altair_chart(chart.properties(height=330), width="stretch")
    st.caption("**How to read this:** each line follows one model as the alert budget grows; every dot is one budget. Going up means more real insiders caught, going right means more innocent people flagged, so the best lines are high and to the left. "
               "The ringed dots are the budget you chose. The final model reaches nearly all insiders before it flags many normal users; the first baseline needs far more false alarms to catch the same number.")
    st.subheader("How we know it works")
    t, cv, ca, gr = facts["tenure_leak"], facts["cross_validation"], facts["causality_check"], facts["graph_controls"]
    items = [
        f"**It caught a shortcut in our first model.** Staying in the data for the whole period alone separated people ({t['normal_users_covering_the_full_period']:.0%} of normal users did, {t['insiders_covering_the_full_period']:.0%} of insiders), because insiders leave early. We stopped using whole-period totals.",
        f"**Every score is from people the model never saw.** {cv['folds']} train/test splits ({cv['repeats']} repeats of 5); people shared between the two sides: {cv['max_users_shared_by_train_and_test']}.",
        f"**Features only use the past.** We rebuilt {ca['user_weeks_checked']} weeks from data cut at that week's end: the biggest difference was {ca['max_abs_difference']:g} across {ca['features_compared']} features, and the same check does fail when we cut mid-week ({ca['control_rows_that_differ']} of {ca['control_rows']} test rows changed).",
        f"**A time-based test.** Trained on earlier weeks and tested on later ones, it keeps {ts[39]['R']:.0%} to {ts[49]['R']:.0%} of its accuracy once enough history exists, and {ts[29]['R']:.0%} at the earliest cutoff.",
        (f"**Graph models added nothing.** Scored {gr['graphsage']:.2f} with connections, {gr['no_edge_mlp']:.2f} without, {gr['graphsage_shuffled_edges']:.2f} with scrambled connections, against {gr['xgboost_same_features']:.2f} for the simple model we use." if gr else "**Graph models** were tested against controls (results not present in this clone)."),
    ]
    for n, line in enumerate(items):
        a, b = st.columns([6, 1])
        a.markdown(line)
        b.button("Details", key=f"det_{n}", on_click=goto, args=("Rigor and limits",))
    st.caption("Synthetic data (CERT r4.2). The Rigor and limits page has the full tables and what this does not show.")


def page_map():
    st.subheader("Threat map")
    st.caption("Flagged user-weeks (the top 2% of all scored weeks) and what they touched. Week view: one calendar week at a time. "
               "Campaign view: flagged users linked through websites, PCs and email shared by at least two of them. "
               "Risk colours follow the alert budgets; named-site categories are labelled.")
    payload, stats = _payload(RESULTS, source.name, bool(eval_view))
    import threat_map
    threat_map.render(payload, PAL, on_open=_map_open, focus=st.session_state.get("map_focus"))
    st.caption("**How to read this:** every large dot is a person in a week when the model put them among the riskiest 2% of all weeks; the redder and bigger the dot, the riskier the week. "
               "Lines join a person to the websites, computers and colleagues they used that week, so anything several flagged people share (a website, a computer) is worth a look. "
               "Websites in a named category (leak, keylogger, job, cloud storage) are always labelled. Click any dot for the evidence behind its score.")
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
    sp = _scores_p(RESULTS)
    alerts = U.rank_alerts(sp, q, wk)
    scope = "all weeks" if wk is None else f"calendar week {int(wk)}"
    st.subheader(f"Top {q}% of weeks ({scope})")
    if not len(alerts):
        st.info("No scored weeks for this selection."); return
    g = metrics["ui_global_lists"][str(q)] if wk is None else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Alerts", f"{len(alerts):,}")
    k2.metric("Different people", f"{alerts['user'].nunique()}")
    k3.metric("Expected real attack weeks", f"{g['week precision']:.0%}" if g else "n/a", "measured on evaluation data" if g else "all-weeks lists only", delta_color="off")
    k4.metric("Insiders caught (of 70)", f"{g['insiders caught']}" if g else "n/a", "global list, evaluation data" if g else None, delta_color="off")
    if wk is None:
        st.caption("This is a global top-q% list over the whole period, which matches the reported results and is easier than a real weekly queue: pick a calendar week in the sidebar to see a weekly queue.")
    c1, c2 = st.columns([2, 1])
    display = c1.radio("Show risk as", U.DISPLAYS, horizontal=True, key="al_display", help="Raw scores of the top rows all read 0.9999, so the default is the percentile (how many weeks score lower).")
    show_sd = c2.checkbox("Show spread over the 5 repeats", value=False)
    t = U.alerts_frame(alerts, display, show_sd)
    if ev is not None:
        t = t.merge(ev[["user", "week_start", "label", "scenario"]].rename(columns={"user": "Person", "week_start": "Week", "label": "Attack week (truth)", "scenario": "Scenario (truth)"}), on=["Person", "Week"], how="left")
    lo, hi = {"Percentile": (float(t["Risk"].min()) - 0.05, 100.0), "Log-odds": (float(t["Risk"].min()) - 0.5, float(t["Risk"].max()) + 0.5), "Raw score": (0.0, 1.0)}[display]
    fmt = {"Percentile": "%.2f", "Log-odds": "%.1f", "Raw score": "%.4f"}[display]
    cfg = {"Week": st.column_config.DateColumn("Week", format="YYYY-MM-DD"), "Risk": st.column_config.ProgressColumn("Risk (final model)", format=fmt, min_value=lo, max_value=hi),
           "Without named sites": st.column_config.NumberColumn("Without named sites", format=fmt, help="Rating from features that do not use the named-site (leak, job, cloud, keylogger, competitor) categories."),
           "Why": st.column_config.ListColumn("Main reasons (push up)")}
    event = st.dataframe(t, hide_index=True, width="stretch", column_config=cfg, on_select="rerun", selection_mode="single-row", key="al_tbl", height=min(35 * (len(t) + 1) + 3, 520))
    st.caption("**How to read this:** each row is one person in one week. The bar shows how risky the model rated that week (a fuller bar is riskier; in percentile mode it is the share of all weeks that scored lower). "
               "'Without named sites' is the same rating with the near-label named-site features removed. 'Main reasons' are the two kinds of behaviour that pushed the score up the most. Click a row to select it.")
    rows = event.selection.rows if event is not None and event.selection else []
    i = rows[0] if rows else 0
    if not rows:
        st.caption("No row selected: the buttons act on the top alert.")
    u, w = alerts.iloc[i]["user"], alerts.iloc[i]["week_start"]
    st.session_state.sel_user, st.session_state.sel_week = u, w
    st.markdown(f"**Selected:** {u}, week of {w:%Y-%m-%d}")
    a, b, _ = st.columns([1, 1, 4])
    a.button("Open in Investigate", on_click=goto, args=("Investigate", u, w))
    b.button("Open in Threat map", on_click=goto, args=("Threat map", u, w))


def page_investigate():
    sp = _scores_p(RESULTS)
    users = sorted(scores["user"].unique())
    su = st.session_state.sel_user
    c0, c1 = st.columns(2)
    user = c0.selectbox("Person", users, index=users.index(su) if su in users else 0)
    tl = U.user_timeline(scores, user)
    weeks_l = list(tl["week_start"])
    sw = st.session_state.sel_week
    pick = c1.selectbox("Week", weeks_l, index=weeks_l.index(sw) if (su == user and sw in weeks_l) else len(weeks_l) - 1, format_func=lambda d: f"{d:%Y-%m-%d}")
    row = scores[(scores["user"] == user) & (scores["week_start"] == pick)].iloc[0]
    prow = sp[(sp["user"] == user) & (sp["week_start"] == pick)].iloc[0]
    cuts = U.budget_cutoffs(scores)
    c1_, c2_, c3_ = st.columns(3)
    c1_.metric("Risk percentile", f"{prow['pct']:.2f}", f"score {row['score']:.4f} ± {row['score_sd']:.4f}", delta_color="off")
    c2_.metric("Without named sites", f"{prow['pct_g']:.2f}", f"score {row['score_generic']:.4f}", delta_color="off")
    c3_.metric("Alert level", next((f"top {b}%" for b in U.BUDGETS if row["score"] >= cuts[b]), "not an alert"))
    st.subheader("This person over time")
    long = pd.concat([tl.assign(series="Final model")[["week_start", "score", "series"]], tl.assign(series="Without named sites", score=tl["score_generic"])[["week_start", "score", "series"]]])
    colour = alt.Color("series:N", scale=alt.Scale(domain=["Final model", "Without named sites"], range=[PAL["accent"], PAL["muted"]]), legend=alt.Legend(title=None, orient="top"))
    y = alt.Y("score:Q", scale=alt.Scale(domain=[0, 1], clamp=True, nice=False), title="Risk score (0 = lowest, 1 = highest)")
    x = alt.X("week_start:T", title="Week")
    edges = [0.0] + [cuts[b] for b in sorted(cuts, reverse=True)][::-1] + [1.0]
    bands = pd.DataFrame({"lo": [cuts[2.0], cuts[1.0], cuts[0.5], cuts[0.25]], "hi": [cuts[1.0], cuts[0.5], cuts[0.25], 1.0], "level": ["top 2%", "top 1%", "top 0.5%", "top 0.25%"], "i": [0, 1, 2, 3]})
    band_layer = alt.Chart(bands).mark_rect(opacity=0.14).encode(y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 1], clamp=True, nice=False)), y2="hi:Q", color=alt.Color("level:N", scale=alt.Scale(domain=list(bands["level"]), range=PAL["risk"]), legend=None))
    lab = alt.Chart(bands.assign(x=tl["week_start"].max())).mark_text(align="right", dx=-4, dy=-6, color=PAL["text2"], fontSize=11).encode(x="x:T", y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 1], clamp=True, nice=False)), text="level:N")
    lines = alt.Chart(long).mark_line(interpolate="linear", strokeWidth=2, clip=True).encode(x=x, y=y, color=colour)
    dots = alt.Chart(long).mark_point(filled=True, size=26, clip=True).encode(x=x, y=y, color=alt.Color("series:N", scale=alt.Scale(domain=["Final model", "Without named sites"], range=[PAL["accent"], PAL["muted"]]), legend=None))
    sel = alt.Chart(pd.DataFrame({"week_start": [pick], "score": [row["score"]]})).mark_point(size=170, color=PAL["text"], filled=False, strokeWidth=2).encode(x="week_start:T", y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1], clamp=True, nice=False)))
    layers = [band_layer, lines, dots, lab, sel]
    if ev is not None:
        e = ev[ev["user"] == user]
        if e["window_start"].notna().any():
            layers.insert(0, alt.Chart(pd.DataFrame({"s": [e["window_start"].iloc[0]], "e": [e["window_end"].iloc[0]]})).mark_rect(color=PAL["text"], opacity=0.10).encode(x="s:T", x2="e:T"))
            st.error(f"EVALUATION VIEW: the pale vertical band is the true attack window {e['window_start'].iloc[0]:%Y-%m-%d} to {e['window_end'].iloc[0]:%Y-%m-%d} (scenario {int(e['scenario'].iloc[0])}). Not available to analysts.")
        else:
            st.error("EVALUATION VIEW: this person is not a listed insider.")
    st.altair_chart(alt.layer(*layers).resolve_scale(color="independent").properties(height=330), width="stretch")
    st.caption("**How to read this:** each dot is one week of this person; the line joins them in time and a higher dot means a riskier week. The coloured bands mark the alert levels: a dot inside a band would be an alert at that budget "
               "(the top band is the strictest). The ringed dot is the week selected above. Weeks before a person's fifth active week are not scored.")
    st.subheader(f"Why the week of {pick:%Y-%m-%d} scored as it did")
    g = pd.DataFrame({"group": list(U.GROUP_COLS.values()), "contribution": [float(row[c]) for c in U.GROUP_COLS]})
    st.altair_chart(alt.Chart(g).mark_bar().encode(y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230, labelOverlap=False)), x=alt.X("contribution:Q", title="Pushes the risk up (right) or down (left)"),
                    color=alt.condition(alt.datum.contribution > 0, alt.value(PAL["positive"]), alt.value(PAL["negative"]))).properties(height=300), width="stretch")
    st.caption("**How to read this:** each bar is one kind of behaviour. A bar to the right pushed this week's risk up, a bar to the left pulled it down, and a longer bar means a bigger push; the longest bar is what mattered most. "
               f"The starting point before any behaviour is counted is {float(row['bias']):+.2f} on the model's log-odds scale. These are explanations of the score, not proof of why someone is risky.")
    st.button("Show this person in the Threat map", on_click=goto, args=("Threat map", user, pick))


def page_whatif():
    meta = _json(RESULTS, "final_model_meta.json")
    if meta is None or not (Path(RESULTS) / "final_xgb_main.json").exists():
        st.error("The what-if scorer needs the final models. Create them with `python -m itgnn.ui_export model`."); return
    models, rows = _models(RESULTS), _final_rows(RESULTS)
    st.markdown("### What if this week looked different?")
    st.write("Start from a typical week, move the sliders to change what a person did, and see how the model's risk rating responds. "
             "Everything is computed live by the final model; nothing is looked up.")
    typical = W.median_baseline(meta)
    mode_ = st.radio("Start from", ["A typical week", "A flagged week from the data"], horizontal=True, key="wi_mode")
    if mode_ == "A typical week":
        base, base_id = typical, "typical"
    else:
        opts = list(range(len(rows)))
        pick = st.selectbox("Flagged week", opts, format_func=lambda i: f"{rows.loc[i, 'user']}  {rows.loc[i, 'week_start']:%Y-%m-%d}", key="wi_pick")
        base, base_id = {c: float(rows.loc[pick, c]) for c in meta["features_main"]}, f"row{pick}"
        st.caption("This week is in the final model's own training data, so its score is optimistic compared with a week the model has never seen.")
    smax = {k: W.slider_max(meta, k, base) for k in W.SLIDERS}
    if st.session_state.get("wi_base_id") != base_id:
        st.session_state["wi_base_id"] = base_id
        for k in W.SLIDERS:
            st.session_state[f"wi_{k}"] = int(min(base[k], smax[k]))
    pre = W.presets({k: base[k] for k in W.SLIDERS})

    def apply_preset(vals):
        for k in W.SLIDERS:
            st.session_state[f"wi_{k}"] = int(min(vals.get(k, base[k]), smax[k]))
    st.caption("One-click scenarios, written from the one-sentence scenario descriptions in the dataset (not from any insider's data). The numbers are illustrative multiples of the starting week.")
    pc = st.columns(len(pre))
    for col, (name, vals) in zip(pc, pre.items()):
        col.button(name, on_click=apply_preset, args=(vals,), width="stretch")
    with st.container(key="wi_row"):
        left, right = st.columns([1.15, 1])
        with left:
            cur = {}
            for k in W.SLIDERS:
                cur[k] = st.slider(W.LABELS[k], 0, smax[k], key=f"wi_{k}", help=f"A typical week has {typical[k]:g}; 99% of real weeks are at or below {meta['p99'][k]:g}. Larger values can be typed in the Advanced editor below.")
            feats0, notes = W.rebuild(base, cur)
            with st.expander("Advanced: every feature, exact entry (values beyond the sliders go here)"):
                st.caption("Type any value to override it. 'Set by sliders' and 'Worked out' values follow the sliders until you type over them; 'Stays at the starting week' ones cannot be worked out from the sliders. Values far outside real weeks get a warning.")
                kind = {}
                for c in W.feature_columns():
                    b = c.replace("hist_mean_", "").replace("dev_", "")
                    kind[c] = ("Set by sliders" if c in cur else "Worked out" if c.startswith("dev_") and b not in W.FROZEN else "Typed here or starting week" if c in W.ADVANCED_KNOBS else "Stays at the starting week")
                tbl = pd.DataFrame({"Feature": list(feats0), "Plain name": [W.LABELS.get(c.replace("hist_mean_", "").replace("dev_", ""), c) + (" (recent average)" if c.startswith("hist_mean_") else " (change vs recent average)" if c.startswith("dev_") else "") for c in feats0],
                                    "Value": [round(v, 4) for v in feats0.values()], "How it is set": [kind[c] for c in feats0], "Typical": [round(meta["medians"][c], 3) for c in feats0], "99th percentile": [round(meta["p99"][c], 3) for c in feats0]})
                epoch = st.session_state.setdefault("wi_adv_epoch", 0)
                st.button("Clear typed values", on_click=lambda: st.session_state.update(wi_adv_epoch=epoch + 1))
                edited = st.data_editor(tbl, hide_index=True, disabled=["Feature", "Plain name", "How it is set", "Typical", "99th percentile"], key=f"wi_adv_{base_id}_{epoch}", height=300)
        overrides = {r.Feature: float(r.Value) for r in edited.itertuples() if abs(float(r.Value) - round(feats0[r.Feature], 4)) > 1e-9}
        feats, notes = W.rebuild(base, cur, overrides)
        res = W.predict(models, meta, pd.DataFrame([feats]))
        score, score_g = float(res["score"].iloc[0]), float(res["score_generic"].iloc[0])
        pct, pct_g = W.percentile(meta, "score", score), W.percentile(meta, "score_generic", score_g)
        v = W.verdict(meta, score)
        with right:
            with st.container(key="wi_rating", border=True):   # pinned (sticky) so the rating stays in view while the sliders are used
                st.caption("Risk rating of this week (final model)")
                st.markdown(f"#### Riskier than {pct:.2f}% of real weeks")
                if v is not None:
                    st.success(f"Would be an alert at the {v:g}% budget (score {score:.4f}).")
                else:
                    st.info(f"Would not be an alert at any budget up to 2% (score {score:.4f}).")
                a, b = st.columns(2)
                a.metric("All features: riskier than", f"{pct:.1f}%", f"of weeks · score {score:.4f}", delta_color="off")
                b.metric("No named sites: riskier than", f"{pct_g:.1f}%", f"of weeks · score {score_g:.4f}", delta_color="off")
            for n in notes:
                st.info(n)
            for w_ in W.range_warnings(meta, feats, list(cur) + [k for k in overrides if k in meta["p99"]]):
                st.warning(w_)
            g = pd.DataFrame({"group": list(U.GROUP_COLS.values()), "contribution": [float(res[k].iloc[0]) for k in meta["groups"]]})
            st.altair_chart(alt.Chart(g).mark_bar().encode(y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230, labelOverlap=False)), x=alt.X("contribution:Q", title="Pushes the risk up (right) or down (left)"),
                            color=alt.condition(alt.datum.contribution > 0, alt.value(PAL["positive"]), alt.value(PAL["negative"]))).properties(height=330), width="stretch")
            st.caption("**How to read this:** each bar is one kind of behaviour. A bar to the right pushed this week's risk up, a bar to the left pulled it down, and a longer bar means a bigger push. The longest bar is what moved this rating the most.")
    st.caption("This rating is a ranking signal (how this week compares with real weeks), not the probability that someone is an insider. Named-site features (leak, keylogger, job, cloud and competitor sites) are close to label markers in this synthetic data, so look at the 'no named sites' rating too. "
               "The final model has seen every week, so its ratings differ slightly from the out-of-sample scores shown on the other pages.")
    st.subheader("Sweep one behaviour")
    st.write("Pick one behaviour and see how the rating changes from none to a lot, with everything else held where you set it. The list starts with the behaviour that moves this rating the most.")
    cur_all = dict(cur)
    for k in W.ADVANCED_KNOBS:
        cur_all[k] = feats[k]
    eff = W.knob_effects(models, meta, base, cur_all, overrides)
    order = list(eff["knob"])
    knob = st.selectbox("Behaviour to sweep", order, format_func=lambda k: W.LABELS[k] + ("  (biggest effect right now)" if k == order[0] else ""), key="wi_sweep")
    ov = {k: v_ for k, v_ in overrides.items() if k not in (knob, f"dev_{knob}")}
    df = W.sweep(models, meta, base, cur_all, knob, n=160, overrides=ov)
    df["z"], df["z_g"] = W.logit(df["score"]), W.logit(df["score_generic"])
    cuts = pd.DataFrame({"z": [float(W.logit(c)) for c in meta["budget_cutoffs"].values()], "label": [f"top {qq}% cut-off" for qq in meta["budget_cutoffs"]]})
    lo, hi = min(df["z"].min(), df["z_g"].min(), cuts["z"].min()) - 1, max(df["z"].max(), df["z_g"].max(), cuts["z"].max()) + 1
    long = pd.concat([df.assign(series="All features", y=df["z"]), df.assign(series="No named sites", y=df["z_g"])])
    colour = alt.Color("series:N", scale=alt.Scale(domain=["All features", "No named sites"], range=[PAL["accent"], PAL["muted"]]), legend=alt.Legend(title=None, orient="top", labelLimit=300))
    ysc = alt.Scale(domain=[lo, hi], nice=False)
    line = alt.Chart(long).mark_line(strokeWidth=2, interpolate="step-after").encode(x=alt.X("value:Q", title=W.LABELS[knob]), y=alt.Y("y:Q", scale=ysc, title="Risk score (log-odds: higher = riskier)"), color=colour, tooltip=["series", "value", "y"])
    rules = alt.Chart(cuts).mark_rule(strokeDash=[4, 4], color=PAL["text2"]).encode(y=alt.Y("z:Q", scale=ysc))
    xmax = float(df["value"].max())
    names_ = alt.Chart(cuts.assign(x=xmax)).mark_text(align="right", baseline="bottom", dx=-4, dy=-2, color=PAL["text"], fontSize=11).encode(x="x:Q", y=alt.Y("z:Q", scale=ysc), text="label:N")
    here = alt.Chart(pd.DataFrame({"x": [cur_all[knob]]})).mark_rule(color=PAL["text"], strokeWidth=1.5).encode(x="x:Q")
    st.altair_chart((line + rules + names_ + here).properties(height=320), width="stretch")
    st.caption("**How to read this:** the line shows how risky the model thinks the week is as you turn just this one behaviour up from none to a lot; higher is riskier. The dashed lines are the alert cut-offs: where the blue line is above one, "
               "that week would be an alert at that budget. The white vertical line is where the slider is now. A flat line means the model hardly reacts to this behaviour on its own; steps are normal, because the model decides in thresholds.")
    cut = meta["budget_cutoffs"][str(q)]
    cr = W.first_crossing(df, cut, cur_all[knob], score)
    lab_, now_, typ_, p99_ = W.LABELS[knob], cur_all[knob], typical[knob], meta["p99"][knob]
    if cr is None:
        st.markdown(f"**What would it take?** Within the range {df['value'].min():g} to {df['value'].max():g}, changing only *{lab_.lower()}* " + ("never makes this week an alert" if score < cut else "never stops this week being an alert") + f" at the {q}% budget (pick another budget in the sidebar).")
    elif cr[0] == "up":
        st.markdown(f"**What would it take?** *{lab_}* would need to reach about **{cr[1]:g}** (now {now_:g}; a typical week has {typ_:g}) for this week to become an alert at the {q}% budget." + (f" That is beyond what 99% of real weeks show ({p99_:g})." if cr[1] > p99_ else ""))
    else:
        st.markdown(f"**What would it take?** This week stops being an alert at the {q}% budget if *{lab_.lower()}* falls below about **{cr[1]:g}** (now {now_:g}).")
    st.caption("**How to read this:** this sentence reads the same line as the chart above: it finds the first value where the blue line crosses the dashed cut-off of the alert budget chosen in the sidebar, keeping every other slider where it is.")


def page_rigor():
    T = U.metrics_tables(metrics)
    curve, facts = _json(RESULTS, "ui_budget_curve.json"), _json(RESULTS, "ui_rigor_facts.json")
    st.subheader("Does it rank attack weeks above ordinary weeks?")
    pr = metrics["pr_auc_25fold"]
    names = {"W2": "First baseline (counts + history)", "W2+calendar": "Baseline + calendar week (a trick)", "W2+E+H generic": "Email + web, no named-site features", "W2+E+H": "Final model (email + web + named sites)"}
    d = pd.DataFrame({"model": [names[k] for k in names], "score": [pr[k]["pr_auc_mean"] for k in names], "lo": [pr[k]["pr_auc_mean"] - pr[k]["pr_auc_std"] for k in names], "hi": [pr[k]["pr_auc_mean"] + pr[k]["pr_auc_std"] for k in names]})
    bars = alt.Chart(d).mark_bar(color=PAL["accent"]).encode(y=alt.Y("model:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=320)), x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="Ranking quality (PR-AUC, 1.0 = perfect)"))
    err = alt.Chart(d).mark_errorbar(color=PAL["text"], ticks=True, thickness=2).encode(y=alt.Y("model:N", sort=list(names.values())), x=alt.X("lo:Q", title=""), x2="hi:Q")
    st.altair_chart((bars + err).properties(height=190), width="stretch")
    st.caption("**How to read this:** each bar is how well a model ranks real attack weeks above ordinary ones (longer is better; 1.0 would be perfect and random guessing is about "
               f"{metrics['attack_weeks'] / metrics['n_scored_weeks']:.3f}). The thin line is the spread over the 25 train/test splits, so bars whose lines do not overlap really differ.")
    if curve:
        st.subheader("Real insiders caught against false alarms")
        short = {"W2": "First baseline", "W2+E+H generic": "No named sites", "W2+E+H": "Final model"}
        rows = [{"model": short[k], "budget": b, "caught": curve["sets"][k]["caught"][i], "false alarms": curve["sets"][k]["normal_flagged"][i]} for k in short for i, b in enumerate(curve["budgets"])]
        df = pd.DataFrame(rows)
        base = alt.Chart(df).encode(x=alt.X("budget:Q", scale=alt.Scale(type="log"), title="Alert budget (% of all weeks flagged, log scale)"), color=alt.Color("model:N", scale=alt.Scale(domain=list(short.values()), range=[PAL["muted"], PAL["text2"], PAL["accent"]]), legend=alt.Legend(title=None, orient="top", symbolType="stroke")))
        c1, c2 = st.columns(2)
        c1.altair_chart((base.mark_line(strokeWidth=2).encode(y=alt.Y("caught:Q", scale=alt.Scale(domain=[0, curve["n_insiders"]]), title=f"Real insiders caught (of {curve['n_insiders']})")) + base.mark_circle(size=30).encode(y="caught:Q")).properties(height=280), width="stretch")
        c2.altair_chart((base.mark_line(strokeWidth=2).encode(y=alt.Y("false alarms:Q", title=f"Normal people flagged (of {curve['n_normal_users']})")) + base.mark_circle(size=30).encode(y="false alarms:Q")).properties(height=280), width="stretch")
        st.caption("**How to read this:** left, a higher line means more real insiders found; right, a lower line means fewer innocent people flagged. Each dot is one alert budget (how many weeks may be flagged), "
                   "moving right allows more alerts. The best model is high on the left chart and low on the right chart at the same budget.")
    st.subheader("Does it hold up on weeks it has not seen?")
    ts = pd.DataFrame(metrics["time_split"]).assign(test="Same people, later weeks")
    dh = pd.DataFrame(metrics["double_holdout"]).rename(columns={"pr_auc_pooled": "pr_auc"}).assign(test="New people, later weeks")
    both = pd.concat([ts, dh]); both["Model"] = both["set"].map({"W2": "First baseline", "W2+E+H": "Final model", "W2+E+H generic": "No named sites"}); both["cutoff"] = "from week " + (both["T"] + 1).astype(str)
    cols = st.columns(2)
    for col, test in zip(cols, ["Same people, later weeks", "New people, later weeks"]):
        ch = alt.Chart(both[both["test"] == test]).mark_bar().encode(x=alt.X("cutoff:N", title=test + " (trained before, tested from this week)", axis=alt.Axis(labelAngle=0)), xOffset="Model:N",
              y=alt.Y("R:Q", scale=alt.Scale(domain=[0, 1.1]), title="Share of its normal accuracy kept"),
              color=alt.Color("Model:N", scale=alt.Scale(domain=["First baseline", "No named sites", "Final model"], range=[PAL["muted"], PAL["text2"], PAL["accent"]]), legend=alt.Legend(title=None, orient="top")),
              tooltip=["Model", "cutoff", alt.Tooltip("R:Q", format=".2f")])
        col.altair_chart(ch.properties(height=250), width="stretch")
    st.caption("**How to read this:** a tall bar means the model kept most of the accuracy it shows in the main evaluation when it only had earlier weeks to learn from; 1.0 means no loss. "
               "Bars are short for the earliest cutoff because very few attacks had happened yet, and shorter again when the test people are completely new.")
    if facts and facts.get("graph_controls"):
        gc = facts["graph_controls"]
        st.subheader("Do connections between people add anything?")
        gd = pd.DataFrame({"model": ["No connections (same features)", "With connections", "Connections scrambled", "Simple model we use"], "score": [gc["no_edge_mlp"], gc["graphsage"], gc["graphsage_shuffled_edges"], gc["xgboost_same_features"]]})
        st.altair_chart(alt.Chart(gd).mark_bar(color=PAL["accent"]).encode(y=alt.Y("model:N", sort=list(gd["model"]), title=None, axis=alt.Axis(labelLimit=300)), x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="Ranking quality (PR-AUC)")).properties(height=170), width="stretch")
        st.caption("**How to read this:** the first three bars are the same neural network with real connections, none, and scrambled ones; if connections mattered the second bar would clearly beat the others. It does not, and the simple model (last bar) beats all three.")
    st.subheader("Limitations")
    for line in U.limitations(metrics):
        st.markdown(f"- {line}")
    st.subheader("All the numbers")
    with st.expander("Global list versus a real weekly queue"):
        st.dataframe(T["lists"].reset_index(), hide_index=True, width="stretch")
        st.caption("**How to read this:** each row is one alert budget. 'Insiders caught' is how many of the 70 insiders appear in the list and 'normal users flagged' how many innocent people; the weekly-queue rows are what you would get picking the riskiest weeks inside each calendar week instead.")
    with st.expander("Ranking quality over 25 splits"):
        st.dataframe(T["pr_auc"], width="stretch"); st.dataframe(T["paired"], width="stretch")
        st.caption("**How to read this:** first table, a higher PR-AUC is better and the spread column shows how much it moved between splits. Second table, 'mean PR-AUC difference' is the gain of the first model over the second on the same splits; 'folds positive' counts the splits where it won (25/25 = every time).")
    with st.expander("Alert budget over 25 splits"):
        st.dataframe(T["budget"], hide_index=True, width="stretch")
        st.caption("**How to read this:** for each model and budget, how many of the 70 insiders were caught and how many of the 930 normal people were flagged; more caught and fewer flagged is better.")
    with st.expander("Time-based tests"):
        st.dataframe(T["time_split"], hide_index=True, width="stretch"); st.dataframe(T["double_holdout"], hide_index=True, width="stretch")
        st.caption("**How to read this:** 'retention R' is the time-based result divided by the main result on the same weeks, so 1.0 means nothing was lost; the first table uses the same people on both sides, the second completely new people.")


{"Overview": page_overview, "Threat map": page_map, "Alerts": page_alerts, "Investigate": page_investigate, "What-if": page_whatif, "Rigor and limits": page_rigor}[page]()
