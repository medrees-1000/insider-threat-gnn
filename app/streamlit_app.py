"""Analyst UI over saved results only (read-only: no training, no writes). Run: streamlit run app/streamlit_app.py"""
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from itgnn import ui_data as U
from itgnn.neo4j_ui import auto_source

RESULTS = os.getenv("ITGNN_RESULTS", "results")
BLUE, ORANGE, RED, GREY, INK = "#2a78d6", "#eb6834", "#e34948", "#9c9a92", "#0b0b0b"
CAT_COLOR = {"leak": "#e34948", "job": "#6250d6", "cloud": "#1baf7a", "download": "#eda100", "competitor": "#e87ba4"}

st.set_page_config(page_title="Insider threat analyst view", layout="wide")
st.title("Insider threat analyst view")
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


scores, weeks, metrics = _scores(RESULTS), _weeks(RESULTS), _metrics(RESULTS)
source, backend_note = _source(RESULTS)

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

ev = U.load_eval(RESULTS) if eval_view else None  # the labels are loaded ONLY here, only on request
alerts = U.rank_alerts(scores, q, wk)
tab_alerts, tab_user, tab_graph, tab_metrics = st.tabs(["Alerts", "User detail", "Neighbourhood", "Metrics and limitations"])

sel_user, sel_week = None, None
with tab_alerts:
    scope = "all weeks" if wk is None else f"calendar week {int(wk)}"
    st.subheader(f"Top {q}% of weeks ({scope}): {len(alerts):,} alerts, {alerts['user'].nunique()} distinct users")
    if wk is None:
        st.caption("Ranked globally over every scored user-week, which matches the reported metrics. A global top-q% is easier than a real weekly queue: pick a calendar week in the sidebar to see one.")
    if len(alerts):
        t = U.alert_table(alerts)
        if ev is not None:
            t = t.merge(ev[["user", "week_start", "label", "scenario"]].rename(columns={"label": "attack week (truth)", "scenario": "scenario (truth)"}), on=["user", "week_start"], how="left")
        st.dataframe(t.rename(columns={"score": "score (W2+E+H)", "score_sd": "sd over 5 repeats", "score_generic": "score (generic features)"}),
                     hide_index=True, width="stretch", column_config={"week_start": st.column_config.DateColumn("week", format="YYYY-MM-DD")})
        st.caption("The generic score uses no named-site (category-domain) features, so it shows the evidence that does not rely on the near-label markers.")
        i = st.selectbox("Open an alert", range(len(alerts)), format_func=lambda i: f"{alerts.loc[i, 'user']}  {alerts.loc[i, 'week_start']:%Y-%m-%d}  score {alerts.loc[i, 'score']:.3f}")
        sel_user, sel_week = alerts.loc[i, "user"], alerts.loc[i, "week_start"]
    else:
        st.info("No scored weeks for this selection.")

users = sorted(scores["user"].unique())
with tab_user:
    user = st.selectbox("User", users, index=users.index(sel_user) if sel_user in users else 0)
    tl = U.user_timeline(scores, user)
    pick = st.selectbox("Week", list(tl["week_start"]), index=list(tl["week_start"]).index(sel_week) if (sel_user == user and sel_week in list(tl["week_start"])) else len(tl) - 1,
                        format_func=lambda d: f"{d:%Y-%m-%d}")
    row = scores[(scores["user"] == user) & (scores["week_start"] == pick)].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Score (W2+E+H)", f"{row['score']:.3f}", f"sd {row['score_sd']:.3f} over 5 repeats", delta_color="off")
    c2.metric("Score (generic features)", f"{row['score_generic']:.3f}")
    c3.metric("Within the top", next((f"{b}%" for b in U.BUDGETS if row["score"] >= U.budget_cutoffs(scores)[b]), "more than 2% of weeks (not alerted)"))
    long = pd.concat([tl.assign(series="W2+E+H score")[["week_start", "score", "series"]],
                      tl.assign(series="generic-feature score", score=tl["score_generic"])[["week_start", "score", "series"]]])
    colour = alt.Color("series:N", scale=alt.Scale(domain=["W2+E+H score", "generic-feature score"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top"))
    base = alt.Chart(long).encode(x=alt.X("week_start:T", title="week"), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="score (out-of-fold, mean of 5 repeats)"))
    layers = [base.mark_line(strokeWidth=2).encode(color=colour), base.mark_point(size=28, filled=True).encode(color=colour)]
    cut = pd.DataFrame({"cutoff": list(U.budget_cutoffs(scores).values()), "budget": [f"top {b}%" for b in U.budget_cutoffs(scores)]})
    layers.append(alt.Chart(cut).mark_rule(strokeDash=[4, 4], color=GREY).encode(y="cutoff:Q", tooltip=["budget", "cutoff"]))
    layers.append(alt.Chart(pd.DataFrame({"week_start": [pick], "score": [row["score"]]})).mark_point(size=160, color=INK, filled=False, strokeWidth=2).encode(x="week_start:T", y="score:Q"))
    if ev is not None:
        e = ev[ev["user"] == user]
        if e["window_start"].notna().any():
            layers.insert(0, alt.Chart(pd.DataFrame({"s": [e["window_start"].iloc[0]], "e": [e["window_end"].iloc[0]]})).mark_rect(color=RED, opacity=0.12).encode(x="s:T", x2="e:T"))
            st.error(f"EVALUATION VIEW: red band = true attack window {e['window_start'].iloc[0]:%Y-%m-%d} to {e['window_end'].iloc[0]:%Y-%m-%d} (scenario {int(e['scenario'].iloc[0])}). Not available to analysts.")
        else:
            st.error("EVALUATION VIEW: this user is not a listed insider.")
    st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
    st.caption("Dashed lines: score cut-offs of the four global alert budgets. Weeks before a user's fifth active week are not scored.")
    g = pd.DataFrame({"group": list(U.GROUP_COLS.values()), "contribution (log-odds)": [float(row[c]) for c in U.GROUP_COLS]})
    st.subheader(f"Why this week scored as it did ({pick:%Y-%m-%d})")
    st.altair_chart(alt.Chart(g).mark_bar().encode(y=alt.Y("group:N", sort="-x", title=None), x=alt.X("contribution (log-odds):Q"),
                                                   color=alt.condition(alt.datum["contribution (log-odds)"] > 0, alt.value(BLUE), alt.value(GREY))).properties(height=260), width="stretch")
    st.caption(f"Positive bars push the score up. Baseline (bias) {float(row['bias']):+.2f} log-odds. XGBoost TreeSHAP contributions grouped into feature groups; not causal.")

with tab_graph:
    uw = weeks[weeks["user"] == user]
    st.caption("Neighbourhoods are exported for the top 2% of weeks (the largest budget).")
    if uw.empty:
        st.info(f"{user} has no week in the top 2%; pick a flagged user or week.")
    else:
        wopts = list(uw["week_start"])
        gw = st.selectbox("Flagged week", wopts, index=wopts.index(pick) if pick in wopts else 0, format_func=lambda d: f"{d:%Y-%m-%d}", key="graph_week")
        nb = source.neighbourhood(user, gw)
        nodes = [Node(id="center", label=f"{user}\n{gw:%Y-%m-%d}", size=26, color=BLUE)]
        edges = []
        for r in nb.itertuples():
            nid = f"{r.other_kind}:{r.other_id}"
            if r.other_kind == "PC":
                nodes.append(Node(id=nid, label=r.other_id, size=14, color=ORANGE if r.other_users == 0 else GREY, title=f"PC used by {int(r.other_users)} other users that week"))
            elif r.other_kind == "Domain":
                nodes.append(Node(id=nid, label=r.other_id, size=10 if not r.category else 14, color=CAT_COLOR.get(r.category, "#c9c8c0"),
                                  title=(f"{U.CATEGORY_NOTE}: {r.category}" if r.category else "domain")))
            else:
                nodes.append(Node(id=nid, label=r.other_id, size=12, color="#008300", title="emailed user"))
            edges.append(Edge(source="center", target=nid, width=float(np.clip(np.log1p(r.weight), 1, 6))))
        agraph(nodes=nodes, edges=edges, config=Config(width=950, height=520, directed=False, physics=True))
        legend = "  ".join([f"<span style='color:{BLUE}'>●</span> user-week", f"<span style='color:{GREY}'>●</span> PC", f"<span style='color:{ORANGE}'>●</span> PC used by nobody else that week",
                            "<span style='color:#008300'>●</span> emailed user", "<span style='color:#c9c8c0'>●</span> domain"] +
                           [f"<span style='color:{c}'>●</span> {k} (named-site)" for k, c in CAT_COLOR.items()])
        st.markdown(legend, unsafe_allow_html=True)
        st.caption(f"{U.CATEGORY_NOTE}. Every category-domain visit of this week is shown; other domains are the top 10 by visits; recipients are the top 5 internal users mailed. "
                   f"External recipients that week: {int(uw.loc[uw['week_start'] == gw, 'n_external_recipients'].iloc[0])}.")
        st.dataframe(nb.assign(category=nb["category"].replace("", np.nan)), hide_index=True, width="stretch")

with tab_metrics:
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
