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
    if st.session_state.get("wi_base_id") != base_id:
        st.session_state["wi_base_id"] = base_id
        for k in W.SLIDERS:
            st.session_state[f"wi_{k}"] = int(min(base[k], meta["knob_max"][k]))
    pre = W.presets({k: base[k] for k in W.SLIDERS})

    def apply_preset(vals):
        for k in W.SLIDERS:
            st.session_state[f"wi_{k}"] = int(min(vals.get(k, base[k]), meta["knob_max"][k]))
    st.caption("One-click scenarios, written from the one-sentence scenario descriptions in the dataset (not from any insider's data). The numbers are illustrative multiples of the starting week.")
    pc = st.columns(len(pre))
    for col, (name, vals) in zip(pc, pre.items()):
        col.button(name, on_click=apply_preset, args=(vals,), width="stretch")
    left, right = st.columns([1.15, 1])
    with left:
        cur = {}
        for k in W.SLIDERS:
            cur[k] = st.slider(W.LABELS[k], 0, meta["knob_max"][k], key=f"wi_{k}", help=f"A typical week has {typical[k]:g}; 99% of real weeks are at or below {meta['p99'][k]:g}.")
    feats0, notes = W.rebuild(base, cur)
    with left:
        with st.expander("Advanced: every feature, exact entry"):
            st.caption("Type any value to override it. 'Set by sliders' and 'Worked out' values follow the sliders until you type over them; 'Stays at the starting week' ones cannot be worked out from the sliders.")
            kind = {}
            for c in W.feature_columns():
                b = c.replace("hist_mean_", "").replace("dev_", "")
                kind[c] = ("Set by sliders" if c in cur else "Worked out" if c.startswith("dev_") and b not in W.FROZEN else "Typed here or starting week" if c in W.ADVANCED_KNOBS else
                           "Stays at the starting week")
            tbl = pd.DataFrame({"Feature": list(feats0), "Plain name": [W.LABELS.get(c.replace("hist_mean_", "").replace("dev_", ""), c) + (" (recent average)" if c.startswith("hist_mean_") else " (change vs recent average)" if c.startswith("dev_") else "") for c in feats0],
                                "Value": [round(v, 4) for v in feats0.values()], "How it is set": [kind[c] for c in feats0], "Typical": [round(meta["medians"][c], 3) for c in feats0],
                                "99th percentile": [round(meta["p99"][c], 3) for c in feats0]})
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
        with st.container(border=True):
            st.caption("Risk rating of this week (final model)")
            st.markdown(f"## Riskier than {pct:.2f}% of real weeks")
            if v is not None:
                st.success(f"Would be an alert at the {v:g}% budget (score {score:.4f}).")
            else:
                st.info(f"Would not be an alert at any budget up to 2% (score {score:.4f}).")
            a, b = st.columns(2)
            a.metric("All features: riskier than", f"{pct:.1f}%", f"of weeks · score {score:.4f}", delta_color="off")
            b.metric("No named sites: riskier than", f"{pct_g:.1f}%", f"of weeks · score {score_g:.4f}", delta_color="off")
        for n in notes:
            st.info(n)
        for w in W.range_warnings(meta, feats, list(cur) + [k for k in overrides if k in meta["p99"]]):
            st.warning(w)
        g = pd.DataFrame({"group": list(U.GROUP_COLS.values()), "contribution": [float(res[k].iloc[0]) for k in meta["groups"]]})
        st.altair_chart(alt.Chart(g).mark_bar().encode(y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230, labelOverlap=False)), x=alt.X("contribution:Q", title="Pushes the risk up (right) or down (left)"),
                        color=alt.condition(alt.datum.contribution > 0, alt.value(PAL["positive"]), alt.value(PAL["negative"]))).properties(height=330), width="stretch")
        st.caption("**How to read this:** each bar is one kind of behaviour. A bar to the right pushed this week's risk up, a bar to the left pulled it down, and a longer bar means a bigger push. The longest bar is what moved this rating the most.")
    st.caption("This rating is a ranking signal (how this week compares with real weeks), not the probability that someone is an insider. Named-site features (leak, keylogger, job, cloud and competitor sites) are close to label markers in this synthetic data, so look at the 'without named-site features' rating too. "
               "The final model has seen every week, so its ratings differ slightly from the out-of-sample scores shown on the other pages.")
    st.subheader("Sweep one behaviour")
    st.write("Pick one behaviour and see how the rating changes from none to a lot, with everything else held where you set it.")
    sw_opts = W.SLIDERS + W.ADVANCED_KNOBS
    knob = st.selectbox("Behaviour to sweep", sw_opts, format_func=lambda k: W.LABELS[k], key="wi_sweep")
    cur_all = dict(cur)
    for k in W.ADVANCED_KNOBS:
        cur_all[k] = feats[k]
    df = W.sweep(models, meta, base, cur_all, knob, n=50, overrides={k: v_ for k, v_ in overrides.items() if k not in (knob, f"dev_{knob}")})
    long = pd.concat([df.assign(series="All features", p=df["pct"]), df.assign(series="Without named-site features", p=df["pct_generic"])])
    colour = alt.Color("series:N", scale=alt.Scale(domain=["All features", "Without named-site features"], range=[PAL["accent"], PAL["muted"]]), legend=alt.Legend(title=None, orient="top", labelLimit=300))
    line = alt.Chart(long).mark_line(strokeWidth=2).encode(x=alt.X("value:Q", title=W.LABELS[knob]), y=alt.Y("p:Q", scale=alt.Scale(domain=[0, 100]), title="Riskier than this % of real weeks"), color=colour)
    cuts = pd.DataFrame({"y": [100 - float(q) for q in meta["budget_cutoffs"]], "label": [f"alert at top {q}%" for q in meta["budget_cutoffs"]]})
    rules = alt.Chart(cuts).mark_rule(strokeDash=[4, 4], color=PAL["muted"]).encode(y="y:Q", tooltip=["label"])
    note = alt.Chart(pd.DataFrame({"x": [0], "y": [92], "t": ["dashed lines: alert budgets (top 2%, 1%, 0.5%, 0.25%)"]})).mark_text(align="left", dx=4, color=PAL["text2"], fontSize=11).encode(x="x:Q", y="y:Q", text="t:N")
    here = alt.Chart(pd.DataFrame({"x": [cur_all[knob]]})).mark_rule(color=PAL["text"], strokeWidth=1.5).encode(x="x:Q")
    st.altair_chart((line + rules + here + note).properties(height=300), width="stretch")
    st.caption("**How to read this:** the line shows how risky the model thinks the week is as you turn just this one behaviour up from none to a lot. Higher means riskier. The dashed lines mark the alert budgets: once a line crosses one, that week would become an alert. "
               "The white vertical line is where the slider is now. A flat line means the model hardly reacts to this behaviour on its own.")


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
