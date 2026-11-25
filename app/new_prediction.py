"""The "New prediction" tab: build a hypothetical worker (usual week + this week + optional connections), rate it with the saved final models,
and keep it in a session-only list. Nothing is written to disk or to any results file; labels are never loaded here."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from itgnn import ui_hypo as H
from itgnn import whatif as W
from itgnn import ui_data as U


LABELS = {**W.LABELS, "em_n_bcc": "Emails with a Bcc recipient", "em_n_attach": "Emails with an attachment", "em_size_sum": "Total size of emails sent (bytes)",
          "em_n_new_recip": "New email recipients (never mailed before)", "wb_n_new_domains": "New websites (never visited before)", "wb_share_rare": "Share of visits to rare websites",
          "wb_n_domains": "Different websites visited", "n_new_pcs": "New computers (never used before)"}   # plain names for the counts whatif.LABELS does not cover


@st.cache_data
def _pcs(path):
    e = pd.read_parquet(Path(path) / "ui_edges.parquet", columns=["edge_type", "other_id"])
    return sorted(e.loc[e["edge_type"] == "LOGGED_ON", "other_id"].unique())


@st.cache_data
def _domains(path):
    return H.load_domains(path)


def _site_label(row):
    return row.domain + (f"   [named-site category: {H.CAT_NAME[row.category]}]" if row.category else "")


def _contrib_chart(groups, PAL, height=250):
    g = pd.DataFrame({"group": list(U.GROUP_COLS.values()), "contribution": [float(v) for v in groups.values()]})
    return alt.Chart(g).mark_bar().encode(y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230, labelOverlap=False)), x=alt.X("contribution:Q", title="Pushes the risk up (right) or down (left)"),
                                          color=alt.condition(alt.datum.contribution > 0, alt.value(PAL["positive"]), alt.value(PAL["negative"]))).properties(height=height)


def _alert_text(meta, score, q):
    return score >= meta["budget_cutoffs"][str(q)]


def render(ctx):
    """ctx: results, meta, models, scores, PAL, howto, budget_control, cap, real_users."""
    meta, models, scores, PAL, howto = ctx["meta"], ctx["models"], ctx["scores"], ctx["PAL"], ctx["howto"]
    typical = W.median_baseline(meta)
    real = ctx["scores"]["score"].to_numpy()
    last = scores.loc[scores["cal_week"].idxmax()]
    wk_lo, wk_hi = int(scores["cal_week"].min()), int(last["cal_week"])
    domains, dnote = _domains(ctx["results"])
    workers = st.session_state.setdefault("np_workers", [])
    cap = ctx["cap"]

    st.markdown("### Rate a new, hypothetical worker")
    st.write("Describe a worker's usual week and this week, optionally say which websites, computers and colleagues were involved, and the final model rates the week. "
             "The worker exists only in this browser session: nothing is saved, and no real week, score, label or number elsewhere in the app changes.")
    st.warning("**Read this first.** Named-site features (leak, keylogger, job, cloud and competitor sites) are near-label markers in this synthetic data: "
               "a single visit to a leak site is enough to lift a typical week into the top 2% (the loosest alert budget). Always read the **without named sites** rating beside the main one.", icon="⚠️")
    st.caption("The rating is a ranking signal (how this week compares with real weeks), not the probability that someone is an insider. Rank placement is approximate: "
               "the final model has already seen every real week, so its scores differ slightly from the out-of-sample scores the ranking is read against.")
    if dnote:
        st.info(dnote)

    left, right = st.columns([1.6, 1])
    with left:
        c1, c2 = st.columns([2, 1])
        name = c1.text_input("Worker name", key="np_name", placeholder="e.g. Test worker 1", max_chars=30, help="Must not look like a real user id (three letters and four digits) and must be unique in this session.")
        cw = c2.number_input("Calendar week", wk_lo, wk_hi, value=wk_hi, step=1, key="np_week", help=f"Week {wk_hi} is the latest week in the data. The model has no calendar feature, so the week only places the worker on the map and in lists; it does not change the score.")
        wstart = H.week_start_of(cw, wk_hi, last["week_start"])
        c2.caption(f"week of {wstart:%Y-%m-%d}")
        bad = H.check_name(name, [w["name"] for w in workers], ctx["real_users"]) if name.strip() else "Give the worker a name."
        if name.strip() and bad:
            c1.error(bad)
        smax = {k: W.slider_max(meta, k, typical) for k in W.SLIDERS}
        for k in W.SLIDERS:
            for side in ("u", "t"):
                st.session_state.setdefault(f"np_{side}_{k}", int(min(typical[k], smax[k])))
        st.markdown("#### The week")
        cu, ct = st.columns(2)
        cu.markdown("**Their usual week** (what they normally do)")
        ct.markdown("**This week**")
        usual, this_week = {}, {}
        for k in W.SLIDERS:
            usual[k] = cu.slider(W.LABELS[k], 0, smax[k], key=f"np_u_{k}", help=f"A typical week has {typical[k]:g}; 99% of real weeks are at or below {meta['p99'][k]:g}. Larger values can be typed in the Advanced editor.")
            this_week[k] = ct.slider(W.LABELS[k], 0, smax[k], key=f"np_t_{k}", label_visibility="visible")
        st.caption("Starting values are a typical week. If the two columns are equal, every 'change versus usual' feature is 0.")

        st.markdown("#### Connections (optional)")
        by_domain = {r.domain: r for r in domains.itertuples()}
        picked = st.multiselect("Websites visited (type to search)", list(by_domain), format_func=lambda d: _site_label(by_domain[d]), key="np_sites")
        sites = {}
        for d in picked:
            sites[d] = (by_domain[d].category, float(st.number_input(f"Visits to {d}", 1, 1000, value=1, step=1, key=f"np_v_{d}")))
        pcs_all = _pcs(ctx["results"])
        pcs = list(st.multiselect("Computers used (type to search)", pcs_all, key="np_pcs"))
        if st.checkbox("Also used a new computer no one else uses (hypothetical)", key="np_newpc"):
            pcs.append(H.NEW_PC)
        colleagues = list(st.multiselect("Colleagues emailed (type to search)", ctx["real_users"], key="np_colleagues"))
        if not (sites or pcs or colleagues):
            st.caption("None chosen: the worker has no connections and is rated on the sliders alone.")
        else:
            st.caption("A chosen connection can raise a matching count (for example 3 visits to a leak site raise 'Leak-site visits' to at least 3) but never lowers a value you set. Every raise is listed beside the rating.")

        feats0, _, cur0 = H.build_features(meta, typical, usual, this_week, sites, pcs, colleagues)
        with st.expander("Advanced: every feature, exact entry"):
            st.caption("Type any value to override it. Values far outside real weeks get a warning.")
            tbl = pd.DataFrame({"Feature": list(feats0), "Plain name": [LABELS.get(c.replace("hist_mean_", "").replace("dev_", ""), c) + (" (recent average)" if c.startswith("hist_mean_") else " (change vs recent average)" if c.startswith("dev_") else "") for c in feats0],
                                "Value": [round(v, 4) for v in feats0.values()], "Typical": [round(meta["medians"][c], 3) for c in feats0], "99th percentile": [round(meta["p99"][c], 3) for c in feats0]})
            epoch = st.session_state.setdefault("np_adv_epoch", 0)
            st.button("Clear typed values", key="np_adv_clear", on_click=lambda: st.session_state.update(np_adv_epoch=epoch + 1))
            edited = st.data_editor(tbl, hide_index=True, disabled=["Feature", "Plain name", "Typical", "99th percentile"], key=f"np_adv_{epoch}", height=300)
        overrides = {r.Feature: float(r.Value) for r in edited.itertuples() if abs(float(r.Value) - round(feats0[r.Feature], 4)) > 1e-9}

        st.markdown("#### Left at the typical week")
        raised = {k for k in cur0 if cur0[k] != this_week.get(k, float(typical[k]))}
        stay = [{"What": LABELS.get(c, c), "Value": round(feats0[c], 3),
                 "Why": "Cannot be worked out from the sliders" if c in W.FROZEN else "No slider: type it in Advanced to change it"}
                for c in W.BASE_COLS + ["n_new_pcs"] if c not in W.SLIDERS and c not in raised and c not in overrides]
        st.dataframe(pd.DataFrame(stay), hide_index=True, width="stretch", height=35 * min(len(stay), 8) + 38)
        howto("each row is a count the sliders cannot set, so it keeps the typical-week value (and its change-versus-usual is 0) unless you type over it.")

    worker = H.make_worker(models, meta, typical, real, name or "Preview", cw, wstart, usual, this_week, sites, pcs, colleagues, overrides)
    with right:
        with st.container(key="np_rating", border=True):
            st.caption("Risk rating of this worker's week (final model)")
            st.markdown(f"#### Riskier than {worker['pct']:.2f}% of real weeks")
            q = ctx["budget_control"](st, "q_np")
            if _alert_text(meta, worker["score"], q):
                st.success(f"Would be an alert at the {q:g}% budget.")
            else:
                tight = worker["verdict"]
                st.info(f"Would not be an alert at the {q:g}% budget." + (f" It would be at the {tight:g}% budget." if tight is not None else " It would not be an alert at any budget up to 2%."))
            a, b = st.columns(2)
            a.metric("All features: riskier than", f"{worker['pct']:.1f}%", help=f"Score {worker['score']:.4f}.")
            b.metric("Without named sites: riskier than", f"{worker['pct_generic']:.1f}%", help=f"The same rating without the named-site features (score {worker['score_generic']:.4f}).")
            if worker["pct"] - worker["pct_generic"] > 20:
                st.warning("The two ratings are far apart: the alert comes from named-site features, which are near-label markers.")
            for n in worker["notes"]:
                st.info(n)
            for w_ in W.range_warnings(meta, worker["features"], list(this_week) + [k for k in overrides if k in meta["p99"]]):
                st.warning(w_)
            st.altair_chart(_contrib_chart(worker["groups"], PAL), width="stretch")
            howto("each bar is one kind of behaviour, to the right pushed this rating up, to the left pulled it down, and the longest bar moved it the most.")
            st.caption(f"Would rank about #{worker['rank']:,} of {worker['n_weeks']:,} scored weeks (approximate; see the note at the top).")

            def _add(w=worker):
                try:
                    st.session_state["np_workers"] = H.add_worker(st.session_state.get("np_workers", []), w, cap, ctx["real_users"])
                    st.session_state["np_msg"] = ("ok", f"Added {w['name']} to this session.")
                    st.session_state["np_name"] = ""
                except ValueError as e:
                    st.session_state["np_msg"] = ("err", str(e))
            st.button("Add to this session's list", key="np_add", type="primary", disabled=bool(bad) or len(workers) >= cap, on_click=_add,
                      help=f"Up to {cap} hypothetical workers; the list is cleared when the page is refreshed.")
            kind, text = st.session_state.pop("np_msg", (None, None))
            if kind:
                (st.success if kind == "ok" else st.error)(text)

    st.subheader(f"This session's hypothetical workers ({len(workers)} of {cap})")
    if not workers:
        st.caption("None yet. Build one above and add it; it will be listed here and (in the next stage) shown on the Threat map, Alerts and Investigate pages.")
        return
    h = st.columns([2, 1.2, 1.4, 1.6, 1.8, 1])
    for col, t in zip(h, ["Worker", "Week", "Riskier than", "Without named sites", "Verdict at budget", ""]):
        col.caption(t)
    q = st.session_state.get("_q", 0.5)
    for i, w in enumerate(workers):
        c = st.columns([2, 1.2, 1.4, 1.6, 1.8, 1])
        c[0].markdown(f"**{w['name']}** · hypothetical")
        c[1].write(f"{w['week_start']:%Y-%m-%d}")
        c[2].write(f"{w['pct']:.1f}%")
        c[3].write(f"{w['pct_generic']:.1f}%")
        c[4].write("alert" if _alert_text(meta, w["score"], q) else "no alert")
        c[5].button("Remove", key=f"np_rm_{w['name']}", on_click=lambda n=w["name"]: st.session_state.update(np_workers=[x for x in st.session_state["np_workers"] if x["name"] != n]))
    d1, d2, _ = st.columns([1, 1, 3])
    d1.button("Clear all", key="np_clear", on_click=lambda: st.session_state.update(np_workers=[]))
    d2.download_button("Download CSV", H.to_csv(workers, W.feature_columns()), "hypothetical_workers.csv", "text/csv", key="np_csv", help="One row per worker with every input, the 79 features, the ratings and hypothetical = True.")
    st.caption("These workers are not in any score, label, metric or percentile table, and refreshing the page clears them.")
