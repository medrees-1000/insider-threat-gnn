"""Pure helpers behind the analyst UI (no Streamlit, no training libraries): reading saved results, alert lists,
timelines, explanations and the neighbourhood backends. The app only calls these."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLS = {  # parquet column -> the ONE plain-English label used for the contribution groups on every page (bars, reasons, map panel)
    "w2_base": "everyday activity", "w2_hist_dev": "change from their usual", "email_volume": "email volume", "email_recipients": "who they emailed",
    "web_volume": "web browsing", "web_domains": "variety of sites", "web_after_hours": "after-hours browsing", "named_site": "named-site visits", "first_seen": "new things"}
BUDGETS = [0.25, 0.5, 1.0, 2.0]
CATEGORY_NOTE = "Named-site category (also an XGBoost feature; evidence for the analyst)"
EDGE_COLS = ["edge_type", "other_id", "other_kind", "weight", "category", "other_users"]


def week_key(user, week_start):
    return f"{user}|{pd.Timestamp(week_start):%Y-%m-%d}"


def top_groups(row, k=3):
    """The k groups pushing the score up the most: [(label, contribution in log-odds)], largest first."""
    vals = sorted(((GROUP_COLS[c], float(row[c])) for c in GROUP_COLS), key=lambda t: -t[1])
    return vals[:k]


def format_groups(pairs):
    return "; ".join(f"{n} {v:+.2f}" for n, v in pairs)


def n_alerts(n_rows, q):
    return int(np.ceil(q / 100 * n_rows))


def rank_alerts(scores, q, week=None):
    """Top alerts by `score`. Global (default): the top q% of ALL scored weeks. With `week` (a calendar-week number),
    the top q% of the weeks inside that calendar week only. Ties are broken by user and week (deterministic)."""
    s = scores if week is None else scores[scores["cal_week"] == week]
    n = n_alerts(len(s), q)
    return s.sort_values(["score", "user", "week_start"], ascending=[False, True, True]).head(n).reset_index(drop=True)


def alert_table(alerts, k=3):
    t = alerts[["user", "week_start", "cal_week", "score", "score_sd", "score_generic"]].copy()
    t["top groups (log-odds)"] = [format_groups(top_groups(r, k)) for _, r in alerts.iterrows()]
    return t


def user_timeline(scores, user):
    return scores[scores["user"] == user].sort_values("week_start")[["week_start", "cal_week", "score", "score_sd", "score_generic"]]


def budget_cutoffs(scores):
    """Score cut-off of each global budget (the lowest score still inside the top q%)."""
    s = np.sort(scores["score"].to_numpy())[::-1]
    return {q: float(s[n_alerts(len(s), q) - 1]) for q in BUDGETS}


def load_scores(results="results"):
    p = Path(results)
    s = pd.read_parquet(p / "ui_scores.parquet")
    return s


def load_eval(results="results"):
    """Ground-truth labels and attack windows. ONLY for the explicitly labelled evaluation view."""
    return pd.read_parquet(Path(results) / "ui_eval.parquet")


def load_metrics(results="results"):
    p = Path(results) / "ui_metrics.json"
    return json.load(open(p)) if p.exists() else None


class ParquetSource:
    """Neighbourhood of a user-week from the exported edge table (the fallback backend)."""

    name = "parquet"

    def __init__(self, results="results"):
        self.edges = pd.read_parquet(Path(results) / "ui_edges.parquet")
        self.edges["key"] = [week_key(u, w) for u, w in zip(self.edges["user"], self.edges["week_start"])]

    def has_week(self, user, week_start):
        return bool((self.edges["key"] == week_key(user, week_start)).any())

    def all_edges(self):
        """Every exported edge row (user, week_start, edge_type, other_id, other_kind, weight, category, other_users)."""
        return self.edges.drop(columns="key").sort_values(["user", "week_start", "edge_type", "other_id"]).reset_index(drop=True)

    def neighbourhood(self, user, week_start):
        e = self.edges[self.edges["key"] == week_key(user, week_start)]
        out = e[EDGE_COLS].copy()
        out["weight"] = out["weight"].astype(float)
        return out.sort_values(["edge_type", "weight", "other_id"], ascending=[True, False, True]).reset_index(drop=True)


def normalise_rows(rows):
    """Neo4j rows (list of dicts) -> the same frame ParquetSource returns."""
    out = pd.DataFrame(rows, columns=EDGE_COLS)
    out["weight"] = out["weight"].astype(float)
    out["category"] = out["category"].fillna("")
    return out.sort_values(["edge_type", "weight", "other_id"], ascending=[True, False, True]).reset_index(drop=True)


def metrics_tables(m):
    """The metrics panel as DataFrames, from ui_metrics.json (never hand-typed numbers)."""
    t = {}
    pr = pd.DataFrame(m["pr_auc_25fold"]).T
    t["pr_auc"] = pd.DataFrame({"PR-AUC mean": pr["pr_auc_mean"], "std over 25 folds": pr["pr_auc_std"], "ROC-AUC": pr["roc_auc_mean"]}).loc[["W2", "W2+calendar", "W2+E+H generic", "W2+E+H"]]
    t["paired"] = pd.DataFrame(m["paired_vs"]).T.rename(columns={"mean": "mean PR-AUC difference", "folds_positive": "folds positive"})
    rows = []
    for model in ["W2", "W2+E+H generic", "W2+E+H"]:
        for q in BUDGETS:
            b = m["alert_budget_25fold"][model][str(q)]
            rows.append({"model": model, "budget %": q, "insiders caught (of 70)": b["caught"][0], "normal users flagged (of 930)": b["normal flagged"][0]})
    t["budget"] = pd.DataFrame(rows)
    t["time_split"] = pd.DataFrame(m["time_split"]).rename(columns={"pr_auc": "PR-AUC (time split)", "R": "retention R", "user_split_same_rows": "user-split PR-AUC, same rows"})
    t["double_holdout"] = pd.DataFrame(m["double_holdout"]).rename(columns={"pr_auc_pooled": "PR-AUC (unseen users x future weeks)", "R": "retention R", "user_split_same_rows": "user-split PR-AUC, same rows"})
    t["lists"] = pd.concat({"global top q% of all weeks (what the alert list shows)": pd.DataFrame(m["ui_global_lists"]).T,
                            "top q% inside each calendar week (a real weekly queue)": pd.DataFrame(m["ui_weekly_queue_lists"]).T}, names=["queue type", "budget %"])
    return t


def limitations(m):
    """Plain-language limitations, with the numbers taken from the saved metrics."""
    ts = {(r["T"], r["set"]): r for r in m["time_split"]}
    lk, dl = m["marker_overlap"]["leak"], m["marker_overlap"]["download"]
    return [
        "Synthetic data: CERT r4.2 is a generated dataset; results say nothing certain about a real organisation.",
        f"Near-label markers: wikileaks.org has {lk['visitors']} visitors and {lk['visitors that are scenario insiders']} of them are scenario-{lk['scenario']} insiders "
        f"({lk['normal-user visitors']} normal users); the keylogger domains have {dl['visitors']} visitors, {dl['visitors that are scenario insiders']} are scenario-{dl['scenario']} insiders "
        f"({dl['normal-user visitors']} normal). A named-site explanation is therefore close to a label here.",
        f"Cold start: with a training period that holds few attacks the model is weak. At the earliest cutoff (T=29, {ts[(29, 'W2+E+H')]['train_attack_weeks']} training attack weeks) "
        f"the time-split PR-AUC is {ts[(29, 'W2+E+H')]['pr_auc']:.2f} against {ts[(29, 'W2+E+H')]['user_split_same_rows']:.2f} under user-split CV on the same weeks.",
        "Evaluation protocol: the scores shown are out-of-fold from 5x5 user-split CV (no row is scored by a model that saw its user); they are averaged over the 5 repeats, which is not what one deployed model would output.",
        "A global top-q% is easier than a real weekly queue (see the first table under 'All the numbers' below): a weekly queue of the same size is less precise.",
        "Labels are noisy: any week overlapping an attack window counts as an attack week, and most windows are longer than 30 days.",
        "The first four active weeks of each user are not scored (history needs four weeks).",
        "Explanations are XGBoost TreeSHAP contributions grouped into feature groups: they show what pushed a score up, not why someone is malicious; correlated features share credit.",
        "Graph models (user-PC, domain and email edges) added nothing over these features, so the app uses the tabular model; Neo4j is only used to browse neighbourhoods.",
    ]


BANNED_AGGREGATE_KEYS = {"user", "users", "label", "labels", "scenario", "window", "window_start", "window_end", "week_start", "insider", "insiders", "truth", "is_insider"}


def assert_aggregates_only(obj, path="$", max_list=64):
    """Raise if a JSON-able object holds anything per-row: banned key names (user, label, scenario, window, week...), long lists, or user-id-like strings.
    Used on every file the Overview reads (budget curve, rigor facts) and by the tests."""
    import re

    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in BANNED_AGGREGATE_KEYS:
                raise AssertionError(f"{path}.{k}: per-row field name")
            assert_aggregates_only(v, f"{path}.{k}", max_list)
    elif isinstance(obj, list):
        if len(obj) > max_list:
            raise AssertionError(f"{path}: list of {len(obj)} items looks per-row")
        for i, v in enumerate(obj):
            assert_aggregates_only(v, f"{path}[{i}]", max_list)
    elif isinstance(obj, str) and re.fullmatch(r"[A-Z]{3}\d{4}", obj):
        raise AssertionError(f"{path}: looks like a user id ({obj})")


GROUP_SHORT = GROUP_COLS   # old name, same map


def with_percentiles(scores):
    """Adds pct / pct_g (rank percentile 0-100 of the score among ALL scored weeks) and z / z_g (log-odds) for the final and the no-named-site score."""
    s = scores.copy()
    for col, p, z in (("score", "pct", "z"), ("score_generic", "pct_g", "z_g")):
        s[p] = s[col].rank(pct=True) * 100
        v = s[col].to_numpy(float).clip(1e-6, 1 - 1e-6)
        s[z] = np.log(v / (1 - v))
    return s


def reasons(row, k=2):
    """The k kinds of behaviour pushing this week's score up the most, in plain words (positive contributions only)."""
    top = sorted(((GROUP_SHORT[c], float(row[c])) for c in GROUP_SHORT), key=lambda t: -t[1])[:k]
    return [f"{n} +{v:.1f}" for n, v in top if v > 0] or ["nothing stands out"]


def top_label(pct):
    """'top 0.034%' from a percentile (0-100, share of weeks that score lower)."""
    return f"top {max(100 - float(pct), 0.0001):.2g}%"


def _alert_cols(a, floor):
    return {"Position": [top_label(p) for p in a["pct"]], "Why": [reasons(r) for _, r in a.iterrows()],
            "Needs named sites": ["yes" if g < floor else "" for g in a["pct_g"]]}   # would not rank in the same top-q without the named-site features


def alerts_by_week(alerts, show_sd=False):
    """One row per alerted week: rank, person, week, risk (log-odds), position among all weeks, main reasons, optional spread."""
    floor = alerts["pct"].min()
    c = _alert_cols(alerts, floor)
    t = pd.DataFrame({"Rank": np.arange(1, len(alerts) + 1), "Person": alerts["user"].to_numpy(), "Week": alerts["week_start"].to_numpy(), "Risk": alerts["z"].round(2).to_numpy(), **c})
    if show_sd:
        t["Spread over 5 repeats"] = alerts["score_sd"].round(4).to_numpy()
    return t


def alerts_by_person(alerts):
    """One row per person: number of flagged weeks, their worst week (highest score) with its risk, position and main reasons. Rank follows the worst week."""
    floor = alerts["pct"].min()
    a = alerts.sort_values(["score", "user", "week_start"], ascending=[False, True, True])
    n = a.groupby("user").size()
    w = a.drop_duplicates("user").reset_index(drop=True)
    c = _alert_cols(w, floor)
    return pd.DataFrame({"Rank": np.arange(1, len(w) + 1), "Person": w["user"], "Flagged weeks": n[w["user"]].to_numpy(), "Worst week": w["week_start"], "Risk": w["z"].round(2), **c})


USUAL_ROWS = [("n_logon", "Logons"), ("n_after_hours", "Logons outside 07:00-18:00"), ("n_device", "Removable-drive connections"), ("n_file", "Files copied to removable media"),
              ("em_n", "Emails sent"), ("em_n_ext_recip", "Recipients outside the company"), ("wb_n", "Web pages visited"),
              ("wb_n_named", "Named-site visits (leak, job, cloud, keylogger, competitor)")]
NAMED_CATEGORY = {"leak": "leak site", "download": "keylogger site", "job": "job site", "cloud": "cloud storage", "competitor": "competitor"}


def usual_marker(value, usual):
    """Plain comparison with the person's own usual (past-only average): far above = at least double and at least 3 more."""
    d = value - usual
    if d >= 3 and value >= 2 * usual:
        return "▲ far above usual"
    if d >= 1 and value > 1.25 * usual:
        return "↑ above usual"
    if -d >= 1 and value < 0.75 * usual:
        return "↓ below usual"
    return "about usual"


def named_sites_visited(neigh):
    """The named sites (leak, job, cloud, keylogger, competitor) a person visited in one week, with visit counts, from the neighbourhood edge frame."""
    e = neigh[(neigh["edge_type"] == "VISITED") & neigh["category"].isin(NAMED_CATEGORY)]
    t = pd.DataFrame({"Site": e["other_id"], "Category": e["category"].map(NAMED_CATEGORY), "Visits": e["weight"].astype(float)})
    return t.sort_values(["Visits", "Site"], ascending=[False, True]).reset_index(drop=True)


def week_vs_usual(facts_row):
    """'This week vs their usual' table from one ui_week_facts row (current count and the person's past-only mean)."""
    rows = [{"Activity": label, "This week": float(facts_row[k]), "Their usual": float(facts_row[f"hist_mean_{k}"]), "Compared with usual": usual_marker(float(facts_row[k]), float(facts_row[f"hist_mean_{k}"]))} for k, label in USUAL_ROWS]
    return pd.DataFrame(rows)
