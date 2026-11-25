"""Pure helpers behind the analyst UI (no Streamlit, no training libraries): reading saved results, alert lists,
timelines, explanations and the neighbourhood backends. The app only calls these."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLS = {  # parquet column -> label shown to the analyst
    "w2_base": "W2 base counts", "w2_hist_dev": "W2 history/deviation", "email_volume": "email volume",
    "email_recipients": "email recipients/attachments", "web_volume": "web volume", "web_domains": "web domain counts",
    "web_after_hours": "web after-hours", "named_site": "named-site (category domains)", "first_seen": "first-seen"}
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
        "A global top-q% is easier than a real weekly queue (see the table above): a weekly queue of the same size is less precise.",
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
