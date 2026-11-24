import json

import numpy as np
import pandas as pd
import pytest

GROUPS = ["w2_base", "w2_hist_dev", "email_volume", "email_recipients", "web_volume", "web_domains", "web_after_hours", "named_site", "first_seen"]


@pytest.fixture
def ui_results(tmp_path):
    """A tiny, synthetic results folder with every file the UI reads."""
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2010-06-07", periods=8, freq="7D")
    rows = [(u, w, 20 + i) for u in ("U1", "U2", "U3") for i, w in enumerate(weeks)]
    sc = pd.DataFrame(rows, columns=["user", "week_start", "cal_week"])
    sc["score"] = np.round(rng.random(len(sc)), 4)
    sc["score_sd"] = 0.01
    sc["score_generic"] = np.round(rng.random(len(sc)), 4)
    for g in GROUPS:
        sc[g] = rng.normal(size=len(sc))
    sc["bias"] = -4.0
    sc[[c for c in sc.columns if c not in ("user", "week_start", "cal_week")]] = sc[[c for c in sc.columns if c not in ("user", "week_start", "cal_week")]].astype("float32")
    sc.to_parquet(tmp_path / "ui_scores.parquet", index=False)
    ev = sc[["user", "week_start"]].copy()
    ev["label"] = ((ev["user"] == "U1") & (ev["week_start"] >= weeks[3]) & (ev["week_start"] <= weeks[5])).astype(int)
    ev["scenario"] = np.where(ev["user"] == "U1", 2, 0).astype("int8")
    ev["window_start"] = np.where(ev["user"] == "U1", weeks[3], pd.NaT)
    ev["window_end"] = np.where(ev["user"] == "U1", weeks[5], pd.NaT)
    ev["window_start"], ev["window_end"] = pd.to_datetime(ev["window_start"]), pd.to_datetime(ev["window_end"])
    ev.to_parquet(tmp_path / "ui_eval.parquet", index=False)
    top = sc.sort_values("score", ascending=False).head(6).copy()
    top["key"] = [f"{u}|{w:%Y-%m-%d}" for u, w in zip(top["user"], top["week_start"])]
    top["top_groups"] = "named-site +1.00"
    top["n_external_recipients"] = 2
    top.to_parquet(tmp_path / "ui_weeks.parquet", index=False)
    edges = []
    for r in top.itertuples():
        edges += [(r.user, r.week_start, "LOGGED_ON", "PC-1", "PC", 4.0, "", 0.0), (r.user, r.week_start, "LOGGED_ON", "PC-2", "PC", 1.0, "", 3.0),
                  (r.user, r.week_start, "VISITED", "wikileaks.org", "Domain", 2.0, "leak", np.nan), (r.user, r.week_start, "VISITED", "news.com", "Domain", 9.0, "", np.nan),
                  (r.user, r.week_start, "EMAILED", "U9", "User", 5.0, "", np.nan)]
    pd.DataFrame(edges, columns=["user", "week_start", "edge_type", "other_id", "other_kind", "weight", "category", "other_users"]).to_parquet(tmp_path / "ui_edges.parquet", index=False)
    m = {"pr_auc_25fold": {k: {"pr_auc_mean": 0.5, "pr_auc_std": 0.1, "roc_auc_mean": 0.9} for k in ["W2", "W2+calendar", "W2+E+H generic", "W2+E+H"]},
         "paired_vs": {"W2+E+H - W2": {"mean": 0.4, "folds_positive": "25/25"}},
         "alert_budget_25fold": {mm: {str(q): {"caught": [50.0, 1.0], "normal flagged": [5.0, 1.0]} for q in [0.25, 0.5, 1.0, 2.0]} for mm in ["W2", "W2+E+H generic", "W2+E+H"]},
         "time_split": [{"T": 29, "set": s, "pr_auc": 0.5, "user_split_same_rows": 0.8, "R": 0.6, "train_attack_weeks": 43, "test_attack_weeks": 279} for s in ["W2", "W2+E+H", "W2+E+H generic"]],
         "double_holdout": [{"T": 29, "set": "W2+E+H", "pr_auc_pooled": 0.4, "user_split_same_rows": 0.8, "R": 0.5, "test_attack_weeks": 279}],
         "ui_global_lists": {str(q): {"alerts": 10, "attack weeks flagged": 5, "week precision": 0.5, "distinct users": 3, "insiders caught": 2, "normal users flagged": 1} for q in [0.25, 0.5, 1.0, 2.0]},
         "ui_weekly_queue_lists": {str(q): {"alerts": 11, "attack weeks flagged": 4, "week precision": 0.4, "distinct users": 3, "insiders caught": 2, "normal users flagged": 1} for q in [0.25, 0.5, 1.0, 2.0]},
         "marker_overlap": {"leak": {"visitors": 30, "scenario": 1, "scenario insiders": 30, "visitors that are scenario insiders": 30, "normal-user visitors": 0},
                            "download": {"visitors": 9, "scenario": 3, "scenario insiders": 10, "visitors that are scenario insiders": 9, "normal-user visitors": 0}}}
    json.dump(m, open(tmp_path / "ui_metrics.json", "w"))
    return tmp_path
