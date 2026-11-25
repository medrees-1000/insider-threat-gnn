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


@pytest.fixture
def ui_results_full(ui_results):
    """ui_results plus the Stage 2 files: budget curve, rigor facts, and tiny final models trained on synthetic features."""
    import xgboost as xgb

    from itgnn import whatif as W
    from itgnn.commslogs import comms_feature_sets
    from itgnn.ui_export import GENERIC, MAIN, explanation_groups
    from itgnn.weekly_features import FEATURE_SETS

    rng = np.random.default_rng(1)
    sets = comms_feature_sets(FEATURE_SETS["W2"])
    fm, fg = sets[MAIN], sets[GENERIC]
    X = pd.DataFrame(np.abs(rng.normal(3, 2, size=(400, len(fm)))), columns=fm)
    y = (X["n_after_hours"] + X["wb_n_leak"] + rng.normal(0, 1, 400) > 8).astype(int)
    for name, cols, f in (("main", fm, "final_xgb_main.json"), ("generic", fg, "final_xgb_generic.json")):
        m = xgb.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0).fit(X[cols], y)
        m.get_booster().save_model(str(ui_results / f))
    sc = pd.read_parquet(ui_results / "ui_scores.parquet")
    grid = np.r_[np.arange(0, 0.99, 0.01), np.arange(0.99, 1.00001, 0.001)]
    q = {k: {"grid": [float(g) for g in grid], "values": [float(v) for v in np.quantile(sc[k].to_numpy(float), grid)]} for k in ("score", "score_generic")}
    meta = {"features_main": fm, "features_generic": fg, "groups": explanation_groups(fm), "medians": {c: float(np.median(X[c])) for c in fm},
            "p01": {c: float(np.quantile(X[c], 0.01)) for c in fm}, "p99": {c: float(np.quantile(X[c], 0.99)) for c in fm},
            "knob_max": {k: 30 for k in W.LABELS}, "budget_cutoffs": {"0.25": 0.95, "0.5": 0.9, "1.0": 0.8, "2.0": 0.7}, "oof_quantiles": q, "sliders": W.SLIDERS}
    json.dump(meta, open(ui_results / "final_model_meta.json", "w"))
    rows = X.head(5).copy(); rows.insert(0, "week_start", pd.date_range("2010-06-07", periods=5, freq="7D")); rows.insert(0, "user", [f"U{i}" for i in range(5)])
    rows.to_parquet(ui_results / "final_model_rows.parquet", index=False)
    from itgnn.ui_export import WEEK_FACT_COLS
    wf = sc[["user", "week_start"]].copy()
    for k in WEEK_FACT_COLS:
        wf[k] = rng.integers(0, 12, len(wf)).astype("float32"); wf[f"hist_mean_{k}"] = np.float32(2.0)
    wf.to_parquet(ui_results / "ui_week_facts.parquet", index=False)
    grid_b = [0.1, 0.25, 0.5, 1.0, 2.0]
    sets_c = {k: {"caught": [40.0, 50.0, 55.0, 60.0, 65.0], "caught_sd": [1.0] * 5, "normal_flagged": [0.0, 1.0, 3.0, 9.0, 30.0], "normal_flagged_sd": [1.0] * 5, "weeks_flagged": [60.0, 150.0, 300.0, 600.0, 1200.0]}
              for k in ("W2", "W2+E+H generic", "W2+E+H")}
    json.dump({"budgets": grid_b, "n_insiders": 70, "n_normal_users": 930, "n_scored_weeks": 63238, "sets": sets_c}, open(ui_results / "ui_budget_curve.json", "w"))
    facts = {"tenure_leak": {"active_days_auc_for_insider": 0.053, "typical_full_tenure_days": 346, "normal_users_covering_the_full_period": 0.755, "insiders_covering_the_full_period": 0.0},
             "cross_validation": {"folds": 25, "repeats": 5, "max_users_shared_by_train_and_test": 0},
             "causality_check": {"user_weeks_checked": 200, "features_compared": 79, "max_abs_difference": 0.0, "control_rows_that_differ": 18, "control_rows": 18},
             "graph_controls": {"no_edge_mlp": 0.708, "graphsage": 0.718, "graphsage_shuffled_edges": 0.717, "xgboost_same_features": 0.879, "folds": 5, "gate_passed": False}}
    json.dump(facts, open(ui_results / "ui_rigor_facts.json", "w"))
    m = json.load(open(ui_results / "ui_metrics.json")); m.update({"n_scored_weeks": 63238, "attack_weeks": 322})
    m["time_split"] = [{"T": t, "set": "W2+E+H", "pr_auc": 0.5, "user_split_same_rows": 0.8, "R": r, "train_attack_weeks": 43, "test_attack_weeks": 279} for t, r in ((29, 0.6), (39, 0.85), (49, 0.97))]
    m["ui_global_lists"]["0.25"].update({"alerts": 159, "insiders caught": 51}); m["alert_budget_25fold"]["W2+E+H"]["0.25"] = {"caught": [52.8, 3.0], "normal flagged": [0.2, 0.4]}
    json.dump(m, open(ui_results / "ui_metrics.json", "w"))
    return ui_results
