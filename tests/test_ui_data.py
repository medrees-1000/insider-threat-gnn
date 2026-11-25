import ast
from pathlib import Path

import numpy as np
import pandas as pd

from itgnn import ui_data as U


def test_rank_alerts_global_weekly_and_ties(ui_results):
    sc = U.load_scores(ui_results)
    for q in U.BUDGETS:
        assert len(U.rank_alerts(sc, q)) == int(np.ceil(q / 100 * len(sc)))
    a = U.rank_alerts(sc, 25)
    assert a["score"].is_monotonic_decreasing and set(a["score"]) <= set(sc["score"])
    w = U.rank_alerts(sc, 50, week=22)
    assert (w["cal_week"] == 22).all() and len(w) == 2  # 3 rows in that week -> ceil(1.5) = 2
    tied = sc.assign(score=0.5)
    assert U.rank_alerts(tied, 10).equals(U.rank_alerts(tied, 10))  # deterministic ties
    cuts = U.budget_cutoffs(sc)
    assert cuts[0.25] >= cuts[2.0]


def test_explanations_and_alert_table_carry_no_labels(ui_results):
    sc = U.load_scores(ui_results)
    row = sc.iloc[0]
    top = U.top_groups(row, 3)
    assert len(top) == 3 and top[0][1] >= top[1][1] >= top[2][1] and top[0][0] in U.GROUP_COLS.values()
    t = U.alert_table(U.rank_alerts(sc, 25))
    assert not any("label" in c or "truth" in c or "scenario" in c for c in t.columns) and "score_generic" in t.columns and "score_sd" in t.columns
    assert not any("label" in c for c in sc.columns)  # ui_scores itself is label-free
    assert list(U.user_timeline(sc, "U1")["week_start"]) == sorted(U.user_timeline(sc, "U1")["week_start"])


def test_parquet_source_and_neo4j_row_normalisation(ui_results):
    src = U.ParquetSource(ui_results)
    w = pd.read_parquet(ui_results / "ui_weeks.parquet").iloc[0]
    nb = src.neighbourhood(w["user"], w["week_start"])
    assert list(nb.columns) == U.EDGE_COLS and len(nb) == 5 and src.has_week(w["user"], w["week_start"]) and not src.has_week("nobody", w["week_start"])
    assert nb.loc[nb["other_id"] == "wikileaks.org", "category"].iloc[0] == "leak"
    rows = [dict(edge_type=r.edge_type, other_id=r.other_id, other_kind=r.other_kind, weight=r.weight, category=(r.category or None), other_users=(None if pd.isna(r.other_users) else r.other_users))
            for r in nb.itertuples()]
    pd.testing.assert_frame_equal(U.normalise_rows(rows[::-1]), nb, check_dtype=False)  # same frame whatever order Neo4j returns rows in


def test_metrics_tables_and_limitations_use_the_saved_numbers(ui_results):
    m = U.load_metrics(ui_results)
    t = U.metrics_tables(m)
    assert set(t) == {"pr_auc", "paired", "budget", "time_split", "double_holdout", "lists"} and len(t["budget"]) == 12
    text = " ".join(U.limitations(m))
    assert "30 visitors" in text and "43 training attack weeks" in text and "weekly queue" in text
    assert U.load_metrics(ui_results / "nowhere") is None


def test_ui_modules_do_not_train_or_write():
    banned = {"xgboost", "sklearn", "torch", "torch_geometric"}
    banned_modules = {"itgnn.ui_export", "itgnn.comms_graph", "itgnn.comms_eval", "itgnn.weekly_gnn"}
    for f in ["app/streamlit_app.py", "src/itgnn/ui_data.py", "src/itgnn/neo4j_ui.py"]:
        src = Path(f).read_text()
        full = {n for node in ast.walk(ast.parse(src)) for n in (
            [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])}
        assert not {n.split(".")[0] for n in full} & banned and not full & banned_modules, (f, full)
    # the app only reads: no training, fitting, writing or opening files (checked on the syntax tree, not on text)
    forbidden_attrs = {"fit", "train", "to_parquet", "to_csv", "write_text", "write_bytes", "save_model", "XGBClassifier", "XGBRegressor"}
    for f in ["app/streamlit_app.py", "app/threat_map.py", "src/itgnn/whatif.py"]:
        for node in ast.walk(ast.parse(Path(f).read_text())):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert name not in forbidden_attrs, (f, name)
                if name == "open" and isinstance(node.func, ast.Name):   # reading is fine; writing (mode w/a/x) is not
                    mode = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)] + [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                    assert not any(set(str(m)) & set("wax+") for m in mode), (f, mode)


def test_percentiles_alert_frame_and_reasons(ui_results):
    sp = U.with_percentiles(U.load_scores(ui_results))
    assert sp["pct"].between(0, 100).all() and sp["pct"].iloc[sp["score"].argmax()] == 100 and sp["pct"].iloc[sp["score"].argmin()] < 5
    assert (np.sign(sp["z"].diff().fillna(0)) == np.sign(sp["score"].diff().fillna(0))).mean() > 0.9     # log-odds order follows the score
    top = U.rank_alerts(sp, 25)
    for display, col in (("Percentile", "pct"), ("Log-odds", "z"), ("Raw score", "score")):
        t = U.alerts_frame(top, display)
        assert list(t.columns) == ["Person", "Week", "Risk", "Without named sites", "Why"] and len(t) == len(top)
        assert np.allclose(t["Risk"], top[col].round(4 if display == "Raw score" else 2))
    assert "Spread over 5 repeats" in U.alerts_frame(top, "Percentile", show_sd=True).columns
    why = U.alerts_frame(top)["Why"].iloc[0]
    assert 1 <= len(why) <= 2 and all(isinstance(x, str) for x in why)
    assert not any("label" in c.lower() or "truth" in c.lower() for c in U.alerts_frame(top).columns)      # the default table carries no label columns
