from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itgnn.evaluation import alert_budget


def test_alert_budget_small_case():
    # 1 repeat, 1 fold, 10 rows; q=25 -> top ceil(2.5)=3 rows flagged
    oof = pd.DataFrame({"model": "m", "repeat": 0, "fold": 0,
                        "user": ["I1", "I1", "I2", "N1", "N1", "N2", "N3", "N4", "N5", "N6"],
                        "score": [0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.05],
                        "label": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]})
    out = alert_budget(oof, ["I1", "I2"], {"I1": 1, "I2": 2}, [25, 100], seed=0)
    r = out.set_index("q%").loc[25]
    assert (r["caught"], r["normal flagged"], r["weeks flagged"], r["caught s1"], r["caught s2"], r["attack weeks s1"]) == (2, 1, 3, 1, 1, 1)
    assert out.set_index("q%").loc[100, "normal flagged"] == 6
    pd.testing.assert_frame_equal(out, alert_budget(oof, ["I1", "I2"], {"I1": 1, "I2": 2}, [25, 100], seed=0))  # deterministic


NB4 = {  # notebook 04 output (mean, std over 5 repeats) for RF W2 and XGB W2: caught, normal flagged
    "RF W2": {0.25: ((51.4, 2.3), (3.2, 0.4)), 0.5: ((56.6, 1.7), (9.2, 1.1)), 1: ((61.2, 0.8), (18.2, 3.6)), 2: ((65.4, 0.9), (35.6, 4.6))},
    "XGB W2": {0.25: ((48.0, 3.1), (2.8, 0.8)), 0.5: ((52.6, 0.9), (5.4, 0.5)), 1: ((56.6, 1.3), (10.2, 1.8)), 2: ((62.6, 1.3), (30.2, 6.8))},
}


def test_alert_budget_reproduces_notebook_04():
    path, raw = Path("results/weekly_v2_oof_scores.parquet"), Path("data/raw/answers/insiders.csv")
    if not (path.exists() and raw.exists()):
        pytest.skip("saved notebook-04 scores or raw answers not available")
    ins = pd.read_csv(raw)
    ins = ins[ins["dataset"].astype(str) == "4.2"]
    oof = pd.read_parquet(path)
    oof = oof[oof["model"].isin(NB4)]
    out = alert_budget(oof, ins["user"], dict(zip(ins["user"], ins["scenario"])), list(NB4["RF W2"]), seed=42)
    agg = out.groupby(["model", "q%"])[["caught", "normal flagged"]].agg(["mean", "std"])
    for model, by_q in NB4.items():
        for q, (c, n) in by_q.items():
            row = agg.loc[(model, q)]
            assert (round(row[("caught", "mean")], 1), round(row[("caught", "std")], 1)) == c
            assert (round(row[("normal flagged", "mean")], 1), round(row[("normal flagged", "std")], 1)) == n
