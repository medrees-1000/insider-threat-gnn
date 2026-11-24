import numpy as np
import pandas as pd
import pytest

from itgnn.commslogs import comms_feature_sets
from itgnn.evaluation import make_folds
from itgnn.ui_export import GENERIC, GROUP_KEYS, MAIN, aggregate_scores, eval_table, explanation_groups, fold_scores
from itgnn.weekly_features import FEATURE_SETS


@pytest.fixture(scope="module")
def synth():
    sets = comms_feature_sets(FEATURE_SETS["W2"])
    sets = {MAIN: sets[MAIN], GENERIC: sets[GENERIC]}
    rng = np.random.default_rng(0)
    users = [f"u{i:02d}" for i in range(40)]
    insiders = set(users[:10])
    weeks = pd.date_range("2010-01-04", periods=10, freq="7D")
    rows = [(u, w) for u in users for w in weeks]
    df = pd.DataFrame(rows, columns=["user", "week_start"])
    for c in sets[MAIN]:
        df[c] = rng.normal(size=len(df))
    df["label"] = (df["user"].isin(insiders) & (rng.random(len(df)) < 0.5)).astype(int)
    df.loc[df["label"] == 1, "wb_n_leak"] += 3  # a learnable signal
    folds = make_folds(users, insiders, 2, 3, 7)
    return df, sets, folds, insiders


def test_explanation_groups_are_a_partition():
    cols = comms_feature_sets(FEATURE_SETS["W2"])[MAIN]
    g = explanation_groups(cols)
    flat = [c for v in g.values() for c in v]
    assert len(flat) == len(set(flat)) == len(cols) and list(g) == list(GROUP_KEYS)
    assert "wb_n_leak" in g["named-site"] and "dev_n_logon" in g["W2 history/deviation"] and "n_new_pcs" in g["W2 base"]


def test_no_row_is_scored_by_a_model_that_trained_on_it(synth):
    df, sets, folds, _ = synth
    long = fold_scores(df, folds, sets, seed=1, n_estimators=15)
    test_of = {(f["repeat"], f["fold"]): set(f["test"]) for f in folds}
    train_of = {(f["repeat"], f["fold"]): set(f["train"]) for f in folds}
    for (r, fo), g in long.groupby(["repeat", "fold"]):
        assert set(g["user"]) <= test_of[(r, fo)] and not set(g["user"]) & train_of[(r, fo)]
    per = long[long["set"] == MAIN].groupby(["user", "week_start"]).size()
    assert (per == 3).all()  # exactly once per repeat


def test_contributions_sum_to_the_margin_and_the_score(synth):
    df, sets, folds, _ = synth
    m = fold_scores(df, folds, sets, seed=1, n_estimators=15).query("set == @MAIN")
    s = m[list(GROUP_KEYS.values()) + ["bias"]].sum(axis=1)
    assert np.allclose(s, m["margin"], atol=1e-4)
    assert np.allclose(1 / (1 + np.exp(-m["margin"])), m["score"], atol=1e-5)


def test_aggregate_and_eval_table_keep_labels_apart(synth, tmp_path):
    df, sets, folds, insiders = synth
    long = fold_scores(df, folds, sets, seed=1, n_estimators=15)
    a = aggregate_scores(long)
    assert len(a) == len(df) and a["n_repeats"].eq(3).all() and a["score_generic"].notna().all()
    assert not any("label" in c for c in a.columns)
    win = pd.DataFrame({"user": sorted(insiders), "start": pd.Timestamp("2010-02-01"), "end": pd.Timestamp("2010-02-20")})
    import io
    ins = pd.read_csv(io.StringIO("dataset,scenario,user\n" + "\n".join(f"4.2,1,{u}" for u in sorted(insiders))))
    ins_path = tmp_path / "ins.csv"; ins.to_csv(ins_path, index=False)
    e = eval_table(df, win, ins_path)
    assert e["scenario"].gt(0).sum() == df["user"].isin(insiders).sum() and e["label"].equals(df["label"])
