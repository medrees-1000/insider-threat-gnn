import numpy as np
import pandas as pd
import pytest

from itgnn.evaluation import evaluate_scores, make_folds, summarize

USERS = [f"U{i}" for i in range(100)]
LABELS = {f"U{i}" for i in range(10)}


def test_folds_disjoint_and_cover_within_repeat():
    folds = make_folds(USERS, LABELS, 5, 3, seed=0)
    assert len(folds) == 15
    for r in range(3):
        tests = [f["test"] for f in folds if f["repeat"] == r]
        flat = [u for t in tests for u in t]
        assert sorted(flat) == sorted(USERS)  # disjoint and complete
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        assert sorted(f["train"] + f["test"]) == sorted(USERS)


def test_folds_balance_malicious_counts():
    for f in make_folds(USERS, LABELS, 5, 2, seed=1):
        assert sum(u in LABELS for u in f["test"]) == 2


def test_folds_reproducible_and_seed_dependent():
    a = make_folds(USERS, LABELS, 5, 2, seed=7)
    assert a == make_folds(list(reversed(USERS)), LABELS, 5, 2, seed=7)
    assert a != make_folds(USERS, LABELS, 5, 2, seed=8)


def test_metrics_hand_example():
    m = evaluate_scores([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.2, 0.1], k=2)
    assert m["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)
    assert m["roc_auc"] == pytest.approx(5 / 6)
    assert m["precision_at_k"] == pytest.approx(0.5)
    assert m["recall_at_k"] == pytest.approx(0.5)


def test_ties_use_expected_value():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # prevalence 0.2
    m = evaluate_scores(y, np.ones(10), k=4)
    assert m["precision_at_k"] == pytest.approx(0.2)
    assert m["recall_at_k"] == pytest.approx(0.4)
    # boundary inside a tie: top score is a positive, then 3 tied (1 pos of 3), k=3
    m = evaluate_scores([1, 1, 0, 0], [1.0, 0.5, 0.5, 0.5], k=3)
    assert m["precision_at_k"] == pytest.approx((1 + 2 * 1 / 3) / 3)


def test_single_class_raises():
    with pytest.raises(ValueError):
        evaluate_scores([0, 0], [0.1, 0.2], k=1)


def test_summarize():
    df = pd.DataFrame({"model": ["a", "a", "b", "b"], "pr_auc": [0.2, 0.4, 0.5, 0.5]})
    s = summarize(df)
    assert s.loc["a", ("pr_auc", "mean")] == pytest.approx(0.3)
    assert s.loc["b", ("pr_auc", "std")] == pytest.approx(0.0)
