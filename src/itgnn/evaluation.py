"""Repeated cross-validation folds over users and ranking metrics for model comparison."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def make_folds(users, labels, n_splits, n_repeats, seed):
    """Repeated stratified k-fold over users. Deterministic for a given seed.

    Returns a list of {repeat, fold, train, test}; train/test are lists of user ids.
    Within one repeat the test folds are disjoint and cover every user.
    """
    users = sorted(users)
    y = np.array([u in labels for u in users])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    arr = np.array(users)
    return [
        {"repeat": i // n_splits, "fold": i % n_splits, "train": list(arr[tr]), "test": list(arr[te])}
        for i, (tr, te) in enumerate(rskf.split(arr, y))
    ]


def _expected_tp_at_k(y_true, y_score, k):
    """Expected positives in the top k when ties at the k boundary are broken at random."""
    order = np.argsort(-y_score, kind="stable")
    y, s = y_true[order], y_score[order]
    k = min(k, len(s))
    boundary = s[k - 1]
    above = s > boundary
    tied = s == boundary
    n_above = above.sum()
    return y[above].sum() + (k - n_above) * y[tied].mean(), k


def evaluate_scores(y_true, y_score, k):
    """PR-AUC (average precision), ROC-AUC, precision@k and recall@k.

    precision@k and recall@k are tie-aware: if the k boundary falls inside a group of
    equal scores, the expected value under random tie-breaking is used.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() in (0, len(y_true)):
        raise ValueError("y_true needs both classes")
    if k < 1:
        raise ValueError("k must be >= 1")
    tp, k = _expected_tp_at_k(y_true, y_score, k)
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision_at_k": tp / k,
        "recall_at_k": tp / y_true.sum(),
    }


def summarize(results_df):
    """Mean and std (sample, ddof=1) of every metric column per model.

    results_df has one row per (model, repeat, fold) with a `model` column and metric columns.
    """
    metrics = [c for c in results_df.columns if c in
               ("pr_auc", "roc_auc", "precision_at_k", "recall_at_k")]
    return results_df.groupby("model")[metrics].agg(["mean", "std"])


def alert_budget(oof, labels, scenario_of, qs, seed):
    """Alert-budget table from out-of-fold scores (same logic as notebooks 04 and 05).

    oof needs model, repeat, fold, user, score, label. In every (model, repeat, fold) the top
    ceil(q% of rows) are flagged (ties broken at random, rng seeded with seed + repeat); the folds of
    a repeat are then pooled so each user appears once. Returns one row per (model, q%, repeat):
    caught (insiders with >=1 flagged attack week), normal flagged (distinct users not in labels),
    weeks flagged, and per scenario s: `caught s{s}` and `attack weeks s{s}` (flagged attack weeks).
    """
    labels = set(labels)
    scens = sorted(set(scenario_of.values()))
    rows = []
    for (name, r), g in oof.groupby(["model", "repeat"]):
        rng = np.random.default_rng(seed + r)
        parts = {q: [] for q in qs}
        for _, gf in g.groupby("fold"):
            order = np.lexsort((rng.random(len(gf)), -gf["score"].to_numpy()))
            for q in qs:
                parts[q].append(gf.iloc[order[: int(np.ceil(q / 100 * len(gf)))]])
        for q in qs:
            fl = pd.concat(parts[q])
            att = fl[fl["label"] == 1].assign(scenario=lambda x: x["user"].map(scenario_of))
            row = {"model": name, "q%": q, "repeat": r, "caught": att["user"].nunique(),
                   "normal flagged": fl.loc[~fl["user"].isin(labels), "user"].nunique(), "weeks flagged": len(fl)}
            for s in scens:
                row[f"caught s{s}"] = att.loc[att["scenario"] == s, "user"].nunique()
                row[f"attack weeks s{s}"] = int((att["scenario"] == s).sum())
            rows.append(row)
    return pd.DataFrame(rows)
