"""Step 7 helpers: the cached weekly frame with email/web features, time-based splits, user-cluster bootstrap."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from itgnn.commslogs import add_comms_features
from itgnn.weekly_features import add_history_features
from itgnn.windows import label_weeks, make_weekly_features

CACHE = "data/processed/comms_weekly_full.parquet"


def build_full(d, windows):
    """All user-week rows (warm-up included) with label, cal_week, W2 history and the email/web features + history."""
    full = make_weekly_features(d["logon"], d["device"], d["file"], d["users"])
    full["label"] = label_weeks(full, windows)
    full["cal_week"] = (full["week_start"] - full["week_start"].min()).dt.days // 7
    full = add_history_features(full, d["logon"])
    email = pd.read_parquet("data/processed/email.parquet")
    agg = pd.read_parquet("data/processed/http_user_week_domain.parquet")
    full, _ = add_comms_features(full, email, agg)
    return full


def load_full(d, windows, cache=CACHE):
    """Cached `build_full` (delete the parquet to rebuild)."""
    if Path(cache).exists():
        return pd.read_parquet(cache)
    full = build_full(d, windows)
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(cache, index=False)
    return full


def time_split_masks(weekly, T):
    """(train, test) row masks: calendar week <= T vs > T. The same users may be on both sides."""
    return (weekly["cal_week"] <= T).to_numpy(), (weekly["cal_week"] > T).to_numpy()


def double_holdout_masks(weekly, fold, T):
    """Training-fold users x weeks <= T for training; test-fold users x weeks > T for testing."""
    tr_t, te_t = time_split_masks(weekly, T)
    return tr_t & weekly["user"].isin(set(fold["train"])).to_numpy(), te_t & weekly["user"].isin(set(fold["test"])).to_numpy()


def user_bootstrap_ap(y, scores, users, n_boot=200, seed=42):
    """PR-AUC of every score vector on `n_boot` bootstrap resamples of USERS (all their rows move together).

    scores: dict name -> array aligned with y. Returns a frame (n_boot x names); the same resample is used for
    every name, so differences between names are paired. Resamples without a positive are redrawn.
    """
    y, users = np.asarray(y), np.asarray(users)
    ids, inv = np.unique(users, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(ids))]
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_boot:
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(ids), len(ids))])
        if y[idx].sum() == 0:
            continue
        out.append({k: average_precision_score(y[idx], np.asarray(v)[idx]) for k, v in scores.items()})
    return pd.DataFrame(out)


def same_rows_user_split_ap(oof, model, T):
    """Reference for the time split: PR-AUC under user-split CV on the rows with calendar week > T.

    Per repeat the test predictions of all folds are pooled (each user once); returns the mean over repeats.
    """
    g = oof[(oof["model"] == model) & (oof["cal_week"] > T)]
    return float(np.mean([average_precision_score(x["label"], x["score"]) for _, x in g.groupby("repeat")]))
