"""Extra weekly features: past-only history and same-week graph statistics.

Causality rule: the value for week w uses data from weeks <= w. History features use
strictly earlier weeks only; graph statistics use week w only. Predictions are made at
the end of week w, so all of week w is known.
"""
import numpy as np
import pandas as pd

from itgnn.windows import WEEK_FEATURE_COLS, _week_start

N_HIST = 4  # previous active weeks (weeks with a Logon) used for the history mean
HIST_COLS = [f"hist_mean_{c}" for c in WEEK_FEATURE_COLS] + [f"dev_{c}" for c in WEEK_FEATURE_COLS] + ["n_new_pcs"]
GRAPH_COLS = ["peers_mean", "peers_max", "share_solo_pcs"]
DEV_COLS = [c for c in HIST_COLS if c.startswith("dev_")]
FEATURE_SETS = {
    "W1": WEEK_FEATURE_COLS,
    "W2": WEEK_FEATURE_COLS + HIST_COLS,
    "W3": WEEK_FEATURE_COLS + GRAPH_COLS,
    "W4": WEEK_FEATURE_COLS + HIST_COLS + GRAPH_COLS,
}


def signed_log1p(x):
    """sign(x) * log1p(|x|); for deviations that can be negative."""
    return np.sign(x) * np.log1p(np.abs(x))


def _user_week_pcs(logon, weekly):
    lg = logon[(logon["activity"] == "Logon") & logon["user"].isin(set(weekly["user"]))]
    return (
        pd.DataFrame({"user": lg["user"].to_numpy(), "week_start": _week_start(lg["date"]).to_numpy(), "pc": lg["pc"].to_numpy()})
        .drop_duplicates()
    )


def add_history_features(weekly, logon):
    """Add hist_mean_*, dev_* and n_new_pcs. Returns a copy sorted by (user, week_start).

    hist_mean_* = mean over the user's previous N_HIST active rows (NaN if fewer exist, i.e. in
    the first N_HIST active weeks). dev_* = this week minus that mean. n_new_pcs = PCs used this
    week that the user never used in an earlier week.
    """
    w = weekly.sort_values(["user", "week_start"]).reset_index(drop=True)
    g = w.groupby("user")
    for c in WEEK_FEATURE_COLS:
        w[f"hist_mean_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(N_HIST, min_periods=N_HIST).mean())
        w[f"dev_{c}"] = w[c] - w[f"hist_mean_{c}"]
    pcs = _user_week_pcs(logon, w)
    first = pcs.groupby(["user", "pc"])["week_start"].transform("min")
    new = pcs[pcs["week_start"] == first].groupby(["user", "week_start"]).size().rename("n_new_pcs")
    w = w.merge(new, left_on=["user", "week_start"], right_index=True, how="left")
    w["n_new_pcs"] = w["n_new_pcs"].fillna(0).astype(int)
    return w


def add_graph_stats(weekly, logon):
    """Add peers_mean, peers_max, share_solo_pcs from the same week's logons only.

    For each PC the user logged on to that week, peers = number of OTHER users with a Logon on
    that PC in that week. Statistics are taken over the user's PCs.
    """
    pcs = _user_week_pcs(logon, weekly)
    pcs["peers"] = pcs.groupby(["week_start", "pc"])["user"].transform("nunique") - 1
    pcs["solo"] = pcs["peers"] == 0
    agg = pcs.groupby(["user", "week_start"]).agg(
        peers_mean=("peers", "mean"), peers_max=("peers", "max"), share_solo_pcs=("solo", "mean")
    )
    return weekly.merge(agg, left_on=["user", "week_start"], right_index=True, how="left")


def drop_warmup(weekly, n=N_HIST):
    """Drop each user's first n active weeks (rows must exist only for active weeks)."""
    w = weekly.sort_values(["user", "week_start"])
    return w[w.groupby("user").cumcount() >= n].reset_index(drop=True)
