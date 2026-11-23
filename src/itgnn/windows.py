"""User-per-week dataset: attack windows, weekly features and week labels.

Weeks are Monday-based 7-day blocks. Feature columns carry no tenure or calendar
information; `user` and `week_start` are ids only.
"""
from pathlib import Path

import pandas as pd

WEEK_FEATURE_COLS = ["n_logon", "n_after_hours", "n_pcs", "n_device", "n_file"]
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"  # month first: the 4.2 rows have first field <= 12, second up to 31


def load_attack_windows(raw_dir, release="4.2"):
    """user, start, end (Timestamps) per insider for one release. Raises on unparseable dates."""
    df = pd.read_csv(Path(raw_dir) / "answers" / "insiders.csv")
    df = df[df["dataset"].astype(str) == release]
    out = pd.DataFrame(
        {
            "user": df["user"].to_numpy(),
            "start": pd.to_datetime(df["start"].to_numpy(), format=DATE_FORMAT),  # strict: bad rows raise
            "end": pd.to_datetime(df["end"].to_numpy(), format=DATE_FORMAT),
        }
    )
    assert (out["end"] >= out["start"]).all(), "attack window ends before it starts"
    return out.reset_index(drop=True)


def _week_start(dates):
    d = dates.dt.normalize()
    return d - pd.to_timedelta(d.dt.day_of_week, unit="D")


def make_weekly_features(logon, device, file, users):
    """One row per (user, week_start) with at least one Logon; counts within that week only.

    Same definitions as user_features. Device/file events in weeks with no Logon are dropped.
    """
    lg = logon[(logon["activity"] == "Logon") & logon["user"].isin(users)].copy()
    lg["week_start"] = _week_start(lg["date"])
    lg["after_hours"] = ~lg["date"].dt.hour.between(7, 18)
    g = lg.groupby(["user", "week_start"])
    weekly = pd.DataFrame(
        {"n_logon": g.size(), "n_after_hours": g["after_hours"].sum(), "n_pcs": g["pc"].nunique()}
    )
    for col, df in [("n_device", device), ("n_file", file)]:
        df = df[df["user"].isin(users)]
        counts = df.groupby(["user", _week_start(df["date"]).rename("week_start")]).size()
        weekly[col] = counts.reindex(weekly.index).fillna(0)
    weekly = weekly.astype(int).reset_index()
    return weekly[["user", "week_start"] + WEEK_FEATURE_COLS]


def label_weeks(weekly_df, attack_windows):
    """Return a 0/1 Series: 1 if [week_start, week_start + 7d) overlaps a user's [start, end]."""
    m = weekly_df[["user", "week_start"]].reset_index(drop=True).reset_index().merge(
        attack_windows, on="user", how="left"
    )
    hit = (m["week_start"] <= m["end"]) & (m["week_start"] + pd.Timedelta(days=7) > m["start"])
    flag = hit.groupby(m["index"]).any()  # any window of the user; NaT comparisons give False
    return pd.Series(flag.to_numpy().astype(int), index=weekly_df.index, name="label")


def week_masks(weekly_df, fold):
    """Boolean (train, test) row masks: all weeks of a user fall on the same side."""
    test = weekly_df["user"].isin(set(fold["test"]))
    return ~test & weekly_df["user"].isin(set(fold["train"])), test
