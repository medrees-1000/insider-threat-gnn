import numpy as np
import pandas as pd

from itgnn.weekly_features import (DEV_COLS, FEATURE_SETS, N_HIST, add_graph_stats, add_history_features,
                                   drop_warmup, signed_log1p)
from itgnn.windows import make_weekly_features

T = pd.Timestamp
MONDAYS = pd.date_range("2010-01-04", periods=8, freq="7D")  # 8 consecutive Mondays


def _logon(rows):
    """rows: (user, pc, week index, hour)"""
    df = pd.DataFrame(
        {"user": [r[0] for r in rows], "pc": [r[1] for r in rows],
         "date": [MONDAYS[r[2]] + pd.Timedelta(hours=r[3], days=1) for r in rows], "activity": "Logon"}
    )
    return df


def _empty():
    return pd.DataFrame({"user": pd.Series(dtype=object), "date": pd.Series(dtype="datetime64[ns]")})


def _build(logon):
    users = set(logon["user"])
    w = make_weekly_features(logon, _empty(), _empty(), users)
    return add_graph_stats(add_history_features(w, logon), logon)


# U1 uses PC1 for weeks 0-5 (week k has k+1 logons), then PC2 in week 6. U2 shares PC1 in week 2 only.
BASE = (
    [("U1", "PC1", k, 9) for k in range(6) for _ in range(k + 1)]
    + [("U1", "PC2", 6, 9), ("U1", "PC1", 6, 9)]
    + [("U2", "PC1", 2, 10), ("U2", "PC9", 2, 10), ("U2", "PC9", 3, 10)]
)


def _row(df, user, wk):
    return df[(df["user"] == user) & (df["week_start"] == MONDAYS[wk])].iloc[0]


def test_history_and_new_pcs():
    f = _build(_logon(BASE))
    r = _row(f, "U1", 5)  # previous 4 active weeks: 1..4 -> n_logon 2,3,4,5? weeks 1-4 have 2,3,4,5 logons
    assert r["hist_mean_n_logon"] == np.mean([2, 3, 4, 5])
    assert r["dev_n_logon"] == 6 - 3.5
    assert _row(f, "U1", 0)["n_new_pcs"] == 1 and _row(f, "U1", 1)["n_new_pcs"] == 0
    assert _row(f, "U1", 6)["n_new_pcs"] == 1  # PC2 new, PC1 old
    assert _row(f, "U2", 3)["n_new_pcs"] == 0  # PC9 first used in week 2
    assert np.isnan(_row(f, "U1", 3)["hist_mean_n_logon"])  # only 3 earlier weeks


def test_graph_stats_same_week_only():
    f = _build(_logon(BASE))
    r = _row(f, "U1", 2)  # week 2: PC1 shared with U2
    assert (r["peers_mean"], r["peers_max"], r["share_solo_pcs"]) == (1, 1, 0)
    r2 = _row(f, "U2", 2)  # PC1 (1 peer), PC9 (0 peers)
    assert (r2["peers_mean"], r2["peers_max"], r2["share_solo_pcs"]) == (0.5, 1, 0.5)
    r3 = _row(f, "U1", 3)  # U2 was on PC1 in week 2, not week 3 -> no peers
    assert (r3["peers_mean"], r3["peers_max"], r3["share_solo_pcs"]) == (0, 0, 1)


def test_causality_and_positive_control():
    cols = [c for s in FEATURE_SETS.values() for c in s]
    before = _build(_logon(BASE))
    # change data in week 6 and 7 only: earlier weeks must be identical
    later = BASE + [("U2", "PC1", 6, 9), ("U2", "PC7", 7, 9), ("U1", "PC8", 7, 9), ("U1", "PC1", 6, 20)]
    after = _build(_logon(later))
    key = ["user", "week_start"]
    b = before[before["week_start"] < MONDAYS[6]].set_index(key)[cols].sort_index()
    a = after[after["week_start"] < MONDAYS[6]].set_index(key)[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)
    # positive control: the same kind of edit in the CURRENT week must change that week
    cur = _row(before, "U1", 6)
    new = _row(after, "U1", 6)
    assert new["n_logon"] != cur["n_logon"] and new["peers_mean"] != cur["peers_mean"]
    # and editing week 6 changes week 7's history but not week 6's history
    assert _row(after, "U1", 6)["hist_mean_n_logon"] == _row(before, "U1", 6)["hist_mean_n_logon"]


def test_drop_warmup_first_four_active_weeks():
    # U1 has a gap (week 3 missing); active weeks are 0,1,2,4,5,6,7 -> keep from the 5th active week (idx 5)
    rows = [("U1", "PC1", k, 9) for k in (0, 1, 2, 4, 5, 6, 7)] + [("U2", "PC1", k, 9) for k in range(3)]
    f = drop_warmup(_build(_logon(rows)))
    assert list(f.loc[f["user"] == "U1", "week_start"]) == [MONDAYS[k] for k in (5, 6, 7)]
    assert "U2" not in set(f["user"])  # fewer than N_HIST+1 active weeks
    assert N_HIST == 4
    assert f[sum(FEATURE_SETS.values(), [])].notna().all().all()  # no NaN after the warm-up


def test_signed_log1p():
    x = np.array([-3.5, 0.0, 2.0])
    out = signed_log1p(x)
    assert np.isfinite(out).all() and out[0] < 0 and out[1] == 0 and out[2] > 0
    assert all(c.startswith("dev_") for c in DEV_COLS) and len(DEV_COLS) == 5
