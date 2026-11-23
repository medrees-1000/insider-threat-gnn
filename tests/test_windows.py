import pandas as pd
import pytest

from itgnn.evaluation import make_folds
from itgnn.windows import (WEEK_FEATURE_COLS, label_weeks, load_attack_windows,
                           make_weekly_features, week_masks)

T = pd.Timestamp


def _wk(user, start):
    return {"user": user, "week_start": T(start)}


def _label(windows, weeks):
    w = pd.DataFrame(windows, columns=["user", "start", "end"])
    w[["start", "end"]] = w[["start", "end"]].apply(pd.to_datetime)
    return label_weeks(pd.DataFrame(weeks), w).tolist()


def test_overlap_boundaries():
    # week = Mon 2010-01-11 00:00 up to (not including) Mon 2010-01-18 00:00
    weeks = [_wk("U1", "2010-01-11")]
    cases = {
        "before": ("2010-01-01", "2010-01-10 23:59:59", 0),
        "after": ("2010-01-18 00:00:00", "2010-01-20", 0),  # starts exactly when the week ends
        "touch_start": ("2010-01-05", "2010-01-11 00:00:00", 1),  # end inclusive
        "partly": ("2010-01-05", "2010-01-12", 1),
        "inside": ("2010-01-13 10:00", "2010-01-13 11:00", 1),
        "covers": ("2010-01-01", "2010-02-01", 1),
    }
    for name, (s, e, want) in cases.items():
        assert _label([("U1", s, e)], weeks) == [want], name


def test_multiple_windows_and_other_users():
    weeks = [_wk("U1", "2010-01-04"), _wk("U1", "2010-01-18"), _wk("U1", "2010-02-01"), _wk("U2", "2010-01-18")]
    windows = [("U1", "2010-01-05", "2010-01-06"), ("U1", "2010-02-02", "2010-02-03")]
    assert _label(windows, weeks) == [1, 0, 1, 0]


def _logs():
    lg = lambda ts, u, pc, a="Logon": {"date": T(ts), "user": u, "pc": pc, "activity": a}
    logon = pd.DataFrame([
        lg("2010-01-04 08:00", "U1", "PC1"), lg("2010-01-05 23:00", "U1", "PC2"),
        lg("2010-01-05 23:30", "U1", "PC2", "Logoff"), lg("2010-01-17 09:00", "U1", "PC1"),  # Sunday, week of Mon 01-11
        lg("2010-01-25 09:00", "U1", "PC1"),
        lg("2010-01-04 09:00", "U2", "PC1"),
    ])
    device = pd.DataFrame({"date": [T("2010-01-04 12:00"), T("2010-01-13 12:00")], "user": ["U1", "U1"],
                           "pc": ["PC1", "PC1"], "activity": ["Connect", "Connect"]})  # 01-13: week with no logon
    file = pd.DataFrame({"date": [T("2010-01-06 12:00")] * 2, "user": ["U1", "U2"], "pc": ["PC1", "PC1"]})
    return logon, device, file


def test_weekly_counts_and_missing_weeks():
    w = make_weekly_features(*_logs(), ["U1", "U2"]).set_index(["user", "week_start"])
    # U1 has logon weeks 01-04, 01-11 (the Sunday 01-17), 01-25; no row for 01-18
    assert list(w.loc["U1"].index) == [T("2010-01-04"), T("2010-01-11"), T("2010-01-25")]
    r = w.loc[("U1", T("2010-01-04"))]
    assert (r["n_logon"], r["n_after_hours"], r["n_pcs"], r["n_device"], r["n_file"]) == (2, 1, 2, 1, 1)
    # device event on 01-13 falls in week 01-11 which has a logon -> counted there
    assert w.loc[("U1", T("2010-01-11")), "n_device"] == 1
    assert w.loc[("U2", T("2010-01-04")), "n_file"] == 1


def test_dropped_events_in_logonless_week():
    logon, device, file = _logs()
    device.loc[1, "date"] = T("2010-01-20 12:00")  # week 01-18: no logon -> dropped
    w = make_weekly_features(logon, device, file, ["U1", "U2"])
    assert w["n_device"].sum() == 1


def test_feature_columns_have_no_tenure_or_calendar():
    w = make_weekly_features(*_logs(), ["U1", "U2"])
    assert WEEK_FEATURE_COLS == ["n_logon", "n_after_hours", "n_pcs", "n_device", "n_file"]
    assert not {"active_days", "week", "week_index", "date"} & set(WEEK_FEATURE_COLS)
    assert set(w.columns) == {"user", "week_start"} | set(WEEK_FEATURE_COLS)


def test_attack_window_date_parsing(tmp_path):
    (tmp_path / "answers").mkdir()
    (tmp_path / "answers" / "insiders.csv").write_text(
        "dataset,scenario,details,user,start,end\n"
        "2,1,x,OLD,3/6/2010 1:41:56,3/20/2010 8:10:12\n"
        "4.2,1,x,A,3/6/2010 1:41:56,3/20/2010 8:10:12\n"
        "4.2,2,x,B,07/01/2010 01:24:58,07/16/2010 06:52:00\n"
    )
    w = load_attack_windows(tmp_path)
    assert list(w["user"]) == ["A", "B"]
    assert w.loc[0, "start"] == T("2010-03-06 01:41:56") and w.loc[1, "start"] == T("2010-07-01 01:24:58")


def test_folds_never_split_a_user():
    users = [f"U{i}" for i in range(40)]
    labels = {f"U{i}" for i in range(8)}
    weekly = pd.DataFrame({"user": [u for u in users for _ in range(3)]})
    for f in make_folds(users, labels, 5, 2, seed=0):
        tr, te = week_masks(weekly, f)
        assert not set(weekly.loc[tr, "user"]) & set(weekly.loc[te, "user"])
        assert (tr | te).all()
