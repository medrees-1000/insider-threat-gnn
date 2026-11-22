import pandas as pd

from itgnn.data import make_split, subsample_users
from itgnn.graph import FEATURE_COLS, logon_edges, user_features


def _fake_logs():
    logon = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2010-01-04 08:00", "2010-01-04 23:00", "2010-01-05 09:00", "2010-01-05 10:00"]
            ),
            "user": ["U1", "U1", "U2", "U2"],
            "pc": ["PC-1", "PC-2", "PC-1", "PC-1"],
            "activity": ["Logon", "Logon", "Logon", "Logoff"],
        }
    )
    device = pd.DataFrame(
        {"date": pd.to_datetime(["2010-01-04 12:00"]), "user": ["U1"], "pc": ["PC-1"], "activity": ["Connect"]}
    )
    file = pd.DataFrame(
        {"date": pd.to_datetime(["2010-01-04 12:30"] * 2), "user": ["U1", "U2"], "pc": ["PC-1", "PC-1"]}
    )
    return logon, device, file


def test_user_features_shape_and_values():
    logon, device, file = _fake_logs()
    feats = user_features(logon, device, file, ["U1", "U2"])
    assert list(feats.columns) == FEATURE_COLS
    assert feats.loc["U1", "n_logon"] == 2
    assert feats.loc["U1", "n_after_hours"] == 1
    assert feats.loc["U1", "n_pcs"] == 2
    assert feats.loc["U2", "n_logon"] == 1


def test_logon_edges_counts():
    logon, _, _ = _fake_logs()
    edges = logon_edges(logon, ["U1", "U2"])
    assert len(edges) == 3


def test_split_keeps_users_disjoint_and_complete():
    users = [f"U{i}" for i in range(100)]
    labels = {f"U{i}" for i in range(10)}
    sp = make_split(users, labels, val_size=0.15, test_size=0.30, seed=0)
    all_users = sp["train"] + sp["val"] + sp["test"]
    assert sorted(all_users) == sorted(users)
    assert any(u in labels for u in sp["test"])


def test_subsample_keeps_malicious_share():
    users = [f"U{i}" for i in range(200)]
    labels = {f"U{i}" for i in range(20)}
    picked = subsample_users(users, labels, 100, seed=0)
    assert len(picked) == 100
    assert sum(u in labels for u in picked) == 10
