import numpy as np
import pandas as pd
import pytest
import torch

from itgnn.weekly_features import add_history_features
from itgnn.weekly_gnn import WeeklyNet, choose_epoch, class_weight, train_scores
from itgnn.weekly_graph import N_FEAT, build_weekly_graph, fold_masks, node_features
from itgnn.windows import make_weekly_features

MONDAYS = pd.date_range("2010-01-04", periods=8, freq="7D")
USERS = [f"U{i}" for i in range(6)]
GCFG = {"layers": 2, "hidden": 8, "dropout": 0.3, "lr": 0.01, "weight_decay": 0.0, "class_weight": "sqrt", "max_class_weight": 50}


def _empty():
    return pd.DataFrame({"user": pd.Series(dtype=object), "date": pd.Series(dtype="datetime64[ns]")})


def _logon(extra=()):
    rng = np.random.default_rng(0)
    rows = [(u, f"PC{(i + k) % 3}", k, 8 + int(rng.integers(0, 10))) for i, u in enumerate(USERS) for k in range(8) for _ in range(1 + int(rng.integers(0, 3)))]
    rows += list(extra)
    return pd.DataFrame({"user": [r[0] for r in rows], "pc": [r[1] for r in rows], "activity": "Logon",
                         "date": [MONDAYS[r[2]] + pd.Timedelta(hours=r[3], days=1) for r in rows]})


def _graph(logon, labels=None):
    w = make_weekly_features(logon, _empty(), _empty(), set(USERS))
    w["label"] = 0
    if labels:
        for u, k in labels:
            w.loc[(w["user"] == u) & (w["week_start"] == MONDAYS[k]), "label"] = 1
    return build_weekly_graph(add_history_features(w, logon), logon)


LABELS = [("U0", 6), ("U0", 7), ("U1", 7), ("U3", 5)]


def _triples(g, before):
    n = g.n_user
    u, p = g.edge_index[:, : g.edge_index.shape[1] // 2].numpy()
    keep = g.rows["week_start"].to_numpy()[u] < before
    return sorted(zip(g.rows["user"].to_numpy()[u][keep], g.rows["week_start"].to_numpy()[u][keep], g.pc_name[p[keep] - n]))


def test_no_edge_between_weeks_and_bipartite():
    g = _graph(_logon())
    s, d = g.edge_index.numpy()
    assert (g.node_week[s] == g.node_week[d]).all()
    assert ((s < g.n_user) != (d < g.n_user)).all()  # user <-> PC only
    assert g.edge_index.shape[1] % 2 == 0


def test_later_week_change_leaves_earlier_weeks_unchanged():
    base = _graph(_logon())
    new = _graph(_logon([("U2", "PC9", 7, 9), ("U0", "PC9", 7, 10), ("U5", "PC0", 6, 22)]))
    cut = MONDAYS[6]
    assert _triples(base, cut) == _triples(new, cut)
    a = base.rows["week_start"] < cut
    b = new.rows["week_start"] < cut
    np.testing.assert_array_equal(base.user_raw[a.to_numpy()], new.user_raw[b.to_numpy()])
    pcw = lambda g: {(w, p): v for w, p, v in zip(g.node_week[g.n_user :], g.pc_name, g.pc_logn) if w < cut}
    assert pcw(base) == pcw(new)
    # positive control: the edit does change the weeks it touches
    assert _triples(base, MONDAYS[7] + pd.Timedelta(days=1)) != _triples(new, MONDAYS[7] + pd.Timedelta(days=1))


def test_labels_are_not_inputs():
    g = _graph(_logon(), LABELS)
    fit = g.rows["has_history"].to_numpy()
    g0 = _graph(_logon())  # no labels at all
    np.testing.assert_array_equal(node_features(g, fit)[0].numpy(), node_features(g0, fit)[0].numpy())
    tr, te = fold_masks(g, USERS[:4], USERS[4:])
    x, _ = node_features(g, tr)
    y = g.rows["label"].to_numpy().copy()
    y2 = y.copy()
    y2[te] = 1 - y2[te]  # flip the TEST labels
    y2[~g.rows["has_history"].to_numpy()] = 1  # and the warm-up labels
    s1, _ = train_scores(x, g.edge_index, y, np.flatnonzero(tr), 5, "sage", GCFG, 0)
    s2, _ = train_scores(x, g.edge_index, y2, np.flatnonzero(tr), 5, "sage", GCFG, 0)
    np.testing.assert_array_equal(s1, s2)


def test_scaler_fit_on_training_rows_only():
    g = _graph(_logon(), LABELS)
    tr, te = fold_masks(g, USERS[:4], USERS[4:])
    _, sc = node_features(g, tr)
    np.testing.assert_allclose(sc.mean_, g.user_raw[tr].mean(0))
    g.user_raw[te] += 100.0  # changing test rows must not move the scaler
    _, sc2 = node_features(g, tr)
    np.testing.assert_array_equal(sc.mean_, sc2.mean_)
    np.testing.assert_array_equal(sc.scale_, sc2.scale_)


def test_warmup_rows_excluded_from_loss_and_eval():
    g = _graph(_logon(), LABELS)
    h = g.rows["has_history"].to_numpy()
    assert h.sum() == 6 * 4 and (~h).sum() == 6 * 4  # 8 active weeks each -> 4 warm-up
    tr, te = fold_masks(g, USERS[:4], USERS[4:])
    assert not (tr & ~h).any() and not (te & ~h).any() and not (tr & te).any()
    assert set(g.rows.loc[tr, "user"]) == set(USERS[:4]) and set(g.rows.loc[te, "user"]) == set(USERS[4:])
    assert tr.sum() == 16 and te.sum() == 8
    x, _ = node_features(g, tr)
    assert x.shape == (g.n_node, N_FEAT) and (x[: g.n_user, -3] == torch.tensor(h, dtype=torch.float32)).all()


def test_class_weight_and_epoch_choice_and_mlp_ignores_edges():
    assert class_weight(np.r_[np.ones(1), np.zeros(99)], 50) == pytest.approx(np.sqrt(99))
    assert class_weight(np.r_[np.ones(1), np.zeros(9999)], 50) == 50
    curves = pd.DataFrame({"epoch": np.tile(np.arange(1, 21), 2), "val_ap": np.tile(np.r_[np.linspace(0, 0.5, 10), np.full(10, 0.5)], 2)})
    assert choose_epoch(curves, 1, 0.01) == 10  # first epoch within 0.01 of the plateau
    assert choose_epoch(curves, 5, 0.0) == 14  # trailing mean over 5 epochs reaches 0.5 at epoch 14
    x = torch.randn(7, 4)
    m = WeeklyNet(4, "mlp", 2, 8, 0.0).eval()
    ei = torch.tensor([[0, 1], [1, 0]])
    assert torch.equal(m(x, ei), m(x, ei[:, :0]))
    with pytest.raises(ValueError):
        WeeklyNet(4, "gat")
