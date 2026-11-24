import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.nn import SAGEConv

from itgnn.comms_graph import (DOM, EMAIL, PC, address_coverage, adjacency, build_graph_v2, domains_to_drop, edge_table,
                               explode_recipients, node_features_v2, shuffle_edges)
from itgnn.weekly_gnn import WSageConv

W = pd.date_range("2010-01-04", periods=8, freq="7D")
USERS = ["u1", "u2", "u3"]
COLS = ["a", "dev_a"]
ADDR = {f"{u}@dtaa.com": u for u in USERS}


def _full():
    return pd.DataFrame([(u, w, 0, 1.0 + i, 0.5) for i, u in enumerate(USERS) for w in W], columns=["user", "week_start", "label", "a", "dev_a"])


def _logon(extra=()):
    rows = [(u, w + pd.Timedelta(hours=9), "P1" if u != "u3" else "P2", "Logon") for u in USERS for w in W] + list(extra)
    return pd.DataFrame(rows, columns=["user", "date", "pc", "activity"])


def _agg(extra=()):
    rows = [(u, w, d, n, 0) for u in USERS for w in W for d, n in [("d1.com", 5), ("d2.com", 3), ("d3.com", 2), ("bad.com", 9)]] + list(extra)
    return pd.DataFrame(rows, columns=["user", "week_start", "domain", "n", "n_ah"])


def _email(extra=()):
    rows = [(u, w + pd.Timedelta(days=1, hours=10), f"{v}@dtaa.com;x@gmail.com", "", "") for u in USERS for v in USERS if u != v for w in W] + list(extra)
    return pd.DataFrame(rows, columns=["user", "date", "to", "cc", "bcc"])


def _graph(logon=None, agg=None, email=None, k_dom=2, k_em=1, drop=("bad.com",)):
    return build_graph_v2(_full(), logon if logon is not None else _logon(), agg if agg is not None else _agg(),
                          explode_recipients(email if email is not None else _email()), ADDR, COLS, k_dom, k_em, frozenset(drop))


def test_wsage_equals_sage_with_unit_weights():
    torch.manual_seed(0)
    n, d = 12, 5
    ei = torch.tensor([[0, 1, 2, 3, 4, 5, 5, 6, 7], [1, 2, 0, 0, 0, 6, 7, 5, 5]])
    ei = torch.cat([ei, ei.flip(0)], 1).unique(dim=1)
    x = torch.randn(n, d)
    sage = SAGEConv(d, 4)
    A = torch.zeros(n, n)
    A[ei[1], ei[0]] = 1.0  # row i = neighbours j that send to i
    A = A / A.sum(1, keepdim=True).clamp(min=1)
    w = WSageConv(d, 4)
    w.lin_l.load_state_dict(sage.lin_l.state_dict()); w.lin_r.load_state_dict(sage.lin_r.state_dict())
    out = w(x.requires_grad_(), A.to_sparse_csr())
    assert torch.allclose(out, sage(x, ei), atol=1e-5)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()  # autograd flows through the CSR product


def test_graph_structure_exclusions_and_same_week_only():
    g = _graph()
    assert (g.etype == PC).sum() == 24 and (g.etype == DOM).sum() == 3 * 8 * 2 and (g.etype == EMAIL).sum() == 3 * 8 * 1
    assert "bad.com" not in set(g.dom_name)
    assert (g.node_week[g.eu] == g.node_week[g.ev]).all(), "an edge joins two different weeks"
    t = edge_table(g)
    assert set(t.loc[t["type"] == DOM, "other"]) == {"d1.com", "d2.com"}  # top-2 by visits
    assert (t.loc[t["type"] == EMAIL, "other"].isin(USERS)).all()  # only mapped recipients (x@gmail.com has no user)
    X, _ = node_features_v2(g, np.ones(g.n_user, bool))
    assert X.shape == (g.n_node, len(COLS) + 7) and torch.isfinite(X).all()
    A = adjacency(g)
    assert np.allclose(np.asarray(A.to_dense().sum(1))[np.asarray(A.to_dense().sum(1)) > 0], 1.0, atol=1e-5)


def test_edges_are_causal_and_positive_control():
    base = edge_table(_graph())
    later = _graph(logon=_logon([("u1", W[6] + pd.Timedelta(hours=10), "PX", "Logon")]),
                   agg=_agg([("u1", W[6], "d9.com", 99, 0), ("u2", W[7], "d9.com", 50, 0)]),
                   email=_email([("u2", W[6] + pd.Timedelta(days=2), "u3@dtaa.com", "", "")]))
    after = edge_table(later)
    pd.testing.assert_frame_equal(after[after["week_start"] <= W[5]].reset_index(drop=True), base[base["week_start"] <= W[5]].reset_index(drop=True))
    assert len(after) > len(base)  # positive control: the extra week-6/7 events do add edges there
    assert not after[after["week_start"] == W[6]].equals(base[base["week_start"] == W[6]])


def test_shuffle_preserves_degrees_weights_and_weeks():
    g = _graph(k_dom=3, k_em=2)
    s, drop = shuffle_edges(g, seed=1)
    assert (s.node_week[s.eu] == s.node_week[s.ev]).all()
    assert s.rows.equals(g.rows) and np.array_equal(s.user_raw, g.user_raw)  # nodes and labels untouched
    for t in (PC, DOM):
        a, b = g.etype == t, s.etype == t
        assert np.array_equal(np.bincount(g.eu[a], minlength=g.n_node), np.bincount(s.eu[b], minlength=g.n_node))
        assert np.array_equal(np.bincount(g.ev[a], minlength=g.n_node), np.bincount(s.ev[b], minlength=g.n_node))
        assert sorted(g.ew[a].round(9)) == sorted(s.ew[b].round(9))
    assert 0.0 <= drop <= 1.0


def test_address_coverage_and_domain_drop():
    long = explode_recipients(_email([("u1", W[0], "ghost@dtaa.com", "", "")]))
    cov = address_coverage(long, ADDR, [])
    assert cov["internal addresses"] == 4 and abs(cov["addresses mapped to exactly one user"] - 0.75) < 1e-9
    assert address_coverage(long, ADDR, ["u2@dtaa.com"])["ambiguous addresses (>1 user)"] == 1
    agg = _agg([("u1", W[0], "rare.com", 1, 0)])
    drop, rep = domains_to_drop(agg, {"c": ["d3.com"]}, ["d2.com"], min_users=2)
    assert drop == {"d3.com", "d2.com", "rare.com"} and rep["domains dropped (union)"] == 3
