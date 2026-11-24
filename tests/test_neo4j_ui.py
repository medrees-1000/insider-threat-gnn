import numpy as np
import pandas as pd
import pytest

from itgnn import neo4j_ui as N
from itgnn import ui_data as U


class FakeTx:
    def __init__(self, store): self.store = store
    def run(self, q, **kw): return FakeResult(self.store.get("rows", []), q)


class FakeResult:
    def __init__(self, rows, q): self.rows, self.q = rows, q
    def __iter__(self): return iter([FakeRec(r) for r in self.rows])
    def single(self): return {"n": 1}


class FakeRec:
    def __init__(self, d): self.d = d
    def data(self): return self.d


class FakeSession:
    def __init__(self, store): self.store = store
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, q, **kw): self.store["calls"].append((q, kw)); return FakeResult(self.store.get("rows", []), q)
    def execute_read(self, fn): return fn(FakeTx(self.store))


class FakeDriver:
    def __init__(self, rows=None): self.store = {"calls": [], "rows": rows or []}
    def session(self): return FakeSession(self.store)


def _inputs(ui_results):
    w = pd.read_parquet(ui_results / "ui_weeks.parquet")
    return w, pd.read_parquet(ui_results / "ui_edges.parquet")


def test_row_builders_and_batches(ui_results):
    w, e = _inputs(ui_results)
    wr = N.week_rows(w)
    assert len(wr) == len(w) and set(wr[0]["props"]) == set(N.WEEK_PROPS) | {"week_start"} and wr[0]["key"].startswith(wr[0]["user"] + "|")
    er = N.edge_rows(e, "VISITED")
    assert len(er) == (e["edge_type"] == "VISITED").sum() and {r["category"] for r in er} == {"", "leak"}
    assert [len(b) for b in N.batches(list(range(12)), 5)] == [5, 5, 2]


def test_loader_sends_everything_and_never_a_label(ui_results):
    w, e = _inputs(ui_results)
    d = FakeDriver()
    sent = N.load_ui_graph(d, w, e, batch=4)
    assert sent == {"weeks": len(w), "LOGGED_ON": int((e["edge_type"] == "LOGGED_ON").sum()), "VISITED": int((e["edge_type"] == "VISITED").sum()), "EMAILED": int((e["edge_type"] == "EMAILED").sum())}
    calls = d.store["calls"]
    assert sum("CREATE CONSTRAINT" in q for q, _ in calls) == 4
    text = " ".join(q for q, _ in calls).lower()
    assert not any(w_ in text for w_ in ("label", "malicious", "scenario", "window", "insider"))
    for _, kw in calls:
        for r in kw.get("rows", []):
            assert not any(k in str(r).lower() for k in ("malicious", "scenario", "insider"))
    note_calls = [kw for q, kw in calls if "VISITED" in q]
    assert note_calls and all(kw["note"] == U.CATEGORY_NOTE for kw in note_calls)


def test_neo4j_source_returns_the_same_frame_as_parquet(ui_results):
    w, e = _inputs(ui_results)
    p = U.ParquetSource(ui_results)
    r = w.iloc[0]
    expect = p.neighbourhood(r["user"], r["week_start"])
    rows = [dict(edge_type=x.edge_type, other_id=x.other_id, other_kind=x.other_kind, weight=x.weight, category=(x.category or None),
                 other_users=(None if pd.isna(x.other_users) else x.other_users)) for x in expect.itertuples()]
    got = N.Neo4jSource(FakeDriver(rows)).neighbourhood(r["user"], r["week_start"])
    pd.testing.assert_frame_equal(got, expect, check_dtype=False)
    assert N.Neo4jSource(FakeDriver()).has_week("U1", r["week_start"])


def test_env_and_auto_source_fallback(tmp_path, monkeypatch, ui_results):
    for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        N.get_driver(str(tmp_path / "missing.env"))  # no default password anywhere in code
    env = tmp_path / ".env"; env.write_text("NEO4J_URI=bolt://x:1\nNEO4J_USER=u\nNEO4J_PASSWORD=pw\n")
    monkeypatch.setenv("NEO4J_USER", "already")
    N.load_env(str(env))
    import os
    assert os.environ["NEO4J_URI"] == "bolt://x:1" and os.environ["NEO4J_USER"] == "already"  # never overrides the environment
    monkeypatch.setattr(N, "get_driver", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    src, note = N.auto_source(str(ui_results))
    assert src.name == "parquet" and "fallback" in note


def test_queries_file_and_browser_links():
    q = N.load_queries()
    assert set(q) == {"neighbourhood", "visitors", "shared_pc", "campaign"} and all(v["cypher"].startswith("MATCH") and v["what"] for v in q.values())
    link = N.browser_link(q["neighbourhood"]["cypher"], key="U1|2010-07-12")
    assert link.startswith("http://localhost:7474/browser/?cmd=edit&arg=") and "%27U1%7C2010-07-12%27" in link and "%24key" not in link
    assert "wikileaks.org" in N.browser_link(q["visitors"]["cypher"], domain="wikileaks.org").replace("%27", "'")


def test_all_edges_is_identical_on_both_backends(ui_results):
    p = U.ParquetSource(ui_results).all_edges()
    rows = [dict(user=r.user, week_start=r.week_start.strftime("%Y-%m-%d"), edge_type=r.edge_type, other_id=r.other_id, other_kind=r.other_kind, weight=r.weight,
                 category=(r.category or None), other_users=(None if pd.isna(r.other_users) else r.other_users)) for r in p.itertuples()]
    got = N.Neo4jSource(FakeDriver(rows)).all_edges()
    pd.testing.assert_frame_equal(got[p.columns], p, check_dtype=False)
