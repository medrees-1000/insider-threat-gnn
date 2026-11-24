import json

import numpy as np
import pandas as pd

from itgnn import ui_data as U
from itgnn.ui_graph import MAX_PER_WEEK, T_DOM, T_MAIL_USER, T_PC, build_payload

BANNED_KEYS = ("label", "scenario", "window", "insider", "ins", "truth", "evaluation", "attack")


def _inputs(ui_results):
    return U.load_scores(ui_results), pd.read_parquet(ui_results / "ui_weeks.parquet"), U.ParquetSource(ui_results).all_edges()


def _keys(o, out=None):
    out = set() if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(k); _keys(v, out)
    elif isinstance(o, list):
        for v in o:
            _keys(v, out)
    return out


def test_default_payload_has_no_label_fields(ui_results):
    sc, wk, e = _inputs(ui_results)
    payload, stats = build_payload(sc, wk, e)
    assert not set(BANNED_KEYS) & _keys(payload), set(BANNED_KEYS) & _keys(payload)
    assert "evaluation" not in payload["meta"] and not any("ins" in r for r in payload["uw"]) and not any("ins" in u for u in payload["campaign"]["users"])
    json.dumps(payload)  # JSON-able


def test_evaluation_flags_only_when_insiders_are_passed(ui_results):
    sc, wk, e = _inputs(ui_results)
    payload, _ = build_payload(sc, wk, e, insiders={"U1"})
    assert payload["meta"]["evaluation"] is True
    assert {r["ins"] for r in payload["uw"] if r["u"] == "U1"} == {1} and {r["ins"] for r in payload["uw"] if r["u"] != "U1"} <= {0}
    assert all("ins" in u for u in payload["campaign"]["users"])


def test_payload_counts_levels_and_edges(ui_results):
    sc, wk, e = _inputs(ui_results)
    p, st = build_payload(sc, wk, e)
    assert len(p["uw"]) == len(wk) == st["flagged_user_weeks"] and st["dropped_by_cap"] == 0
    assert len(p["edges"]) == len(e)                       # nothing lost when no cap applies
    assert {x[2] for x in p["edges"]} <= {T_PC, T_DOM, 2, T_MAIL_USER}
    assert all(0 <= r["lvl"] <= 3 and 0 <= r["p"] <= 100 and len(r["c"]) == len(p["meta"]["groups"]) for r in p["uw"])
    assert [c["cw"] for c in p["meta"]["calendar"]] == sorted({r["cw"] for r in p["uw"]})
    cats = {c["cat"] for c in p["conn"] if c["k"] == "domain"}
    assert "leak" in cats and "" in cats                   # named-site category is carried to the map


def test_cap_per_calendar_week_drops_the_lowest_scores(ui_results):
    sc, wk, e = _inputs(ui_results)
    p, st = build_payload(sc, wk, e, max_per_week=1)
    per = pd.Series([r["cw"] for r in p["uw"]]).value_counts()
    assert per.max() == 1 and st["dropped_by_cap"] == len(wk) - len(p["uw"]) > 0
    kept = {(r["u"], r["d"]) for r in p["uw"]}
    full = wk.assign(d=wk["week_start"].dt.strftime("%Y-%m-%d"))
    for cw, g in full.groupby("cal_week"):
        top = g.sort_values("score", ascending=False).iloc[0]
        assert (top["user"], top["d"]) in kept
    assert MAX_PER_WEEK == 60


def test_campaign_is_deterministic_and_positions_are_finite(ui_results):
    sc, wk, e = _inputs(ui_results)
    a, _ = build_payload(sc, wk, e); b, _ = build_payload(sc, wk, e)
    assert a["campaign"] == b["campaign"]
    c = a["campaign"]
    assert c["louvain_seed"] == 42 and isinstance(c["modularity"], float)
    assert np.isfinite([[u["x"], u["y"]] for u in c["users"]]).all() and np.isfinite([[x["x"], x["y"]] for x in c["conn"]]).all()
    assert {u["u"] for u in c["users"]} == set(wk["user"])
