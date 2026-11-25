import json
import pandas as pd
import pytest

from itgnn import ui_data
from itgnn import ui_hypo as H
from itgnn import whatif as W
from itgnn import neo4j_ui

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
from pathlib import Path

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_check_name_rules():
    assert H.check_name("Test worker 1", ["Other"], ["AAF0535"]) is None
    assert H.check_name("") and H.check_name("   ")
    assert "real user id" in H.check_name("AAF0535")          # looks like a real id
    assert "real user id" in H.check_name("zzz1234")
    assert "already" in H.check_name("test WORKER 1", ["Test worker 1"])
    assert "already" in H.check_name("Zed", [], ["zed"])
    assert H.check_name("<script>")                           # characters outside the whitelist
    assert H.check_name("x" * 31)


def test_week_start_is_consecutive_weeks():
    assert H.week_start_of(72, 72, "2011-05-16") == pd.Timestamp("2011-05-16") and H.week_start_of(70, 72, "2011-05-16") == pd.Timestamp("2011-05-02")


def test_coupling_only_raises_and_explains():
    start = {"wb_n_leak": 0.0, "wb_n_job": 7.0, "wb_n": 100.0, "n_pcs": 3.0, "em_n": 10.0, "em_n_recip": 20.0}
    cur, notes = H.couple(start, {"wikileaks.org": ("leak", 3.0), "indeed.com": ("job", 2.0), "news.com": ("", 5.0)}, ["PC-1"], ["U1", "U2"])
    assert cur["wb_n_leak"] == 3 and cur["wb_n_job"] == 7 and cur["wb_n"] == 100 and cur["n_pcs"] == 3 and cur["em_n_recip"] == 20     # a value the analyst set is never lowered
    assert any("Leak-site visits raised from 0 to 3" in n for n in notes) and not any("Job-site" in n for n in notes)
    cur, notes = H.couple({**start, "wb_n": 4.0, "n_pcs": 1.0, "em_n_recip": 1.0}, {"a.com": ("", 9.0)}, ["PC-1", "PC-2"], ["U1", "U2"])
    assert cur["wb_n"] == 9 and cur["n_pcs"] == 2 and cur["em_n_recip"] == 2 and len(notes) == 3
    assert H.couple(start, {}, [], []) == (start, [])


def test_worker_rating_and_equal_usual_means_zero_deviation(ui_results_full):
    meta, models = W.load_meta(ui_results_full), W.load_models(ui_results_full)
    typ = W.median_baseline(meta)
    real = pd.read_parquet(ui_results_full / "ui_scores.parquet")["score"].to_numpy()
    this = {k: typ[k] for k in W.SLIDERS}
    w = H.make_worker(models, meta, typ, real, "Test 1", 72, "2011-05-16", this, this)
    assert all(abs(w["features"][f"dev_{k}"]) < 1e-9 for k in W.SLIDERS) and w["hypothetical"] is True and 1 <= w["rank"] <= len(real) + 1
    up = H.make_worker(models, meta, typ, real, "Test 2", 72, "2011-05-16", this, {**this, "n_after_hours": 20}, {"wikileaks.org": ("leak", 3.0)})
    assert up["features"]["wb_n_leak"] == 3 and up["features"]["dev_n_after_hours"] == 20 - this["n_after_hours"] and up["features"]["wb_n"] >= 3
    assert any("raised" in n for n in up["notes"])


def test_cap_duplicates_and_csv(ui_results_full):
    meta, models = W.load_meta(ui_results_full), W.load_models(ui_results_full)
    typ = W.median_baseline(meta)
    real = pd.read_parquet(ui_results_full / "ui_scores.parquet")["score"].to_numpy()
    this = {k: typ[k] for k in W.SLIDERS}
    mk = lambda n: H.make_worker(models, meta, typ, real, n, 72, "2011-05-16", this, this, {"wikileaks.org": ("leak", 1.0)}, ["PC-1"], ["U1"])
    ws = H.add_worker([], mk("A"), 2, ["U1"])
    ws = H.add_worker(ws, mk("B"), 2, ["U1"])
    with pytest.raises(ValueError, match="at most 2"):
        H.add_worker(ws, mk("C"), 2)
    with pytest.raises(ValueError, match="already"):
        H.add_worker(ws[:1], mk("a"), 2)
    csv = pd.read_csv(__import__("io").StringIO(H.to_csv(ws, W.feature_columns())))
    assert list(csv["name"]) == ["A", "B"] and csv["hypothetical"].all() and len(W.feature_columns()) == 79 and set(W.feature_columns()) <= set(csv.columns)
    assert "wikileaks.org (leak) x1" in csv["websites"].iloc[0] and "label" not in " ".join(csv.columns)


def test_domain_table_is_label_free_and_tags_categories():
    agg = pd.DataFrame({"user": [f"u{i}" for i in range(25)] * 2 + ["a", "b"] * 2, "domain": ["news.com"] * 25 + ["rare.com"] * 25 + ["wikileaks.org", "wikileaks.org", "blackhat.org", "blackhat.org"]})
    t = H.domain_table(agg, {"leak": ["wikileaks.org"]}, ["blackhat.org"], 20)
    assert list(t.columns) == ["domain", "category", "n_users"] and set(t["domain"]) == {"news.com", "rare.com", "wikileaks.org"}
    assert t.loc[t["domain"] == "wikileaks.org", "category"].item() == "leak" and t["n_users"].min() == 2


def test_domain_fallback_uses_ui_edges_when_file_missing(ui_results_full):
    t, note = H.load_domains(ui_results_full)
    assert note and "ui_domains" in note and "wikileaks.org" in set(t["domain"]) and t.loc[t["domain"] == "wikileaks.org", "category"].item() == "leak"
    t.to_parquet(ui_results_full / "ui_domains.parquet", index=False)
    assert H.load_domains(ui_results_full)[1] is None


def _app(ui_results, monkeypatch, **env):
    monkeypatch.setenv("ITGNN_RESULTS", str(ui_results))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(neo4j_ui, "auto_source", lambda results: (ui_data.ParquetSource(results), "test backend"))
    return AppTest.from_file(APP, default_timeout=90)


def _go(at, page):
    at.segmented_control[0].set_value(page).run()
    assert not at.exception, (page, [e.value for e in at.exception])
    return at


def test_tab_never_loads_labels_adds_removes_and_changes_nothing_else(ui_results_full, monkeypatch):
    def boom(*a, **k): raise AssertionError("labels loaded")
    monkeypatch.setattr(ui_data, "load_eval", boom)
    at = _app(ui_results_full, monkeypatch).run()
    before = {p: (ui_data.load_scores(ui_results_full).copy()) for p in ("x",)}["x"]
    _go(at, "New prediction")
    assert len(at.slider) == 24 and any("near-label markers" in w.value for w in at.warning)                      # caveat is on the tab, prominent
    assert any("without named sites" in w.value.lower() for w in at.warning) and any("approximate" in c.value for c in at.caption)
    assert at.metric[0].label.startswith("All features") and at.metric[1].label.startswith("Without named sites")        # the generic rating sits right next to the main one
    at.text_input(key="np_name").set_value("AAF0535").run()                                                       # looks like a real id
    assert at.button(key="np_add").disabled and any("real user id" in e.value for e in at.error)
    at.text_input(key="np_name").set_value("Test worker 1").run()
    at.multiselect(key="np_sites").set_value(["wikileaks.org"]).run()
    at.number_input(key="np_v_wikileaks.org").set_value(100).run()                                                 # above the fixture's typical value, so the count must be raised
    assert not at.exception and any("Leak-site visits raised" in i.value for i in at.info)                          # the raise is visible
    at.button(key="np_add").click().run()
    assert not at.exception and len(at.session_state["np_workers"]) == 1 and at.session_state["np_workers"][0]["name"] == "Test worker 1"
    at.text_input(key="np_name").set_value("Test worker 1").run()                                                 # duplicate name
    assert any("already used" in e.value for e in at.error)
    at.button(key="np_rm_Test worker 1").click().run()
    assert not at.exception and at.session_state["np_workers"] == []
    pd.testing.assert_frame_equal(before, ui_data.load_scores(ui_results_full))                                    # nothing in the loaded scores changed


def test_cap_clear_all_and_other_pages_unaffected(ui_results_full, monkeypatch):
    import yaml
    cfg = ui_results_full / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"new_prediction": {"enabled": True, "max_workers": 2}}))
    at = _app(ui_results_full, monkeypatch, ITGNN_CONFIG=str(cfg)).run()
    _go(at, "New prediction")
    for n in ("W1", "W2"):
        at.text_input(key="np_name").set_value(n).run(); at.button(key="np_add").click().run()
    assert len(at.session_state["np_workers"]) == 2
    at.text_input(key="np_name").set_value("W3").run()
    assert at.button(key="np_add").disabled
    for page in ("Overview", "Threat map", "Alerts", "Investigate", "What-if", "Rigor and limits"):
        _go(at, page)
    _go(at, "New prediction")
    assert len(at.session_state["np_workers"]) == 2                                                               # the list survives page switches
    at.button(key="np_clear").click().run()
    assert at.session_state["np_workers"] == []


def test_switch_off_gives_six_tabs(ui_results_full, monkeypatch):
    import yaml
    cfg = ui_results_full / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"new_prediction": {"enabled": False}}))
    at = _app(ui_results_full, monkeypatch, ITGNN_CONFIG=str(cfg)).run()
    assert at.segmented_control[0].options == ["Overview", "Threat map", "Alerts", "Investigate", "What-if", "Rigor and limits"]


def _payload_and_worker(ui_results_full, **kw):
    from itgnn.ui_graph import build_payload
    sc, wk = ui_data.load_scores(ui_results_full), pd.read_parquet(ui_results_full / "ui_weeks.parquet")
    payload, _ = build_payload(sc, wk, ui_data.ParquetSource(ui_results_full).all_edges())
    meta, models = W.load_meta(ui_results_full), W.load_models(ui_results_full)
    typ = W.median_baseline(meta)
    this = {k: typ[k] for k in W.SLIDERS}
    cw = int(payload["uw"][0]["cw"]); u0 = payload["uw"][0]["u"]
    args = dict(sites={"wikileaks.org": ("leak", 3.0), "brand-new.example": ("", 2.0)}, pcs=["PC-1", H.NEW_PC], colleagues=[u0, "U9"])
    args.update(kw)
    w = H.make_worker(models, meta, typ, ui_data.load_scores(ui_results_full)["score"].to_numpy(), "Test 1", cw, payload["uw"][0]["d"], this, this, **args)
    return payload, w, meta, u0


def test_augment_is_identity_without_workers_and_never_mutates_the_real_payload(ui_results_full):
    payload, w, meta, _ = _payload_and_worker(ui_results_full)
    before = json.dumps(payload, sort_keys=True)
    assert H.augment_payload(payload, [], meta["budget_cutoffs"]) is payload                                    # switch off / no workers: the same object, so byte-identical
    out = H.augment_payload(payload, [w], meta["budget_cutoffs"])
    assert json.dumps(payload, sort_keys=True) == before                                                          # the cached real payload is untouched
    n = len(payload["uw"])
    assert out["uw"][:n] == payload["uw"] and out["edges"][:len(payload["edges"])] == payload["edges"] and out["conn"][:len(payload["conn"])] == payload["conn"]   # real indices unchanged
    assert out["campaign"] is payload["campaign"]                                                                  # the Campaign view never sees a worker


def test_augment_adds_one_node_and_one_edge_per_connection(ui_results_full):
    payload, w, meta, u0 = _payload_and_worker(ui_results_full)
    out = H.augment_payload(payload, [w], meta["budget_cutoffs"])
    i = len(payload["uw"]); rec = out["uw"][i]
    assert rec["hyp"] == 1 and rec["u"] == "Test 1" and rec["i"] == i and len(out["uw"]) == i + 1
    mine = [e for e in out["edges"][len(payload["edges"]):]]
    assert len(mine) == 2 + 2 + 2 and all(e[0] == i for e in mine)
    names = {(out["conn"][e[1]]["n"] if e[2] != 3 else "user:" + out["uw"][e[1]]["u"]): e for e in mine}
    n_old = len(payload["conn"])
    wl = next(k for k, c in enumerate(payload["conn"]) if c["n"] == "wikileaks.org")
    assert names["wikileaks.org"][1] == wl and names["wikileaks.org"][1] < n_old                                    # existing website reused
    assert names["brand-new.example"][1] >= n_old and "NEW PC of Test 1 (hypothetical)" in names and names["U9"][2] == 2   # new entries appended; an unflagged colleague is a contact
    assert names["user:" + u0][2] == 3 or u0 in names                                                                # a flagged colleague that week is a direct person-to-person edge
    assert "ins" not in rec and not [k for k in rec if k in ("label", "scenario", "window")]                       # no label fields


def test_worker_without_connections_is_isolated_and_missing_week_extends_the_calendar(ui_results_full):
    payload, w, meta, _ = _payload_and_worker(ui_results_full, sites={}, pcs=[], colleagues=[])
    w["cal_week"], w["week_start"] = 99, pd.Timestamp("2012-01-02")
    out = H.augment_payload(payload, [w], meta["budget_cutoffs"])
    assert out["edges"] == payload["edges"] and {"cw": 99, "d": "2012-01-02"} in out["meta"]["calendar"]
    assert [c["cw"] for c in out["meta"]["calendar"]] == sorted(c["cw"] for c in out["meta"]["calendar"])
    assert out["uw"][-1]["nr"] in (0, 1)


def test_map_page_renders_with_workers_and_loads_no_labels(ui_results_full, monkeypatch):
    def boom(*a, **k): raise AssertionError("labels loaded")
    monkeypatch.setattr(ui_data, "load_eval", boom)
    at = _app(ui_results_full, monkeypatch).run()
    _go(at, "New prediction")
    at.text_input(key="np_name").set_value("Map worker").run()
    at.multiselect(key="np_sites").set_value(["wikileaks.org"]).run()
    at.button(key="np_add").click().run()
    assert not at.exception and len(at.session_state["np_workers"]) == 1
    at.button(key="np_map_Map worker").click().run()                                                              # "Show on map"
    assert not at.exception and at.session_state["nav"] == "Threat map" and at.session_state["map_focus"]["user"] == "Map worker"
    for page in ("Alerts", "Investigate", "Overview", "Threat map"):
        _go(at, page)
