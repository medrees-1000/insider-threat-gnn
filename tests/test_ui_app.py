from pathlib import Path

import pytest

import json

import numpy as np

from itgnn import neo4j_ui, ui_data
from itgnn import whatif as W

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
PAGES = ["Overview", "Threat map", "Alerts", "Investigate", "What-if", "Rigor and limits"]


def _app(ui_results, monkeypatch):
    ui_results = ui_results   # may be ui_results_full
    monkeypatch.setenv("ITGNN_RESULTS", str(ui_results))
    monkeypatch.setattr(neo4j_ui, "auto_source", lambda results: (ui_data.ParquetSource(results), "test backend"))  # never touch a real Neo4j
    return AppTest.from_file(APP, default_timeout=90)


def _go(at, page):
    at.segmented_control[0].set_value(page).run()
    assert not at.exception, (page, [e.value for e in at.exception])
    return at


def test_every_page_renders_and_nav_order(ui_results_full, monkeypatch):
    at = _app(ui_results_full, monkeypatch).run()
    assert not at.exception and at.segmented_control[0].options == PAGES
    for page in PAGES:
        _go(at, page)
    assert at.session_state["nav"] == "Rigor and limits"


def test_default_path_never_loads_the_label_file(ui_results_full, monkeypatch):
    def boom(*a, **k): raise AssertionError("ground-truth labels were loaded on the default path")
    monkeypatch.setattr(ui_data, "load_eval", boom)
    at = _app(ui_results_full, monkeypatch).run()
    for page in PAGES:                                   # includes the Threat map: its payload must not touch the label file either
        _go(at, page)
    for page in ("Alerts", "Threat map"):                 # the evaluation view is the ONLY thing that may load them
        _go(at, page)
        at.sidebar.checkbox[0].check().run()
        assert at.exception and "ground-truth labels were loaded" in str(at.exception[0].value), page
        at.sidebar.checkbox[0].uncheck()


def test_controls_and_evaluation_view(ui_results_full, monkeypatch):
    at = _app(ui_results_full, monkeypatch).run()
    _go(at, "Alerts")
    assert len(at.dataframe[0].value) == 1               # 0.5% of 24 weeks -> 1 alert
    assert not any("truth" in c for c in at.dataframe[0].value.columns)
    at.sidebar.radio[0].set_value(2.0).run()
    assert not at.exception and "Top 2.0%" in at.subheader[0].value
    at.sidebar.selectbox[0].select(22).run()
    assert not at.exception and "calendar week 22" in at.subheader[0].value
    at.sidebar.selectbox[0].select("All weeks (global ranking)").run()
    at.sidebar.checkbox[0].check().run()
    assert not at.exception and {"Attack week (truth)", "Scenario (truth)"} <= set(at.dataframe[0].value.columns)
    assert any("EVALUATION VIEW" in e.value for e in at.error)
    _go(at, "Threat map")                                 # the map payload carries insider flags only now


def test_navigation_callback_switches_page_and_selection(ui_results_full, monkeypatch):
    at = _app(ui_results_full, monkeypatch).run()
    _go(at, "Alerts")
    at.button[0].click().run()                            # "Open in Investigate"
    assert not at.exception and at.session_state["nav"] == "Investigate" and at.session_state["sel_user"] is not None
    assert any("scored as it did" in s.value for s in at.subheader)


def test_app_stops_with_instructions_when_results_are_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("ITGNN_RESULTS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and any("ui_export" in c.value for c in at.code) and any("Missing results" in e.value for e in at.error)


def test_overview_numbers_come_from_the_files_and_whatif_responds(ui_results_full, monkeypatch):
    at = _app(ui_results_full, monkeypatch).run()
    text = " ".join(m.value for m in at.markdown)
    assert "51 of 70" in text and "0.50 vs 0.50" in text and "85% to 97%" in text            # headline cards: values from ui_metrics / ui_budget_curve
    assert any("How to read this" in c.value for c in at.caption)                            # the chart caption is there
    assert at.select_slider[0].value == 0.5
    at.select_slider[0].set_value(2.0).run()
    assert not at.exception and any("65.0 of 70" in m.value for m in at.metric)
    _go(at, "What-if")
    assert len(at.slider) == 12 and any("Riskier than" in m.value for m in at.markdown)
    before = at.metric[0].delta
    at.slider[1].set_value(20).run()                                                           # n_after_hours: the rating must respond live
    assert not at.exception
    assert at.metric[0].delta != before, "the rating did not respond to the slider"
    at.button[0].click().run()                                                                  # a preset button
    assert not at.exception
    assert sum("How to read this" in c.value for c in at.caption) >= 2                        # contribution chart + sweep chart


def test_label_file_is_never_read_by_the_overview_or_whatif(ui_results_full, monkeypatch):
    def boom(*a, **k): raise AssertionError("labels loaded")
    monkeypatch.setattr(ui_data, "load_eval", boom)
    at = _app(ui_results_full, monkeypatch).run()
    for page in ("Overview", "What-if"):
        _go(at, page)


def test_alerts_display_options_and_investigate_chart(ui_results_full, monkeypatch):
    at = _app(ui_results_full, monkeypatch).run()
    _go(at, "Alerts")
    assert at.radio(key="al_display").value == "Percentile"                     # percentile is the default display
    pct = at.dataframe[0].value["Risk"].iloc[0]
    assert 0 <= pct <= 100
    for disp in ("Log-odds", "Raw score"):
        at.radio(key="al_display").set_value(disp).run()
        assert not at.exception, disp
    assert any("How to read this" in c.value for c in at.caption)
    _go(at, "Investigate")
    assert any("How to read this" in c.value for c in at.caption) and sum("How to read this" in c.value for c in at.caption) >= 2


def test_whatif_pins_the_rating_caps_sliders_and_reads_out_the_flip(ui_results_full, monkeypatch):
    at = _app(ui_results_full, monkeypatch).run()
    _go(at, "What-if")
    assert any("position: sticky" in m.value for m in at.markdown)               # the pinned-rating rule is injected (its behaviour was checked in a browser)
    meta = json.load(open(ui_results_full / "final_model_meta.json"))
    base = W.median_baseline(meta)
    for s in at.slider:                                                          # every slider is capped near the 99th percentile (or the starting value), never at the 3x sweep range
        key = [k for k in W.SLIDERS if W.LABELS[k] == s.label][0]
        assert s.max == W.slider_max(meta, key, base) and s.max < meta["knob_max"][key] + 1
    sweep = at.selectbox(key="wi_sweep")
    assert "biggest effect" in sweep.options[0]                                  # the sweep starts at the knob with the largest effect
    text = " ".join(m.value for m in at.markdown)
    assert "What would it take?" in text
    assert sum("How to read this" in c.value for c in at.caption) >= 3          # contribution chart, sweep chart, readout
