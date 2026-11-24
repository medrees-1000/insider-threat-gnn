from pathlib import Path

import pytest

from itgnn import neo4j_ui, ui_data

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _app(ui_results, monkeypatch):
    monkeypatch.setenv("ITGNN_RESULTS", str(ui_results))
    monkeypatch.setattr(neo4j_ui, "auto_source", lambda results: (ui_data.ParquetSource(results), "test backend"))  # never touch a real Neo4j
    return AppTest.from_file(APP, default_timeout=60)


def test_default_path_never_loads_the_label_file(ui_results, monkeypatch):
    def boom(*a, **k): raise AssertionError("ground-truth labels were loaded on the default path")
    monkeypatch.setattr(ui_data, "load_eval", boom)
    at = _app(ui_results, monkeypatch).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Alerts", "User detail", "Neighbourhood", "Metrics and limitations"]
    assert not any("truth" in c for c in at.dataframe[0].value.columns)
    at.sidebar.checkbox[0].check().run()   # the evaluation view is the ONLY thing that may load them
    assert at.exception and "ground-truth labels were loaded" in str(at.exception[0].value)


def test_controls_and_evaluation_view(ui_results, monkeypatch):
    at = _app(ui_results, monkeypatch).run()
    assert not at.exception and len(at.dataframe[0].value) == 1  # 0.5% of 24 weeks -> 1 alert
    at.sidebar.radio[0].set_value(2.0).run()
    assert not at.exception and "Top 2.0%" in at.subheader[0].value and len(at.dataframe[0].value) == 1  # ceil(2% of 24) = 1
    at.sidebar.selectbox[0].select(22).run()
    assert not at.exception and "calendar week 22" in at.subheader[0].value
    at.sidebar.selectbox[0].select("All weeks (global ranking)").run()
    at.sidebar.checkbox[0].check().run()
    assert not at.exception and {"attack week (truth)", "scenario (truth)"} <= set(at.dataframe[0].value.columns)
    assert any("EVALUATION VIEW" in e.value for e in at.error)


def test_app_stops_with_instructions_when_results_are_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("ITGNN_RESULTS", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and any("ui_export" in c.value for c in at.code) and any("Missing results" in e.value for e in at.error)
