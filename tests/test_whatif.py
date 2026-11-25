import json

import numpy as np
import pandas as pd
import pytest

from itgnn import whatif as W
from itgnn.commslogs import comms_feature_sets
from itgnn.ui_data import assert_aggregates_only
from itgnn.weekly_features import FEATURE_SETS


def _real_like_row(rng):
    """A consistent synthetic week: every deviation equals current minus history mean, totals respect their parts."""
    row = {}
    n_logon = int(rng.integers(5, 40)); n_after = int(rng.integers(0, min(6, n_logon))); em = int(rng.integers(0, 30)); wb = int(rng.integers(50, 500))
    base = {"n_logon": n_logon, "n_after_hours": n_after, "n_pcs": int(rng.integers(1, 4)), "n_device": int(rng.integers(0, 8)), "n_file": int(rng.integers(0, 12)),
            "em_n": em, "em_n_after_hours": int(rng.integers(0, em + 1)), "em_n_recip": em + 5, "em_n_ext_recip": int(rng.integers(0, 6)), "em_max_recip": 3, "em_n_bcc": 1, "em_n_attach": 1,
            "em_size_sum": float(em * 28000), "em_n_new_recip": 2, "em_n_from_external": int(rng.integers(0, em + 1)), "em_n_competitor_recip": 0,
            "wb_n": wb, "wb_n_after_hours": int(rng.integers(0, 20)), "wb_n_domains": 30, "wb_n_new_domains": 1, "wb_share_rare": 0.0,
            "wb_n_leak": 0, "wb_n_job": int(rng.integers(0, 5)), "wb_n_cloud": int(rng.integers(0, 5)), "wb_n_download": 0, "wb_n_competitor": int(rng.integers(0, 5))}
    for c in W.BASE_COLS:
        row[c] = float(base[c]); h = float(rng.uniform(0, 1) * base[c] + 0.5); row[f"hist_mean_{c}"] = h; row[f"dev_{c}"] = row[c] - h
    row["n_new_pcs"] = 1.0
    return row


def test_feature_columns_are_the_79_model_columns():
    assert sorted(W.feature_columns()) == sorted(comms_feature_sets(FEATURE_SETS["W2"])["W2+E+H"]) and len(W.feature_columns()) == 79


def test_round_trip_rebuilds_every_stored_feature_within_1e_6():
    rng = np.random.default_rng(0)
    for _ in range(200):
        row = _real_like_row(rng)
        rebuilt, notes = W.rebuild(row, {c: row[c] for c in W.BASE_COLS if c not in W.FROZEN})
        assert max(abs(rebuilt[c] - row[c]) for c in row) < 1e-6 and not notes


def test_a_knob_changes_only_its_current_value_and_deviation():
    row = _real_like_row(np.random.default_rng(1))
    new, notes = W.rebuild(row, {"n_file": row["n_file"] + 7})
    changed = {c for c in row if abs(new[c] - row[c]) > 1e-9}
    assert changed == {"n_file", "dev_n_file"} and abs(new["dev_n_file"] - (row["dev_n_file"] + 7)) < 1e-9
    assert new["hist_mean_n_file"] == row["hist_mean_n_file"]
    with pytest.raises(ValueError):
        W.rebuild(row, {"n_new_pcs": 3})                  # frozen: cannot be set from a knob
    ov, _ = W.rebuild(row, {}, {"dev_n_logon": 5.0})
    assert ov["dev_n_logon"] == 5.0


def test_consistency_rules_raise_totals_and_say_so():
    row = _real_like_row(np.random.default_rng(2)); row.update({"n_logon": 4.0, "n_pcs": 1.0, "wb_n": 100.0})
    new, notes = W.rebuild(row, {"n_after_hours": 10, "wb_n_leak": 150})
    assert new["n_logon"] == 10 and new["wb_n"] >= 150 and len(notes) >= 2 and all("raised" in n for n in notes)
    assert new["dev_n_logon"] == new["n_logon"] - new["hist_mean_n_logon"]


def test_percentile_verdict_warnings_and_presets():
    grid = list(np.linspace(0, 1, 101))
    meta = {"oof_quantiles": {"score": {"grid": grid, "values": [g ** 2 for g in grid]}}, "budget_cutoffs": {"0.25": 0.99, "0.5": 0.97, "1.0": 0.9, "2.0": 0.8},
            "p01": {"n_logon": 1.0}, "p99": {"n_logon": 30.0}}
    assert W.percentile(meta, "score", 0.0) == 0 and W.percentile(meta, "score", 1.0) == 100 and 49 < W.percentile(meta, "score", 0.25) < 51
    assert W.verdict(meta, 0.995) == 0.25 and W.verdict(meta, 0.92) == 1.0 and W.verdict(meta, 0.5) is None
    assert W.range_warnings(meta, {"n_logon": 80.0}, ["n_logon"]) and not W.range_warnings(meta, {"n_logon": 10.0}, ["n_logon"])
    base = {k: 2.0 for k in W.SLIDERS}
    p = W.presets(base)
    assert list(p) == ["Normal week", "After-hours data copying", "Leak-site visit", "Job-hunting week"] and p["Normal week"] == {}
    assert p["Leak-site visit"]["wb_n_leak"] == 3 and set().union(*p.values()) <= set(W.SLIDERS)


def test_live_predictor_matches_the_model_and_contributions_add_up(ui_results_full):
    import xgboost as xgb
    meta = json.load(open(ui_results_full / "final_model_meta.json"))
    models = W.load_models(ui_results_full)
    base = W.median_baseline(meta)
    feats, _ = W.rebuild(base, {"n_after_hours": 9, "wb_n_leak": 4})
    res = W.predict(models, meta, pd.DataFrame([feats]))
    direct = models["main"].predict(xgb.DMatrix(pd.DataFrame([feats])[meta["features_main"]], feature_names=meta["features_main"]))
    assert abs(float(res["score"].iloc[0]) - float(direct[0])) < 1e-9
    margin = sum(float(res[g].iloc[0]) for g in meta["groups"]) + float(res["bias"].iloc[0])
    assert abs(1 / (1 + np.exp(-margin)) - float(res["score"].iloc[0])) < 1e-5
    sw = W.sweep(models, meta, base, {k: base[k] for k in W.SLIDERS}, "wb_n_leak", n=50)
    assert 20 <= len(sw) <= 50 and sw["value"].is_monotonic_increasing and sw["pct"].between(0, 100).all()
    at_base = sw.loc[(sw["value"] - base["wb_n_leak"]).abs().idxmin()]
    one, _ = W.rebuild(base, {"wb_n_leak": at_base["value"]})
    assert abs(float(W.predict(models, meta, pd.DataFrame([one]))["score"].iloc[0]) - at_base["score"]) < 1e-9


def test_aggregate_files_have_no_per_row_fields(ui_results_full):
    for f in ("ui_rigor_facts.json", "ui_budget_curve.json"):
        assert_aggregates_only(json.load(open(ui_results_full / f)))
    for bad in ({"users": ["A"]}, {"x": {"label": 1}}, {"rows": list(range(100))}, {"id": "ABC1234"}, {"week_start": "2010-01-04"}):
        with pytest.raises(AssertionError):
            assert_aggregates_only(bad)
    from pathlib import Path
    for f in ("ui_rigor_facts.json", "ui_budget_curve.json"):   # the real files, when this clone has them (ui_metrics.json is an older file and is not changed)
        p = Path("results") / f
        if p.exists():
            assert_aggregates_only(json.load(open(p)))
