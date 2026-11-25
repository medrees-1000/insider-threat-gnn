"""What-if scoring (inference only): rebuild the model's 79 features from a baseline week plus a few knobs, then score with the
saved final models. Nothing here trains a model; xgboost is imported only to load boosters and predict.

Rules (fixed in the plan): a knob changes the CURRENT-WEEK value of a count; every deviation feature is recomputed as
current minus the baseline's history mean; history means stay at the baseline; the features listed in FROZEN cannot be
recomputed from the knobs and stay at the baseline (with their history/deviation columns)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from itgnn.commslogs import EMAIL_GENERIC, EMAIL_NAMED, WEB_COLS, WEB_NAMED
from itgnn.windows import WEEK_FEATURE_COLS

BASE_COLS = WEEK_FEATURE_COLS + EMAIL_GENERIC + EMAIL_NAMED + WEB_COLS + WEB_NAMED     # counts that have hist_mean_ and dev_ columns
FROZEN = ["n_new_pcs", "em_n_new_recip", "wb_n_new_domains", "wb_share_rare", "wb_n_domains", "em_size_sum", "em_n_bcc", "em_n_attach"]
LABELS = {  # plain-English names (current-week counts)
    "n_logon": "Logons this week", "n_after_hours": "Logons outside 07:00-18:00", "n_pcs": "Different computers used", "n_device": "Removable-drive connections",
    "n_file": "Files copied to removable media", "em_n": "Emails sent", "em_n_after_hours": "Emails sent outside 07:00-18:00", "em_n_recip": "Recipients (all emails)",
    "em_n_ext_recip": "Recipients outside the company", "em_max_recip": "Most recipients in one email", "em_n_from_external": "Emails sent from a personal address",
    "em_n_competitor_recip": "Emails to competitor companies", "wb_n": "Web pages visited", "wb_n_after_hours": "Web pages visited outside 07:00-18:00",
    "wb_n_leak": "Leak-site visits", "wb_n_job": "Job-site visits", "wb_n_cloud": "Cloud-storage visits", "wb_n_download": "Keylogger-site visits", "wb_n_competitor": "Competitor-site visits",
}
SLIDERS = ["n_logon", "n_after_hours", "n_pcs", "n_device", "n_file", "em_n", "em_n_ext_recip", "em_max_recip", "wb_n", "wb_n_job", "wb_n_cloud", "wb_n_leak"]   # the dozen main knobs
ADVANCED_KNOBS = [c for c in LABELS if c not in SLIDERS]
CATEGORY_COUNTS = ["wb_n_leak", "wb_n_job", "wb_n_cloud", "wb_n_download", "wb_n_competitor"]


def feature_columns():
    """The 79 W2+E+H columns in a fixed order (current, history mean, deviation for every base count; n_new_pcs once)."""
    cols = []
    for c in BASE_COLS:
        cols += [c, f"hist_mean_{c}", f"dev_{c}"]
    return cols + ["n_new_pcs"]


def enforce(v):
    """Make the knob values consistent (a total can never be smaller than its parts); returns plain-English notes. Totals are RAISED."""
    notes = []

    def up(k, to, why):
        if v[k] < to:
            notes.append(f"{LABELS[k]} raised from {v[k]:g} to {to:g}: {why}.")
            v[k] = to
    up("n_logon", v["n_after_hours"], "after-hours logons are part of all logons")
    up("n_logon", v["n_pcs"], "each computer needs at least one logon")
    up("em_n", v["em_n_after_hours"], "after-hours emails are part of all emails")
    up("em_n", v["em_n_from_external"], "emails from a personal address are part of all emails")
    up("em_n_recip", v["em_n"], "every email has at least one recipient")
    up("em_n_recip", v["em_n_ext_recip"], "outside recipients are part of all recipients")
    up("em_n_recip", v["em_max_recip"], "the biggest email cannot have more recipients than all emails together")
    up("wb_n", v["wb_n_after_hours"], "after-hours visits are part of all visits")
    up("wb_n", sum(v[c] for c in CATEGORY_COUNTS), "named-site visits are part of all visits")
    return notes


def rebuild(baseline, current=None, overrides=None):
    """All 79 features from a baseline (dict with the 79 columns) and new current-week counts. Returns (features dict, notes).

    current: {base count: new current-week value} for non-frozen counts. Deviation = current - baseline history mean (recomputed
    for every count). overrides: exact values for any feature, applied last (advanced entry)."""
    cur = {c: float(baseline[c]) for c in BASE_COLS}
    for k, val in (current or {}).items():
        if k in FROZEN or k not in cur:
            raise ValueError(f"{k} cannot be set from a knob")
        cur[k] = float(val)
    notes = enforce(cur)
    out = {}
    for c in BASE_COLS:
        out[c] = cur[c]
        out[f"hist_mean_{c}"] = float(baseline[f"hist_mean_{c}"])
        out[f"dev_{c}"] = cur[c] - out[f"hist_mean_{c}"]
    out["n_new_pcs"] = float(baseline["n_new_pcs"])
    for k, val in (overrides or {}).items():
        if k not in out:
            raise ValueError(f"unknown feature {k}")
        out[k] = float(val)
    return out, notes


def median_baseline(meta):
    """The label-free 'typical week': median of every current-week count, median history means, deviations recomputed, then made consistent."""
    med = meta["medians"]
    base = {c: float(med[c]) for c in feature_columns()}
    base = {c: float(np.round(v)) if c in LABELS and c not in FROZEN else v for c, v in base.items()}
    feats, _ = rebuild(base)
    return feats


def load_meta(results="results"):
    p = Path(results) / "final_model_meta.json"
    return json.load(open(p)) if p.exists() else None


def load_models(results="results"):
    import xgboost as xgb  # inference only: loading boosters and predicting

    out = {}
    for name, f in (("main", "final_xgb_main.json"), ("generic", "final_xgb_generic.json")):
        b = xgb.Booster()
        b.load_model(str(Path(results) / f))
        out[name] = b
    return out


def predict(models, meta, rows):
    """rows: DataFrame with the 79 feature columns. Returns DataFrame: score, score_generic, and one contribution column per explanation group."""
    import xgboost as xgb

    fm, fg = meta["features_main"], meta["features_generic"]
    dm = xgb.DMatrix(rows[fm].astype(float), feature_names=fm)
    out = pd.DataFrame({"score": models["main"].predict(dm)}, index=rows.index)
    out["score_generic"] = models["generic"].predict(xgb.DMatrix(rows[fg].astype(float), feature_names=fg))
    contrib = models["main"].predict(dm, pred_contribs=True)
    for g, cols in meta["groups"].items():
        out[g] = contrib[:, [fm.index(c) for c in cols]].sum(axis=1)
    out["bias"] = contrib[:, -1]
    return out


def percentile(meta, key, score):
    """Risk percentile of a score against the out-of-fold score distribution (key: 'score' or 'score_generic')."""
    q = meta["oof_quantiles"][key]
    return float(np.interp(score, q["values"], q["grid"]) * 100)


def verdict(meta, score):
    """Tightest alert budget (in %) at which this score would be an alert, or None. Cut-offs come from the out-of-fold scores."""
    hit = [float(q) for q, c in meta["budget_cutoffs"].items() if score >= c]
    return min(hit) if hit else None


def range_warnings(meta, feats, keys):
    """Plain-English warnings for the given features whose value is outside the 1st-99th percentile of the real weeks."""
    out = []
    for k in keys:
        lo, hi = meta["p01"][k], meta["p99"][k]
        if feats[k] > hi:
            out.append(f"{LABELS.get(k, k)} = {feats[k]:g} is above what 99% of real weeks show ({hi:g}); the score here is an extrapolation.")
        elif feats[k] < lo:
            out.append(f"{LABELS.get(k, k)} = {feats[k]:g} is below what 99% of real weeks show ({lo:g}).")
    return out


def sweep(models, meta, baseline, current, knob, n=50, overrides=None):
    """Score and risk percentile as ONE knob sweeps 0..its maximum, everything else held at its current value. Computed live from the models."""
    vmax = meta["knob_max"][knob]
    vals = np.unique(np.round(np.linspace(0, vmax, n)))
    rows = []
    for v in vals:
        cur = dict(current); cur[knob] = v
        f, _ = rebuild(baseline, cur, overrides)
        rows.append(f)
    df = pd.DataFrame(rows)
    p = predict(models, meta, df)
    return pd.DataFrame({"value": vals, "score": p["score"].to_numpy(), "score_generic": p["score_generic"].to_numpy(),
                         "pct": [percentile(meta, "score", s) for s in p["score"]], "pct_generic": [percentile(meta, "score_generic", s) for s in p["score_generic"]]})


PRESET_NOTE = ("Presets are written from the one-sentence scenario descriptions only (not from any insider's data); the numbers are illustrative multiples of a typical week.")


def presets(baseline):
    """Name -> {knob: value}, derived from answers/scenarios.txt text and the typical week: round multiples with small floors."""
    b = {k: baseline[k] for k in SLIDERS}
    return {
        "Normal week": {},
        "After-hours data copying": {"n_after_hours": max(b["n_after_hours"], 8), "n_device": max(b["n_device"] * 3, b["n_device"] + 6), "n_file": max(b["n_file"] * 3, b["n_file"] + 10)},
        "Leak-site visit": {"wb_n_leak": 3, "n_after_hours": max(b["n_after_hours"], 6)},
        "Job-hunting week": {"wb_n_job": 15, "em_n_ext_recip": b["em_n_ext_recip"] + 5, "n_device": max(b["n_device"] * 3, b["n_device"] + 6)},
    }
