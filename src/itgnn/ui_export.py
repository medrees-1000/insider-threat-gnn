"""Export everything the analyst UI shows, from out-of-fold models only.

`python -m itgnn.ui_export scores|edges|metrics|all`. Scores: XGBoost W2+E+H and generic-feature scores per user-week,
each from the fold model that did NOT train on that user, averaged over the 5 repeats; explanations are XGBoost's own
`pred_contribs` (TreeSHAP in log-odds) grouped into a partition of the feature columns.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from itgnn.comms_audit import HIST_PREFIXES, audit_groups
from itgnn.comms_eval import load_full
from itgnn.commslogs import comms_feature_sets
from itgnn.data import load_config, prepare
from itgnn.evaluation import make_folds
from itgnn.weekly_features import FEATURE_SETS, drop_warmup
from itgnn.windows import load_attack_windows

GROUP_KEYS = {  # explanation group -> parquet column
    "W2 base": "w2_base", "W2 history/deviation": "w2_hist_dev", "email volume": "email_volume",
    "email recipients/attachments": "email_recipients", "web volume": "web_volume", "web domain counts": "web_domains",
    "web after-hours": "web_after_hours", "named-site": "named_site", "first-seen": "first_seen"}
MAIN, GENERIC = "W2+E+H", "W2+E+H generic"


def explanation_groups(cols):
    """A PARTITION of `cols` into the 9 explanation groups (every column in exactly one group).

    Same families as the audit (each with its own hist/dev columns); the audit's cross-cutting "history/deviation"
    group is replaced by "W2 history/deviation" (the hist/dev columns of the five W2 counts), so contributions add up.
    """
    cols = list(cols)
    groups = {k: v for k, v in audit_groups(cols).items() if k != "history/deviation"}
    used = {c for v in groups.values() for c in v}
    groups["W2 history/deviation"] = [c for c in cols if c.startswith(HIST_PREFIXES) and c not in used]
    flat = [c for v in groups.values() for c in v]
    assert sorted(flat) == sorted(cols), "explanation groups must partition the columns"
    assert set(groups) == set(GROUP_KEYS)
    return {k: groups[k] for k in GROUP_KEYS}


def make_xgb(ytr, seed, n_estimators=200):
    """The fixed notebook-04/07 XGBoost recipe (not tuned)."""
    return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=3, learning_rate=0.1,
                             scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), n_jobs=4, random_state=seed, verbosity=0)


def fold_scores(weekly, folds, sets, seed, n_estimators=200):
    """One row per (set, fold, test row): out-of-fold score; for the main set also the group contributions + bias.

    weekly needs user, week_start, label and every column of `sets` (dict name -> columns). Each fold model is fit on the
    training users' rows only and scores the test users' rows only, so no row is scored by a model that trained on it.
    """
    y = weekly["label"].to_numpy()
    users = weekly["user"]
    groups = explanation_groups(sets[MAIN])
    parts = []
    for f in folds:
        tr, te = users.isin(set(f["train"])).to_numpy(), users.isin(set(f["test"])).to_numpy()
        assert not (tr & te).any()
        for name, cols in sets.items():
            model = make_xgb(y[tr], seed, n_estimators).fit(weekly.loc[tr, cols], y[tr])
            Xte = weekly.loc[te, cols]
            p = pd.DataFrame({"user": weekly.loc[te, "user"].to_numpy(), "week_start": weekly.loc[te, "week_start"].to_numpy(), "repeat": f["repeat"],
                              "fold": f["fold"], "set": name, "score": model.predict_proba(Xte)[:, 1]})
            if name == MAIN:
                c = model.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)  # (n, n_cols + 1); last column = bias
                ci = {col: i for i, col in enumerate(cols)}
                for g, gc in groups.items():
                    p[GROUP_KEYS[g]] = c[:, [ci[x] for x in gc]].sum(axis=1)
                p["bias"] = c[:, -1]
                p["margin"] = c.sum(axis=1)
            parts.append(p)
    return pd.concat(parts, ignore_index=True)


def aggregate_scores(long):
    """Mean over the repeats per user-week: score, score_sd (sample sd over repeats), score_generic and contributions."""
    main = long[long["set"] == MAIN]
    gen = long[long["set"] == GENERIC]
    k = ["user", "week_start"]
    contrib = list(GROUP_KEYS.values()) + ["bias"]
    out = main.groupby(k).agg(score=("score", "mean"), score_sd=("score", "std"), n_repeats=("score", "size"), **{c: (c, "mean") for c in contrib}).reset_index()
    out["score_generic"] = out[k].merge(gen.groupby(k)["score"].mean().rename("score_generic").reset_index(), on=k, how="left")["score_generic"].to_numpy()
    out["score_sd"] = out["score_sd"].fillna(0.0)
    return out


def eval_table(weekly, windows, insiders_csv):
    """Labels for the evaluation view ONLY (never joined into ui_scores)."""
    ins = pd.read_csv(insiders_csv)
    scen = ins[ins["dataset"].astype(str) == "4.2"].set_index("user")["scenario"]
    w = windows.set_index("user")
    e = weekly[["user", "week_start", "label"]].copy()
    e["scenario"] = e["user"].map(scen).fillna(0).astype("int8")
    e["window_start"] = e["user"].map(w["start"])
    e["window_end"] = e["user"].map(w["end"])
    return e


def export_scores(cfg, out_dir="results", n_estimators=200, log=print):
    t0 = time.time()
    d = prepare(cfg)
    windows = load_attack_windows(cfg["raw_dir"])
    full = load_full(d, windows)
    weekly = drop_warmup(full).reset_index(drop=True)
    allsets = comms_feature_sets(FEATURE_SETS["W2"])
    sets = {MAIN: allsets[MAIN], GENERIC: allsets[GENERIC]}
    folds = make_folds(d["users"], d["labels"], cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], cfg["seed"])
    log(f"{len(weekly):,} scored rows, {len(folds)} folds x {len(sets)} sets = {len(folds) * len(sets)} fits")
    long = fold_scores(weekly, folds, sets, cfg["seed"], n_estimators)
    log(f"fits done in {time.time() - t0:.0f}s")

    check = {"n_rows": int(len(weekly)), "n_folds": len(folds)}
    m = long[long["set"] == MAIN]
    check["max_abs(sigmoid(sum of contributions + bias) - score)"] = float(np.abs(1 / (1 + np.exp(-m["margin"])) - m["score"]).max())
    gsum = m[list(GROUP_KEYS.values()) + ["bias"]].sum(axis=1)
    check["max_abs(sum of group contributions + bias - margin)"] = float(np.abs(gsum - m["margin"]).max())
    saved = pd.read_parquet(Path(out_dir) / "comms_25fold_oof_scores.parquet", columns=["user", "week_start", "repeat", "fold", "model", "score"])
    for sname, mname in [(MAIN, "XGB W2+E+H"), (GENERIC, "XGB W2+E+H generic")]:
        a = long[long["set"] == sname].merge(saved[saved["model"] == mname], on=["user", "week_start", "repeat", "fold"], suffixes=("", "_saved"))
        assert len(a) == len(long[long["set"] == sname]) == len(saved[saved["model"] == mname]), "rows differ from the saved scores"
        check[f"max_abs_diff_vs_saved_oof[{sname}]"] = float((a["score"] - a["score_saved"]).abs().max())
    agg = aggregate_scores(long)
    assert agg["n_repeats"].eq(cfg["cv"]["n_repeats"]).all(), "every row must be scored once per repeat"
    cal = weekly[["user", "week_start", "cal_week"]]
    agg = agg.merge(cal, on=["user", "week_start"], how="left")
    check.update({"n_repeats_per_row": [int(agg["n_repeats"].min()), int(agg["n_repeats"].max())], "n_users": int(agg["user"].nunique()),
                  "groups": {k: len(v) for k, v in explanation_groups(sets[MAIN]).items()}})
    num = [c for c in agg.columns if c not in ("user", "week_start", "n_repeats", "cal_week")]
    agg[num] = agg[num].astype("float32")
    agg = agg.drop(columns="n_repeats").sort_values(["user", "week_start"]).reset_index(drop=True)
    assert not any("label" in c for c in agg.columns)
    out = Path(out_dir)
    agg.to_parquet(out / "ui_scores.parquet", index=False)
    eval_table(weekly, windows, Path(cfg["raw_dir"]) / "answers" / "insiders.csv").to_parquet(out / "ui_eval.parquet", index=False)
    check["file_mb"] = {f: round((out / f).stat().st_size / 2**20, 2) for f in ["ui_scores.parquet", "ui_eval.parquet"]}
    check["seconds"] = round(time.time() - t0, 1)
    json.dump(check, open(out / "ui_scores_check.json", "w"), indent=1)
    log(json.dumps(check, indent=1))
    return check


def export_edges(cfg, out_dir="results", budget=2.0, n_domains=10, n_recipients=5, log=print):
    """Neighbourhoods of the flagged weeks (top `budget`% of ui_scores): ui_weeks.parquet (one row per flagged user-week) and
    ui_edges.parquet (LOGGED_ON / VISITED / EMAILED rows). Every category-domain visit of a flagged week is kept; the other
    domains are the top `n_domains` by visits; recipients are the internal users mailed most (via the static LDAP table)."""
    from itgnn.comms_graph import address_map_from_ldap, explode_recipients
    from itgnn.commslogs import WEB_CATEGORIES, match_suffix
    from itgnn.ui_data import CATEGORY_NOTE, format_groups, rank_alerts, top_groups, week_key  # noqa: F401
    from itgnn.windows import _week_start

    t0 = time.time()
    out = Path(out_dir)
    sc = pd.read_parquet(out / "ui_scores.parquet")
    fl = rank_alerts(sc, budget)
    wk = fl[["user", "week_start"]]
    weeks = fl.copy()
    weeks["key"] = [week_key(u, w) for u, w in zip(weeks["user"], weeks["week_start"])]
    weeks["top_groups"] = [format_groups(top_groups(r, 3)) for _, r in fl.iterrows()]
    log(f"{len(fl)} flagged weeks (top {budget}% of {len(sc):,})")

    d = prepare(cfg)
    lg = d["logon"][d["logon"]["activity"] == "Logon"]
    lg = pd.DataFrame({"user": lg["user"].to_numpy(), "week_start": _week_start(lg["date"]).to_numpy(), "pc": lg["pc"].to_numpy()})
    peers = lg.drop_duplicates().groupby(["week_start", "pc"])["user"].nunique().rename("n_users").reset_index()
    pc = lg.merge(wk, on=["user", "week_start"]).groupby(["user", "week_start", "pc"]).size().rename("weight").reset_index()
    pc = pc.merge(peers, on=["week_start", "pc"], how="left")
    pc = pd.DataFrame({"user": pc["user"], "week_start": pc["week_start"], "edge_type": "LOGGED_ON", "other_id": pc["pc"], "other_kind": "PC",
                       "weight": pc["weight"].astype(float), "category": "", "other_users": (pc["n_users"] - 1).astype(float)})

    agg = pd.read_parquet("data/processed/http_user_week_domain.parquet").merge(wk, on=["user", "week_start"])
    cat = {dom: k for k, v in WEB_CATEGORIES.items() for dom in agg["domain"].drop_duplicates()[match_suffix(agg["domain"].drop_duplicates(), v)]}
    agg["category"] = agg["domain"].map(cat).fillna("")
    agg = agg.sort_values(["user", "week_start", "n", "domain"], ascending=[True, True, False, True])
    rank = agg[agg["category"] == ""].groupby(["user", "week_start"]).cumcount()
    keep = agg[(agg["category"] != "") | agg.index.isin(agg[agg["category"] == ""].index[rank < n_domains])]
    dom = pd.DataFrame({"user": keep["user"], "week_start": keep["week_start"], "edge_type": "VISITED", "other_id": keep["domain"], "other_kind": "Domain",
                        "weight": keep["n"].astype(float), "category": keep["category"], "other_users": np.nan})

    addr, _ = address_map_from_ldap(cfg["raw_dir"])
    em = explode_recipients(pd.read_parquet("data/processed/email.parquet"))
    em["week_start"] = _week_start(em["date"])
    em = em.merge(wk, on=["user", "week_start"])
    em["ruser"] = em["recip"].map(addr)
    ext = em[em["ruser"].isna()].groupby(["user", "week_start"]).size().rename("n_external_recipients")
    inn = em[em["ruser"].notna() & (em["ruser"] != em["user"])].groupby(["user", "week_start", "ruser"]).size().rename("weight").reset_index()
    inn = inn.sort_values(["user", "week_start", "weight", "ruser"], ascending=[True, True, False, True])
    inn = inn[inn.groupby(["user", "week_start"]).cumcount() < n_recipients]
    mail = pd.DataFrame({"user": inn["user"], "week_start": inn["week_start"], "edge_type": "EMAILED", "other_id": inn["ruser"], "other_kind": "User",
                         "weight": inn["weight"].astype(float), "category": "", "other_users": np.nan})
    weeks = weeks.merge(ext.reset_index(), on=["user", "week_start"], how="left")
    weeks["n_external_recipients"] = weeks["n_external_recipients"].fillna(0).astype("int32")
    edges = pd.concat([pc, dom, mail], ignore_index=True).sort_values(["user", "week_start", "edge_type", "other_id"]).reset_index(drop=True)
    weeks.to_parquet(out / "ui_weeks.parquet", index=False)
    edges.to_parquet(out / "ui_edges.parquet", index=False)
    stats = {"flagged_weeks": len(weeks), "flagged_users": int(weeks["user"].nunique()), "edges": {k: int(v) for k, v in edges["edge_type"].value_counts().items()},
             "pcs": int(edges.loc[edges["edge_type"] == "LOGGED_ON", "other_id"].nunique()), "domains": int(edges.loc[edges["edge_type"] == "VISITED", "other_id"].nunique()),
             "category_domain_edges": int((edges["category"] != "").sum()), "recipient_users": int(edges.loc[edges["edge_type"] == "EMAILED", "other_id"].nunique()),
             "file_mb": {f: round((out / f).stat().st_size / 2**20, 3) for f in ["ui_weeks.parquet", "ui_edges.parquet"]}, "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(out / "ui_edges_check.json", "w"), indent=1)
    log(json.dumps(stats, indent=1))
    return stats


def build_caches(cfg, processed="data/processed", log=print):
    """Parquet caches the exports read, built from data/raw only if missing: email.parquet (content dropped, about 25 s) and
    http_user_week_domain.parquet (streamed once, about 100 s). load_log / load_full cache the rest themselves."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    from itgnn.commslogs import aggregate_http, combine_parts
    from itgnn.windows import DATE_FORMAT

    raw, proc = Path(cfg["raw_dir"]), Path(processed)
    proc.mkdir(parents=True, exist_ok=True)
    e, h = proc / "email.parquet", proc / "http_user_week_domain.parquet"
    if not e.exists():
        cols = ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments"]
        t = pacsv.read_csv(raw / "email.csv", parse_options=pacsv.ParseOptions(newlines_in_values=True),
                           convert_options=pacsv.ConvertOptions(include_columns=cols, column_types={c: pa.string() for c in cols if c != "size"}))
        df = t.to_pandas()
        df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)  # strict
        df.to_parquet(e, index=False)
        log(f"built {e} ({len(df):,} rows)")
    if not h.exists():
        st = aggregate_http(raw / "http.csv", proc / "http_parts")
        combine_parts(proc / "http_parts").to_parquet(h, index=False)
        log(f"built {h} ({st['rows']:,} http rows, {st['url_fail']} unparsable URLs)")


BUDGET_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def export_model(cfg, out_dir="results", log=print):
    """Final models for the what-if scorer: XGBoost W2+E+H and generic trained on ALL scored rows (same fixed hyperparameters as the CV runs),
    saved as JSON, plus final_model_meta.json (feature order, label-free medians and 1st/99th percentiles, budget cut-offs and score ->
    percentile tables from the OUT-OF-FOLD scores, explanation groups, a feature round-trip check) and final_model_rows.parquet
    (features of the flagged top-2% weeks, for the optional real-week baseline; no labels)."""
    from itgnn.ui_data import budget_cutoffs, rank_alerts
    from itgnn.whatif import BASE_COLS, FROZEN, LABELS, SLIDERS, feature_columns, rebuild

    t0 = time.time()
    out = Path(out_dir)
    d = prepare(cfg)
    windows = load_attack_windows(cfg["raw_dir"])
    weekly = drop_warmup(load_full(d, windows)).reset_index(drop=True)
    y = weekly["label"].to_numpy()
    allsets = comms_feature_sets(FEATURE_SETS["W2"])
    fm, fg = allsets[MAIN], allsets[GENERIC]
    assert sorted(fm) == sorted(feature_columns())
    for name, cols, f in (("main", fm, "final_xgb_main.json"), ("generic", fg, "final_xgb_generic.json")):
        m = make_xgb(y, cfg["seed"]).fit(weekly[cols], y)
        m.get_booster().save_model(str(out / f))
    sc = pd.read_parquet(out / "ui_scores.parquet")
    cut = budget_cutoffs(sc)
    grid = np.r_[np.arange(0, 0.99, 0.001), np.arange(0.99, 1.00001, 0.0001)]
    oofq = {k: {"grid": [round(float(g), 5) for g in grid], "values": [float(v) for v in np.quantile(sc[k].to_numpy(float), grid)]} for k in ("score", "score_generic")}
    X = weekly[fm]
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(weekly), 2000, replace=False)
    diffs = []
    for i in idx:   # round trip: rebuild all 79 features from (current counts, history) and compare with the stored values
        row = weekly.loc[i, fm].astype(float).to_dict()
        rebuilt, _ = rebuild(row, {c: row[c] for c in BASE_COLS if c not in FROZEN})
        diffs.append(max(abs(rebuilt[c] - row[c]) for c in fm))
    knob_max = {k: int(max(10, np.ceil(3 * float(np.quantile(weekly[k], 0.99))))) for k in LABELS}
    meta = {"features_main": fm, "features_generic": fg, "groups": explanation_groups(fm), "medians": {c: float(np.median(X[c])) for c in fm},
            "p01": {c: float(np.quantile(X[c], 0.01)) for c in fm}, "p99": {c: float(np.quantile(X[c], 0.99)) for c in fm}, "knob_max": knob_max,
            "budget_cutoffs": {str(q): float(v) for q, v in cut.items()}, "oof_quantiles": oofq, "n_training_rows": int(len(weekly)), "n_attack_weeks": int(y.sum()),
            "roundtrip_check": {"rows": len(idx), "max_abs_diff_all_79_features": float(max(diffs))}, "xgboost": xgb.__version__, "sliders": SLIDERS}
    json.dump(meta, open(out / "final_model_meta.json", "w"), indent=1)
    top = rank_alerts(sc, 2.0)[["user", "week_start"]]
    rows = top.merge(weekly[["user", "week_start"] + fm], on=["user", "week_start"], how="left")
    rows[fm] = rows[fm].astype("float32")
    assert rows[fm].notna().all().all() and not any("label" in c for c in rows.columns)
    rows.to_parquet(out / "final_model_rows.parquet", index=False)
    log(json.dumps({"seconds": round(time.time() - t0, 1), "rows": len(weekly), "roundtrip": meta["roundtrip_check"],
                    "sizes_mb": {f: round((out / f).stat().st_size / 2**20, 3) for f in ["final_xgb_main.json", "final_xgb_generic.json", "final_model_meta.json", "final_model_rows.parquet"]}}, indent=1))
    return meta


WEEK_FACT_COLS = ["n_logon", "n_after_hours", "n_device", "n_file", "em_n", "em_n_ext_recip", "wb_n"]


def export_week_facts(cfg, out_dir="results", log=print):
    """ui_week_facts.parquet: for every scored user-week, seven plain activity counts and the person's own past-only average of each (hist_mean_*),
    for the Investigate 'this week vs their usual' table. Existing feature values only; no labels, no scores."""
    t0 = time.time()
    d = prepare(cfg)
    windows = load_attack_windows(cfg["raw_dir"])
    weekly = drop_warmup(load_full(d, windows)).reset_index(drop=True)
    cols = WEEK_FACT_COLS + [f"hist_mean_{c}" for c in WEEK_FACT_COLS]
    out = weekly[["user", "week_start"] + cols].copy()
    out[cols] = out[cols].astype("float32")
    assert out[cols].notna().all().all() and not any("label" in c for c in out.columns)
    out.to_parquet(Path(out_dir) / "ui_week_facts.parquet", index=False)
    log(json.dumps({"seconds": round(time.time() - t0, 1), "rows": len(out), "mb": round((Path(out_dir) / "ui_week_facts.parquet").stat().st_size / 2**20, 2)}))
    return out


def export_curve(cfg, out_dir="results", log=print):
    """ui_budget_curve.json: insiders caught and normal users flagged (mean and sd over the 5 repeats of the 5x5 user-split CV) for W2, generic and
    W2+E+H on a grid of alert budgets. Same alert_budget function and seeds as ui_metrics.json, so the shared budgets agree."""
    from itgnn.evaluation import alert_budget

    t0 = time.time()
    d = prepare(cfg)
    windows = load_attack_windows(cfg["raw_dir"])
    weekly = drop_warmup(load_full(d, windows)).reset_index(drop=True)
    allsets = comms_feature_sets(FEATURE_SETS["W2"])
    sets = {k: allsets[k] for k in ["W2", GENERIC, MAIN]}
    labels = d["labels"]
    ins = pd.read_csv(Path(cfg["raw_dir"]) / "answers" / "insiders.csv")
    scen = ins[ins["dataset"].astype(str) == "4.2"].set_index("user")["scenario"]
    folds = make_folds(d["users"], labels, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], cfg["seed"])
    long = fold_scores(weekly, folds, sets, cfg["seed"])
    oof = long[["user", "week_start", "repeat", "fold", "set", "score"]].rename(columns={"set": "model"}).merge(weekly[["user", "week_start", "label"]], on=["user", "week_start"])
    bud = alert_budget(oof[["user", "repeat", "fold", "model", "score", "label"]], labels, scen.to_dict(), BUDGET_GRID, cfg["seed"])
    res = {"budgets": BUDGET_GRID, "n_insiders": len(labels), "n_normal_users": len(d["users"]) - len(labels), "n_scored_weeks": int(len(weekly)),
           "protocol": "XGBoost, 5x5 user-split CV; mean and sd over the 5 repeats", "sets": {}}
    for name in sets:
        g = bud[bud["model"] == name].groupby("q%")
        res["sets"][name] = {"caught": [round(float(g["caught"].mean().loc[q]), 2) for q in BUDGET_GRID], "caught_sd": [round(float(g["caught"].std().loc[q]), 2) for q in BUDGET_GRID],
                             "normal_flagged": [round(float(g["normal flagged"].mean().loc[q]), 2) for q in BUDGET_GRID], "normal_flagged_sd": [round(float(g["normal flagged"].std().loc[q]), 2) for q in BUDGET_GRID],
                             "weeks_flagged": [round(float(g["weeks flagged"].mean().loc[q]), 1) for q in BUDGET_GRID]}
    res["seconds"] = round(time.time() - t0, 1)
    from itgnn.ui_data import assert_aggregates_only
    assert_aggregates_only(res)
    json.dump(res, open(Path(out_dir) / "ui_budget_curve.json", "w"), indent=1)
    log(f"saved ui_budget_curve.json in {res['seconds']}s")
    return res


def export_facts(cfg, out_dir="results", log=print):
    """ui_rigor_facts.json: evaluation AGGREGATES only (scalars and small dicts; no per-row labels, scenarios or windows)."""
    from sklearn.metrics import roc_auc_score

    from itgnn.comms_audit import draw_sample, truncation_check

    t0 = time.time()
    out = Path(out_dir)
    d = prepare(cfg)
    windows = load_attack_windows(cfg["raw_dir"])
    full = load_full(d, windows)
    weekly = drop_warmup(full).reset_index(drop=True)
    labels = d["labels"]
    # 1. the tenure leak the first baseline had (users who stay the whole period are almost all normal)
    lg = d["logon"][d["logon"]["activity"] == "Logon"]
    days = lg.assign(day=lg["date"].dt.normalize()).groupby("user")["day"].nunique()
    is_ins = days.index.isin(labels)
    full_period = int(days[~is_ins].mode().iloc[0])   # the typical full tenure: the most common number of active days among normal users
    tenure = {"active_days_auc_for_insider": round(float(roc_auc_score(is_ins, days.to_numpy())), 3), "typical_full_tenure_days": int(full_period),
              "normal_users_covering_the_full_period": round(float((days[~is_ins] == full_period).mean()), 3),
              "insiders_covering_the_full_period": round(float((days[is_ins] == full_period).mean()), 3)}
    # 2. the cross-validation protocol
    folds = make_folds(d["users"], labels, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], cfg["seed"])
    cv = {"folds": len(folds), "repeats": cfg["cv"]["n_repeats"], "max_users_shared_by_train_and_test": max(len(set(f["train"]) & set(f["test"])) for f in folds)}
    # 3. causality: features recomputed from logs cut at the end of each sampled week
    allsets = comms_feature_sets(FEATURE_SETS["W2"])
    email = pd.read_parquet("data/processed/email.parquet")
    agg = pd.read_parquet("data/processed/http_user_week_domain.parquet")
    sample = draw_sample(weekly, n=200, n_attack=30, n_weeks=15, seed=cfg["seed"])
    chk = truncation_check(sample, d["logon"], d["device"], d["file"], email, agg, full, allsets[MAIN])
    ctrl = truncation_check(sample[sample["week_start"] == sample["week_start"].iloc[0]], d["logon"], d["device"], d["file"], email, agg, full, allsets[MAIN], cut_days=3)
    causality = {"user_weeks_checked": int(len(chk)), "features_compared": len(allsets[MAIN]), "max_abs_difference": float(chk["max_abs_diff"].max()),
                 "control_rows_that_differ": int((ctrl["max_abs_diff"] > 1e-9).sum()), "control_rows": int(len(ctrl))}
    # 4. graph models vs controls (only if the stage-1 graph run is present in this clone)
    graph = None
    g1, o25 = out / "gnn2_stage1_oof_scores.parquet", out / "comms_25fold_oof_scores.parquet"
    if g1.exists() and o25.exists():
        from itgnn.evaluation import evaluate_scores
        g = pd.read_parquet(g1)
        x = pd.read_parquet(o25)
        x = x[(x["repeat"] == 0) & (x["model"] == "XGB W2+E+H")]
        mean_ap = lambda df: round(float(np.mean([evaluate_scores(gg["label"], gg["score"], 100)["pr_auc"] for _, gg in df.groupby("fold")])), 3)
        graph = {"no_edge_mlp": mean_ap(g[g["model"] == "M0 MLP"]), "graphsage": mean_ap(g[g["model"] == "M1 GraphSAGE"]), "graphsage_shuffled_edges": mean_ap(g[g["model"] == "M2 GraphSAGE shuffled"]),
                 "xgboost_same_features": mean_ap(x), "folds": 5, "gate_passed": False}
    res = {"tenure_leak": tenure, "cross_validation": cv, "causality_check": causality, "graph_controls": graph, "seconds": round(time.time() - t0, 1)}
    from itgnn.ui_data import assert_aggregates_only
    assert_aggregates_only(res)
    json.dump(res, open(out / "ui_rigor_facts.json", "w"), indent=1)
    log(json.dumps(res, indent=1))
    return res


def _budget_rows(bud):
    """alert_budget output -> {model: {q: {caught mean/std, normal flagged mean/std, per scenario}}} (JSON-friendly)."""
    out = {}
    for (m, q), g in bud.groupby(["model", "q%"]):
        out.setdefault(m, {})[str(q)] = {c: [round(float(g[c].mean()), 2), round(float(g[c].std(ddof=1)) if len(g) > 1 else 0.0, 2)]
                                         for c in ["caught", "normal flagged", "caught s1", "caught s2", "caught s3", "weeks flagged"]}
    return out


def _list_stats(flagged, ev, insiders):
    f = flagged.merge(ev[["user", "week_start", "label"]], on=["user", "week_start"])
    return {"alerts": int(len(f)), "attack weeks flagged": int(f["label"].sum()), "week precision": round(float(f["label"].mean()), 3),
            "distinct users": int(f["user"].nunique()), "insiders caught": int(f.loc[f["label"] == 1, "user"].nunique()),
            "normal users flagged": int(f.loc[~f["user"].isin(insiders), "user"].nunique())}


def export_metrics(cfg, out_dir="results", log=print):
    """ui_metrics.json: everything the metrics panel shows, recomputed from XGBoost fits (no notebook outputs needed)."""
    from itgnn.comms_eval import double_holdout_masks, same_rows_user_split_ap, time_split_masks
    from itgnn.commslogs import WEB_CATEGORIES, match_suffix
    from itgnn.evaluation import alert_budget, evaluate_scores
    from itgnn.ui_data import BUDGETS, rank_alerts

    t0 = time.time()
    out = Path(out_dir)
    d = prepare(cfg)
    windows = load_attack_windows(cfg["raw_dir"])
    weekly = drop_warmup(load_full(d, windows)).reset_index(drop=True)
    y = weekly["label"].to_numpy()
    allsets = comms_feature_sets(FEATURE_SETS["W2"])
    sets = {k: allsets[k] for k in ["W2", "W2+calendar", GENERIC, MAIN]}
    labels, seed = d["labels"], cfg["seed"]
    ins = pd.read_csv(Path(cfg["raw_dir"]) / "answers" / "insiders.csv")
    scen = ins[ins["dataset"].astype(str) == "4.2"].set_index("user")["scenario"]
    folds = make_folds(d["users"], labels, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)

    long = fold_scores(weekly, folds, sets, seed)
    lab = weekly[["user", "week_start", "label", "cal_week"]]
    oof = long[["user", "week_start", "repeat", "fold", "set", "score"]].rename(columns={"set": "model"}).merge(lab, on=["user", "week_start"])
    log(f"25-fold fits done {time.time() - t0:.0f}s")
    per = []
    for (m, r, f), g in oof.groupby(["model", "repeat", "fold"]):
        ev = evaluate_scores(g["label"], g["score"], 100)
        per.append({"model": m, "repeat": r, "fold": f, "pr_auc": ev["pr_auc"], "roc_auc": ev["roc_auc"]})
    per = pd.DataFrame(per)
    pr = {m: {"pr_auc_mean": round(float(g["pr_auc"].mean()), 4), "pr_auc_std": round(float(g["pr_auc"].std()), 4), "roc_auc_mean": round(float(g["roc_auc"].mean()), 4)}
          for m, g in per.groupby("model")}
    pv = per.assign(i=per["repeat"] * 5 + per["fold"]).pivot_table(index="i", columns="model", values="pr_auc")
    paired = {}
    for hi, lo in [(MAIN, "W2"), (GENERIC, "W2"), ("W2+calendar", "W2"), (MAIN, "W2+calendar"), (MAIN, GENERIC), (GENERIC, "W2+calendar")]:
        dl = pv[hi] - pv[lo]
        paired[f"{hi} - {lo}"] = {"mean": round(float(dl.mean()), 4), "folds_positive": f"{int((dl > 0).sum())}/{len(dl)}"}
    bud = alert_budget(oof[["user", "repeat", "fold", "model", "score", "label"]], labels, scen.to_dict(), BUDGETS, seed)
    log(f"25-fold metrics {time.time() - t0:.0f}s")

    # time split (same users) and double holdout (unseen users x future weeks); reference = user-split on the same rows
    ref_oof = oof.copy()
    ts, dh, ts_frames, dh_frames = [], [], [], []
    for T in (29, 39, 49):
        tr, te = time_split_masks(weekly, T)
        for name in ("W2", MAIN, GENERIC):
            cols = sets[name]
            s = make_xgb(y[tr], seed).fit(weekly.loc[tr, cols], y[tr]).predict_proba(weekly.loc[te, cols])[:, 1]
            ev = evaluate_scores(y[te], s, 100)
            ref = same_rows_user_split_ap(ref_oof, name, T)
            ts.append({"T": T, "set": name, "pr_auc": round(ev["pr_auc"], 4), "user_split_same_rows": round(ref, 4), "R": round(ev["pr_auc"] / ref, 3),
                       "train_attack_weeks": int((y[tr] == 1).sum()), "test_attack_weeks": int((y[te] == 1).sum())})
            ts_frames.append(pd.DataFrame({"user": weekly.loc[te, "user"].to_numpy(), "repeat": 0, "fold": 0, "model": f"{name}|{T}", "score": s, "label": y[te]}))
        parts = {n: [] for n in ("W2", MAIN, GENERIC)}
        for f in [f for f in folds if f["repeat"] == 0]:
            tr, te = double_holdout_masks(weekly, f, T)
            for name in parts:
                cols = sets[name]
                s = make_xgb(y[tr], seed).fit(weekly.loc[tr, cols], y[tr]).predict_proba(weekly.loc[te, cols])[:, 1]
                parts[name].append(pd.DataFrame({"y": y[te], "s": s}))
        for name, ps in parts.items():
            pp = pd.concat(ps)
            ap = evaluate_scores(pp["y"], pp["s"], 100)["pr_auc"]
            ref = same_rows_user_split_ap(ref_oof, name, T)
            dh.append({"T": T, "set": name, "pr_auc_pooled": round(ap, 4), "user_split_same_rows": round(ref, 4), "R": round(ap / ref, 3), "test_attack_weeks": int(pp["y"].sum())})
    tb = alert_budget(pd.concat(ts_frames, ignore_index=True), labels, scen.to_dict(), BUDGETS, seed)
    log(f"time split / double holdout {time.time() - t0:.0f}s")

    # the UI's own alert lists (global top-q% of the averaged OOF scores) vs a real weekly queue (top-q% inside each calendar week)
    sc = pd.read_parquet(out / "ui_scores.parquet")
    ev_t = pd.read_parquet(out / "ui_eval.parquet")
    insiders = set(ev_t.loc[ev_t["scenario"] > 0, "user"])
    glob = {str(q): _list_stats(rank_alerts(sc, q), ev_t, insiders) for q in BUDGETS}
    weekly_q = {str(q): _list_stats(pd.concat([rank_alerts(sc, q, w) for w in sorted(sc["cal_week"].unique())], ignore_index=True), ev_t, insiders) for q in BUDGETS}

    agg = pd.read_parquet("data/processed/http_user_week_domain.parquet", columns=["user", "domain"]).drop_duplicates()
    marker = {}
    for cat, sc_id in [("leak", 1), ("download", 3)]:
        vis = set(agg.loc[match_suffix(agg["domain"], WEB_CATEGORIES[cat]), "user"])
        s_ins = set(scen[scen == sc_id].index)
        marker[cat] = {"visitors": len(vis), "scenario": sc_id, "scenario insiders": len(s_ins), "visitors that are scenario insiders": len(vis & s_ins),
                       "normal-user visitors": len(vis - set(scen.index))}
    res = {"n_scored_weeks": int(len(weekly)), "attack_weeks": int(y.sum()), "protocol": "XGBoost, repeated 5x5 user-split CV; time split and double holdout on repeat-0 folds",
           "pr_auc_25fold": pr, "paired_vs": paired, "alert_budget_25fold": _budget_rows(bud), "time_split": ts, "double_holdout": dh,
           "time_split_alert_budget": _budget_rows(tb), "ui_global_lists": glob, "ui_weekly_queue_lists": weekly_q, "marker_overlap": marker, "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(out / "ui_metrics.json", "w"), indent=1, default=str)
    log(f"saved ui_metrics.json ({(out / 'ui_metrics.json').stat().st_size / 1024:.0f} KB) in {time.time() - t0:.0f}s")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["caches", "scores", "edges", "metrics", "model", "curve", "facts", "weekfacts", "all"])
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.stage in ("caches", "all"):
        build_caches(cfg)
    if a.stage in ("scores", "all"):
        export_scores(cfg, cfg["results_dir"])
    if a.stage in ("edges", "all"):
        export_edges(cfg, cfg["results_dir"])
    if a.stage in ("metrics", "all"):
        export_metrics(cfg, cfg["results_dir"])
    if a.stage in ("model", "all"):
        export_model(cfg, cfg["results_dir"])
    if a.stage in ("curve", "all"):
        export_curve(cfg, cfg["results_dir"])
    if a.stage in ("facts", "all"):
        export_facts(cfg, cfg["results_dir"])
    if a.stage in ("weekfacts", "all"):
        export_week_facts(cfg, cfg["results_dir"])


if __name__ == "__main__":
    main()
