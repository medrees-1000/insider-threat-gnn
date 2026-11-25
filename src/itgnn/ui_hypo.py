"""Hypothetical workers for the "New prediction" tab (inference only, session only, never saved, never labelled).

A worker is a dict built from two sets of 12 slider values ("their usual week" -> history means, "this week" -> current counts)
plus optional connections (websites, PCs, colleagues). Connections RAISE matching counts (never lower a value the analyst set),
then `whatif.rebuild` recomputes the 79 features and the saved final models score them. Percentiles and ranks are read against the
saved out-of-fold tables, so adding a worker moves nothing for real weeks."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from itgnn import whatif as W

CAT_COUNT = {"leak": "wb_n_leak", "job": "wb_n_job", "cloud": "wb_n_cloud", "download": "wb_n_download", "competitor": "wb_n_competitor"}
CAT_NAME = {"leak": "leak site", "job": "job site", "cloud": "cloud-storage site", "download": "keylogger site", "competitor": "competitor site"}
NEW_PC = "NEW-PC (hypothetical)"
_REAL_ID = re.compile(r"^[A-Za-z]{3}\d{4}$")          # real CERT user ids look like AAF0535
_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,29}$")


def check_name(name, taken=(), real_users=()):
    """None if the name is usable, else the plain-English reason. Must not look like a real id, must be unique (case-insensitive)."""
    n = (name or "").strip()
    if not n:
        return "Give the worker a name."
    if not _NAME_OK.match(n):
        return "Use 1-30 letters, digits, spaces, '-', '_' or '.', starting with a letter or digit."
    if _REAL_ID.match(n):
        return "That looks like a real user id (three letters and four digits); pick a name that cannot be mistaken for a real person."
    low = {t.lower() for t in list(taken) + list(real_users)}
    if n.lower() in low:
        return "That name is already used."
    return None


def week_start_of(cal_week, anchor_week, anchor_start):
    """Monday of a calendar week, from one known (week number, start date) pair; scored weeks are consecutive."""
    return pd.Timestamp(anchor_start) + pd.Timedelta(days=7 * (int(cal_week) - int(anchor_week)))


def _plural(n, word):
    return f"{n:g} {word}" + ("" if n == 1 else "s")


def couple(current, sites, pcs, colleagues):
    """RAISE-ONLY coupling. current: {count: value} (the analyst's this-week values, not modified). sites: {domain: (category, visits)}.
    Returns (new current dict, notes). A count is raised only when a chosen connection needs it; nothing is ever lowered."""
    cur, notes = dict(current), []

    def up(k, to, why):
        if to > cur.get(k, 0):
            notes.append(f"{W.LABELS[k]} raised from {cur.get(k, 0):g} to {to:g}: {why}.")
            cur[k] = float(to)

    for cat, col in CAT_COUNT.items():
        chosen = {d: v for d, (c, v) in sites.items() if c == cat and v > 0}
        if chosen:
            up(col, sum(chosen.values()), "you chose " + ", ".join(f"{d} ({_plural(v, 'visit')})" for d, v in chosen.items()))
    total = sum(v for _, v in sites.values())
    if total:
        up("wb_n", total, f"you chose {_plural(len(sites), 'website')} with {_plural(total, 'visit')} in all")
    if pcs:
        up("n_pcs", len(pcs), f"you chose {_plural(len(pcs), 'computer')}")
    if colleagues:
        up("em_n", 1, "you chose colleagues who were emailed")
        up("em_n_recip", len(colleagues), f"you chose {_plural(len(colleagues), 'colleague')} as recipients")
    return cur, notes


def build_features(meta, typical, usual, this_week, sites=(), pcs=(), colleagues=(), overrides=None):
    """79 features for a worker. usual / this_week: {slider knob: value}. sites: {domain: (category, visits)}.
    'usual' sets the history mean of each slider count (made consistent like a real week); 'this_week' sets the current counts; deviations are recomputed.
    Returns (features dict, notes, current dict after coupling)."""
    base = dict(typical)
    cur_u = {c: float(typical[c]) for c in W.BASE_COLS}
    cur_u.update({k: float(v) for k, v in usual.items()})
    notes = [f"Usual week: {n}" for n in W.enforce(cur_u)]
    for k in usual:
        base[f"hist_mean_{k}"] = cur_u[k]
    for k in W.BASE_COLS:                                   # counts raised by the consistency rules (e.g. a total) move their history mean too
        if k not in usual and cur_u[k] != float(typical[k]):
            base[f"hist_mean_{k}"] = cur_u[k]
    start = {c: float(typical[c]) for c in W.BASE_COLS}       # coupling compares with the starting week's own value of every count
    start.update({k: float(v) for k, v in this_week.items()})
    cur, cnotes = couple(start, dict(sites), list(pcs), list(colleagues))
    feats, rnotes = W.rebuild(base, {k: v for k, v in cur.items() if k not in W.FROZEN}, overrides)
    return feats, notes + cnotes + rnotes, cur


def score_worker(models, meta, feats, scores_score):
    """Rating of one feature dict against the saved out-of-fold tables (nothing is added to them). scores_score: the real out-of-fold scores (array)."""
    res = W.predict(models, meta, pd.DataFrame([feats]))
    s, sg = float(res["score"].iloc[0]), float(res["score_generic"].iloc[0])
    return {"score": s, "score_generic": sg, "pct": W.percentile(meta, "score", s), "pct_generic": W.percentile(meta, "score_generic", sg), "verdict": W.verdict(meta, s),
            "verdict_generic": W.verdict(meta, sg), "rank": rank_among(scores_score, s), "n_weeks": int(len(scores_score)), "groups": {g: float(res[g].iloc[0]) for g in meta["groups"]}}


def rank_among(real_scores, score):
    """Where a score would sit among the real out-of-fold weeks (1 = riskiest). Approximate: the final model has seen every real week."""
    return int(1 + (np.asarray(real_scores) > score).sum())


def make_worker(models, meta, typical, real_scores, name, cal_week, week_start, usual, this_week, sites=(), pcs=(), colleagues=(), overrides=None):
    """One session worker: inputs, features, notes and ratings. Plain dict (kept in st.session_state only)."""
    sites = dict(sites)
    feats, notes, cur = build_features(meta, typical, usual, this_week, sites, pcs, colleagues, overrides)
    w = {"name": name.strip(), "cal_week": int(cal_week), "week_start": pd.Timestamp(week_start), "usual": dict(usual), "this_week": dict(this_week), "sites": sites,
         "pcs": list(pcs), "colleagues": list(colleagues), "features": feats, "notes": notes, "hypothetical": True}
    w.update(score_worker(models, meta, feats, real_scores))
    return w


def add_worker(workers, w, cap, real_users=()):
    """Append to the session list (returns a NEW list) or raise ValueError: cap reached or bad/duplicate name."""
    if len(workers) >= cap:
        raise ValueError(f"The session holds at most {cap} hypothetical workers; remove one first.")
    bad = check_name(w["name"], [x["name"] for x in workers], real_users)
    if bad:
        raise ValueError(bad)
    return list(workers) + [w]


def to_csv(workers, feature_cols):
    """One row per worker for download: name, week, inputs, 79 features, ratings, verdicts. Every row says hypothetical = True."""
    rows = []
    for w in workers:
        r = {"name": w["name"], "calendar_week": w["cal_week"], "week_start": f"{w['week_start']:%Y-%m-%d}",
             "websites": "; ".join(f"{d} ({c}) x{v:g}" for d, (c, v) in w["sites"].items()), "pcs": "; ".join(w["pcs"]), "colleagues_emailed": "; ".join(w["colleagues"])}
        r.update({c: w["features"][c] for c in feature_cols})
        r.update({"riskier_than_pct_of_real_weeks": round(w["pct"], 3), "riskier_than_pct_no_named_sites": round(w["pct_generic"], 3),
                  "alert_at_budget_pct": w["verdict"] if w["verdict"] is not None else "", "alert_at_budget_pct_no_named_sites": w["verdict_generic"] if w["verdict_generic"] is not None else "",
                  "approx_rank_among_real_weeks": w["rank"], "hypothetical": True})
        rows.append(r)
    return pd.DataFrame(rows).to_csv(index=False)


def domain_table(agg, categories, excluded, min_users=20):
    """Label-free domain list for the picker: every domain the models keep (>= min_users distinct users, not excluded) plus every named-site
    category domain present. Columns: domain, category ('' for ordinary sites), n_users (distinct users over the whole period)."""
    from itgnn.commslogs import match_suffix

    dom = agg["domain"].drop_duplicates()
    cat = {d: k for k, v in categories.items() for d in dom[match_suffix(dom, v)]}
    nu = agg.groupby("domain")["user"].nunique()
    keep = {d for d in nu.index if nu[d] >= min_users and d not in set(excluded)} | set(cat)
    t = pd.DataFrame({"domain": sorted(keep)})
    t["category"] = t["domain"].map(cat).fillna("")
    t["n_users"] = t["domain"].map(nu).astype("int32")
    return t.sort_values(["category", "n_users", "domain"], ascending=[False, False, True]).reset_index(drop=True)


def load_domains(results):
    """(table, note). ui_domains.parquet if present; otherwise the domains in ui_edges (flagged weeks only) with a note."""
    p = Path(results) / "ui_domains.parquet"
    if p.exists():
        return pd.read_parquet(p), None
    e = pd.read_parquet(Path(results) / "ui_edges.parquet")
    e = e[e["edge_type"] == "VISITED"]
    t = e.groupby("other_id").agg(category=("category", "first"), n_users=("user", "nunique")).reset_index().rename(columns={"other_id": "domain"})
    t = t.sort_values(["category", "n_users", "domain"], ascending=[False, False, True]).reset_index(drop=True)
    return t, f"results/ui_domains.parquet is missing, so only the {len(t)} websites seen in flagged weeks can be chosen. Create the full list with `python -m itgnn.ui_export domains`."


def augment_payload(payload, workers, cutoffs):
    """Threat-map payload plus the session's hypothetical workers. Returns the SAME object when there are no workers (so the payload is
    unchanged byte for byte); otherwise a new dict that shares everything except new lists: real records keep their indices, workers are appended
    after them, each connection is one edge (an existing website/PC/contact entry is reused, a new one appended), and the calendar gains the
    worker's week when no flagged week exists there. cutoffs: {budget %: score cut-off} of the final model (same as the rating)."""
    if not workers:
        return payload
    uw, conn, edges = list(payload["uw"]), list(payload["conn"]), list(payload["edges"])
    real_by = {(r["cw"], r["u"]): r["i"] for r in payload["uw"]}
    cidx = {(c["k"], c["n"]): i for i, c in enumerate(conn)}
    cal = {c["cw"]: c["d"] for c in payload["meta"]["calendar"]}

    def conn_of(kind, name, cat=""):
        if (kind, name) not in cidx:
            cidx[(kind, name)] = len(conn); conn.append({"k": kind, "n": name, "cat": cat})
        return cidx[(kind, name)]

    cut = {float(q): c for q, c in cutoffs.items()}
    for w in workers:
        i = len(uw)
        s = w["score"]
        inside = s >= cut[2.0]
        uw.append({"i": i, "u": w["name"], "cw": w["cal_week"], "d": f"{w['week_start']:%Y-%m-%d}", "s": round(s, 5), "p": round(w["pct"], 3), "z": round(float(np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))), 2),
                   "g": round(w["score_generic"], 5), "sd": 0.0, "c": [round(v, 2) for v in w["groups"].values()], "lvl": int((s >= cut[1.0]) + (s >= cut[0.5]) + (s >= cut[0.25])) if inside else 0,
                   "ne": int(w["features"]["em_n_ext_recip"]), "hyp": 1, "nr": 0 if inside else 1})
        cal.setdefault(w["cal_week"], f"{w['week_start']:%Y-%m-%d}")
        for dom, (cat, visits) in w["sites"].items():
            edges.append([i, conn_of("domain", dom, cat), 1, float(visits), -1])
        npc = max(len(w["pcs"]), 1)
        for pc in w["pcs"]:
            new = pc == NEW_PC
            edges.append([i, conn_of("pc", f"NEW PC of {w['name']} (hypothetical)" if new else pc), 0, round(max(w["features"]["n_logon"], 1) / npc, 2), 0 if new else -1])
        for col in w["colleagues"]:
            j = real_by.get((w["cal_week"], col))
            edges.append([i, j, 3, 1.0, -1] if j is not None else [i, conn_of("contact", col), 2, 1.0, -1])
    meta = {**payload["meta"], "hyp_conn_from": len(payload["conn"]), "calendar": [{"cw": c, "d": d} for c, d in sorted(cal.items())], "hyp": [f"{w['name']}|{w['score']:.6f}|{len(w['sites'])}{len(w['pcs'])}{len(w['colleagues'])}|{w['cal_week']}" for w in workers]}   # only used to remount the component when a worker changes
    return {**payload, "meta": meta, "uw": uw, "conn": conn, "edges": edges}


def facts_row(w):
    """The seven 'this week vs usual' counts (+ named-site total) of a worker, in the layout of one ui_week_facts row: current value and history mean."""
    from itgnn.ui_data import USUAL_ROWS

    f, row = w["features"], {}
    for k, _ in USUAL_ROWS:
        cols = W.CATEGORY_COUNTS if k == "wb_n_named" else [k]
        row[k], row[f"hist_mean_{k}"] = sum(f[c] for c in cols), sum(f[f"hist_mean_{c}"] for c in cols)
    return row


def named_sites(w):
    """Named sites the analyst chose for the worker (leak, job, cloud, keylogger, competitor), same columns as ui_data.named_sites_visited."""
    from itgnn.ui_data import NAMED_CATEGORY

    t = pd.DataFrame([{"Site": d, "Category": NAMED_CATEGORY[c], "Visits": float(v)} for d, (c, v) in w["sites"].items() if c in NAMED_CATEGORY], columns=["Site", "Category", "Visits"])
    return t.sort_values(["Visits", "Site"], ascending=[False, True]).reset_index(drop=True)
