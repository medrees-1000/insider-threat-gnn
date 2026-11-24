"""Payloads for the threat map (pure functions, no Streamlit): one JSON-able dict with a WEEK view (flagged user-weeks of a
calendar week plus their neighbours) and a CAMPAIGN view (flagged users linked through shared websites, PCs and email,
with Louvain communities). Evaluation labels are added ONLY when `insiders` is passed; otherwise the payload has no such field."""
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from itgnn.ui_data import BUDGETS, GROUP_COLS, budget_cutoffs

T_PC, T_DOM, T_MAIL, T_MAIL_USER = 0, 1, 2, 3          # edge type codes (T_MAIL_USER links two flagged user-weeks directly)
KIND_CODE = {"PC": "pc", "Domain": "domain", "User": "contact"}
EDGE_TYPE = {"LOGGED_ON": T_PC, "VISITED": T_DOM, "EMAILED": T_MAIL}
MAX_PER_WEEK = 60            # flagged user-weeks drawn per calendar week (the rest are dropped, counted in meta)
CAMPAIGN_CAPS = {"domain": 120, "pc": 80, "contact": 80}
CAMPAIGN_MAX_SHARE = 0.5     # a connector visited/used by more than this share of flagged users is background noise
MIN_CLUSTER = 3              # communities smaller than this are shown as "other"
SEED = 42
LAYOUT_SCALE = 900.0         # campaign layout extent (pixels at zoom 1); the map fits it to the screen
CAMPAIGN_TOP_CONNECTORS = 40  # connectors drawn by default (plus every named-site one); the rest behind "All neighbours"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def risk_levels(scores):
    """0..3 for a score inside the top 2% / 1% / 0.5% / 0.25% of all scored weeks (same cut-offs as the timeline)."""
    cut = budget_cutoffs(scores)
    return lambda s: int((s >= cut[1.0]) + (s >= cut[0.5]) + (s >= cut[0.25]))


def build_payload(scores, weeks, edges, insiders=None, max_per_week=MAX_PER_WEEK, seed=SEED):
    """scores: ui_scores; weeks: ui_weeks (flagged user-weeks); edges: ui_edges; insiders: optional set of true insider user ids
    (evaluation view only). Returns (payload, stats)."""
    lvl = risk_levels(scores)
    pct = scores.assign(p=scores["score"].rank(pct=True) * 100)[["user", "week_start", "p"]]
    w = weeks.merge(pct, on=["user", "week_start"], how="left").sort_values(["cal_week", "score", "user"], ascending=[True, False, True])
    over = w.groupby("cal_week").cumcount() >= max_per_week
    stats = {"flagged_user_weeks": int(len(w)), "dropped_by_cap": int(over.sum()), "max_user_weeks_in_a_week": int(w.groupby("cal_week").size().max())}
    w = w[~over].reset_index(drop=True)
    gcols = list(GROUP_COLS)
    uw = []
    for i, r in enumerate(w.itertuples(index=False)):
        d = r._asdict()
        c = [round(float(d[g]), 2) for g in gcols]
        rec = {"i": i, "u": d["user"], "cw": int(d["cal_week"]), "d": pd.Timestamp(d["week_start"]).strftime("%Y-%m-%d"), "s": round(float(d["score"]), 5),
               "p": round(float(d["p"]), 3), "z": round(float(_logit(d["score"])), 2), "g": round(float(d["score_generic"]), 5), "sd": round(float(d["score_sd"]), 4),
               "c": c, "lvl": lvl(d["score"]), "ne": int(d["n_external_recipients"])}
        if insiders is not None:
            rec["ins"] = int(d["user"] in insiders)
        uw.append(rec)
    key = {(r["u"], r["d"]): r["i"] for r in uw}
    by_week_user = {(r["cw"], r["u"]): r["i"] for r in uw}

    e = edges.copy()
    e["d"] = pd.to_datetime(e["week_start"]).dt.strftime("%Y-%m-%d")
    e["uw"] = [key.get((u, d)) for u, d in zip(e["user"], e["d"])]
    e = e[e["uw"].notna()].astype({"uw": int})
    cw_of = {i: r["cw"] for i, r in enumerate(uw)}
    conn_idx, conn = {}, []
    out = []
    for r in e.itertuples(index=False):
        t = EDGE_TYPE[r.edge_type]
        ou = -1 if pd.isna(r.other_users) else int(r.other_users)
        if t == T_MAIL:  # recipient is also a flagged user-week in the same calendar week -> direct user-user edge
            j = by_week_user.get((cw_of[r.uw], r.other_id))
            if j is not None and j != r.uw:
                out.append([r.uw, j, T_MAIL_USER, round(float(r.weight), 2), -1]); continue
        k = (r.other_kind, r.other_id)
        if k not in conn_idx:
            conn_idx[k] = len(conn); conn.append({"k": KIND_CODE[r.other_kind], "n": r.other_id, "cat": r.category or ""})
        out.append([r.uw, conn_idx[k], t, round(float(r.weight), 2), ou])
    cal = w.groupby("cal_week")["week_start"].min().sort_index()
    cut = budget_cutoffs(scores)
    meta = {"groups": [GROUP_COLS[g] for g in gcols], "budgets": BUDGETS, "cutoffs": [round(cut[q], 6) for q in BUDGETS],
            "calendar": [{"cw": int(c), "d": pd.Timestamp(d).strftime("%Y-%m-%d")} for c, d in cal.items()], "n_scored_weeks": int(len(scores))}
    if insiders is not None:
        meta["evaluation"] = True
    camp, cstats = build_campaign(scores, weeks, edges, insiders, seed)
    payload = {"meta": meta, "uw": uw, "conn": conn, "edges": out, "campaign": camp}
    stats.update({"user_weeks_drawn": len(uw), "connectors": len(conn), "edges": len(out), **cstats})
    return payload, stats


def _cluster_layout(G, clusters, n, seed):
    """Deterministic positions: community centres on a circle, a spring layout inside each community (so communities are visible
    even when modularity is low); users outside any community of at least MIN_CLUSTER sit in a small disc at the centre."""
    xy, placed = {}, set()
    R = 0.62 * LAYOUT_SCALE
    for k, members in enumerate(clusters):
        members = sorted(members)
        ang = 2 * np.pi * k / max(len(clusters), 1)
        cx, cy = R * np.cos(ang), R * np.sin(ang)
        pos = nx.spring_layout(G.subgraph(members), weight="weight", seed=seed, iterations=300) if len(members) > 1 else {members[0]: np.zeros(2)}
        ext = max(float(np.abs(np.array(list(pos.values()))).max()), 1e-9)
        rad = 62 * np.sqrt(len(members))
        for m in members:
            xy[m] = (cx + float(pos[m][0]) / ext * rad, cy + float(pos[m][1]) / ext * rad); placed.add(m)
    for j, m in enumerate(sorted(set(range(n)) - placed)):
        a = j * 2.399963
        xy[m] = (40 * np.sqrt(j + 1) * np.cos(a), 40 * np.sqrt(j + 1) * np.sin(a))
    return xy


def build_campaign(scores, weeks, edges, insiders=None, seed=SEED):
    """Flagged users linked through shared connectors, with Louvain communities (seed fixed; modularity reported)."""
    lvl = risk_levels(scores)
    pct = scores.assign(p=scores["score"].rank(pct=True) * 100)[["user", "week_start", "p"]]
    w = weeks.merge(pct, on=["user", "week_start"], how="left")
    users = sorted(w["user"].unique())
    uidx = {u: i for i, u in enumerate(users)}
    g = w.sort_values("score", ascending=False).groupby("user")
    best = g.head(1).set_index("user")
    nweeks = w.groupby("user").size()
    e = edges.copy()
    e = e[e["user"].isin(uidx)]
    flagged = set(users)
    mail_users = e[(e["edge_type"] == "EMAILED") & e["other_id"].isin(flagged) & (e["other_id"] != e["user"])]
    conn_e = e[~e.index.isin(mail_users.index)]
    agg = conn_e.groupby(["user", "other_kind", "other_id", "category"], dropna=False).agg(weight=("weight", "sum"), wk=("week_start", "nunique")).reset_index()
    deg = agg.groupby(["other_kind", "other_id"])["user"].nunique().rename("deg").reset_index()
    tw = agg.groupby(["other_kind", "other_id"])["weight"].sum().rename("tw").reset_index()
    cn = deg.merge(tw, on=["other_kind", "other_id"]).merge(agg[["other_kind", "other_id", "category"]].drop_duplicates(["other_kind", "other_id"]), on=["other_kind", "other_id"])
    cn = cn[(cn["deg"] >= 2) & (cn["deg"] <= CAMPAIGN_MAX_SHARE * len(users))]
    keep = []
    for kind, cap in [("Domain", CAMPAIGN_CAPS["domain"]), ("PC", CAMPAIGN_CAPS["pc"]), ("User", CAMPAIGN_CAPS["contact"])]:
        c = cn[cn["other_kind"] == kind]
        named = c[c["category"].fillna("") != ""]
        rest = c[c["category"].fillna("") == ""].sort_values(["deg", "tw", "other_id"], ascending=[False, False, True]).head(max(cap - len(named), 0))
        keep.append(pd.concat([named, rest]))
    cn = pd.concat(keep).reset_index(drop=True)
    cidx = {(r.other_kind, r.other_id): i for i, r in enumerate(cn.itertuples())}
    cedges = []
    for r in agg.itertuples(index=False):
        j = cidx.get((r.other_kind, r.other_id))
        if j is not None:
            cedges.append([uidx[r.user], j, EDGE_TYPE[{"PC": "LOGGED_ON", "Domain": "VISITED", "User": "EMAILED"}[r.other_kind]], round(float(r.weight), 1), int(r.wk)])
    mu = mail_users.groupby(["user", "other_id"]).agg(weight=("weight", "sum"), wk=("week_start", "nunique")).reset_index()
    for r in mu.itertuples(index=False):
        cedges.append([uidx[r.user], uidx[r.other_id], T_MAIL_USER, round(float(r.weight), 1), int(r.wk)])
    G = nx.Graph(); G.add_nodes_from(range(len(users)))
    members = defaultdict(set)
    for ue, j, t, wgt, nwk in cedges:
        if t != T_MAIL_USER:
            members[j].add(ue)
    for j, us in members.items():
        wt = 1.0 / np.log(1 + cn.loc[j, "deg"])
        for a, b in itertools.combinations(sorted(us), 2):
            G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0.0})["weight"] + wt)
    for ue, j, t, wgt, nwk in cedges:
        if t == T_MAIL_USER:
            G.add_edge(ue, j, weight=G.get_edge_data(ue, j, {"weight": 0.0})["weight"] + float(np.log1p(wgt)))
    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=1.0), key=lambda s: (-len(s), min(s)))
    cl = {}
    big = [c for c in comms if len(c) >= MIN_CLUSTER]
    for k, c in enumerate(big):
        for node in c:
            cl[node] = k
    mod = float(nx.community.modularity(G, comms, weight="weight")) if G.number_of_edges() else 0.0
    xy = _cluster_layout(G, big, len(users), seed)
    unodes = []
    for u in users:
        b = best.loc[u]
        rec = {"u": u, "x": round(xy[uidx[u]][0], 1), "y": round(xy[uidx[u]][1], 1), "wk": int(nweeks[u]), "s": round(float(b["score"]), 5), "p": round(float(b["p"]), 3), "z": round(float(_logit(b["score"])), 2),
               "lvl": lvl(b["score"]), "bd": pd.Timestamp(b["week_start"]).strftime("%Y-%m-%d"), "cl": cl.get(uidx[u], -1)}
        if insiders is not None:
            rec["ins"] = int(u in insiders)
        unodes.append(rec)
    wsum, xsum, ysum = defaultdict(float), defaultdict(float), defaultdict(float)
    for ue, j, t, wgt, nwk in cedges:
        if t != T_MAIL_USER:
            wt = float(np.log1p(wgt)) + 0.1
            wsum[j] += wt; xsum[j] += wt * xy[ue][0]; ysum[j] += wt * xy[ue][1]
    conns = []
    for j, r in enumerate(cn.itertuples()):
        a = j * 2.399963   # deterministic jitter so connectors with the same centre do not overlap
        conns.append({"k": KIND_CODE[r.other_kind], "n": r.other_id, "cat": r.category if isinstance(r.category, str) else "", "deg": int(r.deg), "tw": round(float(r.tw), 1),
                      "x": round(xsum[j] / wsum[j] + 22 * np.cos(a), 1), "y": round(ysum[j] / wsum[j] + 22 * np.sin(a), 1)})
    camp = {"top_connectors": CAMPAIGN_TOP_CONNECTORS, "users": unodes, "conn": conns, "edges": cedges, "clusters": [len(c) for c in big], "modularity": round(mod, 3), "louvain_seed": seed}
    stats = {"campaign_users": len(users), "campaign_connectors": len(conns), "campaign_edges": len(cedges), "campaign_clusters(size>=3)": [len(c) for c in big],
             "campaign_modularity": round(mod, 3), "campaign_unclustered_users": int(sum(1 for u in range(len(users)) if u not in cl))}
    return camp, stats
