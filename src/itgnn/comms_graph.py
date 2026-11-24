"""Graph v2: user-week, PC-week and domain-week nodes with SAME-WEEK edges only.

Edge types: user-PC (logon, weight 1), user-domain (top-K domains per user-week by visits, weight log1p(visits)),
user-user email (top-K recipients per sender-week by mails, weight log1p(mails); recipients are resolved through a
static address table, never through behaviour). Every edge is stored once (undirected) with its type; `adjacency`
builds the symmetric row-normalised weighted adjacency used by the `wsage` layer.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from itgnn.weekly_features import N_HIST, signed_log1p
from itgnn.windows import _week_start

PC, DOM, EMAIL = 0, 1, 2
DOM_LOG_USERS, DOM_LOG_VISITS = 7.0, 12.0  # fixed divisors (about log of 1,000 users / 160,000 visits), not fitted


@dataclass
class GraphV2:
    rows: pd.DataFrame  # one row per user node: user, week_start, label, has_history
    n_pc: int
    n_dom: int
    node_week: np.ndarray  # week_start of every node (users, then PC-weeks, then domain-weeks)
    eu: np.ndarray  # left endpoint (user node id) of every edge
    ev: np.ndarray  # right endpoint (PC-week node, domain-week node or user node)
    ew: np.ndarray  # weight
    etype: np.ndarray  # PC / DOM / EMAIL
    user_raw: np.ndarray  # transformed (log1p / signed log) user features, NOT scaled; no history -> 0
    user_cols: list
    pc_logn: np.ndarray
    pc_name: np.ndarray  # pc id per PC-week node
    dom_name: np.ndarray  # domain per domain-week node
    dom_feat: np.ndarray  # (n_dom, 3): log1p users / 7, log1p visits / 12, after-hours share (that week only)

    @property
    def n_user(self):
        return len(self.rows)

    @property
    def n_node(self):
        return self.n_user + self.n_pc + self.n_dom


def address_map_from_ldap(raw_dir):
    """Static identity table {address (lowercase): user_id} from the LDAP monthly files; addresses that appear
    with more than one user_id are left out and returned separately."""
    frames = [pd.read_csv(f, usecols=["user_id", "email"]) for f in sorted((Path(raw_dir) / "LDAP").glob("*.csv"))]
    m = pd.concat(frames).dropna().drop_duplicates()
    m["email"] = m["email"].str.lower()
    n = m.groupby("email")["user_id"].nunique()
    ok = m[m["email"].isin(n[n == 1].index)].drop_duplicates("email")
    return dict(zip(ok["email"], ok["user_id"])), sorted(n[n > 1].index)


def explode_recipients(email):
    """One row per (mail, recipient address): user, date, recip. A recipient listed twice in a mail counts once."""
    m = email[["user", "date", "to", "cc", "bcc"]].reset_index(drop=True)
    r = (m["to"].fillna("") + ";" + m["cc"].fillna("") + ";" + m["bcc"].fillna("")).str.lower().str.split(";")
    long = pd.DataFrame({"mail": np.arange(len(m)), "user": m["user"], "date": m["date"], "recip": r}).explode("recip")
    return long[long["recip"] != ""].drop_duplicates(["mail", "recip"]).reset_index(drop=True)


def address_coverage(long, addr_map, ambiguous, org="dtaa.com"):
    """Share of internal recipient addresses / entries that resolve to exactly one user (the owner's >= 90% rule)."""
    internal = long[long["recip"].str.endswith("@" + org)]
    amb = set(ambiguous)
    ok = internal["recip"].isin(addr_map.keys()) & ~internal["recip"].isin(amb)
    addrs = internal.drop_duplicates("recip")
    ok_a = addrs["recip"].isin(addr_map.keys()) & ~addrs["recip"].isin(amb)
    return {"internal entries": len(internal), "internal addresses": len(addrs), "entries mapped to exactly one user": float(ok.mean()),
            "addresses mapped to exactly one user": float(ok_a.mean()), "ambiguous addresses (>1 user)": len(amb),
            "internal addresses not in LDAP": int((~addrs["recip"].isin(addr_map.keys()) & ~addrs["recip"].isin(amb)).sum())}


def domains_to_drop(agg, categories, excluded, min_users=20):
    """Domains kept out of the edges: category lists, the owner's excluded domains, and domains with < min_users
    distinct users over the whole period (unlabeled counts only). Returns (set, report dict)."""
    from itgnn.commslogs import match_suffix
    dom = agg["domain"].drop_duplicates()
    cat = set(dom[match_suffix(dom, [d for v in categories.values() for d in v])])
    exc = set(excluded) & set(dom)
    nu = agg.groupby("domain")["user"].nunique()
    rare = set(nu[nu < min_users].index)
    drop = cat | exc | rare
    rows = int(agg["domain"].isin(drop).sum())
    return drop, {"domains total": len(dom), "category domains": len(cat), "excluded domains": len(exc), "< %d users" % min_users: len(rare),
                  "domains dropped (union)": len(drop), "domains kept": len(dom) - len(drop), "aggregate rows dropped": rows,
                  "aggregate rows kept": len(agg) - rows, "aggregate rows total": len(agg)}


def build_graph_v2(full, logon, agg, long_email, addr_map, user_cols, k_domain=5, k_email=5, drop_domains=frozenset()):
    """full: all user-week rows (warm-up included) with label and `user_cols`; long_email from `explode_recipients`
    (None = no email edges). Weeks are Monday-based `week_start` values; no edge ever joins two weeks."""
    full = full.sort_values(["user", "week_start"]).reset_index(drop=True)
    rows = full[["user", "week_start", "label"]].copy()
    rows["has_history"] = (full.groupby("user").cumcount() >= N_HIST).to_numpy()
    uid = pd.Series(np.arange(len(rows)), index=pd.MultiIndex.from_frame(rows[["user", "week_start"]]))
    n_user = len(rows)
    lookup = lambda u, w: uid.reindex(pd.MultiIndex.from_arrays([u, w])).to_numpy()

    lg = logon[(logon["activity"] == "Logon") & logon["user"].isin(set(full["user"]))]
    e = pd.DataFrame({"user": lg["user"].to_numpy(), "week_start": _week_start(lg["date"]).to_numpy(), "pc": lg["pc"].to_numpy()}).drop_duplicates()
    e["u"] = lookup(e["user"], e["week_start"])
    assert not np.isnan(e["u"]).any(), "logon week without a feature row"
    e["p"] = e.groupby(["week_start", "pc"]).ngroup()
    pc = e.groupby("p").agg(week_start=("week_start", "first"), pc=("pc", "first"), n=("user", "size"))
    n_pc = len(pc)
    eu, ev, ew, et = [e["u"].to_numpy(int)], [e["p"].to_numpy(int) + n_user], [np.ones(len(e))], [np.full(len(e), PC)]

    a = agg[~agg["domain"].isin(drop_domains)].copy()
    a["u"] = lookup(a["user"], a["week_start"])
    a = a[~np.isnan(a["u"])]
    a["u"] = a["u"].astype(int)
    a["d"] = a.groupby(["week_start", "domain"]).ngroup()
    dom_stat = a.groupby("d").agg(week_start=("week_start", "first"), domain=("domain", "first"), users=("u", "size"), visits=("n", "sum"), ah=("n_ah", "sum"))
    top = a.sort_values(["u", "n", "domain"], ascending=[True, False, True])
    top = top[top.groupby("u").cumcount() < k_domain]
    used = np.sort(top["d"].unique())
    dmap = pd.Series(np.arange(len(used)), index=used)
    ds = dom_stat.loc[used]
    dom_feat = np.c_[np.log1p(ds["users"]) / DOM_LOG_USERS, np.log1p(ds["visits"]) / DOM_LOG_VISITS, ds["ah"] / ds["visits"]]
    n_dom = len(used)
    eu.append(top["u"].to_numpy(int)); ev.append(dmap.loc[top["d"]].to_numpy(int) + n_user + n_pc)
    ew.append(np.log1p(top["n"].to_numpy(float))); et.append(np.full(len(top), DOM))

    if long_email is not None:
        m = long_email[long_email["recip"].isin(addr_map.keys())].copy()
        m["week_start"] = _week_start(m["date"])
        m["ruser"] = m["recip"].map(addr_map)
        m["u"] = lookup(m["user"], m["week_start"])
        m["v"] = lookup(m["ruser"], m["week_start"])
        m = m[~np.isnan(m["u"]) & ~np.isnan(m["v"])]
        m["u"], m["v"] = m["u"].astype(int), m["v"].astype(int)
        m = m[m["u"] != m["v"]]
        c = m.groupby(["u", "v"]).size().rename("n").reset_index()
        c = c.sort_values(["u", "n", "v"], ascending=[True, False, True])
        c = c[c.groupby("u").cumcount() < k_email]
        eu.append(c["u"].to_numpy(int)); ev.append(c["v"].to_numpy(int)); ew.append(np.log1p(c["n"].to_numpy(float))); et.append(np.full(len(c), EMAIL))

    raw = full[user_cols].fillna(0.0)
    dev = [c for c in user_cols if c.startswith("dev_")]
    nonneg = [c for c in user_cols if c not in dev]
    user_raw = pd.concat([np.log1p(raw[nonneg]), signed_log1p(raw[dev])], axis=1)[user_cols].to_numpy(float)
    assert np.isfinite(user_raw).all()
    node_week = np.r_[rows["week_start"].to_numpy(), pc["week_start"].to_numpy(), ds["week_start"].to_numpy()]
    g = GraphV2(rows, n_pc, n_dom, node_week, np.concatenate(eu), np.concatenate(ev), np.concatenate(ew), np.concatenate(et),
                user_raw, list(user_cols), np.log(pc["n"].to_numpy(float)), pc["pc"].to_numpy(), ds["domain"].to_numpy(), dom_feat)
    return g


def shuffle_edges(g, seed):
    """Degree-preserving shuffle: within each week and edge type, the right endpoint and the weight of the edges are
    permuted TOGETHER, so every node keeps its degree and the weight multiset per week is unchanged; weeks never mix.
    Email self-loops created by the shuffle are dropped. Returns (new graph, share of email edges dropped)."""
    rng = np.random.default_rng(seed)
    ev, ew = g.ev.copy(), g.ew.copy()
    wk = pd.factorize(g.node_week[g.eu])[0]
    for t in (PC, DOM, EMAIL):
        i = np.flatnonzero(g.etype == t)
        if len(i) == 0:
            continue
        a = i[np.argsort(wk[i], kind="stable")]
        b = i[np.lexsort((rng.random(len(i)), wk[i]))]  # same week blocks as `a`, random order inside each block
        ev[a], ew[a] = g.ev[b], g.ew[b]
    keep = ~((g.etype == EMAIL) & (g.eu == ev))
    dropped = float(1 - keep[g.etype == EMAIL].mean()) if (g.etype == EMAIL).any() else 0.0
    from dataclasses import replace
    return replace(g, eu=g.eu[keep], ev=ev[keep], ew=ew[keep], etype=g.etype[keep]), dropped


def adjacency(g):
    """Symmetric weighted adjacency, row-normalised (weighted mean over neighbours), as a torch sparse CSR tensor."""
    r, c, v = np.r_[g.eu, g.ev], np.r_[g.ev, g.eu], np.r_[g.ew, g.ew]
    A = sp.coo_matrix((v, (r, c)), shape=(g.n_node, g.n_node)).tocsr()
    s = np.asarray(A.sum(1)).ravel()
    A = sp.diags(1.0 / np.where(s > 0, s, 1.0)) @ A
    A = A.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)), torch.from_numpy(A.indices.astype(np.int64)),
                                   torch.from_numpy(A.data), size=A.shape)


def node_features_v2(g, fit_mask):
    """(n_node, n_user_cols + 7) tensor; the user scaler is fit on rows where fit_mask is True only.
    Columns: user features | has_history | is_pc, log(users on the PC) | is_domain, 3 domain-week statistics."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(g.user_raw[fit_mask])
    nu = len(g.user_cols)
    X = np.zeros((g.n_node, nu + 7), dtype=np.float32)
    X[: g.n_user, :nu] = sc.transform(g.user_raw)
    X[: g.n_user, nu] = g.rows["has_history"].to_numpy()
    p0, d0 = g.n_user, g.n_user + g.n_pc
    X[p0:d0, nu + 1] = 1.0
    X[p0:d0, nu + 2] = g.pc_logn
    X[d0:, nu + 3] = 1.0
    X[d0:, nu + 4 :] = g.dom_feat
    assert np.isfinite(X).all()
    return torch.from_numpy(X), sc


def edge_table(g):
    """Canonical edge list (type, user, week_start, other, weight) with names instead of node ids; sorted.

    `other` is the PC name, the domain name or the recipient user; the week is the user's week for every edge.
    Used to compare graphs built from different data (causality checks) independent of node numbering.
    """
    u = g.rows[["user", "week_start"]].to_numpy()
    n_u, n_p = g.n_user, g.n_pc
    other = np.empty(len(g.ev), dtype=object)
    pcm, dm, em = g.etype == PC, g.etype == DOM, g.etype == EMAIL
    other[pcm] = g.pc_name[g.ev[pcm] - n_u]
    other[dm] = g.dom_name[g.ev[dm] - n_u - n_p]
    other[em] = g.rows["user"].to_numpy()[g.ev[em]]
    t = pd.DataFrame({"type": g.etype, "user": u[g.eu, 0], "week_start": u[g.eu, 1], "other": other, "weight": np.round(g.ew, 9)})
    return t.sort_values(["type", "user", "week_start", "other", "weight"]).reset_index(drop=True)


EXCLUDED_DOMAINS = ["relytec.com", "wellresearchedreviews.com", "blackhatworld.com", "download.cnet.com", "custhelp.com"]  # out of every model (log 6b)
ADDR_MIN_SHARE = 0.90  # the owner's rule for keeping the email edges


def load_inputs(cfg):
    """Everything the graph needs, loaded once: data dict, windows, cached feature frame, aggregates, address map."""
    from itgnn.comms_eval import load_full
    from itgnn.commslogs import WEB_CATEGORIES, comms_feature_sets
    from itgnn.data import prepare
    from itgnn.weekly_features import FEATURE_SETS
    from itgnn.windows import load_attack_windows

    d = prepare(cfg)
    windows = load_attack_windows(cfg["raw_dir"])
    full = load_full(d, windows)
    agg = pd.read_parquet("data/processed/http_user_week_domain.parquet")
    long = explode_recipients(pd.read_parquet("data/processed/email.parquet"))
    addr, amb = address_map_from_ldap(cfg["raw_dir"])
    drop, drop_rep = domains_to_drop(agg, WEB_CATEGORIES, EXCLUDED_DOMAINS, 20)
    cov = address_coverage(long, addr, amb)
    cols = comms_feature_sets(FEATURE_SETS["W2"])["W2+E+H"]
    use_email = cov["entries mapped to exactly one user"] >= ADDR_MIN_SHARE and cov["addresses mapped to exactly one user"] >= ADDR_MIN_SHARE
    return dict(d=d, windows=windows, full=full, agg=agg, long=long, addr=addr, amb=amb, drop=drop, drop_rep=drop_rep, cov=cov,
                cols=cols, use_email=use_email)


def build_from_inputs(inp, k_domain=5, k_email=5):
    return build_graph_v2(inp["full"], inp["d"]["logon"], inp["agg"], inp["long"] if inp["use_email"] else None, inp["addr"],
                          inp["cols"], k_domain, k_email, frozenset(inp["drop"]))


def pilot_curves_v2(g, A, train_users, gcfg, conv, seed):
    """Same method as weekly_gnn.pilot_curves (inner user splits inside the training users), on graph v2."""
    from sklearn.model_selection import StratifiedShuffleSplit

    from itgnn.weekly_gnn import train_scores
    from itgnn.weekly_graph import fold_masks

    pcfg = gcfg["pilot"]
    users = np.array(sorted(train_users))
    ins = set(g.rows.loc[g.rows["label"] == 1, "user"])
    y_user = np.array([u in ins for u in users])
    y = g.rows["label"].to_numpy()
    out = []
    sss = StratifiedShuffleSplit(pcfg["n_inner_splits"], test_size=pcfg["inner_val_size"], random_state=seed)
    for i, (tr, va) in enumerate(sss.split(users, y_user)):
        fit, val = fold_masks(g, users[tr], users[va])
        x, _ = node_features_v2(g, fit)
        _, h = train_scores(x, A, y, np.flatnonzero(fit), pcfg["max_epochs"], conv, gcfg, seed, np.flatnonzero(val))
        out.append(h.assign(split=i, model=conv))
    return pd.concat(out, ignore_index=True)
