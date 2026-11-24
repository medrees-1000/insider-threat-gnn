"""Web (http.csv) and email logs -> per user-week features.

Causality rule (same as weekly_features): the value for week w uses events from weeks <= w;
"first seen" and "rare" are defined from weeks strictly before w (rarity) or up to w (first seen).
http.csv is huge, so it is streamed once and reduced to (user, week_start, domain) counts.
"""
import io
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from itgnn.windows import DATE_FORMAT, _week_start

RARE_MAX_USERS = 5  # a domain is "rare" if fewer than this many users visited it before week w (fixed, not tuned)
_HOST_RE = r"^(?:[a-z][a-z0-9+.\-]*://)?(?:[^/?#@]*@)?(?P<host>[^/?#:]+)"


def parse_domain(urls):
    """Lowercased hostname with a leading 'www.' stripped; null where no host can be parsed.

    urls: pyarrow string array/chunked array. Returns a pyarrow string array.
    """
    low = pc.utf8_lower(urls)
    host = pc.struct_field(pc.extract_regex(low, _HOST_RE), "host")
    return pc.replace_substring_regex(host, r"^www\.", "")


def match_suffix(domains, entries):
    """Boolean array: domain equals an entry or ends with '.'+entry (exact-domain / suffix match)."""
    entries = [e.lower() for e in entries]
    uniq = pd.Series(domains.unique())
    ok = uniq.map(lambda d: isinstance(d, str) and any(d == e or d.endswith("." + e) for e in entries))
    return domains.map(dict(zip(uniq, ok))).fillna(False).astype(bool)


def aggregate_http(path, out_dir, block_mb=64, max_bytes=None, flush_every=8):
    """Stream http.csv once -> parquet parts of (user, week_start, domain, n, n_ah) in out_dir.

    Reads only date, user, url (content is never kept) with newlines_in_values=True. Each batch is
    reduced by groupby; every `flush_every` batches the partial sums are spilled to a part file, so
    RAM stays bounded. Returns a stats dict (rows, url_fail, date range, bytes read, seconds,
    hashes of ids are NOT kept). Stops after ~max_bytes read (timing probe) if given.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = dict(pcs=set(), rows=0, url_fail=0, date_min=None, date_max=None, seconds=0.0, bytes=0, id_hashes=[])
    t0 = time.time()
    pending, part = [], 0

    def flush():
        nonlocal pending, part
        if not pending:
            return
        df = pd.concat(pending).groupby(["user", "week_start", "domain"], as_index=False, observed=True).sum()
        df.to_parquet(out / f"part-{part:05d}.parquet", index=False)
        pending, part = [], part + 1

    with open(path, "rb") as f:
        src = f
        if max_bytes is not None:  # exact prefix (f.tell() would count the reader's read-ahead), cut at a line end
            buf = f.read(max_bytes)
            buf = buf[: buf.rfind(b"\n") + 1]
            src, stats["bytes"] = io.BytesIO(buf), len(buf)
        reader = pacsv.open_csv(
            src,
            read_options=pacsv.ReadOptions(block_size=block_mb << 20),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=["id", "date", "user", "pc", "url"],
                column_types={c: pa.string() for c in ["id", "date", "user", "pc", "url"]},
            ),
        )
        for batch in reader:
            dom = parse_domain(batch.column("url"))
            stats["rows"] += batch.num_rows
            stats["url_fail"] += dom.null_count
            ids = pd.Series(batch.column("id").to_numpy(zero_copy_only=False))
            stats["pcs"].update(pc.unique(batch.column("pc")).to_pylist())
            stats["id_hashes"].append(pd.util.hash_array(ids.to_numpy()))
            d = pd.DataFrame(
                {
                    "user": batch.column("user").to_numpy(zero_copy_only=False),
                    "date": pd.to_datetime(batch.column("date").to_numpy(zero_copy_only=False), format=DATE_FORMAT),  # strict
                    "domain": dom.to_numpy(zero_copy_only=False),
                }
            )
            lo, hi = d["date"].min(), d["date"].max()
            stats["date_min"] = lo if stats["date_min"] is None else min(stats["date_min"], lo)
            stats["date_max"] = hi if stats["date_max"] is None else max(stats["date_max"], hi)
            d = d.dropna(subset=["domain"])
            d["n"] = 1
            d["n_ah"] = (~d["date"].dt.hour.between(7, 18)).astype(int)  # same after-hours rule as logon
            d["week_start"] = _week_start(d["date"])
            pending.append(d.groupby(["user", "week_start", "domain"], as_index=False)[["n", "n_ah"]].sum())
            if len(pending) >= flush_every:
                flush()
            if max_bytes is None:
                stats["bytes"] = f.tell()  # approximate: includes read-ahead
    flush()
    stats["seconds"] = time.time() - t0
    h = np.concatenate(stats.pop("id_hashes")) if stats["id_hashes"] else np.array([], dtype="uint64")
    stats["duplicate_ids"] = int(len(h) - len(np.unique(h)))  # 64-bit hash: collisions negligible at this size
    return stats


def combine_parts(parts_dir):
    """Sum the spilled parts into one (user, week_start, domain, n, n_ah) frame."""
    t = pa.concat_tables([pq.read_table(p) for p in sorted(Path(parts_dir).glob("part-*.parquet"))])
    t = t.group_by(["user", "week_start", "domain"]).aggregate([("n", "sum"), ("n_ah", "sum")])
    return t.rename_columns(["user", "week_start", "domain", "n", "n_ah"]).to_pandas()


WEB_COLS = ["wb_n", "wb_n_after_hours", "wb_n_domains", "wb_n_new_domains", "wb_share_rare"]


def web_features(agg, categories):
    """Per (user, week_start) web features from the user-week-domain aggregate.

    wb_n_new_domains: domains this user never visited in an earlier week (first seen in week w).
    wb_share_rare: share of visits to domains that fewer than RARE_MAX_USERS users had visited
    by the end of week w-1 (past-only; a domain nobody visited yet counts as rare).
    categories: {name: [domain, ...]} -> one wb_n_<name> count column each (exact/suffix match).
    """
    a = agg.copy()
    first = a.groupby(["user", "domain"])["week_start"].transform("min")
    a["is_new"] = a["week_start"] == first
    # distinct users who had visited each domain strictly before week w
    new_users = (
        a[a["is_new"]].groupby(["domain", "week_start"]).size().rename("k").reset_index().sort_values("week_start")
    )
    new_users["prior_cum"] = new_users.groupby("domain")["k"].cumsum()
    q = a[["domain", "week_start"]].drop_duplicates().sort_values("week_start")
    q = pd.merge_asof(q, new_users[["domain", "week_start", "prior_cum"]], on="week_start", by="domain", allow_exact_matches=False)
    a = a.merge(q, on=["domain", "week_start"], how="left")
    a["rare_n"] = np.where(a["prior_cum"].fillna(0) < RARE_MAX_USERS, a["n"], 0)
    a["new_dom"] = a["is_new"].astype(int)
    for name, entries in categories.items():
        a[f"wb_n_{name}"] = np.where(match_suffix(a["domain"], entries), a["n"], 0)
    cat_cols = [f"wb_n_{k}" for k in categories]
    g = a.groupby(["user", "week_start"])
    out = pd.DataFrame(
        {
            "wb_n": g["n"].sum(),
            "wb_n_after_hours": g["n_ah"].sum(),
            "wb_n_domains": g.size(),
            "wb_n_new_domains": g["new_dom"].sum(),
            "wb_share_rare": g["rare_n"].sum() / g["n"].sum(),
        }
    )
    for c in cat_cols:
        out[c] = g[c].sum()
    return out.reset_index()


def add_history(weekly, cols, n_hist=4):
    """hist_mean_<c> (mean of previous n_hist active rows) and dev_<c> for each c in cols.

    Same recipe as weekly_features.add_history_features; `weekly` must be sorted by (user, week_start).
    """
    w = weekly.copy()
    g = w.groupby("user")
    for c in cols:
        w[f"hist_mean_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(n_hist, min_periods=n_hist).mean())
        w[f"dev_{c}"] = w[c] - w[f"hist_mean_{c}"]
    return w


def attach_features(weekly, feats, cols):
    """Left-join `feats` (user, week_start, cols...) onto the weekly rows; missing -> 0.

    Returns (frame, n_dropped_events_rows): feature rows with no weekly row (user-weeks without a Logon).
    """
    keys = weekly[["user", "week_start"]].merge(feats[["user", "week_start"]], how="right", indicator=True)
    dropped = int((keys["_merge"] == "right_only").sum())
    out = weekly.merge(feats[["user", "week_start"] + cols], on=["user", "week_start"], how="left")
    out[cols] = out[cols].fillna(0)
    return out, dropped


# Approved by the owner (see docs/PROJECT_LOG.md 6b) before any label was looked at. Exact domain or '.'-suffix match.
COMPETITOR = ["lockheedmartin.com", "northropgrumman.com", "raytheon.com", "boeing.com", "harris.com"]
WEB_CATEGORIES = {
    "leak": ["wikileaks.org"],
    "job": ["careerbuilder.com", "indeed.com", "monster.com", "simplyhired.com", "job-hunt.org", "jobhuntersbible.com", "taleo.net"],
    "cloud": ["dropbox.com", "mediafire.com", "yousendit.com", "4shared.com", "megaupload.com", "filesonic.com", "fileserve.com"],
    "download": ["refog.com", "dailykeylogger.com", "actualkeylogger.com", "keylogpc.com", "spectorsoft.com",
                 "webwatchernow.com", "best-spy-soft.com", "softactivity.com"],
    "competitor": COMPETITOR,
}
ORG_DOMAIN = "dtaa.com"
EMAIL_GENERIC = ["em_n", "em_n_after_hours", "em_n_recip", "em_n_ext_recip", "em_max_recip", "em_n_bcc", "em_n_attach",
                 "em_size_sum", "em_n_new_recip", "em_n_from_external"]
EMAIL_NAMED = ["em_n_competitor_recip"]
WEB_NAMED = [f"wb_n_{k}" for k in WEB_CATEGORIES]
NEW_BASE = EMAIL_GENERIC + EMAIL_NAMED + WEB_COLS + WEB_NAMED
FIRST_SEEN = ["em_n_new_recip", "wb_n_new_domains", "wb_share_rare"]


def email_features(df, org=ORG_DOMAIN, competitor=COMPETITOR):
    """Per (user, week_start) email features from the slim email frame (to/cc/bcc are ';'-separated lists).

    em_n_new_recip: distinct recipient addresses the user never mailed in an earlier week (first seen in week w).
    em_n_from_external: mails whose sender domain is not `org`. em_max_recip: most recipients in one mail.
    """
    m = df[["user", "date", "to", "cc", "bcc", "from", "size", "attachments"]].reset_index(drop=True)
    m["week_start"] = _week_start(m["date"])
    m["after_hours"] = ~m["date"].dt.hour.between(7, 18)
    m["from_ext"] = m["from"].str.rsplit("@", n=1).str[-1].str.lower() != org
    m["has_att"] = m["attachments"].astype(str) != "0"
    m["has_bcc"] = m["bcc"].fillna("") != ""
    m["mail"] = np.arange(len(m))
    r = (m["to"].fillna("") + ";" + m["cc"].fillna("") + ";" + m["bcc"].fillna("")).str.lower().str.split(";")
    long = pd.DataFrame({"user": m["user"], "week_start": m["week_start"], "mail": m["mail"], "recip": r}).explode("recip")
    long = long[long["recip"] != ""].reset_index(drop=True)
    dom = long["recip"].str.rsplit("@", n=1).str[-1]
    long["ext"] = dom != org
    long["comp"] = match_suffix(dom, competitor).to_numpy()
    k = ["user", "week_start"]
    gm, gl = m.groupby(k), long.groupby(k)
    out = pd.DataFrame(
        {
            "em_n": gm.size(),
            "em_n_after_hours": gm["after_hours"].sum(),
            "em_n_bcc": gm["has_bcc"].sum(),
            "em_n_attach": gm["has_att"].sum(),
            "em_size_sum": gm["size"].sum(),
            "em_n_from_external": gm["from_ext"].sum(),
            "em_n_recip": gl.size(),
            "em_n_ext_recip": gl["ext"].sum(),
            "em_n_competitor_recip": gl["comp"].sum(),
            "em_max_recip": long.groupby(k + ["mail"]).size().groupby(k).max(),
        }
    )
    uniq = long[k + ["recip"]].drop_duplicates()
    first = uniq.groupby(["user", "recip"])["week_start"].transform("min")
    out["em_n_new_recip"] = (uniq["week_start"] == first).groupby([uniq["user"], uniq["week_start"]]).sum()
    return out.fillna(0).astype(float).reset_index()


def add_comms_features(weekly, email, agg):
    """Add email + web features and their history/deviation columns to the weekly rows.

    weekly: rows sorted by (user, week_start) BEFORE the warm-up drop (history needs the early weeks).
    Returns (frame, dropped) with dropped = {"email": n, "web": n} user-weeks with events but no Logon row.
    """
    ef, wf = email_features(email), web_features(agg, WEB_CATEGORIES)
    out, d_e = attach_features(weekly, ef, EMAIL_GENERIC + EMAIL_NAMED)
    out, d_w = attach_features(out, wf, WEB_COLS + WEB_NAMED)
    return add_history(out, NEW_BASE), {"email": d_e, "web": d_w}


def _cols(base):
    return [x for c in base for x in (c, f"hist_mean_{c}", f"dev_{c}")]


def comms_feature_sets(w2):
    """The seven pre-registered feature sets (docs/PROJECT_LOG.md 6b), given the W2 column list."""
    e_all, h_all = _cols(EMAIL_GENERIC + EMAIL_NAMED), _cols(WEB_COLS + WEB_NAMED)
    named = set(_cols(EMAIL_NAMED + WEB_NAMED))
    seen = set(_cols(FIRST_SEEN))
    both = e_all + h_all
    return {
        "W2": list(w2),
        "W2+E": list(w2) + e_all,
        "W2+H": list(w2) + h_all,
        "W2+E+H": list(w2) + both,
        "W2+E+H generic": list(w2) + [c for c in both if c not in named],
        "W2+E+H no-first-seen": list(w2) + [c for c in both if c not in seen],
        "W2+calendar": list(w2) + ["cal_week"],
    }
