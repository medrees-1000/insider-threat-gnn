"""Post-hoc audit helpers for the email/web probe (notebook 06). None of this feeds back into any feature."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from itgnn.commslogs import EMAIL_NAMED, FIRST_SEEN, WEB_NAMED, add_comms_features
from itgnn.weekly_features import add_history_features
from itgnn.windows import WEEK_FEATURE_COLS, make_weekly_features

HIST_PREFIXES = ("hist_mean_", "dev_")


def audit_groups(cols):
    """Feature groups for permutation importance / leave-one-group-out, as {name: columns}.

    Families include their own hist_mean_/dev_ columns; "history/deviation" is cross-cutting (every hist_mean_*
    and dev_* column, W2's included), so it overlaps the families. W2's five weekly counts and n_new_pcs sit in
    "W2 base" (current-week values only). Every column of `cols` must land in at least one group.
    """
    cols = list(cols)
    fam = lambda base: [x for c in base for x in (c, f"hist_mean_{c}", f"dev_{c}") if x in cols]
    groups = {
        "W2 base": [c for c in WEEK_FEATURE_COLS + ["n_new_pcs"] if c in cols],
        "email volume": fam(["em_n", "em_n_after_hours", "em_size_sum", "em_n_from_external"]),
        "email recipients/attachments": fam(["em_n_recip", "em_n_ext_recip", "em_max_recip", "em_n_bcc", "em_n_attach"]),
        "web volume": fam(["wb_n"]),
        "web domain counts": fam(["wb_n_domains"]),
        "web after-hours": fam(["wb_n_after_hours"]),
        "named-site": fam(WEB_NAMED + EMAIL_NAMED),
        "first-seen": fam(FIRST_SEEN),
        "history/deviation": [c for c in cols if c.startswith(HIST_PREFIXES)],
    }
    missing = set(cols) - {c for v in groups.values() for c in v}
    assert not missing, f"columns without a group: {sorted(missing)}"
    return groups


def grouped_permutation_importance(model, X, y, groups, n_shuffles=5, seed=0):
    """PR-AUC drop when one group's columns are permuted TOGETHER (same row permutation) on held-out data.

    Returns a frame with one row per (group, shuffle): `drop` = baseline PR-AUC - permuted PR-AUC.
    Caveat: correlated groups share information, so a drop understates how much a group matters.
    """
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for g, cols in groups.items():
        for k in range(n_shuffles):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            rows.append({"group": g, "shuffle": k, "drop": base - average_precision_score(y, model.predict_proba(Xp)[:, 1])})
    return pd.DataFrame(rows)


def draw_sample(weekly, n=200, n_attack=30, n_weeks=15, seed=0):
    """Random user-weeks for the truncation check: n rows, n_attack of them attack weeks, from few distinct weeks.

    The rows come from `n_weeks` randomly chosen calendar weeks (more are added until enough attack weeks exist)
    so that each recompute (one per distinct week) stays cheap.
    """
    rng = np.random.default_rng(seed)
    weeks = list(rng.permutation(weekly["week_start"].unique()))
    k = n_weeks
    while True:
        pool = weekly[weekly["week_start"].isin(weeks[:k])]
        if (pool["label"] == 1).sum() >= n_attack or k >= len(weeks):
            break
        k += 1
    att = pool[pool["label"] == 1].sample(min(n_attack, int((pool["label"] == 1).sum())), random_state=seed)
    nor = pool[pool["label"] == 0].sample(n - len(att), random_state=seed)
    return pd.concat([att, nor])[["user", "week_start", "label"]].reset_index(drop=True)


def truncation_check(sample, logon, device, file, email, agg, full, cols, cut_days=7):
    """Recompute features for each sampled user-week from logs truncated at the END of that week.

    full: the full-run frame (all rows, before the warm-up drop) holding `cols`. Returns a frame with one row per
    sampled user-week and the max |difference| over `cols`; 0 everywhere means no future information was used.
    Web rarity depends on every user, so the web aggregate keeps all users; email / logon are per-user and are
    restricted to the sampled users. cut_days < 7 truncates inside the week (a positive control: must differ).
    """
    ref = full.set_index(["user", "week_start"])
    out = []
    for wk, g in sample.groupby("week_start"):
        end = wk + pd.Timedelta(days=cut_days)
        us = set(g["user"])
        lg, dv, fl = (x[(x["date"] < end) & x["user"].isin(us)] for x in (logon, device, file))
        em = email[(email["date"] < end) & email["user"].isin(us)]
        ag = agg[agg["week_start"] <= wk]
        w = add_history_features(make_weekly_features(lg, dv, fl, us), lg)
        w, _ = add_comms_features(w, em, ag)
        got = w[w["week_start"] == wk].set_index("user")
        for u in g["user"]:
            if u not in got.index:
                out.append({"user": u, "week_start": wk, "max_abs_diff": np.inf, "worst_col": "row missing"})
                continue
            a, b = got.loc[u, cols].to_numpy(float), ref.loc[(u, wk), cols].to_numpy(float)
            d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
            out.append({"user": u, "week_start": wk, "max_abs_diff": float(np.nanmax(np.where(np.isnan(d), np.inf, d))),
                        "worst_col": cols[int(np.argmax(d))]})
    return pd.DataFrame(out)


def flag_top_q(oof, q, seed):
    """Boolean Series (aligned to oof.index): top ceil(q%) rows by score within each fold, ties broken at random."""
    rng = np.random.default_rng(seed)
    flagged = pd.Series(False, index=oof.index)
    for _, g in oof.groupby("fold"):
        order = np.lexsort((rng.random(len(g)), -g["score"].to_numpy()))
        flagged.loc[g.index[order[: int(np.ceil(q / 100 * len(g)))]]] = True
    return flagged


def attack_week_position(oof, windows):
    """For label==1 rows add `position` (only/first/middle/last among the user's attack weeks) and `win_days`."""
    a = oof[oof["label"] == 1].copy()
    a["rank"] = a.groupby("user")["week_start"].rank(method="first").astype(int) - 1
    a["n"] = a.groupby("user")["week_start"].transform("size")
    a["position"] = np.select([a["n"] == 1, a["rank"] == 0, a["rank"] == a["n"] - 1], ["only", "first", "last"], "middle")
    w = windows.set_index("user")
    a["win_days"] = (a["user"].map(w["end"]) - a["user"].map(w["start"])).dt.days + 1
    return a


def window_phase(oof, windows, span_days=28):
    """Tag rows of insiders: inside (label 1), before (the span before the window start), after (span after its end)."""
    w = windows.set_index("user")
    o = oof[oof["user"].isin(w.index)].copy()
    st, en = o["user"].map(w["start"]), o["user"].map(w["end"])
    span = pd.Timedelta(days=span_days)
    o["phase"] = np.select(
        [o["label"] == 1, (o["week_start"] < st) & (o["week_start"] >= st - span), (o["week_start"] > en) & (o["week_start"] <= en + span)],
        ["inside", "before", "after"], "other")
    return o
