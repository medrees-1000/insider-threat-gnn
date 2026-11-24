import numpy as np
import pandas as pd
import pyarrow as pa

from itgnn.commslogs import (RARE_MAX_USERS, add_history, aggregate_http, attach_features, combine_parts,
                             match_suffix, parse_domain, web_features)

MONDAYS = pd.date_range("2010-01-04", periods=8, freq="7D")


def test_parse_domain_lowercase_www_and_failures():
    urls = pa.array(["HTTP://WWW.Example.COM/a/b?x=1", "https://user@news.site.org:8080/p", "www.foo.com/x", "", None])
    out = parse_domain(urls).to_pylist()
    assert out[:3] == ["example.com", "news.site.org", "foo.com"]
    assert out[3] is None and out[4] is None  # unparseable -> null (counted as failures)


def test_match_suffix_is_exact_or_dot_suffix_not_substring():
    d = pd.Series(["wikileaks.org", "cdn.wikileaks.org", "notwikileaks.org", "wikileaks.org.evil.com", "a.com"])
    assert match_suffix(d, ["wikileaks.org"]).tolist() == [True, True, False, False, False]


def _agg(rows):
    """rows: (user, week idx, domain, n)"""
    return pd.DataFrame(
        {"user": [r[0] for r in rows], "week_start": [MONDAYS[r[1]] for r in rows], "domain": [r[2] for r in rows],
         "n": [r[3] for r in rows], "n_ah": [0] * len(rows)}
    )


def test_web_features_values_and_job_category():
    agg = _agg([("u1", 0, "a.com", 2), ("u1", 1, "a.com", 1), ("u1", 1, "jobs.com", 3), ("u2", 1, "a.com", 1)])
    f = web_features(agg, {"job": ["jobs.com"]}).set_index(["user", "week_start"])
    r = f.loc[("u1", MONDAYS[1])]
    assert (r["wb_n"], r["wb_n_domains"], r["wb_n_new_domains"], r["wb_n_job"]) == (4, 2, 1, 3)  # only jobs.com is new in week 1
    assert r["wb_share_rare"] == 1.0  # all domains have < RARE_MAX_USERS prior visitors
    assert f.loc[("u1", MONDAYS[0]), "wb_n_new_domains"] == 1


def test_web_rarity_uses_only_earlier_weeks():
    # a.com is visited by RARE_MAX_USERS users in week 0 -> common in week 1, rare in week 0 for everyone
    users = [f"u{i}" for i in range(RARE_MAX_USERS)]
    base = [(u, 0, "a.com", 1) for u in users] + [("late", 1, "a.com", 1)]
    f = web_features(_agg(base), {}).set_index(["user", "week_start"])
    assert f.loc[("u0", MONDAYS[0]), "wb_share_rare"] == 1.0
    assert f.loc[("late", MONDAYS[1]), "wb_share_rare"] == 0.0
    # causality: adding a week-2 event must not change any week <= 1 value
    f2 = web_features(_agg(base + [("x", 2, "a.com", 5), ("late", 2, "b.com", 1)]), {}).set_index(["user", "week_start"])
    pd.testing.assert_frame_equal(f2.loc[f.index], f)


def test_aggregate_http_streams_newlines_and_counts_failures(tmp_path):
    csv = tmp_path / "http.csv"
    csv.write_text(
        'id,date,user,pc,url,content\n'
        '{1},01/05/2010 08:00:00,u1,PC-1,http://www.A.com/x,"text with\nnewline"\n'
        '{2},01/05/2010 23:00:00,u1,PC-1,http://a.com/y,words\n'
        '{3},01/12/2010 09:00:00,u2,PC-2,,words\n'
    )
    stats = aggregate_http(csv, tmp_path / "parts", block_mb=1, flush_every=1)
    assert stats["rows"] == 3 and stats["url_fail"] == 1 and stats["duplicate_ids"] == 0
    agg = combine_parts(tmp_path / "parts").set_index(["user", "week_start", "domain"])
    assert agg.loc[("u1", MONDAYS[0], "a.com")].tolist() == [2, 1]  # www. stripped, one after-hours visit


def test_add_history_and_attach_features():
    w = pd.DataFrame({"user": ["u"] * 6, "week_start": MONDAYS[:6], "c": [1, 2, 3, 4, 5, 10]})
    h = add_history(w, ["c"])
    assert h["hist_mean_c"].isna().sum() == 4 and h.loc[5, "hist_mean_c"] == 3.5 and h.loc[5, "dev_c"] == 6.5
    feats = pd.DataFrame({"user": ["u", "u"], "week_start": [MONDAYS[0], MONDAYS[7]], "x": [7, 9]})
    out, dropped = attach_features(w, feats, ["x"])
    assert dropped == 1 and out["x"].tolist() == [7, 0, 0, 0, 0, 0]


def test_email_features_values_and_causality():
    from itgnn.commslogs import email_features
    T = pd.Timestamp
    def mails(rows):  # (user, date, to, cc, bcc, from, size, attachments)
        return pd.DataFrame(rows, columns=["user", "date", "to", "cc", "bcc", "from", "size", "attachments"])
    rows = [
        ("u", T("2010-01-05 08:00"), "a@dtaa.com;x@boeing.com", "", "", "u@dtaa.com", 100, "0"),
        ("u", T("2010-01-05 22:00"), "A@dtaa.com", "b@gmail.com", "c@gmail.com", "me@yahoo.com", 50, "2"),
        ("u", T("2010-01-12 09:00"), "a@dtaa.com;z@dtaa.com", "", "", "u@dtaa.com", 10, "0"),
    ]
    f = email_features(mails(rows)).set_index(["user", "week_start"])
    w0, w1 = f.loc[("u", MONDAYS[0])], f.loc[("u", MONDAYS[1])]
    assert (w0["em_n"], w0["em_n_after_hours"], w0["em_n_recip"], w0["em_max_recip"]) == (2, 1, 5, 3)
    assert (w0["em_n_ext_recip"], w0["em_n_competitor_recip"], w0["em_n_bcc"], w0["em_n_attach"]) == (3, 1, 1, 1)
    assert (w0["em_n_from_external"], w0["em_size_sum"], w0["em_n_new_recip"]) == (1, 150, 4)  # a@ (case-folded), x@, b@, c@
    assert w1["em_n_new_recip"] == 1  # only z@ is new in week 1; a@ was mailed before
    f2 = email_features(mails(rows + [("u", T("2010-01-19 09:00"), "q@dtaa.com", "", "", "u@dtaa.com", 1, "0")])).set_index(["user", "week_start"])
    pd.testing.assert_frame_equal(f2.loc[f.index], f)  # a later week must not change earlier weeks


def test_comms_feature_sets_membership():
    from itgnn.commslogs import comms_feature_sets
    s = comms_feature_sets(["a", "b"])
    gen, full = set(s["W2+E+H generic"]), set(s["W2+E+H"])
    assert not {"wb_n_leak", "dev_wb_n_job", "em_n_competitor_recip", "hist_mean_wb_n_competitor"} & gen
    assert {"wb_n_leak", "em_n_competitor_recip"} <= full
    nfs = set(s["W2+E+H no-first-seen"])
    assert not {"em_n_new_recip", "dev_wb_share_rare", "hist_mean_wb_n_new_domains"} & nfs and "wb_n_leak" in nfs
    assert all(c in full for c in nfs - {"a", "b"}) and s["W2+calendar"] == ["a", "b", "cal_week"]


def test_audit_groups_cover_every_column_and_history_group():
    from itgnn.comms_audit import audit_groups
    from itgnn.commslogs import comms_feature_sets
    from itgnn.weekly_features import FEATURE_SETS
    cols = comms_feature_sets(FEATURE_SETS["W2"])["W2+E+H"]
    g = audit_groups(cols)  # asserts full coverage
    assert len(g) == 9 and set(g["history/deviation"]) == {c for c in cols if c.startswith(("hist_mean_", "dev_"))}
    assert "wb_n_leak" in g["named-site"] and "dev_wb_n_leak" in g["named-site"] and "wb_n_leak" not in g["web volume"]
    assert "n_new_pcs" in g["W2 base"] and "em_n_new_recip" in g["first-seen"]


def test_grouped_permutation_importance_finds_signal_and_not_noise():
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from itgnn.comms_audit import grouped_permutation_importance
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"sig": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = (X["sig"] + 0.3 * rng.normal(size=n) > 1.2).astype(int)
    m = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=0).fit(X[:300], y[:300])
    imp = grouped_permutation_importance(m, X[300:], y[300:], {"s": ["sig"], "n": ["noise"]}, n_shuffles=3).groupby("group")["drop"].mean()  # held-out rows
    assert imp["s"] > 0.2 and abs(imp["n"]) < 0.1


def test_truncation_check_passes_and_positive_control_fails():
    import numpy as np
    from itgnn.comms_audit import truncation_check
    from itgnn.commslogs import NEW_BASE, add_comms_features
    T = pd.Timestamp
    wk = lambda k: MONDAYS[k]
    lg = pd.DataFrame([(u, wk(k) + pd.Timedelta(days=d, hours=9), f"PC{(k + d) % 3}", "Logon") for u in ("a", "b") for k in range(7) for d in (0, 3)],
                      columns=["user", "date", "pc", "activity"])
    empty = pd.DataFrame({"user": pd.Series(dtype=object), "date": pd.Series(dtype="datetime64[ns]")})
    em = pd.DataFrame([(u, wk(k) + pd.Timedelta(days=d, hours=10), f"r{k}@x.com;q@dtaa.com", "", "", "u@dtaa.com", 10 + k, "0")
                       for u in ("a", "b") for k in range(7) for d in (1, 4)],
                      columns=["user", "date", "to", "cc", "bcc", "from", "size", "attachments"])
    ag = pd.DataFrame([(u, wk(k), dom, 1 + k, 0) for u in ("a", "b") for k in range(7) for dom in ("d1.com", f"d{k}.com")],
                      columns=["user", "week_start", "domain", "n", "n_ah"])
    from itgnn.windows import make_weekly_features
    full = add_history_features_for_test(lg, empty)
    full, _ = add_comms_features(full, em, ag)
    cols = NEW_BASE + [f"hist_mean_{c}" for c in NEW_BASE] + ["n_logon", "n_new_pcs"]
    sample = full[full["week_start"] == wk(6)][["user", "week_start"]]
    ok = truncation_check(sample, lg, empty, empty, em, ag, full, cols)
    assert (ok["max_abs_diff"] == 0).all()
    bad = truncation_check(sample, lg, empty, empty, em, ag, full, cols, cut_days=2)  # cuts days 2-6 of the week
    assert (bad["max_abs_diff"] > 0).any()


def add_history_features_for_test(lg, empty):
    from itgnn.weekly_features import add_history_features
    from itgnn.windows import make_weekly_features
    return add_history_features(make_weekly_features(lg, empty, empty, set(lg["user"])), lg)


def test_position_and_phase_helpers():
    from itgnn.comms_audit import attack_week_position, flag_top_q, window_phase
    d = pd.date_range("2010-01-04", periods=8, freq="7D")
    oof = pd.DataFrame({"user": ["u"] * 8, "week_start": d, "fold": 0, "score": np.arange(8, dtype=float),
                        "label": [0, 0, 1, 1, 1, 0, 0, 0]}) if False else pd.DataFrame(
        {"user": ["u"] * 8, "week_start": d, "fold": 0, "score": [float(i) for i in range(8)], "label": [0, 0, 1, 1, 1, 0, 0, 0]})
    win = pd.DataFrame({"user": ["u"], "start": [d[2] + pd.Timedelta(days=1)], "end": [d[4] + pd.Timedelta(days=2)]})
    a = attack_week_position(oof, win)
    assert list(a["position"]) == ["first", "middle", "last"] and (a["win_days"] == 16).all()
    ph = window_phase(oof, win).set_index("week_start")["phase"]
    assert ph[d[1]] == "before" and ph[d[3]] == "inside" and ph[d[5]] == "after" and ph[d[7]] == "after"
    assert flag_top_q(oof, 25, 0).sum() == 2 and flag_top_q(oof, 25, 0)[[6, 7]].all()


def test_time_split_masks_and_bootstrap_and_reference():
    from itgnn.comms_eval import double_holdout_masks, same_rows_user_split_ap, time_split_masks, user_bootstrap_ap
    w = pd.DataFrame({"user": list("aabbcc"), "cal_week": [1, 5, 2, 6, 3, 9]})
    tr, te = time_split_masks(w, 4)
    assert tr.tolist() == [True, False, True, False, True, False] and te.tolist() == [not t for t in tr]
    assert set(w.loc[tr, "user"]) & set(w.loc[te, "user"])  # same users on both sides
    tr2, te2 = double_holdout_masks(w, {"train": ["a", "b"], "test": ["c"]}, 4)
    assert w.loc[tr2, "user"].tolist() == ["a", "b"] and w.loc[te2, "user"].tolist() == ["c"]
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(180)].astype(int)
    users = np.repeat(np.arange(40), 5)
    good, bad = y + rng.normal(0, 0.3, 200), rng.normal(size=200)
    b = user_bootstrap_ap(y, {"good": good, "bad": bad}, users, n_boot=30, seed=1)
    assert (b["good"] > b["bad"]).mean() > 0.9 and b.shape == (30, 2)
    b2 = user_bootstrap_ap(y, {"good": good, "bad": bad}, users, n_boot=30, seed=1)
    pd.testing.assert_frame_equal(b, b2)  # deterministic
    oof = pd.DataFrame({"model": "m", "repeat": [0] * 4 + [1] * 4, "cal_week": [1, 5, 6, 7] * 2, "label": [1, 1, 0, 0] * 2, "score": [0.1, 0.8, 0.9, 0.1] * 2})
    assert abs(same_rows_user_split_ap(oof, "m", 4) - 0.5) < 1e-9  # rows with cal_week 5,6,7: labels 1,0,0, scores .8,.9,.1 -> the positive is ranked 2nd -> AP 0.5
