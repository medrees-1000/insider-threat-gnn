"""One user-PC graph per week, batched as a single disjoint graph, and its node features.

Nodes: one per user-week row (warm-up rows included, they are neighbours only) and one per
(PC, week) with at least one Logon. Edges connect a user-week to the PC-weeks it logged on to,
in both directions. No edge ever joins two different weeks. Labels are stored but only the loss
mask uses them.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from itgnn.weekly_features import DEV_COLS, HIST_COLS, N_HIST, signed_log1p
from itgnn.windows import WEEK_FEATURE_COLS, _week_start

USER_COLS = WEEK_FEATURE_COLS + HIST_COLS  # feature set W2
N_FEAT = len(USER_COLS) + 3  # + has_history, is_pc, log(users on this PC that week)


@dataclass
class WeeklyGraph:
    rows: pd.DataFrame  # one row per user node: user, week_start, label, has_history
    edge_index: torch.Tensor  # (2, 2E), user ids first then PC ids
    node_week: np.ndarray  # week_start of every node
    user_raw: np.ndarray  # (n_user, 16) transformed (log1p / signed log), NOT scaled; no history -> 0
    pc_name: np.ndarray  # pc id per PC node
    pc_logn: np.ndarray  # log(number of users on that PC that week)

    @property
    def n_user(self):
        return len(self.rows)

    @property
    def n_node(self):
        return self.n_user + len(self.pc_name)


def build_weekly_graph(full, logon):
    """full: one row per (user, week_start) with label and the W2 columns (history NaN in the warm-up)."""
    full = full.sort_values(["user", "week_start"]).reset_index(drop=True)
    assert not full.duplicated(["user", "week_start"]).any()
    rows = full[["user", "week_start", "label"]].copy()
    rows["has_history"] = (full.groupby("user").cumcount() >= N_HIST).to_numpy()

    lg = logon[(logon["activity"] == "Logon") & logon["user"].isin(set(full["user"]))]
    e = pd.DataFrame({"user": lg["user"].to_numpy(), "week_start": _week_start(lg["date"]).to_numpy(),
                      "pc": lg["pc"].to_numpy()}).drop_duplicates()
    uid = pd.Series(np.arange(len(rows)), index=pd.MultiIndex.from_frame(rows[["user", "week_start"]]))
    e["u"] = uid.reindex(pd.MultiIndex.from_frame(e[["user", "week_start"]])).to_numpy()
    assert not e["u"].isna().any(), "logon week without a feature row"
    e["p"] = e.groupby(["week_start", "pc"]).ngroup().to_numpy()
    n_user = len(rows)
    pc = e.groupby("p").agg(week_start=("week_start", "first"), pc=("pc", "first"), n=("user", "size"))
    u, p = e["u"].to_numpy(), e["p"].to_numpy() + n_user
    edge_index = torch.tensor(np.vstack([np.r_[u, p], np.r_[p, u]]), dtype=torch.long)

    raw = full[USER_COLS].fillna(0.0)
    nonneg = [c for c in USER_COLS if c not in DEV_COLS]
    user_raw = pd.concat([np.log1p(raw[nonneg]), signed_log1p(raw[DEV_COLS])], axis=1)[USER_COLS].to_numpy(float)
    assert np.isfinite(user_raw).all()
    return WeeklyGraph(rows, edge_index, np.r_[rows["week_start"].to_numpy(), pc["week_start"].to_numpy()],
                       user_raw, pc["pc"].to_numpy(), np.log(pc["n"].to_numpy(float)))


def node_features(g, fit_mask):
    """(n_node, N_FEAT) float tensor; the scaler is fit on user rows where fit_mask is True only."""
    sc = StandardScaler().fit(g.user_raw[fit_mask])
    X = np.zeros((g.n_node, N_FEAT), dtype=np.float32)
    X[: g.n_user, : len(USER_COLS)] = sc.transform(g.user_raw)
    X[: g.n_user, len(USER_COLS)] = g.rows["has_history"].to_numpy()
    X[g.n_user :, len(USER_COLS) + 1] = 1.0
    X[g.n_user :, len(USER_COLS) + 2] = g.pc_logn
    assert np.isfinite(X).all()
    return torch.from_numpy(X), sc


def fold_masks(g, train_users, test_users):
    """Boolean masks over user nodes: (loss rows, eval rows). Warm-up rows are in neither."""
    h = g.rows["has_history"].to_numpy()
    return g.rows["user"].isin(set(train_users)).to_numpy() & h, g.rows["user"].isin(set(test_users)).to_numpy() & h
