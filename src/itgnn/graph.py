"""Per-user features and the user-PC access graph."""
import numpy as np
import pandas as pd

FEATURE_COLS = ["n_logon", "n_after_hours", "n_pcs", "n_device", "n_file"]
FEATURE_COLS_B = FEATURE_COLS + ["active_days"] + [f"{c}_per_day" for c in FEATURE_COLS]


def user_features(logon, device, file, users):
    """One row per user: simple activity counts. Used by the baseline and as GNN node features."""
    lg = logon[(logon["activity"] == "Logon") & logon["user"].isin(users)].copy()
    lg["after_hours"] = ~lg["date"].dt.hour.between(7, 18)
    g = lg.groupby("user")
    feats = pd.DataFrame(
        {
            "n_logon": g.size(),
            "n_after_hours": g["after_hours"].sum(),
            "n_pcs": g["pc"].nunique(),
        }
    )
    feats["n_device"] = device[device["user"].isin(users)].groupby("user").size()
    feats["n_file"] = file[file["user"].isin(users)].groupby("user").size()
    return feats.reindex(users).fillna(0)[FEATURE_COLS]


def user_features_extended(logon, device, file, users):
    """Feature set B: the five counts, active_days (distinct days with a Logon) and per-active-day rates."""
    feats = user_features(logon, device, file, users)
    lg = logon[(logon["activity"] == "Logon") & logon["user"].isin(users)]
    days = lg.groupby("user")["date"].apply(lambda d: d.dt.normalize().nunique())
    feats["active_days"] = days.reindex(users).fillna(0)
    for c in FEATURE_COLS:
        feats[f"{c}_per_day"] = (feats[c] / feats["active_days"]).where(feats["active_days"] > 0, 0.0)
    return feats[FEATURE_COLS_B]


def logon_edges(logon, users):
    """User -> PC edges with the number of logons on each pair."""
    lg = logon[(logon["activity"] == "Logon") & logon["user"].isin(users)]
    return lg.groupby(["user", "pc"]).size().reset_index(name="count")


def build_graph(logon, device, file, labels, users):
    """Bipartite user/PC graph as a PyG Data object.

    Nodes 0..n_users-1 are users, the rest are PCs. Edges are undirected logon relations.
    Returns (data, users, pcs). Only user nodes carry labels (data.user_mask marks them).
    """
    import torch
    from torch_geometric.data import Data

    feats = user_features(logon, device, file, users)
    edges = logon_edges(logon, users)
    pcs = sorted(edges["pc"].unique())
    nu, npc = len(users), len(pcs)

    uidx = {u: i for i, u in enumerate(users)}
    pidx = {p: i + nu for i, p in enumerate(pcs)}
    src = edges["user"].map(uidx).to_numpy()
    dst = edges["pc"].map(pidx).to_numpy()
    w = edges["count"].to_numpy(dtype=float)

    edge_index = torch.tensor(np.vstack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long)
    edge_weight = torch.tensor(np.r_[w, w], dtype=torch.float)

    k = feats.shape[1]
    x = torch.zeros(nu + npc, k + 2)
    x[:nu, :k] = torch.tensor(np.log1p(feats.to_numpy()), dtype=torch.float)
    x[:nu, k] = 1.0      # is_user
    x[nu:, k + 1] = 1.0  # is_pc

    y = torch.zeros(nu + npc, dtype=torch.long)
    y[:nu] = torch.tensor([int(u in labels) for u in users], dtype=torch.long)
    user_mask = torch.zeros(nu + npc, dtype=torch.bool)
    user_mask[:nu] = True

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, user_mask=user_mask)
    return data, users, pcs
