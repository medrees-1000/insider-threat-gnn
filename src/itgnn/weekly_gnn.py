"""Light GNN / MLP node classifier for the weekly user-PC graph, full-batch training."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch_geometric.nn import SAGEConv

from itgnn.weekly_graph import fold_masks, node_features

class WSageConv(nn.Module):
    """GraphSAGE with a weighted mean: lin_l(A x) + lin_r(x), A = row-normalised sparse adjacency.

    Same two linear maps as SAGEConv (bias on lin_l only). With unit weights it equals SAGEConv(aggr="mean").
    The forward takes the sparse matrix where the other convs take edge_index.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, A):
        return self.lin_l(torch.sparse.mm(A, x)) + self.lin_r(x)


CONVS = {"sage": SAGEConv, "mlp": nn.Linear, "wsage": WSageConv}  # add gcn / gat here later (GAT needs head handling)


class WeeklyNet(nn.Module):
    """`layers` x (conv -> ReLU -> dropout) then a linear classifier. conv='mlp' ignores the edges."""

    def __init__(self, in_dim, conv="sage", layers=2, hidden=64, dropout=0.3):
        super().__init__()
        if conv not in CONVS:
            raise ValueError(f"conv must be one of {list(CONVS)}")
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(CONVS[conv](dims[i], dims[i + 1]) for i in range(layers))
        self.out = nn.Linear(hidden, 2)
        self.conv, self.dropout = conv, dropout

    def forward(self, x, edge_index):
        for c in self.convs:
            x = c(x) if self.conv == "mlp" else c(x, edge_index)
            x = F.dropout(F.relu(x), self.dropout, self.training)
        return self.out(x)


def class_weight(y, cap):
    """sqrt(neg/pos) of the training rows, capped."""
    pos = int((np.asarray(y) == 1).sum())
    return float(min(np.sqrt((len(y) - pos) / max(pos, 1)), cap))


def train_scores(x, edge_index, y, loss_idx, epochs, conv, gcfg, seed, val_idx=None):
    """Train full-batch for a fixed number of epochs; return (P(malicious) for all nodes, history).

    Only y[loss_idx] enters the loss. With val_idx, the history has loss and validation PR-AUC per epoch.
    The model after the last epoch is used (no checkpoint selection).
    """
    if gcfg["class_weight"] != "sqrt":
        raise ValueError("only class_weight: sqrt is implemented")
    torch.manual_seed(seed)
    model = WeeklyNet(x.shape[1], conv, gcfg["layers"], gcfg["hidden"], gcfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=gcfg["lr"], weight_decay=gcfg["weight_decay"])
    ytr = torch.as_tensor(np.asarray(y)[loss_idx], dtype=torch.long)
    w = torch.tensor([1.0, class_weight(ytr.numpy(), gcfg["max_class_weight"])])
    loss_idx = torch.as_tensor(loss_idx)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[loss_idx], ytr, weight=w)
        loss.backward()
        opt.step()
        if val_idx is not None:
            model.eval()
            with torch.no_grad():
                s = F.softmax(model(x, edge_index)[val_idx], 1)[:, 1].numpy()
            hist.append({"epoch": ep, "loss": loss.item(), "val_ap": average_precision_score(np.asarray(y)[val_idx], s)})
    model.eval()
    with torch.no_grad():
        scores = F.softmax(model(x, edge_index), 1)[:, 1].numpy()
    return scores, pd.DataFrame(hist)


def pilot_curves(g, train_users, gcfg, conv, seed):
    """Inner validation by user inside train_users only: loss and validation PR-AUC per epoch and split."""
    pcfg = gcfg["pilot"]
    users = np.array(sorted(train_users))
    ins = set(g.rows.loc[g.rows["label"] == 1, "user"])
    y_user = np.array([u in ins for u in users])
    y = g.rows["label"].to_numpy()
    out = []
    sss = StratifiedShuffleSplit(pcfg["n_inner_splits"], test_size=pcfg["inner_val_size"], random_state=seed)
    for i, (tr, va) in enumerate(sss.split(users, y_user)):
        fit, val = fold_masks(g, users[tr], users[va])
        x, _ = node_features(g, fit)
        _, h = train_scores(x, g.edge_index, y, np.flatnonzero(fit), pcfg["max_epochs"], conv, gcfg, seed, np.flatnonzero(val))
        out.append(h.assign(split=i, model=conv))
    return pd.concat(out, ignore_index=True)


def choose_epoch(curves, smooth, tol):
    """Earliest epoch whose smoothed (trailing mean over `smooth` epochs) mean-over-splits PR-AUC is within tol of the best."""
    m = curves.groupby("epoch")["val_ap"].mean().rolling(smooth, min_periods=smooth).mean().dropna()
    return int(m[m >= m.max() - tol].index[0])
