"""Train the GNN and export per-user risk scores."""
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from itgnn.data import load_config, prepare
from itgnn.graph import build_graph
from itgnn.models import GraphSAGE


def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    torch.manual_seed(cfg["seed"])

    d = prepare(cfg)
    data, users, _pcs = build_graph(d["logon"], d["device"], d["file"], d["labels"], d["users"])
    uidx = {u: i for i, u in enumerate(users)}
    idx = {s: torch.tensor([uidx[u] for u in us]) for s, us in d["split"].items()}

    # Weight the rare malicious class more heavily
    y_train = data.y[idx["train"]]
    n_pos = int(y_train.sum())
    weight = torch.tensor([1.0, (len(y_train) - n_pos) / max(n_pos, 1)])

    model = GraphSAGE(data.x.shape[1], mcfg["hidden"], mcfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])

    def scores():
        model.eval()
        with torch.no_grad():
            return F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]

    best_val, best_state = -1.0, None
    for epoch in range(1, mcfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[idx["train"]], data.y[idx["train"]], weight=weight)
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            s = scores()
            val_ap = average_precision_score(data.y[idx["val"]].numpy(), s[idx["val"]].numpy())
            if val_ap > best_val:
                best_val = val_ap
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val PR-AUC {val_ap:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    s = scores()
    test_ap = average_precision_score(data.y[idx["test"]].numpy(), s[idx["test"]].numpy())
    print(f"GNN test PR-AUC: {test_ap:.3f}")

    split_of = {u: sp for sp, us in d["split"].items() for u in us}
    out = pd.DataFrame(
        {
            "gnn_score": s[: len(users)].numpy(),
            "label": data.y[: len(users)].numpy(),
            "split": [split_of[u] for u in users],
        },
        index=pd.Index(users, name="user"),
    )
    Path(cfg["results_dir"]).mkdir(exist_ok=True)
    out.to_csv(Path(cfg["results_dir"]) / "gnn_scores.csv")


if __name__ == "__main__":
    main()
