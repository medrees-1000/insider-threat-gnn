"""Tabular baseline: random forest on per-user features, no graph structure."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from itgnn.data import load_config, prepare
from itgnn.graph import user_features


def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    d = prepare(cfg)
    feats = user_features(d["logon"], d["device"], d["file"], d["users"])
    y = pd.Series({u: int(u in d["labels"]) for u in d["users"]})

    sp = d["split"]
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=cfg["seed"], n_jobs=-1
    )
    clf.fit(feats.loc[sp["train"]], y.loc[sp["train"]])

    scores = pd.Series(clf.predict_proba(feats)[:, 1], index=feats.index, name="baseline_score")
    test_ap = average_precision_score(y.loc[sp["test"]], scores.loc[sp["test"]])
    print(f"Baseline test PR-AUC: {test_ap:.3f}")

    split_of = {u: s for s, us in sp.items() for u in us}
    out = pd.DataFrame(
        {"baseline_score": scores, "label": y, "split": pd.Series(split_of)}
    ).rename_axis("user")
    Path(cfg["results_dir"]).mkdir(exist_ok=True)
    out.to_csv(Path(cfg["results_dir"]) / "baseline_scores.csv")


if __name__ == "__main__":
    main()
