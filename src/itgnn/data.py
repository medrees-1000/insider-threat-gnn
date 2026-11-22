"""Load the CERT r4.2 logs, ground-truth labels, and build user subsamples and splits."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

LOG_SPECS = {
    "logon": ("logon.csv", ["date", "user", "pc", "activity"]),
    "device": ("device.csv", ["date", "user", "pc", "activity"]),
    "file": ("file.csv", ["date", "user", "pc"]),
}


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_log(raw_dir, name):
    """Read one CERT log (logon, device or file) with only the columns we need."""
    filename, cols = LOG_SPECS[name]
    df = pd.read_csv(Path(raw_dir) / filename, usecols=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_labels(raw_dir, release="4.2"):
    """Return the set of malicious user ids from the answer key.

    Expects data/raw/answers/insiders.csv with a `dataset` and a `user` column.
    Check the readme in answers.tar.bz2 if your copy differs.
    """
    df = pd.read_csv(Path(raw_dir) / "answers" / "insiders.csv")
    df = df[df["dataset"].astype(str) == release]
    return set(df["user"])


def subsample_users(users, labels, n, seed):
    """Pick n users, keeping the malicious share the same as in the full set."""
    users = sorted(users)
    if n >= len(users):
        return users
    y = [u in labels for u in users]
    picked, _ = train_test_split(users, train_size=n, stratify=y, random_state=seed)
    return sorted(picked)


def make_split(users, labels, val_size, test_size, seed):
    """Stratified train/val/test split of users. Reuse it for baseline and GNN."""
    y = [u in labels for u in users]
    train_val, test, y_tv, _ = train_test_split(
        users, y, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=y_tv, random_state=seed
    )
    return {"train": train, "val": val, "test": test}


def prepare(cfg):
    """Load everything once: logs, labels, chosen users and the split."""
    raw = cfg["raw_dir"]
    logon = load_log(raw, "logon")
    device = load_log(raw, "device")
    file = load_log(raw, "file")
    labels = load_labels(raw)
    users = subsample_users(logon["user"].unique(), labels, cfg["n_users"], cfg["seed"])
    split = make_split(users, labels, cfg["val_size"], cfg["test_size"], cfg["seed"])
    return {
        "logon": logon,
        "device": device,
        "file": file,
        "labels": labels,
        "users": users,
        "split": split,
    }
