# Insider threat detection with graph neural networks

Detect malicious insiders in the CERT Insider Threat dataset (r4.2) by modeling
user activity as a graph and classifying users with a GNN, compared against a
tabular baseline. Results are written to Neo4j so an analyst UI can explore each
flagged user's neighborhood.

## Pipeline

```
CERT r4.2 logs -> clean + subsample -> user/PC graph -> baseline + GNN -> evaluate -> Neo4j -> Streamlit UI
```

## Setup

1. **Python environment** (3.10 or newer)

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

   Install PyTorch for your machine first (pick CPU or CUDA at https://pytorch.org/get-started/locally/),
   then PyTorch Geometric (https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html),
   then the rest:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Dataset.** Download from https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
   (DOI: https://doi.org/10.1184/R1/12841247.v1). You need two files:

   - the `r4.2` archive (the logs)
   - `answers.tar.bz2` (the ground-truth list of malicious users)

   Extract both into `data/raw/` so you end up with:

   ```
   data/raw/logon.csv
   data/raw/device.csv
   data/raw/file.csv
   data/raw/answers/insiders.csv
   ```

   Check the readme inside each archive for the exact columns. If a column name
   differs from what `src/itgnn/data.py` expects, fix it there.

3. **Neo4j** (needed for the UI phase, not for training)

   ```bash
   cp .env.example .env               # then edit the password
   docker compose up -d
   ```

   Neo4j Browser opens at http://localhost:7474.

## Run

```bash
python -m itgnn.baseline     # tabular baseline -> results/baseline_scores.csv
python -m itgnn.train        # GNN              -> results/gnn_scores.csv
python -m itgnn.neo4j_io     # load graph and scores into Neo4j
streamlit run app/streamlit_app.py
pytest                       # smoke tests
```

## Layout

| Path | Purpose |
| --- | --- |
| `configs/config.yaml` | Paths, subsample size, seed, training settings |
| `src/itgnn/data.py` | Load logs, labels, subsample, train/val/test split |
| `src/itgnn/graph.py` | User features and PyG graph construction |
| `src/itgnn/baseline.py` | Random forest on per-user features (no graph) |
| `src/itgnn/models.py` | GraphSAGE node classifier |
| `src/itgnn/train.py` | Training, evaluation, score export |
| `src/itgnn/neo4j_io.py` | Write the graph and scores to Neo4j, neighborhood queries |
| `app/streamlit_app.py` | Analyst UI: ranked users plus neighborhood graph |
| `tests/` | Smoke tests on synthetic data |

## Notes for the report

- CERT r4.2 is a "dense needles" release with an unrealistically high amount of
  malicious activity, so scores here will look better than they would in practice.
  Consider re-running on r5.2 or r6.2 as a robustness check.
- Report PR-AUC, precision and recall, not accuracy. Malicious users are rare.
- Keep the same train/validation/test users for the baseline and the GNN.
