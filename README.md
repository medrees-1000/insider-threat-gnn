# Insider threat detection on CERT r4.2: weekly scores and an analyst UI

Every user-week of the CERT r4.2 dataset (1,000 users, 70 insiders) is scored with XGBoost on weekly logon, device,
file, email and web features. The scores, per-week explanations and neighbourhoods are shown in a read-only Streamlit
app. Graph neural networks were tried (user-PC, user-domain and email edges) and added nothing over these features, so
the model is tabular; Neo4j is used only to browse neighbourhoods. The dataset is synthetic.
**Read `docs/PROJECT_LOG.md` for every decision, result and caveat.**

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```
The UI path needs no GPU and no PyTorch. PyTorch and PyTorch Geometric are only needed to re-run the graph notebooks
(`05`, `09`, `10`); install them first following https://pytorch.org/get-started/locally/ and
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html. Notebooks use the kernel `itgnn`:
`python -m ipykernel install --user --name itgnn --display-name "Python (itgnn)"`.

## 2. Data (not in the repo)

Download CERT r4.2 and `answers.tar.bz2` from https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
and extract so that `data/raw/` contains `logon.csv`, `device.csv`, `file.csv`, `email.csv`, `http.csv`, `LDAP/*.csv` and
`answers/insiders.csv`. `http.csv` is about 14.5 GB and the `.tar.bz2` must be decompressed end to end (about 25 minutes). The
two large files can be taken in one pass (their member paths in the archive are `r4.2/email.csv` and `r4.2/http.csv`):

```bash
mkdir -p /path/to/new_empty_dir
bzip2 -dc r4.2.tar.bz2 | tar -xv -C /path/to/new_empty_dir --no-same-owner --no-same-permissions --wildcards '*/email.csv' '*/http.csv'
mv -n /path/to/new_empty_dir/r4.2/*.csv data/raw/
```
`data/raw/`, `data/processed/` and `results/` are gitignored.

## 3. Build everything the UI shows (about 6 minutes on a 20-core machine, cold)

```bash
python -m itgnn.ui_export all        # = caches, scores, edges, metrics; each stage can run alone
```
| Stage | What it writes | Time |
| --- | --- | --- |
| `caches` | `data/processed/email.parquet`, `http_user_week_domain.parquet` (built only if missing; the http file is streamed once) | about 2 min |
| `scores` | `results/ui_scores.parquet` (out-of-fold scores, no labels), `ui_eval.parquet` (labels, evaluation view only), `ui_scores_check.json` | about 0.5 min (+1.5 min the first time: feature cache) |
| `edges` | `results/ui_weeks.parquet`, `ui_edges.parquet`, `ui_edges_check.json` (neighbourhoods of the top 2% of weeks) | seconds |
| `metrics` | `results/ui_metrics.json` (recomputed from XGBoost fits; no notebook output needed) | about 1 min |

Scores are out of fold only: 25 XGBoost fold models (5x5 user-split CV), every row scored by models that did not train on its
user, averaged over the 5 repeats; explanations are XGBoost's `pred_contribs` grouped into 9 feature groups.

## 4. Neo4j (optional; the app falls back to the parquet files)

```bash
cp .env.example .env                 # then set NEO4J_PASSWORD in .env (gitignored; never put it in code)
docker compose up -d                 # first start pulls the neo4j:5 image (about 1 GB); retry if the pull is interrupted
python -m itgnn.neo4j_ui load        # about 5 s; loads the top-2% weeks: UserWeek, User, PC, Domain nodes
docker compose down                  # stop it (data stays in ./neo4j_data)
```
The app uses Neo4j when it is reachable and loaded, otherwise `results/ui_edges.parquet`; both return identical rows (tested).
No label, scenario or attack window is ever written to Neo4j. Neo4j Browser: http://localhost:7474.

## 5. Run the app

```bash
streamlit run app/streamlit_app.py   # open http://localhost:8501 (also from a Windows browser when run inside WSL2)
```
Panels: alert list (budget 0.25/0.5/1/2% and an optional calendar week), user detail (score timeline, generic-feature score,
explanation bars), neighbourhood graph, metrics and limitations. The **evaluation view** (sidebar, off by default) is the only
place that loads ground-truth labels. The app is read-only: it never trains and never writes. `ITGNN_RESULTS=<dir>` points it at
another results folder.

## 6. Tests

```bash
pytest          # about 10 s, synthetic data only (no data, no Neo4j needed)
```

## Layout

| Path | Purpose |
| --- | --- |
| `app/streamlit_app.py` | the analyst UI (read-only) |
| `src/itgnn/ui_export.py` | `python -m itgnn.ui_export caches|scores|edges|metrics|all` |
| `src/itgnn/ui_data.py` | pure helpers behind the app (alert lists, timelines, explanations, parquet backend, metrics tables) |
| `src/itgnn/neo4j_ui.py` | Neo4j loader and query backend, `auto_source` fallback; `python -m itgnn.neo4j_ui load` |
| `src/itgnn/commslogs.py`, `comms_eval.py`, `comms_audit.py`, `comms_graph.py` | email/web features, time-split helpers, audit helpers, graph v2 |
| `src/itgnn/windows.py`, `weekly_features.py`, `evaluation.py`, `data.py` | weekly dataset, history features, folds and metrics, loaders |
| `notebooks/` | `01`-`05` early work, `06`-`08` email/web probe, audit and time split, `09`-`10` graph v2 (gate failed) |
| `configs/config.yaml` | paths, seed, CV, GNN settings |
| `docs/PROJECT_LOG.md` | decisions, results, caveats, open questions |

The first-commit scripts `itgnn.baseline`, `itgnn.train` and `itgnn.neo4j_io` (user-level, never run in this work) are
unverified and are not part of the path above.
