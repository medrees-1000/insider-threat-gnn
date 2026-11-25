# Insider threat detection on CERT r4.2: weekly risk scores and an analyst app

Every week, every person's logons, USB use, file copies, email and web browsing are turned into a risk score. The riskiest weeks become
alerts, and a Streamlit app lets an analyst see *who*, *why*, and *what they touched*. It is built on the synthetic CERT r4.2 dataset
(1,000 users, 70 insiders) and evaluated honestly: every score the app shows comes from people the model had never seen.

![Overview](docs/screenshots/01-overview.png)

## The result in one paragraph

At the tightest alert budget (the riskiest 0.25% of weeks) the model catches **51 of the 70 insiders while flagging 0 of 930 normal users**
(159 alerts, 96% of them real attack weeks). Across 25 train/test splits it ranks attack weeks with PR-AUC **0.87 against 0.46** for the
first baseline, and it keeps **85-97% of that accuracy** when trained only on earlier weeks and tested on later ones (60% at the earliest
cutoff, when few attacks had happened). The numbers above come from `results/ui_metrics.json` and are read live by the app.
The same app shows the limits: a global top-q% list is easier than a real weekly queue, the named-site features (for example
`wikileaks.org`, visited by 30 users who are all insiders) are near-label markers in this synthetic data, and graph models tested against
no-edge and shuffled-edge controls added nothing, so the model is a plain gradient-boosted tree (XGBoost).

## Try it: a 60-second tour

1. **Overview** - read the three cards, then drag the *alert budget* slider and watch insiders caught and false alarms change live.
2. **Threat map** - press *Play* to step through the weeks; switch to *Campaign view* to see flagged people linked through shared websites, PCs and email; click a dot for the evidence and press *Open in Investigate*.
3. **Alerts** - the top weeks as a table (percentile by default; switch to log-odds or raw score); each row lists its main reasons.
4. **Investigate** - one person's risk over time with the alert bands, and the bar chart of what pushed the week up.
5. **What-if** - press *Leak-site visit* (or move any slider) and watch the pinned rating respond; open *Sweep one behaviour* to see where the verdict would flip.
6. **Rigor and limits** - the checks behind the numbers (leak caught in the first baseline, 5x5 user-split CV, causality check, time-based test, graph controls) and what this does *not* show.

Optional: tick *Evaluation view* in the sidebar to see ground-truth labels and attack windows (off by default; analysts never see them).

### Screenshots to save (exact names under `docs/screenshots/`)

| File | What to capture |
| --- | --- |
| `01-overview.png` | Overview page, budget slider at 0.5% |
| `02-threat-map-week.png` | Threat map, week view of a busy week (for example calendar week 36), one person focused |
| `03-threat-map-campaign.png` | Threat map, campaign view |
| `04-alerts.png` | Alerts page, percentile display |
| `05-investigate.png` | Investigate page for a flagged person |
| `06-what-if.png` | What-if page after pressing *Leak-site visit* |
| `07-what-if-sweep.png` | The sweep chart with the "What would it take?" sentence |
| `08-rigor.png` | Rigor and limits page (charts) |
| `09-evaluation-view.png` | Threat map with *Evaluation view* on (double rings on true insiders) |
| `10-neo4j-browser.png` | Neo4j Browser showing the neighbourhood query (optional) |

Capture at about 1500 px wide in the dark theme (the default). Streamlit's *Settings* menu switches to the light theme.

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

## 3. Build everything the UI shows (about 9 minutes on a 20-core machine, cold)

```bash
python -m itgnn.ui_export all        # = caches, scores, edges, metrics, model, curve, facts; each stage can run alone
```
| Stage | What it writes | Time |
| --- | --- | --- |
| `caches` | `data/processed/email.parquet`, `http_user_week_domain.parquet` (built only if missing; the http file is streamed once) | about 2 min |
| `scores` | `results/ui_scores.parquet` (out-of-fold scores, no labels), `ui_eval.parquet` (labels, evaluation view only), `ui_scores_check.json` | about 0.5 min (+1.5 min the first time: feature cache) |
| `edges` | `results/ui_weeks.parquet`, `ui_edges.parquet`, `ui_edges_check.json` (neighbourhoods of the top 2% of weeks) | seconds |
| `metrics` | `results/ui_metrics.json` (recomputed from XGBoost fits; no notebook output needed) | about 1 min |
| `model` | `results/final_xgb_main.json`, `final_xgb_generic.json`, `final_model_meta.json`, `final_model_rows.parquet` (the What-if scorer) | about 20 s |
| `curve` | `results/ui_budget_curve.json` (alert-budget curve for the Overview and Rigor pages) | about 1 min |
| `facts` | `results/ui_rigor_facts.json` (evaluation aggregates only: tenure leak, CV, causality check, graph controls) | about 1.5 min |

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
Pages: Overview, Threat map (interactive Cytoscape.js graph), Alerts, Investigate, What-if (live scorer from the saved final models), Rigor and limits. The **evaluation view** (sidebar, off by default) is the only
place that loads ground-truth labels. The app is read-only: it never trains and never writes. `ITGNN_RESULTS=<dir>` points it at
another results folder.

## 6. Tests

```bash
pytest          # about 15 s, synthetic data only (no data, no Neo4j needed)
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
