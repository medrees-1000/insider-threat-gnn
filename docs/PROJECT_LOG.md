# Project log

Last updated: after step 3 (weekly baseline). Everything below comes from the repo, notebook
outputs and `results/` files unless marked **unverified**.

## 1. Goal and dataset
- Goal: detect malicious insiders in CERT r4.2 by modelling activity as a user/PC graph and
  classifying with GNNs, compared against tabular baselines. Later: write scores to Neo4j and
  explore them in a Streamlit UI (code for both exists from the first commit, not exercised yet).
- Data: raw files in `data/raw/` (gitignored): `logon.csv` (854,859 rows), `device.csv` (405,380),
  `file.csv` (445,581), `psychometric.csv`, `LDAP/` (18 monthly files), `answers/` (ground truth).
  Events span 2010-01-02 to 2011-05-17. 1,000 users, ~1,000 PCs, no missing values, no duplicate event ids.
- Label: a user is malicious if listed in `answers/insiders.csv` for release 4.2: 70 insiders
  (scenario 1: 30, scenario 2: 30, scenario 3: 10), 7.0% of users. All 70 appear in the logon log.
- Only 265 users have device events and 264 have file events; every insider has device activity.

## 2. Layout
| Path | What it does |
| --- | --- |
| `configs/config.yaml` | paths, `n_users` (null = all), seed 42, `cv` (5 splits x 5 repeats), GNN settings |
| `src/itgnn/data.py` | `load_log` (optional parquet cache), `load_labels`, `subsample_users`, `make_split`, `prepare` (uses cache `data/processed`) |
| `src/itgnn/graph.py` | `user_features` (5 counts), `user_features_extended` (feature set B: + `active_days` and per-day rates), `logon_edges`, `build_graph` (bipartite user/PC PyG graph) |
| `src/itgnn/evaluation.py` | `make_folds` (repeated stratified k-fold over users), `evaluate_scores` (PR-AUC, ROC-AUC, tie-aware precision@k / recall@k), `summarize` |
| `src/itgnn/windows.py` | `load_attack_windows`, `make_weekly_features`, `label_weeks`, `week_masks` (user-per-week dataset) |
| `src/itgnn/baseline.py`, `train.py`, `models.py`, `neo4j_io.py`, `app/streamlit_app.py` | from the first commit: single-split RF baseline, GraphSAGE training, Neo4j export, UI. Never run in this work (no `results/baseline_scores.csv` or `gnn_scores.csv` exists); **unverified** that they run |
| `notebooks/01_eda.ipynb` | dataset overview, label, imbalance, features, graph stats, split, unused files |
| `notebooks/02_baseline.ipynb` | user-level baselines on the 5x5 folds (feature sets A and B) |
| `notebooks/03_weekly_baseline.ipynb` | weekly dataset, baselines, control models, alert budget |
| `tests/` | 19 tests on synthetic data (smoke, evaluation, extended features, windows) |

## 3. Decisions and reasons
- **Hybrid layout**: reusable logic in `src/`, notebooks tell the story; keeps logic testable.
- **Repeated stratified 5x5 CV by user**: only 70 insiders, so one split is noisy (the old split has 21 test insiders; each is ~5 recall points). Same folds for every model.
- **Tie-aware precision/recall@k**: many scores tie (constant prior, `n_device > 0`); a stable sort would give arbitrary results. Expected value under random tie-breaking is used.
- **User-level first, then weekly windows**: the user-level label cannot say when an attack happened, and notebook 02 showed a tenure shortcut (see 4).
- **Weekly label rule**: a user-week is malicious if `[week_start, week_start+7d)` overlaps the user's attack window `[start, end]` (end inclusive). Weeks start Monday; a row exists only if the user has a Logon that week.
- **Split by user, never by week**, so no user is in both train and test (tested).
- **Graph v1 is logon-only** (what `build_graph` does today); device/file edges are planned as a later ablation (graph v2).
- **GNN trained from scratch** (GraphSAGE in `models.py`); no pretrained model.
- **Neo4j + Streamlit** for the analyst UI (code exists, not yet used).
- Dates in `insiders.csv` are parsed strictly as month/day/year; in the 4.2 rows the first field is never above 12.

## 4. Results so far
**User-level baseline (notebook 02, 1,000 users, 25 folds, PR-AUC mean).** Constant prior 0.070; `n_device > 0` 0.266;
logistic regression 0.415 (A) / 0.589 (B); random forest 0.982 (A) / 0.995 (B); XGBoost 0.987 (A) / 0.985 (B).
**Misleading:** `active_days` has permutation importance 0.342 vs <=0.013 for any other feature. 75.5% of normal
users have 346 active days (the full period); 0% of insiders do (median 213). `active_days` alone has AUC 0.053
(i.e. ~0.95 inverted). Insiders vanish from the data after attacking, so tenure leaks the label, and since all
features span the full period, tenure is baked into every count. Treat 0.98 as an upper bound, not skill.

**Weekly baseline (notebook 03; 67,238 user-weeks, 322 attack weeks = 0.48%, all 70 insiders covered; 25 folds).**
| Model | PR-AUC | ROC-AUC |
| --- | --- | --- |
| constant prior | 0.005 ± 0.001 | 0.500 |
| `n_device > 0` | 0.023 ± 0.004 | 0.896 ± 0.013 |
| logistic regression | 0.054 ± 0.024 | 0.952 ± 0.012 |
| random forest | 0.283 ± 0.071 | 0.969 ± 0.023 |
| XGBoost | 0.343 ± 0.087 | 0.981 ± 0.008 |
| control: calendar week | 0.009 ± 0.002 | 0.719 |
| control: weeks since first | 0.009 ± 0.002 | 0.717 |
| control: weeks until last (uses future, diagnostic) | 0.044 ± 0.005 | 0.941 ± 0.006 |

Alert budget (top q% of weeks per test fold, pooled per repeat, mean over 5 repeats; 70 insiders, 930 normal users):
- q=0.25% (~170 weeks): RF catches 34.4 insiders, flags 10.6 normal users; XGBoost 38.6 and 15.6.
- q=2% (~1,347 weeks): RF 56.2 insiders / 86.4 normal users; XGBoost 53.2 / 58.4; `n_device > 0` 23.4 / 190.8; random 7.6 / 689.8.
- "Caught" = at least one attack week flagged. Feature rows hold the five weekly counts only (no tenure/calendar).
- Weekly OOF scores: `results/weekly_baseline_oof_scores.parquet`; user-level: `results/baseline_oof_scores.csv`.

## 5. Findings and caveats
- Scenario 2 supplies 260 of 322 attack weeks (scenario 1: 52, scenario 3: 10), so weekly results are dominated by it.
- XGBoost vs RF is within noise (spreads overlap; not tested). Repeated-CV folds are correlated, so stds understate uncertainty.
- Labels are noisy: any overlap with an attack window counts, and windows last up to 58 days (median 7).
- The future-looking control (ROC-AUC 0.941) shows leaving-related signal exists in the data: every insider's last
  attack week is within 4 weeks of their last week. The real models cannot see it, but correlated behaviour may leak in.
- Top-q% over the whole period is easier than a weekly analyst workflow (a real one needs a fixed budget per week).
- Weekly features have no per-user history, so a habitual device user looks like a new one.
- First and last calendar weeks are partial. No device/file events are dropped by the "no logon, no row" rule (0 dropped).
- **Charts in notebooks 02 and 03 were never opened by Claude Code**; only printed tables were checked. Notebook 01's charts were also not viewed.
- **Unverified**: that `baseline.py`, `train.py`, `neo4j_io.py` and the Streamlit app run; the contents of `answers/r4.2-*`
  (only the file listing was seen: `r4.2-1` has 30 files named `r4.2-1-<user>.csv`); scenario meanings beyond `answers/scenarios.txt`.

## 6. Roadmap
- **Done**: setup and caching; EDA (01); CV utilities and tests; user-level baseline (02); weekly dataset and baseline with controls (03).
- **Next**: stronger weekly baseline with past-only history features (skip or mark a new user's first weeks) and simple graph
  statistics (e.g. how many other users share this user's PCs that week), with per-scenario results, on the same folds.
- **Then**: read-only look at `answers/r4.2-*` to see whether per-event labels exist; a label-sensitivity check.
- **Later**: GNN on weekly graphs; graph v2 with device and file edges; error analysis; Neo4j + Streamlit; report; optional r5.2 re-run.

## 7. Open questions for the owner
- Per-week alert budget: what review capacity is realistic (weeks or users per week)?
- New users with no history: skip their first weeks or mark them with a flag?
- Should fixed model hyperparameters move from notebooks into `config.yaml`?
- Is the user-based split enough, or do we also want a time-based holdout?
- Remove the 6 device rows that repeat (date, user, pc, activity) with different ids, or keep them?

## 8. Rerun commands
```bash
source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .   # first time only; torch / PyG installed separately (README)
python -m ipykernel install --user --name itgnn --display-name "Python (itgnn)"   # once
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb          # ~15 s
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline.ipynb     # ~1.5 min
jupyter nbconvert --to notebook --execute --inplace notebooks/03_weekly_baseline.ipynb  # ~1.5 min
```
