# Project log

Last updated: end of step 5 (light GNN on weekly graphs) and the decision for the next step (option B: email/web data). Everything below comes from the repo, notebook
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
- **Not in `data/raw/`**: `http.csv` and `email.csv` (verified by listing the folder; only logon, device, file, psychometric, LDAP, answers, readme and license are there). They are in the original archive
  `/mnt/c/Users/idree/Desktop/Downloads/r4.2.tar.bz2` (4,824,287,500 bytes on disk; its contents were NOT listed or read). `answers/scenarios.txt` describes scenario 1 as uploading data to wikileaks.org, scenario 2 as surfing
  job websites and soliciting a competitor, scenario 3 as a mass email sent from a supervisor's account (scenarios 1-3 also involve removable drives): web and email activity are named in all three, which motivates option B below.

## 2. Layout
| Path | What it does |
| --- | --- |
| `configs/config.yaml` | paths, `n_users` (null = all), seed 42, `cv` (5 splits x 5 repeats), GNN settings |
| `src/itgnn/data.py` | `load_log` (optional parquet cache), `load_labels`, `subsample_users`, `make_split`, `prepare` (uses cache `data/processed`) |
| `src/itgnn/graph.py` | `user_features` (5 counts), `user_features_extended` (feature set B: + `active_days` and per-day rates), `logon_edges`, `build_graph` (bipartite user/PC PyG graph) |
| `src/itgnn/evaluation.py` | `make_folds` (repeated stratified k-fold over users), `evaluate_scores` (PR-AUC, ROC-AUC, tie-aware precision@k / recall@k), `summarize` |
| `src/itgnn/windows.py` | `load_attack_windows`, `make_weekly_features`, `label_weeks`, `week_masks` (user-per-week dataset) |
| `src/itgnn/weekly_features.py` | past-only history features (`add_history_features`), same-week graph statistics (`add_graph_stats`), `drop_warmup`, feature sets W1-W4, `signed_log1p` |
| `src/itgnn/weekly_graph.py` | one user-PC graph per week (all weeks as one disjoint graph), W2 node features, train-fold-only scaler, loss/eval masks |
| `src/itgnn/weekly_gnn.py` | `WeeklyNet` (conv `sage` or `mlp`, layers, hidden, dropout configurable), full-batch training, pilot and epoch rule |
| `src/itgnn/baseline.py`, `train.py`, `models.py`, `neo4j_io.py`, `app/streamlit_app.py` | from the first commit: single-split RF baseline, GraphSAGE training, Neo4j export, UI. Never run in this work (no `results/baseline_scores.csv` or `gnn_scores.csv` exists); **unverified** that they run |
| `notebooks/01_eda.ipynb` | dataset overview, label, imbalance, features, graph stats, split, unused files |
| `notebooks/02_baseline.ipynb` | user-level baselines on the 5x5 folds (feature sets A and B) |
| `notebooks/03_weekly_baseline.ipynb` | weekly dataset, baselines, control models, alert budget |
| `notebooks/04_weekly_baseline_v2.ipynb` | W1-W4 feature sets x LR/RF/XGBoost, budgets, per-scenario results, paired differences |
| `notebooks/05_gnn_lite.ipynb` | light GraphSAGE vs same network without edges vs RF/XGBoost W2 (pilot, 25 folds, budgets, per scenario, seed check) |
| `tests/` | 32 tests on synthetic data (smoke, evaluation, extended features, windows, weekly features incl. causality, weekly graph/GNN, alert budget); the alert-budget regression test skips if `results/weekly_v2_oof_scores.parquet` is missing |

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
- **DECISION (owner, end of step 5): pursue option B**, i.e. bring in `email.csv` and `http.csv` before any further GNN work. **Hypothesis (unverified): the scenario signals live mostly in `http.csv` and `email.csv`, which were excluded at extraction.** Evidence so far is only the scenario text above and the step 4-5 results (history features help, graph structure does not); nothing has been measured on email or web data yet.
- **Heavier GNNs (3-4 layers, bigger hidden size, GCN/GAT, device/file edges) are not planned** on the current logon graph, for two reasons: (1) the graph is sparse: 59.6% of (PC, week) nodes have exactly one user and the mean user degree is 1.67, so most user-weeks have no neighbour to aggregate (interpretation, not tested); (2) the light GraphSAGE did not beat the tree models (PR-AUC 0.430 vs RF W2 0.503, XGB W2 0.458) and its gain over the edge-less MLP (0.395) did not show up in the alert budget. The model code keeps depth, hidden size and conv type configurable, so this can be reopened if new data gives the graph more structure.

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

**Weekly baseline v2 (notebook 04; 63,238 user-weeks after the warm-up; same 25 folds, fixed hyperparameters).**
- Rule: week *w* uses data up to *w*. History = mean over the user's previous 4 active weeks of the five counts, deviation (raw difference) from it, and `n_new_pcs` (PCs never used in an earlier week). Graph stats (same week only) = mean/max number of OTHER users on the user's PCs and the share of PCs nobody else used. Predictions are made at the end of each week.
- Warm-up: each user's first 4 active weeks are dropped for ALL models: 4,000 of 67,238 rows (5.9%), **0 attack weeks, 0 insiders lost** (70 of 70 eligible; prevalence 0.48% -> 0.51%). Controls are computed on the full data before the drop.
- LR uses log1p on non-negative columns and signed log on `dev_*`; every feature matrix and score is asserted finite.

| Model (PR-AUC mean ± std, 25 folds) | W1 | W2 (+history) | W3 (+graph) | W4 (both) |
| --- | --- | --- | --- | --- |
| logistic regression | 0.058 ± 0.025 | 0.088 ± 0.058 | 0.056 ± 0.025 | 0.087 ± 0.054 |
| random forest | 0.297 ± 0.075 | 0.503 ± 0.071 | 0.286 ± 0.079 | 0.492 ± 0.076 |
| XGBoost | 0.358 ± 0.091 | 0.458 ± 0.080 | 0.347 ± 0.091 | 0.459 ± 0.074 |

Constant prior 0.005, `n_device > 0` 0.024, controls: calendar week 0.009, weeks since first 0.009, weeks until last (future) 0.044. W1 on restricted rows vs notebook 03: RF 0.297 vs 0.283, XGB 0.358 vs 0.343, LR 0.058 vs 0.054.
Paired PR-AUC differences (mean ± std over folds, correlated folds): RF W2-W1 +0.206 ± 0.064 (100% of folds), W3-W1 -0.011 ± 0.036, W4-W1 +0.196 ± 0.057, W4-W2 -0.011 ± 0.025; XGB W2-W1 +0.100 ± 0.074 (92%), W3-W1 -0.011 ± 0.027, W4-W1 +0.101 ± 0.061, W4-W2 +0.001 ± 0.025.
Alert budget (insiders caught of 70 / normal users flagged of 930, mean of 5 repeats): q=0.25% RF W1 35.0/10.0, RF W2 51.4/3.2, XGB W1 36.6/14.6, XGB W2 48.0/2.8; q=2% RF W1 56.2/84.2, RF W2 65.4/35.6, XGB W1 54.0/58.0, XGB W2 62.6/30.2.
Per scenario (caught at q=0.25%, RF W1 -> W2): s1 11.4 -> 23.0 of 30, s2 22.4 -> 23.4 of 30, s3 1.2 -> 5.0 of 10 (XGB s3 only 3.2 -> 3.8). Full tables, including week-level recall, are in the notebook.
- Decision: history features help (mainly scenario 1); the three graph statistics add nothing for RF/XGBoost. `n_new_pcs` is inflated shortly after the warm-up (mean 0.59 in active weeks 4-7 vs 0.20 from week 40 on; 6.7% vs 4.5% of weeks with a new PC), so part of it is a "young account" signal; no ablation was run.
- OOF scores: `results/weekly_v2_oof_scores.parquet`. Notebook 04 has no charts (tables only), so no chart was left unopened for this step.

**Light GNN on weekly graphs (notebook 05; same 63,238 scored rows and 25 folds as step 4).**
- Graph: per week one node per user-week (67,238, incl. 4,000 warm-up rows as neighbours only) and per (PC, week) (67,375); 112,408 undirected edges, 73 weeks, no edge between weeks. 59.6% of PC-week nodes have a single user; mean user degree 1.67. User features = W2 (signed log on `dev_*`, log1p otherwise, scaler fit on training-fold rows only) + `has_history`; PC nodes: type flag + log(users that week). Warm-up rows never enter loss or evaluation; test nodes stay in the graph, their labels are never used.
- Recipe (config `gnn:`): 2 layers, hidden 64, dropout 0.3, Adam lr 0.01, weight decay 5e-4, full batch, class weight sqrt(neg/pos) capped at 50 (not tuned), last-epoch model. MLP = same network with `Linear` instead of `SAGEConv`.
- Epochs chosen once by a pilot on the training users of repeat 0 fold 0 (3 inner 80/20 splits by user, up to 300 epochs, trailing 5-epoch mean, earliest epoch within 0.01 of the best): **MLP 188, GraphSAGE 224**, written to `config.yaml`. Caveats: curves still creeping up at 300; inner splits noisy; those users are test users in other folds (one integer chosen with their data).
- Results (PR-AUC, 25 folds): MLP 0.395 ± 0.075, GraphSAGE 0.430 ± 0.073, RF W2 0.503 ± 0.071, XGB W2 0.458 ± 0.080. Paired: GraphSAGE - MLP +0.035 ± 0.021 (25/25 folds), GraphSAGE - RF W2 -0.073 ± 0.044 (0/25), GraphSAGE - XGB W2 -0.028 ± 0.057 (7/25).
- Alert budget (caught of 70 / normal flagged of 930): q=0.25% MLP 53.2/9.2, GraphSAGE 52.2/9.2, RF W2 51.4/3.2, XGB W2 48.0/2.8; q=2% MLP 61.0/32.6, GraphSAGE 62.2/40.8, RF W2 65.4/35.6, XGB W2 62.6/30.2.
- Repeat 0 fold 0 (clean fold): PR-AUC MLP 0.348, GraphSAGE 0.443, RF 0.463, XGB 0.435. Seed check on that fold (5 seeds): GraphSAGE 0.388 ± 0.036 (0.351-0.443), MLP 0.358 ± 0.023; the main run's seed was GraphSAGE's best of five.
- **Conclusion: the light GNN does not beat the tree models, and its gain over the edge-less MLP is small in PR-AUC and absent in the alert budget.** Nothing was tuned after seeing test results. `alert_budget` was extracted to `evaluation.py` and reproduces notebook 04's RF W2 / XGB W2 numbers exactly.
- Runtime: CPU only (CPU torch build, 10 threads). 25-fold run about 75 min (per-model split not recorded), pilot about 16 min, seed check about 14 min, whole notebook about 88 min. The first full execution crashed in its last cell (my bug, fixed) after saving the OOF scores; the notebook now loads `results/gnn_lite_oof_scores.parquet` (`RERUN_CV = False`) and the pilot curves (`RERUN_PILOT = False`). `results/gnn_lite_run_info.json` timings are derived from file timestamps.
- Charts opened this step: the pilot curves, the PR-AUC strip plot and the pooled PR curves. Not opened: charts in notebooks 01-03.

## 5. Findings and caveats
- Scenario 2 supplies 260 of 322 attack weeks (scenario 1: 52, scenario 3: 10), so weekly results are dominated by it.
- XGBoost vs RF is within noise (spreads overlap; not tested). Repeated-CV folds are correlated, so stds understate uncertainty.
- Labels are noisy: any overlap with an attack window counts, and windows last up to 58 days (median 7).
- The future-looking control (ROC-AUC 0.941) shows leaving-related signal exists in the data: every insider's last
  attack week is within 4 weeks of their last week. The real models cannot see it, but correlated behaviour may leak in.
- Top-q% over the whole period is easier than a weekly analyst workflow (a real one needs a fixed budget per week).
- W1 features have no per-user history (W2 adds it, from the previous 4 active weeks only).
- First and last calendar weeks are partial. No device/file events are dropped by the "no logon, no row" rule (0 dropped).
- **Step 5 caveats.** (a) The pilot's inner-validation PR-AUC was still rising slowly (and the loss still falling) at the 300-epoch cap, so both networks may be slightly under-trained; the epoch counts (MLP 188, GraphSAGE 224) come from one noisy pilot (about 11 insiders per validation split) on users who are test users in other folds. (b) One seed per fold in the 25-fold run; on repeat 0 fold 0 the seed noise (GraphSAGE 0.388 ± 0.036) is as large as the GraphSAGE-MLP gap, and the main run's seed was GraphSAGE's best of five. (c) `n_new_pcs` is inflated just after the warm-up (mean 0.59 in active weeks 4-7 vs 0.20 from week 40 on), so part of the W2 gain may be a "young account" signal; no ablation was run. (d) Weekly labels mark every week that overlaps an attack window, including quiet weeks inside long windows (up to 58 days), which caps precision for every model. (e) Repeated-CV folds are correlated, so the stds understate uncertainty.
- **Charts in notebooks 01, 02 and 03 were never opened by Claude Code**; only printed tables were checked. In step 5 the pilot curves, strip plot and pooled PR curves of notebook 05 were opened. Notebook 04 has no charts.
- **Unverified**: that `baseline.py`, `train.py`, `neo4j_io.py` and the Streamlit app run; the contents of `answers/r4.2-*`
  (only the file listing was seen: `r4.2-1` has 30 files named `r4.2-1-<user>.csv`); scenario meanings beyond `answers/scenarios.txt`.

## 6. Roadmap
- **Done**: setup and caching; EDA (01); CV utilities and tests; user-level baseline (02); weekly dataset and baseline with controls (03).
- **Done (04)**: stronger weekly baseline with past-only history features and same-week graph statistics, first 4 active weeks skipped, per-scenario results.
- **Done (05)**: light GraphSAGE vs MLP vs RF/XGB W2 on weekly graphs.
- **NEXT (step 6, option B), read-only first, nothing run yet:**
  1. Check disk space (checked today: 910 GB free on `/`, 633 GB on `C:`; the archive is 4.8 GB, so space is not a concern, but a .tar.bz2 must be decompressed end to end: expect it to be slow and run it in the background).
  2. Re-extract **only** `email.csv` and `http.csv` from `/mnt/c/Users/idree/Desktop/Downloads/r4.2.tar.bz2` into a new empty directory (untrusted archive contents: list members first, extract only those two, check member paths). Do not overwrite anything in `data/raw/`.
  3. Report schema, size, row counts and date range of each file; check that user ids and the date range match the logon log. No modelling in this step.
  4. Then test whether simple per-week email/web features (past-only rules as in step 4: week *w* uses data up to *w*) lift RF/XGBoost W2 on the same folds. **Use repeat 0 only (5 folds) for exploration; keep the full 25 folds for the final numbers.** No graph until the tabular features show lift.
- **Then**: read-only look at `answers/r4.2-*` (per-event labels?) and a label-sensitivity check; ablation of `n_new_pcs` and single history features.
- **Later (only if the new data justifies it)**: a graph with email/web/device/file structure; error analysis; Neo4j + Streamlit; report; optional r5.2 re-run.

## 7. Open questions for the owner
- Per-week alert budget: what review capacity is realistic (weeks or users per week)?
- (Decided in step 4) New users: skip their first 4 active weeks for every model.
- Should fixed model hyperparameters move from notebooks into `config.yaml`?
- Is the user-based split enough, or do we also want a time-based holdout?
- Email/web: device/file exist for only ~265 users, while email/web probably cover everyone (unverified): how should missing activity be encoded?
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
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/04_weekly_baseline_v2.ipynb  # ~6.5 min
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/05_gnn_lite.ipynb  # ~15 min with saved scores/pilot; ~90 min with RERUN_CV / RERUN_PILOT = True
```
