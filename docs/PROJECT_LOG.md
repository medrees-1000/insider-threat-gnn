# Project log

Last updated: end of step 7 stage 1 (time split 6g: HOLDS UP / double holdout DEGRADES; graph v2 stage 1 6h: gate NOT passed, no stage 2). Step 6 verdict (6b-6e) stays AMBIGUOUS. Everything below comes from the repo, notebook
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
- **Not in `data/raw/` until step 6 (now extracted, see 6b)**: `http.csv` and `email.csv` (at the end of step 5, verified by listing the folder; only logon, device, file, psychometric, LDAP, answers, readme and license were there). They are in the original archive
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
| `src/itgnn/commslogs.py` | email/web: streaming `aggregate_http`, `parse_domain`, `match_suffix`, `web_features`, `email_features`, approved category lists, `add_comms_features`, `comms_feature_sets` |
| `src/itgnn/comms_audit.py` | post-hoc audit helpers: feature groups, grouped permutation importance, truncation check, window-position helpers |
| `src/itgnn/comms_eval.py` | cached weekly frame with email/web features (`load_full`), time-split / double-holdout masks, user-cluster bootstrap, same-rows user-split reference |
| `src/itgnn/comms_graph.py` | graph v2 (user-PC, user-domain, email edges, domain-week nodes), LDAP address map, domain exclusions, degree-preserving shuffle, weighted adjacency, v2 pilot; `weekly_gnn.py` gained an opt-in `wsage` conv |
| `notebooks/08_time_split.ipynb`, `09_graph_v2_b0.ipynb`, `10_graph_v2_stage1.ipynb` | step 7: time-based evaluation, graph B0 checks, pilot + stage 1 |
| `notebooks/07_email_web_25fold.ipynb` | 25-fold run of W2, W2+calendar, W2+E+H, generic with RF/XGB, paired differences, alert budget |
| `notebooks/06_email_web_probe.ipynb` | step 6: 7 feature sets x RF/XGB, repeat 0, paired differences, alert budget, decision rule |
| `notebooks/05_gnn_lite.ipynb` | light GraphSAGE vs same network without edges vs RF/XGBoost W2 (pilot, 25 folds, budgets, per scenario, seed check) |
| `tests/` | 50 tests (32 earlier + 13 in `test_commslogs.py` incl. audit/time-split helpers + 5 in `test_comms_graph.py`: domain parsing, suffix matching, web/email features incl. causality, streaming, history, feature-set membership) on synthetic data (smoke, evaluation, extended features, windows, weekly features incl. causality, weekly graph/GNN, alert budget); the alert-budget regression test skips if `results/weekly_v2_oof_scores.parquet` is missing |

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

## 6b. Step 6 (option B): email/web probe. PRE-REGISTRATION, written BEFORE any evaluation
Written 2026-11-24 after extraction and profiling, before any model was fit on email/web features. Nothing below was changed after seeing results; any later deviation is listed under "Deviations" at the end of step 6's results.

**Extraction (done).** One pass `bzip2 -dc < r4.2.tar.bz2 | tar -xv --wildcards '*/email.csv' '*/http.csv'` into `/home/nothing/r42_extract` (outside the repo), tar exit 0, empty `.err`, exactly two regular files, then `mv -n` into `data/raw/` (`email.csv` 1,362,101,939 bytes; `http.csv` 14,536,257,467 bytes). The temp dir was empty afterwards and was deleted. Real rate: about 3.2 MB/s compressed in (~10 MB/s out); the pass took about 24.5 min. bzip2's own exit status was not captured (a truncated archive would have failed tar; all 4.82 GB were read).

**Profile (done, read-only).**
| | email.csv | http.csv |
| --- | --- | --- |
| columns | id, date, user, pc, to, cc, bcc, from, size, attachments, content | id, date, user, pc, url, content |
| rows | 2,629,979 | 28,434,423 |
| date range | 2010-01-02 07:11:45 .. 2011-05-16 21:16:26 | 2010-01-02 06:55:16 .. 2011-05-16 21:23:10 |
| users | 1,000 (all in logon; no logon user missing; 70/70 insiders present) | 1,000 (same) |
| PCs not in logon | 0 | 0 of 1,000 |
| duplicate ids / null cells / unparsable URLs | 0 / 0 / n.a. | 0 / n.a. / 0 |
- Email: sorted by date; `attachments` is an integer count stored as text (0 in 79% of mails); recipients per mail median 2, 99% <= 6, max 17; bcc present in 16%; 62% of `from` addresses are `dtaa.com`, the rest personal webmail; a user has a median of 2 `from` addresses (max 72).
- Web: 475 distinct domains (`domain` = lowercased host, leading `www.` stripped), 73 weeks, user-week-domain aggregate of 4,026,477 rows (`data/processed/http_user_week_domain.parquet`); only 9 of 475 domains are visited by fewer than 5 users over the whole period. Streamed with pyarrow, `newlines_in_values=True`, content never kept; full pass 98 s, peak memory 3.6 GB. Duplicate-id check uses 64-bit hashes. Row order of `http.csv` was not checked (the aggregate does not depend on it).
- Rows per user (q25 / median / q75), descriptive: email insiders 489 / 1,796 / 2,515 vs normal 1,002 / 3,013 / 3,712; web insiders 4,932 / 19,608 / 27,442 vs normal 10,034 / 32,871 / 39,444. No zero-row users in either group, so there is no "has any email/web" coverage shortcut. Insiders have fewer rows, which is consistent with the tenure effect (they leave early).
- Cached: `data/processed/email.parquet` (content dropped), `http_user_week_domain.parquet`, `http_top50_domains.csv`, `http_parts/`.

**Features (all past-only; week w uses events <= w; first-seen and rarity as described).**
- Email, per user-week: `em_n`, `em_n_after_hours` (hour outside 07-18, same rule as logon), `em_n_recip` (to+cc+bcc entries), `em_n_ext_recip` (recipient domain != dtaa.com), `em_max_recip` (largest single mail), `em_n_bcc` (mails with a bcc), `em_n_attach` (mails with attachments > 0), `em_size_sum`, `em_n_new_recip` (addresses the user never mailed in an earlier week), `em_n_from_external` (sender domain != dtaa.com), `em_n_competitor_recip` (recipients at a competitor domain).
- **Decision: `em_n_from_external` replaces the planned `em_n_from_mismatch`**, because a user has a median of 2 sender addresses, so a mismatch rule would flag a large share of normal mail. Consequence: scenario 3's "sent as the supervisor" has **no direct feature** (there is no user-to-address map); only `em_max_recip` and `em_size_sum` can carry it.
- Web, per user-week: `wb_n`, `wb_n_after_hours`, `wb_n_domains`, `wb_n_new_domains`, `wb_share_rare` (share of visits to domains visited by fewer than 5 distinct users before week w; fixed, not tuned; a never-seen domain counts as rare), plus category counts `wb_n_leak`, `wb_n_job`, `wb_n_cloud`, `wb_n_download`, `wb_n_competitor`.
- Each new column also gets the step-4 history recipe (`hist_mean_*` over the previous 4 active weeks, `dev_*`). Same weekly rows, same warm-up drop as W2 (63,238 rows expected; asserted against the saved W2 scores). Events in weeks with no Logon are dropped and counted; users with no events get 0.
- **Category lists (approved by the owner; matching is exact domain or `.`-suffix, never substring).** Decided from the unlabeled domain vocabulary and the scenario text, before any insider/label lookup:
  - leak: `wikileaks.org`
  - job: `careerbuilder.com`, `indeed.com`, `monster.com`, `simplyhired.com`, `job-hunt.org`, `jobhuntersbible.com`, `taleo.net`
  - cloud: `dropbox.com`, `mediafire.com`, `yousendit.com`, `4shared.com`, `megaupload.com`, `filesonic.com`, `fileserve.com`
  - download (keylogger/spy software vendors, judged by name only; no site was opened): `refog.com`, `dailykeylogger.com`, `actualkeylogger.com`, `keylogpc.com`, `spectorsoft.com`, `webwatchernow.com`, `best-spy-soft.com`, `softactivity.com`
  - competitor (fifth list, added by the owner before any labels were seen; used for web visits `wb_n_competitor` and email recipients `em_n_competitor_recip`): `lockheedmartin.com`, `northropgrumman.com`, `raytheon.com`, `boeing.com`, `harris.com`
  - **Deliberately out** (deciding on them from labels would be leakage; they may be looked at post-hoc, labelled as such, only after the result is final): `relytec.com`, `wellresearchedreviews.com`, `blackhatworld.com`, `download.cnet.com`, `custhelp.com`. Also out: `linkedin.com`, `craigslist.org`, freelance sites, `amazonaws.com`, `googleusercontent.com`, `filestube.com`, torrent sites.
  - **Caveat on these lists:** `wikileaks.org` has exactly 30 users and scenario 1 has 30 insiders; the keylogger domains have 1-8 users and scenario 3 has 10 insiders. Overlap with insiders was NOT checked. A scenario-1 gain is therefore expected to be easy in this synthetic dataset.

**Feature sets (repeat 0 only, 5 folds, RF and XGBoost, same fixed hyperparameters as notebook 04, 70 fits).**
1. W2 (reference, re-run)
2. W2+E (all email features incl. `em_n_competitor_recip`)
3. W2+H (all web features incl. the five category counts)
4. W2+E+H (**primary**)
5. W2+E+H generic: all email/web features EXCLUDING the named-site features (`wb_n_leak`, `wb_n_job`, `wb_n_cloud`, `wb_n_download`, `wb_n_competitor`, `em_n_competitor_recip`) and their history/deviation columns
6. W2+E+H no-first-seen: W2+E+H minus `em_n_new_recip`, `wb_n_new_domains`, `wb_share_rare` (and their history/deviation columns)
7. W2 + calendar week (negative control)
Metrics: PR-AUC, ROC-AUC, tie-aware precision/recall@100 and the alert budget at q = 0.25/0.5/1/2%, per scenario; paired fold-by-fold difference vs W2 on identical folds. Descriptive burn-in check from the saved predictions (no new fits): PR-AUC restricted to calendar week >= 13, because `wb_share_rare` is unreliable while the domain history is nearly empty. Reproduction check: W2 repeat-0 PR-AUC per fold must match the saved scores in `results/weekly_v2_oof_scores.parquet`.

**Decision rule (fixed before running).** Primary = RF, W2+E+H vs W2, PR-AUC.
- **GO** (the email/web logs carry signal), all of: (1) RF mean delta PR-AUC >= +0.05; (2) >= 4 of 5 folds positive; (3) XGBoost replicates: mean delta >= +0.03 and >= 4 of 5 folds positive; (4) at q = 0.5% RF W2+E+H catches >= 3 more insiders than RF W2 with <= 2 extra normal users flagged (q = 1% is descriptive); (5) the calendar-week control does not lift (RF delta < +0.01).
- **AMBIGUOUS**: RF mean delta in [+0.02, +0.05), or only 3 of 5 folds positive, or the RF delta is >= +0.05 but another GO criterion fails. Then no graph yet; ask the owner whether to spend the full 25-fold run.
- **NO-GO**: RF mean delta < +0.02. Record and move on.
- **What GO means:** the email/web logs carry signal, NOT that a graph helps. If we get there, graph models must be compared against a tabular model on the SAME features plus the no-edge MLP control.
- **Interpretation of the generic set** (RF mean delta of "generic" vs W2): if GO holds for W2+E+H but generic is < +0.02, the finding is "named-site features help; generic email/web behaviour does not", and it is NOT a reason to build a graph; the final tabular model gets the named-site features and a GNN is then run on the same features with the no-edge MLP control. If generic ALSO has RF mean delta >= +0.03, that is the case for a relational email/web graph. (0.02-0.03: grey zone, reported without a conclusion.)
- **Leakage audit:** if W2+E+H PR-AUC exceeds 0.80, audit before believing it, including the per-scenario breakdown, feature importances and the generic ablation (a named-site marker can legitimately explain a large jump). Per-scenario results are reported for every set.
- After the result is final (not before): a labelled post-hoc check of the overlap between category-domain visitors and insiders, and a look at the five excluded domains.
- Runtime: if the projected run exceeds ~15 min, stop and ask.

### 6c. Step 6 results (notebook 06, repeat 0 only = 5 folds; fixed notebook-04 hyperparameters; 70 fits in 103 s)
Rows identical to the saved W2 run (63,238 user-weeks, 322 attack weeks); W2 re-run reproduces the saved repeat-0 scores exactly (PR-AUC diff 0.000 per fold, RF and XGB). Every user-week has email and web activity (coverage 1.000 in both groups; 0 user-weeks with events but no Logon row).
| PR-AUC (mean ± std over 5 folds) | RF | XGB |
| --- | --- | --- |
| W2 | 0.499 ± 0.072 | 0.465 ± 0.097 |
| W2+E | 0.575 ± 0.122 | 0.566 ± 0.107 |
| W2+H | 0.772 ± 0.101 | 0.875 ± 0.075 |
| **W2+E+H (primary)** | **0.743 ± 0.098** | **0.879 ± 0.080** |
| W2+E+H generic | 0.675 ± 0.127 | 0.824 ± 0.112 |
| W2+E+H no-first-seen | 0.709 ± 0.094 | 0.877 ± 0.086 |
| W2+calendar (control) | 0.590 ± 0.105 | 0.565 ± 0.114 |
Paired difference vs W2 (mean ± std, folds positive): RF W2+E+H +0.244 ± 0.091 (5/5), XGB +0.414 ± 0.126 (5/5); RF W2+H +0.273 (5/5), W2+E +0.076 (4/5); generic RF +0.176 (4/5), XGB +0.359 (5/5); no-first-seen RF +0.210, XGB +0.412; **calendar control RF +0.091 (5/5), XGB +0.100 (5/5)**.
- **Verdict by the pre-registered rule: AMBIGUOUS.** Criteria 1-4 are met (RF +0.244, 5/5 folds; XGB +0.414, 5/5; q = 0.5% RF catches 58 vs 54 insiders with 6 vs 10 normal users flagged), but **criterion 5 fails**: adding calendar week to W2 lifts RF by +0.091 and XGB by +0.100, far above the +0.01 limit. No graph decision follows from this; the owner decides whether to spend the full 25-fold run.
- Reading (my interpretation): web features lift well above the calendar effect (RF +0.27, XGB +0.41 vs +0.09 / +0.10); email alone (+0.08 RF, +0.10 XGB) is **not distinguishable from the calendar control**. Why calendar week helps on top of W2 is **unexplained** (hypothesis, unverified: attacks cluster in time).
- The notebook prints "case for a relational graph" for generic RF +0.176 vs the +0.03 rule; that line is mechanical. The rule applies only if GO holds, which it does not, so it is not a conclusion. Generic XGB is clearly above the calendar control; generic RF only moderately.
- W2+E+H RF PR-AUC 0.743 < 0.80: the leakage-audit trigger did not fire. Burn-in check (calendar week >= 13) leaves every attack week (322 of 322) in, so it removes only quiet weeks and changes the deltas little (RF W2+E+H -W2 +0.216, XGB +0.379). `wb_share_rare` prints 0.000 in every calendar-week block (3 decimals; not checked whether exactly constant), so it is practically unused.
- Alert budget (one repeat; caught of 70 / normal flagged of 930): RF W2 q=0.25% 49/3, 0.5% 54/10, 1% 60/19, 2% 65/41; RF W2+E+H 41/2, 58/6, 66/20, 70/67. Mixed for RF: worse at q = 0.25% (41 vs 49 caught), better at 0.5-2% in insiders caught, but more normal users flagged at 2% (67 vs 41). XGB W2+E+H: 56/0, 62/7, 68/27, 69/51 vs XGB W2 51/2, 53/5, 58/12, 62/36.
- Per scenario (caught at q = 1%, of 30/30/10): RF W2 28/26/6 -> W2+E+H 30/29/7; XGB W2 27/27/4 -> W2+E+H 30/30/8 (full tables for every set and budget are in the notebook). The scenario-1 gain is expected to be easy (`wikileaks.org` has exactly 30 users, scenario 1 has 30 insiders; 30 users and 52 user-weeks visit it in the model rows). Scenario 3 has only 10 insiders.
- Descriptive RF impurity importances (fit on fold-0 training users; biased towards continuous columns): top are W2 device/file columns; named-site columns hold 19.8% of total importance (`hist_mean_wb_n_competitor` 6th at 0.046).
- Category coverage in the model rows: wikileaks 30 users / 52 user-weeks, job 902 users, cloud 849, download 9 users / 9 user-weeks, competitor web 513, competitor email recipients 999. Overlap with insiders NOT checked (post-hoc, after the result is final).
- Not run: 25 folds, post-hoc overlap analysis, the five excluded domains, any chart (notebook 06 has tables only). Deviations from the plan: none after the pre-registration. Charts: none.

### 6d. Post-hoc audit and diagnostics (owner-approved after 6c; labelled post-hoc; section 6b is not edited and the verdict stays AMBIGUOUS)
- **Correction to 6c:** the leakage-audit rule (PR-AUC > 0.80) applies to ANY model, not only RF. XGBoost W2+E+H = 0.879 and XGBoost W2+H = 0.875 exceed 0.80, so the audit IS triggered; my 6c/report applied it to RF only. The audit is run BEFORE the 25 folds. If it finds a leak, the 25-fold run does not happen.
- Plan (owner's order): (a) causality test + empirical truncation check on 200 user-weeks incl. 30 attack weeks; (b) grouped permutation importance on held-out folds; (c) leave-one-group-out refits; (d) where the model fires (by position in the window, window length, mean score before/inside/after the window); (e) is `wb_share_rare` exactly constant (if so it is dropped, post-hoc); POST-HOC: (f) attack weeks and W2 features by calendar week, (g) W2+E+H vs W2+calendar and a W2+calendar+E+H set, (h) category-domain visitors vs insiders by scenario, (i) the five excluded domains (report only; they stay out of every model). Then, only if no leak: 25 folds (5x5) for W2, W2+calendar, W2+E+H, W2+E+H generic with RF and XGB.
- **Note for the final report:** CV here splits by user, not by time. Calendar week looks easier under user-split CV (the calendar control lifts W2 by about +0.09/+0.10); a time-based holdout would answer whether the gains survive. **Not run now** (owner decision).

**6d results (notebook 06 part 2, repeat 0, all POST-HOC; charts in `results/figures/06_*.png`, all three opened).**
- **(a) Causality: no future information found.** Existing tests: `test_causality_and_positive_control` (weekly features, edits in weeks 6-7 leave earlier weeks identical) and the causality assertions in `test_commslogs.py` (web rarity and email first-seen). New empirical check: 200 user-weeks (30 attack weeks) from 15 random calendar weeks (to bound runtime), features recomputed from logs truncated at the END of each week (web rarity keeps all users; logon/email restricted to the sampled users): max |difference| over all 79 W2+E+H columns = 0.0, 0 of 200 rows differ. Positive control (week cut after 3 days): 18 of 18 rows differ (`em_size_sum`), so the check can fail. This checks the features, not the label definition.
- **(b) Grouped permutation importance** (PR-AUC drop, 5 shuffles, held-out folds, mean ± std over 5 folds; RF / XGB): history/deviation 0.643 / 0.701, named-site 0.553 / 0.627, W2 base 0.258 / 0.560, web volume 0.047 / 0.171, email recipients/attachments 0.072 / 0.075, first-seen 0.031 / 0.048, web after-hours 0.026 / 0.008, email volume 0.001 / 0.007, web domain counts -0.002 / 0.005. Groups overlap (each family includes its own hist/dev columns; "history/deviation" is cross-cutting), and correlated groups understate each other.
- **(c) Leave-one-group-out** (delta PR-AUC vs W2+E+H, RF / XGB; negative = the group helped): history/deviation -0.157 / -0.211 (5/5 folds worse in both), named-site -0.068 / -0.055 (4/5, 5/5), web volume -0.031 / -0.052, first-seen -0.033 / -0.002; email groups, web after-hours, web domain counts and W2 base about 0 or slightly positive. Named-site columns look huge under permutation (0.55-0.63) but cost only 0.055-0.068 when removed: the generic web features carry much of the same information (redundancy).
- **(d) Where it fires** (recall of attack weeks at q = 1%, RF W2+E+H / XGB W2+E+H): first week 0.80 / 0.94, middle 0.87 / 0.93, last 0.88 / 0.92, single-week windows 0.85 / 0.90 (W2: RF 0.68 / 0.55 / 0.70 / 0.70, XGB 0.58 / 0.56 / 0.64 / 0.60); by window length: <=7 d 0.93 / 0.95, 8-14 d 1.00 / 1.00, >30 d 0.84 / 0.92 (260 of 322 attack weeks are in windows > 30 days). Detection is not concentrated at the end of windows. Mean within-fold score percentile (RF W2+E+H / XGB W2+E+H): 28 days before the window 0.70 / 0.76, inside 0.994 / 0.996, 28 days after 0.92 / 0.63 (only 91 rows, 42 insiders), normal users 0.49 / 0.49. Insiders already rank above normal users before the window starts (partly user-level differences such as device use); the RF's elevated rank after the window is not explained.
- **(e) `wb_share_rare` is not constant** (44 unique values; > 0 in 114 of 63,238 modelled user-weeks = 0.18%, max 0.02; more often in the first calendar weeks), so it is KEPT. It is practically unused.
- **Verdict of the audit: no leakage found.** No future information (a), detection starts at the first attack week (d), and the large lift comes from named-site markers plus history/deviation (b, c). That named-site features are near label markers in this dataset is NOT a leak but is stated plainly: see (h). The 25-fold run therefore went ahead.
- **(f) Why the calendar control lifts (data fact + inference).** No attack week exists before calendar week 23: the first 20 calendar weeks hold 14,871 rows (23.5% of all rows) and 0 attack weeks; attack weeks span calendar weeks 23-69 (scenario 1: 27-61, scenario 2: 24-67, scenario 3: 23-69), 46 of 69 weeks have at least one, and the busiest 10 weeks hold 36%. Prevalence by 10-week block: 0.000, 0.000, 0.004, 0.011, 0.009, 0.006, 0.003. W2 features also drift with calendar week (mean `n_new_pcs` 0.59 -> 0.16, `n_after_hours` 0.55 -> 0.38; Spearman rho with calendar week only -0.02 to -0.07). Inference (not ablated): the trees use calendar week to discard the quiet early period. A user-split CV lets them; a time-based split would not. **Not run now.**
- **(g) Stricter baseline W2+calendar** (paired, 5 folds): RF W2+E+H - W2+calendar +0.153 ± 0.111 (5/5), XGB +0.314 ± 0.143 (5/5); generic - W2+calendar RF +0.085 ± 0.122 (4/5), XGB +0.259 ± 0.156 (5/5). W2+calendar+E+H: RF 0.754 ± 0.105 (+0.012 vs W2+E+H, 3/5), XGB 0.912 ± 0.065 (+0.033, 5/5). So the email/web lift survives the stricter baseline.
- **(h) Category visitors vs insiders (post-hoc).** `wikileaks.org`: 30 visitors, 0 normal users, all 30 are scenario-1 insiders, all visit inside their window (69 visits). The `download` list: 9 visitors, 0 normal users, all scenario-3 insiders (9 of 10), all inside the window (34 visits). Job-site domains: 906 visitors (839 normal, 67 insiders incl. all 30 scenario-2 and all 10 scenario-3 insiders); competitor domains: 513 visitors (463 normal, 50 insiders incl. all 30 scenario-2); cloud: 859 visitors (798 normal, 61 insiders). So wikileaks and the keylogger domains are near-perfect scenario-1 / scenario-3 markers in this dataset; the job/competitor/cloud lists are visited by roughly half or more of the normal users, so only volume can separate.
- **(i) The five excluded domains (report only; they stay out of every model):** `relytec.com`, `wellresearchedreviews.com`, `download.cnet.com`: 5 visitors each, all 5 are scenario-3 insiders, all inside their window; `blackhatworld.com` and `custhelp.com`: 5 visitors each, all normal users. Excluding the first three by name only was conservative and cost nothing in validity.
- **RF q = 0.25% (41 vs 49 caught, repeat 0): concentration, not worse ranking.** At the same 161 flagged weeks, RF W2 flags 111 attack weeks (precision 0.69, 49 distinct insiders) and RF W2+E+H 138 (precision 0.86, 41 distinct insiders; top-5 insiders supply 31% of flagged attack weeks vs 22%; most flagged weeks for one insider 9 vs 6). XGB W2+E+H: 159 of 161 flagged weeks are attack weeks (precision 0.99), 56 distinct insiders. The 25-fold run re-checks this.

### 6e. 25-fold run (notebook 07; 5x5 folds; sets W2, W2+calendar, W2+E+H, W2+E+H generic; RF and XGBoost; 200 fits in 420 s). DESCRIPTIVE: the pre-registered verdict (6b/6c) stays AMBIGUOUS.
Run only after the audit (6d) found no leakage. W2 over 25 folds reproduces the saved step-4 scores exactly (316,190 rows per model, max score difference 0.000). Folds are correlated (the 5 repeats re-use the same users), so stds understate uncertainty; CV splits by user, not by time.
| PR-AUC (mean ± std, 25 folds) | RF | XGBoost |
| --- | --- | --- |
| W2 | 0.503 ± 0.071 | 0.458 ± 0.080 |
| W2+calendar | 0.585 ± 0.091 | 0.555 ± 0.091 |
| W2+E+H generic | 0.661 ± 0.101 | 0.792 ± 0.086 |
| W2+E+H | 0.737 ± 0.125 | 0.867 ± 0.071 |
Paired PR-AUC differences (mean, folds positive of 25; range of the 5 per-repeat means): W2+E+H - W2: RF +0.234 (24/25; 0.174-0.281), XGB +0.409 (25/25; 0.392-0.422). Calendar control - W2: RF +0.082 (24/25), XGB +0.097 (25/25). **W2+E+H - W2+calendar: RF +0.153 (23/25; 0.103-0.191), XGB +0.313 (25/25; 0.300-0.330).** Generic - W2: RF +0.158, XGB +0.334; generic - W2+calendar: RF +0.076 (19/25), XGB +0.237 (25/25). W2+E+H - generic (value of the named-site features): RF +0.077 (19/25), XGB +0.076 (24/25).
Alert budget (mean ± std over 5 repeats; caught of 70 / normal users flagged of 930):
| q | RF W2 | RF W2+E+H | XGB W2 | XGB W2+E+H |
| --- | --- | --- | --- | --- |
| 0.25% | 51.4 / 3.2 | 37.8 / 0.4 | 48.0 / 2.8 | 52.8 / 0.2 |
| 0.5% | 56.6 / 9.2 | 53.4 / 4.2 | 52.6 / 5.4 | 61.6 / 6.6 |
| 1% | 61.2 / 18.2 | 65.8 / 25.0 | 56.6 / 10.2 | 67.6 / 25.6 |
| 2% | 65.4 / 35.6 | 69.4 / 71.0 | 62.6 / 30.2 | 69.0 / 58.6 |
(std and the W2+calendar / generic rows are in the notebook; per scenario at q = 1% (of 30/30/10): RF W2 28.8 / 26.4 / 6.0 -> W2+E+H 30.0 / 27.8 / 8.0; XGB W2 26.4 / 26.2 / 4.0 -> W2+E+H 30.0 / 29.4 / 8.2.)
- **The pre-registered GO criteria on 25 folds (descriptive): criteria 1-3 hold, 4 and 5 do not.** RF delta +0.234 in 24/25 folds, XGB +0.409 in 25/25; the calendar control still lifts (+0.082 / +0.097, rule < +0.01); and criterion 4 (RF at q = 0.5%: >= 3 more insiders caught with <= 2 extra normal flagged) is NOT met: RF W2+E+H catches 53.4 vs 56.6 (-3.2) while flagging 4.2 vs 9.2 normal users. The repeat-0 pass (+4 caught) did not hold up over 25 folds.
- **Reading:** the email/web lift is not just the calendar effect: it stays at +0.153 (RF) and +0.313 (XGB) over the stricter W2+calendar baseline. XGBoost benefits far more than RF, and in the alert budget XGBoost W2+E+H beats XGBoost W2 at every budget (q = 0.5%: 61.6 vs 52.6 caught). RF gains only at q >= 1%, at the price of more normal users flagged at 2% (71.0 vs 35.6). The generic set (no named sites) keeps +0.076 (RF, 19/25 folds) and +0.237 (XGB, 25/25) over W2+calendar; the pre-registered rule about the generic set applies only if GO holds, which it does not, so no relational-graph conclusion follows. If a graph is ever built, it must be compared against a tabular model on the SAME features plus the no-edge MLP control.
- **RF at q = 0.25% (51.4 -> 37.8 caught): explained by concentration (supported by the numbers, mechanism unverified).** Both flag about 160 weeks, but RF W2+E+H flags 137 attack weeks (precision 0.855, vs 112 / 0.697 for W2) on only 37.8 distinct insiders (vs 51.4): the most weeks flagged for one insider rise from 5.6 to 8.8 and the top-5 insiders' share from 0.21 to 0.30. Scenario 1 is hit hardest (15.4 vs 23.0 of 30 caught). XGBoost does NOT concentrate (precision 0.974, 52.8 distinct insiders). Why RF concentrates was not tested.
- Not run: time-based holdout (owner decision), tuning, logistic regression. Notebook 07 has no charts.

## 6f. Step 7: time-based evaluation (Part A) and graph v2 on W2+E+H features (Part B). PRE-REGISTRATION, written BEFORE any run (owner-approved plan)
Written 2026-11-24 after the owner approved the plan. All thresholds below are fixed now; any later deviation is listed under "Deviations" in the results. Order: Part A, then Part B stage 1 (repeat 0), then STOP and report against the gate. **Stage 2 (more folds) needs the owner's approval. Nothing is committed.**

**Part A: forward-chaining time split (RF and XGBoost, fixed hyperparameters of notebook 04, calendar week NOT a feature).**
- Train = user-weeks with calendar week <= T, test = weeks > T; the same users appear on both sides. Sets: W2, W2+E+H, W2+E+H generic. Cutoffs T = 29, 39, 49 (calendar week 0 = week of 2010-01-04). From notebook 06's 10-week blocks: test attack weeks 279 / 173 / 94, train attack weeks 43 / 149 / 228 (T = 29 is thin on the training side). **T = 59 is dropped** (38 test attack weeks, below ~40). Distinct insiders per block are counted in A0 (not available before).
- Metrics: PR-AUC, ROC-AUC, alert budget at q = 0.25/0.5/1/2% of the test rows (caught = distinct insiders with a flagged attack week in the test block), per scenario, cluster bootstrap by test user (200 resamples, seed 42) for PR-AUC and for the W2+E+H - W2 difference.
- **Reference** (same rows, user-split): from the saved 25-fold scores (`results/comms_25fold_oof_scores.parquet`), per repeat pool the test predictions, take the PR-AUC on rows with calendar week > T, average over the 5 repeats.
- **Reading rule.** R = time-split PR-AUC / same-rows user-split PR-AUC for XGBoost W2+E+H. DOES NOT HOLD if R < 0.60 at >= 2 of 3 cutoffs OR the W2+E+H - W2 PR-AUC lift is < +0.05 at >= 2 of 3 cutoffs. HOLDS UP (checked next) if R >= 0.85 at >= 2 of 3 cutoffs AND the lift is >= +0.10 at >= 2 of 3 cutoffs. Otherwise DEGRADES (partly calendar-driven). RF, the generic set, the alert budget and the bootstrap intervals are read descriptively. Why 0.85 / 0.60: the user-split fold std of XGBoost W2+E+H is 0.071 (about 8% of 0.867), so a 15% loss is about two fold-stds and a 40% loss is far outside.
- **Double holdout (owner-approved, tree models only):** repeat-0 folds, train = training-fold users x weeks <= T, test = test-fold users x weeks > T. The pooled test block (every test user once) has the same attack weeks as the time split (>= 40 at every kept cutoff), so the same reading rule applies to it as pass/fail on the pooled predictions; per-fold test blocks are smaller and their attack weeks are reported (descriptive).

**Part B: does a graph add anything on top of the W2+E+H features? Stage 1 = repeat 0 (5 folds).**
- Graph v2 (V1), one disjoint graph with same-week edges only: user-PC (existing logon edges, weight 1), user-domain (per user-week the top K = 5 domains by visits that week, ties by domain name; weight log1p(visits)), user-user email (per week, sender -> recipient, top K = 5 recipients per sender by mails, weight log1p(mails), made undirected), plus domain-week nodes with week-only features (is_domain, log1p(users that week), log1p(visits that week), after-hours share). User-week node features = the 79 W2+E+H columns (log1p for non-negative, signed log for `dev_*`, scaler fit on training-fold users only) + has_history; PC nodes: type flag + log(users that week). Warm-up rows are neighbours only (as in step 5).
- **Domain exclusions for the edges:** the 28 category-list domains, the 5 excluded domains, and every domain with fewer than 20 distinct users over the whole period (unlabeled counts only). The number of domains and aggregate rows removed is reported. The category domains appear only in the separate marker variant (M3), which is NOT run in this step.
- **K = 5**; lowered (to 3) only if the timing probe says stage 1 cannot fit; never changed after any fold result. K used and edge counts are recorded.
- **Email edges depend on the address-map check.** The map is a static identity table: the LDAP `email` / `user_id` columns (18 monthly files), never behaviour from later weeks. Pass if BOTH the share of distinct internal (`@dtaa.com`) recipient addresses and the share of internal recipient entries that map to exactly one user are >= 90%. If it fails, email edges are dropped, the log says so, and the graph runs without them. The map is used for these edges only, never for a feature that looks at weeks after w.
- **Weighted aggregation:** a new opt-in `wsage` conv (weighted mean over neighbours through a row-normalised sparse matrix, same self/neighbour linear maps as SAGEConv); the existing `sage` / `mlp` paths are unchanged. A unit test checks `wsage` with unit weights equals `SAGEConv`.
- **Models on the same folds/rows:** M0 = no-edge MLP on identical features; M1 = GraphSAGE (`wsage`) on V1; M2 = GraphSAGE on a degree-preserving SHUFFLE of the same edges (within each week, the far endpoint and weight of each edge type are permuted together; self-loops dropped and their share reported). Same recipe as step 5 (2 layers, hidden 64, dropout 0.3, Adam lr 0.01, weight decay 5e-4, sqrt class weight capped at 50, last-epoch model, one seed per fold). Epoch counts come from one pilot (same method as step 5: repeat-0 fold-0 training users, 3 inner splits, <= 300 epochs, trailing-5 smoothing, earliest epoch within 0.01 of the best) for MLP and for GraphSAGE V1; M2 reuses GraphSAGE's. If the pilot picks the 300-epoch cap, it is recorded as possibly under-trained and not extended (no tuning).
- **Comparators:** XGBoost W2+E+H and XGBoost generic, taken from the saved 25-fold scores restricted to the repeat-0 folds (same folds, same rows).
- **GATE for stage 2 (fixed now):** GraphSAGE (M1) must beat BOTH M0 and M2 by a mean PR-AUC margin >= +0.05 AND the margin must be positive in at least 4 of 5 folds, for each of the two comparisons. (+0.05 is about 1.2x the sd of a seed-noise difference, 0.043, from step 5: GraphSAGE +/- 0.036, MLP +/- 0.023 on repeat-0 fold 0; the step-5 features differ, so this is a rough guide.)
- **Decision rule.** Gate fails -> edges add nothing detectable over the W2+E+H node features; XGBoost stays the model; no stage 2; the graph stays for Neo4j/UI exploration only. M1 beats M0 but not M2 -> any neighbour averaging helps, no structural claim. Gate passes -> (with owner approval) stage 2: compare M1 with XGBoost W2+E+H on the same folds: >= +0.02 mean PR-AUC in >= 80% of folds = "graph adds on top of tabular" (then edge-type ablations, the marker variant, a GNN time split); within +/-0.02 = tie, tabular preferred; below -0.02 = relational signal exists but tabular is better. Beating XGBoost generic but not W2+E+H = the graph recovers part of the named-site information. Alert budget at q = 0.5% and 1% is secondary.
- **Stage-2 cut rule (if approved later):** the cap is about 2 h of total compute; if 25 folds would exceed it, cut REPEATS first (e.g. 3 repeats = 15 folds), keep every control (never drop the MLP or the shuffled-edge control), and STOP and tell the owner before cutting anything.
- **Stop conditions:** if the timing probe projects stage 1 above ~25 min even at K = 3, or the pilot + stage 1 above ~60 min in total, stop and ask.
- **Edge-building leak checks:** (i) unit test: events added in week w+1 leave every edge and weight with week <= w unchanged (with a positive control: an edit in week w changes week w's edges); no edge joins different weeks; (ii) empirical check on real data (logs truncated at the end of 15 sampled weeks, edges for sampled users compared with the full run); (iii) shuffle test: degrees and weights preserved, no cross-week edges, labels untouched.
- Not in this step: the marker variant M3, stage 2, any commit, any tuning. Unverified until measured: time per epoch, memory, CSR autograd on this torch build (2.15.0+cpu, PyG 2.8.1).

### 6g. Step 7 results, Part A (notebook 08; 18 time-split fits + 90 double-holdout fits, about 150 s; calendar week is not a feature; 25-fold scores used as the user-split reference)
Blocks (weekly rows after the warm-up): T = 29: train 24,581 rows / 43 attack weeks / 17 insiders, test 38,657 / 279 / 62 insiders (26 s1, 30 s2, 6 s3), 9 "seen" insiders with attack weeks on both sides; T = 39: train 149 / 36 insiders, test 173 / 39 insiders (16/19/4), 5 seen; T = 49: train 228 / 53 insiders, test 94 / 21 insiders (8/12/1), 4 seen; T = 59: 38 test attack weeks / 9 insiders (dropped, as pre-registered).
PR-AUC on the test block (time split, same users; [95% user-bootstrap]); R = time split / user-split on the same rows:
| T | set | RF | XGBoost |
| --- | --- | --- | --- |
| 29 | W2 | 0.248 [0.175, 0.349] | 0.206 [0.148, 0.283] |
| 29 | W2+E+H | 0.510 [0.406, 0.647] (R 0.69) | 0.525 [0.419, 0.628] (R 0.60) |
| 29 | generic | 0.370 | 0.412 |
| 39 | W2 | 0.579 | 0.560 |
| 39 | W2+E+H | 0.727 [0.591, 0.885] (R 0.97) | 0.778 [0.654, 0.909] (R 0.85) |
| 39 | generic | 0.637 | 0.696 |
| 49 | W2 | 0.684 | 0.673 |
| 49 | W2+E+H | 0.950 [0.888, 0.990] (R 1.09) | 0.938 [0.857, 0.994] (R 0.97) |
| 49 | generic | 0.855 | 0.872 |
Lift W2+E+H - W2 (paired user-bootstrap 95% interval): XGBoost +0.319 [0.205, 0.421] / +0.218 [0.089, 0.346] / +0.265 [0.151, 0.397] at T = 29 / 39 / 49; RF +0.262 [0.134, 0.402] / +0.148 [0.029, 0.274] / +0.266 [0.156, 0.424]. Generic - W2: XGBoost +0.206 / +0.136 [0.002, 0.225] / +0.200; named-site value (W2+E+H - generic) XGBoost +0.113 [0.043, 0.175] / +0.081 [-0.016, 0.232] / +0.065 [-0.022, 0.162].
- **Pre-registered reading, time split (same users): HOLDS UP.** XGBoost W2+E+H R = 0.598 / 0.853 / 0.970 (>= 0.85 at 2 of 3 cutoffs; < 0.60 only at T = 29, which has just 43 training attack weeks) and the lift is >= +0.10 at all three cutoffs. Caveat: R at T = 39 clears 0.85 by 0.003, and these are single splits (one train/test split per cutoff).
- **Pre-registered reading, double holdout (unseen users x future weeks; pooled test blocks have >= 40 attack weeks so it is pass/fail): DEGRADES.** XGBoost W2+E+H pooled PR-AUC 0.391 / 0.726 / 0.901, R = 0.444 / 0.797 / 0.933 (>= 0.85 only at T = 49), lift +0.215 / +0.207 / +0.286. Per-fold test blocks are thin (T = 49: 33 / 6 / 28 / 3 / 24 attack weeks), so per-fold numbers are descriptive.
- Alert budget (time split, XGBoost; caught of the insiders with attack weeks in the test block / normal flagged of 930): T = 29 (62 insiders) at q = 0.5%: W2 40 / 16, W2+E+H 33 / 19; T = 39 (39 insiders): W2 31 / 7, W2+E+H 34 / 5; T = 49 (21 insiders): W2 19 / 8, W2+E+H 21 / 5. Mixed at T = 29: scenario-1 insiders caught by XGBoost W2+E+H at q = 0.5% were 4 of 26 vs 17 for W2, and at T = 39 15-16 of 16. Inference (not tested): when the training period holds almost no scenario-1 attack weeks (scenario 1 spans calendar weeks 27-61), the model cannot learn the `wikileaks.org` marker; it works once the training period contains such attacks.
- Outputs: `results/time_split_oof_scores.parquet`, `results/double_holdout_oof_scores.parquet`. Notebook 08 has tables only (no charts).

### 6h. Step 7 results, Part B stage 1 (notebooks 09 and 10; repeat 0 = 5 folds, one seed). GATE NOT PASSED: stage 2 is not run.
**B0 (notebook 09, about 70 s).**
- **Address map:** the LDAP `email`/`user_id` table has 1,000 addresses for 1,000 users, none ambiguous; 100% of the 3,547,274 internal recipient entries and 100% of the 1,000 distinct internal addresses map to exactly one user (rule: both >= 90%), so the **email edges are kept**. 58.2% of all recipient entries are internal.
- **Domain exclusions:** of 475 domains, 38 are dropped (28 category-list, 5 excluded, 18 with fewer than 20 distinct users; the union is smaller than the sum because the lists overlap), 437 kept; 232,068 of 4,026,477 aggregate rows removed (5.8%).
- **Graph at K = 5** (K kept at 5: the probe projected stage 1 at 24.2 min and the pilot at 16.9 min, within the pre-registered 25 / 60): 67,238 user-week nodes, 67,375 PC-week nodes, 24,999 domain-week nodes; undirected edges: user-PC 112,408, user-domain 336,159, user-user email 322,392; 1,452,830 non-zero entries in the symmetric weighted adjacency; the shuffle drops 0.11% of the email edges as self-loops. Probe (20 threads): 0.376 s/epoch for the MLP, 0.489 s/epoch for the weighted GraphSAGE; peak memory 4.06 GB.
- **Edge-causality check on real data: passed.** For 15 sampled weeks, the graph rebuilt from logs truncated at the end of the week has edges (type, user, other node, weight) and domain-week node features identical to the full run (0 weeks with any difference). Unit tests (`tests/test_comms_graph.py`) cover `wsage` = `SAGEConv` with unit weights, CSR autograd, same-week-only edges, exclusions, causality with a positive control, shuffle degree preservation, address coverage.
**Pilot (notebook 10, 16.7 min; repeat-0 fold-0 training users, 3 inner splits, <= 300 epochs):** chosen epochs **MLP 64, GraphSAGE 75** (step 5 used 188 / 224), neither at the 300-epoch cap; best mean inner-validation PR-AUC 0.656 (MLP) and 0.660 (GraphSAGE). The curves (chart opened) rise until about epoch 50, then stay flat and noisy around 0.62-0.65 to epoch 300, with no sign of the late creeping seen in step 5, so under-training is not indicated; the early plateau and the earliest-within-0.01 rule choose small counts. Epochs stored in `configs/config.yaml` (`gnn2`).
**Stage 1 (9.9 min of training over 15 runs; PR-AUC per fold, mean ± std over 5 folds):**
| model | PR-AUC | ROC-AUC |
| --- | --- | --- |
| M0 no-edge MLP | 0.708 ± 0.155 | 0.967 |
| M1 GraphSAGE (V1) | 0.718 ± 0.176 | 0.973 |
| M2 GraphSAGE, shuffled edges | 0.717 ± 0.167 | 0.981 |
| XGBoost W2+E+H (same folds) | 0.879 ± 0.080 | 0.998 |
| XGBoost generic | 0.824 ± 0.112 | 0.995 |
| RF W2+E+H | 0.743 ± 0.098 | 0.997 |
Paired (mean diff, folds positive of 5): M1 - M0 **+0.010** (3/5; folds +0.003, +0.014, +0.057, -0.006, -0.021); M1 - M2 **0.000** (3/5); M2 - M0 +0.009 (2/5); M1 - XGBoost W2+E+H **-0.161** (0/5); M1 - XGBoost generic -0.106 (1/5); M1 - RF W2+E+H -0.025 (2/5); M0 - XGBoost W2+E+H -0.170 (0/5).
- **Gate (pre-registered): FAILED.** M1 - M0 mean +0.010 (needs >= +0.05; positive in 3 of 5 folds, needs 4) and M1 - M2 mean 0.000 (needs >= +0.05; 3 of 5 folds). Per the decision rule: edges add nothing detectable over the W2+E+H node features; XGBoost stays the model; no stage 2; the graph remains an option for Neo4j/UI exploration only. The only fold with a visible graph gain (fold 2: +0.057 over the MLP) is matched by the shuffled edges (+0.054), i.e. neighbour averaging, not structure.
- The no-edge MLP alone is 0.17 PR-AUC below XGBoost on the same features, so even a perfect graph effect would have to make up that gap first.
- Alert budget (repeat 0; caught of 70 / normal flagged of 930), secondary and descriptive: q = 0.25%: MLP 63 / 4, GraphSAGE 59 / 4, shuffled 60 / 4, XGBoost W2+E+H 56 / 0, generic 50 / 1; q = 0.5%: MLP 65 / 16, GraphSAGE 63 / 22, shuffled 65 / 24, XGBoost 62 / 7; q = 1%: MLP 65 / 46, GraphSAGE 65 / 48, XGBoost 68 / 27; q = 2%: GraphSAGE 67 / 90, XGBoost 69 / 51. The neural nets catch more distinct insiders at the strictest budgets (scenario 3: 9 of 10 at every budget vs XGBoost 2-4 of 10 at q <= 0.5%) but flag more normal users; the MLP does the same without edges, so it is not an edge effect (one seed, 5 folds).
- Caveats: one seed per fold (step-5 seed noise on one fold was ±0.036 for GraphSAGE); 5 folds; K = 5 keeps only the top 5 domains / recipients per user-week; one weighted-mean aggregation mixes PC, domain and email neighbours; the node features already hold every weekly count and history column, so neighbours add little new; the marker variant M3 and stage 2 were NOT run; the named-site count features are in all node features.
- Charts (opened): `results/figures/10_pilot_curves.png`, `results/figures/10_stage1_pr_auc.png`. Outputs: `results/gnn2_stage1_oof_scores.parquet`, `results/gnn2_pilot_curves.parquet`, `results/gnn2_epochs.json`, `results/graph_v2_b0.json`. Deviations from 6f: none.

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
