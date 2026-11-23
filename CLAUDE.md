# CLAUDE.md

Insider threat detection on the CERT r4.2 dataset (1,000 users, 70 insiders). Users and their
PC logons are modelled as a graph and classified with GNNs (GraphSAGE, trained from scratch),
and compared against tabular baselines (logistic regression, random forest, XGBoost).
Results are meant to feed Neo4j and a Streamlit analyst UI later.

**Read `docs/PROJECT_LOG.md` first** (decisions, results, caveats, roadmap, open questions).
**Update it at the end of every step.**

## How to run
- Environment: `.venv` (Python 3.12). `source .venv/bin/activate`. Install with `uv pip install ...`
  (add new packages to `requirements.txt` too). Package is installed editable (`src/itgnn`).
- Tests: `pytest` (synthetic data only, runs in ~1 s). It must keep passing.
- Notebooks use the Jupyter kernel `itgnn`. Run from the repo root or `notebooks/` (each notebook resolves the repo root).
  Re-run one: `jupyter nbconvert --to notebook --execute --inplace notebooks/03_weekly_baseline.ipynb`
- Raw data is in `data/raw/` (gitignored); parsed logs are cached as parquet in `data/processed/`.
- Config: `configs/config.yaml` (`n_users: null` = all users, `cv: {n_splits: 5, n_repeats: 5}`, seed 42).
- Scripts `python -m itgnn.baseline` / `itgnn.train` / `itgnn.neo4j_io` exist from the first commit and
  are not part of the current notebook work; leave them unchanged.

## Layout
- Reusable logic lives in `src/itgnn/`; notebooks in `notebooks/` are the narrative layer
  (hybrid layout). Do not put analysis logic only in a notebook if it will be reused.

## Working agreement
- Give a short plan and **wait for my go-ahead before writing code**.
- Keep existing function behavior and `pytest` passing; add tests for new logic.
- **Do not commit** unless I ask.
- Say clearly which charts or outputs you did not look at. Never claim a chart looks fine unseen.
- Explain results in plain language: what was checked, what the number means, and the caveats.
- Never tune on test folds. Use the shared folds from `make_folds` (same folds for every model)
  and the tie-aware `evaluate_scores` metrics in `src/itgnn/evaluation.py`.
- Split by user, never by week or event. Report PR-AUC, precision and recall, not accuracy.
- Base every statement on the repo, notebook outputs or `results/` files. If something cannot
  be verified, say so.
- Fixed hyperparameters live in the notebooks (not tuned).
