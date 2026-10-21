# Phase 2 implementation report

> Historical report: superseded on 2026-10-13 by the Vietnamese restaurant data phase.
> Current source, measured results and commands are in FOOD_ACCEPTANCE_REPORT.md and
> PHASE2.md. Product/rating ML CLI commands are disabled; no model execution is enabled.

Date: 2026-10-12, Asia/Ho_Chi_Minh.

## Status

Phase 2 architecture and CLI: implemented. Real-data model acceptance: **BLOCKED —
no external publicly licensed review dataset has been supplied/imported**.
No real model was fitted, no fabricated reviews were added, no sentiment metrics are
claimed, and no Unity Catalog model version was created. Shopee diagnostics were not
rerun. The verified Phase 1 Bronze/Silver/Gold code and bundle were not modified.

## Created files

- `src/ml/{config,data,labels,features,split,train,evaluate,mlflow_tracking,registry,cli}.py`
  and package initializer.
- `src/api/{app,schemas,model_loader,monitoring}.py` and package initializer.
- `tests/test_ml_data.py`, `test_ml_split.py`, `test_ml_training.py`,
  `test_ml_evaluation_registry.py`, `test_sentiment_api.py`.
- `requirements-ml.txt`, `requirements-api.txt`, `params.yaml`.
- `dvc.yaml`, `.dvcignore`, `.dvc/config`, `.dvc/.gitignore`.
- `Dockerfile.api`, `docker-compose.api.yml`.
- `.github/workflows/ci.yml`.
- `PHASE2.md`, this report, `reports/README.md`, reports/models `.gitkeep` placeholders.

Existing files extended: `src/cli.py`, README, `.env.example`, `.gitignore`,
`.dockerignore`, and a descriptive comment in `requirements.txt`. Phase 1's direct
dependency pins remain unchanged. ML dependencies are opt-in/separate.

## Architecture and data boundary

Licensed CSV/Parquet → source/provenance snapshot → UTF-8/NFC normalization, validation
and exact deduplication → labeled local Gold-compatible dataset → leakage-safe groups
and deterministic splits → three TF-IDF/Logistic Regression experiments → validation
macro F1 selection → once-only test evaluation → explicit UC registration → API.

The target contract is `shopee.gold.review_ml_dataset`, but the verified live Gold
producer is unchanged. External development snapshots stay in `data/ml/processed/`
and do not materialize/replace a cloud table. `source` identifies the imported provider,
and `original_source` preserves a supplied input source field. No live review availability
or external-to-Gold cloud publication is claimed.

Licensing is documented by user-supplied license/source URLs and identifier. The
importer does not certify those claims; a legitimate dataset must be provided by the
operator. Fixtures are rejected and cannot satisfy the training provenance guard.

## Actually executed

- Phase 1 + Phase 2 pinned dependency install: PASS.
- `python -m pip check`: PASS — no broken requirements.
- `python -m pytest -q`: **PASS — 123 passed in 17.52 seconds**, including the original
  96 tests and 27 ML/API tests. One existing Starlette/AnyIO deprecation warning.
- `python -m ruff check .`: PASS.
- `python -m compileall -q src databricks scripts`: PASS.
- CLI help/import help: PASS; new commands are exposed.
- `python -m src.cli train-baseline`: expected **exit 1**, with a clear missing-real-
  dataset message. No model fitting or training run started.
- DVC initialization: standalone `no_scm=True` configuration present; this directory
  is not a Git repository. `dvc dag`: PASS, four stages validated.
- `dvc repro`: expected failure at absent `data/ml/source`; no fake data generated.
- Local MLflow SQLite backend smoke: PASS. Run
  `08726c3ed5434913a6839fd5de429a5f` in isolated `phase2-runtime-smoke` logged only
  verification metadata and a JSON artifact, with **no model, reviews or model metrics**.
  Backend: `state/phase2-verification/mlflow.db`.
- Actual Uvicorn startup at `127.0.0.1:8012`: PASS. GET `/health` returned expected
  HTTP 503, `Model unavailable; import real reviews, train, and configure serving`.
  Server closed cleanly after the check. Model-backed predictions were tested only
  with unit doubles, not accepted against a real model.
- Docker CLI availability detected; daemon check failed because the
  `dockerDesktopLinuxEngine` pipe does not exist. **Docker build/run NOT TESTED —
  Docker daemon unavailable.**

Dependency hardening: DVC 3.63 required pathspec 0.12.1; the compatibility pin was
added after an actual import failure with pathspec 1.1.1. The final installed
environment passes dependency consistency checks.

## Unit coverage limits

Tests cover normalization/label rules, column mapping, CSV/Parquet import, rejects,
deduplication, hash tampering, trainability failures, deterministic grouping, duplicate
comment leakage, validation-only model selection, logging all three experiments,
one-time held-out prediction, registration without alias promotion, API readiness,
probabilities/version metadata, and exclusion of private text from logs.

Small manufactured unit inputs and model doubles are temporary test data only. Real
Logistic Regression is not fitted on those inputs. Mocked scores are not experiment
or test-set acceptance results.

## Still requires real data or external execution

- Actual data report/distributions for a licensed review dataset.
- Real splits, fitted models, experiment comparison and held-out performance.
- MLflow model logging/round-trip loading against an actual sentiment model.
- Databricks experiment permissions and `shopee.ml`/`shopee_test.ml` model registration.
- Controlled model promotion review; registration does not promote any alias.
- Complete real-data DVC reproduction and optional remote configuration.
- API prediction acceptance using actual trained/registered weights.
- Docker daemon startup/build/runtime and GitHub Actions execution.

Exact usage commands and dataset format are in `PHASE2.md`. No generated ML data
report, experiment-comparison CSV or trained weights are shipped while the real
dataset is absent.
