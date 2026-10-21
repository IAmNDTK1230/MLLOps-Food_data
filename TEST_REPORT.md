# Local verification

Verified on Windows with Python 3.12. The project's direct dependencies were installed
in `.venv` and Prefect's server dependencies are pinned in `requirements.txt`.

## Passed

- `python -m pytest -q`: **49 passed**. Tests cover model constraints and UTC,
  Vietnamese price/count normalization, rejection persistence, raw file history,
  typed Parquet, manifest identity, upload recovery and overwrite protection,
  cloud job completion, retry policy, restriction detection, checkpoint resume,
  flow success/failure/partial blocking, and empty Prefect API URL configuration.
- Ruff lint and formatting checks, Python syntax compilation.
- `python -m pip check`: no broken requirements.
- Deployment YAML parsing, seven Databricks tasks and timezone-aware Prefect cron
  construction using the installed Prefect API.
- The real CLI/Prefect flow with `--crawl-only --fixture tests/fixtures/products.jsonl`
  starts and stops its temporary server, completes successfully, and emits no event
  worker errors after normalizing blank API URLs.
- Offline output verified: two raw observations, one valid Parquet record priced at
  199000 VND, one rejected record, UTC Arrow timestamp schema, and successful run metadata.
  Demonstration data is isolated in `data/offline-demo`, with corresponding state and
  logs in `state/offline-demo` and `logs/offline-demo`.

## Not verified here

- Live Shopee HTML/browser navigation: Chromium browser binaries were not installed
  for a live crawl. Current public access and selectors must be checked on a permitted
  run. Restrictions are handled explicitly, and successful live collection is not guaranteed.
- Docker build/run: Docker is not installed on this host.
- Actual Databricks authentication, Volume upload, bundle validation/deployment,
  Auto Loader ingestion, Delta transformations and remote job completion: no workspace
  credentials or compute configuration were provided. Uploader/job behavior was tested
  with mocks, and Spark scripts were syntax-checked; they require cloud acceptance testing.
- Reviews: deliberately disabled and modular; no live review implementation is claimed.

Use the README's bootstrap and bundle instructions before the first cloud pipeline run.
