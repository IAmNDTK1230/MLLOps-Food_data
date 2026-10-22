# Food Review Intelligence Platform

[![CI](https://github.com/IAmNDTK1230/MLLOps-Food_data/actions/workflows/ci.yml/badge.svg)](https://github.com/IAmNDTK1230/MLLOps-Food_data/actions/workflows/ci.yml)

A local Vietnamese restaurant-review demo and a separate, frozen sentiment research benchmark. VN-MSASar v3 supplies research training, validation, and test data. First-party Foody reviews supply real-world inference, restaurant analytics, and retrieval documents; they are not treated as gold sentiment labels.

The active demo uses text-only TF-IDF sentiment, a separate aspect classifier, Foody marts, Chroma and BM25 retrieval, RAG V3 evidence checks, a local Qwen 2.5 3B model, Streamlit, and SQLite LLMOps telemetry. [The final architecture](reports/final/FINAL_ARCHITECTURE.md) describes the data and answer path. Local model weights, review data, indexes, and operational state are intentionally excluded from Git and container images.

## Run the accepted local demo

From the project directory on Windows, with the existing `.venv` and frozen local artifacts:

```powershell
.\scripts\start_demo.cmd
```

The launcher runs the read-only `python -m src.cli demo-preflight`, starts the pinned local Qwen server if needed, and opens Streamlit at `http://localhost:8501`. It does not crawl, download, train, or rebuild indexes. For a CPU-only fallback, use `.\scripts\start_demo.cmd -CpuOnly`. Active artifact paths and versions are pinned in [`config/demo.yaml`](config/demo.yaml). The [demo acceptance report](reports/final/FINAL_DEMO_ACCEPTANCE.md), [script](reports/final/DEMO_SCRIPT_VI.md), and [backup plan](reports/final/DEMO_BACKUP_PLAN.md) record the verified local presentation path.

Foody predictions have no measured Foody accuracy. RAG V3 answers use cited synthesis, a labeled extractive fallback, or an explicit insufficient-evidence result. The six-query retrieval development set is small and incomplete; see the [RAG V3 acceptance report](reports/final/RAG_V3_ACCEPTANCE_REPORT.md) for measured limits.

## Research and data

- [Data preparation and frozen split](docs/experiments/PHASE2.md)
- [Rule-based and TF-IDF baselines](docs/experiments/BASELINES.md)
- [Qwen Proposed V1](docs/experiments/PROPOSED_V1.md)
- [Proposed V2](docs/experiments/PROPOSED_V2.md)
- [Aspect-conditioned research](docs/experiments/ASPECT_CONDITIONED.md)
- [Final experiment story](reports/final/TASK2_EXPERIMENT_STORY.md)

The VN-MSASar split, labels, and completed model results are frozen. Foody processing and enrichment preserve raw lineage; ratings are never substituted for sentiment labels. Dataset and local model artifacts are not shipped in the repository. `dvc.yaml`, `dvc.lock`, and `params.yaml` retain the reproducible research protocol without requiring CI to download the dataset or retrain models.

The repository retains historical Shopee/Databricks source and bundle files because current tests and the Proposed V2 frozen-input audit reference them. They are outside the active Foody demo. Local-only Shopee live diagnostics and acceptance evidence are excluded from Git tracking; no local historical file is deleted by this cleanup.

## Checks

Use Python 3.12 in the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q src app scripts
.\.venv\Scripts\python.exe -m src.cli demo-preflight
```

The final preflight requires the local data, models, indexes, and state files. CI uses synthetic fixtures and skips that real-artifact check when they are absent.

## CI/CD and containers

[CI](.github/workflows/ci.yml) runs on `main` and `develop` pushes, pull requests to `main`, and manual dispatch. It checks Ruff, source compilation, pytest, application imports, Docker Compose syntax, and the Streamlit image build without downloading Qwen weights or rebuilding the Foody index. The [release workflow](.github/workflows/release.yml) publishes a versioned image to GHCR on `v*` tags or manual dispatch. It delivers an image; no remote production deployment is configured. See the [CI/CD acceptance report](reports/final/CICD_ACCEPTANCE.md).

`Dockerfile.rag` builds the application image. Docker Compose mounts local `data/`, `models/`, `state/`, and `reports/` at runtime; these assets are not baked into the image. `docker compose config --quiet` validates its configuration. The accepted presentation path is the local launcher above.
