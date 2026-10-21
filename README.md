# shopee-data-mlops-platform

## Quick Demo (Windows, frozen RAG V3)

From this project directory, with the existing `.venv` and frozen artifacts:

```powershell
.\scripts\start_demo.cmd
```

The launcher runs `python -m src.cli demo-preflight`, checks all active artifacts,
starts the pinned local Qwen 3B server if needed, and opens Streamlit at
`http://localhost:8501`. It does not download, train, or reindex. For a slow
CPU-only fallback use `.\scripts\start_demo.cmd -CpuOnly`. The active paths and
versions are pinned in `config/demo.yaml`. Presentation evidence and the backup
plan are in [`reports/final/`](reports/final/FINAL_DEMO_ACCEPTANCE.md).

The active AI Review Analyst uses evidence-first RAG V3: cited Qwen synthesis,
clearly labeled extractive fallback, or an explicit insufficient-evidence answer.
Foody predictions have no trusted Foody accuracy measurement. Docker Compose
configuration passed previously; Docker runtime remains unverified.

## Earlier Foody review retrieval (RAG V1)

This layer reads the existing `data/foody/rag/documents.jsonl`. Activate the
project virtual environment, then install `requirements-rag.txt` alongside the
existing requirements. On first use, `rag-build-index` compares two local
multilingual ONNX embedding models against a six-query, manually inspected
known-positive development set. It selects an embedding by known-positive
top-10 hit rate, then MRR and query latency, and indexes all valid review
documents in persistent Chroma. Model weights and the vector index stay under
ignored local `models/` and `data/` paths.

The current persistent index has 32,457 reviews and passed source-ID, ONNX-hash
and vector-dimension reconciliation. Reuse it for the demo; `rag-build-index`
is for an initial build if the index is absent.

```powershell
python -m scripts.verify_rag_index
python -m src.cli rag-evaluate
python -m src.cli rag-query --restaurant "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu" --question "Nhân viên phục vụ có vấn đề gì không?" --sentiment negative --aspect service
python -m src.cli restaurant-health-report --restaurant "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu"
python -m streamlit run app/app.py
```

The Streamlit demo has Restaurant Intelligence, Review Explorer, AI Review
Analyst and LLMOps Monitor pages. Review and mart pages open without the vector
index or Qwen service; AI actions show a clear unavailable message. Operational
events go to `state/llmops.sqlite` without raw query or review text. Retrieval
supports restaurant key, date range, predicted sentiment and predicted aspect.
The Qwen gateway reuses the pinned local Qwen 2.5 GGUF/llama.cpp runtime, or a
configured `QWEN_BASE_URL`. Generated answers must cite retrieved review IDs;
invalid citations yield an insufficient-evidence response.
The mechanical citation gate checks retrieved ID membership and inline reference
consistency. Manual acceptance review found one unfiltered answer with a
semantically contradictory citation; consult the detailed report before using
generated claims as conclusions.

`docker compose up --build` defines Streamlit, a FastAPI backend, a Qwen
service, and shared local Chroma storage. Docker mounts `data/`, `models/`,
`reports/` and `state/` at runtime; image builds exclude model weights and
review data. The Qwen model must already exist locally under `models/qwen_v1/`.
The Chroma index must be built before AI retrieval is available in the app.
`docker compose config` passed locally, but Docker runtime was not executed
because the Docker daemon was unavailable.

The development judgments cover only a few known relevant reviews. Their
hit rates are not complete retrieval recall or accuracy. VN-MSASar scores
remain research benchmarks. Foody sentiment and aspect predictions have no
measured Foody accuracy; issue detection is heuristic. Restaurant IDs remain
derived from normalized names rather than official Foody IDs.

See [RAG acceptance evidence](reports/rag/RAG_ACCEPTANCE_REPORT.md) and
[the combined product acceptance report](reports/RAG_STREAMLIT_LLMOPS_ACCEPTANCE_REPORT.md).

## First-party Foody intelligence (2026-10-20)

Three distinct data roles:

| Dataset | Role |
| --- | --- |
| VN-MSASar v3, frozen memberships | Research training, validation and benchmark testing |
| `data/foody/source/foody_reviews.csv` | First-party real-world inference and demo analytics |
| External `review.csv` / `review_utf8_20261020.csv` | Supplementary historical audits only; unknown Classification provenance |

The earlier external dataset's unresolved restaurant IDs do not block the first-party
Foody crawl. The new source has RestaurantName and no Classification. Name-derived
SHA256 keys are **not official restaurant IDs**. Ratings never become sentiment labels.
Historical audit evidence remains unchanged.

With the project virtual environment activated:

```powershell
python -m src.cli foody-prepare
python -m src.cli train-aspect-baseline
python -m src.cli run-foody-intelligence
```

Aspect training uses VN-MSASar only: TF-IDF + OneVsRest Logistic Regression, with
validation-selected configuration/threshold and one checkpointed test evaluation.
Repeating the completed training command reuses its frozen artifact and metrics.
`run-foody-intelligence` requires this aspect artifact, and **never trains either model**.
It prepares → enriches → builds marts → writes review RAG documents.

Individual stages are also available:

```powershell
python -m src.cli foody-enrich
python -m src.cli foody-build-mart --minimum-samples 10 --rate-increase 0.2 --rating-decrease 1.0
python -m src.cli foody-prepare-rag
```

Every source row survives processing/enrichment. AI runs on valid, conservative
restaurant+normalized-text groups, then predictions join back to original rows.
Marts and RAG count distinct deterministic review IDs; raw lineage and ratings remain
available. U+FFFD/empty text is excluded from AI without repair. Alerts are historical
adjacent-month comparisons with adequate samples; they are not live restaurant warnings.
Foody predictions are not accuracy measurements. Issue extraction is explicitly heuristic.
No embeddings, vector database or Streamlit are built here.

See [the measured Foody report](reports/FOODY_INTELLIGENCE_DATA_REPORT.md) and
[aspect benchmark](reports/aspect_classifier/ASPECT_ACCEPTANCE_REPORT.md).

Phase 1 data engineering implementation for public Shopee Vietnam product observations.
Python 3.11+, async Playwright, Pydantic, Pandas/PyArrow, Prefect 3, official Databricks
SDK, Unity Catalog Volumes and Delta Lake. Phase 2 research experiments and their
separate food-review protocols are documented below.

## Architecture

```text
Shopee (public pages only)
  ↓
Crawler (robots.txt, global delay, restriction detection, checkpoint)
  ↓
Raw JSONL (original visible fields + card HTML, including invalid records)
  ↓
Validation → rejected JSONL audit trail
  ↓
Parquet (normalized explicit nullable schema, UTC)
  ↓
Databricks Volume (JSONL + Parquet, checksum manifest)
  ↓
Bronze (Auto Loader JSONL, append-only Delta + source metadata)
  ↓
Silver (cleaning, rejection tables, insert-only deduplication + current snapshot)
  ↓
Gold (daily metrics, category/shop summaries, clean review text)
  ↓
Analytics / ML (future Phase 2)
```

Shopee may require login, block automation, change its HTML, or disallow search in
robots.txt. This implementation stops in those cases; it cannot promise live records.
Only rendered public card fields are collected. Optional unavailable fields are null.
The browser identifies itself as `ShopeeDataPlatform/0.1` and follows that robots policy.
No login, imported cookies, hidden/private API calls, proxies, stealth, CAPTCHA solving,
or attempts to bypass restrictions are used. Reviews are a disabled extension point,
not a functioning live collection feature. Enabling them reports an explicit failure.

## Project structure

```text
src/
  crawler/          base interface, products, disabled reviews, DOM assumptions
  models/           Product, Review, CrawlRun
  processing/       normalizer, validator, typed Parquet export
  storage/          exclusive raw writes, atomic state and upload manifest
  databricks/       SDK connection, upload, job trigger and completion check
  orchestration/    Prefect tasks and ShopeeDataPipeline flow
  utils/            JSON logging, retry policy, hashing, UTC
  config.py         validated environment settings
  cli.py            crawl, pipeline, serve
databricks/         00–06 Spark task scripts and shared transformations
tests/              offline and mocked integration tests + synthetic fixture
data/{raw,staged,rejected,metadata}/
state/              durable checkpoints, upload_manifest.json, run snapshots
logs/               pipeline.jsonl
databricks.yml      Databricks Asset Bundle job graph
prefect.yaml        optional process worker deployment
Dockerfile, docker-compose.yml, .env.example
```

## Local environment

Run all commands from this project directory. Python 3.11 or later is required.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m playwright install chromium
# Linux system dependencies (or use Docker):
# python -m playwright install --with-deps chromium
cp .env.example .env
# PowerShell: Copy-Item .env.example .env
```

Fill `.env`; it is ignored by Git and excluded from the Docker build context. PySpark
and Delta are supplied by Databricks Runtime; do not install plain PySpark into the
crawler container. Runtime names and credentials come from environment variables.

| Variable | Purpose |
|---|---|
| `DATABRICKS_CONFIG_PROFILE` | Optional named SDK/CLI profile; example `shopee`, supports cached CLI OAuth |
| `DATABRICKS_HOST` | Optional workspace URL for unified auth or legacy PAT |
| `DATABRICKS_TOKEN` | Optional legacy PAT; unnecessary for OAuth CLI profiles |
| `DATABRICKS_CATALOG` | Default `shopee`; set `shopee_test` only for synthetic cloud acceptance |
| `DATABRICKS_RAW_SCHEMA` | Default `raw` |
| `DATABRICKS_BRONZE_SCHEMA` | Default `bronze` |
| `DATABRICKS_SILVER_SCHEMA` | Default `silver` |
| `DATABRICKS_GOLD_SCHEMA` | Default `gold` |
| `DATABRICKS_VOLUME` | Default `landing` |
| `DATABRICKS_JOB_ID` | Deployed job ID; blank explicitly skips transformations |
| `DATABRICKS_JOB_TIMEOUT_MINUTES` | Completion wait limit, default 60 |
| `CRAWL_CONCURRENCY` | 1–4 page workers; default 1 |
| `CRAWL_DELAY_SECONDS` | Global delay between page navigations, minimum 1, default 5 |
| `CRAWL_TIMEOUT_MS` / `CRAWL_HEADLESS` | Browser timeout / headless mode |
| `REVIEWS_ENABLED` | Keep false; public reviews not implemented |
| `DATA_DIR` / `STATE_DIR` / `LOG_DIR` | Local directories |
| `CRAWL_CRON` / `CRAWL_TIMEZONE` | Schedule and IANA timezone |
| `CRAWL_KEYWORD` / `CRAWL_PAGES` | Served deployment defaults |
| `PREFECT_API_URL` | Optional persistent server API; unset for manual runs |

## Crawl and run the pipeline

```bash
python -m src.cli crawl --keyword "tai nghe bluetooth" --pages 3
python -m src.cli pipeline --keyword "laptop gaming" --pages 5 --skip-reviews
python -m src.cli pipeline --keyword "laptop gaming" --pages 5 --crawl-only
python -m src.cli pipeline --keyword "laptop gaming" --pages 5 --resume --crawl-only
python -m src.cli pipeline --keyword "retry upload" --upload-only --skip-databricks-job
python -m src.cli pipeline --keyword "laptop gaming" --pages 5 --skip-databricks-job
```

`crawl` and `--crawl-only` write JSONL, rejects, Parquet and metadata without Databricks.
`--upload-only` scans historical JSONL/Parquet, skips manifest-confirmed uploads and
then triggers the configured job unless skipped. `--crawl-only` and `--upload-only`
are mutually exclusive. If no job ID is configured the summary explicitly marks the
job skipped: only the Volume is updated, not Bronze/Silver/Gold.

Manual commands work without a running Prefect Server: Prefect starts a temporary
local server. That still requires local networking and permission to run processes.
Use a persistent server for production monitoring and scheduling.

### Offline demonstration

```bash
python -m src.cli pipeline --keyword "offline sample" --pages 1 --crawl-only --fixture tests/fixtures/products.jsonl
```

No browser or external credentials are used. One valid record, one rejected record,
raw JSONL, one Parquet file and run metadata are expected. Fixture mode is local-only
by default; the explicit test-only cloud exception is described below. Use a separate `DATA_DIR` for demos
so it cannot accidentally be included in a later upload-only run.

### Expected files and statuses

```text
data/raw/products/crawl_date=YYYY-MM-DD/crawl_hour=HH/products_<UTC timestamp>_<run_id>_<suffix>.jsonl
data/staged/products/crawl_date=YYYY-MM-DD/products_<UTC timestamp>_<run_id>_<suffix>.parquet
data/rejected/products_<run_id>_<uuid>.jsonl
data/metadata/run_<run_id>_<status>_<suffix>.jsonl
state/checkpoints/<keyword+page-count hash>.json
state/upload_manifest.json
state/run_<run_id>.json
logs/pipeline.jsonl
```

Historical raw/export/summary files are created exclusively, never overwritten.
Checkpoint and manifest state is updated atomically under local file locks. Only one
local pipeline may own a state directory at a time. Multiple hosts need separate
state directories and a distributed scheduler/lock; do not put this lock on arbitrary
network storage. Checkpoints preserve complete page observations before marking the
page complete. `--resume` replays saved pages and only fetches missing ones for the
same keyword/page count. Without `--resume`, a new observation crawl is started.

Logs include run ID, keyword, stage, status, record counts and duration. Final JSON
contains all run metadata and stage outcomes. Restriction statuses are `BLOCKED`,
`CAPTCHA`, `RATE_LIMITED` (exit 2). `FAILED` or `NO_DATA` exits 1; successful runs exit 0.
Partial observations are saved locally after a restriction; no upload/job follows it.
An entirely rejected/empty crawl is not reported as successful.
Failure summaries are saved before the flow raises, so Prefect marks those runs failed.

## Databricks and Unity Catalog

### Authentication

The local preflight, uploader, and job trigger use the official SDK's Unified
Authentication. An explicit `DATABRICKS_CONFIG_PROFILE` selects
`WorkspaceClient(profile=...)`; no PAT is required. Otherwise the project tries
`WorkspaceClient()` with SDK default unified authentication (including the DEFAULT
profile and supported environment credentials). Optional host/token values supplied
through Settings are used as a legacy fallback only if default client configuration
fails. A failing explicit profile reports failure and never silently switches identities.

For your existing OAuth login:

```powershell
$env:DATABRICKS_CONFIG_PROFILE = "shopee"
python -m src.cli databricks-check
```

Alternatively, set `DATABRICKS_CONFIG_PROFILE=shopee` in `.env`; `.env.example`
includes this setting. Keep the Databricks CLI available on PATH and run under the
same user account that performed OAuth login. The SDK manages CLI OAuth credentials;
do not paste OAuth access tokens into `.env` or source code. See
[SDK unified authentication](https://docs.databricks.com/aws/en/dev-tools/sdk-python)
and [configuration profiles](https://docs.databricks.com/gcp/en/dev-tools/auth/config-profiles).

`databricks-check` always attempts SDK authentication instead of requiring a host/token
pair. It checks the authenticated identity, catalog, all four schemas, volume, and
a unique write/read/delete probe. Failures return exit 1 with the operation and safe
exception type; SDK exception text and tokens are excluded. An unavailable or expired
profile is an authentication failure, not a skipped check.

Host OAuth profiles and cached login sessions are not automatically available inside
Docker. Configure unified authentication for the container separately when using
cloud upload there; credential-free crawl-only runs need no Databricks authentication.

Use a workspace with Serverless Jobs enabled and Unity Catalog. All seven Python
Spark tasks reference the shared job environment `default`, with
`environment_version: "2"`. Databricks supplies PySpark and Delta in the serverless
runtime; do not install `pyspark` or `delta-spark` from pip. The job identity needs `USE CATALOG`,
`USE SCHEMA`, `READ VOLUME`, `WRITE VOLUME`, and table creation/modification rights.
The setup task additionally needs catalog/schema/volume creation rights; have an
administrator create these first if the run identity lacks those rights.

The uploader requires the Volume to exist **before the first upload**. Bootstrap it:

```sql
CREATE CATALOG IF NOT EXISTS shopee;
CREATE SCHEMA IF NOT EXISTS shopee.raw;
CREATE SCHEMA IF NOT EXISTS shopee.bronze;
CREATE SCHEMA IF NOT EXISTS shopee.silver;
CREATE SCHEMA IF NOT EXISTS shopee.gold;
CREATE VOLUME IF NOT EXISTS shopee.raw.landing;
```

Configure the official Databricks CLI authentication locally. The bundle uses
Serverless Jobs compute, with no classic compute configuration. Environment variables
`BUNDLE_VAR_*` configure deployment variables, while `DATABRICKS_*` configure the
local pipeline. Match both sets when customizing names.

```bash
databricks bundle validate
databricks bundle deploy
# Bootstrap/test the empty feed and verify the entire job graph:
databricks bundle run data_pipeline
databricks bundle summary
```

Copy the deployed job ID into `DATABRICKS_JOB_ID`. `databricks/common.py` is synced
alongside the task scripts and imported from their directory. The bundle sets all
name arguments for every task, serializes job runs, and supplies the pipeline run ID.
No compute or remote job is created by the local CLI itself. Deployment is a separate
operator step.

### Synthetic cloud acceptance in the test target

Normal/local Settings and the bundle's global catalog default remain `shopee`.
The default `dev` target retains `mode: development`. A separate `test` target
also uses development mode and overrides only the catalog variable to `shopee_test`.
The schemas remain `raw`, `bronze`, `silver`, `gold`, and the volume remains `landing`.
The Serverless environment and seven-task dependency graph are shared unchanged.
Clear any conflicting `BUNDLE_VAR_catalog` override before deploying either target.

Deploy and run the isolated test job:

```powershell
databricks bundle validate -t test --profile shopee
databricks bundle deploy -t test --profile shopee
databricks bundle run -t test data_pipeline --profile shopee
```

Use the job ID from the **test** target (for example, inspect
`databricks bundle summary -t test --profile shopee`). The local SDK trigger uses
`DATABRICKS_JOB_ID`, so do not substitute the dev job ID. The test Volume must exist
before local upload; the initial test bundle run executes setup before ingestion.
Alternatively generate bootstrap SQL with `DATABRICKS_CATALOG=shopee_test` and run
it as an appropriately privileged identity.

Cloud fixtures require all three: `--fixture`, `--allow-synthetic-cloud-test`, and
the exact configured catalog `shopee_test`. The CLI and direct flow both reject an
unsafe combination before reading the fixture or creating run files. The allow flag
requires a fixture and cannot be used to combine fixtures with `--upload-only`.
Existing `crawl --fixture` and `pipeline --crawl-only --fixture` remain allowed.

Windows Command Prompt (CMD):

```bat
set "DATABRICKS_CONFIG_PROFILE=shopee"
set "DATABRICKS_CATALOG=shopee_test"
set "DATABRICKS_JOB_ID=<TEST_JOB_ID>"
python -m src.cli pipeline ^
  --keyword "cloud acceptance" ^
  --pages 1 ^
  --skip-reviews ^
  --fixture tests\fixtures\products.jsonl ^
  --allow-synthetic-cloud-test
```

Use an actual numeric test job ID in place of `<TEST_JOB_ID>`. PowerShell equivalents:

```powershell
$env:DATABRICKS_CONFIG_PROFILE = "shopee"
$env:DATABRICKS_CATALOG = "shopee_test"
$env:DATABRICKS_JOB_ID = "<TEST_JOB_ID>"
python -m src.cli pipeline --keyword "cloud acceptance" --pages 1 --skip-reviews --fixture tests/fixtures/products.jsonl --allow-synthetic-cloud-test
```

The fixture follows the real raw/validation/Parquet/upload/job path. Uploads go to
`/Volumes/shopee_test/raw/landing/shopee/`. The pipeline waits for the configured test
job's Bronze/Silver/Gold tasks; a successful local mocked test is not cloud acceptance.
Cloud fixture outputs are isolated under `synthetic-cloud-test` inside each configured
data/state/log directory, including a separate upload manifest. Normal `--upload-only`
scans do not include that subtree. Clear the test catalog and job ID overrides before
resuming normal cloud runs.

### Serverless compatibility

The processing scripts use Spark DataFrame/SQL and Delta table APIs, without RDDs,
SparkContext, JVM access, custom Spark extensions, or compute-scoped libraries.
`SparkSession.builder.getOrCreate()` uses the task's runtime-provided session.
The UTC session timezone setting is supported. Parameters pass catalog, schema,
volume, and run ID explicitly, so tasks do not depend on compute environment variables.

Auto Loader uses `trigger(availableNow=True)`, a supported serverless trigger, with
durable checkpoints and schema state in the Unity Catalog Volume. The existing
absolute `/Volumes/{catalog}/raw/landing/` paths support Python directory creation and
Spark reads/writes subject to Unity Catalog permissions. No DBFS mount is required.
Bundle deployment syncs the Python task files and sibling `databricks/common.py` to
workspace files. Python script imports resolve from the script directory, rather
than relying on the task working directory; Spark input and checkpoint paths remain
absolute Volume paths.

Defaults are catalog `shopee`, schemas `raw`, `bronze`, `silver`, `gold`, and
volume `landing`. SDK upload and job triggering remain independent of the compute
configuration. A notebook `spark.range(10).show()` confirms basic serverless Spark
access; bundle validation and the full job must still verify permissions, imports,
Auto Loader, and Delta processing for the deployed identity.

References: [serverless bundle configuration](https://docs.databricks.com/aws/en/dev-tools/bundles/examples#job-that-uses-serverless-compute),
[serverless limitations and supported triggers](https://docs.databricks.com/aws/en/compute/serverless/limitations),
[environment version 2](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/two).

Upload destination:

```text
/Volumes/{catalog}/{raw_schema}/{volume}/shopee/
  products/crawl_date=YYYY-MM-DD/crawl_hour=HH/*.jsonl
  products/crawl_date=YYYY-MM-DD/*.parquet
  reviews/crawl_date=YYYY-MM-DD/...
```

The upload manifest records destination, SHA-256, source path, time, run ID and status.
The same bytes at the same destination are skipped. If an upload succeeded before the
manifest was saved, a retry downloads and verifies the remote checksum. Different
bytes at an existing destination cause failure; they are never overwritten. Files
and manifest must be treated as immutable once uploaded. Deleting remote data while
keeping the manifest requires an operator to reconcile the corresponding entries.

### Layer behavior

Bronze ingests **JSONL only**, including invalid observations and raw card HTML, with
Auto Loader `availableNow` and durable checkpoints under the Volume's `_checkpoints`.
Parquet is a normalized export, not a second Bronze source. Reading both formats
would duplicate observations. JSONL and Parquet upload both use immutable paths.
Do not reset Auto Loader checkpoints: deleting them replays historical Bronze input.

Bronze tables: `products_raw`, `reviews_raw`, with `raw_payload`, `_source_file`,
`_run_id`, `_ingestion_timestamp`, `_ingestion_date`. Empty review directories are
supported with explicit input schema and precreated Bronze tables.

Silver trims strings, casts visible Vietnamese prices/counts, uses UTC, filters
invalid values into `products_rejected` / `reviews_rejected`, and deduplicates using
insert-only Delta MERGE. Products use `(product_id, crawl_timestamp)`;
`products_current` uses the latest timestamp per product. Reviews use `review_id`
when available, otherwise SHA-256 of a JSON struct containing product, rating, comment
and review timestamp. Nulls/boundaries are preserved in that hash.

Gold refreshes `daily_product_metrics`, `category_summary`, `shop_summary` and
`review_ml_dataset` as atomic per-table Delta replacements. Daily metrics select
the latest observation per date/product. `total_sold` sums publicly shown cumulative
sold counts, **not units sold that day**. Missing categories or shops form null groups;
missing numeric observations stay null rather than invented zero values. Review text
has whitespace normalized, empty comments excluded, and no sentiment label inferred.
Gold full refreshes and Silver full-history scans are appropriate for this foundation;
large-scale production should add incremental change processing and retention rules.
There is no transaction across all Gold tables: a failed job should be rerun to finish.

The local trigger waits for job completion, checks the result, and includes the remote
run ID/URL. The idempotency token protects a retry of the same trigger. A completion
timeout does not cancel a remote job; check its status before starting another run.

## Docker

```bash
cp .env.example .env
# Configure container-accessible unified authentication and deployed job ID for cloud runs.
docker compose build
docker compose run --rm pipeline python -m src.cli pipeline --keyword "tai nghe bluetooth" --pages 2 --skip-reviews
docker compose run --rm pipeline python -m src.cli crawl --keyword "tai nghe bluetooth" --pages 2
```

The image installs Chromium and Linux browser dependencies and runs as a non-root
user. Credentials are injected at runtime by `.env`; they are never copied into the
image. Linux bind directories must be writable by UID 10001 (or set an appropriate
Compose user). Docker Desktop normally manages Windows bind mount permissions.

## Configurable Prefect schedule

Set `CRAWL_CRON`, `CRAWL_TIMEZONE`, `CRAWL_KEYWORD`, `CRAWL_PAGES` in `.env`.
Scheduling is optional, and the default cron is merely an example.

```bash
docker compose --profile orchestration up --build prefect-server scheduler
```

Open `http://localhost:4200` for the UI. The scheduler serves the Python flow with
one concurrent run. It must remain running to execute scheduled work. Locally, start
`prefect server start`, set `PREFECT_API_URL=http://127.0.0.1:4200/api`, then run
`python -m src.cli serve`. CLI `serve` loads schedule and page count directly from
the environment; it is the recommended scheduling method.

Alternatively, `prefect.yaml` supports a process worker:

```bash
export PROJECT_DIR="/absolute/path/to/shopee-data-mlops-platform"
export CRAWL_CRON="0 2 * * *"
export CRAWL_TIMEZONE="Asia/Ho_Chi_Minh"
export CRAWL_KEYWORD="tai nghe bluetooth"
prefect work-pool create shopee-process --type process
prefect deploy --all
prefect worker start --pool shopee-process
```

Set the YAML deployment page parameter when using that alternative; `serve` reads
`CRAWL_PAGES`. Use one scheduling approach to avoid duplicate scheduled runs.

## Tests and code checks

```bash
python -m pytest -q
python -m ruff check .
python -m compileall -q src databricks
```

Tests need no live Shopee or Databricks access. They cover validation/UTC, Vietnamese
normalization, rejection persistence, append-only storage, explicit Parquet types,
checksum manifest rules, upload retry recovery/collisions, restriction detection,
checkpoint resume, and flow success/partial block/failure behavior. The offline fixture
command also exercises the real Prefect task/flow runtime.

See [TEST_REPORT.md](TEST_REPORT.md) for verification performed in this workspace and
the remaining live/browser, Docker and Databricks acceptance checks. Compatible
Prefect server dependencies are pinned alongside the main libraries; an empty
`PREFECT_API_URL` is normalized to unset for manual runs.

## CI/CD

GitHub Actions CI runs on pushes to `main` and `develop`, pull requests to `main`,
and manual dispatch. It uses Python 3.12 to run Ruff, source compilation, the
artifact-independent pytest suite, a lightweight application import, Docker
Compose configuration validation, and an application image build. The real
`python -m src.cli demo-preflight` remains a local check because it requires
mounted model weights, data, indexes, and operational state.

Pushing a stable `v*` release tag (or manually dispatching the release workflow)
builds and publishes a versioned image to `ghcr.io/<owner>/<repo>` with
`GITHUB_TOKEN`. The image excludes local data, Qwen weights, vector indexes,
SQLite state, and secrets; provide those as runtime mounts. This is image
delivery to GHCR. No remote production server deployment is configured.
See [CI/CD acceptance](reports/final/CICD_ACCEPTANCE.md) for verification status.

For cloud acceptance, bootstrap the Volume, deploy the bundle, fill credentials,
run the pipeline, then query all seven business tables and the two rejection tables.
Run the Databricks job a second time without uploading files; Bronze counts should
remain stable, Silver keys unique, and Gold results unchanged. Runtime execution,
Auto Loader and Delta transactions require an actual Databricks workspace to verify.

## Troubleshooting

- `BLOCKED` / `CAPTCHA` / `RATE_LIMITED`: stop and review site policy. Do not loop
  retries or enable workarounds. Use permitted alternate data or an offline fixture.
- `NO_DATA`: distinguish empty public results from a changed DOM. Review publicly
  visible HTML and update only `src/crawler/selectors.py`. Do not add private API calls.
- Missing browser executable: run `python -m playwright install chromium` again after
  updating Playwright, or rebuild Docker.
- Missing Volume / permissions: bootstrap it, check catalog/schema names and both
  uploader and job identities. The Files API cannot create the Volume itself.
- Job skipped: fill `DATABRICKS_JOB_ID`; upload alone does not execute transformations.
- Manifest collision: keep historical paths immutable; do not replace existing data.
- Lock timeout: a pipeline is already using this state directory. Wait for it to finish.
- Prefect connection refused: unset a stale `PREFECT_API_URL` for manual temporary-server
  runs, or start the configured server.
- Empty review dataset is expected while reviews are disabled. Null optional product
  fields reflect what was publicly visible and should not be fabricated.

## Phase 2 — Vietnamese restaurant data preparation

Phase 1 is complete and verified; live Shopee ingestion remains `PUBLIC_PAGE_EMPTY`.
Do not keep probing Shopee or alter the verified Bronze/Silver/Gold architecture.
Phase 2 uses only [VN-MSASar v3](https://huggingface.co/datasets/phamluan/vn_msa_sar_v3),
filtered exactly to `language == "vi"` and `place_type == "restaurant"`.
The real source has been downloaded: 5,869 valid sentence/aspect pairs, 1,530 reviews.
Gold signed sentiment labels replace rating-derived labels. The shared leakage-safe
split has 4,140 train / 842 validation / 887 test rows. EDA charts, source/license/hash
provenance, RAG documents and tokenizer-only chunk comparisons are generated locally.
The text-only Rule-Based and TF-IDF/Logistic Regression baselines now run explicitly
with `python -m src.cli run-baselines`; see [BASELINES.md](BASELINES.md).
Previous product ML commands remain disabled. Qwen Proposed V1 now has a separate
explicit inference command, `python -m src.cli proposed-v1`; see
[PROPOSED_V1.md](PROPOSED_V1.md). The calibrated text-only V2-A experiment uses
`python -m src.cli proposed-v2`; see [PROPOSED_V2.md](PROPOSED_V2.md).
Baselines/split/V1 stay frozen. No fine-tuning, aspect-conditioned implementation,
external dataset download, embeddings or production registration are included.
See [PHASE2.md](PHASE2.md) for the food import/preparation commands and
[food acceptance report](reports/FOOD_ACCEPTANCE_REPORT.md) for measured results.

Implementation references: [Databricks Python SDK](https://docs.databricks.com/aws/en/dev-tools/sdk-python),
[Auto Loader schema handling](https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/auto-loader/schema),
[SDK Jobs waiter](https://databricks-sdk-py.readthedocs.io/en/stable/wait.html),
[Playwright navigation](https://playwright.dev/python/docs/navigations).
## Aspect-conditioned research experiment

The new text + aspect protocol and commands are in
[ASPECT_CONDITIONED.md](ASPECT_CONDITIONED.md). It adds an aspect-aware TF-IDF
control and three Qwen prompts while freezing previous experiment outputs.
Text-only historical comparisons use a different input contract; reused test
results are exploratory. No Phase 1 changes or production model registration.

## Foody RAG V2

RAG V2 keeps the existing 32,457-document Chroma/MiniLM index and adds a local
SQLite FTS5/BM25 index over the same review IDs. The selected retrieval path uses
dense and lexical top-20 results, reciprocal rank fusion, limited Vietnamese
expansion, complaint/aspect filters, and a local evidence-aware scoring step.
Questions use the local Qwen runtime with claim-level citations and a conservative
semantic support gate. The four-page Streamlit layout is unchanged.

```bash
python -m src.cli rag-v2-build-lexical
python -m src.cli rag-v2-evaluate
python -m src.cli rag-v2-query --restaurant "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu" --question "Nhân viên phục vụ có vấn đề gì không?"
python -m src.cli rag-v2-health-report --restaurant "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu"
streamlit run app/app.py
```

The local Qwen model and llama.cpp runtime configured for RAG V1 are required for
answer generation. See [RAG V2 acceptance](reports/rag_v2/RAG_V2_ACCEPTANCE_REPORT.md)
for the small, incomplete known-relevance benchmark, manual evidence review,
generation limitations, and full-index integrity checks. Operational logs keep
retrieval strategy and timing without raw question or review text.

