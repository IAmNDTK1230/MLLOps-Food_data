# Live Acceptance Report

Updated 2026-10-12, Asia/Ho_Chi_Minh. The remaining blocker is live public Shopee ingestion. The verified Databricks architecture, transformations, authentication and bundle were not changed during this diagnostic task. No ML was implemented.

## Environment

- OS: Windows, PowerShell; Python 3.12.14 in project `.venv`.
- Playwright: 1.51.0; Chromium 134.0.6998.35.
- Browser: visible (`headless=false`), one page, concurrency 1, anonymous fresh context, self-identifying `ShopeeDataPlatform/0.1` user agent.
- Delay: at least 5 seconds between robots policy verification and the search navigation. No navigation retries, login, imported cookies, stealth or proxy rotation.
- Docker: NOT TESTED — Docker unavailable. This task did not change Docker.

## Local tests

- pytest: **PASS — 96 passed in 16.51 seconds** on 2026-10-12.
- Ruff: **PASS** — `python -m ruff check .`.
- Compile: **PASS** — `python -m compileall -q src databricks scripts`.
- Diagnostic tests cover evidence-based classifications, restriction/login detection, dynamic/private resource exclusions and the distinction between caught console errors and uncaught JavaScript failures.
- These are local tests; they do not establish live product ingestion success.

## Browser

**PASS — visible Chromium launched, navigated and closed cleanly.**

The fresh diagnostic ran at 09:46:12–09:46:44 on 2026-10-12 local time (02:46:12–02:46:44 UTC). robots.txt and the main public search document returned HTTP 200. robots policy allowed this search for the configured crawler agent. The previous October 4 evidence remains unchanged under `state/visible-diagnostic/live-20261004/`; that run also returned PUBLIC_PAGE_EMPTY.

Executed from the project directory:

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path (Get-Location) 'state/playwright-browsers'
.\.venv\Scripts\python.exe -m scripts.shopee_visible_diagnostic --output state/visible-diagnostic/live-20261012
```

## Shopee

Exact live diagnostic outcome: **PUBLIC_PAGE_EMPTY**.

Public products visible: **No**. Live products collected: **0**.

Final URL: `https://shopee.vn/search?keyword=tai+nghe+bluetooth&page=0`.

Final title: `Shopee Việt Nam | Mua và Bán Trên Ứng Dụng Di Động Hoặc Website`.

No explicit CAPTCHA, login redirect, main-document HTTP 401/403/429, or visible access-denied message was detected. This does not prove that automation was accepted; it only describes the observed response and rendered page.

### Rendering checkpoints

The HTML lengths are browser DOM measurements, not byte sizes of the saved full HTML files. All five captures have screenshots and rendered HTML. D/E are five/ten seconds after the completed C capture; actual elapsed navigation times are recorded below. Capture operations take time, so B was already `complete` when measured; it is a post-DOMContentLoaded observation, not the exact event timestamp.

| Capture | Seconds from navigation | readyState | Body HTML length | Body text length | Links | Divs | Scripts | Iframes | Main HTTP | Visible cards |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — immediately after navigation commit | 0.485 | loading | 0 | 0 | 0 | 0 | 16 | 0 | 200 | 0 |
| B — after DOMContentLoaded | 4.438 | complete | 7269 | 0 | 0 | 4 | 25 | 3 | 200 | 0 |
| C — after load | 4.860 | complete | 7269 | 0 | 0 | 4 | 25 | 3 | 200 | 0 |
| D — after 5 seconds | 10.032 | complete | 21038 | 0 | 0 | 4 | 32 | 3 | 200 | 0 |
| E — after 10 seconds | 15.047 | complete | 21038 | 0 | 0 | 4 | 32 | 3 | 200 | 0 |

A had an empty title; B–E had the final Shopee title. The URL remained the public search URL at every checkpoint. Visible product links and visible images were zero throughout. The final screenshot was visually inspected and shows a blank grey page.

### Console, JavaScript and public-resource findings

- 43 captured public document/static-resource responses: all HTTP 200. No captured public-resource 4xx/5xx response.
- 17 console messages: 11 debug, 4 warning, 2 error.
- Repeated caught/debug `TypeError: Cannot read properties of null (reading '9')` messages from the public `shopee__web_enhance_sap` JavaScript resource.
- Two failures launching `wvjbscheme://__WVJB_QUEUE_MESSAGE__` iframe documents, recorded as `net::ERR_ABORTED`; console mentions an unregistered handler and a required user gesture. These are custom-scheme failures, not failed HTTPS search navigation.
- Warnings include Canvas2D readback advice and `No available adapters.`
- Uncaught Playwright `pageerror` events: **0**. Screenshot/capture errors: **0**.

**Root cause remains unknown.** The document and public scripts loaded, but the application never produced visible search content. The caught script errors, custom-scheme failures and adapter warning are diagnostic signals; their causal relationship to the empty application has not been established. The evidence does not justify classifying this as CAPTCHA, login required, confirmed anti-bot blocking, confirmed network failure, or a selector mismatch.

XHR/fetch/WebSocket traffic and private API URLs were excluded from diagnostics. No request/response bodies, headers, tokens or cookies were read. Only robots policy, rendered public DOM, browser console/page-error messages and public-resource metadata were examined. URLs in resource logs omit query strings.

### Evidence files

Current evidence directory: `state/visible-diagnostic/live-20261012/`.

- `evidence.json`: URL, title, DOM metrics, status, timing, console messages, errors, failed requests and response statuses.
- `robots.txt`: policy observed before navigation.
- `A_navigation.html` / `.png`.
- `B_domcontentloaded.html` / `.png`.
- `C_load.html` / `.png`.
- `D_after_5s.html` / `.png`.
- `E_after_10s.html` / `.png`.

**Selector changes: not justified; none made.** No visible product content exists to support a parser update. The existing crawler interface remains unchanged and modular.

## Local pipeline

Production live pipeline: **NOT RUN** in this task because live products were unavailable. No live raw JSONL, Parquet, rejected records or pipeline metadata were produced by the standalone browser diagnostic. No normal-catalog cloud upload or job was triggered.

The historical local live pipeline ended with `NO_DATA`, run ID `9b4860a6fa6444869dcdf1e4b19c4ded`, duration 39.023166 seconds, zero products and metadata saved under `data/metadata/`. The new diagnostic refines that observation to `PUBLIC_PAGE_EMPTY` without changing stored historical results.

## Docker

Build and pipeline: **NOT TESTED — Docker unavailable.** No Docker work was requested or performed in this diagnostic task.

## Databricks

### Verified synthetic cloud acceptance

The following are **PASS, verified by the user and supplied in the current request**. They were not rerun by the agent during the browser diagnostic:

- Databricks OAuth: PASS.
- Serverless: PASS.
- Unity Catalog: PASS.
- Volume: PASS.
- Asset Bundle: PASS.
- Synthetic fixture upload: PASS.
- Synthetic cloud E2E: **PASS**.
- Bronze: PASS.
- Silver: PASS.
- Gold: PASS.

Verified test-catalog counts:

| Table | First processing run | Second run without new upload |
|---|---:|---:|
| `shopee_test.bronze.products_raw` | 2 | 2 |
| `shopee_test.silver.products` | 1 | 1 |
| `shopee_test.silver.products_rejected` | 1 | 1 |
| `shopee_test.gold.daily_product_metrics` | 1 | 1 |

These counts are synthetic test acceptance results, not live Shopee production counts. Other table counts were not supplied and are not invented.

Normal/local catalog remains `shopee`; the separate bundle `test` target overrides it to `shopee_test`. The schemas are `raw`, `bronze`, `silver`, `gold` and volume is `landing`. Synthetic cloud execution remains gated by the explicit allow flag and exact test catalog. No changes were made to those controls or to Bronze/Silver/Gold.

## Idempotency

**PASS — verified by the user.** A second processing-job run without another upload completed SUCCESS and the counts above remained stable.

Silver duplicate-key query: **0 rows**, verified by the user.

This acceptance task preserves those results and does not redesign the verified processing pipeline.

## Remaining blockers

Only live public Shopee ingestion remains unresolved: the visible browser produced no public product content. Longer arbitrary retries, selector guesses, private APIs, authentication workarounds and anti-bot bypass are not justified by this evidence.

Files changed for the October 12 rerun:

- `LIVE_ACCEPTANCE_REPORT.md` — fresh live observations and test results; user-verified cloud results retained.
- Generated evidence files under `state/visible-diagnostic/live-20261012/`.

The existing diagnostic script and tests were reused without changes. No crawler,
selector, authentication, bundle, processing or ML source was modified.

Next command: review the evidence, without triggering another crawl or production cloud job:

```powershell
Get-Content .\state\visible-diagnostic\live-20261012\evidence.json
```
