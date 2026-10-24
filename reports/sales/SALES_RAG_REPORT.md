# Sales and review acceptance

## Source and identity

`order_cleaning.csv` has 400,000 unique orders dated 2025-01-01 through 2026-12-31 and 3,647 sales restaurant IDs. The six expected columns are present. There are no invalid dates, duplicate order IDs, negative or zero amounts. Restaurant names are missing on 3,603 rows. Delivered / Cancelled / Refunded counts are 366,164 / 23,783 / 10,053. Amount min / median / p95 / p99 / max are 816 / 97,104 / 347,072 / 597,040 / 4,893,824 VND.

Foody reviews have no official restaurant ID. A Unicode NFC, casefold, whitespace-only normalized name can be linked only when it is unique on both sides and a sales ID has one consistent nonempty name. This yielded **22 verified matches**, **34 ambiguous IDs**, **3,591 unmatched sales IDs**, and **2,088 review-only restaurants**. No ambiguous rows are joined. In particular, a sale from one Hanuri branch is not joined to the default Hanuri review branch.

## Warehouse and metric policy

Migration `003` adds `dim_sales_restaurant`, `fact_order`, `sales_review_restaurant_map`, `sales_source_state`, and five SQL marts. The transaction loaded 400,000 orders and 22 verified mappings. The observed daily / weekly / monthly views have 354,775 / 221,946 / 75,269 rows as of 2026-10-24 Vietnam time. **10,769 future-dated orders** remain in the audit fact table but are excluded from observed metrics.

Realized revenue is Delivered `sales_amount`; Cancelled and Refunded amounts are excluded. AOV divides Delivered revenue by Delivered order count. Cancellation and refund rates divide respective order counts by all observed orders. Completed months are the default. An explicit current-month comparison uses the same elapsed days in the prior month. SQL produces deterministic monthly summary documents on demand (75,269 available); no sales Chroma/BM25 was built because exact numeric queries require SQL and no semantic sales retrieval route is needed.

## Live acceptance

For Thai Market Restaurant - 95 Cao Thắng, sales ID `1853`, May 2025 Delivered revenue was **492,320 VND** over four orders, up **8.25%** from April's 454,784 VND. The SQL CLI and the Streamlit browser returned this value. Its observed summary has 105 orders, 98 Delivered, 6 Cancelled, 1 Refunded, AOV 144,456.98 VND, and a 5.71% cancellation rate. Its highest completed-month revenue was August 2025 at 1,959,760 VND.

A cross-analysis for May 2025 found **one negative review from the same verified restaurant and month**. The Streamlit answer showed the SQL numbers and cited `[E1]` for complaints about small portions, poor taste and curry/tomyum ingredients, then explicitly stated that the overlap does not prove causation. The CLI returned an extractive fallback; the Streamlit run returned a grounded Qwen response, as recorded by SQLite LLMOps. Review-only routing and the accepted RAG V3 implementation were left intact. Preflight still reports 32,457 documents and 32,457 vectors.

SQLite LLMOps recorded `sales_sql`, `cross_analysis`, and the existing `rag_v3_request`. Prometheus scraped `foody_sales_queries_total{status="ok"}=1`, `foody_cross_analysis_requests_total{status="grounded"}=1`, and one sales SQL latency observation from Streamlit. The existing Grafana dashboard was provisioned with three panels for those counters and sales latency. The old dashboard panels and metrics remain.

The loader's second run returned `unchanged` with 400,000 orders and no new migration. The final Review-2 image reported warehouse schema version `003`; FastAPI and Streamlit health endpoints returned HTTP 200. Grafana's API confirmed all 10 original panels plus three new ones. The complete suite passed **314 tests**; Ruff, source compilation, Compose validation, and demo-preflight passed.

Additional live SQL questions returned August 2025 as Thai Market's highest completed month at 1,959,760 VND, a 5.71% cancellation rate (6 of 105 observed orders), and no orders in the current week. A largest-decline cross question found July 2026 at -89.58% but **no same-month reviews**, so it explicitly declined to infer a complaint cause. A separate review-only Hanuri service question still returned a cited `[E1]` through the accepted RAG V3 evidence fallback.

Finalization checks reran 62 focused Sales/RAG/Review-2 tests and 314 full tests. A live cross question for unmatched sales ID `1` returned `unmatched_restaurant` with no review evidence; it did not attempt a name-based join. The current demo script now includes the verified Sales / Review / Cross sequence.

## Storage audit and safe cleanup

The project measured **10,369,016,676 bytes** before cleanup and **9,979,122,711 bytes** after, saving **389,893,965 bytes**. Removed only redundant local package clones (`.flow-tools`, `.verify-tools`, `.lint-tools`, `.pytest-tools`) and Python/pytest/Ruff caches. The active `.venv` contains the required packages. `release/` was already empty, so no ZIP or extracted release was removed. The 2.10 GB active Qwen 3B, the 235 MB MiniLM ONNX, current Foody data, both review indexes, SQLite state, PostgreSQL volume, Terraform cache, and experimental models of distinct identity were retained. No Docker prune was run.

The 400,000-order source CSV, model weights, review indexes, SQLite DB, and Docker volumes remain excluded from Git. Unrelated pre-existing untracked documentation was left untouched.

The only persistent architecture document is [RAG_SALES_ARCHITECTURE.md](../../docs/RAG_SALES_ARCHITECTURE.md). The sales extension did not retrain sentiment, aspect, MiniLM, or Qwen; did not replay 400,000 orders through Kafka; and did not build a new review index.
