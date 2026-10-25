# Sales Query Planner checkpoint — Phase 1

Base commit: `35986b0e1fe4c81b4eac107eca2a6198464ea879` on the current checkout.

## Implemented

- `SINGLE_PERIOD_METRIC` preserves the existing simple Sales execution path.
- `COMPARE_PERIODS` extracts two explicit months, including a shared year such as “tháng 2 với tháng 3 năm 2026”. Explicit years in `YYYY-MM`, `MM/YYYY`, and “tháng M năm YYYY” are supported.
- A single explicit month with “so với tháng trước” or MoM resolves the preceding month, including the December/January boundary. “Cùng kỳ năm trước” or YoY resolves the same month one year earlier.
- Comparison metrics: Delivered revenue, Delivered AOV, and cancellation rate. The answer states both observed values and their signed difference. Revenue and AOV also show relative change when the reference value is nonzero; cancellation rate uses percentage points.
- Each period is looked up using the existing parameterized `monthly_sales` SQL function. Python computes the difference. No LLM-generated SQL is used.
- Missing data, invalid months, ambiguous years, more than two months, and zero reference values do not produce an invented growth percentage. Future months are rejected by the existing SQL query guard.
- The existing Sales, Review RAG, and Cross Analysis routes remain separate. Phase 1 does not alter Sales warehouse data or model behavior.

## Validation

- Focused Phase 1 and existing Sales tests: **40 passed**.
- Full pytest suite: **355 passed**, one dependency deprecation warning.
- Ruff: **PASS**.
- `compileall -q src app scripts`: **PASS**.
- `python -m src.cli demo-preflight`: **PASS**.

## Deferred Phase 2

`MAX_MIN_PERIOD`, `PERIOD_TREND`, range aggregation, yearly totals, average monthly revenue, and multi-step analytical plans are not implemented in this checkpoint.
