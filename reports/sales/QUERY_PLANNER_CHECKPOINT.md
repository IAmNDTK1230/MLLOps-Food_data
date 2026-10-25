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

## Phase 2 implemented

- `MAX_MIN_PERIOD` selects the highest or lowest observed monthly Delivered revenue within a named year.
- `PERIOD_TREND` reports a bounded monthly series for an explicit range or the six most recently completed months.
- `RANGE_AGGREGATE` sums an explicit month range or named quarter. `YEAR_AGGREGATE` sums a year or calculates average monthly revenue from its observed complete months.
- `MULTI_STEP_ANALYSIS` finds the largest month over month decline in a bounded range using the month before the range as the first reference. It also finds a year's highest and lowest months and their absolute spread.
- The current year uses only completed months for yearly extrema and aggregates, and the answer names the effective range. All effective months must have SQL facts; missing periods return `no_data` instead of a partial or invented total. Explicit future months remain rejected by the existing query guard.
- The same planner produces an intent, metric, bounded periods, aggregation, and ordered analytical steps. The executor uses only parameterized `monthly_sales` SQL and deterministic Python math. Technical Details display the plan and SQL mode.
- Forecasting, ML anomaly detection, restaurant ranking, cross-review changes, retraining, and new models remain out of scope.

### Phase 2 validation

- Focused Phase 1, Phase 2, existing Sales, and Sales UI tests: **58 passed**.
- Ruff on changed files: **PASS**.
- Full pytest suite: **367 passed**, one dependency deprecation warning. The unrestricted local run was required because the sandbox stalled on a minimal `asyncio.sleep(0)` call in an unrelated crawler test.
- `compileall -q src app scripts`: **PASS**.
- `python -m src.cli demo-preflight`: **PASS**.
- A live SQL result was not obtained because the local PostgreSQL connection on port 5433 was unavailable in this execution environment. The focused tests use controlled SQL facts and verify requested periods and calculations.
