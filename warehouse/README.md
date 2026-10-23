# Foody analytical warehouse

The loader reads the existing enriched reviews and restaurant marts. It never
recomputes predictions. `schema_migrations` records SQL versions `001` and `002`.
`python -m src.cli warehouse-load` runs migrations and transactionally replaces
only the seven batch tables. `fact_review_events` is intentionally preserved,
with `event_id` as its replay-safe primary key.

`fact_review` counts distinct `review_id` values, while the enriched Parquet
retains every source observation. Daily and monthly rows are copied from their
existing marts; their counts must not be treated as raw observation counts.
PostgreSQL data resides in the `foody_postgres_data` Docker named volume.

Example analytical SQL:

```sql
SELECT r.restaurant_name, COUNT(*) AS reviews
FROM fact_review f JOIN dim_restaurant r USING (restaurant_key)
GROUP BY r.restaurant_name ORDER BY reviews DESC LIMIT 5;

SELECT d.year, d.month, ROUND(AVG(f.average_rating)::numeric, 2) AS mean_rating
FROM fact_restaurant_monthly f JOIN dim_date d ON d.date_key = f.month_key
WHERE f.average_rating IS NOT NULL
GROUP BY d.year, d.month ORDER BY d.year, d.month LIMIT 12;

SELECT a.aspect, SUM(a.predicted_review_count) AS predicted_reviews
FROM fact_aspect a GROUP BY a.aspect ORDER BY predicted_reviews DESC;

SELECT version, applied_at FROM schema_migrations ORDER BY version;
```
