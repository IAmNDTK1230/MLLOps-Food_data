-- Separate review dimension preserves the legacy name-derived review warehouse.
CREATE TABLE IF NOT EXISTS dim_review_restaurant_v2 (
  restaurant_id TEXT PRIMARY KEY REFERENCES dim_sales_restaurant(restaurant_id),
  restaurant_name TEXT NOT NULL,
  source_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_review_v2 (
  review_id TEXT PRIMARY KEY,
  restaurant_id TEXT NOT NULL REFERENCES dim_review_restaurant_v2(restaurant_id),
  date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
  rating NUMERIC(4,2) NOT NULL,
  predicted_sentiment TEXT,
  sentiment_confidence DOUBLE PRECISION,
  predicted_aspects JSONB NOT NULL DEFAULT '[]',
  review_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fact_review_v2_restaurant_date
  ON fact_review_v2(restaurant_id, date_key);
CREATE TABLE IF NOT EXISTS review_source_state_v2 (
  source_version TEXT PRIMARY KEY,
  source_sha256 TEXT NOT NULL,
  review_count INTEGER NOT NULL,
  loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
