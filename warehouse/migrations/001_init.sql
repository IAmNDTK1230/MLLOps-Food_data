CREATE TABLE IF NOT EXISTS dim_restaurant (
  restaurant_key TEXT PRIMARY KEY,
  restaurant_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_date (
  date_key INTEGER PRIMARY KEY,
  calendar_date DATE NOT NULL UNIQUE,
  day SMALLINT NOT NULL,
  month SMALLINT NOT NULL,
  year SMALLINT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_review (
  review_id TEXT PRIMARY KEY,
  restaurant_key TEXT NOT NULL REFERENCES dim_restaurant(restaurant_key),
  date_key INTEGER REFERENCES dim_date(date_key),
  rating NUMERIC(4,2),
  predicted_sentiment TEXT,
  predicted_aspects JSONB NOT NULL DEFAULT '[]',
  review_text TEXT
);
CREATE TABLE IF NOT EXISTS fact_restaurant_daily (
  restaurant_key TEXT NOT NULL REFERENCES dim_restaurant(restaurant_key),
  date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
  review_count INTEGER NOT NULL,
  ai_review_count INTEGER NOT NULL,
  rating_count INTEGER NOT NULL,
  average_rating DOUBLE PRECISION,
  predicted_negative_count INTEGER NOT NULL,
  predicted_neutral_count INTEGER NOT NULL,
  predicted_positive_count INTEGER NOT NULL,
  PRIMARY KEY (restaurant_key, date_key)
);
CREATE TABLE IF NOT EXISTS fact_restaurant_monthly (
  restaurant_key TEXT NOT NULL REFERENCES dim_restaurant(restaurant_key),
  month_key INTEGER NOT NULL REFERENCES dim_date(date_key),
  review_count INTEGER NOT NULL,
  ai_review_count INTEGER NOT NULL,
  rating_count INTEGER NOT NULL,
  average_rating DOUBLE PRECISION,
  predicted_negative_count INTEGER NOT NULL,
  predicted_neutral_count INTEGER NOT NULL,
  predicted_positive_count INTEGER NOT NULL,
  PRIMARY KEY (restaurant_key, month_key)
);
CREATE TABLE IF NOT EXISTS fact_aspect (
  restaurant_key TEXT NOT NULL REFERENCES dim_restaurant(restaurant_key),
  aspect TEXT NOT NULL,
  predicted_review_count INTEGER NOT NULL,
  ai_review_count INTEGER NOT NULL,
  PRIMARY KEY (restaurant_key, aspect)
);
CREATE TABLE IF NOT EXISTS fact_issue (
  restaurant_key TEXT NOT NULL REFERENCES dim_restaurant(restaurant_key),
  issue TEXT NOT NULL,
  predicted_review_count INTEGER NOT NULL,
  ai_review_count INTEGER NOT NULL,
  PRIMARY KEY (restaurant_key, issue)
);
CREATE TABLE IF NOT EXISTS fact_review_events (
  event_id TEXT PRIMARY KEY,
  review_id TEXT NOT NULL,
  restaurant_key TEXT NOT NULL,
  event_timestamp TIMESTAMPTZ,
  predicted_sentiment TEXT,
  predicted_aspects JSONB NOT NULL DEFAULT '[]',
  schema_version TEXT NOT NULL,
  processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
