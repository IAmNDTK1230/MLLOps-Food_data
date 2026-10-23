CREATE INDEX IF NOT EXISTS ix_fact_review_restaurant_date
  ON fact_review(restaurant_key, date_key);
CREATE INDEX IF NOT EXISTS ix_fact_review_sentiment
  ON fact_review(predicted_sentiment);
CREATE INDEX IF NOT EXISTS ix_fact_events_review
  ON fact_review_events(review_id);
