-- Sales IDs are source-owned. Review keys are derived from names and linked only
-- through audited, one-to-one exact normalized-name matches.
CREATE TABLE IF NOT EXISTS dim_sales_restaurant (
  restaurant_id TEXT PRIMARY KEY,
  restaurant_name TEXT
);

CREATE TABLE IF NOT EXISTS sales_review_restaurant_map (
  restaurant_id TEXT PRIMARY KEY REFERENCES dim_sales_restaurant(restaurant_id),
  restaurant_key TEXT NOT NULL UNIQUE REFERENCES dim_restaurant(restaurant_key),
  match_method TEXT NOT NULL CHECK (match_method = 'unique_normalized_name')
);

CREATE TABLE IF NOT EXISTS fact_order (
  order_id TEXT PRIMARY KEY,
  restaurant_id TEXT NOT NULL REFERENCES dim_sales_restaurant(restaurant_id),
  date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
  sales_amount NUMERIC(14,2) NOT NULL CHECK (sales_amount >= 0),
  order_status TEXT NOT NULL CHECK (order_status IN ('Delivered', 'Cancelled', 'Refunded'))
);
CREATE INDEX IF NOT EXISTS ix_fact_order_restaurant_date
  ON fact_order(restaurant_id, date_key);
CREATE INDEX IF NOT EXISTS ix_fact_order_status_date
  ON fact_order(order_status, date_key);

CREATE TABLE IF NOT EXISTS sales_source_state (
  source_name TEXT PRIMARY KEY,
  source_sha256 TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW restaurant_daily_sales AS
SELECT o.restaurant_id, d.calendar_date AS period_start,
       COUNT(*)::INTEGER AS order_count,
       COALESCE(SUM(o.sales_amount),0) AS gross_sales,
       COALESCE(SUM(o.sales_amount) FILTER (WHERE o.order_status='Delivered'),0) AS delivered_sales,
       COUNT(*) FILTER (WHERE o.order_status='Delivered')::INTEGER AS delivered_count,
       COUNT(*) FILTER (WHERE o.order_status='Cancelled')::INTEGER AS cancelled_count,
       COUNT(*) FILTER (WHERE o.order_status='Refunded')::INTEGER AS refunded_count,
       ROUND(COALESCE(SUM(o.sales_amount) FILTER (WHERE o.order_status='Delivered'),0)
             / NULLIF(COUNT(*) FILTER (WHERE o.order_status='Delivered'),0),2) AS aov,
       ROUND(COUNT(*) FILTER (WHERE o.order_status='Cancelled')::NUMERIC / COUNT(*),4) AS cancellation_rate,
       ROUND(COUNT(*) FILTER (WHERE o.order_status='Refunded')::NUMERIC / COUNT(*),4) AS refund_rate
FROM fact_order o JOIN dim_date d ON d.date_key=o.date_key
WHERE d.calendar_date <= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::DATE
GROUP BY o.restaurant_id,d.calendar_date;

CREATE OR REPLACE VIEW restaurant_weekly_sales AS
WITH weekly AS (
  SELECT restaurant_id, DATE_TRUNC('week',period_start)::DATE AS period_start,
         SUM(order_count)::INTEGER AS order_count, SUM(gross_sales) AS gross_sales,
         SUM(delivered_sales) AS delivered_sales, SUM(delivered_count)::INTEGER AS delivered_count,
         SUM(cancelled_count)::INTEGER AS cancelled_count, SUM(refunded_count)::INTEGER AS refunded_count
  FROM restaurant_daily_sales GROUP BY restaurant_id,DATE_TRUNC('week',period_start)::DATE
)
SELECT cur.*,
       ROUND(cur.delivered_sales / NULLIF(cur.delivered_count,0),2) AS aov,
       ROUND(cur.cancelled_count::NUMERIC / NULLIF(cur.order_count,0),4) AS cancellation_rate,
       ROUND(cur.refunded_count::NUMERIC / NULLIF(cur.order_count,0),4) AS refund_rate,
       prev.delivered_sales AS previous_delivered_sales,
       ROUND((cur.delivered_sales-prev.delivered_sales) / NULLIF(prev.delivered_sales,0),4)
         AS wow_growth
FROM weekly cur LEFT JOIN weekly prev ON prev.restaurant_id=cur.restaurant_id
 AND prev.period_start=cur.period_start-7;

CREATE OR REPLACE VIEW restaurant_monthly_sales AS
WITH monthly AS (
  SELECT restaurant_id, DATE_TRUNC('month',period_start)::DATE AS period_start,
         SUM(order_count)::INTEGER AS order_count, SUM(gross_sales) AS gross_sales,
         SUM(delivered_sales) AS delivered_sales, SUM(delivered_count)::INTEGER AS delivered_count,
         SUM(cancelled_count)::INTEGER AS cancelled_count, SUM(refunded_count)::INTEGER AS refunded_count
  FROM restaurant_daily_sales GROUP BY restaurant_id,DATE_TRUNC('month',period_start)::DATE
)
SELECT cur.*,
       ROUND(cur.delivered_sales / NULLIF(cur.delivered_count,0),2) AS aov,
       ROUND(cur.cancelled_count::NUMERIC / NULLIF(cur.order_count,0),4) AS cancellation_rate,
       ROUND(cur.refunded_count::NUMERIC / NULLIF(cur.order_count,0),4) AS refund_rate,
       prev.delivered_sales AS previous_delivered_sales,
       ROUND((cur.delivered_sales-prev.delivered_sales) / NULLIF(prev.delivered_sales,0),4)
         AS mom_growth
FROM monthly cur LEFT JOIN monthly prev ON prev.restaurant_id=cur.restaurant_id
 AND prev.period_start=(cur.period_start-INTERVAL '1 month')::DATE;

CREATE OR REPLACE VIEW restaurant_sales_summary AS
SELECT restaurant_id, SUM(order_count)::INTEGER AS order_count,
       SUM(gross_sales) AS gross_sales, SUM(delivered_sales) AS delivered_sales,
       SUM(delivered_count)::INTEGER AS delivered_count,
       SUM(cancelled_count)::INTEGER AS cancelled_count,
       SUM(refunded_count)::INTEGER AS refunded_count,
       ROUND(SUM(delivered_sales)/NULLIF(SUM(delivered_count),0),2) AS aov,
       ROUND(SUM(cancelled_count)::NUMERIC/NULLIF(SUM(order_count),0),4) AS cancellation_rate,
       ROUND(SUM(refunded_count)::NUMERIC/NULLIF(SUM(order_count),0),4) AS refund_rate
FROM restaurant_daily_sales GROUP BY restaurant_id;

CREATE OR REPLACE VIEW restaurant_order_status_summary AS
SELECT restaurant_id, order_status, COUNT(*)::INTEGER AS order_count,
       SUM(sales_amount) AS gross_sales
FROM fact_order o JOIN dim_date d ON d.date_key=o.date_key
WHERE d.calendar_date <= (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::DATE
GROUP BY o.restaurant_id,o.order_status;
