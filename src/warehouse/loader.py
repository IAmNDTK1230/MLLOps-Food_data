"""Load only canonical, already-enriched Foody artifacts into PostgreSQL."""

import json
import math
import os
from datetime import date
from pathlib import Path

import pandas as pd

TABLES = (
    "dim_restaurant", "dim_date", "fact_review", "fact_restaurant_daily",
    "fact_restaurant_monthly", "fact_aspect", "fact_issue",
)
MART = Path("data/foody/mart")
ENRICHED = Path("data/foody/enriched/reviews.parquet")


def connection():
    import psycopg

    return psycopg.connect(os.environ.get(
        "FOODY_DATABASE_URL",
        "postgresql://foody:foody_local_demo@localhost:5433/foody",
    ))


def migrate(conn, directory=Path("warehouse/migrations")):
    conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
        version TEXT PRIMARY KEY, filename TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""")
    existing = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    applied = []
    for path in sorted(Path(directory).glob("[0-9]*.sql")):
        version = path.name.split("_", 1)[0]
        if version in existing:
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations(version, filename) VALUES (%s, %s)",
                     (version, path.name))
        applied.append(version)
    return applied


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _date_key(value):
    return int(value.strftime("%Y%m%d"))


def _copy(conn, table, columns, rows):
    with conn.cursor() as cursor, cursor.copy(
        f"COPY {table} ({', '.join(columns)}) FROM STDIN"
    ) as copy:
        for row in rows:
            copy.write_row(row)


def _metrics_rows(frame, monthly=False):
    for row in frame.itertuples(index=False):
        period = str(row.period) + ("-01" if monthly else "")
        yield (row.restaurant_key, _date_key(date.fromisoformat(period)),
               int(row.review_count), int(row.ai_review_count), int(row.rating_count),
               _number(row.average_rating), int(row.predicted_negative_count),
               int(row.predicted_neutral_count), int(row.predicted_positive_count))


def load(conn):
    """Transactionally replace batch facts; keep independently consumed events."""
    reviews = pd.read_parquet(ENRICHED).drop_duplicates("review_id")
    marts = {name: pd.read_parquet(MART / f"restaurant_{name}.parquet")
             for name in ("summary", "daily_metrics", "monthly_metrics",
                          "aspect_summary", "issue_summary")}
    daily, monthly = marts["daily_metrics"], marts["monthly_metrics"]
    dates = {v.date() for v in reviews.review_timestamp.dropna()}
    dates.update(date.fromisoformat(v) for v in daily.period)
    dates.update(date.fromisoformat(v + "-01") for v in monthly.period)
    # TRUNCATE is deliberately scoped to batch tables. Event history survives reloads.
    conn.execute("TRUNCATE fact_review, fact_restaurant_daily, fact_restaurant_monthly, "
                 "fact_aspect, fact_issue, dim_date, dim_restaurant")
    _copy(conn, "dim_restaurant", ("restaurant_key", "restaurant_name"),
          ((r.restaurant_key, r.restaurant_name) for r in
           marts["summary"].itertuples(index=False)))
    _copy(conn, "dim_date", ("date_key", "calendar_date", "day", "month", "year"),
          ((_date_key(d), d, d.day, d.month, d.year) for d in sorted(dates)))
    _copy(conn, "fact_review", ("review_id", "restaurant_key", "date_key", "rating",
                                "predicted_sentiment", "predicted_aspects", "review_text"),
          ((r.review_id, r.restaurant_key,
            _date_key(r.review_timestamp.date()) if pd.notna(r.review_timestamp) else None,
            (_number(r.raw_rating) if _number(r.raw_rating) is not None
             and 0 <= _number(r.raw_rating) <= 10 else None),
            r.predicted_sentiment if pd.notna(r.predicted_sentiment) else None,
            json.dumps(list(r.predicted_aspects), ensure_ascii=False),
            r.text if pd.notna(r.text) else None)
           for r in reviews.itertuples(index=False)))
    metric_cols = ("restaurant_key", "date_key", "review_count",
                   "ai_review_count", "rating_count", "average_rating",
                   "predicted_negative_count", "predicted_neutral_count",
                   "predicted_positive_count")
    _copy(conn, "fact_restaurant_daily", metric_cols, _metrics_rows(daily))
    _copy(conn, "fact_restaurant_monthly",
          ("restaurant_key", "month_key", *metric_cols[2:]), _metrics_rows(monthly, True))
    for name, label in (("aspect", "aspect"), ("issue", "issue")):
        _copy(conn, f"fact_{name}",
              ("restaurant_key", label, "predicted_review_count", "ai_review_count"),
              ((r.restaurant_key, getattr(r, label), int(r.predicted_review_count),
                int(r.ai_review_count)) for r in marts[f"{name}_summary"].itertuples(index=False)))
    return counts(conn)


def counts(conn):
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (*TABLES, "fact_review_events")}


def main():
    with connection() as conn:
        applied = migrate(conn)
        result = load(conn)
    print(json.dumps({"migrations_applied": applied, "rows": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
