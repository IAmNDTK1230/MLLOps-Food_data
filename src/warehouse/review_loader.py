"""Idempotent load of the accepted direct-ID review corpus beside legacy facts."""

import json

import pandas as pd

from src.foody.review_paths import DATA_VERSION, ENRICHED, MANIFEST
from src.foody.schema import digest
from src.warehouse.loader import connection, migrate


def load(conn, path=ENRICHED):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["data_version"] != DATA_VERSION or manifest["row_count"] <= 0:
        raise ValueError("Accepted review corpus manifest is missing or invalid")
    reviews = pd.read_parquet(path)
    if len(reviews) != manifest["row_count"] or reviews.review_id.duplicated().any():
        raise ValueError("Enriched review count or IDs differ from accepted manifest")
    ids = set(reviews.restaurant_id.astype(str))
    found = {row[0] for row in conn.execute(
        "SELECT restaurant_id FROM dim_sales_restaurant WHERE restaurant_id = ANY(%s)",
        (list(ids),)).fetchall()}
    if found != ids:
        raise ValueError("Review restaurant IDs are absent from loaded Sales dimension")
    sha = digest(path)
    previous = conn.execute("""SELECT source_sha256,review_count FROM review_source_state_v2
        WHERE source_version=%s""", (DATA_VERSION,)).fetchone()
    current = conn.execute("SELECT COUNT(*) FROM fact_review_v2").fetchone()[0]
    if previous == (sha, len(reviews)) and current == len(reviews):
        return {"status": "unchanged", "reviews": current, "restaurants": len(ids)}
    names = reviews[["restaurant_id", "restaurant_name"]].drop_duplicates()
    if names.restaurant_id.duplicated().any():
        raise ValueError("Review restaurant ID maps to multiple names")
    with conn.cursor() as cursor:
        cursor.executemany("""INSERT INTO dim_review_restaurant_v2
            (restaurant_id,restaurant_name,source_version) VALUES (%s,%s,%s)
            ON CONFLICT (restaurant_id) DO UPDATE SET
              restaurant_name=EXCLUDED.restaurant_name,
              source_version=EXCLUDED.source_version""",
            ((str(row.restaurant_id), row.restaurant_name, DATA_VERSION)
             for row in names.itertuples(index=False)))
        dates = sorted({stamp.date() for stamp in reviews.review_time})
        cursor.executemany("""INSERT INTO dim_date(date_key,calendar_date,day,month,year)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT (date_key) DO NOTHING""",
            ((int(day.strftime("%Y%m%d")), day, day.day, day.month, day.year)
             for day in dates))
        conn.execute("DELETE FROM fact_review_v2")
        with cursor.copy("""COPY fact_review_v2
            (review_id,restaurant_id,date_key,rating,predicted_sentiment,
             sentiment_confidence,predicted_aspects,review_text) FROM STDIN""") as copy:
            for row in reviews.itertuples(index=False):
                copy.write_row((row.review_id, str(row.restaurant_id),
                                int(row.review_time.strftime("%Y%m%d")), float(row.rating),
                                row.predicted_sentiment, float(row.sentiment_confidence),
                                json.dumps(list(row.predicted_aspects), ensure_ascii=False),
                                row.comment))
        conn.execute("""INSERT INTO review_source_state_v2
            (source_version,source_sha256,review_count) VALUES (%s,%s,%s)
            ON CONFLICT (source_version) DO UPDATE SET
              source_sha256=EXCLUDED.source_sha256,
              review_count=EXCLUDED.review_count,loaded_at=now()""",
            (DATA_VERSION, sha, len(reviews)))
    return {"status": "loaded", "reviews": len(reviews), "restaurants": len(ids)}


def main():
    with connection() as conn:
        migrate(conn)
        result = load(conn)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
