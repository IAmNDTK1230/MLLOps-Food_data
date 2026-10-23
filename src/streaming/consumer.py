"""Validate, deduplicate, and persist simulated review events."""

import os
from time import perf_counter

from prometheus_client import Counter, Histogram, start_http_server
from pydantic import ValidationError

from src.streaming.events import TOPIC, ReviewEvent
from src.warehouse.loader import connection

CONSUMED = Counter("foody_kafka_events_consumed_total", "Valid review events consumed")
FAILED = Counter("foody_kafka_events_failed_total", "Invalid or failed review events")
INSERTED = Counter("foody_kafka_events_inserted_total", "Unique events inserted")
DUPLICATES = Counter("foody_kafka_events_duplicate_total", "Repeated event IDs ignored")
PROCESSING = Histogram("foody_kafka_processing_seconds", "Review-event processing time")


def persist(conn, event):
    """Reuse frozen batch predictions when available; do not infer or retrain."""
    from psycopg.types.json import Jsonb

    with conn.cursor() as cursor:
        cursor.execute("""INSERT INTO fact_review_events
            (event_id, review_id, restaurant_key, event_timestamp,
             predicted_sentiment, predicted_aspects, schema_version)
            SELECT %s, %s, %s, %s, predicted_sentiment, predicted_aspects, %s
            FROM fact_review WHERE review_id = %s
            ON CONFLICT (event_id) DO NOTHING""",
                       (event.event_id, event.review_id, event.restaurant_key,
                        event.timestamp, event.schema_version, event.review_id))
        inserted = cursor.rowcount
        if inserted == 0:
            cursor.execute("""INSERT INTO fact_review_events
                (event_id, review_id, restaurant_key, event_timestamp,
                 predicted_aspects, schema_version)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING""",
                           (event.event_id, event.review_id, event.restaurant_key,
                            event.timestamp, Jsonb([]), event.schema_version))
            inserted = cursor.rowcount
    return bool(inserted)


def run():
    from kafka import KafkaConsumer
    from kafka.errors import CommitFailedError

    start_http_server(8001)
    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=os.getenv("FOODY_KAFKA_BOOTSTRAP", "kafka:19092"),
        group_id="foody-review-warehouse-v1", auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    with connection() as conn:
        for message in consumer:
            started = perf_counter()
            try:
                event = ReviewEvent.model_validate_json(message.value)
            except ValidationError:
                FAILED.inc()
                PROCESSING.observe(perf_counter() - started)
                # A malformed event is permanent; skip it without logging review text.
                try:
                    consumer.commit()
                except CommitFailedError:
                    pass
                continue
            try:
                inserted = persist(conn, event)
                conn.commit()
                CONSUMED.inc()
                (INSERTED if inserted else DUPLICATES).inc()
            except Exception:
                conn.rollback()
                FAILED.inc()
                # Transient database failures must replay rather than lose events.
                raise
            finally:
                PROCESSING.observe(perf_counter() - started)
            try:
                consumer.commit()
            except CommitFailedError:
                # A broker restart can revoke membership after the DB commit.
                # The event_id key makes its later replay safe.
                pass


if __name__ == "__main__":
    run()
