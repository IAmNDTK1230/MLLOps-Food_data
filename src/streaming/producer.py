"""Replay a bounded sample of real Foody reviews; never stream the whole file by default."""

import json
import os

import pandas as pd

from src.streaming.events import TOPIC, from_review


def produce(limit=20, bootstrap=None):
    if not 1 <= limit <= 1000:
        raise ValueError("--limit must be between 1 and 1000")
    from kafka import KafkaProducer

    frame = pd.read_parquet("data/foody/enriched/reviews.parquet").drop_duplicates("review_id")
    producer = KafkaProducer(
        bootstrap_servers=bootstrap or os.getenv("FOODY_KAFKA_BOOTSTRAP", "127.0.0.1:9092"),
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
        acks="all", retries=3,
    )
    try:
        for row in frame.head(limit).itertuples(index=False):
            event = from_review(row)
            producer.send(TOPIC, key=event.event_id.encode(),
                          value=event.model_dump(mode="json")).get(timeout=20)
        producer.flush(timeout=20)
    finally:
        producer.close()
    return {"topic": TOPIC, "sent": min(limit, len(frame)), "schema_version": "1.0"}


def main(limit=20):
    print(json.dumps(produce(limit), indent=2))
    return 0
