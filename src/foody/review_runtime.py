"""Explicit review corpus switch; the legacy indexes remain available for rollback."""

import json
import os
from pathlib import Path

import yaml

from src.foody.review_paths import DATA_VERSION, ENRICHED, MANIFEST

CONFIG = Path("config/review_corpus.yaml")
VECTOR = Path("data/foody/vector_store_review_2025_2026")
LEXICAL = Path("data/foody/lexical_index_review_2025_2026")


def new_corpus_active():
    configured = yaml.safe_load(CONFIG.read_text("utf-8"))["active"] if CONFIG.exists() else "legacy"
    return os.getenv("FOODY_REVIEW_CORPUS", configured) == DATA_VERSION


def active_manifest():
    if new_corpus_active():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    legacy = Path("data/foody/vector_store/index_manifest.json")
    return json.loads(legacy.read_text(encoding="utf-8"))


def review_table():
    import pandas as pd

    if new_corpus_active():
        frame = pd.read_parquet(ENRICHED)
        return frame.rename(columns={"review_time": "review_timestamp",
                                     "rating": "raw_rating", "comment": "text"})
    return pd.read_parquet("data/foody/enriched/reviews.parquet")


def review_restaurants():
    frame = review_table()
    if new_corpus_active():
        return frame[["restaurant_id", "restaurant_name"]].drop_duplicates()
    return frame[["restaurant_key", "restaurant_name"]].drop_duplicates()


def review_identity(name_or_id):
    frame = review_restaurants()
    key = "restaurant_id" if new_corpus_active() else "restaurant_key"
    selected = frame.loc[frame[key].astype(str).eq(str(name_or_id)) |
                         frame.restaurant_name.eq(str(name_or_id))]
    if len(selected) != 1:
        raise ValueError(f"Restaurant not found or ambiguous: {name_or_id}")
    return str(selected.iloc[0][key]), selected.iloc[0]
