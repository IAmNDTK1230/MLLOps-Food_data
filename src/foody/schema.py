"""Versioned identities and shared IO contracts."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("data/foody/source/foody_reviews.csv")
SNAPSHOT = SOURCE.with_name("foody_reviews_20261020.csv")
PROCESSED = Path("data/foody/processed/reviews.parquet")
ENRICHED = Path("data/foody/enriched/reviews.parquet")
REPORTS = Path("reports/foody_intelligence")
IDENTITY_TYPE = "derived_from_restaurant_name"
MAPPING = {
    "User": "user_name",
    "Review Time": "raw_review_time",
    "Rating": "raw_rating",
    "Comment": "text_original",
    "RestaurantName": "restaurant_name",
}


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(version, *parts):
    payload = json.dumps([version, *parts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def restaurant_key(name):
    normalized = normalize(name).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else None


def review_id(key, user, raw_time, original_comment):
    return identity("foody-review-v1", key, user, raw_time, original_comment)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        + "\n", encoding="utf-8"
    )


def save_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
