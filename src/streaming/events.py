"""Versioned review events made only from canonical source fields."""

import hashlib
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

TOPIC = "foody.review.events"
SCHEMA_VERSION = "1.0"


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = Field(pattern=r"^1\.0$")
    event_id: str = Field(min_length=64, max_length=64)
    review_id: str = Field(min_length=64, max_length=64)
    restaurant_key: str = Field(min_length=64, max_length=64)
    restaurant_name: str
    timestamp: datetime | None
    rating: float | None
    comment: str


def from_review(row):
    rating = pd.to_numeric(row.raw_rating, errors="coerce")
    return ReviewEvent(
        schema_version=SCHEMA_VERSION,
        event_id=hashlib.sha256(f"foody-review-event-v1:{row.review_id}".encode()).hexdigest(),
        review_id=row.review_id,
        restaurant_key=row.restaurant_key,
        restaurant_name=row.restaurant_name,
        timestamp=pd.Timestamp(row.review_timestamp).to_pydatetime()
        if pd.notna(row.review_timestamp) else None,
        rating=float(rating) if pd.notna(rating) and 0 <= rating <= 10 else None,
        comment=row.text if isinstance(row.text, str) else "",
    )
