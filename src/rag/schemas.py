"""Strict service contracts; review evidence is kept separate from model prose."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    restaurant_key: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    sentiment: Literal["positive", "neutral", "negative"] | None = None
    aspect: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def ordered_dates(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be <= date_to")
        return self


class ReviewHit(BaseModel):
    review_id: str
    restaurant_name: str
    text: str
    score: float
    metadata: dict


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str
    restaurant_name: str
    score: float


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    evidence: list[Evidence]
    model: str
    prompt_version: str


class HealthReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_summary: str
    strengths: list[str]
    main_complaints: list[str]
    aspect_trends: list[str]
    warnings: list[str]
    recommendations: list[str]
    evidence_review_ids: list[str]
