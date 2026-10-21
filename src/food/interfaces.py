"""Definitions only. No classifier, Qwen client, embeddings, or training calls."""

from typing import Literal

from pydantic import BaseModel, Field


class StructuredReviewAnalysis(BaseModel):
    sentiment: Literal["negative", "neutral", "positive"]
    aspect: str  # Must be validated against actual observed aspect labels at evaluation.
    food_quality: str | None = None
    service_experience: str | None = None
    topics: list[str] = Field(default_factory=list)


EXPERIMENT_PLAN = {
    "status": "definitions_only_not_executed",
    "dataset": "vn_msasar_v3",
    "canonical_splits": "data/food/splits/reviews.parquet",
    "unit": "sentence/aspect pair",
    "inputs": ["text", "aspect"],
    "primary_task": "aspect-conditioned sentiment; intensity collapsed by sign",
    "baseline_rule": "Vietnamese rules; lexicon/negation rules frozen using train/validation only",
    "baseline_tfidf": {
        "method": "TF-IDF + Logistic Regression",
        "class_weight": "balanced",
        "fit_vocabulary": "train only",
        "input": "text features plus supplied aspect indicator",
    },
    "proposed_v1": "Qwen structured review analysis; no execution yet",
    "metrics": [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "inference_latency_ms",
    ],
    "primary_metric": "macro_f1",
    "quantitative_fields": {
        "sentiment": "gold normalized sentiment",
        "aspect": "gold source aspect; optional separate detection evaluation, never count supplied aspect as predicted accuracy",
    },
    "enrichment_only": ["food_quality", "service_experience", "topics"],
    "imbalance": {
        "first_strategy": "class_weight=balanced calculated from TRAIN labels only",
        "smote": "not enabled",
        "after_results": "not available; no balancing/model experiment run",
    },
    "external_only": ["ntc_scv", "vlsp2018_restaurant"],
    "selection": "validation macro F1; evaluate frozen selections on the same canonical test split",
}
