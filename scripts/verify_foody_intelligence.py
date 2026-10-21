"""Read-only reconciliation of executed Foody outputs; never fit or predict."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.foody.schema import ENRICHED, MAPPING, PROCESSED, REPORTS, SNAPSHOT, SOURCE, digest
from src.foody.validation import load_source


def main():
    raw = load_source(SOURCE)
    processed = pd.read_parquet(PROCESSED)
    enriched = pd.read_parquet(ENRICHED)
    assert digest(SOURCE) == digest(SNAPSHOT)
    assert len(raw) == len(processed) == len(enriched)
    for original, canonical in MAPPING.items():
        assert raw[original].tolist() == processed[canonical].tolist()
        assert raw[original].tolist() == enriched[canonical].tolist()
    assert processed.review_id.tolist() == enriched.review_id.tolist()
    assert enriched.loc[~enriched.valid_for_ai, "predicted_sentiment"].isna().all()
    assert enriched.loc[enriched.valid_for_ai, "predicted_sentiment"].notna().all()
    assert set(enriched.predicted_sentiment.dropna()) <= {"negative", "neutral", "positive"}
    assert enriched.groupby("observation_id").predicted_sentiment.nunique().le(1).all()
    for _, group in processed.groupby("observation_id", sort=False):
        assert group.duplicate_count.eq(len(group)).all()
        assert list(group.source_rows.iloc[0]) == group.source_row.tolist()
    valid = enriched.loc[enriched.valid_for_ai].drop_duplicates("review_id")
    business = enriched.drop_duplicates("review_id")
    summary = pd.read_parquet("data/foody/mart/restaurant_summary.parquet")
    assert summary.review_count.sum() == len(business)
    assert summary.ai_review_count.sum() == len(valid)
    assert summary.source_row_count.sum() == len(raw)
    numeric_rating = pd.to_numeric(business.raw_rating, errors="coerce")
    assert np.isclose(
        (summary.average_rating * summary.rating_count).sum() / summary.rating_count.sum(),
        numeric_rating[numeric_rating.between(0, 10)].mean(),
    )
    for period in ("daily", "monthly"):
        table = pd.read_parquet(f"data/foody/mart/restaurant_{period}_metrics.parquet")
        assert table.review_count.sum() == business.review_timestamp.notna().sum()
    documents = [
        json.loads(line) for line in Path("data/foody/rag/documents.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {doc["id"] for doc in documents} == set(valid.review_id)
    assert len(documents) == len(valid)
    lookup = valid.set_index("review_id")
    for doc in documents:
        row = lookup.loc[doc["id"]]
        assert doc["text"] == row.text
        assert doc["metadata"]["restaurant_key"] == row.restaurant_key
        assert doc["metadata"]["raw_rating"] == row.raw_rating
        assert doc["metadata"]["predicted_aspects"] == list(row.predicted_aspects)
    baseline = json.loads(Path("reports/baselines/tfidf_selection.json").read_text("utf-8"))
    assert digest("models/tfidf_logreg.joblib") == baseline["model_sha256"]
    preserved = json.loads((REPORTS / "preservation_checks.json").read_text("utf-8"))
    assert all(digest(path) == sha for path, sha in preserved["sha256"].items())
    versions = json.loads((REPORTS / "versions.json").read_text("utf-8"))
    assert versions["enriched_sha256"] == digest(ENRICHED)
    assert versions["processed_sha256"] == digest(PROCESSED)
    print(json.dumps({
        "status": "PASS", "source_rows": len(raw),
        "valid_source_rows": int(enriched.valid_for_ai.sum()),
        "distinct_valid_reviews_and_rag_documents": len(valid),
        "restaurants": len(summary), "baseline_model_unchanged": True,
        "historical_evidence_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
