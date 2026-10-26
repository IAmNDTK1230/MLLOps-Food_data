"""Safety gates for the direct-ID review migration."""

from datetime import date, datetime

import pandas as pd
import pytest

from src.foody.review_migration import (
    align_ids,
    audit_source,
    canonicalize,
    require_accepted,
    stable_review_id,
    write_documents,
)
from src.rag.retriever import where_filter
from src.rag.schemas import RetrievalRequest
from src.rag.vector_store import chroma_metadata, read_documents
from src.sales import service as sales_service


@pytest.fixture
def review_rows():
    return pd.DataFrame([
        {"Review Time": "01/05/2026", "Rating": "8", "Comment": "Món ăn ngon",
         "RestaurantID": "17", "Restaurant Name": "Quán A", "Address": "Địa chỉ A"},
        {"Review Time": "01/05/2026", "Rating": "8", "Comment": "Món ăn ngon",
         "RestaurantID": "17", "Restaurant Name": "Quán A", "Address": "Địa chỉ A"},
        {"Review Time": "03/05/2026", "Rating": "5", "Comment": "Phục vụ chậm",
         "RestaurantID": "18", "Restaurant Name": "Quán B", "Address": "Địa chỉ B"},
    ]).astype("string")


def test_review_schema_dedup_and_stable_ids(review_rows):
    report = audit_source(review_rows)
    assert (report["rows"], report["exact_duplicate_rows"], report["deduplicated_rows"]) == (3, 1, 2)
    assert report["invalid_dates"] == report["invalid_ratings"] == 0
    canonical = canonicalize(review_rows)
    assert len(canonical) == canonical.review_id.nunique() == 2
    assert canonical.restaurant_id.tolist() == ["17", "18"]
    assert canonical.review_id.iloc[0] == stable_review_id(
        "17", datetime(2026, 5, 1), 8, "Món ăn ngon")


def test_id_namespace_accepts_exact_shared_sales_identity(review_rows):
    sales = pd.DataFrame({"restaurant_id": ["17", "18", "19"],
                          "restaurant_name": ["Quán A", "Quán B", "Quán C"]})
    audit = audit_source(review_rows)
    aligned = align_ids(review_rows, sales)
    require_accepted(audit, aligned)
    assert aligned["id_intersection"] == 2
    assert aligned["sales_ids_without_reviews"] == 1
    assert aligned["name_mismatches"] == 0


def test_id_namespace_rejects_same_id_different_name(review_rows):
    sales = pd.DataFrame({"restaurant_id": ["17", "18"],
                          "restaurant_name": ["Unrelated", "Quán B"]})
    aligned = align_ids(review_rows, sales)
    assert aligned["name_mismatches"] == 1
    with pytest.raises(ValueError, match="namespace"):
        require_accepted(audit_source(review_rows), aligned)


def test_id_namespace_rejects_review_id_name_conflict(review_rows):
    frame = review_rows.copy()
    frame.loc[1, "Restaurant Name"] = "Quán khác"
    with pytest.raises(ValueError, match="multiple names"):
        align_ids(frame, pd.DataFrame({"restaurant_id": ["17"],
                                      "restaurant_name": ["Quán A"]}))


def test_new_document_metadata_and_restaurant_date_filter(tmp_path):
    frame = pd.DataFrame([{"review_id": "r26_a", "restaurant_id": "17",
                           "restaurant_name": "Quán A", "review_time": pd.Timestamp("2026-05-01"),
                           "rating": 8.0, "comment": "Món ăn ngon", "source": "foody_review_2025_2026",
                           "predicted_sentiment": "positive", "predicted_aspects": ["food"],
                           "detected_issues": []}])
    path = tmp_path / "documents.jsonl"
    assert write_documents(frame, path)["document_count"] == 1
    doc = next(read_documents(path))
    metadata = chroma_metadata(doc)
    assert metadata["restaurant_id"] == "17"
    assert metadata["review_id"] == "r26_a"
    assert metadata["timestamp_epoch"] is not None
    request = RetrievalRequest(query="món ăn", restaurant_id="17",
                               date_from=datetime(2026, 5, 1), date_to=datetime(2026, 5, 31))
    assert {"restaurant_id": "17"} in where_filter(request)["$and"]


def test_direct_cross_requires_review_id_membership(monkeypatch):
    monkeypatch.setattr("src.foody.review_runtime.new_corpus_active", lambda: True)
    monkeypatch.setattr(sales_service.queries, "restaurant", lambda *_: {
        "restaurant_id": "17", "restaurant_name": "Quán A",
        "review_restaurant_id": None})
    monkeypatch.setattr(sales_service, "_record", lambda *_, **__: None)
    result = sales_service.cross_answer(None, "17", "doanh thu và review tháng 5/2026")
    assert result["status"] == "unmatched_restaurant"
    assert result["review_evidence"] == []


def test_cross_september_never_uses_june_reviews(monkeypatch):
    monkeypatch.setattr("src.foody.review_runtime.new_corpus_active", lambda: True)
    monkeypatch.setattr(sales_service.queries, "restaurant", lambda *_: {
        "restaurant_id": "17", "restaurant_name": "Quán A",
        "review_restaurant_id": "17"})
    monkeypatch.setattr(sales_service.queries, "monthly_comparison", lambda *_, **__: {
        "period_start": date(2026, 9, 1), "delivered_sales": 120000,
        "order_count": 2, "comparison_growth": 0.1})
    monkeypatch.setattr(sales_service, "_record", lambda *_, **__: None)
    periods = []

    def no_reviews(_conn, restaurant_id, start, end):
        periods.append((restaurant_id, start, end))
        return 0

    monkeypatch.setattr(sales_service, "_aligned_review_count", no_reviews)
    result = sales_service.cross_answer(None, "17", "doanh thu và review tháng 9/2026",
                                        today=date(2026, 10, 1))
    assert periods == [("17", date(2026, 9, 1), date(2026, 9, 30))]
    assert result["status"] == "no_aligned_reviews"
    assert result["sql_facts"]["delivered_sales"] == 120000
    assert result["review_evidence"] == []
