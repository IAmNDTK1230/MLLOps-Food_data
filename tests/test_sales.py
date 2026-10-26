"""Small fixtures cover sales identity, routing, and grounded answer boundaries."""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from src.rag.schemas import Evidence
from src.sales import service
from src.sales.audit import audit_orders, require_loadable
from src.sales.queries import _date
from src.sales.router import parse_requested_period, route_question


@pytest.fixture(autouse=True)
def synthetic_legacy_mapping(monkeypatch):
    """The frozen Sales fixtures assert the retained legacy name-mapping path."""
    monkeypatch.setenv("FOODY_REVIEW_CORPUS", "legacy")


def test_sales_audit_only_maps_unique_names(tmp_path):
    source = tmp_path / "orders.csv"
    source.write_text(
        "order_id,order_date,sales_amount,order_status,restaurant_id,restaurant_name\n"
        "1,2025-05-01,100,Delivered,10,Nhà Hàng A\n"
        "2,2025-05-02,80,Cancelled,11,Shared Name\n"
        "3,2025-05-03,70,Refunded,12,Shared Name\n"
        "4,2025-05-04,90,Delivered,13,\n", encoding="utf-8",
    )
    reviews = tmp_path / "reviews.parquet"
    pd.DataFrame({"restaurant_key": ["a", "shared"],
                  "restaurant_name": ["Nhà Hàng A", "Shared Name"]}).to_parquet(reviews)
    audit = audit_orders(source, reviews)
    require_loadable(audit)
    assert audit["rows"] == 4
    assert audit["matched_count"] == 1
    assert audit["ambiguous_count"] == 3
    assert audit["matched"][0]["restaurant_id"] == "10"


@pytest.mark.parametrize(("question", "route"), [
    ("Doanh thu tháng này là bao nhiêu?", "SALES_SQL"),
    ("Tỷ lệ đơn hủy?", "SALES_SQL"),
    ("Khách phàn nàn gì về phục vụ?", "REVIEW_RAG"),
    ("Doanh thu giảm và khách phàn nàn gì?", "CROSS_ANALYSIS"),
    ("Sales va review", "CROSS_ANALYSIS"),
])
def test_deterministic_router(question, route):
    assert route_question(question) == route


def test_invalid_month_rejected():
    with pytest.raises(ValueError):
        _date("2026-13")


@pytest.mark.parametrize(("question", "year", "month"), [
    ("Doanh thu tháng 2/2026", 2026, 2),
    ("Doanh thu tháng 02/2026", 2026, 2),
    ("Doanh thu 2/2026", 2026, 2),
    ("Doanh thu tháng 2 năm 2026", 2026, 2),
    ("Doanh thu tháng 5/2025", 2025, 5),
    ("Doanh thu tháng 12 năm 2025", 2025, 12),
    ("Revenue 2025-05", 2025, 5),
])
def test_explicit_month_parser(question, year, month):
    assert parse_requested_period(question) == {
        "year": year, "month": month, "explicit": True,
    }


def test_period_parser_rejects_month_13_and_omitted_period_is_not_explicit():
    with pytest.raises(ValueError, match="Tháng không hợp lệ"):
        parse_requested_period("Doanh thu tháng 13/2026")
    assert parse_requested_period("Doanh thu gần nhất") == {
        "year": None, "month": None, "explicit": False,
    }


@pytest.mark.parametrize(("question", "expected"), [
    ("Doanh thu tháng 2/2026", "2026-02"),
    ("Doanh thu tháng 5/2025", "2025-05"),
    ("Doanh thu gần nhất", None),
])
def test_sales_uses_exact_requested_sql_month(monkeypatch, question, expected):
    monkeypatch.setattr(service.queries, "restaurant", lambda *_: {
        "restaurant_name": "Nhà Hàng A", "restaurant_key": "a"})
    seen = []

    def comparison(_conn, _restaurant_id, month, *, today):
        seen.append(month)
        period = date(2026, 9, 1) if month is None else date.fromisoformat(month + "-01")
        return {"period_start": period, "delivered_sales": Decimal("100"),
                "order_count": 2, "aov": Decimal("50"),
                "comparison_growth": None, "comparison_basis": "full_prior_month"}

    monkeypatch.setattr(service.queries, "monthly_comparison", comparison)
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    result = service.sales_answer(None, "10", question, today=date(2026, 10, 24))
    assert seen == [expected]
    assert "Trong tháng " + (str(int(expected[5:])) + "/" + expected[:4] if expected else "9/2026") \
        in result["answer"]


def test_invalid_sales_month_returns_clear_error_without_sql(monkeypatch):
    monkeypatch.setattr(service.queries, "restaurant", lambda *_: {
        "restaurant_name": "Nhà Hàng A", "restaurant_key": "a"})
    monkeypatch.setattr(service.queries, "monthly_comparison", lambda *args, **kwargs:
                        pytest.fail("Invalid month must not reach SQL"))
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    result = service.sales_answer(None, "10", "Doanh thu tháng 13/2026")
    assert result["status"] == "invalid_period"
    assert "Tháng không hợp lệ" in result["answer"]


def test_sales_answer_uses_only_sql_facts(monkeypatch):
    monkeypatch.setattr(service.queries, "restaurant", lambda conn, restaurant_id: {
        "restaurant_name": "Nhà Hàng A", "restaurant_key": "a"})
    monkeypatch.setattr(service.queries, "monthly_comparison", lambda *args, **kwargs: {
        "period_start": date(2025, 5, 1), "delivered_sales": Decimal("100.00"),
        "order_count": 2, "aov": Decimal("50.00"),
        "comparison_growth": Decimal("-0.2"), "comparison_basis": "full_prior_month",
    })
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    result = service.sales_answer(None, "10", "Doanh thu 2025-05?", today=date(2026, 10, 24))
    assert result["source"] == "SQL Analytics"
    assert "100 ₫" in result["answer"]
    assert "giảm 20,00%" in result["answer"]
    assert result["review_evidence"] == []


def test_cross_answer_refuses_unverified_identity(monkeypatch):
    monkeypatch.setattr(service.queries, "restaurant", lambda conn, restaurant_id: {
        "restaurant_name": "Other", "restaurant_key": None})
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    result = service.cross_answer(None, "99", "Doanh thu và phàn nàn?", today=date(2026, 10, 24))
    assert result["status"] == "unmatched_restaurant"
    assert result["review_evidence"] == []


def test_cross_answer_carries_only_cited_review_metadata(monkeypatch):
    monkeypatch.setattr(service.queries, "restaurant", lambda *_: {
        "restaurant_name": "Nhà Hàng A", "restaurant_key": "verified"})
    monkeypatch.setattr(service.queries, "monthly_comparison", lambda *args, **kwargs: {
        "period_start": date(2025, 5, 1), "delivered_sales": Decimal("100"),
        "order_count": 2, "mom_growth": Decimal("0.1")})
    monkeypatch.setattr(service, "_aligned_review_count", lambda *args: 1)
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "observe_rag", lambda *args: None)

    class Gateway:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def answer(*_):
        return (SimpleNamespace(answer="Khách phàn nàn phục vụ chậm [E1].",
                                evidence=[Evidence(review_id="a", restaurant_name="A", score=0.8)]),
                [], {"input_tokens": 4, "output_tokens": 5,
                     "generation_mode": "qwen"}, {},
                SimpleNamespace(evidence=({"ref": "E1", "review_id": "a",
                                           "date": "2025-05-12T14:00:00",
                                           "rating": "6.8", "sentiment": "negative",
                                           "aspects": ["service"], "text": "Phục vụ chậm",
                                           "source": "foody"},)))

    monkeypatch.setattr("src.rag_v3.service.answer_question_v3", answer)
    result = service.cross_answer(None, "10", "Doanh thu và khách phàn nàn gì tháng 5/2025?",
                                  today=date(2026, 10, 24),
                                  retriever_factory=lambda: object(),
                                  gateway_factory=Gateway)
    assert result["status"] == "grounded"
    assert result["review_evidence"][0]["rating"] == "6.8"
    assert result["review_evidence"][0]["date"] == "2025-05-12T14:00:00"
    assert result["review_evidence"][0]["sentiment"] == "negative"
    assert result["review_evidence"][0]["source"] == "foody"
    assert "không chứng minh quan hệ nhân quả" in result["answer"]
