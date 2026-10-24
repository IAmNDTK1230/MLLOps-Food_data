"""Small fixtures cover sales identity, routing, and grounded answer boundaries."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from src.sales import service
from src.sales.audit import audit_orders, require_loadable
from src.sales.queries import _date
from src.sales.router import route_question


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
    assert "100 VND" in result["answer"]
    assert "-20.00%" in result["answer"]
    assert result["review_evidence"] == []


def test_cross_answer_refuses_unverified_identity(monkeypatch):
    monkeypatch.setattr(service.queries, "restaurant", lambda conn, restaurant_id: {
        "restaurant_name": "Other", "restaurant_key": None})
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    result = service.cross_answer(None, "99", "Doanh thu và phàn nàn?", today=date(2026, 10, 24))
    assert result["status"] == "unmatched_restaurant"
    assert result["review_evidence"] == []
