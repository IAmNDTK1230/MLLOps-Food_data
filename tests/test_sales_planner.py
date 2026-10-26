"""Phase 1 deterministic Sales comparison planning and SQL fact tests."""

from datetime import date
from decimal import Decimal

import pytest

from src.sales import comparisons, service
from src.sales.planner import build_sales_plan, extract_requested_periods


@pytest.mark.parametrize(
    ("question", "periods"),
    [
        ("doanh thu 2025-12 so với 2026-01", ("2025-12", "2026-01")),
        ("doanh thu tháng 2 với tháng 3 năm 2026", ("2026-02", "2026-03")),
        ("AOV tháng 2/2025 và tháng 3/2026", ("2025-02", "2026-03")),
        ("doanh thu tháng 2 năm 2025 so với tháng 2 năm 2026",
         ("2025-02", "2026-02")),
    ],
)
def test_extract_two_periods_in_question_order(question, periods):
    assert extract_requested_periods(question) == periods
    plan = build_sales_plan(question)
    assert plan.kind == "COMPARE_PERIODS"
    assert plan.periods == periods
    assert plan.comparison == "explicit"


@pytest.mark.parametrize(
    ("question", "metric", "periods", "comparison"),
    [
        ("Doanh thu tháng 1 năm 2026 so với cùng kỳ năm trước",
         "delivered_sales", ("2026-01", "2025-01"), "yoy"),
        ("Doanh thu tháng 1 năm 2026 so với tháng trước",
         "delivered_sales", ("2026-01", "2025-12"), "mom"),
        ("AOV tháng 2 năm 2026 YoY", "aov", ("2026-02", "2025-02"), "yoy"),
        ("Tỷ lệ hủy đơn tháng 3 năm 2026 MoM",
         "cancellation_rate", ("2026-03", "2026-02"), "mom"),
        ("Đơn hủy tháng 3 năm 2026 so với tháng 2 năm 2026",
         "cancellation_rate", ("2026-03", "2026-02"), "explicit"),
    ],
)
def test_relative_and_metric_plans(question, metric, periods, comparison):
    plan = build_sales_plan(question)
    assert (plan.kind, plan.metric, plan.periods, plan.comparison) == (
        "COMPARE_PERIODS", metric, periods, comparison)


def test_simple_sales_remains_single_period():
    assert build_sales_plan("Doanh thu tháng 3 năm 2026").kind == "SINGLE_PERIOD_METRIC"
    assert build_sales_plan("Doanh thu tháng này").kind == "SINGLE_PERIOD_METRIC"


@pytest.mark.parametrize("question", [
    "Doanh thu tháng 13 năm 2026 so với tháng trước",
    "Doanh thu tháng 2 với tháng 3",
    "Doanh thu tháng 1, tháng 2 và tháng 3 năm 2026",
])
def test_invalid_or_unbounded_comparison_rejected(question):
    with pytest.raises(ValueError):
        build_sales_plan(question)


@pytest.mark.parametrize(
    ("question", "field", "first", "second", "expected"),
    [
        ("doanh thu tháng 2 với tháng 3 năm 2026", "delivered_sales",
         Decimal("1200000"), Decimal("1000000"), "cao hơn tháng 3/2026 (1.000.000 ₫) 200.000 ₫"),
        ("AOV tháng 2 với tháng 3 năm 2026", "aov",
         Decimal("120000"), Decimal("100000"), "cao hơn tháng 3/2026 (100.000 ₫) 20.000 ₫"),
        ("tỷ lệ hủy tháng 2 với tháng 3 năm 2026", "cancellation_rate",
         Decimal("0.15"), Decimal("0.10"), "5,00 điểm phần trăm"),
    ],
)
def test_comparison_uses_two_exact_monthly_sql_facts(
    monkeypatch, question, field, first, second, expected,
):
    requested = []
    rows = {"2026-02": {field: first}, "2026-03": {field: second}}

    def monthly_sales(conn, restaurant_id, month, *, today):
        requested.append((restaurant_id, month, today))
        return rows[month]

    monkeypatch.setattr(comparisons.queries, "monthly_sales", monthly_sales)
    monkeypatch.setattr(service.queries, "restaurant", lambda *_: {
        "restaurant_name": "Example"})
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    result = service.sales_answer(object(), "R1", question, today=date(2026, 10, 25))
    assert result["status"] == "ok"
    assert expected in result["answer"]
    assert result["query_plan"]["metric"] == field
    assert [month for _, month, _ in requested] == ["2026-02", "2026-03"]
    assert result["sql_facts"]["2026-02"][field] == float(first)


def test_missing_period_does_not_invent_comparison(monkeypatch):
    monkeypatch.setattr(comparisons.queries, "monthly_sales",
                        lambda _conn, _id, month, *, today: (
                            {"delivered_sales": Decimal("12")} if month == "2026-02"
                            else None))
    plan = build_sales_plan("doanh thu tháng 2 với tháng 3 năm 2026")
    result = comparisons.answer_comparison(
        object(), "R1", {"restaurant_name": "Example"}, plan,
        today=date(2026, 10, 25))
    assert result["status"] == "no_data"
    assert "không thể so sánh" in result["answer"]


def test_zero_reference_does_not_divide_by_zero(monkeypatch):
    monkeypatch.setattr(comparisons.queries, "monthly_sales",
                        lambda _conn, _id, month, *, today: {
                            "delivered_sales": Decimal("12" if month == "2026-02" else "0")})
    plan = build_sales_plan("doanh thu tháng 2 với tháng 3 năm 2026")
    result = comparisons.answer_comparison(
        object(), "R1", {"restaurant_name": "Example"}, plan,
        today=date(2026, 10, 25))
    assert result["status"] == "ok"
    assert "Không tính tỷ lệ thay đổi" in result["answer"]
