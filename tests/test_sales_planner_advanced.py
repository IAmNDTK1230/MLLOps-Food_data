"""Phase 2 Sales plans use bounded monthly SQL facts and deterministic math."""

from datetime import date
from decimal import Decimal

import pytest

from src.sales import analysis, service
from src.sales.planner import build_sales_plan

TODAY = date(2026, 10, 25)


def _run(monkeypatch, question, values, *, today=TODAY):
    requested = []

    def monthly_sales(_conn, _restaurant_id, month, *, today):
        requested.append(month)
        value = values.get(month)
        return {"delivered_sales": Decimal(str(value))} if value is not None else None

    monkeypatch.setattr(analysis.queries, "monthly_sales", monthly_sales)
    monkeypatch.setattr(service.queries, "restaurant", lambda *_: {
        "restaurant_name": "Example"})
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    result = service.sales_answer(object(), "R1", question, today=today)
    return result, requested


def test_max_and_min_month_in_year(monkeypatch):
    values = {f"2026-{month:02d}": month * 100 for month in range(1, 10)}
    maximum, requested = _run(
        monkeypatch, "tháng nào doanh thu cao nhất năm 2026", values)
    assert maximum["query_plan"]["intent"] == "MAX_MIN_PERIOD"
    assert maximum["query_plan"]["aggregation"] == "max"
    assert "tháng 9/2026 có doanh thu cao nhất với 900 ₫" in maximum["answer"]
    assert requested == list(values)
    assert "Chỉ tính các tháng đã hoàn thành" in maximum["answer"]
    minimum, _ = _run(monkeypatch, "tháng nào doanh thu thấp nhất năm 2026", values)
    assert "tháng 1/2026 có doanh thu thấp nhất với 100 ₫" in minimum["answer"]


def test_six_completed_month_trend(monkeypatch):
    values = {f"2026-{month:02d}": month * 100 for month in range(4, 10)}
    result, requested = _run(monkeypatch, "doanh thu 6 tháng gần đây", values)
    assert result["query_plan"]["intent"] == "PERIOD_TREND"
    assert result["query_plan"]["range"] == {
        "start": "2026-04", "end": "2026-09"}
    assert requested == list(values)
    assert "tháng 4/2026 (400 ₫)" in result["answer"]
    assert "tháng 9/2026 (900 ₫)" in result["answer"]


def test_explicit_range_trend(monkeypatch):
    values = {f"2026-{month:02d}": month for month in range(1, 7)}
    result, requested = _run(monkeypatch,
                             "xu hướng doanh thu từ tháng 1 đến tháng 6 năm 2026", values)
    assert result["query_plan"]["intent"] == "PERIOD_TREND"
    assert requested == list(values)


def test_range_and_quarter_total(monkeypatch):
    values = {f"2026-{month:02d}": month * 100 for month in range(1, 7)}
    result, requested = _run(
        monkeypatch, "tổng doanh thu từ tháng 1 đến tháng 6 năm 2026", values)
    assert result["query_plan"]["intent"] == "RANGE_AGGREGATE"
    assert "2.100 ₫" in result["answer"]
    assert requested == list(values)
    quarter, requested = _run(monkeypatch, "tổng doanh thu quý 1 năm 2026", values)
    assert quarter["query_plan"]["range"] == {
        "start": "2026-01", "end": "2026-03"}
    assert "600 ₫" in quarter["answer"]
    assert requested == list(values)[:3]


def test_yearly_total_and_monthly_average(monkeypatch):
    year_2025 = {f"2025-{month:02d}": 100 for month in range(1, 13)}
    total, requested = _run(monkeypatch, "tổng doanh thu năm 2025", year_2025)
    assert total["query_plan"]["intent"] == "YEAR_AGGREGATE"
    assert "1.200 ₫ trên 12 tháng" in total["answer"]
    assert requested == list(year_2025)
    year_2026 = {f"2026-{month:02d}": 200 for month in range(1, 10)}
    average, requested = _run(monkeypatch,
                              "doanh thu trung bình mỗi tháng năm 2026", year_2026)
    assert average["query_plan"]["aggregation"] == "monthly_average"
    assert "200 ₫, tính từ 9 tháng" in average["answer"]
    assert requested == list(year_2026)


def test_largest_mom_decline_in_first_six_months(monkeypatch):
    values = {
        "2025-12": 100, "2026-01": 120, "2026-02": 90,
        "2026-03": 110, "2026-04": 40, "2026-05": 45, "2026-06": 30,
    }
    result, requested = _run(monkeypatch, (
        "Trong 6 tháng đầu năm 2026, tháng nào doanh thu giảm mạnh nhất "
        "so với tháng trước?"), values)
    assert result["query_plan"]["intent"] == "MULTI_STEP_ANALYSIS"
    assert result["query_plan"]["aggregation"] == "largest_mom_decline"
    assert requested == list(values)
    assert "tháng 4/2026" in result["answer"]
    assert "từ 110 ₫ trong tháng 3/2026 xuống 40 ₫" in result["answer"]
    assert "giảm 63,64%" in result["answer"]
    assert result["analysis"]["difference"] == -70


def test_year_max_min_spread(monkeypatch):
    values = {f"2025-{month:02d}": month * 10 for month in range(1, 13)}
    result, _ = _run(monkeypatch, (
        "Trong năm 2025, tháng nào có doanh thu cao nhất và tháng nào "
        "thấp nhất, chênh lệch bao nhiêu?"), values)
    assert result["query_plan"]["aggregation"] == "max_min_spread"
    assert "tháng 12/2025 với 120 ₫" in result["answer"]
    assert "tháng 1/2025 với 10 ₫" in result["answer"]
    assert "110 ₫" in result["answer"]


def test_missing_month_in_range_does_not_produce_partial_total(monkeypatch):
    result, _ = _run(monkeypatch,
                     "tổng doanh thu từ tháng 1 đến tháng 6 năm 2026",
                     {"2026-01": 100})
    assert result["status"] == "no_data"
    assert result["query_plan"]["missing_periods"] == [
        f"2026-{month:02d}" for month in range(2, 7)]
    assert "không tính kết quả" in result["answer"]


@pytest.mark.parametrize("question", [
    "tổng doanh thu từ tháng 6 đến tháng 1 năm 2026",
    "tổng doanh thu quý 5 năm 2026",
    "doanh thu 13 tháng gần đây",
])
def test_invalid_range_rejected(question):
    with pytest.raises(ValueError):
        build_sales_plan(question, today=TODAY)


def test_trace_contains_deterministic_sql_steps(monkeypatch):
    values = {f"2026-{month:02d}": month for month in range(1, 7)}
    result, _ = _run(monkeypatch,
                     "tổng doanh thu từ tháng 1 đến tháng 6 năm 2026", values)
    plan = result["query_plan"]
    assert plan["metric"] == "delivered_sales"
    assert plan["sql_mode"] == "predefined_parameterized_monthly_sales"
    assert plan["analytical_steps"] == ["fetch_monthly_revenue", "sum_revenue"]
