"""Natural Sales wording cannot change the verified monthly SQL facts."""

from datetime import date
from decimal import Decimal

import pytest

from src.sales import comparisons, service
from src.sales.planner import build_sales_plan
from src.sales.response import SalesAnalysisResult, fact_lock_valid
from src.sales.router import route_question
from src.ui.sales_view import sales_result_table

TODAY = date(2026, 10, 25)


def _ask(monkeypatch, question, amounts, *, mapped=False, naturalizer=None):
    seen = []

    def monthly_sales(_conn, _restaurant_id, month, *, today):
        seen.append(month)
        amount = amounts.get(month)
        return {"delivered_sales": Decimal(str(amount))} if amount is not None else None

    monkeypatch.setattr(comparisons.queries, "monthly_sales", monthly_sales)
    monkeypatch.setattr(service.queries, "restaurant", lambda *_: {
        "restaurant_name": "Nhà hàng A", "restaurant_key": "verified" if mapped else None})
    monkeypatch.setattr(service, "_record", lambda *args, **kwargs: None)
    result = service.answer_question(None, "R1", question, today=TODAY,
                                     naturalizer=naturalizer)
    return result, seen


def test_two_named_months_are_set_not_full_year(monkeypatch):
    question = "Doanh thu tháng 5 và tháng 6 trong năm 2025 tổng cộng bao nhiêu"
    plan = build_sales_plan(question, today=TODAY)
    assert plan.kind == "RANGE_AGGREGATE"
    assert plan.period_selection == "EXPLICIT_SET"
    assert plan.periods == ("2025-05", "2025-06")
    result, seen = _ask(monkeypatch, question, {"2025-05": 100, "2025-06": 250})
    assert seen == ["2025-05", "2025-06"]
    assert result["analysis"]["details"]["aggregate_value"] == 350
    assert "350 ₫" in result["answer"]
    assert "100 ₫" in result["answer"] and "250 ₫" in result["answer"]


def test_three_named_months_and_range_have_different_period_selection(monkeypatch):
    named = "tổng doanh thu tháng 3, tháng 4 và tháng 5 năm 2026"
    plan = build_sales_plan(named, today=TODAY)
    assert plan.period_selection == "EXPLICIT_SET"
    assert plan.periods == ("2026-03", "2026-04", "2026-05")
    result, seen = _ask(monkeypatch, named, dict.fromkeys(plan.periods, 100))
    assert seen == list(plan.periods)
    assert "300 ₫" in result["answer"]
    ranged = build_sales_plan("tổng doanh thu tháng 1 đến tháng 6 năm 2026", today=TODAY)
    assert ranged.period_selection == "RANGE"
    assert len(ranged.periods) == 6


@pytest.mark.parametrize(("amounts", "word"), [
    ({"2026-05": 200, "2026-06": 100}, "cao hơn"),
    ({"2026-05": 100, "2026-06": 200}, "thấp hơn"),
    ({"2026-05": 100, "2026-06": 100}, "gần như không thay đổi"),
])
def test_natural_comparison_direction_and_unchanged(monkeypatch, amounts, word):
    result, seen = _ask(monkeypatch,
                        "so sánh doanh thu tháng 5 và tháng 6 năm 2026", amounts)
    assert seen == ["2026-05", "2026-06"]
    assert result["analysis"]["response_mode"] == "COMPARISON"
    assert word in result["answer"]
    if word == "gần như không thay đổi":
        assert "thấp hơn 0,00%" not in result["answer"]
    assert "VND" not in result["answer"]
    assert result["query_plan"]["qwen_used"] is False


def test_yoy_comparison_keeps_both_named_years(monkeypatch):
    result, seen = _ask(monkeypatch,
                        "so sánh doanh thu tháng 5 năm 2026 với tháng 5 năm 2025",
                        {"2026-05": 200, "2025-05": 100})
    assert seen == ["2026-05", "2025-05"]
    assert result["analysis"]["difference"] == 100
    assert result["analysis"]["percent_change"] == 100
    assert "100,00%" in result["answer"]


@pytest.mark.parametrize("mapped", [False, True])
def test_why_question_leads_with_verified_facts_and_cause_limit(monkeypatch, mapped):
    question = "tại sao doanh thu tháng 5/2026 lại tăng so với 5/2025"
    assert route_question(question) == "SALES_SQL"
    result, seen = _ask(monkeypatch, question,
                        {"2026-05": 200, "2025-05": 100}, mapped=mapped)
    assert seen == ["2026-05", "2025-05"]
    assert result["analysis"]["response_mode"] == "EXPLANATION"
    assert "200 ₫" in result["answer"] and "100 ₫" in result["answer"]
    assert "không đủ để xác định nguyên nhân" in result["answer"]
    assert result["analysis"]["details"]["cross_available"] is mapped
    assert ("có thể dùng Cross Analysis" in result["answer"]) is mapped
    assert result["review_evidence"] == []
    assert result["query_plan"]["qwen_used"] is False


def test_fact_lock_rejects_changed_number_and_falls_back(monkeypatch):
    question = "tại sao doanh thu tháng 5/2026 tăng so với 5/2025"
    result, _ = _ask(monkeypatch, question,
                     {"2026-05": 1024896, "2025-05": 540464},
                     naturalizer=lambda _facts: "Tháng 5/2026 đạt 1.024.000 ₫.")
    assert "1.024.896 ₫" in result["answer"]
    assert "1.024.000 ₫" not in result["answer"]
    assert result["query_plan"]["fact_validation_status"] == "fallback_fact_lock"
    assert result["query_plan"]["qwen_used"] is True


def test_fact_lock_accepts_only_whitelisted_numbers():
    verified = SalesAnalysisResult("COMPARE_PERIODS", "EXPLANATION", "delivered_sales",
                                   ["2026-05", "2025-05"],
                                   [Decimal("1024896"), Decimal("540464")],
                                   difference=Decimal("484432"),
                                   percent_change=Decimal("89.6326"))
    assert fact_lock_valid("Tháng 5/2026 đạt 1.024.896 ₫.", verified)
    assert not fact_lock_valid("Tháng 5/2026 đạt 1.024.000 ₫.", verified)


def test_comparison_table_uses_same_structured_values(monkeypatch):
    result, _ = _ask(monkeypatch,
                     "so sánh doanh thu tháng 5 và tháng 6 năm 2026",
                     {"2026-05": 1200000, "2026-06": 900000})
    table = sales_result_table(result)
    assert table.to_dict("records") == [
        {"Period": "05/2026", "Revenue": "1.200.000 ₫"},
        {"Period": "06/2026", "Revenue": "900.000 ₫"},
    ]
