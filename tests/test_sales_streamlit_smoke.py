"""Lightweight Streamlit smoke for the Sales result card and page."""

import tempfile
from datetime import date
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest, app_test

from src.ui import sales_view


def test_verified_analysis_card_renders_comparison_and_trend(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(app_test, "TMP_DIR", SimpleNamespace(name=str(tmp_path)))
    comparison = {
        "answer": "Doanh thu tháng 5/2026 cao hơn tháng 6/2026 100 ₫.",
        "analysis": {"metric": "delivered_sales", "response_mode": "COMPARISON",
                     "percent_change": 100},
        "display_rows": [{"period": "2026-05", "value": 200},
                         {"period": "2026-06", "value": 100}],
    }
    def show(result):
        from src.ui.sales_view import show_sales_result

        show_sales_result(result)

    app = AppTest.from_function(show,
                                args=(comparison,), default_timeout=10).run()
    assert not app.exception
    assert any("Verified Analysis" in item.value for item in app.markdown)
    assert len(app.dataframe) == 1

    trend = {**comparison, "analysis": {"metric": "delivered_sales",
                                       "response_mode": "TREND", "percent_change": None}}
    app = AppTest.from_function(show,
                                args=(trend,), default_timeout=10).run()
    assert not app.exception
    assert len(app.dataframe) == 1


def test_sales_page_smoke_without_live_database(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(app_test, "TMP_DIR", SimpleNamespace(name=str(tmp_path)))
    monthly = {"period_start": date(2025, 5, 1), "delivered_sales": 100,
               "order_count": 2, "aov": 50, "cancellation_rate": 0,
               "mom_growth": None, "comparison_growth": None,
               "comparison_basis": "full_prior_month"}
    monkeypatch.setattr(sales_view, "sales_restaurants", lambda: [
        {"restaurant_id": "1", "restaurant_name": "Fixture", "restaurant_key": None}])
    monkeypatch.setattr(sales_view, "sales_snapshot", lambda _id: {
        "monthly": [monthly], "weekly": [], "statuses": [],
        "summary": {"cancellation_rate": 0, "refund_rate": 0}})
    monkeypatch.setattr(sales_view, "month_snapshot", lambda _id, _month: monthly)
    def show():
        from src.ui.sales_view import sales_page

        sales_page()

    app = AppTest.from_function(show, default_timeout=10).run()
    assert not app.exception
    assert any("Sales Intelligence" in item.value for item in app.markdown)
