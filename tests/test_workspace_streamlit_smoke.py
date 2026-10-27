"""Run the final three Streamlit workspaces without starting AI inference."""

import tempfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest, app_test

from src.foody import review_runtime
from src.ui import monitoring_view, sales_view


def test_three_workspaces_render_without_automatic_ai(monkeypatch, tmp_path):
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
    monkeypatch.setattr(monitoring_view, "prometheus_snapshot", lambda: {
        name: None for name in monitoring_view.QUERIES})
    monkeypatch.setattr(review_runtime, "MANIFEST", tmp_path / "missing.json")
    reviews = pd.DataFrame([{
        "restaurant_id": "1", "restaurant_name": "Fixture", "review_id": "review-1",
        "review_timestamp": pd.Timestamp("2026-01-01"), "raw_rating": 8,
        "text": "Món ăn ngon", "predicted_sentiment": "positive",
        "predicted_aspects": ["food"], "detected_issues": [],
        "sentiment_confidence": 0.9,
    }])
    monkeypatch.setattr(review_runtime, "review_table", lambda: reviews)

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app/app.py",
                            default_timeout=20).run()
    assert not app.exception
    assert [option.split("  ", 1)[-1] for option in app.sidebar.radio[0].options] == [
        "Sales Intelligence", "Review Intelligence", "System Monitor"]
    assert any("Sales Intelligence" in item.value for item in app.markdown)

    app.sidebar.radio[0].set_value("Review Intelligence").run()
    assert not app.exception
    assert len([widget for widget in app.selectbox if widget.label == "Restaurant"]) == 1
    assert app.selectbox[0].options == ["Fixture"]
    assert any("Foody Reviews" in item.value for item in app.markdown)
    assert any("Customer Overview" in item.value for item in app.subheader)
    assert any("Recent Reviews" in item.value for item in app.subheader)
    assert any("AI Review" in item.value for item in app.subheader)

    app.sidebar.radio[0].set_value("System Monitor").run()
    assert not app.exception
    assert any("System Monitor" in item.value for item in app.markdown)
