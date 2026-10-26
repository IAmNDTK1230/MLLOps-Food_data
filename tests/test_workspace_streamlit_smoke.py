"""Run the final three Streamlit workspaces without starting AI inference."""

import tempfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest, app_test

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

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app/app.py",
                            default_timeout=20).run()
    assert not app.exception
    assert [option.split("  ", 1)[-1] for option in app.sidebar.radio[0].options] == [
        "Sales Intelligence", "Review Intelligence", "System Monitor"]
    assert any("Sales Intelligence" in item.value for item in app.markdown)

    app.sidebar.radio[0].set_value("Review Intelligence").run()
    assert not app.exception
    assert len([widget for widget in app.selectbox if widget.label == "Restaurant"]) == 1
    from src.foody.review_runtime import active_manifest, new_corpus_active

    if new_corpus_active():
        assert len(app.selectbox[0].options) == active_manifest()["restaurant_count"]
        assert any(f"{active_manifest()['row_count']:,} Reviews" in item.value
                   for item in app.markdown)
    assert any("Customer Overview" in item.value for item in app.subheader)
    assert any("Recent Reviews" in item.value for item in app.subheader)
    assert any("AI Review" in item.value for item in app.subheader)

    app.sidebar.radio[0].set_value("System Monitor").run()
    assert not app.exception
    assert any("System Monitor" in item.value for item in app.markdown)
