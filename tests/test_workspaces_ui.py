"""Three-workspace navigation and one shared Review restaurant selection."""

from app import app as dashboard
from src.foody import review_runtime
from src.ui.monitoring_view import _value


def test_exactly_three_workspaces():
    assert list(dashboard.WORKSPACES) == [
        "Sales Intelligence", "Review Intelligence", "System Monitor"]


def test_review_workspace_reuses_one_restaurant_selection(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(review_runtime, "MANIFEST", tmp_path / "missing.json")
    monkeypatch.setattr(dashboard, "header", lambda *args: calls.append(args))
    monkeypatch.setattr(dashboard, "choose_restaurant", lambda: (
        calls.append("select") or "key", {"restaurant_name": "Fixture"}))
    monkeypatch.setattr(dashboard, "restaurant_page", lambda key, summary, **kwargs:
                        calls.append(("overview", key, kwargs["embedded"])))
    monkeypatch.setattr(dashboard, "explorer_page", lambda key, **kwargs:
                        calls.append(("reviews", key, kwargs["embedded"])))
    monkeypatch.setattr(dashboard, "analyst_page", lambda key, **kwargs:
                        calls.append(("ai", key, kwargs["embedded"])))
    dashboard.review_workspace()
    assert calls.count("select") == 1
    assert "Foody Reviews" in calls[0][2]
    assert calls[-3:] == [("overview", "key", True), ("reviews", "key", True),
                          ("ai", "key", True)]


def test_prometheus_snapshot_number_parsing_is_bounded():
    assert _value({"status": "success", "data": {"result": [
        {"value": [0, "12"]}]}}) == 12
    assert _value({"status": "success", "data": {"result": []}}) is None
    assert _value({"status": "success", "data": {"result": [
        {"value": [0, "NaN"]}]}}) is None
