"""UI-level sales data boundaries and presentation values."""

import pandas as pd

from src.sales.router import route_question
from src.ui.components import money, percent
from src.ui.sales_view import select_review_snippets


def test_sales_question_suggestions_stay_on_sql_route():
    for question in ("Doanh thu tháng này?", "Tháng doanh thu nào cao nhất?",
                     "Tỷ lệ đơn hủy?", "AOV là bao nhiêu?",
                     "So sánh doanh thu tháng trước?"):
        assert route_question(question) == "SALES_SQL"


def test_cross_evidence_display_never_uses_other_restaurant_text():
    rows = pd.DataFrame([
        {"review_id": "a", "restaurant_key": "verified", "text": "Relevant",
         "raw_rating": 8, "review_timestamp": pd.Timestamp("2025-05-01")},
        {"review_id": "b", "restaurant_key": "other", "text": "Unsafe",
         "raw_rating": 1, "review_timestamp": pd.Timestamp("2025-05-01")},
    ])
    result = select_review_snippets(rows, ("a", "b"), "verified")
    assert list(result) == ["a"]
    assert result["a"]["text"] == "Relevant"


def test_kpi_formatting_uses_actual_numeric_values():
    assert money(1959760) == "1,959,760 ₫"
    assert percent(0.0825) == "8.25%"
    assert percent(None) == "—"
