"""UI-level sales data boundaries and presentation values."""

import numpy as np
import pandas as pd
import pytest

from src.foody.enrichment import predict_observations
from src.sales.router import route_question
from src.ui.components import money, percent, text_sentiment_label
from src.ui.sales_view import cross_evidence_metadata, select_review_snippets


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


def test_cross_evidence_metadata_displays_source_rating_date_and_missing_values():
    label = cross_evidence_metadata({"ref": "E1", "rating": "6.8",
                                     "date": "2025-05-12T14:00:00",
                                     "sentiment": "negative", "aspects": ["service"],
                                     "source": "foody"})
    assert "Rating: 6.8/10" in label
    assert "Date: 2025-05-12" in label
    assert "Sentiment: Negative" in label
    assert "Aspects: service" in label
    assert "Source: foody" in label
    assert "Rating: unavailable" in cross_evidence_metadata({})


def test_review_badge_maps_only_saved_model_classes():
    assert [text_sentiment_label(label) for label in
            ("negative", "neutral", "positive")] == [
                "Text sentiment: Negative", "Text sentiment: Neutral",
                "Text sentiment: Positive",
            ]
    with pytest.raises(ValueError, match="Unsupported sentiment"):
        text_sentiment_label(-1)


def test_sentiment_prediction_uses_model_class_order_not_rating():
    class Sentiment:
        classes_ = ["positive", "neutral", "negative"]

        def predict_proba(self, _texts):
            return [[0.8, 0.1, 0.1]]

    class Aspect:
        def predict_proba(self, _texts):
            return np.array([[0.9]])

    frame = pd.DataFrame([{"observation_id": "a", "valid_for_ai": True,
                           "text": "Món ngon", "raw_rating": "1.0"}])
    predictions, _ = predict_observations(frame, Sentiment(), {
        "pipeline": Aspect(), "classes": ["food"], "threshold": 0.5})
    assert predictions.iloc[0].predicted_sentiment == "positive"
    assert predictions.iloc[0].sentiment_confidence == pytest.approx(0.8)
