import json
import unicodedata

import numpy as np
import pandas as pd
import pytest

from src.foody import pipeline
from src.foody.analytics import aggregate, alerts, business_rows, label_summary
from src.foody.aspect import grouped_targets, scores
from src.foody.cleaning import clean, snapshot
from src.foody.enrichment import extract_issues, predict_observations
from src.foody.schema import normalize, restaurant_key, review_id
from src.foody.validation import load_source, parse_date, validate_processed


def source():
    return pd.DataFrame([
        ["An", "1/2/2024 8:30", "8.5", "  Món  ngon 😀! ", " Quán Ăn "],
        ["Bình", "2/2/2024 08:31:12", "6", "Món ngon 😀!", "quán ăn"],
        ["An", "bad", "0", "", "Quán khác"],
        ["An", "", "10", "Bị \ufffd", "Quán khác"],
    ], columns=["User", "Review Time", "Rating", "Comment", "RestaurantName"])


def test_keys_and_review_ids():
    assert restaurant_key(" Quán  Ăn ") == restaurant_key(
        unicodedata.normalize("NFD", "QUÁN ĂN")
    )
    assert restaurant_key("Quán ăn 2") != restaurant_key("Quán ăn")
    assert restaurant_key(" ") is None
    key = restaurant_key("Quán")
    assert review_id(key, "An", "a", "x") == review_id(key, "An", "a", "x")
    assert review_id(key, "An", "a", "x") != review_id(key, "An", "b", "x")
    assert review_id(key, "a|b", "c", "d") != review_id(key, "a", "b|c", "d")


def test_cleaning_preserves_rows_and_semantics():
    raw = source()
    frame = clean(raw, "2026-10-20T00:00:00+00:00")
    assert len(frame) == 4
    assert frame.text_original.tolist() == raw.Comment.tolist()
    assert frame.restaurant_name.tolist() == raw.RestaurantName.tolist()
    assert frame.raw_rating.tolist() == ["8.5", "6", "0", "10"]
    assert frame.valid_for_ai.tolist() == [True, True, False, False]
    assert frame.duplicate_count.tolist() == [2, 2, 1, 1]
    assert frame.source_rows[0] == [1, 2]
    assert frame.observation_id[0] == frame.observation_id[1]
    assert frame.review_id[0] != frame.review_id[1]
    assert frame.date_status.tolist() == ["parsed", "parsed", "unparseable", "missing"]
    assert pd.isna(frame.review_timestamp[2])
    assert normalize("  Đồ\n  ĂN 😀?! ") == "Đồ ĂN 😀?!"
    validate_processed(frame)
    assert not any("classification" in c.casefold() for c in frame.columns)


@pytest.mark.parametrize("text,expected", [
    ("1/2/2024 8:30", "2024-02-01 08:30:00"),
    ("31/03/2022 03:04:12", "2022-03-31 03:04:12"),
    ("2024-02-01", None), ("31/2/2024 08:00", None), ("", None),
])
def test_dates(text, expected):
    if expected is None:
        assert pd.isna(parse_date(text))
    else:
        assert parse_date(text) == pd.Timestamp(expected)


def test_utf8_snapshot(tmp_path):
    path, target = tmp_path / "raw.csv", tmp_path / "snapshot.csv"
    source().to_csv(path, index=False, encoding="utf-8")
    original = path.read_bytes()
    metadata = snapshot(path, target)
    assert metadata["row_count"] == 4
    assert target.read_bytes() == original == path.read_bytes()
    assert snapshot(path, target) == metadata
    assert load_source(path).Comment.iloc[0] == "  Món  ngon 😀! "
    path.write_bytes(original + b"\xff")
    with pytest.raises(UnicodeDecodeError):
        load_source(path)
    path.write_bytes(original.replace(b"8.5", b"7.5"))
    with pytest.raises(ValueError, match="differs"):
        snapshot(path, target)


def test_external_schema_rejected(tmp_path):
    path = tmp_path / "external.csv"
    source().assign(Classification=1).to_csv(path, index=False)
    with pytest.raises(ValueError, match="five"):
        load_source(path)


class SentimentStub:
    classes_ = np.array(["negative", "neutral", "positive"])

    def __init__(self):
        self.calls = []

    def predict_proba(self, values):
        self.calls.extend(values)
        return np.array([[0.1, 0.2, 0.7]] * len(values))


class AspectStub:
    def predict_proba(self, values):
        return np.array([[0.8, 0.2]] * len(values))


def test_inference_only_valid_deduplicated_text():
    frame = clean(source(), "now")
    model = SentimentStub()
    predictions, latency = predict_observations(
        frame, model, {"pipeline": AspectStub(), "classes": ["food", "service"], "threshold": 0.5}
    )
    assert model.calls == ["Món ngon 😀!"]
    assert predictions.predicted_sentiment.tolist() == ["positive"]
    assert predictions.predicted_aspects.tolist() == [["food"]]
    assert predictions.sentiment_confidence.iloc[0] == 0.7
    assert predictions.aspect_confidences.iloc[0] == {"food": 0.8, "service": 0.2}
    assert latency["sentiment"]["inference_count"] == 1


def test_issue_gates_and_negation():
    text = "Phục vụ chậm, giá quá đắt."
    assert extract_issues(text, ["service", "price"], "negative") == ["expensive", "slow_service"]
    assert extract_issues(text, ["food"], "negative") == []
    assert extract_issues(text, ["service", "price"], "positive") == []
    assert extract_issues("Không đắt, chẳng bẩn", ["price", "cleanliness"], "negative") == []
    assert extract_issues("Không ngon", ["food"], "negative") == ["food_taste"]


def enriched():
    frame = clean(source(), "now")
    frame["predicted_sentiment"] = ["positive", "negative", None, None]
    frame["predicted_aspects"] = [["food"], ["food", "service"], [], []]
    frame["detected_issues"] = [[], ["slow_service"], [], []]
    frame["sentiment_model_version"] = "sentiment_hash"
    frame["aspect_model_version"] = "aspect_hash"
    return frame


def test_mart_counts_actual_reviews_not_inference_groups():
    frame = enriched()
    frame = pd.concat([frame, frame.iloc[[0]]])
    business = business_rows(frame)
    table = aggregate(business, ["restaurant_key"])
    row = table.loc[table.review_count == 2].loc[lambda x: x.ai_review_count == 2].iloc[0]
    assert row.review_count == 2
    assert row.average_rating == 7.25
    assert row.predicted_negative_rate == 0.5
    assert row.predicted_service_complaint_rate == 0.5
    aspects = label_summary(business, "predicted_aspects", "aspect")
    assert aspects.loc[aspects.aspect == "food", "predicted_review_count"].iloc[0] == 2


def monthly_row(period, n, rate, rating=8):
    return {"restaurant_key": "key", "period": period, "ai_review_count": n,
            "rating_count": n, "predicted_negative_rate": rate,
            "predicted_service_complaint_rate": rate,
            "predicted_price_complaint_rate": 0.0, "predicted_food_complaint_rate": 0.0,
            "average_rating": rating}


def test_alert_sample_rules_and_calendar_comparability():
    assert alerts(pd.DataFrame([
        monthly_row("2024-01", 9, 0.0), monthly_row("2024-02", 20, 0.9)
    ]), minimum_samples=10).empty
    assert alerts(pd.DataFrame([
        monthly_row("2024-01", 20, 0.0), monthly_row("2024-03", 20, 0.9)
    ])).empty
    found = alerts(pd.DataFrame([
        monthly_row("2024-01", 10, 0.1), monthly_row("2024-02", 10, 0.5, 6)
    ]))
    assert set(found.metric) == {
        "predicted_negative_rate", "predicted_service_complaint_rate", "average_rating"
    }
    assert found.historical_only.all()


def test_aspect_grouping_and_leakage_guard():
    frame = pd.DataFrame({"text": ["Món ngon", " Món  ngon ", "Giá cao"],
                          "aspect": ["food", "service", "price"],
                          "split": ["train", "train", "test"], "review_id": ["a", "b", "c"]})
    groups = grouped_targets(frame)
    assert groups.loc[groups.split == "train", "aspects"].iloc[0] == ["food", "service"]
    frame.loc[1, "split"] = "validation"
    with pytest.raises(ValueError, match="crosses"):
        grouped_targets(frame)
    metrics, per_class = scores(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 0]]), ["a", "b"])
    assert metrics["exact_match_ratio"] == 0.5
    assert per_class.support.tolist() == [1, 1]


def test_rag_metadata_and_invalid_exclusion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "enriched.parquet"
    frame = enriched()
    frame = pd.concat([frame, frame.iloc[[0]].assign(source_row=5)], ignore_index=True)
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(pipeline, "ENRICHED", path)
    result = pipeline.prepare_rag()
    assert result["document_count"] == 2
    rows = [json.loads(line) for line in (tmp_path / "data/foody/rag/documents.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    assert rows[0]["text"] == "Món ngon 😀!"
    assert rows[0]["metadata"]["source_rows"] == [1, 5]
    assert rows[0]["metadata"]["raw_rating"] == "8.5"
    assert rows[0]["metadata"]["source"] == "foody"
    assert rows[0]["metadata"]["restaurant_identity_type"] == "derived_from_restaurant_name"
