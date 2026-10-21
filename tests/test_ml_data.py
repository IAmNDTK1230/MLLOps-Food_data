import json
import unicodedata

import pandas as pd
import pytest

from src.ml.config import MLConfig
from src.ml.data import data_report, import_dataset, normalize_rows, require_trainable
from src.ml.features import normalize_text
from src.ml.labels import label_rating


def test_vietnamese_normalization_preserves_accents():
    assert (
        normalize_text(unicodedata.normalize("NFD", "  Sản phẩm\n rất tốt  ")) == "Sản phẩm rất tốt"
    )


@pytest.mark.parametrize(
    "rating,label",
    [(1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")],
)
def test_rating_baseline(rating, label):
    assert label_rating(rating, MLConfig().label_rules) == label


def test_configurable_label_rule():
    rules = {**MLConfig().label_rules, 3: "negative"}
    assert label_rating(3, rules) == "negative"
    with pytest.raises(ValueError):
        label_rating(0, rules)


def test_row_mapping_rejections_and_exact_duplicates():
    # Unit-only strings; never used to fit a model or shipped as training data.
    frame = pd.DataFrame(
        [
            {"text": " Unit A ", "stars": "4,0", "id": "a"},
            {"text": "Unit A", "stars": 4, "id": "a"},
            {"text": " ", "stars": 1},
            {"text": "Unit B", "stars": 6},
        ]
    )
    rows, rejected, duplicates = normalize_rows(
        frame, "unit_source", {"comment": "text", "rating": "stars", "review_id": "id"}
    )
    assert len(rows) == 1 and len(rejected) == 2 and duplicates == 1
    assert rows.iloc[0]["rating"] == 4 and rows.iloc[0]["source"] == "unit_source"
    assert rows.iloc[0]["product_id"] is None
    assert "Unit" not in str(rejected)


def test_missing_columns_and_invalid_dataset_fail():
    with pytest.raises(ValueError, match="Required columns"):
        normalize_rows(pd.DataFrame({"comment": ["unit"]}), "unit", {})
    report = data_report(pd.DataFrame())
    assert not report["trainable"] and "Dataset is empty" in report["trainability_errors"]


@pytest.mark.parametrize(
    "labels", [["positive"], ["negative", "invalid"], ["positive", "negative", None]]
)
def test_trainability_rejects_invalid_or_single_labels(labels):
    frame = pd.DataFrame(
        {
            "comment": ["unit"] * len(labels),
            "sentiment": labels,
            "rating": [5] * len(labels),
            "source": ["unit"] * len(labels),
        }
    )
    assert not data_report(frame)["trainable"]


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_import_roundtrip_and_tamper_detection(tmp_path, suffix):
    config = MLConfig(
        data_dir=tmp_path / "data", reports_dir=tmp_path / "reports", models_dir=tmp_path / "models"
    )
    path = tmp_path / ("unit_input" + suffix)
    frame = pd.DataFrame({"comment": ["Unit A", "Unit B", ""], "rating": [1, 5, 3]})
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    report = import_dataset(
        path,
        "unit_public",
        "TEST-LICENSE",
        "https://example.org/dataset",
        "https://example.org/license",
        {},
        config,
    )
    assert report["row_count"] == 2 and report["rejected_rows"] == 1
    assert report["trainable"] and report["null_comments"] == 0
    rows, _ = require_trainable(config)
    assert set(rows["sentiment"]) == {"negative", "positive"}
    assert rows["ingestion_timestamp"].dt.tz is not None
    provenance = json.loads((config.data_dir / "source/provenance.json").read_text())
    snapshot = config.data_dir / "source" / provenance["snapshot_filename"]
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        require_trainable(config)


def test_no_dataset_and_fixture_rejected_before_training(tmp_path):
    config = MLConfig(data_dir=tmp_path / "empty")
    with pytest.raises(ValueError, match="No real review dataset"):
        require_trainable(config)
    with pytest.raises(ValueError, match="synthetic"):
        import_dataset(
            tmp_path / "none.csv",
            "synthetic_fixture",
            "X",
            "https://example.org",
            "https://example.org",
            {},
            config,
        )
    with pytest.raises(ValueError, match="--license"):
        import_dataset(tmp_path / "none.csv", "public", "", "", "", {}, config)
