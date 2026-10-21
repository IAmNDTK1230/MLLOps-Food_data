import pandas as pd
import pytest

from src.ml.config import MLConfig
from src.ml.split import assign_splits, leakage_groups


def test_deterministic_product_and_text_leakage_protection():
    # Metadata-only unit dataset; no estimator is fitted.
    rows = []
    for index in range(60):
        for duplicate in range(2):
            rows.append(
                {
                    "review_id": f"{index}-{duplicate}",
                    "product_id": f"p{index}",
                    "comment": f"unit text {index}",
                    "rating": [1, 3, 5][index % 3],
                    "sentiment": ["negative", "neutral", "positive"][index % 3],
                    "source": "unit",
                }
            )
    frame = pd.DataFrame(rows)
    config = MLConfig()
    first, meta = assign_splits(frame, config)
    second, _ = assign_splits(frame, config)
    assert first["split"].tolist() == second["split"].tolist()
    assert first.groupby("product_id")["split"].nunique().max() == 1
    assert first.groupby("comment")["split"].nunique().max() == 1
    assert set(first["split"]) == {"train", "validation", "test"}
    assert sum(meta["counts"].values()) == len(frame)
    assert meta["random_seed"] == 42


def test_duplicate_comments_connect_different_products():
    frame = pd.DataFrame({"comment": [" Same ", "same", "other"], "product_id": ["a", "b", "b"]})
    assert len(set(leakage_groups(frame, True))) == 1


def test_too_few_independent_groups_fails():
    frame = pd.DataFrame(
        {"comment": ["a", "b"], "product_id": ["one", "one"], "sentiment": ["negative", "positive"]}
    )
    with pytest.raises(ValueError, match="three independent"):
        assign_splits(frame, MLConfig())
