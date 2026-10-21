import pytest
from pydantic import ValidationError

from src.models.product import Product
from src.models.review import Review
from src.processing.normalizer import normalize


def test_utc_and_optional_fields(product):
    result = Product.model_validate(normalize(product))
    assert result.crawl_timestamp.hour == 0
    assert result.stock is None
    assert result.product_name == "Tai nghe"


@pytest.mark.parametrize(
    "field,value",
    [
        ("product_id", ""),
        ("product_name", " "),
        ("price", -1),
        ("rating", 5.1),
        ("rating_count", -1),
        ("sold_count", -1),
        ("price", float("nan")),
        ("crawl_timestamp", "2026-10-03T00:00:00"),
    ],
)
def test_invalid_product(product, field, value):
    record = normalize(product)
    record[field] = value
    with pytest.raises(ValidationError):
        Product.model_validate(record)


def test_review_bounds():
    with pytest.raises(ValidationError):
        Review(product_id="1", rating=6, crawl_timestamp="2026-10-03T00:00:00Z")
