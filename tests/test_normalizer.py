import pytest

from src.processing.normalizer import number


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("₫1.299.000", 1299000),
        ("4,9", 4.9),
        ("Đã bán 1,2k", 1200),
        ("2m", 2000000),
        (None, None),
        ("0", 0),
        ("-10", -10),
    ],
)
def test_number(raw, expected):
    assert number(raw) == expected


@pytest.mark.parametrize("raw", ["contact shop", "12-14", "1,2,3"])
def test_ambiguous_numbers_rejected(raw):
    with pytest.raises(ValueError):
        number(raw)


def test_fractional_counts_rejected():
    with pytest.raises(ValueError):
        number("1,5", count=True)
