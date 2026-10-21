import re
from typing import Any


def number(value: Any, *, count: bool = False) -> float | int | None:
    """Normalize visible Vietnamese currency/counts. Never invent hidden values."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().lower().replace("\u00a0", " ")
    text = re.sub(r"^(đã bán|sold)\s*", "", text)
    text = re.sub(r"[₫đ%\s+]", "", text)
    multiplier = 1
    if text.endswith(("k", "m")):
        multiplier = 1000 if text[-1] == "k" else 1000000
        text = text[:-1].replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")
    else:
        text = text.replace(",", ".")
    result = float(text) * multiplier  # invalid formats go to rejection, never silently to null
    if count:
        if not result.is_integer():
            raise ValueError("Count must be an integer")
        return int(result)
    return result


def normalize(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value.strip() if isinstance(value, str) else value for key, value in record.items()
    }
    for key in ("price", "original_price", "discount_percent", "rating"):
        if key in result:
            result[key] = number(result[key])
    for key in ("rating_count", "sold_count", "stock", "likes"):
        if key in result:
            result[key] = number(result[key], count=True)
    for key in ("product_id", "shop_id", "review_id"):
        if result.get(key) is not None:
            result[key] = str(result[key])
    return result
