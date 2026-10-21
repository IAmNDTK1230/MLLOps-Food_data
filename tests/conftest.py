from typing import Any

import pytest


@pytest.fixture
def product() -> dict[str, Any]:
    return {
        "product_id": "123",
        "shop_id": "42",
        "product_name": " Tai nghe ",
        "price": "₫199.000",
        "rating": "4,8",
        "sold_count": "Đã bán 1,2k",
        "keyword": "tai nghe",
        "crawl_timestamp": "2026-10-03T07:00:00+07:00",
    }
