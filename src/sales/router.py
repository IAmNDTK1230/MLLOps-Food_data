"""Small deterministic router; never asks an LLM to choose SQL."""

import re
import unicodedata

SALES = ("doanh thu", "sales", "revenue", "đơn hàng", "don hang", "aov",
         "đơn hủy", "don huy", "cancelled", "refunded", "delivered",
         "tăng trưởng", "tang truong", "kinh doanh", "bán hàng", "ban hang")
REVIEWS = ("phàn nàn", "phan nan", "khách", "khach", "review", "đánh giá",
           "danh gia", "phản hồi", "phan hoi", "món ăn", "mon an", "phục vụ",
           "phuc vu", "chất lượng", "chat luong", "cải thiện", "cai thien")


def _plain(value):
    value = unicodedata.normalize("NFD", value.casefold()).replace("đ", "d")
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def route_question(question):
    value = _plain(question)
    sales = any(_plain(term) in value for term in SALES)
    reviews = any(_plain(term) in value for term in REVIEWS)
    if sales and reviews:
        return "CROSS_ANALYSIS"
    if sales:
        return "SALES_SQL"
    return "REVIEW_RAG"


def month_in_question(question):
    period = parse_requested_period(question)
    return f"{period['year']}-{period['month']:02d}" if period["explicit"] else None


def parse_requested_period(question):
    """Read an explicit month and year without guessing from the sales data."""
    value = _plain(question)
    patterns = (
        (r"\b(20\d{2})[-/](\d{1,2})\b", "year_month"),
        (r"\b(?:thang\s*)?(\d{1,2})\s*/\s*(20\d{2})\b", "month_year"),
        (r"\b(?:thang|month)\s+(\d{1,2})\s+(?:nam|year)\s+(20\d{2})\b",
         "month_year"),
    )
    for pattern, order in patterns:
        found = re.search(pattern, value)
        if found:
            year, month = (map(int, found.groups()) if order == "year_month"
                           else reversed(tuple(map(int, found.groups()))))
            if not 1 <= month <= 12:
                raise ValueError("Tháng không hợp lệ; vui lòng nhập tháng từ 1 đến 12.")
            return {"year": year, "month": month, "explicit": True}
    return {"year": None, "month": None, "explicit": False}


def wants_current_month(question):
    value = _plain(question)
    return "thang nay" in value or "this month" in value


def wants_current_week(question):
    value = _plain(question)
    return "tuan nay" in value or "this week" in value
