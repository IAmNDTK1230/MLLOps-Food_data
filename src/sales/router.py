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
    if sales and (reviews or "tai sao" in value or "vi sao" in value):
        return "CROSS_ANALYSIS"
    if sales:
        return "SALES_SQL"
    return "REVIEW_RAG"


def month_in_question(question):
    found = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", question)
    return f"{found.group(1)}-{int(found.group(2)):02d}" if found else None


def wants_current_month(question):
    value = _plain(question)
    return "thang nay" in value or "this month" in value


def wants_current_week(question):
    value = _plain(question)
    return "tuan nay" in value or "this week" in value
