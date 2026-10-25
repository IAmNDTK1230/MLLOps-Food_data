"""Deterministic two-period Sales plans; model-generated SQL is never used."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Literal

PlanKind = Literal["SINGLE_PERIOD_METRIC", "COMPARE_PERIODS"]
Metric = Literal["delivered_sales", "aov", "cancellation_rate"]


@dataclass(frozen=True)
class SalesPlan:
    kind: PlanKind
    metric: Metric
    periods: tuple[str, ...]
    comparison: Literal["explicit", "yoy", "mom"] | None = None


def _plain(value):
    value = unicodedata.normalize("NFD", value.casefold()).replace("đ", "d")
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _period(year, month):
    year, month = int(year), int(month)
    if not 1 <= month <= 12:
        raise ValueError("Tháng không hợp lệ; vui lòng nhập tháng từ 1 đến 12.")
    return f"{year:04d}-{month:02d}"


def extract_requested_periods(question):
    """Extract periods in spoken order, sharing one explicit year when needed."""
    value = _plain(question)
    found = []
    patterns = (
        (r"\b(20\d{2})[-/](\d{1,2})\b", "year_month"),
        (r"\b(?:thang\s*)?(\d{1,2})\s*/\s*(20\d{2})\b", "month_year"),
        (r"\b(?:thang|month)\s+(\d{1,2})\s+(?:nam|year)\s+(20\d{2})\b",
         "month_year"),
    )
    for pattern, order in patterns:
        for match in re.finditer(pattern, value):
            if any(match.start() < end and match.end() > start
                   for start, end, _ in found):
                continue
            first, second = match.groups()
            period = (_period(first, second) if order == "year_month"
                      else _period(second, first))
            found.append((match.start(), match.end(), period))
    years = set(re.findall(r"\b20\d{2}\b", value))
    for match in re.finditer(r"\b(?:thang|month)\s+(\d{1,2})\b", value):
        if any(match.start() < end and match.end() > start
               for start, end, _ in found):
            continue
        if not years:
            raise ValueError("Vui lòng nêu năm cho tháng được hỏi.")
        if len(years) != 1:
            raise ValueError("Tháng chưa có năm rõ ràng; vui lòng nêu cả tháng và năm.")
        found.append((match.start(), match.end(), _period(next(iter(years)), match.group(1))))
    found.sort(key=lambda item: item[0])
    return tuple(period for _, _, period in found)


def _metric(value):
    if re.search(r"\baov\b|gia tri don(?: hang)? trung binh", value):
        return "aov"
    if ("ty le huy" in value or "don huy" in value
            or "cancellation rate" in value or "cancel rate" in value):
        return "cancellation_rate"
    return "delivered_sales"


def build_sales_plan(question):
    value = _plain(question)
    periods = extract_requested_periods(question)
    metric = _metric(value)
    if len(periods) > 2:
        raise ValueError("Giai đoạn này chỉ hỗ trợ so sánh hai kỳ dữ liệu.")
    if len(periods) == 2:
        return SalesPlan("COMPARE_PERIODS", metric, periods, "explicit")
    if len(periods) == 1:
        current = date.fromisoformat(periods[0] + "-01")
        if re.search(r"cung ky nam truoc|cung thang nam truoc|cung ky nam ngoai|\byoy\b|year.over.year", value):
            reference = current.replace(year=current.year - 1)
            return SalesPlan("COMPARE_PERIODS", metric,
                             (periods[0], reference.strftime("%Y-%m")), "yoy")
        if re.search(r"so voi thang truoc|\bmom\b|month.over.month", value):
            reference = current.fromordinal(current.toordinal() - 1).replace(day=1)
            return SalesPlan("COMPARE_PERIODS", metric,
                             (periods[0], reference.strftime("%Y-%m")), "mom")
    return SalesPlan("SINGLE_PERIOD_METRIC", metric, periods)
