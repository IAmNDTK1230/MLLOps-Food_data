"""Deterministic Sales plans; model-generated SQL is never used."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Literal

PlanKind = Literal[
    "SINGLE_PERIOD_METRIC", "COMPARE_PERIODS", "MAX_MIN_PERIOD",
    "PERIOD_TREND", "RANGE_AGGREGATE", "YEAR_AGGREGATE", "MULTI_STEP_ANALYSIS",
]
Metric = Literal["delivered_sales", "aov", "cancellation_rate"]


@dataclass(frozen=True)
class SalesPlan:
    kind: PlanKind
    metric: Metric
    periods: tuple[str, ...]
    comparison: Literal["explicit", "yoy", "mom"] | None = None
    aggregation: str | None = None
    analytical_steps: tuple[str, ...] = ()
    range_start: str | None = None
    range_end: str | None = None
    period_selection: Literal["RANGE", "EXPLICIT_SET"] | None = None


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


def _months(start, end):
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    count = (end_year - start_year) * 12 + end_month - start_month + 1
    if not 1 <= count <= 12:
        raise ValueError("Khoảng thời gian phải từ 1 đến 12 tháng theo thứ tự tăng dần.")
    return tuple(_period((start_year * 12 + start_month - 1 + index) // 12,
                         (start_year * 12 + start_month - 1 + index) % 12 + 1)
                 for index in range(count))


def _advanced_plan(value, metric, today):
    recent = re.search(r"\b(\d{1,2})\s+thang\s+gan\s+day\b", value)
    month_range = re.search(
        r"\b(?:tu\s+)?thang\s+(\d{1,2})\s+den\s+thang\s+(\d{1,2})\s+(?:trong\s+)?nam\s+(20\d{2})\b",
        value,
    )
    first_months = re.search(r"\b(\d{1,2})\s+thang\s+dau\s+nam\s+(20\d{2})\b", value)
    quarter = re.search(r"\bquy\s+(\d{1,2})\s+nam\s+(20\d{2})\b", value)
    year_match = re.search(r"\bnam\s+(20\d{2})\b", value)
    if not any((recent, month_range, first_months, quarter, year_match)):
        return None
    if metric != "delivered_sales":
        return None

    # Named months joined by "và"/commas are a set, never an entire year.
    if not month_range and ("tong" in value or "tong cong" in value):
        explicit = extract_requested_periods(value)
        if 2 <= len(explicit) <= 12:
            if len(set(explicit)) != len(explicit):
                raise ValueError("Các tháng trong tập tính tổng phải khác nhau.")
            return SalesPlan("RANGE_AGGREGATE", metric, explicit, aggregation="sum",
                             analytical_steps=("fetch_named_months", "sum_revenue"),
                             period_selection="EXPLICIT_SET")
        if len(explicit) > 12:
            raise ValueError("Chỉ hỗ trợ tối đa 12 tháng được nêu rõ.")

    if recent:
        count = int(recent.group(1))
        if not 1 <= count <= 12:
            raise ValueError("Xu hướng gần đây chỉ hỗ trợ từ 1 đến 12 tháng.")
        last = today.replace(day=1).fromordinal(today.replace(day=1).toordinal() - 1)
        last = last.replace(day=1)
        first_index = last.year * 12 + last.month - count
        first = _period(first_index // 12, first_index % 12 + 1)
        periods = _months(first, last.strftime("%Y-%m"))
        return SalesPlan("PERIOD_TREND", metric, periods, aggregation="monthly_series",
                         analytical_steps=("fetch_monthly_revenue", "order_by_month"),
                         range_start=periods[0], range_end=periods[-1],
                         period_selection="RANGE")

    if month_range:
        start_month, end_month, year = month_range.groups()
        periods = _months(_period(year, start_month), _period(year, end_month))
    elif first_months:
        count, year = first_months.groups()
        periods = _months(_period(year, 1), _period(year, count))
    elif quarter:
        number, year = map(int, quarter.groups())
        if not 1 <= number <= 4:
            raise ValueError("Quý phải từ 1 đến 4.")
        periods = _months(_period(year, (number - 1) * 3 + 1),
                          _period(year, number * 3))
    elif year_match:
        year = year_match.group(1)
        periods = _months(_period(year, 1), _period(year, 12))
    else:
        return None

    if "giam manh nhat" in value and "thang truoc" in value:
        kind, aggregation = "MULTI_STEP_ANALYSIS", "largest_mom_decline"
        steps = ("fetch_monthly_revenue_and_prior_month", "calculate_mom_changes",
                 "select_largest_decline")
    elif "cao nhat" in value and "thap nhat" in value:
        kind, aggregation = "MULTI_STEP_ANALYSIS", "max_min_spread"
        steps = ("fetch_monthly_revenue", "select_max_and_min", "calculate_spread")
    elif "cao nhat" in value or "thap nhat" in value:
        kind, aggregation = "MAX_MIN_PERIOD", (
            "max" if "cao nhat" in value else "min")
        steps = ("fetch_monthly_revenue", "select_extreme")
    elif "xu huong" in value or "trend" in value:
        kind, aggregation = "PERIOD_TREND", "monthly_series"
        steps = ("fetch_monthly_revenue", "order_by_month")
    elif "trung binh moi thang" in value:
        kind, aggregation = "YEAR_AGGREGATE", "monthly_average"
        steps = ("fetch_monthly_revenue", "sum_revenue", "divide_by_month_count")
    elif "tong" in value and year_match and not month_range and not quarter and not first_months:
        kind, aggregation = "YEAR_AGGREGATE", "sum"
        steps = ("fetch_monthly_revenue", "sum_revenue")
    elif "tong" in value and (month_range or quarter or first_months):
        kind, aggregation = "RANGE_AGGREGATE", "sum"
        steps = ("fetch_monthly_revenue", "sum_revenue")
    else:
        return None
    return SalesPlan(kind, metric, periods, aggregation=aggregation,
                     analytical_steps=steps, range_start=periods[0],
                     range_end=periods[-1], period_selection="RANGE")


def build_sales_plan(question, *, today=None):
    value = _plain(question)
    metric = _metric(value)
    advanced = _advanced_plan(value, metric, today or date.today())
    if advanced is not None:
        return advanced
    periods = extract_requested_periods(question)
    if len(periods) > 2:
        raise ValueError("Giai đoạn này chỉ hỗ trợ so sánh hai kỳ dữ liệu.")
    if len(periods) == 2:
        return SalesPlan("COMPARE_PERIODS", metric, periods, "explicit",
                         period_selection="EXPLICIT_SET")
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
