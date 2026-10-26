"""Structured SQL analysis and concise, deterministic Vietnamese responses."""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

ResponseMode = Literal["FACTUAL", "COMPARISON", "AGGREGATE", "TREND", "EXTREME", "EXPLANATION"]


@dataclass
class SalesAnalysisResult:
    intent: str
    response_mode: ResponseMode
    metric: str
    periods: list[str]
    values: list[Decimal]
    difference: Decimal | None = None
    percent_change: Decimal | None = None
    aggregation: str | None = None
    observations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    evidence_source: str = "PostgreSQL"
    period_selection: str | None = None
    details: dict = field(default_factory=dict)

    def as_dict(self):
        def serializable(value):
            if isinstance(value, Decimal):
                return float(value)
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, list):
                return [serializable(item) for item in value]
            return value

        return {
            "intent": self.intent, "response_mode": self.response_mode,
            "metric": self.metric, "periods": self.periods,
            "values": [float(value) for value in self.values],
            "difference": float(self.difference) if self.difference is not None else None,
            "percent_change": (float(self.percent_change)
                               if self.percent_change is not None else None),
            "aggregation": self.aggregation, "observations": self.observations,
            "limitations": self.limitations, "evidence_source": self.evidence_source,
            "period_selection": self.period_selection,
            "details": serializable(self.details),
        }


def response_mode(plan, question=""):
    lowered = question.casefold()
    if any(phrase in lowered for phrase in ("tại sao", "vì sao", "tai sao", "vi sao", "lý do", "why ")):
        return "EXPLANATION"
    if plan.kind == "COMPARE_PERIODS":
        return "COMPARISON"
    if plan.kind in {"RANGE_AGGREGATE", "YEAR_AGGREGATE"}:
        return "AGGREGATE"
    if plan.kind == "PERIOD_TREND":
        return "TREND"
    if plan.kind in {"MAX_MIN_PERIOD", "MULTI_STEP_ANALYSIS"}:
        return "EXTREME"
    return "FACTUAL"


def number(value, decimals=0):
    raw = f"{Decimal(str(value)):,.{decimals}f}"
    return raw.replace(",", "_").replace(".", ",").replace("_", ".")


def money(value):
    return f"{number(value)} ₫"


def percentage(value):
    return f"{number(value, 2)}%"


def month_label(period):
    return f"{int(period[5:7])}/{period[:4]}"


def _metric_value(value, metric):
    return percentage(value * 100) if metric == "cancellation_rate" else money(value)


def _metric_name(metric):
    return {"delivered_sales": "Doanh thu Delivered", "aov": "AOV Delivered",
            "cancellation_rate": "Tỷ lệ hủy đơn"}[metric]


def render(result, restaurant_name):
    """Render only measured facts; observations never become causal claims."""
    periods, values = result.periods, result.values
    metric = _metric_name(result.metric)
    if result.response_mode in {"COMPARISON", "EXPLANATION"}:
        first, second = values
        first_month, second_month = map(month_label, periods)
        first_value, second_value = (_metric_value(value, result.metric)
                                     for value in values)
        if result.difference > 0:
            direction = "cao hơn"
        elif result.difference < 0:
            direction = "thấp hơn"
        else:
            direction = "gần như không thay đổi so với"
        magnitude = (f"{number(abs(result.difference) * 100, 2)} điểm phần trăm"
                     if result.metric == "cancellation_rate" else money(abs(result.difference)))
        if result.difference == 0:
            comparison = (f"{metric} tháng {first_month} đạt {first_value}, "
                          f"gần như không thay đổi so với tháng {second_month} ({second_value}).")
        else:
            comparison = (f"{metric} tháng {first_month} đạt {first_value}, {direction} "
                          f"tháng {second_month} ({second_value}) {magnitude}.")
        if (result.percent_change is not None and result.metric != "cancellation_rate"
                and result.difference != 0):
            comparison += (f" Tương ứng, kỳ đầu {'cao' if result.difference > 0 else 'thấp'} "
                           f"hơn {percentage(abs(result.percent_change))} so với kỳ đối chiếu.")
        elif result.percent_change is None and result.metric != "cancellation_rate":
            comparison += " Không tính tỷ lệ thay đổi vì kỳ đối chiếu bằng 0."
        if result.response_mode == "EXPLANATION":
            comparison += " " + " ".join(result.limitations)
        return comparison.strip()

    if result.response_mode == "AGGREGATE":
        total = result.details["aggregate_value"]
        if result.aggregation == "monthly_average":
            return (f"{metric} trung bình mỗi tháng trong {month_label(periods[0])}–"
                    f"{month_label(periods[-1])} là {money(total)}, tính từ "
                    f"{len(periods)} tháng có dữ liệu đầy đủ.")
        if result.period_selection == "EXPLICIT_SET":
            labels = " và ".join(month_label(period) for period in periods)
            parts = "; ".join(f"tháng {month_label(period)}: {money(value)}"
                              for period, value in zip(periods, values, strict=True))
            return f"Tổng {metric.lower()} của tháng {labels} là {money(total)}. Trong đó {parts}."
        return (f"Tổng {metric.lower()} từ {month_label(periods[0])} đến "
                f"{month_label(periods[-1])} là {money(total)} trên {len(periods)} tháng.")

    if result.response_mode == "TREND":
        high = max(range(len(values)), key=values.__getitem__)
        low = min(range(len(values)), key=values.__getitem__)
        changes = [right - left for left, right in zip(values, values[1:], strict=False)]
        mixed = any(change > 0 for change in changes) and any(change < 0 for change in changes)
        if values[-1] > values[0]:
            direction = "tăng từ đầu đến cuối kỳ"
        elif values[-1] < values[0]:
            direction = "giảm từ đầu đến cuối kỳ"
        else:
            direction = "kết thúc ở cùng mức với đầu kỳ"
        if mixed:
            direction = "biến động qua các tháng và " + direction
        return (f"Trong {len(periods)} tháng từ {month_label(periods[0])} đến "
                f"{month_label(periods[-1])}, {metric.lower()} {direction}. "
                f"Cao nhất là tháng {month_label(periods[high])} ({money(values[high])}); "
                f"thấp nhất là tháng {month_label(periods[low])} ({money(values[low])}).")

    if result.response_mode == "EXTREME":
        if result.aggregation == "largest_mom_decline":
            if not result.details.get("selected_period"):
                return "Trong khoảng được hỏi, không có tháng nào giảm doanh thu so với tháng trước."
            current = month_label(result.details["selected_period"])
            prior = month_label(result.details["previous_period"])
            return (f"Mức giảm doanh thu mạnh nhất xảy ra vào tháng {current}. "
                    f"Doanh thu giảm từ {money(result.details['previous_value'])} "
                    f"trong tháng {prior} xuống {money(result.details['current_value'])}, "
                    f"tương đương giảm {percentage(abs(result.percent_change)) if result.percent_change is not None else 'không tính được tỷ lệ'}.")
        if result.aggregation == "max_min_spread":
            high = result.details["high_period"]
            low = result.details["low_period"]
            return (f"Trong kỳ được hỏi, doanh thu cao nhất ở tháng {month_label(high)} "
                    f"với {money(result.details['high_value'])}; thấp nhất ở tháng "
                    f"{month_label(low)} với {money(result.details['low_value'])}. "
                    f"Khoảng cách giữa hai tháng là {money(result.difference)}.")
        selected = result.details["selected_period"]
        direction = "cao nhất" if result.aggregation == "max" else "thấp nhất"
        return (f"Trong khoảng {month_label(periods[0])}–{month_label(periods[-1])}, "
                f"tháng {month_label(selected)} có doanh thu {direction} với "
                f"{money(result.details['selected_value'])}.")

    answer = (f"Trong tháng {month_label(periods[0])}, {restaurant_name} ghi nhận "
              f"{metric.lower()} {_metric_value(values[0], result.metric)}")
    if result.details.get("order_count") is not None:
        answer += f" từ {result.details['order_count']} đơn đã quan sát"
    answer += "."
    if result.details.get("aov") is not None:
        answer += f" AOV đạt {money(result.details['aov'])}."
    growth = result.details.get("comparison_growth")
    if growth is not None:
        direction = "tăng" if growth > 0 else "giảm" if growth < 0 else "không đổi"
        answer += f" So với kỳ trước, doanh thu {direction} {percentage(abs(growth) * 100)}."
    return answer


def _parse_number(token):
    token = token.removesuffix("%").lstrip("+")
    if "," in token and "." in token:
        token = (token.replace(".", "").replace(",", ".") if token.rfind(",") > token.rfind(".")
                 else token.replace(",", ""))
    elif "," in token or "." in token:
        separator = "," if "," in token else "."
        pieces = token.split(separator)
        token = ("".join(pieces) if len(pieces[-1]) == 3 else
                 "".join(pieces[:-1]) + "." + pieces[-1])
    return Decimal(token)


def fact_lock_valid(text, result):
    """Reject any generated numerical claim absent from the verified result."""
    allowed = set()
    for period in result.periods:
        allowed.update((Decimal(period[:4]), Decimal(period[5:7])))
    allowed.update(result.values)
    allowed.update(Decimal(str(value)) for value in result.details.get("extra_values", []))
    for value in (result.difference, result.percent_change):
        if value is not None:
            allowed.update((value, abs(value), abs(value).quantize(Decimal("0.01"))))
    if result.details.get("aggregate_value") is not None:
        allowed.add(Decimal(str(result.details["aggregate_value"])))
    for value in result.values:
        allowed.add(value * 100 if result.metric == "cancellation_rate" else value)
    for token in re.findall(r"(?<![\w])[-+]?\d[\d.,]*%?", text):
        try:
            if _parse_number(token) not in allowed:
                return False
        except (ArithmeticError, ValueError):
            return False
    return True


def render_with_optional_naturalizer(result, restaurant_name, naturalizer=None):
    fallback = render(result, restaurant_name)
    if naturalizer is None or result.response_mode != "EXPLANATION":
        return fallback, False, "not_used"
    try:
        candidate = naturalizer(result.as_dict())
        if (isinstance(candidate, str) and candidate.strip() and
                fact_lock_valid(candidate, result) and
                not re.search(r"\b(?:do|vì|nhờ|gây ra|khiến|bởi vì|review|đánh giá|phàn nàn)\b",
                              candidate.casefold())):
            return candidate.strip(), True, "pass"
    except (RuntimeError, ValueError, TypeError):
        pass
    return fallback, True, "fallback_fact_lock"
