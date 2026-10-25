"""Execute bounded Sales plans with existing parameterized SQL and Python math."""

from datetime import date, timedelta
from decimal import Decimal

from src.sales import queries


def _money(value):
    return f"{value:,.0f} VND"


def _previous(period):
    current = date.fromisoformat(period + "-01")
    return (current - timedelta(days=1)).strftime("%Y-%m")


def _trace(plan, effective, missing):
    return {
        "intent": plan.kind,
        "metric": plan.metric,
        "periods": list(plan.periods),
        "range": {"start": plan.range_start, "end": plan.range_end},
        "aggregation": plan.aggregation,
        "analytical_steps": list(plan.analytical_steps),
        "sql_mode": "predefined_parameterized_monthly_sales",
        "effective_periods": list(effective),
        "missing_periods": list(missing),
    }


def answer_advanced(conn, restaurant_id, identity, plan, *, today):
    """Return an answer only when each required observed month has SQL facts."""
    last_complete = _previous(today.strftime("%Y-%m"))
    year_scope = plan.kind in {"MAX_MIN_PERIOD", "YEAR_AGGREGATE"} or (
        plan.kind == "MULTI_STEP_ANALYSIS" and plan.aggregation == "max_min_spread")
    effective = tuple(period for period in plan.periods if period <= last_complete) if (
        year_scope and len(plan.periods) == 12) else plan.periods
    required = ((_previous(effective[0]), *effective)
                if effective and plan.aggregation == "largest_mom_decline" else effective)
    rows = {period: queries.monthly_sales(conn, restaurant_id, period, today=today)
            for period in required}
    missing = tuple(period for period in required if rows[period] is None or
                    rows[period].get("delivered_sales") is None)
    trace = _trace(plan, effective, missing)
    base = {"route": "SALES_SQL", "source": "SQL Analytics",
            "restaurant_id": str(restaurant_id), "query_plan": trace,
            "sql_facts": {period: queries.jsonable(row) for period, row in rows.items()},
            "review_evidence": []}
    if not effective or missing:
        detail = ", ".join(missing) if missing else "khoảng đã yêu cầu"
        return {**base, "status": "no_data", "answer": (
            f"Không đủ dữ liệu doanh thu tháng cho {detail}; "
            "không tính kết quả từ khoảng thiếu dữ liệu.")}

    values = {period: Decimal(str(rows[period]["delivered_sales"]))
              for period in required}
    name = identity["restaurant_name"]
    scope = f"{effective[0]} đến {effective[-1]}"
    if len(effective) < len(plan.periods):
        scope += " (chỉ các tháng đã hoàn thành)"

    if plan.kind == "MAX_MIN_PERIOD":
        chosen = (max if plan.aggregation == "max" else min)(
            effective, key=lambda period: values[period])
        direction = "cao nhất" if plan.aggregation == "max" else "thấp nhất"
        answer = (f"Doanh thu Delivered {direction} của {name} trong {scope} "
                  f"là tháng {chosen}: {_money(values[chosen])}.")
    elif plan.kind == "PERIOD_TREND":
        series = "; ".join(f"{period}: {_money(values[period])}" for period in effective)
        answer = f"Xu hướng doanh thu Delivered của {name} từ {scope}: {series}."
    elif plan.kind in {"RANGE_AGGREGATE", "YEAR_AGGREGATE"}:
        total = sum((values[period] for period in effective), Decimal(0))
        if plan.aggregation == "monthly_average":
            average = total / len(effective)
            answer = (f"Doanh thu Delivered trung bình mỗi tháng của {name} "
                      f"trong {scope}: {_money(average)} "
                      f"({len(effective)} tháng, tổng {_money(total)}).")
        else:
            answer = (f"Tổng doanh thu Delivered của {name} trong {scope}: "
                      f"{_money(total)} ({len(effective)} tháng).")
    elif plan.aggregation == "max_min_spread":
        high = max(effective, key=lambda period: values[period])
        low = min(effective, key=lambda period: values[period])
        spread = values[high] - values[low]
        answer = (f"Trong {scope}, doanh thu Delivered của {name} cao nhất tháng "
                  f"{high}: {_money(values[high])}; thấp nhất tháng {low}: "
                  f"{_money(values[low])}; chênh lệch {_money(spread)}.")
    elif plan.aggregation == "largest_mom_decline":
        changes = [(period, values[period] - values[_previous(period)])
                   for period in effective]
        declines = [(period, change) for period, change in changes if change < 0]
        if not declines:
            return {**base, "status": "ok", "answer": (
                f"Trong {scope}, không có tháng nào doanh thu Delivered của {name} "
                "giảm so với tháng trước.")}
        chosen, change = min(declines, key=lambda item: item[1])
        prior = _previous(chosen)
        relative = (f"{change / values[prior] * 100:+.2f}%" if values[prior]
                    else "không tính được vì tháng trước bằng 0")
        answer = (f"Trong {scope}, doanh thu Delivered của {name} giảm mạnh nhất "
                  f"ở tháng {chosen}: tháng trước {prior} = {_money(values[prior])}; "
                  f"tháng này = {_money(values[chosen])}; thay đổi "
                  f"{_money(change)} ({relative}).")
    else:
        raise ValueError("Unsupported deterministic Sales aggregation")
    return {**base, "status": "ok", "answer": answer}
