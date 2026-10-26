"""Execute bounded Sales plans with existing parameterized SQL and Python math."""

from datetime import date, timedelta
from decimal import Decimal

from src.sales import queries
from src.sales.response import SalesAnalysisResult, render, response_mode


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
        "period_selection": plan.period_selection or "RANGE",
        "sql_functions": ["monthly_sales"],
        "calculations": list(plan.analytical_steps[1:]),
        "qwen_used": False,
        "fact_validation_status": "not_needed",
        "grounding_status": "sql_verified",
        "response_mode": response_mode(plan),
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
    mode = response_mode(plan)
    if not effective or missing:
        detail = ", ".join(missing) if missing else "khoảng đã yêu cầu"
        return {**base, "status": "no_data", "answer": (
            f"Không đủ dữ liệu doanh thu tháng cho {detail}; "
            "không tính kết quả từ khoảng thiếu dữ liệu."),
            "analysis": SalesAnalysisResult(plan.kind, mode, plan.metric,
                                            list(effective), [],
                                            limitations=["missing SQL facts"]).as_dict()}

    values = {period: Decimal(str(rows[period]["delivered_sales"]))
              for period in required}
    analysis = SalesAnalysisResult(
        intent=plan.kind, response_mode=mode, metric=plan.metric,
        periods=list(effective), values=[values[period] for period in effective],
        aggregation=plan.aggregation, period_selection=plan.period_selection,
        limitations=(["Chỉ tính các tháng đã hoàn thành."]
                     if len(effective) < len(plan.periods) else []),
    )
    if plan.kind == "MAX_MIN_PERIOD":
        chosen = (max if plan.aggregation == "max" else min)(
            effective, key=lambda period: values[period])
        analysis.details = {"selected_period": chosen, "selected_value": values[chosen]}
        analysis.observations = [f"{plan.aggregation}_month:{chosen}"]
    elif plan.kind == "PERIOD_TREND":
        analysis.difference = values[effective[-1]] - values[effective[0]]
        analysis.observations = ["end_above_start" if analysis.difference > 0 else
                                 "end_below_start" if analysis.difference < 0 else
                                 "end_equals_start"]
    elif plan.kind in {"RANGE_AGGREGATE", "YEAR_AGGREGATE"}:
        total = sum((values[period] for period in effective), Decimal(0))
        if plan.aggregation == "monthly_average":
            analysis.details = {"aggregate_value": total / len(effective), "total": total}
        else:
            analysis.details = {"aggregate_value": total}
    elif plan.aggregation == "max_min_spread":
        high = max(effective, key=lambda period: values[period])
        low = min(effective, key=lambda period: values[period])
        analysis.difference = values[high] - values[low]
        analysis.details = {"high_period": high, "high_value": values[high],
                            "low_period": low, "low_value": values[low]}
        analysis.observations = [f"high:{high}", f"low:{low}"]
    elif plan.aggregation == "largest_mom_decline":
        changes = [(period, values[period] - values[_previous(period)])
                   for period in effective]
        declines = [(period, change) for period, change in changes if change < 0]
        if declines:
            chosen, change = min(declines, key=lambda item: item[1])
            prior = _previous(chosen)
            analysis.difference = change
            analysis.percent_change = change / values[prior] * 100 if values[prior] else None
            analysis.details = {"selected_period": chosen, "previous_period": prior,
                                "previous_value": values[prior],
                                "current_value": values[chosen]}
            analysis.observations = [f"largest_decline:{chosen}"]
        else:
            analysis.details = {"selected_period": None}
            analysis.observations = ["no_decline"]
    else:
        raise ValueError("Unsupported deterministic Sales aggregation")
    answer = render(analysis, identity["restaurant_name"])
    if analysis.limitations:
        answer += " " + " ".join(analysis.limitations)
    return {**base, "status": "ok", "answer": answer,
            "analysis": analysis.as_dict(),
            "display_rows": [{"period": period, "value": float(values[period])}
                             for period in effective]}
