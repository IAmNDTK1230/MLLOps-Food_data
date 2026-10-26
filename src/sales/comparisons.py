"""Compare two predefined monthly SQL aggregates in deterministic Python."""

from decimal import Decimal

from src.sales import queries
from src.sales.response import SalesAnalysisResult, render_with_optional_naturalizer, response_mode

LABELS = {
    "delivered_sales": "doanh thu Delivered",
    "aov": "AOV Delivered",
    "cancellation_rate": "tỷ lệ hủy đơn",
}


def _display_period(period):
    return f"{period[5:7]}/{period[:4]}"


def answer_comparison(conn, restaurant_id, identity, plan, *, today=None,
                      question="", naturalizer=None):
    """Return only values read from the two exact, parameterized SQL periods."""
    rows = [queries.monthly_sales(conn, restaurant_id, period, today=today)
            for period in plan.periods]
    facts = {period: queries.jsonable(row) for period, row in zip(
        plan.periods, rows, strict=True)}
    missing = [period for period, row in zip(plan.periods, rows, strict=True)
               if row is None or row.get(plan.metric) is None]
    trace = {"kind": plan.kind, "intent": plan.kind, "metric": plan.metric,
             "periods": list(plan.periods), "comparison": plan.comparison,
             "range": {"start": plan.periods[0], "end": plan.periods[-1]},
             "aggregation": "difference", "analytical_steps": ["fetch_monthly_metrics",
             "calculate_difference"], "sql_mode": "predefined_parameterized_monthly_sales",
             "sql_functions": ["monthly_sales"], "calculations": ["subtract", "percent_change"],
             "period_selection": plan.period_selection or "EXPLICIT_SET"}
    mode = response_mode(plan, question)
    trace.update(response_mode=mode, qwen_used=False,
                 grounding_status="sql_verified", fact_validation_status="not_needed")
    if missing:
        labels = ", ".join(_display_period(period) for period in missing)
        return {"route": "SALES_SQL", "source": "SQL Analytics",
                "answer": f"Không có dữ liệu {LABELS[plan.metric]} cho kỳ {labels}; không thể so sánh.",
                "restaurant_id": str(restaurant_id), "sql_facts": facts,
                "query_plan": trace, "review_evidence": [], "status": "no_data",
                "analysis": SalesAnalysisResult(plan.kind, mode, plan.metric,
                                                list(plan.periods), [],
                                                limitations=["missing SQL facts"]).as_dict()}
    first, second = (Decimal(str(row[plan.metric])) for row in rows)
    difference = first - second
    percent = difference / second * 100 if second and plan.metric != "cancellation_rate" else None
    limitations = []
    if mode == "EXPLANATION":
        limitations.append("Dữ liệu Sales chỉ xác nhận mức thay đổi, không đủ để xác định nguyên nhân.")
        if identity.get("restaurant_key") is not None:
            limitations.append("Nhà hàng có ánh xạ review đã xác minh; có thể dùng Cross Analysis để đối chiếu review cùng kỳ, nhưng sự đồng thời không chứng minh quan hệ nhân quả.")
        else:
            limitations.append("Chưa có ánh xạ review xác minh; cần thêm dữ liệu về review, giá bán, chiến dịch hoặc vận hành để phân tích nguyên nhân.")
    analysis = SalesAnalysisResult(
        intent=plan.kind, response_mode=mode, metric=plan.metric,
        periods=list(plan.periods), values=[first, second], difference=difference,
        percent_change=percent, aggregation="difference",
        observations=["first_higher" if difference > 0 else
                      "first_lower" if difference < 0 else "unchanged"],
        limitations=limitations, period_selection=plan.period_selection,
        details={"cross_available": identity.get("restaurant_key") is not None},
    )
    answer, used, validation = render_with_optional_naturalizer(
        analysis, identity["restaurant_name"], naturalizer)
    trace.update(qwen_used=used, fact_validation_status=validation,
                 grounding_status="cross_available_not_run" if
                 mode == "EXPLANATION" and identity.get("restaurant_key") is not None
                 else "sales_only_cause_unverified" if mode == "EXPLANATION" else "sql_verified")
    return {"route": "SALES_SQL", "source": "SQL Analytics",
            "answer": answer, "restaurant_id": str(restaurant_id),
            "sql_facts": facts, "query_plan": trace, "analysis": analysis.as_dict(),
            "review_evidence": [], "status": "ok",
            "display_rows": [{"period": period, "value": float(value)}
                             for period, value in zip(plan.periods, analysis.values, strict=True)]}
