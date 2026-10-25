"""Compare two predefined monthly SQL aggregates in deterministic Python."""

from src.sales import queries

LABELS = {
    "delivered_sales": "doanh thu Delivered",
    "aov": "AOV Delivered",
    "cancellation_rate": "tỷ lệ hủy đơn",
}


def _display_period(period):
    return f"{period[5:7]}/{period[:4]}"


def _display_value(value, metric):
    if metric == "cancellation_rate":
        return f"{float(value) * 100:.2f}%"
    return f"{float(value):,.0f} VND"


def answer_comparison(conn, restaurant_id, identity, plan, *, today=None):
    """Return only values read from the two exact, parameterized SQL periods."""
    rows = [queries.monthly_sales(conn, restaurant_id, period, today=today)
            for period in plan.periods]
    facts = {period: queries.jsonable(row) for period, row in zip(
        plan.periods, rows, strict=True)}
    missing = [period for period, row in zip(plan.periods, rows, strict=True)
               if row is None or row.get(plan.metric) is None]
    trace = {"kind": plan.kind, "metric": plan.metric,
             "periods": list(plan.periods), "comparison": plan.comparison}
    if missing:
        labels = ", ".join(_display_period(period) for period in missing)
        return {"route": "SALES_SQL", "source": "SQL Analytics",
                "answer": f"Không có dữ liệu {LABELS[plan.metric]} cho kỳ {labels}; không thể so sánh.",
                "restaurant_id": str(restaurant_id), "sql_facts": facts,
                "query_plan": trace, "review_evidence": [], "status": "no_data"}
    first, second = (row[plan.metric] for row in rows)
    difference = first - second
    first_label, second_label = map(_display_period, plan.periods)
    if plan.metric == "cancellation_rate":
        change = f"{float(difference) * 100:+.2f} điểm phần trăm"
    else:
        absolute = f"{float(difference):+,.0f} VND"
        relative = (f"{float(difference / second) * 100:+.2f}%"
                    if second else "không tính được tỷ lệ vì kỳ đối chiếu bằng 0")
        change = f"{absolute} ({relative})"
    answer = (f"So sánh {LABELS[plan.metric]} của {identity['restaurant_name']}: "
              f"{first_label} = {_display_value(first, plan.metric)}; "
              f"{second_label} = {_display_value(second, plan.metric)}. "
              f"Chênh lệch ({first_label} − {second_label}): {change}.")
    return {"route": "SALES_SQL", "source": "SQL Analytics",
            "answer": answer, "restaurant_id": str(restaurant_id),
            "sql_facts": facts, "query_plan": trace,
            "review_evidence": [], "status": "ok"}
