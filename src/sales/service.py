"""Grounded SQL answers and date-aligned review cross-analysis."""

from datetime import datetime, timedelta
from time import perf_counter

from src.monitoring.metrics import observe_cross_analysis, observe_rag, observe_sales_sql
from src.rag.monitoring import record
from src.sales import queries
from src.sales.router import (
    month_in_question,
    route_question,
    wants_current_month,
    wants_current_week,
)


def _money(value):
    return f"{value:,.0f} VND" if value is not None else "không có dữ liệu"


def _percent(value):
    return f"{value * 100:+.2f}%" if value is not None else "không đủ dữ liệu so sánh"


def _month_for_question(question, today):
    explicit = month_in_question(question)
    if explicit:
        return explicit
    if wants_current_month(question):
        return today.strftime("%Y-%m")
    return None


def _record(route, started, status, *, input_tokens=0, output_tokens=0,
            generation_mode="none", evidence_count=0):
    elapsed_ms = (perf_counter() - started) * 1000
    record(route, "PostgreSQL + RAG V3" if route == "cross_analysis" else "PostgreSQL",
           "003", "sales_sql_v1", elapsed_ms, status,
           input_tokens=input_tokens, output_tokens=output_tokens,
           rag_version="RAG_V3" if route == "cross_analysis" else None,
           query_route=route, generation_mode=generation_mode,
           evidence_count=evidence_count)
    if route == "sales_sql":
        observe_sales_sql(elapsed_ms / 1000, status)
    else:
        observe_cross_analysis(status)


def sales_answer(conn, restaurant_id, question, *, today=None):
    """Answer numeric questions from predefined SQL only."""
    started = perf_counter()
    today = today or queries.vietnam_today()
    identity = queries.restaurant(conn, restaurant_id)
    if identity is None:
        raise ValueError("Unknown sales restaurant_id")
    lowered = question.casefold()
    month = _month_for_question(question, today)
    source = "SQL Analytics"
    if "cao nhất" in lowered or "highest" in lowered:
        facts = queries.highest_sales_month(conn, restaurant_id, today=today)
        answer = (f"Tháng doanh thu Delivered cao nhất của {identity['restaurant_name']} "
                  f"là {facts['period_start']:%Y-%m}: {_money(facts['delivered_sales'])}.") if facts else \
            "Chưa có tháng hoàn chỉnh để xếp hạng doanh thu."
    elif "tuần" in lowered or "tuan" in lowered or "week" in lowered:
        week = today.fromordinal(today.toordinal() - today.weekday()) if \
            wants_current_week(question) else None
        facts = queries.weekly_sales(conn, restaurant_id, week, today=today)
        answer = (f"Doanh thu Delivered tuần bắt đầu {facts['period_start']}: "
                  f"{_money(facts['delivered_sales'])}; {facts['order_count']} đơn, "
                  f"AOV {_money(facts['aov'])}.") if facts else "Không có đơn trong tuần được hỏi."
    elif any(term in lowered for term in ("cancelled", "refunded", "delivered", "đơn hủy",
                                         "tỷ lệ", "aov", "order status")):
        facts = queries.sales_summary(conn, restaurant_id)
        answer = (f"{identity['restaurant_name']}: {facts['order_count']} đơn đã quan sát; "
                  f"Delivered {facts['delivered_count']}, Cancelled {facts['cancelled_count']}, "
                  f"Refunded {facts['refunded_count']}; tỷ lệ hủy "
                  f"{facts['cancellation_rate'] * 100:.2f}%, tỷ lệ hoàn "
                  f"{facts['refund_rate'] * 100:.2f}%, AOV Delivered "
                  f"{_money(facts['aov'])}.") if facts else "Chưa có đơn đã quan sát."
    else:
        facts = queries.monthly_comparison(conn, restaurant_id, month, today=today)
        if facts:
            basis = (" So sánh với cùng số ngày của tháng trước; tháng này chưa hoàn tất."
                     if facts["comparison_basis"] == "same_day_of_month_partial" else "")
            answer = (f"Doanh thu Delivered của {identity['restaurant_name']} trong "
                      f"{facts['period_start']:%Y-%m}: {_money(facts['delivered_sales'])}; "
                      f"{facts['order_count']} đơn, AOV {_money(facts['aov'])}; "
                      f"thay đổi {_percent(facts['comparison_growth'])} so với kỳ trước.{basis}")
        else:
            answer = "Không có dữ liệu đơn hàng trong tháng được hỏi."
    status = "ok" if facts else "no_data"
    _record("sales_sql", started, status)
    return {"route": "SALES_SQL", "source": source, "answer": answer,
            "restaurant_id": str(restaurant_id), "sql_facts": queries.jsonable(facts),
            "review_evidence": [], "status": status}


def _aligned_review_count(conn, restaurant_key, start, end):
    return conn.execute("""SELECT COUNT(*) FROM fact_review r
        JOIN dim_date d ON d.date_key=r.date_key
        WHERE r.restaurant_key=%s AND d.calendar_date BETWEEN %s AND %s""",
        (restaurant_key, start, end)).fetchone()[0]


def _review_trend(conn, restaurant_key, start, end):
    return conn.execute("""SELECT COUNT(*), COUNT(*) FILTER
        (WHERE r.predicted_sentiment='negative') FROM fact_review r
        JOIN dim_date d ON d.date_key=r.date_key
        WHERE r.restaurant_key=%s AND d.calendar_date BETWEEN %s AND %s""",
        (restaurant_key, start, end)).fetchone()


def cross_answer(conn, restaurant_id, question, *, today=None,
                 retriever_factory=None, gateway_factory=None):
    """Combine SQL facts with separately grounded, same-period review evidence."""
    started = perf_counter()
    today = today or queries.vietnam_today()
    identity = queries.restaurant(conn, restaurant_id)
    if identity is None:
        raise ValueError("Unknown sales restaurant_id")
    if identity["restaurant_key"] is None:
        answer = ("Không có ánh xạ ID nhà hàng được xác minh giữa đơn hàng và đánh giá; "
                  "không thể tổng hợp hai nguồn cho nhà hàng này.")
        _record("cross_analysis", started, "unmatched_restaurant")
        return {"route": "CROSS_ANALYSIS", "source": "Hybrid SQL + RAG",
                "answer": answer, "sql_facts": None, "review_evidence": [],
                "restaurant_id": str(restaurant_id), "status": "unmatched_restaurant"}
    month = _month_for_question(question, today)
    lowered = question.casefold()
    if "giảm mạnh nhất" in lowered or "largest decline" in lowered:
        metric = queries.lowest_growth_month(conn, restaurant_id, today=today)
        month = metric["period_start"].strftime("%Y-%m") if metric else None
    else:
        metric = queries.monthly_comparison(conn, restaurant_id, month, today=today)
    if metric is None:
        _record("cross_analysis", started, "no_sales_data")
        return {"route": "CROSS_ANALYSIS", "source": "Hybrid SQL + RAG",
                "answer": "Không có dữ liệu doanh thu cho kỳ được hỏi.",
                "sql_facts": None, "review_evidence": [],
                "restaurant_id": str(restaurant_id), "status": "no_sales_data"}
    start = metric["period_start"]
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = min(next_month - timedelta(days=1), today)
    growth = metric.get("comparison_growth", metric.get("mom_growth"))
    sales_text = (f"Theo SQL, doanh thu Delivered {start:%Y-%m} là "
                  f"{_money(metric['delivered_sales'])}, {metric['order_count']} đơn; "
                  f"thay đổi {_percent(growth)} so với kỳ trước.")
    count = _aligned_review_count(conn, identity["restaurant_key"], start, end)
    if count == 0:
        answer = (sales_text + " Không có đánh giá cùng kỳ cho đúng nhà hàng đã ánh xạ; "
                  "không đủ bằng chứng để liên hệ biến động doanh thu với phản hồi khách hàng.")
        _record("cross_analysis", started, "no_aligned_reviews")
        return {"route": "CROSS_ANALYSIS", "source": "Hybrid SQL + RAG",
                "answer": answer, "sql_facts": queries.jsonable(metric),
                "review_evidence": [], "restaurant_id": str(restaurant_id),
                "aligned_review_count": 0, "status": "no_aligned_reviews"}

    from src.rag.schemas import RetrievalRequest
    from src.rag_v3.cli import selected_v3_gateway
    from src.rag_v3.retriever import RoutedRetriever
    from src.rag_v3.service import answer_question_v3

    retriever_factory = retriever_factory or RoutedRetriever
    gateway_factory = gateway_factory or selected_v3_gateway
    request = RetrievalRequest(
        query="Khách phàn nàn gì về món ăn, phục vụ hoặc giá?",
        restaurant_key=identity["restaurant_key"],
        date_from=datetime.combine(start, datetime.min.time()),
        date_to=datetime.combine(end, datetime.min.time()), top_k=5,
    )
    with gateway_factory() as gateway:
        review_answer, _, telemetry, _, _ = answer_question_v3(
            retriever_factory(), gateway, request,
        )
    observe_rag(telemetry)
    evidence = [item.model_dump() for item in review_answer.evidence]
    if evidence:
        answer = (sales_text + " Phản hồi cùng kỳ: " + review_answer.answer
                  + " Đây là sự trùng thời gian, không chứng minh quan hệ nhân quả.")
        status = "grounded"
    else:
        answer = (sales_text + " Có đánh giá cùng kỳ nhưng RAG V3 không xác nhận "
                  "được bằng chứng phù hợp; không thể quy nguyên nhân cho phản hồi.")
        status = "insufficient_review_evidence"
    _record("cross_analysis", started, status,
            input_tokens=telemetry.get("input_tokens") or 0,
            output_tokens=telemetry.get("output_tokens") or 0,
            generation_mode=telemetry.get("generation_mode") or "none",
            evidence_count=len(evidence))
    return {"route": "CROSS_ANALYSIS", "source": "Hybrid SQL + RAG",
            "answer": answer, "sql_facts": queries.jsonable(metric),
            "review_evidence": evidence, "restaurant_id": str(restaurant_id),
            "aligned_review_count": count, "status": status,
            "review_generation_mode": telemetry.get("generation_mode")}


def answer_question(conn, restaurant_id, question, **kwargs):
    route = route_question(question)
    if route == "SALES_SQL":
        return sales_answer(conn, restaurant_id, question, today=kwargs.get("today"))
    if route == "CROSS_ANALYSIS":
        return cross_answer(conn, restaurant_id, question, **kwargs)
    raise ValueError("Review-only questions must use the unchanged RAG V3 path")
