"""Dedicated SQL-backed Sales Analytics presentation."""

import pandas as pd
import streamlit as st

from src.ui.components import answer_card, badge, empty_state, evidence_card, header, money, percent


@st.cache_data(ttl=300, show_spinner=False)
def sales_restaurants():
    from src.sales import queries
    from src.warehouse.loader import connection

    with connection() as conn:
        return queries.list_restaurants(conn)


@st.cache_data(ttl=120, show_spinner=False)
def sales_snapshot(restaurant_id):
    from src.sales import queries
    from src.warehouse.loader import connection

    with connection() as conn:
        return {
            "monthly": queries.monthly_trend(conn, restaurant_id, limit=36),
            "weekly": queries.weekly_trend(conn, restaurant_id, limit=52),
            "statuses": queries.order_status_summary(conn, restaurant_id),
            "summary": queries.sales_summary(conn, restaurant_id),
        }


@st.cache_data(ttl=120, show_spinner=False)
def month_snapshot(restaurant_id, month):
    from src.sales import queries
    from src.warehouse.loader import connection

    with connection() as conn:
        return queries.monthly_comparison(conn, restaurant_id, month)


@st.cache_data(ttl=300, show_spinner=False)
def review_snippets(review_ids, restaurant_key):
    """Resolve already accepted evidence IDs to local review text for display only."""
    if not review_ids or not restaurant_key:
        return {}
    frame = pd.read_parquet("data/foody/enriched/reviews.parquet",
                            columns=["review_id", "restaurant_key", "text", "raw_rating",
                                     "review_timestamp"])
    return select_review_snippets(frame, review_ids, restaurant_key)


def select_review_snippets(frame, review_ids, restaurant_key):
    """Never display a review from a different restaurant for a cross result."""
    rows = frame.loc[frame.review_id.isin(review_ids) &
                     frame.restaurant_key.eq(restaurant_key)].drop_duplicates("review_id")
    return {row.review_id: {"text": row.text, "rating": row.raw_rating,
                            "date": row.review_timestamp} for row in rows.itertuples()}


def _delta(value):
    if value is None:
        return None
    return f"{float(value):+.2%} vs previous period"


def _charts(snapshot):
    left, right = st.columns(2)
    monthly = pd.DataFrame(snapshot["monthly"])
    weekly = pd.DataFrame(snapshot["weekly"])
    statuses = pd.DataFrame(snapshot["statuses"])
    for frame in (monthly, weekly):
        if not frame.empty:
            frame["delivered_sales"] = pd.to_numeric(frame["delivered_sales"], errors="coerce")
    if not monthly.empty:
        monthly["mom_growth"] = pd.to_numeric(monthly["mom_growth"], errors="coerce")
    with left:
        st.subheader("Monthly revenue")
        if monthly.empty:
            empty_state("No monthly sales yet", "Choose another restaurant.")
        else:
            st.line_chart(monthly.set_index("period_start")["delivered_sales"],
                          color="#4F8EF7", height=260)
    with right:
        st.subheader("Weekly revenue")
        if weekly.empty:
            empty_state("No weekly sales yet")
        else:
            st.line_chart(weekly.set_index("period_start")["delivered_sales"],
                          color="#2ECC71", height=260)
    left, right = st.columns(2)
    with left:
        st.subheader("Order status · all observed dates")
        if statuses.empty:
            empty_state("No orders available")
        else:
            counts = statuses.set_index("order_status")["order_count"]
            counts = counts.reindex(["Delivered", "Cancelled", "Refunded"], fill_value=0)
            st.bar_chart(counts, color="#4F8EF7", height=250)
    with right:
        st.subheader("Month over month growth")
        growth = monthly.dropna(subset=["mom_growth"]) if not monthly.empty else monthly
        if growth.empty:
            empty_state("No prior month for comparison")
        else:
            st.bar_chart(growth.set_index("period_start")["mom_growth"],
                         color="#F5B942", height=250)


def _performance_table(rows):
    if not rows:
        return
    st.subheader("Monthly performance")
    with st.expander("View monthly details"):
        display = pd.DataFrame([
            {"Month": row["period_start"].strftime("%Y-%m"),
             "Revenue": money(row["delivered_sales"]),
             "Orders": f"{row['order_count']:,}",
             "AOV": money(row["aov"]),
             "Cancellation rate": percent(row["cancellation_rate"]),
             "MoM growth": percent(row["mom_growth"])}
            for row in reversed(rows)
        ])
        st.dataframe(display, hide_index=True, width="stretch")


def _ask_sql(restaurant_id):
    from src.sales.router import route_question
    from src.sales.service import answer_question
    from src.warehouse.loader import connection

    st.subheader("Ask about this restaurant")
    st.caption("Answers come from verified PostgreSQL aggregates.")
    suggestions = ["Doanh thu tháng này?", "Tháng doanh thu nào cao nhất?",
                   "Tỷ lệ đơn hủy?", "AOV là bao nhiêu?",
                   "So sánh doanh thu tháng trước?"]
    columns = st.columns(len(suggestions))
    for column, suggestion in zip(columns, suggestions, strict=True):
        if column.button(suggestion, key=f"sales_suggest_{suggestion}"):
            st.session_state["sales_question"] = suggestion
    question = st.text_input("Sales question", key="sales_question",
                             placeholder="Ví dụ: Doanh thu tháng 5/2025 là bao nhiêu?")
    if st.button("Ask Sales SQL", type="primary", key="ask_sales"):
        if not question.strip():
            empty_state("Enter a sales question to continue")
        elif route_question(question) != "SALES_SQL":
            empty_state("Ask a sales question here", "Review and cross questions have their own sections.")
        else:
            try:
                with st.spinner("Querying warehouse..."):
                    with connection() as conn:
                        result = answer_question(conn, restaurant_id, question)
                answer_card("SQL answer", result["answer"], "SQL Verified", "good")
                with st.expander("Technical details"):
                    st.json({key: result.get(key) for key in
                             ("route", "status", "query_plan", "sql_facts")})
            except Exception as error:
                st.error("Unable to query the sales warehouse.")
                with st.expander("Technical details"):
                    st.code(f"{type(error).__name__}: {error}")


def _cross(restaurant):
    from src.sales.router import route_question
    from src.sales.service import answer_question
    from src.warehouse.loader import connection

    st.subheader("Review + Sales Cross Analysis")
    if not restaurant.get("restaurant_key"):
        empty_state("Cross-analysis unavailable",
                    "No verified review mapping exists for this sales restaurant. Sales analytics remains available.")
        return
    badge("Verified Mapping", "good")
    st.caption("Sales facts and review evidence are aligned only for this verified restaurant.")
    examples = ["Doanh thu giảm thì khách phàn nàn gì?",
                "Tháng doanh thu thấp nhất có review tiêu cực nào nổi bật?",
                "Doanh thu và phản hồi khách hàng cho thấy vấn đề nào cần ưu tiên?"]
    columns = st.columns(3)
    for column, example in zip(columns, examples, strict=True):
        if column.button(example, key=f"cross_suggest_{example}"):
            st.session_state["cross_question"] = example
    question = st.text_input("Cross-analysis question", key="cross_question",
                             placeholder="Doanh thu giảm thì khách phàn nàn gì?")
    if st.button("Analyze Sales + Reviews", key="ask_cross"):
        if route_question(question) != "CROSS_ANALYSIS":
            empty_state("Ask about both sales and reviews",
                        "Include a sales metric and customer feedback in the question.")
            return
        try:
            with st.spinner("Analyzing verified evidence..."):
                with connection() as conn:
                    result = answer_question(conn, restaurant["restaurant_id"], question)
            tone = "good" if result["status"] == "grounded" else "warn"
            answer_card("Cross-analysis", result["answer"],
                        "Grounded" if tone == "good" else "Limited evidence", tone)
            ids = tuple(item["review_id"] for item in result.get("review_evidence") or [])
            snippets = review_snippets(ids, restaurant["restaurant_key"])
            for index, item in enumerate(result.get("review_evidence") or [], 1):
                snippet = snippets.get(item["review_id"], {})
                evidence_card(f"E{index}", snippet.get("text", "Review source text unavailable"),
                              f"Rating {snippet.get('rating', 'N/A')} · {snippet.get('date', 'Date unavailable')}")
            with st.expander("Technical details"):
                st.json({key: result.get(key) for key in
                         ("route", "status", "sql_facts", "aligned_review_count")})
        except Exception as error:
            st.error("Unable to complete cross-analysis.")
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")


def sales_page():
    header("Sales Analytics", "Verified metrics from PostgreSQL Data Warehouse",
           ("Source: PostgreSQL", "Structured SQL Analytics"))
    try:
        restaurants = sales_restaurants()
    except Exception as error:
        st.error("Unable to connect to the sales warehouse.")
        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")
        return
    if not restaurants:
        empty_state("No sales restaurants available", "Load the sales warehouse to begin.")
        return
    by_id = {str(item["restaurant_id"]): item for item in restaurants}
    st.caption(f"{len(restaurants):,} sales restaurants · independent from review restaurant coverage")
    if st.button("Use Thai Market demo example", key="thai_demo") and "1853" in by_id:
        st.session_state["sales_restaurant"] = "1853"
        st.session_state["sales_period"] = "2025-05"
        st.session_state["_sales_period_restaurant"] = "1853"
    selected_id = st.selectbox("Sales restaurant", list(by_id),
                               format_func=lambda value: f"{by_id[value]['restaurant_name']} · ID {value}",
                               key="sales_restaurant")
    restaurant = by_id[selected_id]
    try:
        snapshot = sales_snapshot(selected_id)
        months = [row["period_start"].strftime("%Y-%m") for row in snapshot["monthly"]]
        if not months:
            empty_state("No observed sales periods for this restaurant")
            return
        from src.sales.queries import vietnam_today

        completed = [month for month in months if month < vietnam_today().strftime("%Y-%m")]
        if (st.session_state.get("_sales_period_restaurant") != selected_id or
                st.session_state.get("sales_period") not in months):
            st.session_state["sales_period"] = completed[-1] if completed else months[-1]
            st.session_state["_sales_period_restaurant"] = selected_id
        period = st.selectbox("Month", list(reversed(months)), key="sales_period")
        metric = month_snapshot(selected_id, period)
    except Exception as error:
        st.error("Unable to load sales analytics.")
        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")
        return
    if metric is None:
        empty_state("No data available for the selected month")
        return
    st.caption(f"Selected period: {period} · completed months use full prior-month comparison")
    cards = st.columns(5)
    cards[0].metric("Delivered Revenue", money(metric["delivered_sales"]),
                    _delta(metric.get("comparison_growth")))
    cards[1].metric("Total Orders", f"{metric['order_count']:,}")
    cards[2].metric("Average Order Value", money(metric["aov"]))
    summary = snapshot["summary"] or {}
    cards[3].metric("Cancellation Rate", percent(summary.get("cancellation_rate")),
                    help="All observed orders for this restaurant")
    cards[4].metric("Refund Rate", percent(summary.get("refund_rate")),
                    help="All observed orders for this restaurant")
    st.caption("Revenue, orders and AOV use the selected month. Cancellation and refund rates "
               "cover all observed orders for this restaurant.")
    _charts(snapshot)
    _performance_table(snapshot["monthly"])
    _ask_sql(selected_id)
    _cross(restaurant)
