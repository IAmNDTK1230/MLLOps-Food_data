"""Dedicated SQL-backed Sales Analytics presentation."""

import pandas as pd
import streamlit as st

from src.sales.response import money as vietnamese_money
from src.sales.response import percentage as vietnamese_percentage
from src.ui.components import answer_card, empty_state, evidence_card, header, money, percent


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
    from src.foody.review_runtime import ENRICHED, new_corpus_active

    if new_corpus_active():
        frame = pd.read_parquet(ENRICHED,
                                columns=["review_id", "restaurant_id", "comment", "rating",
                                         "review_time"])
        rows = frame.loc[frame.review_id.isin(review_ids) &
                         frame.restaurant_id.astype(str).eq(str(restaurant_key))]
        return {row.review_id: {"text": row.comment, "rating": row.rating,
                                "date": row.review_time} for row in rows.itertuples()}
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


def cross_evidence_metadata(item):
    """Show only metadata carried by the cited review, including missing values."""
    rating = item.get("rating")
    rating_label = f"{rating}/10" if rating is not None and pd.notna(rating) else "unavailable"
    date = item.get("date")
    date_label = str(date)[:10] if date is not None and pd.notna(date) else "unavailable"
    aspects = item.get("aspects") or []
    return (f"Date: {date_label} · Rating: {rating_label} · "
            f"Sentiment: {(item.get('sentiment') or 'unavailable').title()} · "
            f"Aspects: {', '.join(aspects) if aspects else 'unavailable'} · "
            f"Source: {item.get('source') or 'unavailable'}")


def sales_result_table(result):
    """Present verified monthly facts without changing their values."""
    rows = result.get("display_rows") or []
    metric = (result.get("analysis") or {}).get("metric", "delivered_sales")
    title = {"delivered_sales": "Revenue", "aov": "AOV",
             "cancellation_rate": "Cancellation rate"}.get(metric, "Value")
    formatter = vietnamese_percentage if metric == "cancellation_rate" else vietnamese_money
    return pd.DataFrame([{"Period": f"{row['period'][5:7]}/{row['period'][:4]}",
                          title: formatter(row["value"] * 100 if metric ==
                                           "cancellation_rate" else row["value"])}
                         for row in rows])


def show_sales_result(result):
    answer_card("Verified Analysis", result["answer"], "Verified by SQL", "good")
    table = sales_result_table(result)
    if not table.empty and len(table) >= 2:
        st.dataframe(table, hide_index=True, width="stretch")
        if (result.get("analysis") or {}).get("response_mode") == "TREND":
            chart = pd.DataFrame(result["display_rows"])
            st.line_chart(chart.set_index("period")["value"], height=220)
    change = (result.get("analysis") or {}).get("percent_change")
    if change is not None and (result.get("analysis") or {}).get("response_mode") == "COMPARISON":
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        st.caption(f"{direction} {vietnamese_percentage(abs(change))} vs comparison period")


def _delta(value):
    if value is None:
        return None
    return f"{float(value):+.2%}"


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

    st.subheader("✨ AI Sales")
    st.caption("Ask questions about revenue, orders, AOV or trends.")
    question = st.text_input("Ask AI Sales", key="sales_question",
                             placeholder="Ví dụ: So sánh doanh thu tháng 5/2026 với cùng kỳ năm trước")
    if st.button("✨ Analyze", type="primary", key="ask_sales"):
        if not question.strip():
            empty_state("Enter a sales question to continue")
        elif route_question(question) != "SALES_SQL":
            st.info("Ask about revenue or orders here. Customer feedback belongs in Review Intelligence.")
        else:
            try:
                with st.spinner("Querying warehouse..."):
                    with connection() as conn:
                        result = answer_question(conn, restaurant_id, question)
                if result["status"] == "invalid_period":
                    st.warning(result["answer"])
                    return
                show_sales_result(result)
                with st.expander("Technical details"):
                    st.json({key: result.get(key) for key in
                             ("route", "status", "query_plan", "analysis", "sql_facts")})
            except Exception as error:
                st.error("Unable to query the sales warehouse.")
                with st.expander("Technical details"):
                    st.code(f"{type(error).__name__}: {error}")


def _cross(restaurant):
    from src.sales.router import route_question
    from src.sales.service import answer_question
    from src.warehouse.loader import connection

    st.subheader("Sales + Customer Feedback")
    st.caption("Combine verified sales metrics with customer review evidence.")
    from src.foody.review_runtime import new_corpus_active

    review_id = restaurant.get("review_restaurant_id") if new_corpus_active() else \
        restaurant.get("restaurant_key")
    if not review_id:
        empty_state("Customer feedback is not linked to this sales restaurant.")
        return
    question = st.text_input("Ask about sales and reviews", key="cross_question",
                             placeholder="Doanh thu giảm thì khách phàn nàn gì?")
    if st.button("✨ Analyze together", key="ask_cross"):
        if route_question(question) != "CROSS_ANALYSIS":
            empty_state("Ask about both sales and reviews",
                        "Include a sales metric and customer feedback in the question.")
            return
        try:
            with st.spinner("Analyzing verified evidence..."):
                with connection() as conn:
                    result = answer_question(conn, restaurant["restaurant_id"], question)
            if result["status"] == "invalid_period":
                st.warning(result["answer"])
                return
            tone = "good" if result["status"] == "grounded" else "warn"
            answer_card("Cross-analysis", result["answer"],
                        "Grounded" if tone == "good" else "Limited evidence", tone)
            ids = tuple(item["review_id"] for item in result.get("review_evidence") or [])
            snippets = review_snippets(ids, review_id)
            for index, item in enumerate(result.get("review_evidence") or [], 1):
                snippet = snippets.get(item["review_id"], {})
                evidence_card(item.get("ref") or f"E{index}",
                              snippet.get("text", item.get("text") or "Review source text unavailable"),
                              cross_evidence_metadata(item))
            with st.expander("Technical details"):
                st.json({key: result.get(key) for key in
                         ("route", "status", "sql_facts", "aligned_review_count")})
        except Exception as error:
            st.error("Unable to complete cross-analysis.")
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")


def sales_page():
    header("Sales Intelligence", "Explore revenue, orders and business performance using verified warehouse data.",
           ("PostgreSQL", "Verified Analytics"))
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
    st.caption(f"{len(restaurants):,} sales restaurants available")
    selected_id = st.selectbox("Restaurant", list(by_id),
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
