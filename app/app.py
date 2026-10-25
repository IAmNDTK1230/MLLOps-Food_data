"""Five-page Foody analytics demo with evidence-first RAG V3."""

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from src.monitoring.metrics import observe_rag, start_streamlit_metrics
from src.ui.components import (
    answer_card,
    badge,
    badge_row,
    empty_state,
    evidence_card,
    header,
    money,
    percent,
)
from src.ui.sales_view import review_snippets, sales_page
from src.ui.styles import apply_styles


@st.cache_data(show_spinner=False)
def table(path):
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(ttl=15, show_spinner=False)
def operational_events(path="state/llmops.sqlite"):
    if not Path(path).exists():
        return pd.DataFrame()
    with sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(
            "SELECT timestamp,component,status,latency_ms,llm_latency_ms,"
            "input_tokens,output_tokens,query_route,generation_mode,evidence_count "
            "FROM events ORDER BY id DESC", conn,
        )


def available(path):
    if not Path(path).exists():
        empty_state("Data is unavailable", "Run the local demo preflight to inspect required artifacts.")
        return False
    return True


def choose_restaurant():
    path = "data/foody/mart/restaurant_summary.parquet"
    if not available(path):
        return None, None
    summary = table(path)
    names = summary.sort_values("restaurant_name").restaurant_name.tolist()
    from src.demo import demo_config

    preferred = demo_config()["default_restaurant"]
    chosen = st.selectbox("Restaurant", names,
                          index=names.index(preferred) if preferred in names else 0)
    row = summary.loc[summary.restaurant_name == chosen].iloc[0]
    return row.restaurant_key, row


def date_filter(frame, label):
    dated = frame.review_timestamp.dropna()
    if dated.empty:
        return frame
    minimum, maximum = dated.min().date(), dated.max().date()
    start, end = st.date_input(label, (minimum, maximum), min_value=minimum, max_value=maximum)
    return frame.loc[frame.review_timestamp.dt.date.between(start, end)]


def restaurant_page():
    header("Restaurant Intelligence", "Understand guest feedback and verified operational signals",
           ("Foody Reviews", "RAG V3"))
    key, summary = choose_restaurant()
    if key is None or not available("data/foody/enriched/reviews.parquet"):
        return
    frame = table("data/foody/enriched/reviews.parquet")
    frame = frame.loc[frame.restaurant_key == key].drop_duplicates("review_id")
    frame = date_filter(frame, "Review date range")
    if frame.empty:
        empty_state("No reviews for this date range", "Widen the date filter to see insights.")
        return
    predicted = frame.predicted_sentiment.value_counts(normalize=True)
    issues = frame.detected_issues.explode().dropna().value_counts()
    issues.index = [str(label).replace("_", " ").title() for label in issues.index]
    top_issue = str(issues.index[0]) if not issues.empty else "No signal"
    cards = st.columns(5)
    cards[0].metric("Review Count", f"{len(frame):,}")
    cards[1].metric("Average Rating · /10", f"{pd.to_numeric(frame.raw_rating, errors='coerce').mean():.1f}")
    cards[2].metric("Positive", f"{predicted.get('positive', 0):.1%}")
    cards[3].metric("Negative", f"{predicted.get('negative', 0):.1%}")
    cards[4].metric("Main Issue", top_issue)
    trend = frame.loc[frame.review_timestamp.notna()].assign(
        period=lambda x: x.review_timestamp.dt.to_period("M").dt.to_timestamp(),
        negative=lambda x: x.predicted_sentiment.eq("negative"),
    ).groupby("period").negative.mean()
    left, right = st.columns(2)
    with left:
        st.subheader("Negative sentiment trend")
        if trend.empty:
            empty_state("No dated reviews in this selection")
        else:
            st.line_chart(trend, color="#FF6B6B", height=270)
    with right:
        st.subheader("Sentiment distribution")
        st.bar_chart(frame.predicted_sentiment.value_counts(), color="#4F8EF7", height=270)
    st.subheader("Top review aspects")
    aspects = frame.predicted_aspects.explode().dropna().value_counts().head(8)
    if aspects.empty:
        empty_state("No predicted aspects in this selection")
    else:
        st.bar_chart(aspects, color="#4F8EF7", height=240)
    st.subheader("Review insights")
    st.caption(f"{int(frame.valid_for_ai.sum()):,} AI-eligible reviews in this selection")
    with st.expander("Operational Signals · heuristic diagnostics"):
        st.caption("These signals are heuristic and are not measured model accuracy.")
        if issues.empty:
            empty_state("No heuristic signals in this selection")
        else:
            st.bar_chart(issues.head(10), color="#F5B942", height=220)
    alert_path = "data/foody/mart/alerts.parquet"
    if Path(alert_path).exists():
        alerts = table(alert_path)
        with st.expander("Historical rule alerts"):
            selected_alerts = alerts.loc[alerts.restaurant_key == key]
            if selected_alerts.empty:
                empty_state("No historical alerts for this restaurant")
            else:
                st.dataframe(selected_alerts, hide_index=True)
    try:
        from src.sales import queries
        from src.warehouse.loader import connection

        with connection() as conn:
            matched = next((item for item in queries.list_restaurants(conn, matched_only=True)
                            if item["restaurant_key"] == key), None)
            if matched:
                sales = queries.sales_summary(conn, matched["restaurant_id"])
                monthly = queries.monthly_trend(conn, matched["restaurant_id"])
                weekly = queries.weekly_trend(conn, matched["restaurant_id"])
                statuses = queries.order_status_summary(conn, matched["restaurant_id"])
            else:
                sales = monthly = weekly = statuses = None
        st.subheader("Verified Sales Analytics")
        if matched and sales:
            badge("Verified Mapping", "good")
            a, b, c, d = st.columns(4)
            a.metric("Delivered revenue", money(sales["delivered_sales"]))
            b.metric("Observed orders", sales["order_count"])
            c.metric("Delivered AOV", money(sales["aov"]))
            d.metric("Cancellation rate", percent(sales["cancellation_rate"]))
            if monthly:
                st.caption(f"Latest observed month: {monthly[-1]['period_start']:%Y-%m}")
                st.line_chart(pd.DataFrame(monthly).set_index("period_start")["delivered_sales"],
                              color="#4F8EF7", height=220)
            with st.expander("Sales source and order status"):
                st.caption(f"PostgreSQL · sales restaurant ID {matched['restaurant_id']} · "
                           "exact unique normalized-name match. Future orders excluded.")
                if statuses:
                    st.bar_chart(pd.DataFrame(statuses).set_index("order_status")["order_count"])
                if weekly:
                    st.line_chart(pd.DataFrame(weekly).set_index("period_start")["delivered_sales"])
        else:
            empty_state("Sales data is not available for this review restaurant.",
                        "Use Sales Analytics to explore the independent sales dataset.")
    except Exception as error:
        empty_state("Sales warehouse is unavailable", "Review analytics remain available.")
        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")
    if st.button("Generate Qwen health report"):
        try:
            from src.rag_v3.cli import selected_v3_gateway
            from src.rag_v3.health import health_report_v3
            from src.rag_v3.retriever import RoutedRetriever

            with st.spinner("Analyzing verified evidence..."):
                with selected_v3_gateway() as gateway:
                    report, telemetry, verification = health_report_v3(
                        RoutedRetriever(), gateway, key,
                    )
            for label, value in report.model_dump().items():
                if isinstance(value, (str, list)) and value:
                    st.write(f"**{label.replace('_', ' ').title()}**", value)
            with st.expander("Technical details"):
                st.json({"telemetry": telemetry, "verification": verification})
        except Exception as error:
            st.warning("Health report is currently unavailable. Restaurant insights remain visible.")
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")
    with st.expander("Data provenance"):
        st.caption("Restaurant key derives from its name; it is not an official Foody ID. "
                   "Foody AI predictions have no measured Foody accuracy.")


def explorer_page():
    header("Review Explorer", "Browse source reviews alongside clearly labeled AI predictions",
           ("Foody Reviews",))
    key, _ = choose_restaurant()
    if key is None or not available("data/foody/enriched/reviews.parquet"):
        return
    frame = table("data/foody/enriched/reviews.parquet")
    frame = frame.loc[frame.restaurant_key == key].drop_duplicates("review_id")
    filters = st.columns(3)
    with filters[0]:
        sentiment = st.selectbox("Predicted sentiment", ["All", "positive", "neutral", "negative"])
    with filters[1]:
        aspect = st.selectbox("Predicted aspect", ["All"] + sorted(set(
            frame.predicted_aspects.explode().dropna())))
    with filters[2]:
        issue = st.selectbox("Heuristic signal", ["All"] + sorted(set(
            frame.detected_issues.explode().dropna())))
    frame = date_filter(frame, "Review date range")
    if sentiment != "All":
        frame = frame.loc[frame.predicted_sentiment == sentiment]
    if aspect != "All":
        frame = frame.loc[frame.predicted_aspects.map(lambda x: aspect in x)]
    if issue != "All":
        frame = frame.loc[frame.detected_issues.map(lambda x: issue in x)]
    cards = st.columns(3)
    cards[0].metric("Reviews shown", f"{len(frame):,}")
    cards[1].metric("Average rating · /10",
                    f"{pd.to_numeric(frame.raw_rating, errors='coerce').mean():.1f}"
                    if not frame.empty else "—")
    cards[2].metric("Predicted positive", f"{frame.predicted_sentiment.eq('positive').mean():.1%}"
                    if not frame.empty else "—")
    if frame.empty:
        empty_state("No reviews match these filters", "Try a wider date range or fewer filters.")
        return
    st.subheader("Reviews")
    page_size = 10
    page = st.number_input("Page", min_value=1, max_value=max(1, (len(frame) - 1) // page_size + 1),
                           value=1, step=1)
    shown = frame.iloc[(page - 1) * page_size:page * page_size]
    for row in shown.itertuples():
        date = row.review_timestamp.strftime("%d %b %Y") if pd.notna(row.review_timestamp) else "Date unavailable"
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**Rating {row.raw_rating}/10** · {date}")
            with right:
                badge(str(row.predicted_sentiment).title(),
                      "good" if row.predicted_sentiment == "positive" else
                      "warn" if row.predicted_sentiment == "negative" else "muted")
            st.write(row.text)
            if len(row.predicted_aspects):
                st.caption("Predicted aspects: " + ", ".join(row.predicted_aspects))
            with st.expander("Technical details"):
                st.caption(f"Review ID: {row.review_id} · Confidence: {row.sentiment_confidence}")
                st.caption("Heuristic signals: " + ", ".join(row.detected_issues))
    st.caption("Sentiment and aspects are model predictions; heuristic signals are not measured accuracy.")


def analyst_page():
    header("AI Review Analyst", "Ask a question and inspect the reviews behind the answer",
           ("RAG V3", "Qwen2.5 3B"))
    key, _ = choose_restaurant()
    if key is None:
        return
    st.caption("Try: Khách nói gì về chất lượng món ăn? · "
               "Nhân viên phục vụ có vấn đề gì không? · "
               "Khách có phàn nàn về giá không? · "
               "Tại sao đánh giá của nhà hàng này giảm?")
    question = st.text_input("Question", "Nhân viên phục vụ có vấn đề gì không?")
    from src.sales.router import route_question

    route = route_question(question)
    sales_id = None
    if route != "REVIEW_RAG":
        try:
            from src.sales import queries
            from src.warehouse.loader import connection

            with connection() as conn:
                options = queries.list_restaurants(conn, matched_only=route == "CROSS_ANALYSIS")
            if options:
                selected = st.selectbox("Sales restaurant (verified mapping required for cross-analysis)",
                                        options, format_func=lambda item: f"{item['restaurant_name']} "
                                        f"· ID {item['restaurant_id']}",
                                        key="sales_restaurant_id")
                sales_id = selected["restaurant_id"]
        except Exception:
            empty_state("Sales warehouse is unavailable", "Review questions remain available.")
    if st.button("Ask", type="primary"):
        if route != "REVIEW_RAG":
            if sales_id is None:
                st.error("Choose a sales restaurant ID from the warehouse before asking.")
                return
            try:
                from src.sales.service import answer_question
                from src.warehouse.loader import connection

                with st.spinner("Analyzing verified evidence..."):
                    with connection() as conn:
                        result = answer_question(conn, sales_id, question)
                answer_card("Restaurant Analysis", result["answer"],
                            "SQL Verified" if result["route"] == "SALES_SQL" else
                            "Verified Mapping", "good")
                if result.get("review_evidence"):
                    st.subheader("Review evidence")
                    snippets = review_snippets(
                        tuple(item["review_id"] for item in result["review_evidence"]),
                        selected["restaurant_key"],
                    ) if result["route"] == "CROSS_ANALYSIS" else {}
                    for index, item in enumerate(result["review_evidence"], 1):
                        snippet = snippets.get(item["review_id"], {})
                        evidence_card(f"E{index}", snippet.get("text", "Review source text unavailable"),
                                      f"Rating {snippet.get('rating', 'N/A')} · "
                                      f"{snippet.get('date', 'Date unavailable')}")
                with st.expander("Technical details"):
                    st.json({key: result.get(key) for key in
                             ("source", "route", "status", "sql_facts")})
                return
            except Exception as error:
                st.error("Unable to complete restaurant analysis.")
                with st.expander("Technical details"):
                    st.code(f"{type(error).__name__}: {error}")
                return
        try:
            from src.rag.schemas import RetrievalRequest
            from src.rag_v3.cli import selected_v3_gateway
            from src.rag_v3.retriever import RoutedRetriever
            from src.rag_v3.service import answer_question_v3, generation_mode_label

            with st.spinner("Analyzing verified evidence..."):
                with selected_v3_gateway() as gateway:
                    answer, hits, telemetry, debug, packet = answer_question_v3(
                        RoutedRetriever(), gateway,
                        RetrievalRequest(query=question, restaurant_key=key, top_k=5),
                    )
            observe_rag(telemetry)
            mode = answer.generation_mode
            tone = "good" if mode == "qwen" else "warn" if mode == "extractive_fallback" else "muted"
            state_label = ("Grounded synthesis" if mode == "qwen" else
                           "Extractive fallback" if mode == "extractive_fallback" else
                           "Insufficient evidence")
            answer_card("AI Answer", answer.answer if mode != "none" else
                        "Không đủ bằng chứng để kết luận.", state_label, tone)
            badge_row((("RAG V3", "default"),
                       (telemetry["retrieval_method"].replace("_", " ").title(), "default"),
                       ("Qwen2.5" if mode == "qwen" else "Evidence first", "default")))
            st.caption(f"Generation mode: {generation_mode_label(mode)}")
            st.caption("Analysis is limited to the retrieved reviews; it does not "
                       "represent every customer.")
            st.subheader("Evidence")
            cited_ids = {item.review_id for item in answer.evidence}
            snippets = review_snippets(tuple(cited_ids), key)
            for item in packet.evidence:
                if item["review_id"] not in cited_ids:
                    continue
                snippet = snippets.get(item["review_id"], {})
                date = snippet.get("date")
                date_label = date.strftime("%d %b %Y") if pd.notna(date) else "Date unavailable"
                evidence_card(item["ref"], item["text"],
                              f"Rating {item['rating'] or 'N/A'} · {date_label} · "
                              f"Predicted sentiment {item['sentiment'] or 'N/A'} · "
                              f"Aspects {', '.join(item['aspects']) or 'N/A'}")
            if not cited_ids:
                empty_state("No review evidence accepted", "Try a narrower question or another restaurant.")
            with st.expander("Show technical trace"):
                st.write("Review IDs", [item.review_id for item in answer.evidence])
                st.write("Claim mappings", [claim.model_dump() for claim in answer.claims])
                st.write("Evidence ref → review ID", {
                    item["ref"]: item["review_id"] for item in packet.evidence
                })
                st.json({"prompt_version": answer.prompt_version,
                         "generation_mode": answer.generation_mode,
                         "retrieval_method": telemetry["retrieval_method"],
                         "support_gate_status": packet.support_gate_status,
                         "clause_support_status": telemetry["clause_support_status"],
                         "generation_attempts": telemetry["generation_attempts"],
                         "failure_category": telemetry["failure_category"],
                         "retrieval_debug": debug})
                st.caption(f"{len(answer.evidence)} cited of {len(hits)} retrieved · "
                           f"retrieval {telemetry['retrieval_latency_ms']:.1f} ms")
        except Exception as error:
            st.error("AI analysis is temporarily unavailable.")
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")


def monitor_page():
    header("LLMOps Monitor", "Operational health, usage, and evidence quality",
           ("SQLite Telemetry", "Prometheus / Grafana"))
    from src.demo import demo_config

    active = demo_config()
    try:
        events = operational_events()
    except sqlite3.Error as error:
        events = pd.DataFrame()
        empty_state("Operational telemetry is unavailable")
        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")
    rag = (events.loc[events.component.isin(["rag_request", "rag_v2_request",
                                            "rag_v3_request"])].copy()
           if not events.empty else pd.DataFrame())
    cards = st.columns(6)
    total = len(rag)
    latency = pd.to_numeric(rag.llm_latency_ms, errors="coerce") if total else pd.Series(dtype=float)
    cards[0].metric("RAG Requests", f"{total:,}")
    cards[1].metric("Avg LLM Latency", f"{latency.mean() / 1000:.1f} s" if latency.notna().any() else "—")
    cards[2].metric("Input Tokens", f"{pd.to_numeric(rag.input_tokens, errors='coerce').sum():,.0f}" if total else "0")
    cards[3].metric("Output Tokens", f"{pd.to_numeric(rag.output_tokens, errors='coerce').sum():,.0f}" if total else "0")
    cards[4].metric("Grounded", f"{rag.status.eq('grounded').mean():.1%}" if total else "—")
    cards[5].metric("Fallback", f"{rag.generation_mode.eq('extractive_fallback').mean():.1%}" if total else "—")
    if total:
        rag["day"] = pd.to_datetime(rag.timestamp, utc=True, errors="coerce").dt.date
        left, right = st.columns(2)
        with left:
            st.subheader("Requests over time")
            st.line_chart(rag.groupby("day").size(), color="#4F8EF7", height=230)
        with right:
            st.subheader("Generation modes")
            st.bar_chart(rag.generation_mode.fillna("unknown").value_counts(),
                         color="#2ECC71", height=230)
        left, right = st.columns(2)
        with left:
            st.subheader("Retrieval routes")
            st.bar_chart(rag.query_route.fillna("unknown").value_counts(),
                         color="#4F8EF7", height=230)
        with right:
            st.subheader("Token usage")
            tokens = rag.groupby("day")[["input_tokens", "output_tokens"]].sum()
            st.bar_chart(tokens, height=230)
        st.subheader("LLM latency")
        if latency.notna().any():
            st.line_chart(pd.Series(latency.dropna().iloc[::-1].to_numpy()),
                          color="#F5B942", height=200)
        else:
            empty_state("No measured LLM latency yet")
        st.subheader("Recent requests")
        st.dataframe(rag[["timestamp", "status", "generation_mode", "query_route",
                          "llm_latency_ms", "input_tokens", "output_tokens"]].head(20),
                     hide_index=True, width="stretch")
    else:
        empty_state("No RAG requests recorded yet", "Ask a question in AI Review Analyst to begin.")
    with st.expander("Technical details"):
        from src.foody.schema import digest
        from src.rag.monitoring import summary

        st.caption(f"Active {active['rag_version']} · {active['qwen_model']} · "
                   f"{active['retrieval_policy']}")
        if Path("reports/foody_intelligence/versions.json").exists():
            st.json(json_file("reports/foody_intelligence/versions.json"))
        if Path("data/foody/vector_store/index_manifest.json").exists():
            st.json(json_file("data/foody/vector_store/index_manifest.json"))
        if Path(active["qwen_provenance"]).exists():
            qwen = json_file(active["qwen_provenance"])
            st.json({"qwen_model": qwen.get("model"), "revision": qwen.get("revision"),
                     "model_sha256": qwen.get("model_sha256")})
        st.json({"rag_v3_prompt": digest("prompts/rag/rag_v3.txt"),
                 "health_v3_prompt": digest("prompts/restaurant_report/health_v3.txt")})
        st.dataframe(pd.DataFrame(summary()), hide_index=True)
    st.subheader("Research metrics · VN-MSASar")
    benchmarks = {
        "Rule-Based": "reports/baselines/rule_based_test_metrics.json",
        "TF-IDF": "reports/baselines/tfidf_test_metrics.json",
        "Qwen V1": "reports/proposed_v1/qwen_test_metrics.json",
        "Qwen V2-A": "reports/proposed_v2/test_metrics.json",
        "Aspect classifier": "reports/aspect_classifier/test_metrics.json",
    }
    rows = []
    for name, path in benchmarks.items():
        if Path(path).exists():
            data = json_file(path)
            rows.append({"model": name, "macro_f1": data.get("macro_f1"),
                         "micro_f1": data.get("micro_f1"), "test_dataset": "VN-MSASar v3"})
    with st.expander("View research benchmarks and Foody inference distribution"):
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if Path("reports/foody_intelligence/enrichment.json").exists():
            enrichment = json_file("reports/foody_intelligence/enrichment.json")
            st.json(enrichment["sentiment_distribution_unique_reviews"])
    st.caption("Research metrics are from VN-MSASar. Foody predictions do not measure Foody accuracy.")


def main():
    st.set_page_config(page_title="Foody Intelligence Platform", layout="wide",
                       page_icon="🍽️", initial_sidebar_state="expanded")
    apply_styles()
    if metrics_port := os.getenv("FOODY_STREAMLIT_METRICS_PORT"):
        start_streamlit_metrics(metrics_port)
    st.sidebar.markdown("### Foody Intelligence")
    st.sidebar.caption("Sales Analytics • Review Intelligence • RAG • LLMOps")
    st.sidebar.caption("ANALYTICS  ·  AI  ·  OPERATIONS")
    page_icons = {"Restaurant Intelligence": "🏠", "Sales Analytics": "💰",
                  "Review Explorer": "💬", "AI Review Analyst": "🤖",
                  "LLMOps Monitor": "📊"}
    page = st.sidebar.radio("Page", list(page_icons),
                            format_func=lambda name: f"{page_icons[name]}  {name}")
    st.sidebar.divider()
    st.sidebar.caption("Runtime · PostgreSQL  |  RAG V3  |  Qwen2.5 3B")
    {"Restaurant Intelligence": restaurant_page,
     "Sales Analytics": sales_page,
     "Review Explorer": explorer_page,
     "AI Review Analyst": analyst_page,
     "LLMOps Monitor": monitor_page}[page]()


if __name__ == "__main__":
    main()
