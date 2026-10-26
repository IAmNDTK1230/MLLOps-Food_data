"""Three-workspace Foody analytics demo with evidence-first RAG V3."""

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
    empty_state,
    evidence_card,
    header,
    text_sentiment_label,
)
from src.ui.sales_view import review_snippets, sales_page
from src.ui.styles import apply_styles

WORKSPACES = {"Sales Intelligence": "💰", "Review Intelligence": "💬",
              "System Monitor": "📊"}


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
    from src.foody.review_runtime import new_corpus_active, review_restaurants

    summary = review_restaurants()
    names = summary.sort_values("restaurant_name").restaurant_name.tolist()
    from src.demo import demo_config

    config = demo_config()
    preferred = config["default_restaurant"]
    if new_corpus_active():
        default_id = str(config.get("default_review_restaurant_id", ""))
        matching = summary.loc[summary.restaurant_id.astype(str).eq(default_id)]
        if not matching.empty:
            preferred = matching.iloc[0].restaurant_name
    chosen = st.selectbox("Restaurant", names,
                          index=names.index(preferred) if preferred in names else 0)
    row = summary.loc[summary.restaurant_name == chosen].iloc[0]
    return (str(row.restaurant_id) if new_corpus_active() else row.restaurant_key), row


@st.cache_data(ttl=300, show_spinner=False)
def active_reviews():
    from src.foody.review_runtime import review_table

    return review_table()


def selected_reviews(key):
    from src.foody.review_runtime import new_corpus_active

    frame = active_reviews()
    column = "restaurant_id" if new_corpus_active() else "restaurant_key"
    return frame.loc[frame[column].astype(str).eq(str(key))].drop_duplicates("review_id")


def date_filter(frame, label, key):
    dated = frame.review_timestamp.dropna()
    if dated.empty:
        return frame
    minimum, maximum = dated.min().date(), dated.max().date()
    start, end = st.date_input(label, (minimum, maximum), min_value=minimum,
                               max_value=maximum, key=key)
    return frame.loc[frame.review_timestamp.dt.date.between(start, end)]


def restaurant_page(key=None, summary=None, *, embedded=False):
    from src.foody.review_runtime import new_corpus_active

    if not embedded:
        header("Review Intelligence", "Understand ratings, sentiment and customer feedback.",
               ("Foody Reviews", "RAG V3"))
    if key is None:
        key, summary = choose_restaurant()
    if key is None:
        return
    st.subheader("Customer Overview")
    frame = selected_reviews(key)
    frame = date_filter(frame, "Review date range", "overview_review_dates")
    if frame.empty:
        empty_state("No reviews for this date range", "Widen the date filter to see insights.")
        return
    predicted = frame.predicted_sentiment.value_counts(normalize=True)
    aspects = frame.predicted_aspects.explode().dropna().value_counts()
    issues = frame.detected_issues.explode().dropna().value_counts()
    issues.index = [str(label).replace("_", " ").title() for label in issues.index]
    top_aspect = str(aspects.index[0]).replace("_", " ").title() if not aspects.empty else "—"
    cards = st.columns(5)
    cards[0].metric("Review Count", f"{len(frame):,}")
    cards[1].metric("Average Rating · /10", f"{pd.to_numeric(frame.raw_rating, errors='coerce').mean():.1f}")
    cards[2].metric("Positive", f"{predicted.get('positive', 0):.1%}")
    cards[3].metric("Negative", f"{predicted.get('negative', 0):.1%}")
    cards[4].metric("Top Aspect", top_aspect)
    trend = frame.loc[frame.review_timestamp.notna()].assign(
        period=lambda x: x.review_timestamp.dt.to_period("M").dt.to_timestamp(),
    ).groupby("period").review_id.count()
    left, right = st.columns(2)
    with left:
        st.subheader("Review activity by month")
        if trend.empty:
            empty_state("No dated reviews in this selection")
        else:
            st.line_chart(trend, color="#4F8EF7", height=240)
    with right:
        st.subheader("Sentiment distribution")
        st.bar_chart(frame.predicted_sentiment.value_counts(), color="#4F8EF7", height=240)
    st.subheader("Customer Signals")
    aspects = aspects.head(6)
    if aspects.empty:
        empty_state("No predicted aspects in this selection")
    else:
        st.bar_chart(aspects, color="#4F8EF7", height=240)
    with st.expander("Method details"):
        st.caption(f"{int(frame.predicted_sentiment.notna().sum()):,} AI-eligible reviews. "
                   "Signals are heuristic, not measured model accuracy.")
        if issues.empty:
            empty_state("No heuristic signals in this selection")
        else:
            st.bar_chart(issues.head(10), color="#F5B942", height=220)
    alert_path = "data/foody/mart/alerts.parquet"
    if not new_corpus_active() and Path(alert_path).exists():
        alerts = table(alert_path)
        with st.expander("Historical rule alerts"):
            selected_alerts = alerts.loc[alerts.restaurant_key == key]
            if selected_alerts.empty:
                empty_state("No historical alerts for this restaurant")
            else:
                st.dataframe(selected_alerts, hide_index=True)
    with st.expander("More review insights"):
        generate_health = st.button("✨ Generate AI Health Summary")
    if generate_health:
        try:
            from src.rag_v3.cli import selected_v3_gateway
            from src.rag_v3.health import health_report_v3
            from src.rag_v3.retriever import cached_retriever

            with st.spinner("Analyzing verified evidence..."):
                with selected_v3_gateway() as gateway:
                    report, telemetry, verification = health_report_v3(
                        cached_retriever(), gateway, key,
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


def explorer_page(key=None, *, embedded=False):
    if not embedded:
        header("Review Intelligence", "Explore source reviews and saved AI predictions.",
               ("Foody Reviews",))
    if key is None:
        key, _ = choose_restaurant()
    if key is None:
        return
    frame = selected_reviews(key)
    st.subheader("Recent Reviews")
    filters = st.columns(3)
    with filters[0]:
        sentiment = st.selectbox("Sentiment", ["All", "positive", "neutral", "negative"])
    with filters[1]:
        rating = st.selectbox("Minimum rating", ["All", 2, 4, 6, 8])
    with filters[2]:
        aspect = st.selectbox("Aspect", ["All"] + sorted(set(
            frame.predicted_aspects.explode().dropna())))
    frame = date_filter(frame, "Review date range", "explorer_review_dates")
    if sentiment != "All":
        frame = frame.loc[frame.predicted_sentiment == sentiment]
    if rating != "All":
        frame = frame.loc[pd.to_numeric(frame.raw_rating, errors="coerce") >= rating]
    if aspect != "All":
        frame = frame.loc[frame.predicted_aspects.map(lambda x: aspect in x)]
    if frame.empty:
        empty_state("No reviews match these filters", "Try a wider date range or fewer filters.")
        return
    st.caption(f"{len(frame):,} reviews match these filters")
    page_size = 6
    page = st.number_input("Page", min_value=1, max_value=max(1, (len(frame) - 1) // page_size + 1),
                           value=1, step=1)
    shown = frame.iloc[(page - 1) * page_size:page * page_size]
    for row in shown.itertuples():
        date = row.review_timestamp.strftime("%d %b %Y") if pd.notna(row.review_timestamp) else "Date unavailable"
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**Rating: {row.raw_rating}/10** · {date}")
            with right:
                badge(text_sentiment_label(row.predicted_sentiment),
                      "good" if row.predicted_sentiment == "positive" else
                      "warn" if row.predicted_sentiment == "negative" else "muted")
            review_text = str(row.text)
            if len(review_text) > 420:
                st.write(review_text[:420].rsplit(" ", 1)[0] + "…")
                with st.expander("Read full review"):
                    st.write(review_text)
            else:
                st.write(review_text)
            if len(row.predicted_aspects):
                st.caption("Aspects: " + ", ".join(row.predicted_aspects))
            with st.expander("Technical details"):
                st.caption(f"Review ID: {row.review_id} · Confidence: {row.sentiment_confidence}")
                st.caption("Sentiment source: model_prediction")
                st.caption("Heuristic signals: " + ", ".join(row.detected_issues))
    st.caption("Sentiment and aspects are model predictions; heuristic signals are not measured accuracy.")


def analyst_page(key=None, *, embedded=False):
    if not embedded:
        header("Review Intelligence", "Ask grounded questions about customer feedback.",
               ("RAG V3", "AI Grounded"))
    if key is None:
        key, _ = choose_restaurant()
    if key is None:
        return
    st.subheader("AI Review")
    st.caption("Ask grounded questions about customer feedback.")
    question = st.text_input("Ask AI Review", key="review_question",
                             placeholder="Ví dụ: Khách hàng khen món nào nhiều nhất?")
    from src.sales.router import route_question

    route = route_question(question)
    if question.strip() and route != "REVIEW_RAG":
        st.info("Sales and combined questions are available in Sales Intelligence.")
        return
    if st.button("✨ Ask AI Review", type="primary", key="ask_review"):
        if not question.strip():
            st.info("Enter a review question to begin.")
            return
        try:
            from src.foody.review_runtime import new_corpus_active
            from src.rag.schemas import RetrievalRequest
            from src.rag_v3.cli import selected_v3_gateway
            from src.rag_v3.retriever import cached_retriever
            from src.rag_v3.service import answer_question_v3

            with st.spinner("Analyzing verified evidence..."):
                with selected_v3_gateway() as gateway:
                    answer, hits, telemetry, debug, packet = answer_question_v3(
                        cached_retriever(), gateway,
                        RetrievalRequest(query=question, top_k=5,
                                         **({"restaurant_id": str(key)} if new_corpus_active() else
                                            {"restaurant_key": key})),
                    )
            observe_rag(telemetry)
            mode = answer.generation_mode
            tone = "good" if mode == "qwen" else "warn" if mode == "extractive_fallback" else "muted"
            state_label = ("Grounded" if mode == "qwen" else
                           "Evidence fallback" if mode == "extractive_fallback" else
                           "Insufficient evidence")
            answer_card("AI Answer", answer.answer if mode != "none" else
                        "Không đủ bằng chứng để kết luận.", state_label, tone)
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
            with st.expander("Technical details"):
                st.write("Review IDs", [item.review_id for item in answer.evidence])
                st.write("Claim mappings", [claim.model_dump() for claim in answer.claims])
                st.write("Evidence ref → review ID", {
                    item["ref"]: item["review_id"] for item in packet.evidence
                })
                st.json({"rag_version": answer.retrieval_version,
                         "prompt_version": answer.prompt_version,
                         "generation_mode": answer.generation_mode,
                         "retrieval_route": answer.query_route,
                         "retrieval_method": telemetry["retrieval_method"],
                         "support_gate_status": packet.support_gate_status,
                         "clause_support_status": telemetry["clause_support_status"],
                         "input_tokens": telemetry.get("input_tokens"),
                         "output_tokens": telemetry.get("output_tokens"),
                         "llm_latency_ms": telemetry.get("llm_latency_ms"),
                         "generation_attempts": telemetry["generation_attempts"],
                         "failure_category": telemetry["failure_category"],
                         "retrieval_debug": debug})
                st.caption(f"{len(answer.evidence)} cited of {len(hits)} retrieved · "
                           f"retrieval {telemetry['retrieval_latency_ms']:.1f} ms")
        except Exception as error:
            st.error("AI analysis is temporarily unavailable.")
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")


def review_workspace():
    from src.foody.review_runtime import active_manifest

    metadata = active_manifest()
    count = metadata.get("row_count", metadata.get("document_count"))
    header("Review Intelligence", "Understand ratings, sentiment, customer feedback and recurring issues.",
           ((f"{count:,} Reviews" if count is not None else "Foody Reviews"),
            "RAG V3", "AI Grounded"))
    key, summary = choose_restaurant()
    if key is None:
        return
    restaurant_page(key, summary, embedded=True)
    explorer_page(key, embedded=True)
    analyst_page(key, embedded=True)


def monitor_page():
    header("System Monitor", "Operational health, AI activity and pipeline observability.",
           ("DevOps", "LLMOps"))
    from src.demo import demo_config
    from src.ui.monitoring_view import prometheus_snapshot

    active = demo_config()
    metrics = prometheus_snapshot()
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
    sales = events.loc[events.component.eq("sales_sql")].copy() if not events.empty else pd.DataFrame()
    cards = st.columns(6)
    total = len(rag)
    latency = pd.to_numeric(rag.llm_latency_ms, errors="coerce") if total else pd.Series(dtype=float)
    target_up, target_total = metrics["targets_up"], metrics["targets_total"]
    state = ("Healthy" if target_total and target_up == target_total else
             "Partial" if target_up is not None else "Unavailable")
    cards[0].metric("System Status", state)
    cards[1].metric("RAG Requests", f"{total:,}")
    cards[2].metric("Sales Queries", f"{len(sales):,}")
    cards[3].metric("Average AI Latency", f"{latency.mean() / 1000:.1f} s"
                    if latency.notna().any() else "—")
    token_total = (pd.to_numeric(rag.input_tokens, errors="coerce").sum() +
                   pd.to_numeric(rag.output_tokens, errors="coerce").sum()) if total else 0
    cards[4].metric("Token Usage", f"{token_total:,.0f}")
    cards[5].metric("Kafka Events", f"{metrics['kafka_consumed']:,.0f}"
                    if metrics["kafka_consumed"] is not None else "—")
    st.subheader("AI Operations")
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
        with st.expander("Recent AI requests"):
            st.dataframe(rag[["timestamp", "status", "generation_mode", "query_route",
                              "llm_latency_ms", "input_tokens", "output_tokens"]].head(20),
                         hide_index=True, width="stretch")
    else:
        empty_state("No RAG requests recorded yet", "Ask a question in Review Intelligence to begin.")
    st.subheader("Data Platform")
    left, right = st.columns(2)
    with left:
        st.caption(f"Sales SQL requests: {len(sales):,}")
        if not sales.empty:
            st.bar_chart(sales.status.value_counts(), height=210)
        else:
            empty_state("No Sales questions recorded yet")
    with right:
        consumed, failed = metrics["kafka_consumed"], metrics["kafka_failed"]
        st.caption(f"Kafka consumed: {consumed:,.0f} · failures: {failed:,.0f}"
                   if consumed is not None and failed is not None else
                   "Kafka metrics are unavailable")
        if not sales.empty:
            st.caption(f"Average Sales SQL latency: {pd.to_numeric(sales.latency_ms, errors='coerce').mean():.0f} ms")
    st.subheader("System Health")
    error_count, api_count = metrics["api_errors"], metrics["api_requests"]
    h1, h2, h3 = st.columns(3)
    h1.metric("API Error Rate", f"{error_count / api_count:.1%}"
              if api_count and error_count is not None else "—")
    h2.metric("API Latency", f"{metrics['api_latency_seconds'] * 1000:.0f} ms"
              if metrics["api_latency_seconds"] is not None else "—")
    h3.metric("Services Reporting", f"{target_up:.0f}/{target_total:.0f}"
              if target_up is not None and target_total is not None else "—")
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
    st.sidebar.markdown("### FOODY INTELLIGENCE")
    st.sidebar.caption("Restaurant analytics powered by verified data + grounded AI")
    st.sidebar.caption("NAVIGATION")
    page = st.sidebar.radio("Workspace", list(WORKSPACES),
                            format_func=lambda name: f"{WORKSPACES[name]}  {name}")
    st.sidebar.divider()
    st.sidebar.caption("PostgreSQL ●   RAG V3 ●   Qwen2.5 3B ●")
    {"Sales Intelligence": sales_page,
     "Review Intelligence": review_workspace,
     "System Monitor": monitor_page}[page]()


if __name__ == "__main__":
    main()
