"""Four-page Streamlit demo with lazy, visible RAG/Qwen degradation."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.monitoring.metrics import observe_rag, start_streamlit_metrics


@st.cache_data(show_spinner=False)
def table(path):
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def available(path):
    if not Path(path).exists():
        st.warning(f"Data unavailable: {path}. Run the Foody intelligence pipeline first.")
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
    st.caption("Restaurant key is derived from its name; it is not an official Foody ID.")
    return row.restaurant_key, row


def date_filter(frame, label):
    dated = frame.review_timestamp.dropna()
    if dated.empty:
        return frame
    minimum, maximum = dated.min().date(), dated.max().date()
    start, end = st.date_input(label, (minimum, maximum), min_value=minimum, max_value=maximum)
    return frame.loc[frame.review_timestamp.dt.date.between(start, end)]


def restaurant_page():
    st.title("Restaurant Intelligence")
    key, summary = choose_restaurant()
    if key is None or not available("data/foody/enriched/reviews.parquet"):
        return
    frame = table("data/foody/enriched/reviews.parquet")
    frame = frame.loc[frame.restaurant_key == key].drop_duplicates("review_id")
    frame = date_filter(frame, "Review date range")
    st.subheader(summary.restaurant_name)
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", len(frame))
    c2.metric("Average raw rating (0–10)", f"{pd.to_numeric(frame.raw_rating, errors='coerce').mean():.2f}")
    c3.metric("AI-eligible reviews", int(frame.valid_for_ai.sum()))
    predicted = frame.predicted_sentiment.value_counts(normalize=True)
    rates = st.columns(3)
    for column, label in zip(rates, ("positive", "neutral", "negative"), strict=True):
        column.metric(f"Predicted {label}", f"{predicted.get(label, 0):.1%}")
    st.subheader("Predicted sentiment")
    st.bar_chart(frame.predicted_sentiment.value_counts())
    trend = frame.loc[frame.review_timestamp.notna()].assign(
        period=lambda x: x.review_timestamp.dt.to_period("M").dt.to_timestamp(),
        negative=lambda x: x.predicted_sentiment.eq("negative"),
    ).groupby("period").negative.mean()
    st.subheader("Predicted negative share over time")
    st.line_chart(trend)
    st.subheader("Predicted aspects")
    st.bar_chart(frame.predicted_aspects.explode().value_counts())
    st.subheader("Heuristic issues")
    st.bar_chart(frame.detected_issues.explode().value_counts())
    alert_path = "data/foody/mart/alerts.parquet"
    if available(alert_path):
        alerts = table(alert_path)
        st.subheader("Historical rule alerts")
        st.dataframe(alerts.loc[alerts.restaurant_key == key], hide_index=True)
    if st.button("Generate Qwen health report"):
        try:
            from src.rag_v3.cli import selected_v3_gateway
            from src.rag_v3.health import health_report_v3
            from src.rag_v3.retriever import RoutedRetriever

            with st.spinner("Generating from review and mart evidence..."):
                with selected_v3_gateway() as gateway:
                    report, telemetry, verification = health_report_v3(
                        RoutedRetriever(), gateway, key,
                    )
            st.json(report.model_dump())
            st.caption("RAG V3 section modes: " + ", ".join(
                f"{name}={mode}" for name, mode in telemetry["section_modes"].items()))
            st.write("Evidence by section", verification["evidence_by_section"])
        except Exception as error:
            st.warning(f"Health report unavailable ({type(error).__name__}). Marts remain available.")
    st.caption("Foody AI predictions have no measured Foody accuracy; issues are heuristic.")


def explorer_page():
    st.title("Review Explorer")
    key, _ = choose_restaurant()
    if key is None or not available("data/foody/enriched/reviews.parquet"):
        return
    frame = table("data/foody/enriched/reviews.parquet")
    frame = frame.loc[frame.restaurant_key == key].drop_duplicates("review_id")
    sentiment = st.selectbox("Predicted sentiment", ["All", "positive", "neutral", "negative"])
    aspect = st.selectbox("Predicted aspect", ["All"] + sorted(set(
        frame.predicted_aspects.explode().dropna())))
    issue = st.selectbox("Heuristic issue", ["All"] + sorted(set(
        frame.detected_issues.explode().dropna())))
    frame = date_filter(frame, "Review date range")
    if sentiment != "All":
        frame = frame.loc[frame.predicted_sentiment == sentiment]
    if aspect != "All":
        frame = frame.loc[frame.predicted_aspects.map(lambda x: aspect in x)]
    if issue != "All":
        frame = frame.loc[frame.detected_issues.map(lambda x: issue in x)]
    st.caption(f"{len(frame)} distinct reviews")
    shown = frame[["review_id", "text", "raw_rating", "review_timestamp",
                   "predicted_sentiment", "sentiment_confidence",
                   "predicted_aspects", "detected_issues"]].head(500)
    source, ai = st.columns(2)
    with source:
        st.subheader("SOURCE · review, rating, date")
        st.dataframe(shown[["review_id", "text", "raw_rating", "review_timestamp"]],
                     hide_index=True)
    with ai:
        st.subheader("AI · predictions, confidence, aspects, issues")
        st.dataframe(shown[["review_id", "predicted_sentiment", "sentiment_confidence",
                            "predicted_aspects", "detected_issues"]], hide_index=True)


def analyst_page():
    st.title("AI Review Analyst")
    key, _ = choose_restaurant()
    if key is None:
        return
    st.caption("Demo questions: Khách nói gì về chất lượng món ăn? · "
               "Nhân viên phục vụ có vấn đề gì không? · "
               "Khách có phàn nàn về giá không? · "
               "Tại sao đánh giá của nhà hàng này giảm?")
    question = st.text_input("Question", "Nhân viên phục vụ có vấn đề gì không?")
    if st.button("Ask"):
        try:
            from src.rag.schemas import RetrievalRequest
            from src.rag_v3.cli import selected_v3_gateway
            from src.rag_v3.retriever import RoutedRetriever
            from src.rag_v3.service import answer_question_v3, generation_mode_label

            with st.spinner("Retrieving reviews and generating an evidence-linked answer..."):
                with selected_v3_gateway() as gateway:
                    answer, hits, telemetry, debug, packet = answer_question_v3(
                        RoutedRetriever(), gateway,
                        RetrievalRequest(query=question, restaurant_key=key, top_k=5),
                    )
            observe_rag(telemetry)
            st.subheader("AI Analysis")
            st.write(answer.answer if answer.generation_mode != "none" else
                     "Không đủ bằng chứng để kết luận.")
            st.caption(f"Generation mode: {generation_mode_label(answer.generation_mode)}"
                       f" · Retrieval: {telemetry['retrieval_method']}"
                       f" · Model: {answer.model}")
            st.caption("Analysis is limited to the retrieved reviews; it does not "
                       "represent every customer.")
            st.subheader("Evidence used")
            cited_ids = {item.review_id for item in answer.evidence}
            for item in packet.evidence:
                if item["review_id"] not in cited_ids:
                    continue
                st.markdown(f"**Evidence {item['ref']}**")
                st.caption(f"Sentiment: {item['sentiment'] or 'N/A'} · "
                           f"Aspects: {', '.join(item['aspects']) or 'N/A'} · "
                           f"Rating: {item['rating'] or 'N/A'}")
                st.write(item["text"])
            if not cited_ids:
                st.info("No review evidence was accepted for this question.")
            with st.expander("Technical details"):
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
            st.warning(f"RAG unavailable ({type(error).__name__}). Build the index or start Qwen.")


def monitor_page():
    st.title("LLMOps Monitor")
    from src.demo import demo_config

    active = demo_config()
    st.subheader("PRODUCTION / DEMO MONITORING")
    st.caption(f"Active {active['rag_version']} · {active['qwen_model']} · "
               f"{active['retrieval_policy']}")
    st.subheader("Active data and model versions")
    if available("reports/foody_intelligence/versions.json"):
        st.json(json_file("reports/foody_intelligence/versions.json"))
    if Path("data/foody/vector_store/index_manifest.json").exists():
        st.json(json_file("data/foody/vector_store/index_manifest.json"))
    else:
        st.info("Vector index unavailable. Run rag-build-index.")
    if Path(active["qwen_provenance"]).exists():
        qwen = json_file(active["qwen_provenance"])
        st.json({"qwen_model": qwen.get("model"), "revision": qwen.get("revision"),
                 "model_sha256": qwen.get("model_sha256"),
                 "runtime_tag": qwen.get("runtime_tag")})
    from src.foody.schema import digest

    st.json({"rag_v1_prompt": digest("prompts/rag/rag_v1.txt"),
             "health_v1_prompt": digest("prompts/restaurant_report/health_v1.txt"),
             "rag_v2_prompt": digest("prompts/rag/rag_v2.txt"),
             "health_v2_prompt": digest("prompts/restaurant_report/health_v2.txt"),
             "rag_v3_prompt": digest("prompts/rag/rag_v3.txt"),
             "health_v3_prompt": digest("prompts/restaurant_report/health_v3.txt")})
    from src.rag.monitoring import summary

    st.subheader("Operational requests (no raw query/review text logged)")
    events = pd.DataFrame(summary())
    st.dataframe(events, hide_index=True)
    if not events.empty:
        generation = events.loc[events.component.isin(["rag_generation", "health_generation"])]
        if not generation.empty:
            total = generation.request_count.sum()
            valid = generation.loc[generation.status == "valid_json", "request_count"].sum()
            st.metric("LLM JSON validity (requests)", f"{valid / total:.1%}")
        rag = events.loc[events.component.isin(
            ["rag_request", "rag_v2_request", "rag_v3_request"]
        )]
        if not rag.empty:
            st.metric("RAG request count", int(rag.request_count.sum()))
    st.subheader("RESEARCH METRICS · VN-MSASar benchmarks")
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
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.subheader("Foody inference distribution")
    if Path("reports/foody_intelligence/enrichment.json").exists():
        enrichment = json_file("reports/foody_intelligence/enrichment.json")
        st.json(enrichment["sentiment_distribution_unique_reviews"])
    st.caption("Research metrics are from VN-MSASar. Foody predictions do not measure Foody accuracy.")


def main():
    st.set_page_config(page_title="Food Review Intelligence", layout="wide")
    if metrics_port := os.getenv("FOODY_STREAMLIT_METRICS_PORT"):
        start_streamlit_metrics(metrics_port)
    page = st.sidebar.radio("Page", ["Restaurant Intelligence", "Review Explorer",
                                     "AI Review Analyst", "LLMOps Monitor"])
    {"Restaurant Intelligence": restaurant_page,
     "Review Explorer": explorer_page,
     "AI Review Analyst": analyst_page,
     "LLMOps Monitor": monitor_page}[page]()


if __name__ == "__main__":
    main()
