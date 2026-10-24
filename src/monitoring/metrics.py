"""Shared Prometheus metrics for API and Streamlit RAG V3 requests."""

from threading import Lock

from prometheus_client import Counter, Histogram, start_http_server

API_REQUESTS = Counter("foody_api_requests_total", "HTTP requests", ["route", "status"])
API_DURATION = Histogram("foody_api_request_duration_seconds", "HTTP latency", ["route"])
RAG_REQUESTS = Counter("foody_rag_requests_total", "RAG V3 requests", ["status"])
GENERATION = Counter("foody_rag_generation_mode_total", "RAG generation modes",
                     ["generation_mode"])
RETRIEVAL = Counter("foody_rag_retrieval_route_total", "RAG retrieval routes", ["route"])
EVIDENCE = Histogram("foody_rag_evidence_count", "Cited review count")
LLM_LATENCY = Histogram(
    "foody_llm_latency_seconds", "LLM generation latency",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, float("inf")),
)
INPUT_TOKENS = Counter("foody_llm_input_tokens_total", "LLM input tokens")
OUTPUT_TOKENS = Counter("foody_llm_output_tokens_total", "LLM output tokens")
SALES_QUERIES = Counter("foody_sales_queries_total", "SQL sales questions", ["status"])
CROSS_REQUESTS = Counter("foody_cross_analysis_requests_total", "Cross-analysis questions", ["status"])
SALES_DURATION = Histogram("foody_sales_sql_duration_seconds", "SQL sales question latency")

_SERVER_LOCK = Lock()
_SERVER_STARTED = False
_STATUSES = {"grounded", "supported_extractive_fallback", "correct_insufficient",
             "insufficient_evidence", "error"}
_MODES = {"qwen", "extractive_fallback", "none"}
_ROUTES = {"TREND_CAUSATION", "SERVICE_COMPLAINT", "FOOD_COMPLAINT",
           "PRICE_COMPLAINT", "GENERAL_ASPECT", "GENERAL_SUMMARY"}


def start_streamlit_metrics(port):
    """Expose Streamlit's own process metrics once across script reruns/sessions."""
    global _SERVER_STARTED
    with _SERVER_LOCK:
        if not _SERVER_STARTED:
            start_http_server(int(port), addr="0.0.0.0")
            _SERVER_STARTED = True


def observe_rag_error():
    RAG_REQUESTS.labels(status="error").inc()


def observe_rag(telemetry):
    # These labels are internally generated bounded enums, never raw query/user IDs.
    status = telemetry.get("status")
    mode = telemetry.get("generation_mode")
    route = telemetry.get("query_route")
    status = status if status in _STATUSES else "other"
    mode = mode if mode in _MODES else "other"
    route = route if route in _ROUTES else "other"
    RAG_REQUESTS.labels(status=status).inc()
    GENERATION.labels(generation_mode=mode).inc()
    RETRIEVAL.labels(route=route).inc()
    EVIDENCE.observe(telemetry.get("evidence_count") or 0)
    if telemetry.get("llm_latency_ms") is not None:
        LLM_LATENCY.observe(telemetry["llm_latency_ms"] / 1000)
    INPUT_TOKENS.inc(telemetry.get("input_tokens") or 0)
    OUTPUT_TOKENS.inc(telemetry.get("output_tokens") or 0)


def observe_sales_sql(duration_seconds, status):
    """Record one SQL answer with bounded status labels."""
    SALES_QUERIES.labels(status=status if status in {"ok", "no_data", "error"} else "other").inc()
    SALES_DURATION.observe(duration_seconds)


def observe_cross_analysis(status):
    CROSS_REQUESTS.labels(status=status if status in {
        "grounded", "no_aligned_reviews", "unmatched_restaurant", "no_sales_data",
        "insufficient_review_evidence", "error",
    } else "other").inc()
