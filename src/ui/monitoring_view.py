"""Small read-only Prometheus snapshot for the Streamlit System Monitor."""

import math
import os

import requests
import streamlit as st

QUERIES = {
    "kafka_consumed": "sum(foody_kafka_events_consumed_total)",
    "kafka_failed": "sum(foody_kafka_events_failed_total)",
    "api_errors": 'sum(foody_api_requests_total{status=~"5.."})',
    "api_requests": "sum(foody_api_requests_total)",
    "api_latency_seconds": (
        "sum(rate(foody_api_request_duration_seconds_sum[5m])) / "
        "sum(rate(foody_api_request_duration_seconds_count[5m]))"
    ),
    "targets_up": 'sum(up{job=~"foody-.*"})',
    "targets_total": 'count(up{job=~"foody-.*"})',
}


def _value(payload):
    rows = payload.get("data", {}).get("result", [])
    if payload.get("status") != "success" or not rows:
        return None
    value = float(rows[0]["value"][1])
    return value if math.isfinite(value) else None


@st.cache_data(ttl=15, show_spinner=False)
def prometheus_snapshot():
    candidates = [os.getenv("FOODY_PROMETHEUS_URL", "http://prometheus:9090")]
    if "FOODY_PROMETHEUS_URL" not in os.environ:
        candidates.append("http://127.0.0.1:9090")
    for base in candidates:
        try:
            with requests.Session() as session:
                probe = session.get(f"{base}/api/v1/query", params={"query": "up"},
                                    timeout=0.8)
                probe.raise_for_status()
                values = {}
                for name, query in QUERIES.items():
                    response = session.get(f"{base}/api/v1/query",
                                           params={"query": query}, timeout=0.8)
                    response.raise_for_status()
                    values[name] = _value(response.json())
                return values
        except (requests.RequestException, KeyError, TypeError, ValueError):
            continue
    return {name: None for name in QUERIES}
