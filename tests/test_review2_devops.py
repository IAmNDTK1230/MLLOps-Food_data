"""Fast contract checks; no Docker, model weights, or live data needed in CI."""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import api
from src.monitoring import metrics as rag_metrics
from src.streaming.events import ReviewEvent, from_review
from src.warehouse.loader import _date_key, _number


def test_review_event_is_stable_and_versioned():
    row = SimpleNamespace(
        review_id="a" * 64, restaurant_key="b" * 64, restaurant_name="Quán thử",
        review_timestamp=datetime(2024, 2, 3), raw_rating="8.5", text="Món ăn ngon",
    )
    first = from_review(row)
    assert first.event_id == from_review(row).event_id
    assert first.rating == 8.5
    assert ReviewEvent.model_validate_json(first.model_dump_json()) == first
    with pytest.raises(ValidationError):
        ReviewEvent.model_validate_json(first.model_dump_json().replace('"1.0"', '"2.0"'))


def test_warehouse_mapping_uses_existing_calendar_keys():
    assert _date_key(datetime(2024, 2, 3)) == 20240203
    assert _number("8.5") == 8.5
    assert _number("bad") is None
    assert _number(float("nan")) is None


def test_api_exposes_bounded_metrics_and_versions():
    client = TestClient(api)
    assert client.get("/health").status_code == 200
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "foody_api_requests_total" in response.text
    assert "foody_api_request_duration_seconds" in response.text
    version = client.get("/version")
    assert version.status_code == 200
    assert version.json()["warehouse_schema_version"] == "004"
    assert "qwen_model" in version.json()


def test_devops_configs_have_one_topic_and_dashboard():
    dashboard = json.loads(Path("devops/grafana/dashboards/foody-devops.json").read_text())
    assert dashboard["title"] == "Foody DevOps & LLMOps"
    assert len(dashboard["panels"]) == 13
    assert [panel["id"] for panel in dashboard["panels"][:10]] == list(range(1, 11))
    assert {panel["title"] for panel in dashboard["panels"][10:]} == {
        "Sales SQL requests", "Cross-analysis requests", "Sales SQL latency P95",
    }
    compose = Path("docker-compose.devops.yml").read_text()
    assert "foody.review.events" not in compose  # topic is created explicitly by startup
    assert "apache/kafka:3.9.1" in compose
    assert "foody_devops_network" in compose
    assert Path("warehouse/migrations/001_init.sql").exists()
    prometheus = Path("devops/prometheus/prometheus.yml").read_text()
    assert 'targets: ["streamlit:8002"]' in prometheus
    assert 'FOODY_STREAMLIT_METRICS_PORT: "8002"' in compose


def test_shared_rag_metrics_count_once_and_bound_labels():
    before = rag_metrics.RAG_REQUESTS.labels(status="grounded")._value.get()
    rag_metrics.observe_rag({
        "status": "grounded", "generation_mode": "qwen",
        "query_route": "SERVICE_COMPLAINT", "evidence_count": 2,
        "llm_latency_ms": 300, "input_tokens": 12, "output_tokens": 5,
    })
    assert rag_metrics.RAG_REQUESTS.labels(status="grounded")._value.get() == before + 1
    assert rag_metrics.GENERATION.labels(generation_mode="qwen")._value.get() >= 1
    assert rag_metrics.RETRIEVAL.labels(route="SERVICE_COMPLAINT")._value.get() >= 1

    rag_metrics.observe_rag({
        "status": "raw user query", "generation_mode": "restaurant name",
        "query_route": "review_id", "evidence_count": 0,
    })
    assert rag_metrics.RAG_REQUESTS.labels(status="other")._value.get() >= 1
    assert rag_metrics.GENERATION.labels(generation_mode="other")._value.get() >= 1
    assert rag_metrics.RETRIEVAL.labels(route="other")._value.get() >= 1
    assert "raw user query" not in rag_metrics.RAG_REQUESTS.collect()[0].samples[0].labels.values()


def test_streamlit_metrics_server_starts_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(rag_metrics, "_SERVER_STARTED", False)
    monkeypatch.setattr(rag_metrics, "start_http_server", lambda *args, **kwargs:
                        calls.append((args, kwargs)))
    rag_metrics.start_streamlit_metrics("8002")
    rag_metrics.start_streamlit_metrics("8002")
    assert calls == [((8002,), {"addr": "0.0.0.0"})]
