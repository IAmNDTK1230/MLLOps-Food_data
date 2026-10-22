import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.llm.gateway import LLMGateway
from src.rag.cli import portable_artifact_path
from src.rag.embeddings import EmbeddingSpec
from src.rag.evaluation import known_positive_scores
from src.rag.monitoring import record, summary
from src.rag.retriever import Retriever, where_filter
from src.rag.schemas import GroundedAnswer, HealthReport, RetrievalRequest, ReviewHit
from src.rag.service import INSUFFICIENT, answer_question, check_refs
from src.rag.vector_store import build_index, client


class FakeEmbeddings:
    spec = EmbeddingSpec("mock-multilingual", "mock", 2)
    device = "CPU/test"

    def documents(self, texts):
        return np.array([[1., 0.] if "phục vụ" in text else [0., 1.] for text in texts]), 0.01

    def queries(self, texts):
        return self.documents(texts)


def test_saved_windows_artifact_path_resolves_portably(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "models" / "embeddings" / "model.onnx"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"frozen-model")
    assert portable_artifact_path(r"models\embeddings\model.onnx").read_bytes() == b"frozen-model"


def make_documents(path):
    rows = [
        ("a", "r1", "phục vụ chậm", "negative", ["service"], "2024-01-01T10:00:00"),
        ("b", "r2", "món ăn ngon", "positive", ["food"], "2024-02-01T10:00:00"),
        ("c", "r1", "phục vụ tốt", "positive", ["service"], "2024-03-01T10:00:00"),
    ]
    with path.open("w", encoding="utf-8") as stream:
        for review_id, key, text, sentiment, aspects, timestamp in rows:
            item = {"id": review_id, "text": text, "metadata": {
                "review_id": review_id, "restaurant_key": key, "restaurant_name": key,
                "review_timestamp": timestamp, "raw_rating": "7.5",
                "predicted_sentiment": sentiment, "predicted_aspects": aspects,
                "detected_issues": [], "source": "foody",
            }}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def test_vector_persistence_filters_and_retrieval(tmp_path):
    docs, store = tmp_path / "docs.jsonl", tmp_path / "chroma"
    make_documents(docs)
    embedder = FakeEmbeddings()
    metadata = build_index(embedder, "model-hash", "fake-revision", 0.1,
                           documents=docs, store=store, manifest=store / "manifest.json", batch_size=2)
    assert metadata["document_count"] == 3
    assert build_index(embedder, "model-hash", "fake-revision", 0.1,
                       documents=docs, store=store, manifest=store / "manifest.json") == metadata
    assert client(store).get_collection(metadata["collection"]).count() == 3
    retriever = Retriever(embedder, client(store).get_collection(metadata["collection"]),
                          metadata, monitor=False)
    hits, latency = retriever.search(RetrievalRequest(query="phục vụ", restaurant_key="r1",
                                                   sentiment="negative", aspect="service",
                                                   date_from="2024-01-01", date_to="2024-01-31"))
    assert [hit.review_id for hit in hits] == ["a"]
    assert hits[0].metadata["raw_rating"] == "7.5"
    assert isinstance(hits[0].score, float) and latency >= 0
    missing, _ = retriever.search(RetrievalRequest(query="phục vụ", restaurant_key="unknown"))
    assert missing == []
    assert where_filter(RetrievalRequest(query="test")) is None
    with pytest.raises(ValueError, match="manifest differs"):
        build_index(embedder, "different", "fake", 0.1,
                    documents=docs, store=store, manifest=store / "manifest.json")


def test_document_validation_excludes_invalid(tmp_path):
    from src.rag.vector_store import read_documents

    docs = tmp_path / "bad.jsonl"
    docs.write_text(json.dumps({"id": "a", "text": "bad �", "metadata": {
        "review_id": "a", "source": "foody"}}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        list(read_documents(docs))


def test_retrieval_contract_and_known_positive_scope():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="", top_k=0)
    with pytest.raises(ValidationError):
        RetrievalRequest(query="x", date_from="2024-02-01", date_to="2024-01-01")
    result = known_positive_scores(["a", "b"], ["b", "c"])
    assert result["known_positive_recall_at_5"] == 0.5
    assert result["known_positive_mrr"] == 0.5


def fake_hit():
    return ReviewHit(review_id="a", restaurant_name="Quán A", text="phục vụ chậm",
                     score=0.9, metadata={"restaurant_key": "r1"})


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, request):
        return self.hits, 2.0


class FakeGateway:
    name = "Qwen-test"
    version = "model-hash"

    def __init__(self, draft):
        self.draft = draft

    def generate(self, **kwargs):
        return self.draft, {"input_tokens": 20, "output_tokens": 8, "latency_ms": 5.0}


def test_grounded_answer_and_empty_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr("src.rag.service.record", lambda *args, **kwargs: None)
    answer, hits, timing = answer_question(
        FakeRetriever([fake_hit()]), FakeGateway({"answer": "Khách nói phục vụ chậm [R1].",
                                                  "evidence_refs": ["R1"]}),
        RetrievalRequest(query="Dịch vụ thế nào?"),
    )
    assert answer.evidence[0].review_id == "a"
    assert timing["status"] == "grounded"
    assert hits[0].score == 0.9
    empty, _, telemetry = answer_question(FakeRetriever([]), FakeGateway({}),
                                           RetrievalRequest(query="x"))
    assert empty.answer == INSUFFICIENT and empty.evidence == []
    assert telemetry["status"] == "insufficient_retrieval"
    bad, _, telemetry = answer_question(
        FakeRetriever([fake_hit()]), FakeGateway({"answer": "Không có dẫn chứng",
                                                  "evidence_refs": ["R1"]}),
        RetrievalRequest(query="x"),
    )
    assert bad.answer == INSUFFICIENT
    assert telemetry["status"] == "invalid_generation_or_evidence"
    with pytest.raises(ValueError, match="unknown"):
        check_refs(["R2"], "[R2]", [fake_hit()])
    with pytest.raises(ValidationError):
        GroundedAnswer(answer="x", evidence=[], model="x", prompt_version="x", extra="x")


def test_health_schema():
    report = HealthReport(overall_summary="Khách khen món [R1]", strengths=["Món ngon [R1]"],
                          main_complaints=[], aspect_trends=[], warnings=[],
                          recommendations=["Theo dõi thêm [M1]"], evidence_review_ids=["a"])
    assert report.evidence_review_ids == ["a"]
    with pytest.raises(ValidationError):
        HealthReport(overall_summary="x", strengths=[], main_complaints=[], aspect_trends=[],
                     warnings=[], recommendations=[], evidence_review_ids=[], extra="x")


def test_health_report_uses_marts_and_review_ids(tmp_path, monkeypatch):
    from src.foody.schema import restaurant_key
    from src.rag.service import health_report

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "data/foody/mart"
    folder.mkdir(parents=True)
    key = restaurant_key("Quán A")
    pd.DataFrame([{"restaurant_key": key, "restaurant_name": "Quán A", "review_count": 10,
                   "average_rating": 7.5, "predicted_negative_rate": 0.2,
                   "ai_review_count": 10}]).to_parquet(folder / "restaurant_summary.parquet")
    pd.DataFrame([{"restaurant_key": key, "period": "2024-01", "review_count": 10,
                   "average_rating": 7.5,
                   "predicted_negative_rate": 0.2}]).to_parquet(folder / "restaurant_monthly_metrics.parquet")
    pd.DataFrame([], columns=["restaurant_key", "current_period", "metric",
                              "current_value", "previous_value"]).to_parquet(folder / "alerts.parquet")
    prompt_dir = tmp_path / "prompts/restaurant_report"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "health_v1.txt").write_text("Grounded report", encoding="utf-8")
    draft = {"answer": "Một khách phàn nàn phục vụ [R1].", "evidence_refs": ["R1"]}
    monkeypatch.setattr("src.rag.service.record", lambda *args, **kwargs: None)
    result, _, timing = health_report(FakeRetriever([fake_hit()]), FakeGateway(draft), key)
    assert result.evidence_review_ids == ["a"]
    assert timing["input_tokens"] == 40


def test_gateway_json_contract(monkeypatch):
    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"answer":"Đủ [R1]","evidence_refs":["R1"]}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 6}}

    class Session:
        def post(self, *args, **kwargs):
            assert kwargs["json"]["temperature"] == 0
            return Reply()

    gateway = LLMGateway.__new__(LLMGateway)
    gateway.url = "http://127.0.0.1:9999"
    gateway.provenance = {"model": "Qwen-test", "model_sha256": "hash"}
    gateway.session = Session()
    gateway.monitor = False
    value, usage = gateway.generate(system="system", user="question", response_schema={
        "type": "object"}, prompt_version="v1")
    assert value["answer"] == "Đủ [R1]"
    assert usage["input_tokens"] == 10 and usage["output_tokens"] == 6


def test_monitor_logs_no_raw_text(tmp_path):
    path = tmp_path / "telemetry.sqlite"
    record("retrieval", "embedder", "hash", "v1", 12, "ok", top_k=5,
           filters={"restaurant_key": "key", "question": "private question"}, path=path)
    assert summary(path)[0]["request_count"] == 1
    with sqlite3.connect(path) as connection:
        value = connection.execute("SELECT filters_json FROM events").fetchone()[0]
    assert "private question" not in value
    assert json.loads(value) == {"restaurant_key": "key"}


def test_streamlit_and_api_import_smoke():
    import app.app as streamlit_app
    from app.api import api

    assert callable(streamlit_app.main)
    response = TestClient(api).get("/health")
    assert response.status_code == 200
    assert "vector_index_available" in response.json()
