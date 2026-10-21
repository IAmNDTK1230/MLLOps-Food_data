"""Safety and fallback regressions for evidence-first RAG V3."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from scripts.evaluate_rag_v3_generation import load_frozen_packets
from src.rag.monitoring import record
from src.rag.schemas import RetrievalRequest, ReviewHit
from src.rag.service import INSUFFICIENT
from src.rag_v3.evidence import build_packet, validate_packet_citations
from src.rag_v3.health import health_report_v3
from src.rag_v3.routing import route_query
from src.rag_v3.service import V3Draft, answer_question_v3, attach_citation


def hit(review_id, text, sentiment="negative", aspects=("service",)):
    return ReviewHit(review_id=review_id, restaurant_name="Quán A", text=text,
                     score=.8, metadata={
                         "review_id": review_id, "restaurant_key": "key-a",
                         "predicted_sentiment": sentiment,
                         "predicted_aspects": json.dumps(aspects),
                         "detected_issues": "[]", "raw_rating": "5.0",
                     })


class FakeRetriever:
    metadata = {"model_name": "MiniLM"}

    def __init__(self, hits):
        self.hits = hits

    def search_debug(self, request):
        found = self.hits(request) if callable(self.hits) else self.hits
        return found, 2.0, {"retrieval_method": "FAKE",
                            "effective_filters": {}, "candidate_count": len(found)}


class FakeGateway:
    name = "Qwen-test"
    version = "hash"

    def __init__(self, output=None):
        self.output = output
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.output is None:
            raise ValueError("invalid model output")
        return self.output, {"latency_ms": 1.0, "input_tokens": 12,
                             "output_tokens": 10}


@pytest.fixture(autouse=True)
def no_operational_log(monkeypatch):
    monkeypatch.setattr("src.rag_v3.service.record", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.rag_v3.health.record", lambda *args, **kwargs: None)


def test_query_routing():
    assert route_query("Nhân viên phục vụ có vấn đề không?").category == \
        "SERVICE_COMPLAINT"
    assert route_query("Khách có phàn nàn về giá không?").category == "PRICE_COMPLAINT"
    assert route_query("Đồ ăn không ngon?").category == "FOOD_COMPLAINT"
    assert route_query("Khách nói gì về món ăn?").category == "GENERAL_ASPECT"
    assert route_query("Khách nói gì về quán?").category == "GENERAL_SUMMARY"
    assert route_query("Tại sao đánh giá giảm?").category == "TREND_CAUSATION"


def test_packet_deduplicates_and_does_not_grant_frequency():
    request = RetrievalRequest(query="Khách có phàn nàn về phục vụ không?")
    review = hit("a", "Nhân viên phục vụ chậm, thái độ tệ")
    packet = build_packet(request, [review, hit("b", review.text)],
                          route_query(request.query))
    assert packet.support_gate_status == "ACCEPTED"
    assert packet.evidence[0]["ref"] == "E1"
    assert packet.evidence[0]["review_id"] == "a"
    assert packet.aggregates == {"accepted_review_count": 1,
                                 "mixed_sentiment": False,
                                 "frequency_supported": False}


def test_late_complaint_is_preserved_verbatim_in_packet():
    opening = "Quán nằm ở mặt tiền đường. " * 12
    review = hit("late", opening + "Món ăn quá dở, không hợp khẩu vị.",
                 "negative", ("food",))
    request = RetrievalRequest(query="Khách phàn nàn điều gì?")
    packet = build_packet(request, [review], route_query(request.query))
    assert "Món ăn quá dở" in packet.evidence[0]["text"]
    assert packet.evidence[0]["text"] in review.text


def test_programmatic_citation_maps_to_exact_review_and_checks_clauses():
    request = RetrievalRequest(query="Nhân viên phục vụ có vấn đề không?")
    first = hit("a", "Nhân viên phục vụ chậm và thái độ tệ")
    second = hit("b", "Nhân viên phục vụ tệ, không hỗ trợ xếp chỗ ngồi linh động")
    packet = build_packet(request, [first, second], route_query(request.query))
    answer, refs = attach_citation(V3Draft(
        answer="Nhân viên không hỗ trợ xếp chỗ ngồi linh động",
        evidence_review_ids=["E1", "E2"],
    ), packet)
    assert refs == ["E2"] and "[E2]" in answer
    assert [item.review_id for item in validate_packet_citations(
        answer, refs, packet, [first, second], route_query(request.query), request.query,
    )] == ["b"]
    with pytest.raises(ValueError, match="clause"):
        validate_packet_citations(
            "Nhân viên phục vụ chậm, kim chi không được giao [E1].", ["E1"],
            packet, [first, second], route_query(request.query), request.query,
        )


def test_one_citation_cannot_cover_second_unsupported_service_fact():
    request = RetrievalRequest(query="Nhân viên phục vụ có vấn đề không?")
    slow = hit("slow", "Nhân viên phục vụ chậm, khách phải chờ lâu")
    rude = hit("rude", "Nhân viên có thái độ không tốt và nói chuyện tệ")
    route = route_query(request.query)
    packet = build_packet(request, [slow, rude], route)
    with pytest.raises(ValueError, match="clause"):
        validate_packet_citations(
            "Nhân viên phục vụ chậm và thái độ không tốt [E1].", ["E1"],
            packet, [slow, rude], route, request.query,
        )
    cited = validate_packet_citations(
        "Nhân viên phục vụ chậm và thái độ không tốt [E1][E2].", ["E1", "E2"],
        packet, [slow, rude], route, request.query,
    )
    assert {item.review_id for item in cited} == {"slow", "rude"}


@pytest.mark.parametrize("word", ["nhiều", "đa số", "phổ biến", "thường xuyên",
                                  "chủ yếu", "nhiều nhất"])
def test_frequency_claim_requires_aggregate_support(word):
    request = RetrievalRequest(query="Nhân viên phục vụ có vấn đề không?")
    review = hit("slow", "Nhân viên phục vụ chậm, khách phải chờ lâu")
    route = route_query(request.query)
    packet = build_packet(request, [review], route)
    with pytest.raises(ValueError, match="frequency"):
        validate_packet_citations(f"{word} khách phàn nàn phục vụ chậm [E1].",
                                  ["E1"], packet, [review], route, request.query)


def test_generation_replays_exact_same_saved_packets_for_models(tmp_path):
    path = tmp_path / "evidence_packets.json"
    path.write_text(json.dumps({"service": {
        "question": "Nhân viên phục vụ có vấn đề gì không?",
        "route": {"category": "SERVICE_COMPLAINT", "intent": "complaint",
                  "aspect": "service", "retrieval_method": "RAG_V3_BM25_FILTERED"},
        "packet": {"question_intent": "SERVICE_COMPLAINT", "aspect": "service",
                   "restaurant": "Quán A", "evidence": [{"ref": "E1", "review_id": "a",
                       "sentiment": "negative", "aspects": ["service"], "issues": [],
                       "rating": "4", "text": "Nhân viên phục vụ chậm"}],
                   "aggregates": {"accepted_review_count": 1, "mixed_sentiment": False,
                                  "frequency_supported": False},
                   "support_gate_status": "ACCEPTED"},
        "retrieval_latency_ms": 2.0,
        "retrieval_debug": {"retrieval_method": "RAG_V3_BM25_FILTERED",
                            "effective_filters": {"restaurant_key": "key-a"},
                            "final_top_k": 5, "query_route": "SERVICE_COMPLAINT"},
    }}), encoding="utf-8")
    first = load_frozen_packets(path)
    second = load_frozen_packets(path)
    assert list(first) == list(second)
    for name in first:
        assert first[name][1][4] == second[name][1][4]
        assert [hit.review_id for hit in first[name][1][1]] == [
            hit.review_id for hit in second[name][1][1]
        ]


def test_minimal_generation_schema_rejects_extra_fields():
    assert V3Draft.model_validate({"answer": "Món ăn ngon"}).evidence_review_ids == []
    with pytest.raises(ValueError):
        V3Draft.model_validate({"answer": "Món ăn ngon", "rating": 9})


def test_food_generation_failure_uses_traceable_extractive_fallback():
    food = hit("food-1", "Món ăn ngon đa dạng, vị vừa miệng", "positive", ("food",))
    gateway = FakeGateway({"answer": "Nhân viên phục vụ chậm", "evidence_review_ids": ["E1"]})
    response, _, telemetry, _, _ = answer_question_v3(
        FakeRetriever([food]), gateway,
        RetrievalRequest(query="Khách nói gì về chất lượng món ăn?", top_k=5),
    )
    assert response.generation_mode == "extractive_fallback"
    assert "Món ăn ngon" in response.answer and "[E1]" in response.answer
    assert [item.review_id for item in response.evidence] == ["food-1"]
    assert telemetry["generation_attempts"] == 2


def test_price_praise_cannot_support_price_complaint():
    cheap = hit("cheap", "Chờ lâu nhưng giá rẻ", "negative", ("price",))
    gateway = FakeGateway()
    response, _, telemetry, _, packet = answer_question_v3(
        FakeRetriever([cheap]), gateway,
        RetrievalRequest(query="Có phàn nàn về giá không?"),
    )
    assert response.answer == INSUFFICIENT and gateway.calls == 0
    assert packet.support_gate_status == "INSUFFICIENT_EVIDENCE"
    assert telemetry["generation_mode"] == "none"


def test_positive_service_cannot_support_complaint():
    praise = hit("p", "Nhân viên phục vụ rất tốt, nhiệt tình", "positive")
    gateway = FakeGateway()
    response, _, _, _, _ = answer_question_v3(
        FakeRetriever([praise]), gateway,
        RetrievalRequest(query="Nhân viên phục vụ có vấn đề không?"),
    )
    assert response.answer == INSUFFICIENT and gateway.calls == 0


def test_mixed_positive_review_cannot_turn_cold_food_into_strength():
    mixed = hit("mixed", "Món ăn được mang lên nhanh nhưng bị nguội; món khác ngon",
                "positive", ("food",))
    packet = build_packet(RetrievalRequest(query="Khách khen điều gì?"), [mixed],
                          route_query("Khách khen điều gì?"))
    assert packet.support_gate_status == "INSUFFICIENT_EVIDENCE"


def test_causal_question_never_infers_cause_from_reviews():
    gateway = FakeGateway()
    response, _, telemetry, _, _ = answer_question_v3(
        FakeRetriever([hit("n", "Nhân viên tệ")]), gateway,
        RetrievalRequest(query="Tại sao đánh giá của nhà hàng giảm?"),
    )
    assert response.answer == INSUFFICIENT
    assert telemetry["status"] == "correct_insufficient" and gateway.calls == 0


def test_health_sections_fallback_remains_traceable(monkeypatch):
    monkeypatch.setattr("src.rag_v3.health.resolve_restaurant",
                        lambda name: ("key-a", SimpleNamespace(
                            review_count=10, average_rating=7.5,
                        )))
    good = hit("good", "Món ăn ngon và phục vụ tốt", "positive", ("food",))
    bad = hit("bad", "Nhân viên phục vụ chậm và thái độ tệ", "negative")
    retriever = FakeRetriever(lambda request: [good] if request.sentiment == "positive"
                              else [bad])
    report, telemetry, verification = health_report_v3(
        retriever, FakeGateway(), "Quán A",
    )
    assert "10" in report.overall_summary and "[M1]" in report.overall_summary
    assert report.strengths and report.main_complaints and report.recommendations
    assert set(report.evidence_review_ids) == {"good", "bad"}
    assert report.aspect_trends == report.warnings == []
    assert all(mode == "extractive_fallback"
               for mode in telemetry["section_modes"].values())
    assert verification["all_review_sections_traceable"]


def test_llmops_v3_columns_and_no_raw_content(tmp_path):
    path = tmp_path / "events.sqlite"
    record("rag_v3_request", "Qwen-test", "hash", "prompt-v3", 12,
           "supported_extractive_fallback", path=path,
           filters={"restaurant_key": "key-a", "query": "secret question"},
           rag_version="RAG_V3", query_route="FOOD_COMPLAINT",
           generation_model="Qwen-test", generation_mode="extractive_fallback",
           evidence_count=1, support_gate_status="ACCEPTED",
           clause_support_status="extractive_verified",
           retrieval_method="RAG_V3_BM25_FILTERED", retrieval_latency_ms=3,
           llm_latency_ms=2, end_to_end_latency_ms=12,
           input_tokens=10, output_tokens=5)
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT rag_version, query_route, generation_model, "
                                 "generation_mode, evidence_count, support_gate_status, "
                                 "clause_support_status, "
                                 "filters_json FROM events").fetchone()
    assert row[:7] == ("RAG_V3", "FOOD_COMPLAINT", "Qwen-test",
                       "extractive_fallback", 1, "ACCEPTED", "extractive_verified")
    assert "secret question" not in row[7]
