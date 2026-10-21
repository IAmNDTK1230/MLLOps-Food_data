import json

import pytest

from src.rag.schemas import RetrievalRequest, ReviewHit
from src.rag.service import INSUFFICIENT, DraftAnswer
from src.rag_v2.hybrid import HYBRID_EVIDENCE, HYBRID_RRF, V1_DENSE, HybridRetriever, rrf_fuse
from src.rag_v2.intent import detect, evidence_compatible
from src.rag_v2.lexical import BM25Retriever, build_index
from src.rag_v2.service import answer_question_v2, support_gate, validate_claim


def hit(review_id, text, sentiment="negative", aspects=("service",), key="r1",
        date="2024-01-15T10:00:00"):
    from src.rag.vector_store import epoch

    return ReviewHit(review_id=review_id, restaurant_name="Quán A", text=text, score=.7,
                     metadata={"review_id": review_id, "restaurant_key": key,
                               "restaurant_name": "Quán A", "predicted_sentiment": sentiment,
                               "predicted_aspects": json.dumps(aspects),
                               "review_timestamp": date, "timestamp_epoch": epoch(date),
                               "raw_rating": "3"})


def test_bm25_index_and_metadata_filters(tmp_path):
    docs = tmp_path / "documents.jsonl"
    rows = [hit("a", "nhân viên phục vụ chậm", "negative", ("service",)),
            hit("b", "nhân viên phục vụ tốt", "positive", ("service",)),
            hit("c", "món ăn không ngon", "negative", ("food",), key="r2")]
    with docs.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({"id": row.review_id, "text": row.text,
                                     "metadata": {"review_id": row.review_id,
                                                  "restaurant_key": row.metadata["restaurant_key"],
                                                  "restaurant_name": row.restaurant_name,
                                                  "review_timestamp": row.metadata["review_timestamp"],
                                                  "raw_rating": "3",
                                                  "predicted_sentiment": row.metadata["predicted_sentiment"],
                                                  "predicted_aspects": json.loads(
                                                      row.metadata["predicted_aspects"]),
                                                  "detected_issues": [], "source": "foody"}})
                         + "\n")
    db, manifest = tmp_path / "fts.sqlite", tmp_path / "manifest.json"
    assert build_index(docs, db, manifest)["document_count"] == 3
    assert build_index(docs, db, manifest)["document_count"] == 3
    lexical = BM25Retriever(db, manifest, docs)
    matches, _ = lexical.search(RetrievalRequest(
        query="nhân viên phục vụ", restaurant_key="r1", sentiment="negative",
        aspect="service", date_from="2024-01-01", date_to="2024-01-31",
    ))
    assert [item.review_id for item in matches] == ["a"]
    assert lexical.search(RetrievalRequest(query="món ăn", restaurant_key="r1"))[0] == []


def test_rrf_fusion_and_v1_v2_separation():
    a, b, c = hit("a", "phục vụ chậm"), hit("b", "phục vụ kém"), hit("c", "phục vụ tệ")
    fused = rrf_fuse([[a, b], [b, c]], constant=60)
    assert [item.review_id for item, _ in fused] == ["b", "a", "c"]
    assert V1_DENSE.use_lexical is False and HYBRID_RRF.use_lexical is True
    assert HYBRID_EVIDENCE.evidence_rerank is True


def test_intent_detection_and_positive_service_rejection():
    complaint = detect("Nhân viên phục vụ có vấn đề gì không?")
    assert complaint.aspect == "service" and complaint.intent == "complaint"
    neutral = detect("Khách nói gì về phục vụ?")
    assert neutral.aspect == "service" and neutral.intent == "general"
    assert detect("Khách có phàn nàn về giá không?").aspect == "price"
    praise = hit("positive", "Nhân viên rất nhiệt tình, phục vụ tốt", "positive")
    assert not evidence_compatible(praise, "complaint", "service")
    assert support_gate([praise], "complaint", "service", "Nhân viên có vấn đề?") == []
    cheap_but_slow = hit("cheap", "Chờ món lâu nhưng giá rất rẻ", "negative", ("price",))
    expensive = hit("expensive", "Giá khá cao, không đáng tiền", "negative", ("price",))
    assert not evidence_compatible(cheap_but_slow, "complaint", "price")
    assert evidence_compatible(expensive, "complaint", "price")


def test_claim_membership_topic_and_mixed_evidence():
    negative = hit("n", "Món ăn không ngon", "negative", ("food",))
    positive = hit("p", "Món ăn rất ngon", "positive", ("food",))
    draft = DraftAnswer(answer="Món ăn được nhận xét trái chiều: có người chê [R1], "
                               "có người khen [R2].", evidence_refs=["R1", "R2", "R3"])
    evidence, repaired = validate_claim(draft, [negative, positive], "general", "food",
                                        "Khách nói gì về món ăn?")
    assert [item.review_id for item in evidence] == ["n", "p"] and repaired
    with pytest.raises(ValueError, match="Mixed evidence"):
        validate_claim(DraftAnswer(answer="Món ăn ngon [R2].", evidence_refs=["R2"]),
                       [negative, positive], "general", "food", "Món ăn thế nào?")
    with pytest.raises(ValueError, match="requested aspect"):
        validate_claim(DraftAnswer(answer="Phục vụ chậm [R1].", evidence_refs=["R1"]),
                       [negative], "general", "food", "Món ăn thế nào?")
    with pytest.raises(ValueError, match="unknown"):
        validate_claim(DraftAnswer(answer="Món ăn ngon [R9].", evidence_refs=["R9"]),
                       [positive], "general", "food", "Món ăn thế nào?")


class FakeDense:
    metadata = {"model_name": "mini", "model_hash": "hash"}

    def __init__(self, hits):
        self.hits = hits

    def search(self, request):
        return self.hits, 1.0


class FakeLexical:
    def __init__(self, hits):
        self.hits = hits

    def search(self, request, **kwargs):
        return self.hits, 2.0


class FakeGateway:
    name = "Qwen-test"
    version = "model-hash"

    def generate(self, **kwargs):
        raise AssertionError("Qwen must not run when evidence is incompatible")


def test_hybrid_filter_and_insufficient_evidence():
    positive = hit("p", "Nhân viên nhiệt tình, phục vụ tốt", "positive")
    negative = hit("n", "Nhân viên phục vụ chậm", "negative")
    retriever = HybridRetriever(FakeDense([positive, negative]), FakeLexical([positive]),
                                HYBRID_EVIDENCE)
    request = RetrievalRequest(query="Nhân viên phục vụ có vấn đề gì không?", top_k=5)
    # Fake backends do not apply filters, so the hybrid validator must fail closed.
    with pytest.raises(ValueError, match="metadata filters"):
        retriever.search_debug(request)
    valid = HybridRetriever(FakeDense([negative]), FakeLexical([negative]),
                            HYBRID_EVIDENCE)
    hits, _, debug = valid.search_debug(request)
    assert [item.review_id for item in hits] == ["n"]
    assert debug["effective_filters"] == {"sentiment": "negative", "aspect": "service"}


def test_insufficient_no_generation(monkeypatch):
    class EmptyRetriever:
        config = HYBRID_EVIDENCE
        metadata = {"model_name": "mini"}

        def search_debug(self, request):
            return [], 1.0, {"candidate_count": 0, "effective_filters": {},
                             "retrieval_method": HYBRID_EVIDENCE.name}

    monkeypatch.setattr("src.rag_v2.service.record", lambda *args, **kwargs: None)
    answer, _, telemetry, _ = answer_question_v2(
        EmptyRetriever(), FakeGateway(), RetrievalRequest(query="phục vụ chậm"),
    )
    assert answer.answer == INSUFFICIENT and answer.claims == []
    assert telemetry["status"] == "insufficient_evidence"
