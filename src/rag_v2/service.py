"""Evidence-aware RAG V2 answer with one claim and conservative semantic gates."""

import json
import re
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from src.rag.monitoring import record
from src.rag.schemas import Evidence, RetrievalRequest
from src.rag.service import INSUFFICIENT, DraftAnswer, check_refs, context, prompt
from src.rag_v2.intent import ASPECT_TERMS, detect, evidence_compatible, issue_matches


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    evidence_review_ids: list[str]


class GroundedAnswerV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    model: str
    prompt_version: str
    retrieval_version: str = "RAG_V2"


def support_gate(hits, intent, aspect, query):
    if intent not in ("complaint", "praise"):
        return hits
    selected = []
    for hit in hits:
        if not evidence_compatible(hit, intent, aspect):
            continue
        issue = issue_matches(hit.text, query)
        if issue is False:
            continue
        selected.append(hit)
    return selected


def validate_claim(draft, selected, intent, aspect, query):
    inline_refs = list(dict.fromkeys(re.findall(r"\[(R\d+)\]", draft.answer)))
    # The small local model sometimes lists extra IDs not cited in its answer.
    # Only explicit inline citations can become claim evidence; never add IDs.
    cited = check_refs(inline_refs, draft.answer, selected)
    list_repaired = set(inline_refs) != set(draft.evidence_refs)
    if aspect and not any(term in draft.answer.casefold() for term in ASPECT_TERMS[aspect]):
        raise ValueError("Answer does not address requested aspect")
    if intent == "complaint" and re.search(r"nhiều nhất|phổ biến|thường xuyên", draft.answer,
                                            flags=re.IGNORECASE):
        raise ValueError("Unsupported complaint frequency claim")
    if intent == "general":
        labels = {hit.metadata.get("predicted_sentiment") for hit in selected}
        if {"negative", "positive"} <= labels and not re.search(
            r"trái chiều|khác nhau|có người khen.*có người chê|vừa khen.*vừa chê",
            draft.answer, flags=re.IGNORECASE,
        ):
            raise ValueError("Mixed evidence requires a mixed answer")
    by_id = {hit.review_id: hit for hit in selected}
    for evidence in cited:
        hit = by_id[evidence.review_id]
        if aspect and not any(term in hit.text.casefold() for term in ASPECT_TERMS[aspect]):
            raise ValueError("Cited review does not discuss requested aspect")
        if not evidence_compatible(hit, intent, aspect):
            raise ValueError("Cited review conflicts with requested intent/aspect")
        if issue_matches(hit.text, query) is False:
            raise ValueError("Cited review does not discuss requested issue")
    return cited, list_repaired


def answer_question_v2(retriever, gateway, request):
    if not isinstance(request, RetrievalRequest):
        request = RetrievalRequest.model_validate(request)
    started = perf_counter()
    instruction, version = prompt("prompts/rag/rag_v2.txt")
    query_intent = detect(request.query)
    hits, retrieval_ms, debug = retriever.search_debug(request)
    limit = 2 if query_intent.intent in ("complaint", "praise") else 4
    selected = support_gate(hits, query_intent.intent, query_intent.aspect, request.query)[:limit]
    usages = []
    citation_list_repaired = False
    failure_code = None
    status = "insufficient_evidence"
    response = GroundedAnswerV2(answer=INSUFFICIENT, model=gateway.name, prompt_version=version)
    if query_intent.intent == "causal":
        status = "unsupported_causal_inference"
    elif selected:
        payload = {"question": request.query, "intent": query_intent.intent,
                   "aspect": query_intent.aspect, "reviews": context(selected, max_chars=300)}
        for attempt in range(2):
            correction = ("\nLần trước chưa hợp lệ. Nếu bằng chứng đủ, trả lời đúng chủ đề bằng "
                          "MỘT nhận định ngắn có [R1] hoặc [R2] trong answer. "
                          "Nếu trái chiều, nói rõ trái chiều. Không thêm mã không dùng."
                          if attempt else "")
            try:
                draft, usage = gateway.generate(
                    system=instruction + correction,
                    user=json.dumps(payload, ensure_ascii=False),
                    response_schema=DraftAnswer.model_json_schema(),
                    prompt_version=version, max_new_tokens=180,
                    component="rag_v2_generation",
                )
                usages.append(usage)
                draft = DraftAnswer.model_validate(draft)
                if draft.answer.strip() == INSUFFICIENT and not draft.evidence_refs:
                    status = "model_declared_insufficient"
                    continue
                evidence, citation_list_repaired = validate_claim(
                    draft, selected, query_intent.intent, query_intent.aspect, request.query,
                )
                response = GroundedAnswerV2(
                    answer=draft.answer,
                    claims=[Claim(claim=draft.answer,
                                  evidence_review_ids=[item.review_id for item in evidence])],
                    evidence=evidence, model=gateway.name, prompt_version=version,
                )
                status = "grounded"
                failure_code = None
                break
            except (ValueError, KeyError, TypeError) as error:
                status = "invalid_generation_or_semantic_support"
                known = ("Answer does not address requested aspect",
                         "Cited review does not discuss requested aspect",
                         "Answer citations do not match its evidence list",
                         "Answer cites unknown, duplicate or no review IDs",
                         "Unsupported complaint frequency claim",
                         "Mixed evidence requires a mixed answer",
                         "Cited review conflicts with requested intent/aspect",
                         "Cited review does not discuss requested issue")
                failure_code = next((item for item in known if str(error).startswith(item)),
                                    "schema_or_generation_validation")
    telemetry = {"retrieval_latency_ms": retrieval_ms,
                 "llm_latency_ms": sum(item["latency_ms"] for item in usages) if usages else None,
                 "input_tokens": sum(item["input_tokens"] or 0 for item in usages)
                 if usages else (0 if not selected else None),
                 "output_tokens": sum(item["output_tokens"] or 0 for item in usages)
                 if usages else (0 if not selected else None),
                 "generation_attempts": len(usages),
                 "end_to_end_latency_ms": (perf_counter() - started) * 1000,
                 "schema_valid": status == "grounded" if usages else None,
                 "citation_list_repaired": citation_list_repaired,
                 "failure_code": failure_code,
                 "status": status, "prompt_version": version,
                 "model_version": gateway.version,
                 "retrieval_version": "RAG_V2",
                 "retrieval_method": retriever.config.name,
                 "reranker_model": "evidence-rules-v1" if retriever.config.evidence_rerank else None,
                 "candidate_count": debug["candidate_count"],
                 "final_top_k": request.top_k}
    record("rag_v2_request", gateway.name, gateway.version, version,
           telemetry["end_to_end_latency_ms"], status,
           input_tokens=telemetry["input_tokens"], output_tokens=telemetry["output_tokens"],
           top_k=request.top_k,
           filters=debug["effective_filters"],
           embedding_model=retriever.metadata["model_name"],
           retrieval_latency_ms=retrieval_ms, llm_latency_ms=telemetry["llm_latency_ms"],
           end_to_end_latency_ms=telemetry["end_to_end_latency_ms"],
           schema_valid=telemetry["schema_valid"],
           retrieval_version="RAG_V2", retrieval_method=retriever.config.name,
           reranker_model=telemetry["reranker_model"],
           candidate_count=debug["candidate_count"], final_top_k=request.top_k)
    return response, hits, telemetry, debug
