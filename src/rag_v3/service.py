"""Evidence-first V3 answering; the generator may only phrase accepted facts."""

import json
from time import perf_counter

import requests
from pydantic import BaseModel, ConfigDict, Field

from src.rag.monitoring import record
from src.rag.schemas import Evidence, RetrievalRequest
from src.rag.service import INSUFFICIENT, prompt
from src.rag_v3.evidence import (
    build_packet,
    evidence_for_refs,
    extractive_answer,
    validate_packet_citations,
)
from src.rag_v3.routing import route_query


class V3Draft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str
    evidence_review_ids: list[str] = Field(default_factory=list)


class V3Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    evidence_review_ids: list[str]


class GroundedAnswerV3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    claims: list[V3Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    model: str
    prompt_version: str
    retrieval_version: str = "RAG_V3"
    generation_mode: str
    query_route: str
    support_gate_status: str


def _as_evidence(hits):
    return [Evidence(review_id=hit.review_id, restaurant_name=hit.restaurant_name,
                     score=hit.score) for hit in hits]


def attach_citation(draft, packet):
    """Choose a short evidence placeholder when Qwen omits the inline citation."""
    import re

    answer = draft.answer.strip()
    if re.search(r"[\u4e00-\u9fff]", answer):
        raise ValueError("Malformed Vietnamese output")
    known = {item["ref"]: item for item in packet.evidence}
    inline = re.findall(r"\[(E\d+)\]", answer)
    refs = inline or [ref for ref in draft.evidence_review_ids if ref in known]
    if any(ref not in known for ref in inline):
        raise ValueError("Unknown inline citation")
    if inline:
        return answer, list(dict.fromkeys(inline))
    if not refs:
        refs = list(known)
    tokens = set(re.findall(r"[^\W_]{3,}", answer.casefold()))
    ignored = {"khách", "đánh", "giá", "quán", "một", "các", "về", "được",
               "trong", "những", "điều", "này", "rất", "cho", "với", "nhân", "viên"}
    tokens -= ignored
    ranked = sorted(((len(tokens & set(re.findall(r"[^\W_]{3,}",
                                                   known[ref]["text"].casefold()))), ref)
                     for ref in refs), reverse=True)
    overlap, selected_ref = ranked[0]
    if overlap < 2:
        raise ValueError("Generated claim lacks lexical support")
    return f"{answer.rstrip('.')} [{selected_ref}].", [selected_ref]


def prepare_question(retriever, request):
    if not isinstance(request, RetrievalRequest):
        request = RetrievalRequest.model_validate(request)
    route = route_query(request.query)
    hits, retrieval_ms, debug = retriever.search_debug(request)
    packet = build_packet(request, hits, route)
    return route, hits, retrieval_ms, debug, packet


def answer_question_v3(retriever, gateway, request, *, prepared=None,
                       generation_audit=None):
    if not isinstance(request, RetrievalRequest):
        request = RetrievalRequest.model_validate(request)
    started = perf_counter()
    route, hits, retrieval_ms, debug, packet = prepared or prepare_question(
        retriever, request,
    )
    instruction, version = prompt("prompts/rag/rag_v3.txt")
    answer, refs, mode = INSUFFICIENT, [], "none"
    status = "correct_insufficient" if route.category == "TREND_CAUSATION" else \
        "insufficient_evidence"
    failure_category = None
    usages = []
    generation_attempts = 0
    schema_success = False
    citation_success = False
    clause_support_status = "not_checked"
    if packet.support_gate_status == "ACCEPTED":
        for attempt in range(2):
            generation_attempts += 1
            correction = ("\nLần trước không đạt. Chỉ viết MỘT câu ngắn, đúng chủ đề "
                          "với [E1] và evidence_review_ids=[\"E1\"]." if attempt else "")
            try:
                attempt_audit = [] if generation_audit is not None else None
                raw, usage = gateway.generate(
                    system=instruction + correction,
                    user=json.dumps(packet.generation_payload(request.query),
                                    ensure_ascii=False),
                    response_schema=V3Draft.model_json_schema(),
                    prompt_version=version, max_new_tokens=128,
                    component="rag_v3_generation",
                    audit_sink=attempt_audit,
                )
                if generation_audit is not None:
                    generation_audit.append({"attempt": attempt + 1,
                                             "gateway": attempt_audit,
                                             "parsed_output": raw})
                usages.append(usage)
                draft = V3Draft.model_validate(raw)
                schema_success = True
                cited_answer, cited_refs = attach_citation(draft, packet)
                validate_packet_citations(cited_answer, cited_refs,
                                          packet, hits, route, request.query)
                clause_support_status = "passed"
                citation_success = True
                answer, refs, mode = cited_answer, cited_refs, "qwen"
                status = "grounded"
                failure_category = None
                break
            except (ValueError, KeyError, TypeError, requests.RequestException) as error:
                failure_category = (str(error)[:120] if isinstance(error, ValueError)
                                    else type(error).__name__)
                clause_support_status = "failed" if schema_success else "not_checked"
                if generation_audit is not None:
                    if len(generation_audit) < attempt + 1:
                        generation_audit.append({"attempt": attempt + 1,
                                                 "gateway": attempt_audit})
                    generation_audit[-1]["failure"] = failure_category
        if mode != "qwen":
            answer, refs = extractive_answer(packet)
            # Extractive text comes directly from a known review, but IDs must still map.
            evidence_for_refs(packet, refs, hits)
            mode, status = "extractive_fallback", "supported_extractive_fallback"
            citation_success = True
            clause_support_status = "extractive_verified"
    cited_hits = evidence_for_refs(packet, refs, hits) if refs else []
    response = GroundedAnswerV3(
        answer=answer,
        claims=[V3Claim(claim=answer,
                        evidence_review_ids=[hit.review_id for hit in cited_hits])]
        if cited_hits else [],
        evidence=_as_evidence(cited_hits), model=gateway.name,
        prompt_version=version, generation_mode=mode, query_route=route.category,
        support_gate_status=packet.support_gate_status,
    )
    telemetry = {
        "retrieval_latency_ms": retrieval_ms,
        "llm_latency_ms": sum(item["latency_ms"] for item in usages) if usages else None,
        "end_to_end_latency_ms": (perf_counter() - started) * 1000
        + (retrieval_ms if prepared is not None else 0),
        "input_tokens": sum(item["input_tokens"] or 0 for item in usages)
        if usages else (0 if not generation_attempts else None),
        "output_tokens": sum(item["output_tokens"] or 0 for item in usages)
        if usages else (0 if not generation_attempts else None),
        "generation_attempts": generation_attempts,
        "schema_success": schema_success, "citation_success": citation_success,
        "status": status, "failure_category": failure_category,
        "prompt_version": version, "model_version": gateway.version,
        "rag_version": "RAG_V3", "query_route": route.category,
        "retrieval_method": debug["retrieval_method"],
        "generation_model": gateway.name, "generation_mode": mode,
        "evidence_count": len(cited_hits),
        "support_gate_status": packet.support_gate_status,
        "clause_support_status": clause_support_status,
    }
    record("rag_v3_request", gateway.name, gateway.version, version,
           telemetry["end_to_end_latency_ms"], status,
           input_tokens=telemetry["input_tokens"],
           output_tokens=telemetry["output_tokens"], top_k=request.top_k,
           filters=debug["effective_filters"],
           embedding_model=retriever.metadata.get("model_name"),
           retrieval_latency_ms=retrieval_ms,
           llm_latency_ms=telemetry["llm_latency_ms"],
           end_to_end_latency_ms=telemetry["end_to_end_latency_ms"],
           schema_valid=schema_success, retrieval_version="RAG_V3",
           retrieval_method=debug["retrieval_method"],
           reranker_model="evidence-rules-v1" if "EVIDENCE" in debug["retrieval_method"]
           else None,
           candidate_count=debug["candidate_count"], final_top_k=request.top_k,
           rag_version="RAG_V3", query_route=route.category,
           generation_model=gateway.name, generation_mode=mode,
           evidence_count=len(cited_hits),
           support_gate_status=packet.support_gate_status,
           clause_support_status=clause_support_status)
    return response, hits, telemetry, debug, packet
