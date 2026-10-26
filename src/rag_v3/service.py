"""Evidence-first V3 answering; the generator may only phrase accepted facts."""

import json
import re
from time import perf_counter

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.rag.monitoring import record
from src.rag.schemas import Evidence, RetrievalRequest
from src.rag.service import INSUFFICIENT, prompt
from src.rag_v2.intent import ASPECT_TERMS
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


class V3AnalysisClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    evidence_refs: list[str]


class V3AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    claims: list[V3AnalysisClaim] = Field(max_length=3)
    insufficient_evidence: bool = False


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
    answer = draft.answer.strip()
    if re.search(r"[\u4e00-\u9fff]", answer):
        raise ValueError("Malformed Vietnamese output")
    known = {item["ref"]: item for item in packet.evidence}
    # Some small-model outputs use explicit "E1: fact" labels rather than
    # brackets. Convert only that unambiguous formatting; validation below
    # still decides whether the attributed review actually supports the fact.
    if not re.search(r"\[E\d+\]", answer):
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        labeled = []
        for sentence in sentences:
            label = re.match(r"(?:Nhận định (?:ngắn|khác)\s+)?(E\d+):\s*(.+)",
                             sentence, flags=re.IGNORECASE)
            if not label:
                labeled = []
                break
            ref, fact = label.groups()
            if ref not in known:
                raise ValueError("Unknown inline citation")
            labeled.append(f"{fact.rstrip('.')} [{ref}].")
        if labeled:
            answer = " ".join(labeled)
    inline = re.findall(r"\[(E\d+)\]", answer)
    refs = inline or [ref for ref in draft.evidence_review_ids if ref in known]
    if any(ref not in known for ref in inline):
        raise ValueError("Unknown inline citation")
    if inline:
        return answer, list(dict.fromkeys(inline))
    if len(re.split(r"(?<=[.!?])\s+", answer)) > 1:
        raise ValueError("Missing inline citations in multi-sentence answer")
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


def generation_mode_label(mode):
    return {"qwen": "AI synthesis", "extractive_fallback": "Evidence fallback",
            "none": "Insufficient evidence"}.get(mode, mode)


def _repair_instruction(reason, aspect=None):
    targeted = {
        "Answer adds off-topic restaurant facts":
            "Chỉ nói về aspect được hỏi; bỏ toàn bộ chi tiết thuộc aspect khác.",
        "Claim has missing or unknown evidence refs":
            "Mỗi claim phải có evidence_refs chứa đúng E1/E2/E3 từ packet.",
        "Complaint answer lacks complaint wording":
            "Nêu đúng phàn nàn cụ thể trong review, không thay bằng nhận xét chung.",
        "Answer clause lacks cited support":
            "Dùng đúng chi tiết và từ đánh giá của review gắn citation. "
            "Tránh nhận xét chung như 'được đánh giá tốt' nếu review chỉ nói "
            "'ưng ý' hoặc 'ngon'; xóa mệnh đề không được hỗ trợ.",
        "Unsupported frequency claim":
            "Bỏ các khẳng định về số lượng/tần suất khách hàng.",
        "schema_validation_failed":
            "Chỉ trả tối đa 3 claims đúng schema, mỗi claim một nhận định ngắn "
            "với evidence_refs hợp lệ; không thêm trường khác.",
    }
    if reason == "Answer does not address requested aspect" and aspect:
        terms = {"food": "món ăn", "service": "phục vụ", "price": "giá",
                 "ambiance": "không gian", "cleanliness": "vệ sinh"}
        return (f"Nêu rõ khía cạnh '{terms[aspect]}' trong claim; chỉ nhắc chi tiết "
                "mà review gắn citation thực sự nói tới.")
    return targeted.get(reason, "Viết lại ngắn, chỉ với sự kiện được review trích dẫn hỗ trợ.")


def _answer_claims(answer, refs, packet, mode):
    by_ref = {item["ref"]: item["review_id"] for item in packet.evidence}
    sentences = ([answer] if mode == "extractive_fallback" else
                 re.split(r"(?<=[.!?])\s+|;", answer))
    return [V3Claim(
        claim=sentence.strip(),
        evidence_review_ids=[by_ref[ref] for ref in dict.fromkeys(
            re.findall(r"\[(E\d+)\]", sentence)
        ) if ref in by_ref],
    ) for sentence in sentences if re.search(r"\[(E\d+)\]", sentence)]


def _failure_category(error):
    """Persist only bounded validation reasons, never generated review text."""
    if isinstance(error, ValidationError):
        return "schema_validation_failed"
    if isinstance(error, ValueError):
        known = {"Invalid JSON", "Duplicate JSON key",
                 "Answer adds off-topic restaurant facts",
                 "Claim has missing or unknown evidence refs",
                 "Complaint answer lacks complaint wording",
                 "Answer clause lacks cited support", "Unsupported frequency claim",
                 "Answer does not address requested aspect",
                 "Mixed evidence requires a mixed answer",
                 "Model declared insufficient evidence"}
        return str(error) if str(error) in known else "invalid_generated_content"
    return type(error).__name__


def _assemble_analysis(draft, packet, aspect=None):
    if draft.insufficient_evidence or not draft.claims:
        raise ValueError("Model declared insufficient evidence")
    known = {item["ref"] for item in packet.evidence}
    sentences = []
    refs = []
    for claim in draft.claims:
        content = claim.text.strip().rstrip(".!?")
        if (not content or len(content) > 220 or re.search(r"\[(?:E\d+)\]", content)
                or re.search(r"[\u4e00-\u9fff]", content)
                or len(re.split(r"[.!?]", content)) > 1):
            raise ValueError("Claim must be one short uncited Vietnamese sentence")
        if (not claim.evidence_refs or len(claim.evidence_refs) != len(set(claim.evidence_refs))
                or any(ref not in known for ref in claim.evidence_refs)):
            raise ValueError("Claim has missing or unknown evidence refs")
        sentences.append(content + " " + "".join(f"[{ref}]" for ref in claim.evidence_refs) + ".")
        refs.extend(claim.evidence_refs)
    answer = " ".join(sentences)
    if aspect and not any(term in answer.casefold() for term in ASPECT_TERMS[aspect]):
        cue = {"food": "món ăn", "service": "phục vụ", "price": "giá",
               "ambiance": "không gian", "cleanliness": "vệ sinh"}[aspect]
        answer = f"Về {cue}, {answer[0].lower()}{answer[1:]}"
    return answer, list(dict.fromkeys(refs))


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
    generation_elapsed_ms = 0.0
    validation_elapsed_ms = 0.0
    repair_elapsed_ms = 0.0
    schema_success = False
    citation_success = False
    clause_support_status = "not_checked"
    if packet.support_gate_status == "ACCEPTED":
        rejected_answer = ""
        for attempt in range(2):
            attempt_started = perf_counter()
            generation_attempts += 1
            correction = ("\nBản trước bị từ chối. KHÔNG lặp lại nguyên văn. "
                          + _repair_instruction(failure_category, route.aspect)
                          + " Mỗi claim có evidence_refs; chỉ dùng nội dung review đó hỗ trợ."
                          if attempt else "")
            payload = packet.generation_payload(request.query)
            if attempt:
                payload["repair"] = {"rejected_answer": rejected_answer,
                                     "validation_failure": failure_category}
            attempt_audit = []
            validation_started = None
            try:
                generation_started = perf_counter()
                try:
                    raw, usage = gateway.generate(
                        system=instruction + correction,
                        user=json.dumps(payload, ensure_ascii=False),
                        response_schema=V3AnalysisDraft.model_json_schema(),
                        prompt_version=version, max_new_tokens=192,
                        component="rag_v3_generation",
                        audit_sink=attempt_audit,
                    )
                finally:
                    generation_elapsed_ms += (perf_counter() - generation_started) * 1000
                if generation_audit is not None:
                    generation_audit.append({"attempt": attempt + 1,
                                             "gateway": attempt_audit,
                                             "parsed_output": raw})
                usages.append(usage)
                validation_started = perf_counter()
                draft = V3AnalysisDraft.model_validate(raw)
                schema_success = True
                rejected_answer = " ".join(claim.text for claim in draft.claims)[:600]
                cited_answer, cited_refs = _assemble_analysis(draft, packet, route.aspect)
                rejected_answer = cited_answer[:600]
                validate_packet_citations(cited_answer, cited_refs,
                                          packet, hits, route, request.query)
                clause_support_status = "passed"
                citation_success = True
                answer, refs, mode = cited_answer, cited_refs, "qwen"
                status = "grounded"
                failure_category = None
                break
            except (ValueError, KeyError, TypeError, requests.RequestException) as error:
                if attempt_audit and not rejected_answer:
                    rejected_answer = attempt_audit[0].get("raw_output", "")[:600]
                failure_category = _failure_category(error)
                clause_support_status = "failed" if schema_success else "not_checked"
                if generation_audit is not None:
                    if len(generation_audit) < attempt + 1:
                        generation_audit.append({"attempt": attempt + 1,
                                                 "gateway": attempt_audit})
                    generation_audit[-1]["failure"] = failure_category
            finally:
                if validation_started is not None:
                    validation_elapsed_ms += (perf_counter() - validation_started) * 1000
                if attempt:
                    repair_elapsed_ms += (perf_counter() - attempt_started) * 1000
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
        claims=_answer_claims(answer, refs, packet, mode) if cited_hits else [],
        evidence=_as_evidence(cited_hits), model=gateway.name,
        prompt_version=version, generation_mode=mode, query_route=route.category,
        support_gate_status=packet.support_gate_status,
    )
    telemetry = {
        "retrieval_latency_ms": retrieval_ms,
        "llm_latency_ms": generation_elapsed_ms if generation_attempts else None,
        "validation_latency_ms": validation_elapsed_ms if generation_attempts else None,
        "repair_latency_ms": repair_elapsed_ms if generation_attempts > 1 else None,
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
           clause_support_status=clause_support_status,
           validation_latency_ms=telemetry["validation_latency_ms"],
           repair_latency_ms=telemetry["repair_latency_ms"],
           generation_attempts=generation_attempts,
           failure_category=failure_category)
    return response, hits, telemetry, debug, packet
