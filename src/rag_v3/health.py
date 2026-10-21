"""Section-level, evidence-first restaurant report with explicit fallback modes."""

import json
import re
from time import perf_counter

import requests

from src.rag.monitoring import record
from src.rag.schemas import HealthReport, RetrievalRequest
from src.rag.service import prompt, resolve_restaurant
from src.rag_v3.evidence import validate_packet_citations
from src.rag_v3.routing import Route
from src.rag_v3.service import V3Draft, answer_question_v3, attach_citation


def _short_generation(gateway, instruction, version, payload):
    raw, usage = gateway.generate(
        system=instruction, user=json.dumps(payload, ensure_ascii=False),
        response_schema=V3Draft.model_json_schema(), prompt_version=version,
        max_new_tokens=96, component="health_v3_generation",
    )
    return V3Draft.model_validate(raw), usage


def health_report_v3(retriever, gateway, name_or_key):
    started = perf_counter()
    key, summary = resolve_restaurant(name_or_key)
    instruction, version = prompt("prompts/restaurant_report/health_v3.txt")
    usages = []
    modes = {}
    validation = {}
    fallback_count = 0
    metric_answer = (f"Dữ liệu có {int(summary.review_count)} đánh giá; "
                     f"điểm trung bình {float(summary.average_rating):.2f} [M1].")
    try:
        draft, usage = _short_generation(gateway, instruction, version, {
            "section": "overall_summary",
            "metrics": {"review_count": int(summary.review_count),
                        "average_rating": round(float(summary.average_rating), 2),
                        "citation": "M1"},
        })
        usages.append(usage)
        if (draft.evidence_review_ids or "[M1]" not in draft.answer
                or str(int(summary.review_count)) not in draft.answer
                or f"{float(summary.average_rating):.2f}" not in draft.answer
                or not re.fullmatch(r"[^\d]*\d+[^\d]*\d+\.\d+[^\d]*\[M1\]\.?",
                                    draft.answer)):
            raise ValueError("Metric summary differs from supplied source")
        overall = draft.answer
        modes["overall_summary"] = "qwen"
        validation["overall_summary"] = "metric_verified"
    except (ValueError, KeyError, TypeError, requests.RequestException):
        overall = metric_answer
        modes["overall_summary"] = "extractive_fallback"
        validation["overall_summary"] = "metric_fallback_verified"
        fallback_count += 1

    section_results = {}
    section_packets = {}
    section_hits = {}
    section_evidence = {}
    telemetry_parts = []
    for section, query, sentiment in (
        ("strengths", "Khách khen điều gì?", "positive"),
        ("complaints", "Khách phàn nàn điều gì?", "negative"),
    ):
        answer, hits, telemetry, _, packet = answer_question_v3(
            retriever, gateway,
            RetrievalRequest(query=query, restaurant_key=key, sentiment=sentiment,
                             top_k=5),
        )
        telemetry_parts.append(telemetry)
        section_packets[section] = packet
        section_hits[section] = hits
        section_evidence[section] = [item.review_id for item in answer.evidence]
        section_results[section] = [answer.answer] if answer.evidence else []
        modes[section] = answer.generation_mode
        validation[section] = telemetry["clause_support_status"]
        fallback_count += answer.generation_mode == "extractive_fallback"

    recommendations = []
    complaint_packet = section_packets["complaints"]
    if complaint_packet.evidence:
        fallback = ("Nên xác minh phản ánh trong đánh giá "
                    f"[{complaint_packet.evidence[0]['ref']}] trước khi thay đổi vận hành.")
        try:
            draft, usage = _short_generation(gateway, instruction, version, {
                "section": "recommendations",
                "instruction": "Chỉ đề nghị xác minh phản ánh trước khi hành động.",
                "evidence": list(complaint_packet.evidence),
            })
            usages.append(usage)
            cited_answer, cited_refs = attach_citation(draft, complaint_packet)
            validate_packet_citations(
                cited_answer, cited_refs, complaint_packet,
                section_hits["complaints"],
                Route("HEALTH_RECOMMENDATION", "complaint", None,
                      "RAG_V3_ROUTED"),
                "Khách phàn nàn điều gì?",
            )
            if not re.search(r"xác minh|kiểm tra", cited_answer, flags=re.IGNORECASE):
                raise ValueError("Recommendation lacks verification caveat")
            recommendations = [cited_answer]
            modes["recommendations"] = "qwen"
            validation["recommendations"] = "passed"
        except (ValueError, KeyError, TypeError, requests.RequestException):
            recommendations = [fallback]
            modes["recommendations"] = "extractive_fallback"
            validation["recommendations"] = "extractive_verified"
            fallback_count += 1
    else:
        modes["recommendations"] = "none"
        validation["recommendations"] = "not_checked"

    used_reviews = list(dict.fromkeys(
        [*section_evidence["strengths"], *section_evidence["complaints"]],
    ))
    recommendation_refs = set(re.findall(r"\[(E\d+)\]", " ".join(recommendations)))
    for item in complaint_packet.evidence:
        if item["ref"] in recommendation_refs and item["review_id"] not in used_reviews:
            used_reviews.append(item["review_id"])
    result = HealthReport(
        overall_summary=overall,
        strengths=section_results["strengths"],
        main_complaints=section_results["complaints"],
        recommendations=recommendations,
        aspect_trends=[], warnings=[], evidence_review_ids=used_reviews,
    )
    # Neither aspect-time evidence nor a validated current alert is asserted.
    telemetry = {
        "retrieval_latency_ms": sum(item["retrieval_latency_ms"] for item in telemetry_parts),
        "llm_latency_ms": sum(item["latency_ms"] for item in usages)
        + sum(item["llm_latency_ms"] or 0 for item in telemetry_parts),
        "end_to_end_latency_ms": (perf_counter() - started) * 1000,
        "input_tokens": sum(item["input_tokens"] or 0 for item in usages)
        + sum(item["input_tokens"] or 0 for item in telemetry_parts),
        "output_tokens": sum(item["output_tokens"] or 0 for item in usages)
        + sum(item["output_tokens"] or 0 for item in telemetry_parts),
        "generation_mode": "qwen" if all(mode == "qwen" for mode in modes.values())
        else "extractive_fallback",
        "fallback_sections": fallback_count,
        "section_modes": modes, "section_validation": validation,
        "clause_support_status": "passed_or_extractive_verified" if all(
            value in ("passed", "extractive_verified", "metric_verified",
                      "metric_fallback_verified") for value in validation.values()
        ) else "not_checked",
        "status": "traceable" if used_reviews else "metrics_only",
        "prompt_version": version,
    }
    record("health_v3_request", gateway.name, gateway.version, version,
           telemetry["end_to_end_latency_ms"], telemetry["status"],
           input_tokens=telemetry["input_tokens"], output_tokens=telemetry["output_tokens"],
           top_k=5, filters={"restaurant_key": key},
           embedding_model=retriever.metadata.get("model_name"),
           retrieval_latency_ms=telemetry["retrieval_latency_ms"],
           llm_latency_ms=telemetry["llm_latency_ms"],
           end_to_end_latency_ms=telemetry["end_to_end_latency_ms"],
           schema_valid=True, retrieval_version="RAG_V3",
           retrieval_method="RAG_V3_ROUTED", candidate_count=len(used_reviews),
           final_top_k=5, rag_version="RAG_V3", query_route="HEALTH_SECTIONS",
           generation_model=gateway.name,
           generation_mode=telemetry["generation_mode"],
           evidence_count=len(used_reviews), support_gate_status=telemetry["status"],
           clause_support_status=telemetry["clause_support_status"])
    evidence_by_section = {}
    for section in ("strengths", "complaints"):
        cited = set(section_evidence[section])
        evidence_by_section[section] = [
            {"review_id": hit.review_id, "text": hit.text,
             "sentiment": hit.metadata.get("predicted_sentiment"),
             "aspects": hit.metadata.get("predicted_aspects"),
             "rating": hit.metadata.get("raw_rating")}
            for hit in section_hits[section] if hit.review_id in cited
        ]
    evidence_by_section["recommendations"] = [
        {"review_id": item["review_id"], "text": item["text"],
         "sentiment": item["sentiment"], "aspects": item["aspects"],
         "rating": item["rating"]}
        for item in complaint_packet.evidence if item["ref"] in recommendation_refs
    ]
    verification = {"section_modes": modes, "section_validation": validation,
                    "all_review_sections_traceable": bool(
                        section_results["strengths"] and section_results["complaints"]),
                    "evidence_by_section": evidence_by_section,
                    "aspect_trends": "empty: no aspect-specific time evidence",
                    "warnings": "empty: no validated current alert",
                    "recommendation_source": complaint_packet.evidence[0]["review_id"]
                    if recommendations else None}
    return result, telemetry, verification
