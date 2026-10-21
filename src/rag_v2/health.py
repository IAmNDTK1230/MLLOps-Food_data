"""Post-generation semantic checks for the existing health-report flow."""

import re
from time import perf_counter

import pandas as pd

from src.rag.monitoring import record
from src.rag.service import DraftAnswer, context, health_report, prompt, resolve_restaurant
from src.rag_v2.intent import evidence_compatible
from src.rag_v2.service import validate_claim


def _extractive(hit, ref):
    words = hit.text[:150].rsplit(" ", 1)[0]
    return f"Một đánh giá viết: “{words}…” [{ref}]."


def health_report_v2(retriever, gateway, name_or_key):
    started = perf_counter()
    report, hits, telemetry = health_report(retriever, gateway, name_or_key)
    by_ref = {f"R{index}": hit for index, hit in enumerate(hits, 1)}
    rejected = []

    def accepted(items, intent):
        result = []
        for item in items:
            refs = re.findall(r"\[(R\d+)\]", item)
            if refs and all(ref in by_ref and evidence_compatible(
                by_ref[ref], intent,
            ) for ref in refs):
                result.append(item)
            else:
                rejected.append({"section": intent, "reason": "citation or polarity incompatible"})
        return result

    strengths = accepted(report.strengths, "praise")
    complaints = accepted(report.main_complaints, "complaint")
    focused_usage = []
    focused_success = 0
    extractive_fallback = 0
    instruction, focused_version = prompt("prompts/restaurant_report/health_v2.txt")
    for intent, output, question in (
        ("praise", strengths, "Khách khen điều gì?"),
        ("complaint", complaints, "Khách phàn nàn điều gì?"),
    ):
        if output:
            continue
        eligible = [(ref, hit) for ref, hit in by_ref.items()
                    if evidence_compatible(hit, intent)]
        if not eligible:
            continue
        ref, hit = eligible[0]
        try:
            import json

            draft, usage = gateway.generate(
                system=instruction,
                user=json.dumps({"question": question,
                                 "reviews": context([hit], max_chars=250)},
                                ensure_ascii=False),
                response_schema=DraftAnswer.model_json_schema(),
                prompt_version=focused_version, max_new_tokens=120,
                component="health_v2_generation",
            )
            focused_usage.append(usage)
            draft = DraftAnswer.model_validate(draft)
            if draft.answer == "Không đủ bằng chứng từ các đánh giá được truy xuất.":
                raise ValueError("model declared insufficient")
            validate_claim(draft, [hit], intent, None, question)
            # The focused context renumbers its sole review as R1.
            output.append(draft.answer.replace("[R1]", f"[{ref}]"))
            focused_success += 1
        except (ValueError, KeyError, TypeError):
            output.append(_extractive(hit, ref))
            extractive_fallback += 1
    used_refs = set(re.findall(r"\[(R\d+)\]", " ".join([*strengths, *complaints])))
    recommendations = [item for item in report.recommendations
                       if set(re.findall(r"\[(R\d+)\]", item)) <= used_refs
                       and re.search(r"\[(?:R\d+|M[123])\]", item)]
    key, _ = resolve_restaurant(name_or_key)
    alerts = pd.read_parquet("data/foody/mart/alerts.parquet")
    has_alerts = not alerts.loc[alerts.restaurant_key == key].empty
    warnings = [item for item in report.warnings if has_alerts and "[M3]" in item]
    # The current mart has no aspect-specific time series.
    result = report.model_copy(update={
        "strengths": strengths, "main_complaints": complaints,
        "aspect_trends": [], "warnings": warnings,
        "recommendations": recommendations,
        "evidence_review_ids": [hit.review_id for ref, hit in by_ref.items() if ref in used_refs],
    })
    telemetry = {**telemetry,
                 "input_tokens": telemetry["input_tokens"] + sum(
                     item["input_tokens"] or 0 for item in focused_usage),
                 "output_tokens": telemetry["output_tokens"] + sum(
                     item["output_tokens"] or 0 for item in focused_usage),
                 "llm_latency_ms": telemetry["llm_latency_ms"] + sum(
                     item["latency_ms"] for item in focused_usage),
                 "generation_calls": telemetry["generation_calls"] + len(focused_usage),
                 "end_to_end_latency_ms": (perf_counter() - started) * 1000,
                 "status": "grounded_v2_focused" if focused_success and not extractive_fallback
                 else "extractive_fallback_after_qwen" if extractive_fallback else telemetry["status"],
                 "focused_prompt_version": focused_version}
    record("health_v2_request", gateway.name, gateway.version, focused_version,
           telemetry["end_to_end_latency_ms"], telemetry["status"],
           input_tokens=telemetry["input_tokens"], output_tokens=telemetry["output_tokens"],
           top_k=5, filters={"restaurant_key": key},
           embedding_model=retriever.metadata["model_name"],
           retrieval_latency_ms=telemetry["retrieval_latency_ms"],
           llm_latency_ms=telemetry["llm_latency_ms"],
           end_to_end_latency_ms=telemetry["end_to_end_latency_ms"],
           schema_valid=bool(focused_success or strengths or complaints),
           retrieval_version="RAG_V2", retrieval_method=retriever.config.name,
           reranker_model="evidence-rules-v1", candidate_count=len(hits), final_top_k=5)
    verification = {"retrieval_version": "RAG_V2",
                    "retrieval_method": retriever.config.name,
                    "accepted_strengths": len(strengths),
                    "accepted_complaints": len(complaints),
                    "focused_qwen_successes": focused_success,
                    "extractive_fallback_sections": extractive_fallback,
                    "rejected_items": rejected,
                    "warnings_require_alerts": True,
                    "aspect_trends": "empty: no aspect-specific time series"}
    return result, hits, telemetry, verification
