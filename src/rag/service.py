"""Grounded answer and restaurant health report service."""

import json
import re
from pathlib import Path
from time import perf_counter

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.foody.schema import digest, restaurant_key
from src.rag.monitoring import record
from src.rag.schemas import Evidence, GroundedAnswer, HealthReport, RetrievalRequest

INSUFFICIENT = "Không đủ bằng chứng từ các đánh giá được truy xuất."


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    evidence_refs: list[str] = Field(default_factory=list)


def resolve_restaurant(name_or_key, summary_path="data/foody/mart/restaurant_summary.parquet"):
    summary = pd.read_parquet(summary_path)
    key = name_or_key if re.fullmatch(r"[0-9a-f]{64}", name_or_key) else restaurant_key(name_or_key)
    matching = summary.loc[summary.restaurant_key == key]
    if matching.empty:
        raise ValueError(f"Restaurant not found: {name_or_key}")
    return key, matching.iloc[0]


def prompt(path):
    file = Path(path)
    return file.read_text(encoding="utf-8"), f"{file.stem}:{digest(file)[:12]}"


def context(hits, max_chars=400):
    return [{"ref": f"R{index}", "review_id": hit.review_id,
             "restaurant_name": hit.restaurant_name,
             "review_text": hit.text[:max_chars],
             "review_timestamp": hit.metadata.get("review_timestamp"),
             "raw_rating": hit.metadata.get("raw_rating")}
            for index, hit in enumerate(hits, 1)]


def check_refs(refs, answer, hits):
    valid = {f"R{i}": hit for i, hit in enumerate(hits, 1)}
    if not refs or len(refs) != len(set(refs)) or any(ref not in valid for ref in refs):
        raise ValueError("Answer cites unknown, duplicate or no review IDs")
    mentioned = set(re.findall(r"\[R\d+\]", answer))
    if mentioned != {f"[{ref}]" for ref in refs}:
        raise ValueError("Answer citations do not match its evidence list")
    return [Evidence(review_id=valid[ref].review_id,
                     restaurant_name=valid[ref].restaurant_name,
                     score=valid[ref].score) for ref in refs]


def insufficient(model, version):
    return GroundedAnswer(answer=INSUFFICIENT, evidence=[], model=model, prompt_version=version)


def answer_question(retriever, gateway, request, min_score=0.25):
    if not isinstance(request, RetrievalRequest):
        request = RetrievalRequest.model_validate(request)
    start = perf_counter()
    instruction, version = prompt("prompts/rag/rag_v1.txt")
    hits, retrieval_ms = retriever.search(request)
    strong = [hit for hit in hits if hit.score >= min_score]
    if not strong:
        response = insufficient(gateway.name, version)
        telemetry = {"retrieval_latency_ms": retrieval_ms, "llm_latency_ms": None,
                     "input_tokens": 0, "output_tokens": 0,
                     "schema_valid": None, "status": "insufficient_retrieval"}
    else:
        data = {"question": request.query, "reviews": context(strong[:4])}
        usage = None
        try:
            draft, usage = gateway.generate(system=instruction,
                                            user=json.dumps(data, ensure_ascii=False),
                                            response_schema=DraftAnswer.model_json_schema(),
                                            prompt_version=version, max_new_tokens=256)
            draft = DraftAnswer.model_validate(draft)
            evidence = check_refs(draft.evidence_refs, draft.answer, strong[:4])
            response = GroundedAnswer(answer=draft.answer, evidence=evidence,
                                      model=gateway.name, prompt_version=version)
            telemetry = {"retrieval_latency_ms": retrieval_ms,
                         "llm_latency_ms": usage["latency_ms"],
                         "input_tokens": usage["input_tokens"],
                         "output_tokens": usage["output_tokens"],
                         "schema_valid": True, "status": "grounded"}
        except (ValueError, KeyError, TypeError):
            response = insufficient(gateway.name, version)
            telemetry = {"retrieval_latency_ms": retrieval_ms,
                         "llm_latency_ms": usage["latency_ms"] if usage else None,
                         "input_tokens": usage["input_tokens"] if usage else None,
                         "output_tokens": usage["output_tokens"] if usage else None,
                         "schema_valid": False,
                         "status": "invalid_generation_or_evidence"}
    telemetry["end_to_end_latency_ms"] = (perf_counter() - start) * 1000
    telemetry["prompt_version"] = version
    telemetry["model_version"] = gateway.version
    telemetry["generation_calls"] = 1 if strong else 0
    record("rag_request", gateway.name, gateway.version, version,
           telemetry["end_to_end_latency_ms"], telemetry["status"],
           input_tokens=telemetry["input_tokens"], output_tokens=telemetry["output_tokens"],
           top_k=request.top_k,
           filters=request.model_dump(exclude={"query", "top_k"}, exclude_none=True),
           embedding_model=getattr(retriever, "metadata", {}).get("model_name"),
           retrieval_latency_ms=retrieval_ms, llm_latency_ms=telemetry["llm_latency_ms"],
           end_to_end_latency_ms=telemetry["end_to_end_latency_ms"],
           schema_valid=telemetry["schema_valid"])
    return response, hits, telemetry


def health_report(retriever, gateway, name_or_key):
    start = perf_counter()
    key, summary = resolve_restaurant(name_or_key)
    positive, positive_ms = retriever.search(RetrievalRequest(
        query="khách khen món ăn, phục vụ, giá và không gian",
        restaurant_key=key, sentiment="positive", top_k=2,
    ))
    negative, negative_ms = retriever.search(RetrievalRequest(
        query="khách phàn nàn về món ăn, thái độ phục vụ, thời gian chờ, giá",
        restaurant_key=key, sentiment="negative", top_k=3,
    ))
    hits = list({hit.review_id: hit for hit in [*positive, *negative]}.values())
    retrieval_ms = positive_ms + negative_ms
    if not hits:
        raise ValueError("No review evidence for this restaurant")
    warning = pd.read_parquet("data/foody/mart/alerts.parquet")
    warning = warning.loc[warning.restaurant_key == key]
    instruction, version = prompt("prompts/restaurant_report/health_v1.txt")
    strengths, complaints, evidence_ids, usages = [], [], [], []
    generation_calls = 0
    for question, subset, output in (
        ("Khách khen điều gì?", positive, strengths),
        ("Khách phàn nàn điều gì?", negative, complaints),
    ):
        if not subset:
            continue
        refs = [f"R{hits.index(hit) + 1}" for hit in subset]
        payload = {"question": question,
                   "reviews": [item for item in context(hits, max_chars=240)
                               if item["ref"] in refs]}
        try:
            generation_calls += 1
            draft, usage = gateway.generate(
                system=instruction, user=json.dumps(payload, ensure_ascii=False),
                response_schema=DraftAnswer.model_json_schema(), prompt_version=version,
                max_new_tokens=160, component="health_generation",
            )
            usages.append(usage)
            draft = DraftAnswer.model_validate(draft)
            evidence = check_refs(draft.evidence_refs, draft.answer, hits)
        except (ValueError, KeyError, TypeError):
            continue
        output.append(draft.answer)
        evidence_ids.extend(item.review_id for item in evidence)
    # The mart has monthly rating/sentiment summaries, but no reliable
    # aspect-specific time series for this report.
    trends = []
    warnings = []
    if not warning.empty:
        warnings.append(f"Có {len(warning)} cảnh báo quy tắc lịch sử; không phản ánh "
                        "tình trạng hiện tại [M3].")
    recommendations = []
    if complaints:
        ref = re.search(r"\[R\d+\]", complaints[0]).group()
        recommendations.append(f"Xác minh thêm phản ánh được dẫn ở {ref} trước khi hành động.")
    result = HealthReport(
        overall_summary=f"Dữ liệu có {int(summary.review_count)} đánh giá; "
                        f"điểm trung bình {float(summary.average_rating):.2f} [M1].",
        strengths=strengths, main_complaints=complaints, aspect_trends=trends,
        warnings=warnings, recommendations=recommendations,
        evidence_review_ids=list(dict.fromkeys(evidence_ids)),
    )
    telemetry = {"retrieval_latency_ms": retrieval_ms,
                 "llm_latency_ms": sum(item["latency_ms"] for item in usages),
                 "input_tokens": sum(item["input_tokens"] or 0 for item in usages),
                 "output_tokens": sum(item["output_tokens"] or 0 for item in usages),
                 "status": "grounded" if strengths or complaints else
                 "metrics_only_insufficient_generation",
                 "generation_calls": generation_calls,
                 "prompt_version": version,
                 "model_version": gateway.version,
                 "end_to_end_latency_ms": (perf_counter() - start) * 1000}
    record("health_request", gateway.name, gateway.version, version,
           telemetry["end_to_end_latency_ms"], telemetry["status"],
           input_tokens=telemetry["input_tokens"], output_tokens=telemetry["output_tokens"],
           top_k=5, filters={"restaurant_key": key},
           embedding_model=getattr(retriever, "metadata", {}).get("model_name"),
           retrieval_latency_ms=retrieval_ms, llm_latency_ms=telemetry["llm_latency_ms"],
           end_to_end_latency_ms=telemetry["end_to_end_latency_ms"],
           schema_valid=bool(strengths or complaints),
           retrieval_version="RAG_V2" if hasattr(retriever, "config") else "RAG_V1_DENSE",
           retrieval_method=getattr(getattr(retriever, "config", None), "name",
                                    "RAG_V1_DENSE"),
           reranker_model="evidence-rules-v1" if getattr(
               getattr(retriever, "config", None), "evidence_rerank", False) else None,
           candidate_count=len(hits), final_top_k=5)
    return result, hits, telemetry
