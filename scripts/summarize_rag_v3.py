"""Summarize frozen V3 generation replay without altering raw audit records."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.foody.schema import save_json
from src.rag_v3.service import V3Draft


def failure_class(message):
    text = (message or "").casefold()
    if "frequency" in text or "plural-review" in text:
        return "unsupported_frequency"
    if "off-topic" in text or "aspect" in text:
        return "topic_mismatch"
    if "clause" in text or "lexical support" in text:
        return "unsupported_clause"
    if "citation" in text:
        return "citation_error"
    if "schema" in text or "validation" in text:
        return "schema_failure"
    if "json" in text:
        return "invalid_json"
    if "request" in text or "timeout" in text:
        return "runtime_failure"
    return "other_validation_failure"


def main():
    root = Path("reports/rag_v3")
    comparison = json.loads((root / "generator_comparison.json").read_text(encoding="utf-8"))
    summary = {}
    failures = {}
    for model, result in comparison.items():
        cases = result["cases"]
        attempts = []
        case_results = {}
        for name, case in cases.items():
            telemetry = case["telemetry"]
            response = case["response"]
            audits = case["generation_audit"]
            attempts.extend({"case": name, "attempt": attempt["attempt"],
                             "category": failure_class(attempt.get("failure")),
                             "failure": attempt.get("failure"),
                             "raw_retained": bool(attempt["gateway"])}
                            for attempt in audits if attempt.get("failure"))
            valid_attempts = sum(
                1 for attempt in audits
                if _schema_valid(attempt.get("parsed_output"))
            )
            supported = response["generation_mode"] in ("qwen", "extractive_fallback")
            case_results[name] = {
                "valid_generation": valid_attempts,
                "supported": supported and name != "complaints",
                "supported_but_incomplete": supported and name == "complaints",
                "unsupported": False,
                "correct_insufficient": name in ("price", "rating_drop", "empty_period")
                and response["generation_mode"] == "none",
                "incorrect_insufficient": name in ("complaints", "food", "service")
                and response["generation_mode"] == "none",
                "extractive_fallback": response["generation_mode"] == "extractive_fallback",
                "citation_valid": supported and bool(response["evidence"]),
                "clause_support_valid": telemetry["clause_support_status"] in
                ("passed", "extractive_verified"),
                "input_tokens": telemetry["input_tokens"],
                "output_tokens": telemetry["output_tokens"],
                "LLM_latency_ms": telemetry["llm_latency_ms"],
                "end_to_end_latency_ms": telemetry["end_to_end_latency_ms"],
            }
        counts = {key: sum(value[key] for value in case_results.values())
                  for key in ("valid_generation", "supported", "supported_but_incomplete",
                              "unsupported", "correct_insufficient", "incorrect_insufficient",
                              "extractive_fallback", "citation_valid",
                              "clause_support_valid")}
        active = [value["LLM_latency_ms"] for value in case_results.values()
                  if value["LLM_latency_ms"] is not None]
        summary[model] = {"counts": counts, "cases": case_results,
                          "total_input_tokens": sum(value["input_tokens"] or 0
                                                    for value in case_results.values()),
                          "total_output_tokens": sum(value["output_tokens"] or 0
                                                     for value in case_results.values()),
                          "mean_active_LLM_latency_ms": mean(active) if active else None,
                          "mean_end_to_end_latency_ms": mean(
                              value["end_to_end_latency_ms"]
                              for value in case_results.values()),
                          "resources": result["resources"]}
        failures[model] = {"rejected_attempt_count": len(attempts),
                           "categories": dict(Counter(item["category"] for item in attempts)),
                           "attempts": attempts}
    save_json(root / "generation_metrics.json", summary)
    prior = json.loads((root / "generation_failure_analysis.json").read_text(encoding="utf-8"))
    prior["v3_raw_generation_artifact"] = "reports/rag_v3/generator_comparison.json"
    prior["v3_rejected_attempts"] = failures
    save_json(root / "generation_failure_analysis.json", prior)


def _schema_valid(value):
    try:
        V3Draft.model_validate(value)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
