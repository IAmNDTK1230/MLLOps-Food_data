"""Research-only three-family replay of frozen RAG V3 evidence packets."""

import csv
import json
import os
from hashlib import sha256
from pathlib import Path

import numpy as np

import src.rag_v3.service as service
from scripts.evaluate_rag_v3_generation import ReplayRetriever, load_frozen_packets
from src.llm.gateway import LLMGateway
from src.utils.hashing import file_checksum

OUT = Path("reports/task2")
LOCAL = Path("state/task2_llm_benchmark")
EVIDENCE = Path("reports/rag_v3/evidence_packets.json")
MODELS = (
    {"family": "Qwen", "name": "Qwen2.5-3B-Instruct", "parameter_class": "3B",
     "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF", "revision":
     "cc1e68eea5f05f88f41a6de1fc73110178f23715",
     "file": "models/qwen_v3/qwen2.5-3b-instruct-q4_k_m.gguf", "port": 8194,
     "config": "qwen_v3_3b.yaml", "provenance": "models/qwen_v3/provenance.json"},
    {"family": "Llama", "name": "Llama-3.2-3B-Instruct", "parameter_class": "3B",
     "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF", "revision":
     "5ab33fa94d1d04e903623ae72c95d1696f09f9e8",
     "file": "models/task2_llm/llama/Llama-3.2-3B-Instruct-Q4_K_M.gguf", "port": 8195},
    {"family": "DeepSeek", "name": "DeepSeek-R1-Distill-Qwen-1.5B", "parameter_class": "1.5B",
     "repo": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "revision":
     "9cc28b17e86fa2415fcb070f8ee5ec27c965aa61",
     "file": "models/task2_llm/deepseek/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
     "port": 8196},
)


def paths_for(spec, server):
    if "config" in spec:
        return spec["config"], spec["provenance"]
    path = Path(spec["file"])
    config = {
        "model_repo": spec["repo"], "model_revision": spec["revision"],
        "model_file": path.name, "model_dir": str(path.parent),
        "quantization": "Q4_K_M", "runtime": "llama.cpp", "port": spec["port"],
        "gpu_layers": 99, "threads": 8, "context_size": 2048,
        "temperature": 0.0, "seed": 42,
        "server_log_path": f"logs/task2_{spec['family'].lower()}_server.log",
    }
    provenance = {"model": spec["repo"], "model_sha256": file_checksum(path),
                  "server_executable": server}
    config_path = LOCAL / f"{spec['family'].lower()}_config.json"
    provenance_path = LOCAL / f"{spec['family'].lower()}_provenance.json"
    # JSON is a YAML subset and accepted by the existing gateway.
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return str(config_path), str(provenance_path)


def analyze(spec, cases, runtime):
    accepted_cases = ("complaints", "food", "service")
    insufficient_cases = ("price", "rating_drop", "empty_period")
    generation_latency = [cases[name]["telemetry"]["llm_latency_ms"]
                          for name in accepted_cases
                          if cases[name]["telemetry"]["llm_latency_ms"] is not None]
    known_input = [cases[name]["telemetry"]["input_tokens"]
                   for name in accepted_cases
                   if cases[name]["telemetry"]["input_tokens"] is not None]
    known_output = [cases[name]["telemetry"]["output_tokens"]
                    for name in accepted_cases
                    if cases[name]["telemetry"]["output_tokens"] is not None]
    answerable_wall = [cases[name]["telemetry"]["end_to_end_latency_ms"]
                       for name in accepted_cases]
    supported = sum(cases[name]["telemetry"]["status"] in
                    ("grounded", "supported_extractive_fallback")
                    for name in ("food", "service"))
    incomplete = int(cases["complaints"]["telemetry"]["status"] in
                     ("grounded", "supported_extractive_fallback"))
    accepted = [case for case in cases.values() if case["telemetry"]["status"] in
                ("grounded", "supported_extractive_fallback")]
    unsupported = sum(case["telemetry"]["clause_support_status"] not in
                      ("passed", "extractive_verified") for case in accepted)
    return {
        "family": spec["family"], "model": spec["name"],
        "parameter_class": spec["parameter_class"], "repo": spec["repo"],
        "revision": spec["revision"], "sha256": file_checksum(Path(spec["file"])),
        "quantization": "Q4_K_M", "runtime": "llama.cpp b11429",
        "model_size_bytes": Path(spec["file"]).stat().st_size,
        "model_size_mb": Path(spec["file"]).stat().st_size / 1_000_000,
        "valid_schema_cases": sum(cases[name]["telemetry"]["schema_success"]
                                  for name in accepted_cases),
        "schema_applicable_cases": len(accepted_cases),
        "supported_complete": supported,
        "supported_incomplete": incomplete,
        "unsupported_accepted_validator_detected": unsupported,
        "correct_insufficient": sum(cases[name]["telemetry"]["status"] in
                                    ("correct_insufficient", "insufficient_evidence")
                                    for name in insufficient_cases),
        "incorrect_insufficient": sum(cases[name]["telemetry"]["status"] in
                                      ("correct_insufficient", "insufficient_evidence")
                                      for name in accepted_cases),
        "fallback_count": sum(case["telemetry"]["generation_mode"] ==
                              "extractive_fallback" for case in cases.values()),
        "input_tokens": sum(known_input) if known_input else None,
        "output_tokens": sum(known_output) if known_output else None,
        "unknown_token_usage_cases": len(accepted_cases) - len(known_input),
        "mean_generation_latency_ms": float(np.mean(generation_latency))
        if generation_latency else None,
        "p95_generation_latency_ms": float(np.percentile(generation_latency, 95))
        if generation_latency else None,
        "mean_answerable_end_to_end_latency_ms": float(np.mean(answerable_wall)),
        "p95_answerable_end_to_end_latency_ms": float(np.percentile(answerable_wall, 95)),
        "peak_server_rss_bytes": runtime.peak_rss if runtime else None,
        "peak_system_gpu_memory_mib": max((row[1] for row in runtime.gpu_samples),
                                           default=None) if runtime else None,
        "gpu_samples": len(runtime.gpu_samples) if runtime else 0,
        "case_count": len(cases), "status": "MEASURED",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    os.environ.pop("QWEN_BASE_URL", None)  # Always use each pinned local model.
    prepared = load_frozen_packets(EVIDENCE)
    server = str(Path(".qwen-runtime/b11429/llama-server.exe").resolve())
    provenance = json.loads(Path("models/qwen_v3/provenance.json").read_text(encoding="utf-8"))
    if Path(provenance["server_executable"]).resolve() != Path(server):
        raise ValueError("Research benchmark must use accepted llama.cpp executable")
    # The replay must not add events to accepted operational LLMOps state.
    service.record = lambda *args, **kwargs: None
    retriever = ReplayRetriever()
    rows = []
    for spec in MODELS:
        print("BENCHMARK", spec["family"], flush=True)
        config_path, provenance_path = paths_for(spec, server)
        cases = {}
        with LLMGateway(config_path=config_path, provenance_path=provenance_path,
                        monitor=False) as gateway:
            for name, (request, state) in prepared.items():
                response, _, telemetry, _, _ = service.answer_question_v3(
                    retriever, gateway, request, prepared=state
                )
                cases[name] = {"telemetry": telemetry,
                               "response_sha256": sha256(
                                   response.answer.encode("utf-8")).hexdigest()}
                print(spec["family"], name, telemetry["status"], flush=True)
            runtime = gateway.runtime
            row = analyze(spec, cases, runtime)
        (LOCAL / f"{spec['family'].lower()}_case_metrics.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(row)
        with (OUT / "llm_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "llm_benchmark_provenance.json").write_text(json.dumps({
        "evidence_packets_sha256": file_checksum(EVIDENCE), "prompt": "prompts/rag/rag_v3.txt",
        "prompt_sha256": file_checksum(Path("prompts/rag/rag_v3.txt")),
        "temperature": 0, "seed": 42, "max_new_tokens": 128,
        "runtime": "llama.cpp b11429", "runtime_executable": server,
        "gpu_layers_requested": 99, "threads": 8, "context_size": 2048,
        "retrieval_reexecuted": False, "accepted_demo_modified": False,
    }, indent=2), encoding="utf-8")
    print("Saved", OUT / "llm_comparison.csv")


if __name__ == "__main__":
    main()
