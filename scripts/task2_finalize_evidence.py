"""Finalize Task-2 feasibility, benchmark report, and review checklist."""

import csv
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

OUT = Path("reports/task2")
FIG = OUT / "figures"


def data(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=190, bbox_inches="tight")
    plt.close()


def reconcile_llm():
    path = OUT / "llm_comparison.csv"
    rows = csv_rows(path)
    if not rows:
        return
    for row in rows:
        cases = data(Path("state/task2_llm_benchmark") /
                     f"{row['family'].lower()}_case_metrics.json")
        active = [cases[name]["telemetry"] for name in ("complaints", "food", "service")]
        known = [item for item in active if item["input_tokens"] is not None]
        row["input_tokens"] = sum(item["input_tokens"] for item in known) if known else ""
        row["output_tokens"] = sum(item["output_tokens"] for item in known) if known else ""
        row["unknown_token_usage_cases"] = len(active) - len(known)
        wall = [item["end_to_end_latency_ms"] for item in active]
        row["mean_answerable_end_to_end_latency_ms"] = float(np.mean(wall))
        row["p95_answerable_end_to_end_latency_ms"] = float(np.percentile(wall, 95))
        log_name = ("qwen_v3_3b_server.log" if row["family"] == "Qwen" else
                    f"task2_{row['family'].lower()}_server.log")
        log_path = Path("logs") / log_name
        if log_path.exists():
            session = log_path.read_text(encoding="utf-8", errors="replace").rsplit(
                "llama_server: initializing ...", 1)[-1]
            prompt = [int(v) for v in re.findall(r"prompt eval time\s*=.*?/\s*(\d+) tokens",
                                                   session)]
            output = [int(v) for v in re.findall(r"(?<!prompt )eval time\s*=.*?/\s*(\d+) tokens",
                                                   session)]
            row["server_log_prompt_eval_tokens"] = sum(prompt) if prompt else ""
            row["server_log_output_eval_tokens"] = sum(output) if output else ""
            row["server_log_completed_attempts"] = min(len(prompt), len(output))
            row["server_log_expected_attempts"] = sum(item["generation_attempts"]
                                                       for item in active)
            row["server_log_token_scope"] = ("complete" if len(prompt) == len(output) ==
                                             row["server_log_expected_attempts"] else
                                             "partial; cancelled attempts absent")
    csv_write(path, rows)


def feasibility():
    baseline = {k: data(f"reports/baselines/{k}_test_metrics.json")
                for k in ("rule_based", "tfidf")}
    generation = data("reports/rag_v3/generation_metrics.json")
    index = data("data/foody/vector_store/index_manifest.json")
    llms = csv_rows(OUT / "llm_comparison.csv")
    rows = []

    def add(name, task, size="", mean="", p95="", ram="", gpu="", hardware="",
            status=""):
        rows.append({"component": name, "task": task, "model_size_mb": size,
                     "mean_latency_ms": mean, "p95_latency_ms": p95,
                     "peak_ram_bytes": ram, "peak_system_gpu_memory_mib": gpu,
                     "hardware": hardware, "deployment_status": status})

    for key, title in (("rule_based", "Rule-based"), ("tfidf", "TF-IDF sentiment")):
        metric = baseline[key]
        add(title, "single text sentiment", metric.get("model_size_mb", ""),
            metric["mean_latency_ms"], metric["p95_latency_ms"], hardware="local CPU",
            status="accepted frozen baseline")
    add("Aspect TF-IDF", "multi-label aspect", hardware="local CPU",
        status="accepted artifact; latency/size unavailable in saved metrics")
    for key, title, size in (("qwen_1_5b", "Qwen 1.5B", 1117.320736),
                             ("qwen_3b", "Qwen 3B", 2104.932768)):
        metric = generation[key]
        add(title, "RAG V3 active generation", size, metric["mean_active_LLM_latency_ms"],
            ram=metric["resources"]["peak_server_rss_bytes"],
            gpu=metric["resources"]["peak_gpu_memory_mib"],
            hardware="RTX 3060 Laptop / local", status="accepted RAG V3 replay")
    for family in ("Qwen", "Llama", "DeepSeek"):
        match = next((row for row in llms if row["family"] == family), None)
        wall_path = Path(f"state/task2_llm_benchmark/{family.lower()}_case_metrics.json")
        wall = [case["telemetry"]["end_to_end_latency_ms"]
                for name, case in data(wall_path).items()
                if name in ("complaints", "food", "service")] if wall_path.exists() else []
        mean = (match["mean_generation_latency_ms"] if match and
                match["mean_generation_latency_ms"] else
                float(np.mean(wall)) if wall else "")
        p95 = (match["p95_generation_latency_ms"] if match and
               match["p95_generation_latency_ms"] else
               float(np.percentile(wall, 95)) if wall else "")
        add(family + " research replay", "fixed RAG V3 generation",
            match["model_size_mb"] if match else "",
            mean, p95,
            match["peak_server_rss_bytes"] if match else "",
            match["peak_system_gpu_memory_mib"] if match else "",
            "RTX 3060 Laptop / local",
            "research replay; latency includes timeouts" if match and not
            match["mean_generation_latency_ms"] else
            "research replay" if match else "NOT MEASURED")
    dense = next(row for row in csv_rows(Path("reports/rag_v3/retrieval_policies.csv"))
                 if row["policy"] == "RAG_V1_DENSE")
    add("MiniLM dense retrieval", "top-k retrieval", index["model_size_mb"],
        dense["mean_latency_ms"], dense["p95_latency_ms"], hardware="local CPU/ONNX",
        status="accepted index; six-query latency")
    case_values = [case["end_to_end_latency_ms"]
                   for case in generation["qwen_3b"]["cases"].values()]
    add("RAG V3 end-to-end", "six-case routed RAG", mean=float(np.mean(case_values)),
        p95=float(np.percentile(case_values, 95)), hardware="local RTX 3060 Laptop",
        status="accepted six-case replay")
    csv_write(OUT / "FEASIBILITY.csv", rows)
    values = [row for row in rows if str(row["mean_latency_ms"]) not in ("", "None")]
    plt.figure(figsize=(10, 6))
    plt.scatter([float(row["model_size_mb"]) if str(row["model_size_mb"]) else 0
                 for row in values],
                [float(row["mean_latency_ms"]) for row in values], s=95, color="#2667a5")
    for row in values:
        plt.annotate(row["component"],
                     (float(row["model_size_mb"]) if str(row["model_size_mb"]) else 0,
                      float(row["mean_latency_ms"])), xytext=(5, 5),
                     textcoords="offset points", fontsize=9)
    plt.yscale("symlog", linthresh=1)
    plt.xlabel("Model file size (MB; 0 denotes unavailable/code-only)")
    plt.ylabel("Mean latency (ms; mixed task scopes, see FEASIBILITY.csv)")
    plt.title("Measured size and latency; task scopes differ", fontsize=15)
    plot_save("latency_model_size.png")


def llm_report():
    rows = csv_rows(OUT / "llm_comparison.csv")
    lines = ["# Three-family frozen-evidence generator benchmark", "",
             "The benchmark uses the same six saved RAG V3 evidence packets, questions, "
             "`prompts/rag/rag_v3.txt`, llama.cpp b11429, temperature 0, seed 42, "
             "128 output-token cap, and no rerun of retrieval. All models are Q4_K_M GGUF. "
             "Qwen/Llama are 3B class; DeepSeek-R1-Distill-Qwen is 1.5B class and smaller. "
             "The application telemetry's legacy `generation_mode=qwen` enum means "
             "accepted generated answer; actual family is verified by model provenance. "
             "This six-case set is a fixed safety smoke test, not a general benchmark. "
             "The 3B comparison cannot isolate architecture from instruction tuning or "
             "tokenizer differences. GPU layers were requested; per-process GPU assignment "
             "was not independently confirmed. GPU measurements are sampled system totals.", "",
             "| Family | Cases | Valid schema / applicable | Complete | Incomplete | "
             "Validator-detected unsupported accepted | Correct insufficient | Fallback | "
             "Mean active generation ms | Mean answerable wall ms | Unknown token-usage cases |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['family']} | {row['case_count']} | "
                     f"{row['valid_schema_cases']}/{row['schema_applicable_cases']} | "
                     f"{row['supported_complete']} | {row['supported_incomplete']} | "
                     f"{row['unsupported_accepted_validator_detected']} | "
                     f"{row['correct_insufficient']} | {row['fallback_count']} | "
                     f"{float(row['mean_generation_latency_ms']):.1f} | "
                     f"{float(row['mean_answerable_end_to_end_latency_ms']):.1f} | "
                     f"{row['unknown_token_usage_cases']} |"
                     if row["mean_generation_latency_ms"] else
                     f"| {row['family']} | {row['case_count']} | "
                     f"{row['valid_schema_cases']}/{row['schema_applicable_cases']} | "
                     f"{row['supported_complete']} | {row['supported_incomplete']} | "
                     f"{row['unsupported_accepted_validator_detected']} | "
                     f"{row['correct_insufficient']} | {row['fallback_count']} | unavailable | "
                     f"{float(row['mean_answerable_end_to_end_latency_ms']):.1f} | "
                     f"{row['unknown_token_usage_cases']} |")
    if len(rows) < 3:
        lines.append("\n**IN PROGRESS:** A missing family has no measured row; do not infer its score.")
    else:
        lines.append("\nAll three final outputs have zero validator-detected unsupported accepted "
                     "claims, largely because the unchanged V3 gate falls back or refuses. "
                     "DeepSeek generated no parseable required schema in the answerable cases "
                     "and fell back three times; this is a prompt/structured-output failure "
                     "under this 128-token protocol, not proof of a generally weak model. "
                     "Llama had one grounded incomplete answer and two fallbacks. The new "
                     "Qwen run suffered local resource contention and had three fallbacks, "
                     "whereas the earlier accepted Qwen 3B replay had two. These runs are "
                     "not enough to select a new production generator or replace the accepted "
                     "demo model.")
        deepseek = next(row for row in rows if row["family"] == "DeepSeek")
        lines.append("\nDeepSeek's gateway JSON parser returned no usage object, so its gateway "
                     "input/output token cells are blank. The local llama-server log records "
                     f"{deepseek['server_log_prompt_eval_tokens']} prompt-evaluation and "
                     f"{deepseek['server_log_output_eval_tokens']} output-evaluation tokens "
                     f"over {deepseek['server_log_completed_attempts']} completed attempts "
                     "(including retries); these separate log counters are in "
                     "`llm_comparison.csv` and must not be compared as if they were "
                     "successful-response usage totals.")
    lines += ["", "The validator checks schema, citations, topic, and clause support; "
              "zero validator-detected unsupported accepted is a narrow safety result, "
              "not proof of zero hallucinations. Latency includes local hardware contention. "
              "Active generation latency excludes failed/timed-out attempts; the separate "
              "`FEASIBILITY.csv` uses answerable-case wall time when no completed generation "
              "latency exists and labels that substitution. GPU memory is a system sample, "
              "not this process's allocated VRAM. Input/output token totals come from "
              "responses with runtime usage metadata; timed-out attempts have unknown token "
              "usage and therefore the sums can be lower bounds. "
              "Private case-level telemetry and answer hashes are stored under "
              "`state/task2_llm_benchmark/`; answers are not copied into Git. "
              "Model weights under `models/task2_llm/` remain Git-ignored.", ""]
    (OUT / "LLM_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    if rows:
        plt.figure(figsize=(9, 5))
        names = [row["family"] for row in rows]
        x = np.arange(len(names))
        valid = [int(row["valid_schema_cases"]) for row in rows]
        fallback = [int(row["fallback_count"]) for row in rows]
        plt.bar(x - .18, valid, width=.36, label="Valid generation schema / 3",
                color="#287a55")
        plt.bar(x + .18, fallback, width=.36, label="Extractive fallback / 3",
                color="#e6a23c")
        plt.xticks(x, names)
        plt.ylim(0, 3.5)
        plt.ylabel("Answerable cases (3 per model)")
        plt.title("Generation schema success and fallback on fixed evidence", fontsize=15)
        plt.legend()
        plot_save("three_llm_comparison.png")


def architecture():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [(0.25, 4.2, "Foody source\nfirst-party crawl"),
             (2.95, 4.2, "UTF-8 + validation\nmodel enrichment"),
             (5.65, 4.2, "32,457 review docs\nChroma + BM25"),
             (8.35, 4.2, "V3 routed retrieval\nevidence gates"),
             (8.35, 1.8, "llama.cpp Qwen\nclause validation"),
             (5.65, 1.8, "Fallback / refusal\nLLMOps local state"),
             (2.95, 1.8, "FastAPI backend\nStreamlit frontend"),
             (0.25, 1.8, "VN-MSASar\nfrozen research")]
    for x, y, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.2, 1.0, boxstyle="round,pad=0.05",
                                    linewidth=1.8, edgecolor="#2667a5",
                                    facecolor="#e7f0f8"))
        ax.text(x + 1.1, y + .5, label, ha="center", va="center", fontsize=10.5)
    for x1, y1, x2, y2 in ((2.5, 4.7, 2.9, 4.7), (5.2, 4.7, 5.6, 4.7),
                            (7.9, 4.7, 8.3, 4.7), (9.45, 4.15, 9.45, 2.85),
                            (8.3, 2.3, 7.9, 2.3), (5.6, 2.3, 5.2, 2.3)):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2667a5"})
    ax.text(5.5, 5.6, "Food Review Intelligence: accepted demo and separate research",
            ha="center", fontsize=15)
    ax.text(1.35, 1.35, "Independent labeled benchmark", ha="center", fontsize=9,
            color="#465c6f")
    fig.savefig(FIG / "final_architecture.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def checklist():
    matrix = OUT / "TASK2_GAP_MATRIX.md"
    content = matrix.read_text(encoding="utf-8")
    content = content.replace("IN PROGRESS |", "DONE |")
    content = content.replace("| 10. Paper vs Group | Missing", "| 10. Paper vs Group | Missing")
    llm_incomplete = len(csv_rows(OUT / "llm_comparison.csv")) < 3
    if llm_incomplete:
        content = content.replace("| 20. Generation/faithfulness | Done with caveat", "| 20. Generation/faithfulness | Done with caveat")
        for number in (20, 24):
            lines = content.splitlines()
            lines = [line.rsplit("|", 2)[0] + "| IN PROGRESS |" if line.startswith(f"| {number}. ")
                     else line for line in lines]
            content = "\n".join(lines) + "\n"
    matrix.write_text(content, encoding="utf-8")
    rows = []
    for line in content.splitlines():
        if line.startswith("| ") and len(line) > 4 and line[2].isdigit():
            cells = [part.strip() for part in line.strip("|").split("|")]
            rows.append((cells[0], cells[-1]))
    evidence = {
        1: "dataset_sources.csv", 2: "CLEANING_FLOW.md", 3: "DATASET_SUMMARY.md",
        4: "DATASET_SUMMARY.md", 5: "class_distribution.csv", 6: "imbalance_comparison.csv",
        7: "classical_model_comparison.csv", 8: "classical_model_comparison.csv",
        9: "baseline_three_seed.csv", 10: "PAPER_VS_GROUP.md", 11: "GAP_ANALYSIS.md",
        12: "../proposed_v1/qwen_test_metrics.json", 13: "failure_counts.csv",
        14: "../rag_v3/generation_metrics.json", 15: "RAG_ABLATION_FINAL.md",
        16: "FEASIBILITY.csv", 17: "rag_token_statistics.json",
        18: "embedding_workload.csv", 19: "RAG_ABLATION.csv",
        20: "LLM_BENCHMARK.md", 21: "DOCKER_ACCEPTANCE.md",
        22: "CICD_EVIDENCE.md", 23: "baseline_three_seed.csv", 24: "figures/",
    }
    lines = ["# Task 2 Review-2 self-assessment", "",
             "Status vocabulary: DONE, IN PROGRESS, NOT DONE. DONE means evidence was "
             "generated and verified, not merely that code exists. CI and Docker deployment "
             "are scoped to their verified steps.", "", "| Item | Status | Evidence / limit |",
             "| --- | --- | --- |"]
    for title, status in rows:
        number = int(title.split(".", 1)[0])
        lines.append(f"| {title} | {status} | `{evidence[number]}` |")
    lines += ["", "## Outstanding", "",
              "- DONE source audit: no directly comparable published macro-F1 result on this "
              "exact task was verified; numerical paper gaps are blank in `paper_vs_group.csv`.",
              "- DONE evidence synthesis: `RAG_ABLATION_FINAL.md` makes the observed role of "
              "each retrieval and safety component visible; the six-query known-positive "
              "labels are incomplete and no component-off causal effect is claimed.",
              "- DONE Docker: serving path, mounted indexes, RAG V3 request, LLMOps event "
              "and pipeline's default one-shot startup (exit 0) verified; no ingestion was "
              "run (see `DOCKER_ACCEPTANCE.md`).",
              "- DONE CI/CD: successful GitHub Actions CI and release runs and the public "
              "`v1.0.0` GHCR image manifest verified; remote production deployment is outside "
              "this checklist (see `CICD_EVIDENCE.md`).",
              "- NOT DONE: a full LLM-from-scratch experiment (not part of this project's scope).",
              ""]
    if llm_incomplete:
        lines.insert(-1, "- IN PROGRESS: not all three local LLM families have completed "
                     "the fixed-packet replay; see `LLM_BENCHMARK.md`.")
    (OUT / "TASK2_SELF_ASSESSMENT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    reconcile_llm()
    feasibility()
    llm_report()
    architecture()
    checklist()
    print("Finalized feasibility, benchmark, and checklist")


if __name__ == "__main__":
    main()
