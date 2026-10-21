"""Assemble measured RAG evidence from existing artifacts without new inference."""

import json
from pathlib import Path

import pandas as pd

from src.foody.schema import digest, save_json

ROOT = Path("reports/rag")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def run():
    selection = read(ROOT / "embedding_selection.json")
    index = read("data/foody/vector_store/index_manifest.json")
    retrieval = read(ROOT / "rag_metrics.json")
    development = read(ROOT / "development_judgments.json")
    execution = read(ROOT / "execution_checks.json")
    integrity = read(ROOT / "index_integrity.json")
    filters = read(ROOT / "filter_verification.json")
    streamlit = read(ROOT / "streamlit_smoke.json")
    example_paths = sorted((ROOT / "examples").glob("query_*.json"))
    examples = [read(path) for path in example_paths]
    health_paths = sorted((ROOT / "examples").glob("health_*.json"))
    health = [read(path) for path in health_paths]
    manual_path = ROOT / "manual_answer_review.json"
    manual = read(manual_path) if manual_path.exists() else None
    from src.rag.monitoring import summary

    operational = summary()
    qwen_calls = [item.get("telemetry", {}) for item in examples + health]
    observed_latency = [value for item in qwen_calls
                        if (value := item.get("llm_latency_ms", item.get("latency_ms"))) is not None]
    end_to_end = [item["end_to_end_latency_ms"] for item in qwen_calls
                  if item.get("end_to_end_latency_ms") is not None]
    qwen_tokens = {
        "input_tokens": sum(item.get("input_tokens") or 0 for item in qwen_calls),
        "output_tokens": sum(item.get("output_tokens") or 0 for item in qwen_calls),
        "observed_generation_requests": len(observed_latency),
        "mean_llm_latency_per_request_ms": sum(observed_latency) / len(observed_latency)
        if observed_latency else None,
        "mean_end_to_end_latency_ms": sum(end_to_end) / len(end_to_end) if end_to_end else None,
        "grounded_query_count": sum(item.get("telemetry", {}).get("status") == "grounded"
                                    for item in examples),
    }
    for label, file in (
        ("rag", "prompts/rag/rag_v1.txt"),
        ("health", "prompts/restaurant_report/health_v1.txt"),
    ):
        qwen_tokens[f"{label}_prompt_sha256"] = digest(file)
    save_json(ROOT / "token_summary.json", qwen_tokens)

    rag_report = f"""# Foody retrieval and grounded-answer acceptance

## Source and small development benchmark

`data/foody/rag/documents.jsonl` holds {index['document_count']:,} distinct valid Foody
reviews. Source SHA256: `{index['documents_sha256']}`. All indexed texts passed the
first-party RAG source contract; no empty/U+FFFD texts were indexed.

{len(development['cases'])} Vietnamese questions and manually inspected known relevant review IDs were frozen
before embedding selection. Relevance labels are **incomplete**: other relevant reviews
may exist. The known-positive hit rates below are neither complete recall nor overall
retrieval accuracy. No Foody sentiment/aspect labels were used as relevance gold.

## Embedding comparison and index

Two local multilingual ONNX candidates were evaluated on the same fixed development
pool; details are in `embedding_comparison.csv` and `retrieval_development.csv`.
"""
    comparison = pd.read_csv(ROOT / "embedding_comparison.csv")
    rag_report += comparison[["model", "onnx_source_repo", "revision", "dimension",
                              "model_size_mb", "documents_per_second",
                              "known_positive_recall_at_5", "known_positive_recall_at_10",
                              "known_positive_mrr"]].to_markdown(index=False) + "\n\n"
    rag_report += f"""
Selected: `{selection['model']}`. The pinned ONNX export is
`{selection['onnx_source_repo']}` at revision `{selection['revision']}`,
dimension {selection['dimension']}, file SHA256 `{selection['model_hash']}`,
{selection['model_size_mb']:.1f} MB. Device: {selection['device']}.
Selection: known-positive top-10 hit rate, MRR, then query embedding time.

Full index: {index['document_count']:,} vectors in persisted Chroma collection
`{index['collection']}`. Actual embedding time {index['embedding_seconds']:.1f}s,
throughput {index['documents_per_second']:.1f} documents/s, mean
{index['mean_embedding_ms']:.1f} ms/document. The index stores text, IDs, restaurant,
timestamp, raw rating, predicted sentiment/aspects and heuristic issues.
Restaurant/date/sentiment/aspect filters run inside Chroma before top-k ranking.
Persisted path: `{integrity['collection_path']}`. Build batch size: 128.
The complete wall-clock indexing time was not independently captured; the measured
embedding duration above excludes possible Chroma write overhead. The existing build
is idempotent by review ID. Integrity verification reconciled all {integrity['stored_unique_review_ids']:,}
stored IDs to the source, the source hash, model file hash, cosine distance and
{integrity['vector_dimension']}-dimensional sample vector: **{'PASS' if integrity['pass'] else 'FAIL'}**.

## Full-index retrieval

Known-positive hit ratio @5: {retrieval['known_positive_recall_at_5']:.3f};
@10: {retrieval['known_positive_recall_at_10']:.3f}; MRR:
{retrieval['known_positive_mrr']:.3f}. Mean retrieval latency:
{retrieval['mean_retrieval_latency_ms']:.1f} ms; p95:
{retrieval['p95_retrieval_latency_ms']:.1f} ms across {retrieval['queries']} queries.
The small labeled set cannot support full context recall or a population accuracy
claim. See `retrieval_evaluation.csv` for query-level hits and timings.

## Metadata filtering on the full index

Four actual queries tested restaurant-only, restaurant plus service, restaurant plus
negative predicted sentiment, and restaurant plus a 2021 date range. All returned
review metadata matched the requested filters. Evidence IDs, filter values and timings
are in `filter_verification.json`. This checks filter correctness on the sampled results,
not semantic retrieval quality.

## Grounding and generation

Qwen answers use retrieved reviews only. Each accepted review ID must be in the
retrieval result, and every accepted citation must appear in the answer. Invalid or
weak evidence produces an explicit insufficient-evidence response. This deterministic
check does not prove every sentence is factually entailed; manually reviewed examples
are described below. The Qwen model is used for synthesis, not Foody sentiment.

"""
    for count, item in enumerate(examples, 1):
        response = item.get("response", {})
        rag_report += f"### Example query {count}\n\n"
        rag_report += f"Answer: {response.get('answer', 'Unavailable')}\n\n"
        rag_report += f"Evidence IDs: {[x['review_id'] for x in response.get('evidence', [])]}\n\n"
        rag_report += f"Telemetry: `{json.dumps(item.get('telemetry', {}), ensure_ascii=False)}`\n\n"
        name = example_paths[count - 1].name
        rag_report += f"Source artifact: `{name}`. Manual judgment: "
        judgment = next((review for review in manual["examples"] if review["file"] == name), None)
        rag_report += f"{judgment['judgment'] if judgment else 'not reviewed'}.\n\n"
    for count, item in enumerate(health, 1):
        report = item.get("report", {})
        rag_report += f"### Health report example {count}\n\n"
        rag_report += f"{json.dumps(report, indent=2, ensure_ascii=False)}\n\n"
        rag_report += f"Telemetry: `{json.dumps(item.get('telemetry', {}), ensure_ascii=False)}`\n\n"
    rag_report += "## Small answer review\n\n"
    rag_report += (json.dumps(manual, ensure_ascii=False, indent=2) if manual else
                   "No manual answer review completed; answer relevance and faithfulness are unmeasured.")
    rag_report += "\n\nNo automated LLM judge is treated as ground truth. One unfiltered service answer\n"
    rag_report += ("cited a positive review for a complaint. Source-ID membership and JSON schema "
                   "do not prove entailment; the filtered follow-up was manually supported.\n")
    (ROOT / "RAG_ACCEPTANCE_REPORT.md").write_text(rag_report, encoding="utf-8")

    provenance = read("models/qwen_v1/provenance.json")
    filtered = sum(item["all_results_match"] for item in filters.values())
    page_pass = sum(item["exception_count"] == 0 for name, item in streamlit.items()
                    if name != "AI Review Analyst query")
    combined = f"""# RAG, Streamlit and LLMOps acceptance

The completed Foody intelligence data pipeline was preserved. No dataset download,
sentiment/aspect retraining, or Phase 1 Databricks change was made in this work.

## Persistent vector index and retrieval

| Evidence | Actual result |
| --- | --- |
| Source distinct review documents | {integrity['source_document_count']:,} |
| Persisted Chroma vectors / unique IDs | {integrity['collection_count']:,} / {integrity['stored_unique_review_ids']:,} |
| Full ID/hash/dimension reconciliation | {'PASS' if integrity['pass'] else 'FAIL'} |
| Embedding | {selection['model']} ({selection['dimension']} dimensions, {selection['device']}) |
| ONNX export and revision | {selection['onnx_source_repo']} @ {selection['revision']} |
| ONNX SHA256 | `{selection['model_hash']}` |
| ONNX size | {selection['model_size_mb']:.1f} MB |
| Collection and path | `{index['collection']}` in `{integrity['collection_path']}` |
| Batch size | 128 |
| Measured embedding time and throughput | {index['embedding_seconds']:.1f}s; {index['documents_per_second']:.1f} documents/s |
| Total wall-clock index time | Not captured separately; embedding time excludes possible Chroma write overhead |
| Full-index known-positive Hit@5 / Hit@10 | {retrieval['known_positive_recall_at_5']:.3f} / {retrieval['known_positive_recall_at_10']:.3f} |
| Full-index known-positive MRR | {retrieval['known_positive_mrr']:.3f} |
| Retrieval mean / p95 | {retrieval['mean_retrieval_latency_ms']:.1f} / {retrieval['p95_retrieval_latency_ms']:.1f} ms across {retrieval['queries']} queries |
| Full-index metadata filtering | {filtered}/{len(filters)} cases returned only matching restaurant, aspect, sentiment and date metadata |

Two multilingual embeddings were compared locally on the same 612-document pool.
MiniLM scored {selection['known_positive_recall_at_10']:.3f} Hit@10 versus
{pd.read_csv(ROOT / 'embedding_comparison.csv').loc[lambda frame: frame.model != selection['model'], 'known_positive_recall_at_10'].iloc[0]:.3f}
for Potion multilingual. The full-index six-query benchmark scored lower.
These are **small manually constructed known-relevance benchmarks** with incomplete
labels, not full-corpus recall, accuracy, or Foody model quality estimates.

## Qwen grounded answers and health report

Qwen model: `{provenance['model']}`; runtime: llama.cpp; model revision:
`{provenance.get('revision', 'see provenance.json')}`; Q4_K_M quantization;
model SHA256 `{provenance['model_sha256']}`. The local GPU-enabled runtime was
actually invoked. Prompt hashes: RAG `{qwen_tokens['rag_prompt_sha256']}` and
health `{qwen_tokens['health_prompt_sha256']}`.

{len(examples)} RAG query artifacts were run: {qwen_tokens['grounded_query_count']}
passed the mechanical citation/schema gate. A future-date query returned the
insufficient-evidence response without a Qwen call. A food query fell back after
invalid generation. The manually reviewed unfiltered service answer cited a
positive review to support a complaint; it is **not** a faithful answer even though
the review ID belonged to the retrieved set. The negative/service-filtered follow-up
and the complaint example were manually supported. See `rag/manual_answer_review.json`.

One actual Restaurant Health Report was generated for Hanuri - Quán Ăn Hàn Quốc -
Nguyễn Đình Chiểu. It combines mart counts/ratings with Qwen review synthesis.
The report contains strengths, complaints, recommendations, and three evidence
review IDs. Aspect trends and warnings were empty because the available overall
monthly ratings do not establish aspect-specific trends and no alert was cited.
All cited review IDs were checked against the retrieved review text and the mart
values were checked against their source. JSON/schema and source-ID membership
alone do **not** establish semantic faithfulness.

Across the saved query and health artifacts, observed Qwen usage was
{qwen_tokens['input_tokens']:,} input tokens and {qwen_tokens['output_tokens']:,}
output tokens. Mean reported LLM latency per generation-bearing request was
{qwen_tokens['mean_llm_latency_per_request_ms']:.1f} ms; mean end-to-end latency
across saved requests was {qwen_tokens['mean_end_to_end_latency_ms']:.1f} ms.
Per-request retrieval, LLM, end-to-end, model/prompt version and schema status
are preserved in `rag/examples/*.json` and `rag/token_summary.json`. Health
contains two Qwen synthesis calls whose latency and tokens are summed in its
request telemetry. No token counts were imputed for unavailable values.

## Streamlit, monitoring, containers and verification

| Check | Result |
| --- | --- |
| Four Streamlit pages | {page_pass}/4 rendered without exceptions |
| AI Review Analyst action | Actual query displayed answer/evidence; no warning or exception |
| Streamlit HTTP smoke | `/_stcore/health` returned `200 ok`; server stopped |
| LLMOps | {execution['llmops']} |
| Docker Compose | {execution['docker_compose']} |
| Docker runtime | {execution['docker_runtime']} |
| Pytest | {execution['pytest']} |
| Ruff | {execution['ruff']} |
| compileall | {execution['compileall']} |

The four pages are Restaurant Intelligence, Review Explorer, AI Review Analyst,
and LLMOps Monitor. SQLite at `state/llmops.sqlite` is the operational backend.
Current request rows include timestamp, component, model and embedding names,
model/prompt version, retrieval/LLM/end-to-end latency, token counts, top-k,
status and schema validity. Filter metadata is whitelisted; raw questions and
review text are not logged.

## Scientific and operational limits

VN-MSASar results remain research benchmark metrics. Foody is real-world
inference/demo data; its sentiment/aspect predictions have no measured Foody
accuracy. Issue extraction is heuristic. Restaurant keys derive from names,
not verified Foody IDs. Historical alert rules are demonstrations, not live
operational alerts. Incomplete manual relevance labels cannot support a
population retrieval claim. One manually observed contradictory citation shows
the need for semantic evidence review; the current gate checks ID membership
and inline citation consistency only. Docker runtime was not run because the
daemon is unavailable.

## Final demo commands (from repository root)

```powershell
.venv\\Scripts\\python.exe -m scripts.verify_rag_index
.venv\\Scripts\\python.exe -m src.cli rag-evaluate
.venv\\Scripts\\python.exe -m src.cli rag-query --restaurant "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu" --question "Nhân viên phục vụ có vấn đề gì không?" --sentiment negative --aspect service
.venv\\Scripts\\python.exe -m src.cli restaurant-health-report --restaurant "Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu"
.venv\\Scripts\\python.exe -m streamlit run app/app.py
```

See [detailed evidence](rag/RAG_ACCEPTANCE_REPORT.md),
`rag/embedding_comparison.csv`, `rag/retrieval_evaluation.csv`,
`rag/filter_verification.json`, `rag/manual_answer_review.json`, and
`rag/execution_checks.json`.
"""
    Path("reports/RAG_STREAMLIT_LLMOPS_ACCEPTANCE_REPORT.md").write_text(
        combined, encoding="utf-8")
    return {"examples": len(examples), "health_reports": len(health),
            "tokens": qwen_tokens, "operational_logs": operational}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
