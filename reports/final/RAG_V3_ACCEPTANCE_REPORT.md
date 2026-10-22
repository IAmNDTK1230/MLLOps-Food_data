# RAG V3 acceptance report

**Decision: ACCEPTED for the local evidence-grounded demo.** The fixed six-question replay produced zero unsupported *accepted* answers for either generator. The selected 3B generator produced one validated synthesis, two verified extractive fallbacks, and three correct insufficient-evidence responses. This is a safety and product acceptance result on a small fixed set, not a general model-quality claim.

## Frozen inputs and retrieval

The generation comparison replays `evidence_packets.json` for both generators. `scripts/evaluate_rag_v3_generation.py` loads that saved JSON; its retriever raises if a fresh search is attempted. Thus both models received the same review snippets, metadata, routes, and prompt (`rag_v3:81e19ef10f78`). The prior retrieval benchmark was not rerun.

| Retrieval policy | Known-positive Hit@5 | Hit@10 | MRR | Mean latency, ms | p95, ms |
|---|---:|---:|---:|---:|---:|
| V1 dense | 0.000 | 0.083 | 0.019 | 16.52 | 21.18 |
| V2 BM25 | 0.083 | 0.083 | 0.167 | 240.01 | 620.72 |
| V2 RRF hybrid | 0.083 | 0.083 | 0.042 | 120.09 | 162.25 |
| V2 evidence rerank | 0.083 | 0.083 | 0.056 | 600.29 | 1742.72 |
| V3 routed | 0.083 | 0.083 | 0.167 | 140.97 | 195.15 |

V3 routes aspect-specific complaints to filtered BM25 with evidence reranking, general aspect/summary questions to RRF hybrid, and causal questions to the mart trend guard. The V3 MRR ties BM25 on six manually constructed queries with incomplete known-relevance labels. Latency is one local run, not a stable performance estimate. The benchmark source is `retrieval_policies.csv`.

## Generators and measured outcomes

Both models use Q4_K_M llama.cpp b11429 with deterministic temperature 0 and seed 42. The local artifact checksums are verified by `LLMGateway` before use.

| Generator | Revision | SHA256 | Valid-schema attempts / attempts | Accepted Qwen | Fallback | Correct insufficient | Supported complete | Supported incomplete | Unsupported accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct-GGUF | `91cad51170dc346986eccefdc2dd33a9da36ead9` | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` | 6/6 | 0 | 3 | 3 | 2 | 1 | 0 |
| Qwen2.5-3B-Instruct-GGUF | `cc1e68eea5f05f88f41a6de1fc73110178f23715` | `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d` | 6/6 | 1 | 2 | 3 | 2 | 1 | 0 |

“Supported incomplete” is the most-common-complaint question: one accepted complaint review cannot establish which complaint is most common. The answer gives one cited example without claiming frequency. Price, rating-cause, and empty-period cases correctly returned insufficient evidence. The 3B service synthesis is supported by E2; its complaint and food answers use fallback. The 1.5B model used fallback for all three answerable cases. Citation and clause-support checks pass for all three non-empty final answers per model. No fabricated aggregate “accuracy” score is used.

| Generator | Input tokens | Output tokens | Mean active LLM latency | Mean end-to-end latency, six cases | Peak server RSS | Peak sampled GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| 1.5B | 4,288 | 453 | 1,580 ms | 920 ms | 1.50 GB | 1,301 MiB |
| 3B | 4,288 | 218 | 1,498 ms | 870 ms | 2.47 GB | 2,226 MiB |

The latency figures are from this single replay. The end-to-end value includes saved retrieval latency for each case and zero-call refusals; the active LLM value covers only the three answerable cases. Peak GPU samples are sparse. **3B remains the demo default** because it produced one supported synthesis with fewer fallbacks and no unsupported accepted answer, despite higher memory cost. This small set does not establish broad superiority.

## Grounding and failure handling

The validator checks inline placeholders against the evidence packet, then checks each sentence and factual clause against its own cited review(s). A trailing `[E1]` applies to all clauses in that sentence; an unsupported second clause fails. `[E1][E2]` can jointly support separate subclaims when each subclaim matches at least one cited review. Domain cues check service delay, staff attitude and polarity, food qualities, price polarity, and missing-order complaints. Unsupported frequency words and generated numeric details fail. This is conservative domain-aware matching, not general-purpose natural-language inference. Failure after at most two attempts produces a verified extractive answer using an exact review snippet.

Raw 1.5B and 3B generation strings are retained **only** in `generator_comparison.json`. Operational SQLite logs keep model, route, mode, evidence count, support/clause status, token counts, and timings; they do not store queries, review text, or raw generations. In this replay, all twelve generated attempts had parseable required schema, but failed attempts included unsupported clauses and topic mismatch. The 1.5B attempt that misassigned the person waiting for an order was rejected. Detailed categories and attempt references are in `generation_failure_analysis.json` and `generation_metrics.json`.

The earlier V2 food failure was recorded as an answer-topic mismatch after two attempts. V2 did **not** retain rejected raw output, so its exact JSON/citation failure mode cannot be recovered. V3 food evidence is preserved through a cited extractive answer when Qwen synthesis fails. Positive service praise cannot support a service complaint. A service sentence combining slow service and bad attitude with only a slow-service citation fails; two supporting citations pass. Positive price evidence cannot support a price complaint. The rating-causation route does not infer cause from nearby reviews. Hanuri's latest sparse monthly observations do not establish a recent decline, so this case returned insufficient evidence without asserting that a decline occurred.

## Health report, UI, and LLMOps

The final health report has an exact metric summary (`133` reviews, mean rating `7.38`), a positive-review strength, a negative-review complaint, and a verification-first recommendation. All four sections used clearly labeled fallback in the final run. Each section has a validation status; review sections and the recommendation map to review IDs. The complaint excerpt selector now preserves a late negative span verbatim, avoiding a neutral opening that concealed the actual complaint. Aspect trends and warnings remain empty because the report has no validated aspect-time or current-alert basis. See `health_report.json`.

The Streamlit smoke test passed Restaurant Intelligence, Review Explorer, AI Review Analyst, and LLMOps Monitor, including a real V3 answer with visible generation mode, evidence IDs, supporting review text, and metadata. See `streamlit_smoke.json`. New `rag_v3_request` and `health_v3_request` rows include route, retrieval method, model, generation mode, evidence count, support and clause status, prompt version, timings, tokens, and status. Older historical rows can have null clause status; new rows populate it.

## Verification and limits

`python -m pytest -q`: **285 passed**, one upstream deprecation warning. `python -m ruff check .`: **passed**. `python -m compileall -q src app`: **passed**. Local Qwen comparison, health run, and Streamlit smoke were executed with access to the local llama.cpp endpoint.

The six-question set and known-positive relevance labels are too small to estimate general reliability. Lexical/domain clause checks can reject valid paraphrases, and extractive answers may be incomplete. The V3 health and demo remain traceable under these limits; they should not be treated as a causal analysis or a statistically measured frequency ranking.
