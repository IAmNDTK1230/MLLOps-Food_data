# Task 2 Review-2 self-assessment

Status vocabulary: DONE, IN PROGRESS, NOT DONE. DONE means evidence was generated and verified, not merely that code exists. CI and Docker deployment are scoped to their verified steps.

| Item | Status | Evidence / limit |
| --- | --- | --- |
| 1. Data source table | DONE | `dataset_sources.csv` |
| 2. Cleaning/labeling | DONE | `CLEANING_FLOW.md` |
| 3. Train/Val/Test | DONE | `DATASET_SUMMARY.md` |
| 4. Leakage checks | DONE | `DATASET_SUMMARY.md` |
| 5. Imbalance | DONE | `class_distribution.csv` |
| 6. Imbalance handling | DONE | `imbalance_comparison.csv` |
| 7. Baseline 1 | DONE | `classical_model_comparison.csv` |
| 8. Baseline 2 | DONE | `classical_model_comparison.csv` |
| 9. 3-seed stability | DONE | `baseline_three_seed.csv` |
| 10. Paper vs Group | DONE | `PAPER_VS_GROUP.md` |
| 11. Gap Analysis | DONE | `GAP_ANALYSIS.md` |
| 12. Proposed V1 | DONE | `../proposed_v1/qwen_test_metrics.json` |
| 13. Failure analysis | DONE | `failure_counts.csv` |
| 14. V2/V3 | DONE | `../rag_v3/generation_metrics.json` |
| 15. Ablation | DONE | `RAG_ABLATION_FINAL.md`, `figures/rag_ablation_final.png` (observed contributions; no component-off control) |
| 16. Latency/model size | DONE | `FEASIBILITY.csv` |
| 17. RAG chunking/token accounting | DONE | `rag_token_statistics.json` |
| 18. Embedding/vector DB | DONE | `embedding_workload.csv` |
| 19. Retrieval metrics | DONE | `RAG_ABLATION.csv` |
| 20. Generation/faithfulness | DONE | `LLM_BENCHMARK.md` |
| 21. Docker/microservices | DONE | `DOCKER_ACCEPTANCE.md` (serving path and pipeline startup accepted) |
| 22. GitHub Actions CI/CD | DONE | `CICD_EVIDENCE.md` (CI, release job and GHCR manifest verified) |
| 23. Reproducibility | DONE | `baseline_three_seed.csv` |
| 24. Slide-ready figures | DONE | `figures/` |

## Outstanding

- DONE source audit: no directly comparable published macro-F1 result on this exact task was verified; numerical paper gaps are blank in `paper_vs_group.csv`.
- DONE evidence synthesis: `RAG_ABLATION_FINAL.md` makes the observed role of each retrieval and safety component visible; the six-query known-positive labels are incomplete and no component-off causal effect is claimed.
- DONE Docker: serving path, mounted indexes, RAG V3 request, LLMOps event and pipeline's default one-shot startup (exit 0) verified; no ingestion was run (see `DOCKER_ACCEPTANCE.md`).
- DONE CI/CD: successful GitHub Actions CI and release runs and the public `v1.0.0` GHCR image manifest verified; remote production deployment is outside this checklist (see `CICD_EVIDENCE.md`).
- NOT DONE: a full LLM-from-scratch experiment (not part of this project's scope).
