# Task 2 Review-2 gap matrix

This matrix was opened before new experiments. “Existing” refers to saved project evidence; the final column is updated only after the new evidence is verified. Private data, model weights, and accepted demo outputs remain unchanged.

| Requirement | Current status | Existing evidence | Missing evidence | Action in this task | Final status |
| --- | --- | --- | --- | --- | --- |
| 1. Data source table | Partial | `reports/food_dataset_report.json`, Foody provenance reports | Side-by-side source/role table | Build dataset CSV and summary | DONE |
| 2. Cleaning/labeling | Partial | Food and Foody cleaning reports | One slide-ready flow; label distinction | Document flow | DONE |
| 3. Train/Val/Test | Done | Frozen split metadata, `KEY_NUMBERS.json` | Compact table | Reuse, do not resplit | DONE |
| 4. Leakage checks | Done | Split metadata: zero cross-split review IDs/text | Review-2 explanation | Reuse evidence | DONE |
| 5. Imbalance | Partial | 3-class counts in dataset report | Plot and exact ratio | Generate count/ratio figure | DONE |
| 6. Imbalance handling | Partial | `reports/baselines/class_weight_comparison.csv` | Per-class effect presentation | Reuse measured comparison and compute weights | DONE |
| 7. Baseline 1 | Done | Rule test metrics | Slide comparison | Reuse | DONE |
| 8. Baseline 2 | Done | TF-IDF test metrics | Slide comparison | Reuse | DONE |
| 9. 3-seed stability | Missing | One frozen accepted seed | Same-split independent stability runs | Run separate classical research comparison | DONE |
| 10. Paper vs Group | Missing | Dataset card and project references | Comparable result/citation audit | Source-based table | DONE |
| 11. Gap Analysis | Partial | Failure reports | Evidence vs hypothesis separation | Write analysis | DONE |
| 12. Proposed V1 | Done | Qwen V1 test report | Compact comparison | Reuse | DONE |
| 13. Failure analysis | Partial | Baseline/V1/V2/V3 failure artifacts | Unified taxonomy | Aggregate verified counts | DONE |
| 14. V2/V3 | Done with caveats | V2 exploratory report; RAG V3 acceptance | Slide-ready distinction | Reuse, label V2 exploratory | DONE |
| 15. Ablation | Partial | V1/V2/V3 retrieval policy CSVs and saved six-case safety traces | One applicability-aware table and figure | Reconstruct observed component progression; no causal component-off claim | DONE |
| 16. Latency/model size | Partial | Model/resource JSON | Unified table | Reuse measured values, blank unavailable | DONE |
| 17. RAG chunking/token accounting | Partial | 32,457-review corpus, old token summaries | Selected-tokenizer document analysis | Count tokens, simulate within-review policies | DONE |
| 18. Embedding/vector DB | Done | Chroma/BM25 manifests, embedding selection | Consolidated schema/workload | Document and estimate | DONE |
| 19. Retrieval metrics | Done with caveat | Six-query policy comparison | Figure and caveat | Reuse incomplete known-positive results | DONE |
| 20. Generation/faithfulness | Done with caveat | Fixed V3 six-case replay | Cross-family benchmark | Reuse Qwen; attempt feasible Llama/DeepSeek | DONE |
| 21. Docker/microservices | Partial | Dockerfiles/Compose and config validation | Runtime smoke if daemon available | Verify Qwen 3B, Streamlit, API health, mounted index, live V3 request and LLMOps; rebuild pipeline and verify exit (0) | DONE |
| 22. GitHub Actions CI/CD | Partial | CI/release workflows; user reports CI passing | Run URL and GHCR release status | Verify successful CI and release runs plus public GHCR manifest | DONE |
| 23. Reproducibility | Partial | DVC, pinned hashes, frozen split | Review-2 protocol summary | Consolidate evidence and limitations | DONE |
| 24. Slide-ready figures | Missing | Scattered historical plots | Unified set | Generate figures with missing-data labels | DONE |
