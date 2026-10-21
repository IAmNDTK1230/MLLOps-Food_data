# Final student-demo acceptance

| Area | Status | Evidence and limit |
|---|---|---|
| Demo startup and preflight | **PASS** | `python -m src.cli demo-preflight` returned nine PASS checks. `scripts/start_demo.ps1` was exercised on temporary port 8519: it passed preflight, started the pinned Qwen 3B llama.cpp server, and started Streamlit. The temporary run was stopped afterward. Missing artifacts stop the launcher before serving. |
| Data readiness | **PASS** | Foody enriched parquet has 33,946 source-preserved rows. The restaurant mart has 2,110 names. Research split counts and leakage checks are frozen in `KEY_NUMBERS.json`. |
| Models | **PASS** | Sentiment TF-IDF, aspect TF-IDF, and Qwen 3B paths/checksums passed read-only verification. The demo configuration pins MiniLM embeddings, RAG V3 prompt, top-k 5, Qwen 3B Q4_K_M, and the vector collection. No model was retrained or downloaded. |
| Foody enrichment and marts | **PASS** | Existing report records 31,694 AI-valid inference observations. Summary, monthly, aspect, issue, and alert marts exist. Foody has no trusted gold prediction accuracy; issue detection is heuristic. |
| RAG V3 | **PASS** | Chroma and BM25 each contain 32,457 documents. The accepted six-case 3B run has one validated synthesis, two extractive fallbacks, three correct refusals, and zero unsupported accepted answers. The set is small and incomplete for general reliability. |
| Streamlit | **PASS** | Four-page smoke test passed with a real RAG V3 health-report action and an analyst query. The answer, generation mode, citation ID, supporting review text, and metadata are visible. Presentation UI distinguishes source from AI fields and research metrics from operational monitoring. |
| LLMOps SQLite | **PASS** | Existing `state/llmops.sqlite` was opened read-only by preflight. V3 request logging distinguishes generated/fallback/refusal modes and support status without raw query/review text. |
| Docker configuration | **PASS** | The previously recorded `docker compose config` check passed. No Docker changes were made. |
| Docker runtime | **NOT VERIFIED** | The daemon was unavailable in the prior acceptance work. This demo uses the verified local Windows path; no Docker runtime claim is made. |
| Tests and static checks | **PASS** | `python -m pytest -q`: **287 passed**, one upstream deprecation warning. `python -m ruff check .`: PASS. `python -m compileall -q src app`: PASS. JUnit results: `test_results.xml`. |

The model comparison table and `KEY_NUMBERS.json` are generated from prior metric JSON/CSV artifacts plus the final JUnit file. Unavailable metrics are blank, not zero. Qwen V2-A test reuse remains exploratory. Retrieval Hit@k/MRR uses a six-query set with incomplete known positives. The saved Foody and V3 artifacts were not rebuilt.

## Final demo commands

From the existing project directory in PowerShell:

```powershell
python -m src.cli demo-preflight
.\scripts\start_demo.cmd
```

Open `http://localhost:8501`. If GPU inference is unavailable and time permits, stop the launcher and use `.\scripts\start_demo.cmd -CpuOnly`; CPU generation may be slow. The default restaurant is **Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu**, while other restaurants remain selectable. Follow `DEMO_SCRIPT_VI.md`; use `DEMO_BACKUP_PLAN.md` if a live component fails. Do not present saved evaluation output as a fresh live result.
