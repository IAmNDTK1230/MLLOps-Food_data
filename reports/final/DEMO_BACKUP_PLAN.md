# Demo backup plan

Check `python -m src.cli demo-preflight` before presenting. The backups below show **saved, verified artifacts**; do not describe them as a new live run. Keep the application and terminal visible so the switch is honest.

| Failure | Safe presentation response | Verified artifact |
|---|---|---|
| Qwen server unavailable | Show the preflight result and server log, then open the saved V3 six-case comparison. Explain the one accepted synthesis, two fallbacks, and three evidence refusals as the earlier local evaluation. Do not click **Ask** repeatedly or claim the model responded live. | `reports/rag_v3/generator_comparison.json`, `reports/rag_v3/RAG_V3_ACCEPTANCE_REPORT.md` |
| Chroma unavailable | Stop live RAG use. Continue the mart/dashboard and research story; use the saved retrieval comparison and evidence packets to explain routing. Do not rebuild the index during the talk. | `reports/rag_v3/retrieval_policies.csv`, `reports/rag_v3/evidence_packets.json` |
| Streamlit AI action fails | Leave the error visible briefly, then show the saved generated answer and its exact cited review in the acceptance report/comparison JSON. If only the AI action fails, continue with Restaurant Intelligence and Review Explorer. | `reports/rag_v3/generator_comparison.json`, `reports/rag_v3/streamlit_smoke.json` |
| GPU unavailable | Stop the launcher and retry ` .\scripts\start_demo.cmd -CpuOnly ` if time allows; CPU may be much slower. Otherwise show the saved evidence and results, explicitly saying live GPU inference is unavailable. | `reports/rag_v3/generation_metrics.json`, `reports/rag_v3/health_report.json` |
| Docker unavailable | Use the supported local Windows launcher. State that `docker compose config` passed previously but runtime has not been verified because the daemon was unavailable. Do not attempt to repair Docker Desktop during the presentation. | `README.md` Docker section, `reports/final/FINAL_DEMO_ACCEPTANCE.md` |

In every fallback, distinguish the saved artifact's measured date/run from what was observed on the current presentation machine. Never fill missing results with invented numbers.
