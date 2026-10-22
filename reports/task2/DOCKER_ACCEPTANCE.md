# Docker RAG V3 runtime acceptance — 2026-10-22

**Decision: PASS for the containerized RAG V3 serving path and the historical pipeline startup.** Docker daemon 29.8.1, Compose syntax, targeted image builds, container startup, Qwen 3B HTTP health, API HTTP health, Streamlit HTTP health, a real V3 request, and the Streamlit Analyst handler were verified locally. The pipeline's documented default command now exits successfully after displaying CLI help. No Foody data, Chroma index, or model was rebuilt.

## Exact FileNotFoundError and fix

The Streamlit page catches exceptions and displays only `RAG unavailable (FileNotFoundError)`. Replaying its RAG V3 call inside the **Streamlit container** with `python -m src.cli rag-v3-query` produced this complete traceback before the fix:

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/src/cli.py", line 156, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/app/src/cli.py", line 68, in main
    return rag_v3_cli.execute(args)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/src/rag_v3/cli.py", line 47, in execute
    retriever = RoutedRetriever()
                ^^^^^^^^^^^^^^^^^
  File "/app/src/rag_v3/retriever.py", line 11, in __init__
    self.dense = dense or selected_retriever()
                          ^^^^^^^^^^^^^^^^^^^^
  File "/app/src/rag/cli.py", line 34, in selected_retriever
    if digest(model_path) != selection["model_hash"]:
       ^^^^^^^^^^^^^^^^^^
  File "/app/src/foody/schema.py", line 32, in digest
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/pathlib.py", line 1019, in read_bytes
    with self.open(mode='rb') as f:
         ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'models\\embeddings\\models--Qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q\\snapshots\\faf4aa4225822f3bc6376869cb1164e8e3feedd0\\model_optimized.onnx'
```

`reports/rag/embedding_selection.json` was saved on Windows with backslashes in its project-relative `model_file`. Linux `Path` treats those backslashes as literal filename characters. The actual ONNX file was **already mounted and readable** at `/app/models/embeddings/models--Qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/snapshots/faf4aa4225822f3bc6376869cb1164e8e3feedd0/model_optimized.onnx`. `src/rag/cli.py` now normalizes separators while resolving this saved artifact path; the model checksum check, embedding, retrieval, and RAG behavior remain unchanged. Qwen was running and was not the cause of this exception. The API container had the same mount and would encounter the same saved-path mismatch if it entered this retriever path.

## Host, mount, and configured path audit

`<repo>` below is `C:\Users\ASUS\hoc code\vsc (python)\MLOP_soopi\shopee-data-mlops-platform`. Docker inspect confirmed `./data → /app/data` (RW), `./models → /app/models` (RO), `./state → /app/state` (RW), and `./reports → /app/reports` (RO) in **both** Streamlit and API containers. The Qwen container mounts `./models → /models` (RO). File existence and readability were checked inside both Python containers after the fix.

| Artifact | Host path | Container path | Exists/readable? | Application configured path | Mismatch? |
| --- | --- | --- | --- | --- | --- |
| Chroma manifest + SQLite | `<repo>/data/foody/vector_store/` | `/app/data/foody/vector_store/` | Yes / yes, both | `data/foody/vector_store/` | No |
| Foody RAG documents | `<repo>/data/foody/rag/documents.jsonl` | `/app/data/foody/rag/documents.jsonl` | Yes / yes, both | `data/foody/rag/documents.jsonl` | No |
| Foody enriched data | `<repo>/data/foody/enriched/reviews.parquet` | `/app/data/foody/enriched/reviews.parquet` | Yes / yes, both | `data/foody/enriched/reviews.parquet` | No |
| Foody marts | `<repo>/data/foody/mart/restaurant_summary.parquet` | `/app/data/foody/mart/restaurant_summary.parquet` | Yes / yes, both | `data/foody/mart/restaurant_summary.parquet` | No |
| BM25 SQLite + manifest | `<repo>/data/foody/lexical_index/` | `/app/data/foody/lexical_index/` | Yes / yes, both | `data/foody/lexical_index/` | No; manifest's historical absolute `path` is metadata, not used to open DB |
| Selected ONNX embedding | `<repo>/models/embeddings/.../model_optimized.onnx` | `/app/models/embeddings/.../model_optimized.onnx` | Yes / yes, both | `reports/rag/embedding_selection.json` contained Windows `models\embeddings\...` | **Yes before fix; resolved by separator normalization** |
| Qwen 3B GGUF + provenance | `<repo>/models/qwen_v3/` | `/app/models/qwen_v3/` and `/models/qwen_v3/` for Qwen | Yes / yes | `config/demo.yaml`, `qwen_v3_3b.yaml`, Compose model default | No after Compose change |
| LLMOps state | `<repo>/state/llmops.sqlite` | `/app/state/llmops.sqlite` | Yes / yes; new event written | `state/llmops.sqlite` | No |
| Saved RAG selection/report | `<repo>/reports/rag/embedding_selection.json` | `/app/reports/rag/embedding_selection.json` | Yes / yes, both | `reports/rag/embedding_selection.json` | No mount mismatch; serialized inner path fixed at read time |

The mounted accepted index is unchanged: Foody RAG documents **32,457**, Chroma collection `foody_39897b1e_634d0f66c29d` **32,457**, and BM25 SQLite **32,457**. The API container independently read all three counts; Streamlit's `RoutedRetriever` loaded the same collection and BM25 count. The existing manifest reports the same document hash (`52019e46d6f5205247bbf523fd57b089f5c55ef62e4769e4b368e3213e66a4f3`). No index build was run; the existing BM25 manifest was validated.

## Serving checks and real request

| Check | Result |
| --- | --- |
| `docker info` | PASS; server 29.8.1 |
| `docker compose config --quiet` | PASS |
| `docker compose build streamlit rag-backend` | PASS after path fix; images contain code/config only, not data or weights |
| `docker compose up -d streamlit rag-backend qwen` | PASS |
| Qwen | `docker compose ps`: **healthy**; `/health` HTTP 200 `{"status":"ok"}`; log confirms `/models/qwen_v3/qwen2.5-3b-instruct-q4_k_m.gguf` loaded on port 8193 |
| API/backend | Process up; `/health` HTTP 200 with `status=ready`, index/weights availability true. This endpoint does not exercise a V3 API query. |
| Streamlit | Process up; `/_stcore/health` HTTP 200 `ok`; `/` HTTP 200 |
| Streamlit Analyst handler | In-container `AppTest` selected AI Review Analyst and clicked Ask: 0 exceptions, 0 warnings, Answer and evidence sections present, generation mode visible, 1 cited review from 5 retrieved |

The real containerized query used Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu and “Nhân viên phục vụ có vấn đề gì không?”. The Qwen 3B server received generation attempts. The accepted result was `extractive_fallback`, status `supported_extractive_fallback`, retrieval method `RAG_V3_BM25_FILTERED`, with one cited review ID (`eeefa8287a89ddf678cda23fc243399db24b8e5ae11ae0c978e31f5645ce7159`), readable supporting review text, and `negative` / `service` metadata. No FileNotFoundError occurred. The Streamlit Analyst handler produced the same valid mode and citation count. The fallback is expected when generated clauses fail the unchanged V3 validator; it is not a claim of successful Qwen synthesis.

LLMOps event IDs **400** (CLI) and **403** (Analyst handler) appeared after the request baseline ID 397. Event 403 records `rag_version=RAG_V3`, `generation_model=Qwen/Qwen2.5-3B-Instruct-GGUF`, `generation_mode=extractive_fallback`, `retrieval_method=RAG_V3_BM25_FILTERED`, `evidence_count=1`, `support_gate_status=ACCEPTED`, `clause_support_status=extractive_verified`, retrieval latency about 5,027 ms, LLM latency about 41,969 ms, end-to-end about 47,415 ms, and `status=supported_extractive_fallback`. Event fields contain no raw question or review text; filters hold only restaurant key, aspect, and sentiment. These are local CPU timings from this run, not a new benchmark.

## Pipeline startup and dependency acceptance

Before the fix, `docker compose logs --tail=200 pipeline` showed `ModuleNotFoundError: No module named 'psutil'`. The import chain is `src.cli` → `src.rag_v3.cli` → `src.llm.gateway` → `src.proposed_v1.runtime`, which imports `psutil`. `Dockerfile` installs `requirements.txt`; that file now pins `psutil==7.2.2`, matching the existing demo dependency pin. `docker compose build pipeline` succeeded, and `docker compose up -d pipeline` started the rebuilt image. `docker compose ps -a pipeline` then reported **Exited (0)** after its default `python -m src.cli --help` command, and the log showed CLI usage without a traceback. An isolated `docker compose run --rm --no-deps pipeline` import probe loaded both `psutil` (7.2.2) and `src.cli`. No ingestion or training command was run.

After `docker compose up -d`, `docker compose ps -a` showed `pipeline` **Exited (0)**, `qwen` **Up (healthy)**, `rag-backend` **Up**, and `streamlit` **Up**. Qwen `/health`, backend `/health`, and Streamlit `/_stcore/health` each returned HTTP **200**. Optional Prefect profile services were not enabled.

## GPU and remaining boundary

`GPU_CONTAINER_RUNTIME = NOT VERIFIED`. The Windows host reports an RTX 3060 and Docker lists an `nvidia` runtime, but Compose does not request a GPU, the Qwen container has no `/dev/nvidia*` device, and llama.cpp logs `no usable GPU found` while continuing on CPU. The inspected image's CUDA capability is unproven; the observed immediate issue is missing GPU assignment to this container. CPU inference completed the real request, so GPU support does not block serving acceptance.

The `pipeline` service is a one-shot CLI help command under the default Compose configuration, so its successful **Exited (0)** state is expected; this check does not claim an ingestion run. The API `/ask` endpoint is a separate older RAG path and was not used for the accepted V3 request. The accepted serving path remains Qwen, Streamlit Analyst, mounted retrieval, and the API health endpoint. No ML/RAG artifact was rebuilt or changed.
