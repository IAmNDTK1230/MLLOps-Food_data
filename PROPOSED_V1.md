# Qwen Proposed V1 — frozen text-only prompt classification

The dataset, canonical split and both baseline models/results remain unchanged.
V1 performs inference only: no fine-tuning, LoRA, aspect-conditioned features,
embedding construction, registration or production promotion.

## Model/runtime and hardware

Official Qwen/Qwen2.5-1.5B-Instruct-GGUF, revision
`91cad51170dc346986eccefdc2dd33a9da36ead9`, Apache 2.0, 1.54 billion parameters.
Quantization Q4_K_M; GGUF bytes 1,117,320,736 (about 1.04 GiB).
Model checksum is verified against the publisher's LFS SHA-256.
llama.cpp b11429, commit d81235049384534c167caea52b85a694f6103d14,
Windows CUDA 12.4 binaries; both binary ZIPs verified against pinned release digests.

Hardware inspection discovered an NVIDIA RTX 3060 Laptop GPU, 6144 MiB VRAM,
driver 566.07, in addition to the Ryzen 7 6800H/16 GB RAM described in the request.
V1 uses this GPU, requesting all layers offloaded, with eight CPU threads and a
2048-token context. Weight size is only a lower bound on memory requirement;
runtime KV/compute buffers and other device users add overhead. Actual sampled
host RSS and device memory are recorded instead of claiming weight size equals RAM.
This configuration is chosen for the available hardware; no other model family
or runtime is silently substituted. Download/initialization failure is a failed
execution, not fabricated model acceptance.

## Commands

```powershell
.\.venv\Scripts\Activate.ps1
# Only model/runtime assets; NEVER the restaurant dataset:
python -m scripts.setup_qwen_v1
python -m src.cli proposed-v1
```

Config: qwen_v1.yaml. Model weights live in ignored models/qwen_v1 and binaries
in ignored .qwen-runtime. Do not rerun setup after the experiment is frozen;
the existing pinned model/runtime provenance is part of the selected protocol.
No weights are included in Git or the DVC outputs.

## Prompts, inputs and structured outputs

Three candidates under prompts/food_sentiment:

- v1_a.txt: minimal zero-shot classification.
- v1_b.txt: explicit negative/neutral/positive definitions, negation and factual/mixed text.
- v1_c.txt: definitions and exactly three TRAIN-only examples, one per class.

Few-shot selection is deterministic: train only, unambiguous within train, text
length 30..120 characters, sorted review_id/text. v1_c is materialized before
inference; manifest records example origin and every prompt hash. IDs, aspects,
ratings, source, split and the current sample's gold sentiment never enter the
inference request. TRAIN example labels are the explicitly permitted few-shot context.

The request contains a system instruction and user JSON with only `text`.
The official Qwen GGUF chat template is applied by llama-server. Temperature 0,
seed 42, max_new_tokens 32; no sampling/fine-tuning search. Prefix caching is
explicitly disabled for evaluation requests. One train-only warmup per candidate.

Schema-constrained decoding permits exactly {"sentiment":"negative|neutral|positive"}.
Parsing validates the entire JSON object, rejects duplicate/extra keys, markdown,
arbitrary free text and unsupported label values. One retry with a fixed format
correction is allowed. Raw responses, retries, first/final validity and token counts
are saved. Terminal invalid output remains an abstention/error; it is never mapped
silently to neutral or another sentiment. Validity reflects constrained decoding,
not a claim about unconstrained model output reliability.

## Selection and once-only held-out evaluation

Use ONLY data/food/splits/reviews.parquet, with all existing checksum/membership
checks. All 842 validation rows are evaluated for each prompt. Rank by validation
macro F1, then final JSON validity, then lower measured mean latency. Freeze the
model/runtime/quantization/prompt/settings before the 887-row test evaluation.

Each prediction is durably checkpointed; completed rows are reused without another
model call. A pending test-request marker precedes inference. If a crash leaves an
uncheckpointed in-flight request, stop for inspection instead of silently repeating
it. Completed test runs verify saved artifacts and return cached evidence. Changing
the frozen protocol refuses a new automatic held-out evaluation. Baselines are
read as saved results; no baseline training/evaluation command is called.

## Reports and MLflow

reports/proposed_v1 contains:
prompt_manifest.json; per-prompt validation metrics/predictions/checkpoints;
qwen_prompt_comparison.csv; selection.json; test_state.json; test_checkpoint.jsonl;
qwen_test_metrics.json; qwen_predictions.parquet; qwen_confusion_matrix.png;
qwen_classification_report.csv; model_comparison.csv/.png;
qwen_failure_cases.csv; qwen_error_by_aspect.csv; failure_analysis.json;
qwen_resource_report.json.

Baseline JSON validity is unavailable/not applicable, so the comparison CSV leaves
that field blank. Macro-F1/neutral-F1 deltas and improvement labels use measured
values. Improvements over rules do not establish superiority over TF-IDF.

Resource sampling is once per second: llama-server host RSS/CPU and device-wide
NVIDIA utilization/memory. These are sampled peaks, not continuous exact maxima;
GPU memory can include other applications. Token counts use actual runtime usage,
including retries. Warmup usage is separate. Duration/throughput include request and
checkpoint overhead; latency is end-to-end single-sample HTTP inference including
retries. Qwen uses GPU while frozen baselines used CPU; timing is not a controlled
same-device kernel benchmark.

Existing MLflow infrastructure logs each prompt's model/revision/quantization,
runtime tag, prompt/hash/settings/data hashes and validation scores/resources.
Only the validation-selected run receives test metrics. No registry operation.
reports/task2_proposed_v1_evidence.json supports Slides 10–13 without rewriting
the frozen baseline evidence.

## Failure analysis and V2

Aspect/sarcasm/gold labels are joined ONLY after inference for analysis. Sarcasm and
neutral flags use source gold labels; mixed/negation/implicit-word flags are lexical
proxies, explicitly not verified phenomenon annotations. Long=200+ characters;
short=20 or fewer. Test labels and predictions are never edited.

The known 81 global / 11 test conflicting normalized texts make text-only targets
ambiguous when aspects have different gold labels. Aspect conditioning is a separate
V2 task, not a permitted V1 workaround. V2 recommendations are grounded in actual
V1 errors; no V2 code or run is implemented.

## DVC and checks

Food preparation and all three baseline stages remain frozen. proposed_v1 depends
on the existing split/baseline evidence, prompts/config and pinned model provenance.
It never downloads weights. After execution:

```powershell
python -m dvc repro proposed_v1
python -m pytest -q
python -m ruff check .
python -m compileall -q src databricks scripts
```

Full measured acceptance results are in reports/PROPOSED_V1_ACCEPTANCE_REPORT.md.
The executed run required reporting-only recovery after all 887 outputs were saved.
No additional review predictions were made. Evaluation peak utilization/memory
samples were lost; reports mark them unavailable and distinguish a separate
zero-inference loaded-model capacity probe. Resource sampling now persists to logs.
