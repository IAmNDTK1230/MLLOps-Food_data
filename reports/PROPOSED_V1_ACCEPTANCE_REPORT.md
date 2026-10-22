# Qwen Proposed V1 — measured acceptance

**Inference and reporting completed. No improvement over TF-IDF on macro F1 or neutral F1.**
The 5869-row dataset and 4140/842/887 split, seed 42, text-only inputs, test labels,
baseline models/results and Phase 1 Databricks pipeline are unchanged.
No fine-tuning, Proposed V2 implementation, embeddings or production registration ran.
Execution timestamps are retained in UTC; local timezone is Asia/Saigon.

## Model/runtime actually used

- Official [Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), Apache 2.0, 1.54B parameters.
- Model revision: `91cad51170dc346986eccefdc2dd33a9da36ead9`.
- Quantization: Q4_K_M; 1,117,320,736 bytes (1117.320736 MB).
- Model SHA-256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`, verified against publisher LFS metadata.
- llama.cpp b11429, commit `d81235049384534c167caea52b85a694f6103d14`, Windows CUDA 12.4.
- Hardware inspection found RTX 3060 Laptop GPU, 6144 MiB VRAM, driver 566.07,
  in addition to Ryzen 7 6800H / 16 logical CPUs / about 16 GB RAM.
- Local GPU execution; 99 requested GPU layers, 8 CPU threads, context 2048.
  NVIDIA process inspection confirmed llama-server as a GPU compute process.
- Temperature 0, seed 42, max_new_tokens 32, strict schema-constrained JSON.
  No model-family substitution or training.

## Frozen prompts and actual validation comparison

| Prompt | Definition | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| V1-A | minimal zero-shot | 0.667458 | 0.646242 | 0.727171 | 0.589172 | 0.737557 |
| V1-B | detailed definitions | 0.741093 | 0.627636 | 0.725847 | 0.626862 | 0.771257 |
| **V1-C** | definitions + three TRAIN examples | **0.758907** | **0.638674** | **0.735168** | **0.641252** | **0.790673** |

All candidates evaluated the same 842 validation rows. Final and first-attempt
JSON validity were 100%; invalid outputs 0. Selection: validation macro F1,
then validity, then lower latency. V1-C won on macro F1; no tie-break was needed.
All prompt hashes and train-only example provenance are in prompt_manifest.json.
Only text is sent for each sample. Few-shot labels are the explicitly permitted
TRAIN examples; current sample gold/metadata are never passed to the model.

One selected neutral TRAIN example is English despite source language=vi. This is
source-tag noise preserved by the exact filter, not substituted data. Prompts and
examples were not edited after observing validation/test outcomes.

## Once-only actual test results

887 unique sample indices; 887 saved response attempts; 0 invalid/retried responses.
Strict JSON object with exactly one sentiment key and one of three allowed labels.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Neutral F1 |
|---|---:|---:|---:|---:|---:|---:|
| Rule-Based | 0.505073 | 0.619395 | 0.596898 | 0.431279 | 0.592298 | 0.167715 |
| TF-IDF + LR | 0.809470 | 0.672730 | 0.695250 | **0.682854** | **0.812104** | **0.413043** |
| Qwen Proposed V1 | 0.732807 | 0.632795 | **0.733282** | 0.621789 | 0.774715 | 0.257778 |

Qwen neutral precision **0.159341**, recall **0.674419**, F1 **0.257778**.
It predicts neutral 182 times with 29 correct, while there are only 43 neutral rows.
Higher neutral recall than TF-IDF comes with substantially worse precision/F1.

| Required improvement delta | Value | Signal |
|---|---:|---|
| Qwen macro F1 − Rule macro F1 | +0.190510 | POSITIVE IMPROVEMENT SIGNAL over rules |
| Qwen macro F1 − TF-IDF macro F1 | −0.061065 | No positive signal |
| Qwen neutral F1 − TF-IDF neutral F1 | −0.155266 | No positive signal |

V1 is not better than the strongest baseline on the primary metric or neutral F1.
Baseline JSON validity is not applicable and remains blank in model_comparison.csv.

## Tokens, latency, size and resources

| Partition/prompt | Input tokens | Output tokens | Total tokens | Mean ms | p95 ms | Duration s |
|---|---:|---:|---:|---:|---:|---:|
| validation A | 79,349 | 5,894 | 85,243 | 66.254 | 81.371 | 58.158 |
| validation B | 152,603 | 5,894 | 158,497 | 75.553 | 89.453 | 66.012 |
| validation C | 237,645 | 5,894 | 243,539 | 93.412 | 114.651 | 81.059 |
| selected C test | 249,190 | 6,210 | 255,400 | 93.660 | 116.343 | 105.530 |

Validation+test totals: **718,787 input + 23,892 output = 742,679 tokens**.
Three train-only warmups add 548 input + 21 output = 569 tokens, separately tracked.
Test mean tokens/sample 287.94; p95 input tokens 307; maximum input tokens 424.
Test throughput 8.405 samples/s; 58.846 output tokens/s including request/checkpoint
overhead. Latency is end-to-end single-sample inference, not a kernel-only benchmark.
Qwen uses GPU; baselines used CPU, so this is not an identical-device latency test.

Memory/resource limitation: the initial in-memory sampler was lost after a reporting
exception. **Inference peak host memory, CPU utilization, GPU utilization and inference
peak device memory are unavailable**, explicitly null in qwen_resource_report.json.
They were not fabricated or measured by repeating test inference.

A separate loaded-model idle probe made **zero review prediction calls**. It observed
server RSS 1,382,436,864 bytes and device-wide GPU memory 1153 MiB. This demonstrates
capacity to load on this machine; it is not claimed as the inference peak. Weight
size is about 1.04 GiB; runtime buffers and other GPU applications add overhead.
The sampler now persists future telemetry to logs so reporting errors cannot lose it.

## Actual failure evidence

Qwen made **237/887 errors**. Largest confusion: **121 positive→neutral** and
32 negative→neutral. Also 58 positive→negative, 12 negative→positive,
9 neutral→negative and 5 neutral→positive. Neutral overprediction is the main
calibration problem, rather than failure to recall neutral rows.

| Analysis cohort | Samples | Errors | Error rate |
|---|---:|---:|---:|
| source gold sarcasm | 37 | 9 | 24.32% |
| mixed-marker proxy | 121 | 45 | 37.19% |
| negation-marker proxy | 208 | 42 | 20.19% |
| no obvious polarity-word proxy | 630 | 200 | 31.75% |
| source gold neutral | 43 | 14 | 32.56% |
| long text, 200+ chars | 20 | 0 | 0% |
| short text, ≤20 chars | 93 | 33 | 35.48% |
| conflicting-label text rows | 24 | 16 | 66.67% |

Lexical proxies are not verified annotations for mixed/implicit sentiment. Cohorts
overlap. No evidence here justifies making long-text context the first V2 change.

81 globally conflicting normalized texts, 11 within test, remain unchanged. The
11 test texts cover 24 rows; identical input can have different aspect-level labels.
Any deterministic text-only classifier must make at least 11 errors across this test
set (maximum label-majority correct count 876/887). This does not explain all 237
Qwen errors; not every error in the conflicting cohort is intrinsically unavoidable.
Aspect is joined only for post-hoc analysis, never inference.

## Evidence-driven V2 recommendation — not implemented

First investigate neutral versus mild praise/criticism with diverse, verified
Vietnamese TRAIN-only contrast examples, selecting prompts/settings on validation
only. Evidence: 153 non-neutral rows were predicted neutral; mixed/short cohorts
also have elevated error rates. The current three-example prompt improved validation
macro F1 but did not fix neutral precision. No post-test V1 prompt changes were made.

Aspect-conditioned V2 can address the documented contradictory same-text targets,
but must be presented as a distinct input/task, not a text-only improvement.
A larger quantized Qwen or later LoRA can be separately benchmarked if contextual
errors persist; neither is implemented or proven better by this experiment.

## Reporting recovery and integrity

After all 887 outputs were checkpointed, pandas dropped the sample_index axis name
during a reset_index, causing a KeyError while assembling reports. The index handling
was fixed and a regression assertion added. Recovery reconstructed predictions and
reports from checkpoints, verifying membership, text hashes and agreement with the
already saved metrics. **No additional test or validation response was generated.**
Original execution source hashes are preserved. Repeated proposed-v1 now verifies
model/settings/prompt/data and artifact checksums, returning the original cached
execution even after reporting-only code fixes. It does not claim to evaluate new code.

Dataset SHA-256: `0264dba42d54f2b497e5ec92a56ef4311538a1d234b8d183518b12ff3cb928df`.
Split SHA-256: `860387be19c64220f91cc2b5e764aa3c99aa18daba9c776fde38e05da30a5ff9`.
The baseline evidence, weights and canonical split were only read.

## Files and validation

Created code/config: src/proposed_v1/{__init__,runtime,runner,analysis,reporting}.py;
qwen_v1.yaml; scripts/setup_qwen_v1.py; scripts/recover_qwen_reports.py;
prompts/food_sentiment/v1_a.txt, v1_b.txt, v1_c.txt; tests/test_proposed_v1.py;
docs/experiments/PROPOSED_V1.md; this report.

Changed: src/ml/cli.py; dvc.yaml/dvc.lock; tests/test_food_pipeline.py;
README.md; reports/README.md; .gitignore/.dvcignore. No Phase 1 files changed.

Generated reports/proposed_v1: prompt_manifest.json; per-prompt validation metrics,
predictions and checkpoints; qwen_prompt_comparison.csv; selection.json; test_state.json;
test_checkpoint.jsonl; qwen_test_metrics.json; qwen_predictions.parquet;
qwen_confusion_matrix.png; qwen_classification_report.csv; model_comparison.csv/.png;
qwen_failure_cases.csv; qwen_error_by_aspect.csv; failure_analysis.json;
qwen_resource_report.json. Slide evidence: reports/task2_proposed_v1_evidence.json.
Ignored assets: official GGUF weights/card/license/provenance; pinned CUDA binaries;
MLflow artifacts and DVC cache. No model weights in Git.

- Full pytest: **191 passed**, one existing Starlette/AnyIO warning.
- Ruff: **PASS**. compileall src/databricks/scripts: **PASS**.
- Actual inference: all three validation prompts and 887 selected test responses.
- DVC proposed_v1: **PASS**, cached results; upstream data/baseline stages frozen.
- MLflow: all three prompt runs FINISHED; only selected V1-C has test metrics.
- Model/runtime setup repeated safely: verified existing files, no downloads or
  changed provenance. No production registration.

Command: `python -m src.cli proposed-v1` (returns completed saved evidence now).
V2 remains a separate future experiment, requiring its own frozen protocol.
