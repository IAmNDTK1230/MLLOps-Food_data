# Proposed V2-A measured acceptance report

Recorded 2026-10-15 (Asia/Saigon). Implementation and actual inference completed.
**V2-A did not improve V1 or surpass TF-IDF on macro F1 or neutral F1.**
The calibration hypothesis was not supported by this experiment.

## Frozen protocol and test hygiene

VN-MSASar v3: 5,869 rows, TRAIN 4,140 / VALIDATION 842 / TEST 887, seed 42.
Only `data/food/splits/reviews.parquet` supplies benchmark membership. No preparation,
splitting, baseline training, V1 inference, fine-tuning or Phase 1 changes occurred.
The final SHA audit confirmed all 60 protected input/artifact files unchanged.

Dataset SHA256: `0264dba42d54f2b497e5ec92a56ef4311538a1d234b8d183518b12ff3cb928df`.
Split SHA256: `860387be19c64220f91cc2b5e764aa3c99aa18daba9c776fde38e05da30a5ff9`.

Only TRAIN supplied examples; only VALIDATION selected prompts. Historical V1 test
outcomes had already been inspected. The final reused TEST result is therefore
**exploratory**, not untouched independent confirmation. No test sentences, labels,
or saved failure examples were used to construct prompts or choose settings.
Post-hoc comparisons began after selected V2 inference. Test completed once with
887 unique indices, 887 attempts, no retries or invalid structured outputs.

Input is text only; no aspect, sarcasm, review ID, source or held-out gold label.
Output is strict single-field JSON with negative/neutral/positive schema validation.
Invalid outputs are counted with one permitted retry; no arbitrary text mapping.
Completed checkpoints are immutable and return saved results without inference.

## Model and runtime

Same as V1: Qwen/Qwen2.5-1.5B-Instruct-GGUF (1.54B parameters), Q4_K_M,
revision `91cad51170dc346986eccefdc2dd33a9da36ead9`.
Weights: 1,117,320,736 bytes, SHA256
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
llama.cpp b11429 (`d81235049384534c167caea52b85a694f6103d14`), CUDA 12.4;
RTX 3060 Laptop GPU (6 GB), Ryzen 7 6800H, 16 logical CPUs, approximately 16 GB RAM.
No model/runtime download or model-family change was needed.

Temperature 0, seed 42, max_new_tokens 32, context 2048, CPU threads 8,
requested GPU layers 99, one slot, batch 512 / ubatch 256, cache_prompt=False.
Runtime config equals the frozen V1 selection settings; model/server hashes verified.

## TRAIN contrasts and validation selection

The deterministic selector excludes TRAIN texts with conflicting labels, sorts by
review_id/text/sentiment, applies fixed lexical/length predicates and chooses unused
texts. It records IDs, text, original gold label, selection reason and split=train
in `proposed_v2/train_example_manifest.json`. Categories are selection proxies,
not new independently annotated labels. The original labels remain unchanged.

Candidates were frozen before inference under `prompts/food_sentiment_v2/`:

| Prompt | Calibration change | Examples | SHA256 |
|---|---|---:|---|
| V2-A1 | Tighter neutral definition | 6 | `8af1d035bef40b1b293caead4349fe6ba99a613d2759d9f1cbd07b720c2c086c` |
| V2-A2 | Same definitions plus mild positive/negative contrasts | 9 | `25fde9ff9b72bffd6c90540d6d53387bb9f1a9a88df52538cbd368975ce1fba7` |
| V2-A3 | Decision-oriented dominant-polarity guidance | 8 | `bffe4fc3bdb479c615d77f69564bcfe4515c1a6dec6f882aadf3e181f26d81c6` |

Neutral requires no clear positive/negative judgment. Mild wording can carry
polarity; short/mixed/factual phrasing and absent strong adjectives do not
automatically imply neutral. Examples cover clear/mild polarity, true neutral,
mixed sentiment, negation and short ambiguity.

| Validation prompt | Accuracy | Macro F1 | Neutral precision | Neutral recall | Neutral F1 | Neutral predictions |
|---|---:|---:|---:|---:|---:|---:|
| Historical V1-C | 0.758907 | 0.641252 | 0.187500 | 0.627907 | 0.288770 | 144 |
| V2-A1 | 0.730404 | 0.623801 | 0.165854 | 0.790698 | 0.274194 | 205 |
| V2-A2 | 0.738717 | 0.623353 | 0.168539 | 0.697674 | 0.271493 | 178 |
| **V2-A3** | **0.750594** | **0.632579** | **0.177515** | **0.697674** | **0.283019** | **169** |

V2-A3 won among V2 candidates by validation macro F1. No tie-break was needed.
All V2 candidates scored below V1-C on validation macro F1 and neutral F1.
Higher neutral recall did not determine selection. Complete precision/recall,
weighted F1, JSON validity and resource values are in `prompt_comparison.csv`,
`v1_vs_v2_validation.csv` and per-candidate classification reports.

## One final test evaluation

| Model | Accuracy | Macro F1 | Weighted F1 | Neutral F1 | Mean latency ms | p95 ms | JSON validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-Based | 0.505073 | 0.431279 | 0.592298 | 0.167715 | 0.0238 | 0.0485 | N/A |
| TF-IDF + Logistic Regression | 0.809470 | 0.682854 | 0.812104 | 0.413043 | 0.3652 | 0.4172 | N/A |
| Qwen V1 | 0.732807 | 0.621789 | 0.774715 | 0.257778 | 93.6603 | 116.3429 | 100% |
| Qwen V2-A3 | 0.723788 | 0.614487 | 0.778542 | 0.247104 | 378.2446 | 1053.2570 | 100% |

V2 macro precision 0.646854; macro recall 0.728008. There were 642 correct
predictions and 245 errors. Historical metrics are read from frozen artifacts.
Latency measurements occurred in different sessions and are not a controlled
causal comparison of prompt changes alone.

| Required delta | Measured difference |
|---|---:|
| V2 macro F1 − V1 macro F1 | −0.007302 |
| V2 macro F1 − TF-IDF macro F1 | −0.068367 |
| V2 neutral F1 − V1 neutral F1 | −0.010674 |
| V2 neutral F1 − TF-IDF neutral F1 | −0.165939 |

**No positive improvement signal on these four measures.** Weighted F1 increased
slightly versus V1 (+0.003827), but the primary metric and neutral F1 deteriorated.
V2 is not accepted as a better classifier or production model.

## Neutral behavior and failures

| Test neutral measure | V1 | V2 | Change |
|---|---:|---:|---:|
| Precision | 0.159341 | 0.148148 | −0.011193 |
| Recall | 0.674419 | 0.744186 | +0.069767 |
| F1 | 0.257778 | 0.247104 | −0.010674 |
| Predicted neutral | 182 | 216 | +34 |
| Correct neutral | 29 | 32 | +3 |

True neutral support is 43; V2 produced 184 false-positive neutral predictions.
The prompt change increased neutral overprediction rather than resolving it.

| Confusion | V1 | V2 | Change |
|---|---:|---:|---:|
| positive → neutral | 121 | 120 | −1 |
| negative → neutral | 32 | 64 | +32 |
| positive → negative | 58 | 32 | −26 |
| negative → positive | 12 | 18 | +6 |

| Test cohort | Rows | V1 errors | V2 errors |
|---|---:|---:|---:|
| Short, <=20 characters | 93 | 33 | 26 |
| Negation proxy | 208 | 42 | 52 |
| Mixed sentiment proxy | 121 | 45 | 34 |
| Source sarcasm annotation | 37 | 9 | 14 |
| Implicit sentiment word-absence proxy | 630 | 200 | 208 |

Cohorts overlap. Lexical/length proxies do not establish verified mixed/implicit
phenomena or causal explanations. Validation cohorts and cases are separately
saved in `validation_failure_analysis.json` and candidate failure CSVs.
Test rows/cases/aspects are used only after inference and never alter predictions.

Post-hoc V2 errors by aspect: ambiance 31/71, cleanliness 11/29, food 61/347,
location 22/25, overall 28/117, price 58/139, service 34/159. Small groups have
unstable rates; these results do not justify feeding aspect into this comparison.
There are 81 globally conflicting normalized texts and 11 in test. V2 makes 17
errors among conflicting-text rows. Text-only prediction cannot distinguish
identical text with different aspect-level gold labels.

## Tokens, latency and observed resources

| Evaluation | Input tokens | Output tokens | Mean ms | p95 ms | Duration s |
|---|---:|---:|---:|---:|---:|
| A1 validation | 414465 | 5894 | 110.992 | 123.053 | 94.657 |
| A2 validation | 524767 | 5894 | 135.551 | 148.016 | 115.963 |
| A3 validation | 534029 | 5894 | 287.467 | 1003.322 | 249.584 |
| A3 test | 561414 | 6209 | 378.245 | 1053.257 | 369.946 |

Validation + test: 2,034,675 input + 23,891 output = **2,058,566 tokens**.
Three TRAIN warmups separately used 1,740 input + 21 output tokens. These API
counts exclude native server startup/dummy work. Test: 567,623 total tokens,
639.936 mean tokens/sample, p95 input 659, maximum input 776; throughput 2.398
samples/s and 16.784 output tokens/s. All measured outputs were valid first attempt.

One-second session sampling (including warmups) observed peak server RSS
1,439,875,072 bytes (1.44 GB decimal), mean server CPU 88.39% on the scale where
one core is 100%, peak device-wide GPU memory 5,710 MiB and mean device-wide GPU
utilization 83.87%. GPU values can include other applications; they are not an
isolated model VRAM requirement. RSS excludes GPU memory. Latency tails were
observed; no controlled attribution to contention or prompt length is claimed.

## Future protocols only

`proposed_v2/aspect_conditioned_protocol.md` recommends a separately declared
Qwen(text + aspect) experiment (V2-B / V3), motivated by conflicting aspect labels.
It requires its own validation-only selection, immutable label/input contract and
appropriate aspect-aware comparisons. It is a distinct task, not directly
equivalent to text-only baselines. No implementation or fine-tuning was added.

`proposed_v2/external_evaluation_recommendation.md` recommends later untouched
licensed Restaurant data, potentially VLSP2018. Official access/user-agreement
requirements must be satisfied first: https://vlsp.org.vn/resources-vlsp2018 .
ABSA aspect sentiment is not automatically global review sentiment. Pre-register
the mapping/task, audit exact/near/source overlap with all VN-MSASar, withhold final
labels from development and freeze every model/prompt before one external scoring.
Project-untouched does not establish absence from Qwen pretraining. Prefer a newly
collected licensed corpus if adequate independence cannot be established.
No external data was downloaded/evaluated and no access request was sent.

## Artifacts, tracking and verification

Implementation: `src/proposed_v2/{examples,runner,analysis}.py`, package initializer,
`qwen_v2.yaml`, three prompt files and `tests/test_proposed_v2.py`.
Integration: `src/ml/cli.py`, `tests/test_food_pipeline.py`, `dvc.yaml`, `dvc.lock`,
`.gitignore`, `README.md`, `reports/README.md`, `docs/experiments/PROPOSED_V2.md` and this report.

`proposed_v2/` contains example/prompt manifests, protocol/selection snapshots,
validation checkpoints/predictions/metrics/classification reports/failures,
`prompt_comparison.csv`, `v1_vs_v2_validation.csv`, validation diagnostics/plot,
completed test checkpoint/state, `test_metrics.json`, `predictions.parquet`,
`classification_report.csv`, `confusion_matrix.png`, `comparison.csv/.png`,
`confusion_changes.csv`, `failure_cases.csv`, `error_by_aspect.csv`,
`failure_analysis.json`, resource reports and both future protocols.
`task2_proposed_v2_evidence.json` provides slide-ready evidence for slides 11–14.
Generated comparison/validation plots were visually checked.

All three MLflow runs are FINISHED and contain validation metrics, model/prompt/
data hashes, settings, neutral metrics, latency and tokens. Only selected v2_a3
contains final test metrics. No production model was registered.

| Verification | Result |
|---|---|
| `python -m pytest -q` | PASS: 200 tests; one existing Starlette/AnyIO deprecation warning |
| `python -m ruff check .` | PASS |
| `python -m compileall -q src databricks scripts` | PASS |
| `python -m src.cli proposed-v2-prepare` | PASS: TRAIN manifests/prompts frozen |
| `python -m src.cli proposed-v2` | PASS: 3 validation runs + selected test executed |
| `python -m dvc repro proposed_v2` | PASS: prior stages frozen; completed V2 returned cached results; lock updated |
| Protected artifact SHA audit | PASS: 60 unchanged |
| MLflow selected-only test logging | PASS |

Run commands from the project virtual environment as documented in `docs/experiments/PROPOSED_V2.md`.
A completed `python -m src.cli proposed-v2` validates recorded hashes and returns
saved results without further inference. Do not delete checkpoints to retune on test.
