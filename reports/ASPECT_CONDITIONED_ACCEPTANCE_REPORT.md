# Proposed V2-B / V3 — measured text + aspect experiment

Recorded 2026-10-15 (Asia/Saigon). Both new experiments were implemented and
executed on the frozen VN-MSASar v3 split. **Qwen aspect-aware did not beat
aspect-aware TF-IDF overall.** There is a small descriptive improvement over
historical text-only V1 on conflicting texts, with substantial remaining failures.

## Scientific framing and immutable inputs

This is a distinct input protocol: **text + one original aspect**. The fair
classical control also receives text + aspect. Historical text-only results are
descriptive comparisons with different inputs, not apples-to-apples superiority.

Use exactly `data/food/splits/reviews.parquet`: TRAIN 4140 / VALIDATION 842 /
TEST 887, total 5869, seed 42. No data preparation, split, old baseline fitting,
V1 inference or V2-A inference was repeated. Phase 1 remains unchanged.
All 106 protected data/model/report/Phase 1 files passed final checksum checks.
Existing review ID / normalized-text leakage protections remain active.

Dataset SHA256: `0264dba42d54f2b497e5ec92a56ef4311538a1d234b8d183518b12ff3cb928df`.
Split SHA256: `860387be19c64220f91cc2b5e764aa3c99aa18daba9c776fde38e05da30a5ff9`.

Only TRAIN constructs examples and fits classical vocabulary/IDF/coefficients.
Only VALIDATION selects both models. Both selections were frozen before either
new TEST evaluation. Scored input excludes gold sentiment, original_sentiment,
sarcasm, review_id, source and split. No labels or predictions were modified.
V1/V2-A test outcomes had already been inspected: these reused test results are
**exploratory** and require untouched licensed external confirmation later.

## Classical control

Input: `__aspect_food__ <text>` (using each row's original aspect). TF-IDF uses
max_features=20000, min_df=2, sublinear_tf=True, preserved Vietnamese accents;
Logistic Regression uses class_weight=balanced, random_state=42, max_iter=1500.
Vocabulary/IDF/classifier are fit on TRAIN alone. No post-selection validation refit.

| Validation trial | Accuracy | Macro F1 | Neutral F1 |
|---|---:|---:|---:|
| Unigram, C=1 | 0.779097 | 0.660046 | 0.372093 |
| Unigram + bigram, C=1 | 0.819477 | 0.725451 | 0.520000 |
| **Unigram + bigram, C=2** | **0.828979** | **0.725912** | **0.505747** |

Macro F1 selected C=2. Test accuracy 0.807215, macro precision 0.654164,
macro recall 0.680324, macro F1 0.665514, weighted F1 0.810875. Neutral
precision/recall/F1: 0.320755 / 0.395349 / 0.354167. Serialized model 0.360566 MB.

The requested aspect token is a minimal additive control. It cannot represent
all aspect-specific word interactions; this limitation matters for future work.

## Qwen model, prompts and selection

Same model/runtime as V1: Qwen/Qwen2.5-1.5B-Instruct-GGUF (1.54B), Q4_K_M,
revision `91cad51170dc346986eccefdc2dd33a9da36ead9`, llama.cpp b11429 CUDA 12.4.
Weight SHA256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
Existing model/server assets were checksum-verified and reused; no download.
RTX 3060 Laptop GPU, Ryzen 7 6800H. Temperature 0, seed 42, max_new_tokens 32,
context 2048, CPU threads 8, requested GPU layers 99; all V1 runtime settings retained.

| Prompt | Design | Validation accuracy | Macro F1 | Neutral F1 | Mean / p95 ms |
|---|---|---:|---:|---:|---:|
| V2-B1 | Zero-shot text + aspect | 0.768409 | 0.620898 | 0.235294 | 68.992 / 73.789 |
| V2-B2 | Aspect + sentiment definitions | 0.786223 | 0.601223 | 0.184211 | 98.924 / 107.234 |
| **V2-B3** | Definitions + five TRAIN examples | **0.807601** | **0.626947** | **0.198020** | **180.898 / 352.058** |

V2-B3 selected by validation macro F1. No tie-break was needed. All candidates
had 100% JSON validity. Full precision/recall/weighted and per-class metrics are
in `aspect_conditioned/qwen_prompt_comparison.csv` and classification reports.

Prompt SHA256:

- B1: `8d19050db66c593018689f11f65424447bc80ea5f4953941bcedd52b23186bec`
- B2: `d3a36f591c9d55ac2ab903cd64f116e4a30a15df5ebebf5ec39349f9779d5f46`
- B3: `526d9b92283a5ddcd88b6fef92c8dc7a7e2be3cfbd445ff1879c54034ed33abf`

Examples come from deterministic TRAIN-only ordering, including the same text
about fresh beef and non-cheap price with positive food / negative price labels.
Manifest records review_id/text/aspect/gold label/split/selection reason; model
examples expose only text+aspect and TRAIN example answers. Prompts were frozen
before validation and not edited after scores were observed.

## Final test and improvement assessment

| Model | Input protocol | Accuracy | Macro F1 | Weighted F1 | Neutral F1 |
|---|---|---:|---:|---:|---:|
| Rule-Based | text-only, historical | 0.505073 | 0.431279 | 0.592298 | 0.167715 |
| TF-IDF | text-only, historical | 0.809470 | 0.682854 | 0.812104 | 0.413043 |
| Qwen V1 | text-only, historical | 0.732807 | 0.621789 | 0.774715 | 0.257778 |
| Qwen V2-A | text-only, historical | 0.723788 | 0.614487 | 0.778542 | 0.247104 |
| **TF-IDF** | **text + aspect** | **0.807215** | **0.665514** | **0.810875** | **0.354167** |
| **Qwen B3** | **text + aspect** | **0.786922** | **0.602203** | **0.801965** | **0.148760** |

Qwen test macro precision 0.596315, macro recall 0.626240; 698/887 correct.
Strict schema JSON validity 100%, first-attempt validity 100%, invalid outputs 0.
The test checkpoint contains exactly 887 unique responses/attempts, with no retries.

Fair Qwen minus aspect-TF-IDF differences:

- Accuracy **−0.020293**.
- Macro F1 **−0.063311**.
- Neutral F1 **−0.205406**.

**No overall positive improvement signal over the fair control.** Higher overall
accuracy than old V1 does not compensate for worse macro F1 or establish a better
text-only model. Different-input historical macro F1 deltas: Qwen aspect minus
V1 −0.019586; aspect-TF-IDF minus text-only TF-IDF −0.017340.

Qwen neutral precision/recall/F1 is **0.115385 / 0.209302 / 0.148760**, versus
control **0.320755 / 0.395349 / 0.354167**. Qwen predicted neutral 78 times with
9 correct (43 true neutral rows). Fewer neutral predictions than old V1/V2-A
did not solve neutral classification: both precision and recall remain poor.

## Conflicting texts: 11 texts / 24 test rows

Membership uses the existing normalization across the frozen dataset. There are
81 global conflicting texts, including 11 in TEST; their test rows total 24.

| Model | All-test accuracy / macro F1 | Conflicting accuracy / macro F1 | Nonconflicting accuracy / macro F1 |
|---|---:|---:|---:|
| TF-IDF text-only | 0.809470 / 0.682854 | 0.500000 / 0.323810 | 0.818076 / 0.689077 |
| Qwen V1 text-only | 0.732807 / 0.621789 | 0.333333 / 0.273684 | 0.743917 / 0.632510 |
| TF-IDF text+aspect | 0.807215 / 0.665514 | 0.500000 / 0.323810 | 0.815759 / 0.671895 |
| Qwen text+aspect | 0.786922 / 0.602203 | 0.458333 / 0.329883 | 0.796060 / 0.609705 |

Descriptive Qwen versus historical V1: conflicting accuracy +0.125000 (8→11
correct), macro F1 +0.056198. TF-IDF aspect versus historical text-only control:
no conflicting-subset gain. Fair Qwen versus aspect-TF-IDF: accuracy −0.041667
(11 vs 12 correct), macro F1 +0.006073. This mixed, small-subset signal does not
establish superiority or causality. Prompts also changed, so aspect input alone
cannot be credited as the cause.

Nonconflicting subset has 863 rows. Subset macro F1 always uses all three labels;
absent labels contribute zero. No true neutral labels occur in the conflicting
subset, so its neutral F1 is zero and does not measure neutral discrimination.
Historical TF-IDF outputs were reconstructed from the exhaustive saved union-of-
baseline-errors CSV; absent rows were correct for both models. Reconstructed
aggregate metrics match frozen reports, with no old model inference.

## Metrics by aspect (fair new models)

| Aspect | Test n | TF-IDF accuracy | Qwen accuracy | TF-IDF macro F1 | Qwen macro F1 |
|---|---:|---:|---:|---:|---:|
| food | 347 | 0.835735 | 0.832853 | 0.649863 | 0.584069 |
| service | 159 | 0.886792 | 0.905660 | 0.594998 | 0.609498 |
| price | 139 | 0.719424 | 0.539568 | 0.668509 | 0.448224 |
| ambiance | 71 | 0.802817 | 0.845070 | 0.653050 | 0.693711 |
| overall | 117 | 0.717949 | 0.794872 | 0.567602 | 0.598677 |
| cleanliness | 29 | 0.931034 | 0.965517 | 0.619281 | 0.643232 |
| location | 25 | 0.680000 | 0.360000 | 0.586942 | 0.382574 |

Room/amenities have no supported TEST rows to report. Small groups and absent
labels limit per-aspect macro F1 interpretation. Price and location remain major
weaknesses despite gains on some other aspects.

## Failure evidence

189 Qwen errors are saved with original indices/text/aspect/gold/prediction.
Wrong-aspect interpretation is an observational hypothesis, not an inferred
internal explanation: CSV marks cases as not manually adjudicated.

Illustrative post-hoc cases (no label/prediction edits):

- Rows 81–83: the same text praises food/value and criticizes cramped space.
  Qwen correctly predicts negative ambiance but predicts neutral food and
  negative price despite both gold labels being positive: incomplete target
  separation, consistent with sentiment bleeding across aspects.
- Row 842: a text praises food and describes small space; ambiance gold negative
  is predicted positive. Row 843 food positive is correct. This shows remaining
  same-text/aspect-specific difficulty.
- Row 119: an explicitly easy-to-find location is predicted neutral, gold positive.
- Row 180: a rhetorical question about generous value for 30k is predicted
  negative, gold positive; implicit value judgments remain difficult.
- Row 115: a numeric-price statement has positive gold but negative prediction.
  Some labels require interpretation not transparently recoverable from text;
  aspect input does not remove every annotation/context ambiguity.

Overlapping cohorts: short <=20 characters 21/93 errors; negation proxy 35/208;
mixed proxy 29/121; source sarcasm annotation 6/37; implicit word-absence proxy
163/630. These proxy cohorts do not establish verified linguistic causes.
Validation failure cohorts are separately saved and never use TEST for selection.

Test confusions: positive→neutral 58, negative→neutral 11, positive→negative 66,
negative→positive 20; neutral→negative 22 and neutral→positive 12. The saved
confusion matrix was visually checked.

## Tokens, latency and resources

Qwen validation: 932,171 input + 18,450 output tokens. TEST: 507,723 input +
7,030 output = **514,753 tokens**. Total scored reviews across validation/test:
**1,465,374 tokens** (1,439,894 input + 25,480 output). Two recorded TRAIN
warmups add 944 input + 14 output; one original warmup measurement is unavailable
and explicitly counted as missing, not fabricated as zero.

TEST mean tokens/sample 580.330, p95 input 598, maximum input 715. Evaluation
duration 337.657 s, throughput 2.627 samples/s. Qwen mean latency **340.127 ms**,
p95 **703.440 ms**. Aspect-TF-IDF mean **0.776 ms**, p95 **1.296 ms**. These are
local single-sample measurements, not production benchmarks; separate session
conditions prevent attributing latency differences solely to prompt design.

Main resumed GPU session (B2/B3/test): peak sampled server RSS 1,424,355,328 bytes
(1.424 GB decimal), mean server CPU 84.61% where one core=100%, peak device-wide
GPU memory 5,909 MiB, mean device-wide GPU utilization 79.92%. First B1 session:
peak server RSS 1,415,786,496 bytes, mean CPU 95.95%, peak device-wide GPU memory
1,162 MiB, mean device-wide utilization 74.15%. One-second sampling can miss peaks;
GPU values can include other applications and are not isolated model VRAM needs.

## Recovery integrity, tracking and tests

B1 completed 842 validation responses before a pandas index-name reporting bug.
Original responses and MLflow measurements were recovered before TEST without
inference. Original checkpoint/protocol/resources are preserved; only reporting
source/protocol audit metadata was updated before continuing validation.

A second index-name issue affected post-hoc analysis after all 887 TEST responses
were durably saved. Explicit index-name restoration fixed reporting. An audit pins
the exact corrected reporting sources, preserves the original inference protocol
and source copies, and rejects any input/model/prompt/settings change. Reporting
recovery requires complete test measurements and cannot enter model inference.
No validation or test prediction was repeated during either recovery.

All six MLflow runs (3 new classical trials + 3 Qwen candidates) are FINISHED.
Validation metrics/hashes/protocol/settings/resources are logged; only selected
classical and Qwen runs receive TEST metrics. No production registry operations.
`proposed_v2_aspect` DVC stage completed using cached verified results and updated
`dvc.lock`. Previous stages remained frozen. An initial DVC attempt selected the
system Python 3.9; activating the project environment resolved that child-PATH
issue without inference or dependency installation.

| Verification | Result |
|---|---|
| `python -m pytest -q` | PASS: 210 tests, one existing Starlette/AnyIO deprecation warning |
| `python -m ruff check .` | PASS |
| `python -m compileall -q src databricks scripts` | PASS |
| Actual 3 classical validation trials + selected test | PASS |
| Actual 3 Qwen validation candidates + selected once-only test | PASS |
| `python -m scripts.verify_aspect_results` | PASS: 106 protected files, both 887-row outputs, selected-only MLflow test logging |
| `python -m dvc repro proposed_v2_aspect` (project environment active) | PASS; cached results, no additional inference |

## Files and next research step

Created `src/aspect_conditioned/{prompts,runtime,runner,analysis}.py`, initializer,
`qwen_aspect.yaml`, three `prompts/food_sentiment_aspect/v2_b*.txt` files,
`tests/test_aspect_conditioned.py`, recovery/verification scripts,
`docs/experiments/ASPECT_CONDITIONED.md`, this report, `reports/aspect_conditioned/` artifacts and
`reports/task2_aspect_v2_evidence.json` (slides 12–14). Updated CLI dispatch,
food DAG test, README/report index, ignore rules, `dvc.yaml` and `dvc.lock`.

Required artifacts include classical validation/test files; Qwen prompt comparison,
selection, test metrics/predictions/confusion matrix/classification report;
`conflicting_text_comparison.csv`, `metrics_by_aspect.csv`, `failure_cases.csv`,
`comparison.csv`, validation diagnostics, resource reports and recovery audits.
Weights remain ignored by Git; new classical models are captured by DVC.

Recommended next experiment: pre-register an aspect-by-text interaction classical
control on TRAIN/VALIDATION, plus a matched Qwen aspect-focused prompt ablation
with TRAIN examples covering implicit price/location and neutral boundaries.
The minimal additive control showed no conflicting-subset gain, and Qwen still
fails explicit target separation, price and neutral judgments. Freeze all choices
before untouched licensed external confirmation. Do not keep optimizing this
reused test. No LoRA/QLoRA, larger model, RAG, embeddings or serving was implemented.

Commands are in `docs/experiments/ASPECT_CONDITIONED.md`. Completed CLI/DVC execution returns saved
results; it does not repeat test inference. Read-only integrity check:

```powershell
python -m scripts.verify_aspect_results
```
