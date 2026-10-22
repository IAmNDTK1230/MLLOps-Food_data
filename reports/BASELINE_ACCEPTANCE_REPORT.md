# Task 2 baseline acceptance — actual execution

Date: 2026-10-13, Asia/Saigon. **PASS: both text-only baselines executed**.
Phase 1 is unchanged. No data was redownloaded, cleaned again or resplit.
No Qwen, Proposed V1, embeddings, registration or production promotion ran.

## Frozen experiment protocol

VN-MSASar v3, vi/restaurant: 5869 rows; train 4140, validation 842, test 887; seed 42.
Only text is a feature. The target is the prepared normalized sentiment; aspect,
original_sentiment, ratings and test labels are not feature inputs.

- Dataset SHA-256: `0264dba42d54f2b497e5ec92a56ef4311538a1d234b8d183518b12ff3cb928df`.
- Split SHA-256: `860387be19c64220f91cc2b5e764aa3c99aa18daba9c776fde38e05da30a5ff9`.
- Saved review/text leakage checks remain zero. Both hashes match preparation.
- Python 3.12.14, Windows 11 (10.0.26200), AMD Ryzen 7 6800H, 16 logical CPUs.
- Available total RAM: 16,387,956,736 bytes; detailed environment/timestamp/packages
  are recorded in baselines/environment.json.

## Measured validation and test results

| Model / split | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Rule-Based / validation | 0.492874 | 0.607949 | 0.576858 | 0.425375 | 0.577602 |
| Rule-Based / test | 0.505073 | 0.619395 | 0.596898 | 0.431279 | 0.592298 |
| TF-IDF + LR / validation | 0.819477 | 0.700561 | 0.720342 | 0.709242 | 0.821549 |
| TF-IDF + LR / test | 0.809470 | 0.672730 | 0.695250 | 0.682854 | 0.812104 |

Test macro F1 difference TF-IDF minus rules: **+0.251575** (25.16 percentage points).
These are held-out observations, not a promised production improvement.

| Balanced candidate | Ngrams | C | Validation macro F1 |
|---|---|---:|---:|
| A_unigram_c1 | 1 | 1 | 0.648722 |
| B_bigram_c1 | 1–2 | 1 | 0.700294 |
| **C_bigram_c2** | **1–2** | **2** | **0.709242** |

Selected configuration: ngram_range=(1,2), C=2, max_features=20000, min_df=2,
sublinear_tf=True, max_iter=1500, solver=lbfgs, class_weight=balanced, seed=42.
Vocabulary/IDF/LR fitted on train only; no train+validation refit.
Selection used validation macro F1 only. Test prediction was performed once per
sample for the selected TF-IDF model and rule baseline. Durable completed state
and artifact checksums make repeated commands return cached evidence.

## Class-weight ablation — validation only

All other hyperparameters are identical to selected C_bigram_c2.

| Class | Weight | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| negative | None | 0.813230 | 0.774074 | 0.793169 |
| negative | balanced | 0.750000 | 0.833333 | 0.789474 |
| neutral | None | 1.000000 | 0.093023 | 0.170213 |
| neutral | balanced | 0.456522 | 0.488372 | 0.471910 |
| positive | None | 0.839931 | 0.922495 | 0.879279 |
| positive | balanced | 0.895161 | 0.839319 | 0.866341 |

Balanced weighting improves neutral recall/F1 substantially, with precision and
majority-class tradeoffs. No resampling, SMOTE or fabricated AFTER counts are used.
The unweighted model was not evaluated on test or used to choose test performance.
Slide 6 chart: baselines/class_weight_comparison.png.

## Measured latency and size

| Model | Test mean ms/sample | Test p95 ms/sample | Saved learned model MB |
|---|---:|---:|---:|
| Rule-Based | 0.023798 | 0.048480 | 0 |
| TF-IDF + LR | 0.365186 | 0.417170 | 0.342442 |

Sequential batch size 1; three non-test warmups; normalization included. Timing is
specific to this machine/session. Model size is compressed joblib bytes / 1,000,000.
Rule-Based has no learned artifact; its source/lexicon definition is stored separately.
Selected model: models/tfidf_logreg.joblib, SHA-256
`212f4c36ca6f5e54f34401048689545606b20033828db95b02db7636107f85b6`.

## Actual failure analysis

Rule baseline: 439/887 errors, including 215 positive→neutral and 179 negative→neutral.
Its fixed vocabulary often produces a zero total for sentiment-bearing text. There
are 32 positive-class false positives and 228 positive-class false negatives.

TF-IDF: 169/887 errors. Largest confusions are 68 positive→negative and 47
negative→positive. Neutral remains weakest: precision 0.387755, recall 0.441860,
F1 0.413043; 14 neutral→positive and 10 neutral→negative, out of 43 neutral examples.
Positive-class false positives total 61; false negatives total 89.

By aspect, TF-IDF has 56 food errors/347 rows, 35 price/139, 29 overall/117,
24 service/159, 15 ambiance/71, 8 location/25 and 2 cleanliness/29. Location's
32% error rate has only 25 samples; food has the largest absolute count.
Amenities and room have no test support, so no test performance is claimed for them.

There are 81 globally normalized texts with multiple gold sentiment labels across
aspect rows, including 11 within test. This creates ambiguity for text-only inputs;
aspect labels and predictions remain unchanged. Rule negation is deliberately local;
sarcasm/context/idioms are limitations, not additional models implemented here.

Full actual sentences, review IDs, aspects and both predictions are in
baselines/failure_cases.csv; representative saved examples and confusion counts
are in baselines/failure_analysis_summary.json. No post-test lexicon/model tuning ran.

## MLflow and DVC evidence

Existing local MLflow backend sqlite:///state/mlflow.db, experiment
food-restaurant-baselines. Successful candidate run IDs:

- A: `1876c5b6e3f24f3cb4ebe04b4de03791`.
- B: `9f3b598de8314630a55218e1f9a299fa`.
- Selected C: `d5de702307104c81a901839b7937987e`.

All three runs are FINISHED and log validation metrics, hashes/parameters, confusion
matrices/classification reports and complete sklearn Pipeline model artifacts.
Only selected C has test metrics/artifacts. The separate unweighted run is validation
ablation only. No model was registered or designated production.

DVC baseline_rule/baseline_tfidf/baseline_evaluate stages are recorded in dvc.lock.
The four preparation stages are frozen and were skipped during DVC reproduction.
Repeated evaluation reused saved predictions/results. Slide reporting refresh reads
saved CSV predictions only and corrects literal class_weight=None serialization;
it does not refit or predict. Strict JSON evidence is available in
reports/task2_baseline_evidence.json, including existing Slides 3–6 and new Slides 7–9.

## Files created/changed

Created: src/baselines/__init__.py, rule_based.py, tfidf_logreg.py, metrics.py,
runner.py, reporting.py; tests/test_food_baselines.py; docs/experiments/BASELINES.md;
reports/BASELINE_ACCEPTANCE_REPORT.md.

Changed: src/ml/cli.py; dvc.yaml; dvc.lock; tests/test_food_pipeline.py (frozen DAG
expectations); README.md; docs/experiments/PHASE2.md; .gitignore.

Generated: models/tfidf_logreg.joblib; reports/baselines/environment.json,
rule_definition.json, rule_based_validation_metrics.json, rule_based_test_metrics.json,
tfidf_validation_metrics.json, tfidf_test_metrics.json, tfidf_selection.json,
tfidf_experiments.csv, class_weight_comparison.csv/.png, rule_based and tfidf
confusion_matrix.png/classification_report.csv, failure_cases.csv, error_by_aspect.csv,
failure_analysis_summary.json, baseline_comparison.csv/.png, evaluation_state.json,
per-candidate validation confusion plots; reports/task2_baseline_evidence.json;
local MLflow database/model artifacts and DVC cache entries.

## Validation and exact next step

- python -m pytest -q: **174 passed**, one existing Starlette/AnyIO warning.
- python -m ruff check .: **PASS**.
- python -m compileall -q src databricks scripts: **PASS**.
- python -m src.cli run-baselines: **PASS**, real train/validation/test execution.
- python -m dvc repro baseline_evaluate: **PASS**, frozen preparation preserved.
- MLflow verification: candidate runs FINISHED; only selected run has test metrics.

Initial sandbox artifact logging hit PermissionError; the authorized execution was
retried outside the sandbox before any test evaluation. No dataset download/resplit
or repeated held-out prediction was used to resolve it.

Command: `python -m src.cli run-baselines` (returns frozen completed evidence now).
Next Proposed V1 step: define and review a frozen text-only Qwen prompt/inference
protocol on these exact splits, using train/validation for prompt selection and the
same three labels/metrics. Choose model/runtime/latency protocol and keep enrichment
fields separate. Qwen implementation/execution remains a separate future request.
