# Task 2 text-only Vietnamese restaurant baselines

Use the existing, verified VN-MSASar v3 split without downloading or resplitting.
Only `text` is passed to either classifier. The normalized `sentiment` column is
the target; aspect/source/ordinal sentiment/rating are never model features.
The target is now explicitly text-only, overriding the aspect-conditioned baseline
idea in the initial preparation plan. Source per-aspect gold rows remain unchanged;
identical text can therefore have different target labels across aspect annotations.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.cli run-baselines
```

Separate stages: `baseline-rule`, `baseline-tfidf`, `baseline-evaluate`. All accept
`--params params.yaml`, using the existing food paths only. The loader validates
5869 rows, 4140/842/887 membership, seed 42, raw/processed/split checksums and saved
leakage protections. No preparation function or network download is invoked.

Repeated commands return frozen existing evidence. A completed test evaluation
verifies its artifact checksums and does not predict again. A durable attempt marker
precedes ANY test prediction; incomplete attempts stop for inspection instead of
silently repeating held-out evaluation. Changed protocol/model artifacts also stop.
Do not delete evaluation state to manufacture another test score.

## Rule baseline

The manually authored domain lexicon in `src/baselines/rule_based.py` is frozen
before test evaluation. `reports/baselines/rule_definition.json` contains every
positive/negative phrase, negator, intensifier, boundary and threshold.

1. NFC and whitespace normalization, then casefold for rule matching.
2. Longest non-overlapping lexicon phrase contributes +1/-1.
3. An odd number of negators in the previous three tokens flips polarity.
   Punctuation or adversatives end the scope. `không hề` counts one negator.
4. An adjacent intensifier multiplies the sentiment hit by 1.5.
5. Positive/negative total produces positive/negative; zero produces neutral.

No lexicon mining, learned weights or ML is used. Local negation is approximate;
idioms, context, aspect-specific sentiment and sarcasm are known limitations.
No rules are adjusted from held-out failure cases.

## TF-IDF / Logistic Regression

The sklearn Pipeline normalizes Vietnamese text, vectorizes it and classifies it.
Vocabulary, IDF and LR are fitted ONLY on the 4140 training texts/labels.
Accents/case are preserved by the TF-IDF preprocessing. Fixed common settings:
max_features=20000, min_df=2, sublinear_tf=True, max_iter=1500, solver=lbfgs,
class_weight=balanced, random_state=42.

Three controlled candidates: A unigrams/C=1; B unigrams+bigrams/C=1;
C unigrams+bigrams/C=2. Select by validation macro F1 with stable candidate-order
tie breaking. No test score is available to selection. The selected training-fitted
pipeline is saved as `models/tfidf_logreg.joblib`; no train+validation refit occurs.

An additional unweighted model uses the selected hyperparameters, changing ONLY
class_weight. Its comparison with balanced weighting is on VALIDATION only,
including per-class precision, recall and F1. It cannot replace the prescribed
balanced baseline using test performance. No SMOTE is used.

## Evaluation and artifacts

Both baselines receive identical validation/test text inputs. Metrics include
accuracy, macro precision/recall/F1 and weighted F1; macro F1 is primary. Latency
uses sequential single-sample predict calls (including normalization), measured
with perf_counter_ns. Three non-test warmups precede each evaluation; every held-out
sample is predicted once. Mean/p95 values are machine-specific, not a production SLA.
TF-IDF size is actual compressed joblib bytes / 1,000,000. Rule-based size is zero
learned-model bytes; rule/source text has a separate definition artifact.

`reports/baselines/` contains environment.json; validation/test metric JSON files;
TF-IDF experiment/selection evidence; class weight CSV/PNG; test classification
report CSV and confusion matrix PNG for each model; failure_cases.csv;
error_by_aspect.csv; baseline_comparison.csv/.png; evaluation_state.json.

Failure rows preserve review_id/text/aspect/true_label and both actual predictions.
False-positive/false-negative categories explicitly use positive versus the rest;
neutral-as-positive/negative are distinguished. Aspect is used for analysis only.
`reports/task2_baseline_evidence.json` incorporates the existing preparation evidence
and adds Slides 7/8/9 plus validation-only Slide 6 class-weight evidence.

## MLflow

Existing TrackingConfig/configure_tracking infrastructure is reused. Default local
backend: sqlite:///state/mlflow.db; experiment: food-restaurant-baselines. Explicit
MLFLOW_TRACKING_URI/MLFLOW_EXPERIMENT_NAME settings still take precedence. No tokens
or credentialed URI are serialized into reports.

All three balanced candidate runs log dataset/split hashes, seed, vectorizer/LR
parameters, validation metrics/confusion matrices/classification reports and complete
Pipeline model artifacts. The unweighted ablation logs validation evidence only.
Test metrics and artifacts attach only to the validation-selected balanced run.
No model registration or production alias is performed.

## DVC

The four verified food preparation stages are now frozen during baseline acceptance.
Three downstream stages are baseline_rule → baseline_tfidf → baseline_evaluate.
Model/selection/evaluation state and evidence outputs persist across reproduction.

```powershell
python -m dvc repro baseline_evaluate
```

This records/reuses the baseline outputs while preserving the frozen source and
split. Do not unfreeze preparation or force a new split during this comparison.

## Next Proposed V1 step

Define a frozen **text-only** Qwen inference/evaluation protocol on the same split,
with the same three labels and metrics. Choose prompts/settings on train/validation
only, distinguish enrichment-only fields, then obtain authorization for execution.
No Qwen implementation or run is included in this baseline task.
