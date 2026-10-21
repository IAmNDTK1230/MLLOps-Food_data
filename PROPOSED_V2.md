# Proposed V2-A — calibrated text-only Qwen

V2-A is a prompt-only calibration experiment on the existing model and fixed data.
No dataset preparation, splitting, baseline fitting, V1 inference, fine-tuning,
aspect-conditioned implementation, external download or model registration runs.

## Research/test hygiene

V1 test outcomes were previously inspected. The user-supplied historical finding
of neutral overprediction motivates a hypothesis, not a source of prompt sentences.
Only TRAIN rows construct examples; only VALIDATION scores select the V2 prompt.
No test sentences, labels, saved failure examples or per-row V1 test predictions
are used in either step. Historical test files are byte-hashed for immutability,
not parsed for prompt construction. Test predictions are compared post-hoc only.

The final reused test evaluation is **exploratory**, not untouched independent
confirmation. This caveat travels with metrics, selection and slide evidence.
An untouched, licensed, overlap-audited external confirmation set is required later.

## Model/settings remain equal to V1

Qwen/Qwen2.5-1.5B-Instruct-GGUF, revision
91cad51170dc346986eccefdc2dd33a9da36ead9, Q4_K_M, llama.cpp b11429 CUDA 12.4.
RTX 3060 Laptop GPU; 99 requested GPU layers, 8 CPU threads, context 2048,
temperature 0, seed 42, max_new_tokens 32, one slot, cache_prompt=False.
The V1 config is read directly and checked against its frozen selection snapshot;
model/server executable checksums are verified. No setup/downloader is invoked.

The shared LocalQwen inference client/parser is reused; instantiating it does not
invoke the V1 experiment. V2 uses separate reports, runs and checkpoints. The shared
low-level server/telemetry log filenames retain their historical qwen_v1 prefix;
no frozen V1 report/model file is rewritten. V2 telemetry is durably snapshotted
in its own resource_session.json even if downstream reporting fails.

## TRAIN-only selector and frozen candidates

The selector requires every supplied row to have split=train and projects only
review_id/text/normalized sentiment/split. It excludes exact-text label conflicts
within TRAIN, uses text length 8..150, and requires a Vietnamese diacritic as a
language proxy. This does not relabel/filter the evaluation dataset.

Fixed lexical/length predicates select the first unused text sorted by
review_id/text/sentiment for each category: clear positive, clear negative, true
neutral, mild positive, mild negative, mixed, negation, short ambiguous, plus a
second neutral contrast. No fallback to held-out rows occurs. These categories
are documented selection proxies, not additional independently annotated gold
phenomenon labels. Original normalized sentiment labels remain intact.

Manifest: reports/proposed_v2/train_example_manifest.json, including IDs, text,
gold label, selection reason and split=train. Model examples contain only text
and the explicitly permitted TRAIN sentiment, never IDs/aspect/source/scores.

- V2-A1: precise neutral definition and six contrasts.
- V2-A2: same definitions and nine contrasts, including mild positive/negative.
- V2-A3: decision-oriented guidance and eight contrasts.

All specify that mild evaluations have polarity, absence of strong adjectives is
not neutral, and mixed statements use a clearly dominant polarity when present.
Neutral is reserved for no clear positive/negative judgment. Prompt files live in
prompts/food_sentiment_v2, with hashes frozen before validation. No prompt edits
are made after seeing experiment outcomes.

## Execution, selection and once-only test

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.cli proposed-v2-prepare
python -m src.cli proposed-v2
```

The same 842 validation rows are evaluated for all three prompts. Selection is
validation macro F1, then neutral precision, JSON validity and lower mean latency
only for ties. Neutral F1/precision/recall and prediction counts are diagnostics;
high neutral recall cannot override worse macro F1.

Only after selection.json is frozen does the selected model receive the 887 test
texts. User content is JSON with only text. Strict JSON constrained decoding and
schema parsing accept exactly one sentiment value in negative/neutral/positive.
Invalid responses are counted/retried once; terminal invalid output remains an
abstention/error rather than a silently assigned sentiment.

Checkpoints preserve original sample indices and text hashes, flushing each
response durably. Unknown in-flight test requests stop instead of silently repeating.
Completed runs validate artifact/protocol checksums and return saved results with
no further model calls. Reporting can resume from complete checkpoints after a
post-inference failure; it cannot select another prompt from test scores.

## Evidence

reports/proposed_v2 includes train_example_manifest.json, prompt_manifest.json,
protocol.json, per-candidate validation metrics/classification reports/predictions/
failure cases, prompt_comparison.csv, v1_vs_v2_validation.csv,
validation_failure_analysis.json and validation_comparison.png.

Final outputs: selection.json, test_state.json, test_metrics.json, confusion_matrix.png,
classification_report.csv, predictions.parquet, comparison.csv/.png,
confusion_changes.csv, failure_cases.csv, error_by_aspect.csv, failure_analysis.json,
resource_session.json, resource_report.json, aspect_conditioned_protocol.md and
external_evaluation_recommendation.md. Slides 11–14 are in
reports/task2_proposed_v2_evidence.json. The acceptance report records measured results.

No unavailable baseline JSON-validity score is invented. Test deltas distinguish
improvement over V1 from exceeding TF-IDF on macro F1 and neutral F1. Mixed/negation/
implicit/short cohorts are lexical proxies; source sarcasm labels are post-hoc only.
Class labels, aspects and predictions are never changed to improve reported scores.

MLflow uses the existing backend/experiment. Every V2 candidate logs model/prompt
hashes, settings, validation/neutral metrics, latency/tokens; only the selected run
receives test metrics. No production registry operations.

## Frozen DVC and future protocols

All prior data/baseline/V1 stages are frozen. proposed_v2 is the only new active stage:

```powershell
python -m dvc repro proposed_v2
python -m pytest -q
python -m ruff check .
python -m compileall -q src databricks scripts
```

Aspect-conditioned Qwen(text+aspect) is a DISTINCT future input/task; it is not
presented as an equivalent text-only comparison. Protocol only, no implementation.
VLSP2018 Restaurant is a candidate for later untouched external robustness/final
confirmation, subject to official access/user agreement, label-contract alignment
and source-overlap auditing. No external dataset download, evaluation, access
request or automatic merging is part of this task.

"Untouched" means withheld from this project's development/selection, not proven
absent from a foundation model's unknown pretraining corpus. Remove public task
examples and source overlaps before final scoring; prefer a prospectively collected,
newly licensed corpus if independent VLSP confirmation cannot be established.
