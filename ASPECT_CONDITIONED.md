# Proposed V2-B / V3: distinct text + aspect protocol

This experiment changes input from text to text + one target aspect. The fair
comparison is Qwen(text+aspect) versus TF-IDF(text+aspect). Prior text-only rows
are descriptive historical comparisons, not proof of apples-to-apples superiority.
V1/V2-A test outcomes have been inspected; the reused test is exploratory.

## Frozen data and models

Use only `data/food/splits/reviews.parquet`: 4140 TRAIN / 842 VALIDATION / 887 TEST,
seed 42. Existing review-ID/normalized-text leakage checks remain active through
the read-only protocol loader. No import, split, old baseline fit or old Qwen
inference is called. Protected data, prior reports, old TF-IDF model and Phase 1
files are SHA-audited before/after execution. Prior DVC stages are frozen.

The new classical control prefixes `__aspect_food__` (or the original aspect) to
text. TRAIN alone fits vocabulary/IDF and balanced Logistic Regression (seed 42).
Three fixed validation trials: unigram C=1, unigram+bigram C=1, unigram+bigram C=2;
max_features=20000, min_df=2. Macro F1 selects with fixed candidate-order tie-break.
No refit on validation is done. This additive aspect token is a minimal control;
it does not supply all aspect-by-text feature interactions.

Qwen2.5-1.5B-Instruct-GGUF Q4_K_M uses the exact frozen V1 llama.cpp configuration,
weights and executable. Temperature 0, seed 42, max_new_tokens=32. Nothing is
downloaded, fine-tuned or registered. The scoped transport adapter changes only
the user message to JSON containing text and aspect; it retains shared schema
constrained generation, strict parsing, retry counting and telemetry. It never
adds gold label, sarcasm, IDs, source or split to a scored input.

## Prompts and selection

Three prompts under `prompts/food_sentiment_aspect` are frozen before inference:

- V2-B1: zero-shot target-aspect instruction.
- V2-B2: explicit aspect and sentiment definitions.
- V2-B3: definitions plus reproducibly selected TRAIN-only examples.

The selector projects only TRAIN review_id/text/aspect/sentiment/split; it sorts
IDs/text/aspect/label, prioritizes the first eligible short text with conflicting
labels across aspects, then adds the first example of each sentiment as needed.
Review IDs/split/provenance stay in the manifest, not model examples. No test
cases are searched, selected or inserted. The manifest and all prompt hashes are
immutable. All three prompts evaluate the same validation membership.

Validation macro F1 selects Qwen; JSON validity and lower latency break ties.
Both classical and Qwen selections freeze before either new test evaluation.
Only selected runs receive test metrics. Invalid JSON is counted/retried once;
terminal invalid output is an error/abstention, never silently mapped to a label.

## Execution and recovery

From the project virtual environment:

```powershell
python -m src.cli proposed-v2-aspect-prepare
python -m src.cli proposed-v2-aspect
python -m dvc repro proposed_v2_aspect
```

The preparation command makes immutable prompts/manifests without inference.
Checkpoints bind sample index, text+aspect hash and protocol hash. A pending test
request without a durable response blocks automatic repeat inference. Completed
runs verify output hashes and return cached metrics without starting Qwen.
If reporting fails after all predictions, saved test_measurements permit
report-only recovery. Classical test also has a pre-inference durable marker.

Low-level shared runtime logs keep their historical qwen_v1 filename; reuse of
the runtime class does not execute an old experiment. One-second GPU samples are
device-wide and can include other applications. Latency across separate sessions
is descriptive, not controlled causal evidence.

The initial B1 validation completed all 842 responses but reporting failed when
pandas dropped an index name. The explicit sample_index fix was regression-tested.
`scripts/recover_aspect_validation_reporting.py` recovered original responses and
MLflow measurements without model calls before any TEST attempt, preserving the
original checkpoint/protocol and resource snapshot under logs, with copies in
`reports/aspect_conditioned/recovery_originals` for DVC preservation. Source audit keys
were canonicalized before resuming; prompt/model/settings/input hashes did not
change. One original TRAIN warmup token measurement is unavailable and reported
as missing; all validation/test review token counts are retained. The recovery
script refuses execution after any TEST attempt or completed B1 report.

A second index-name issue occurred only during final post-hoc analysis after all
887 TEST responses were saved. The analysis now explicitly restores sample_index
after reindexing. `posttest_reporting_recovery.json` pins the exact corrected
reporting source hashes while retaining the original inference protocol snapshot.
Recovery requires complete saved test measurements, permits only runner/analysis
reporting source changes, rejects all prompt/runtime/data/settings changes and
cannot enter inference. Original source copies are preserved in recovery_originals.
Regression tests cover index identity and recovery rejection rules. No additional
TEST or validation model calls occurred during either report recovery.

## Outputs and limitations

`reports/aspect_conditioned` contains classical validation/trial/selection files,
new classical model weights, selected test metrics/predictions; Qwen validation
checkpoints/metrics/classification reports, prompt comparison/selection, once-only
test metrics/predictions/confusion matrix/classification report; comparison table,
conflicting-text comparison/cases, metrics by aspect, validation/test failure
analysis and resource reports. Weights stay out of Git. Slide 12–14 evidence is
`reports/task2_aspect_v2_evidence.json`.

Conflicting membership uses the project's normalized text across the full frozen
dataset (no inference inputs use that metadata). Report all test rows, conflicting
rows and nonconflicting rows separately. Fixed-three-label macro F1 assigns zero
to absent labels; subgroup counts accompany scores. Amenities/room are suppressed
if test support is below 20. Small overlapping text clusters limit interpretation;
paired outcomes do not prove causality.

Historical TF-IDF subset predictions are reconstructed from its saved union-of-
baseline-errors CSV: rows absent from that exhaustive file were correct for both
old models. Full reconstructed aggregate scores must equal frozen reported scores.
No old inference is needed. Historical V1 predictions are read after new inference.

Read-only integrity/MLflow verification: `python -m scripts.verify_aspect_results`.
The measured outcome is in `reports/ASPECT_CONDITIONED_ACCEPTANCE_REPORT.md`.

Failure analysis distinguishes lexical proxies (mixed/negation/implicit), source
sarcasm annotations and unadjudicated wrong-aspect hypotheses. A same-text case
table supports manual assessment; it does not fabricate verified error causes.
No LoRA, larger Qwen, RAG, embeddings or production serving is introduced.
