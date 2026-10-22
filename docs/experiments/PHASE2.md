# Phase 2 — Vietnamese food / restaurant data engineering

This document describes the completed preparation phase. The authorized text-only
baseline phase now has separate commands in [BASELINES.md](BASELINES.md).
Phase 1 Databricks jobs, crawler, authentication and
Bronze/Silver/Gold transformations are unchanged. Food datasets stay local under
`data/food`; they are never uploaded into the Shopee Gold tables. Previous product
ML CLI commands (including training, evaluation and registration) are disabled.
The older product ML/API infrastructure remains inactive. Preparation commands do
not fit estimators; the explicit food baseline commands train classical models.
No Qwen inference, fine-tuning, embeddings or SMOTE experiments run.

## Verified primary source

[VN-MSASar v3, phamluan/vn_msa_sar_v3](https://huggingface.co/datasets/phamluan/vn_msa_sar_v3)
at revision `6f3f17b390dcf1bf411804e1bc8246bf88ea1bf4`.
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) attribution is retained in
the source card, provenance, rows and RAG documents. The publisher describes v3 as
real reviews with earlier LLM-generated rows removed. We record that as a publisher
statement; we have not independently audited how each review was collected.

The three downloaded Parquet files contain 76,785 rows. Actual source columns:

| Column | Actual type / meaning |
|---|---|
| review_id | string; original review grouping ID |
| source | string; original platform |
| place_type | string; exact restaurant filter |
| language | string; exact vi filter |
| sentence | string; original sentence, mapped to text |
| aspect | string; gold aspect, preserved |
| sentiment | int64; observed -3, -2, -1, 0, 1, 2, 3 |
| sarcasm | int64; 0/1 |

One row is a sentence/aspect pair. The actual source has no restaurant ID, rating,
review timestamp or intact full-review text. We leave restaurant IDs null and do
not fabricate the absent fields. All nine source aspects also occur after filtering,
including the rare `room` and `amenities` labels; these labels remain unchanged.

## Install and commands

Run from the project root with the existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-ml.txt
python -m src.cli food-data-import --dataset vn_msasar_v3
python -m src.cli food-prepare
python -m src.cli food-split
python -m src.cli food-analysis
```

`food-data-import` downloads only the configured public source revision. Location,
output roots, seed, ratios and tokenizer reference are configured in `params.yaml`
under `food`. No alternative dataset is substituted on failure. For offline import,
download **all three original split Parquet files plus README.md** from the pinned
repository revision into a directory, then:

```powershell
python -m src.cli food-data-import --dataset vn_msasar_v3 --input C:\datasets\vn_msasar_v3
python -m src.cli food-prepare
python -m src.cli food-split
python -m src.cli food-analysis
```

Manual imports record the configured source/revision as operator declarations,
not a verified remote download. CSV/JSONL files with the exact source columns are
also supported. The importer snapshots bytes and records actual schema and values.
If tokenizer download is unavailable, download only its `tokenizer.json` and use
`food-analysis --tokenizer-file C:\datasets\tokenizer.json`; local tokenizer provenance
is marked unverified, with its SHA-256. Word counts are never substituted for tokens.
`food-pipeline --dataset vn_msasar_v3` runs all four data stages, with no model stages.

## Canonical contract and cleaning

Required canonical fields:
`review_id, restaurant_id, text, aspect, sentiment, sarcasm, source, dataset_name,
language, place_type, split, ingestion_timestamp`.
`restaurant_id` is nullable and `split` is initially null, populated in the canonical
split file. Extra retained fields include `original_text`, `original_sentiment`,
`original_split`, `license`, `source_url`, `license_url`. Optional source rating and
timestamp are preserved when present but never used for sentiment.

Workflow: byte snapshot + SHA-256 → exact vi/restaurant filters → NFC and whitespace
normalization → empty-text/missing-metadata/invalid-label rejection → documented
signed gold label normalization → exact duplicate removal → shared group splits →
EDA and RAG preparation. Accents and case are preserved. Rejected original rows and
reasons are stored locally under `data/food/rejected`, never dumped by the CLI.
Conflicting sentiment labels on different aspects are not discarded as duplicates.

The published card defines the **signed ordinal -3..+3 scale**. Our explicit
three-class task collapses intensity by sign: -3/-2/-1 negative, 0 neutral, 1/2/3
positive. `original_sentiment` stays intact; no star-rating proxy is used.
Mapping, observed values, source semantics and revision are recorded in
`reports/food_label_mapping.json`.

## One canonical split

Default seed 42, requested 70/15/15. Source splits are recombined for this new
food-only experiment, retained as `original_split`; this is not a claim to reproduce
the publisher's original benchmark. Connected components join ALL shared review IDs
and identical NFC/whitespace-normalized text, transitively. A deterministic group
search considers ratio/class balance without model scores. Groups are never broken
to meet an exact percentage; actual counts and distributions are recorded.
No new fuzzy duplicate detection is claimed.

`data/food/splits/reviews.parquet` and `metadata.json` are the sole shared splits for
Rule-Based, TF-IDF + Logistic Regression and Proposed V1. Checksums detect edited raw,
processed and split files; loading also checks seed, ratios and leakage. Future text
vocabularies, class weights and tuning must use train/validation only.

## EDA and Task 2 evidence

- `reports/food_dataset_report.json`: raw/filter/valid/rejected/duplicate counts,
  unique reviews, sentence/aspect pairs, class/aspect/source distributions, text
  lengths, before imbalance and split distributions.
- `reports/sentiment_distribution.csv/.png`, `aspect_distribution.csv/.png`,
  `text_length_distribution.csv/.png`, `sentiment_by_aspect.csv/.png`: chart-ready
  values and exportable plots. Length counts are Unicode characters and whitespace
  words, separately from tokenizer counts. These are sentence/aspect-row statistics.
- `reports/task2_slide_evidence.json`: Slide 3 source/count/schema, Slide 4 cleaning
  and mapping, Slide 5 split counts/imbalance, Slide 6 proposed imbalance strategy,
  plus RAG token/chunk/metadata evidence.
- `reports/FOOD_ACCEPTANCE_REPORT.md`: human-readable measured acceptance results.

Use `class_weight="balanced"` computed from training labels as the first classical
ML strategy. No SMOTE is enabled. AFTER distributions and model metrics are absent
until an actual experiment is authorized and run.

## Future comparison definitions — not executed

The historical `reports/food_experiment_plan.json` defines rule-based and TF-IDF/Logistic Regression
baselines and Qwen Proposed V1. The primary task is sentiment of the supplied aspect
for each sentence/aspect pair. All methods receive the same text and aspect inputs,
including an aspect indicator for the future classical baseline. This avoids treating
different gold labels for different aspects of a sentence as contradictory targets.
The current authorized baseline comparison overrides this early idea: **text only**,
with aspect excluded from features. See BASELINES.md and task2_baseline_evidence.json.
Macro F1 is primary; accuracy, macro precision/recall, weighted F1 and inference
latency will also be reported on the same canonical splits.

`src/food/interfaces.py` and `reports/qwen_output_schema.json` define structured Qwen
output: sentiment, aspect, nullable food_quality/service_experience, topics list.
Sentiment has quantitative gold labels; aspect has source gold labels for a separately
defined detection evaluation. Supplied aspect inputs cannot count as predicted aspect
accuracy. Food quality, service experience and topics are enrichment-only with no
direct gold labels. No Qwen client or model execution is added.
NTC-SCV and VLSP2018 Restaurant have explicit external-evaluation-only interfaces
that currently fail pending verified schemas/licenses. They cannot be imported into
the primary dataset by these commands.

## RAG preparation and tokenizer analysis

`data/food/rag/review_documents.jsonl` groups unique source sentences per review;
`sentence_documents.jsonl` groups aspects per review/sentence. Documents retain
review_id/source/aspect/sentiment/dataset_name/split/license/source_url and aligned
per-aspect annotations. Reconstructed reviews use stable lexical sentence order;
the original review order/full text is unavailable and is not claimed recovered.
Future RAG evaluations must build retrieval indexes from training documents only.

Tokenizer-only reference: `Qwen/Qwen2.5-7B-Instruct`, pinned revision
`a09a35458c702b33eeacc393d103063234e8bc28`; Apache 2.0. Only tokenizer.json is loaded
through `tokenizers`, with special tokens disabled. This is a reference for analysis,
not a final model choice. No model weights, generation or embeddings are used.

`reports/token_statistics.json` compares review and source-sentence documents using
actual token counts (total, mean, median, p95, max). `reports/chunking_comparison.csv`
adds all 128/256/512 × 0/32/64 size/overlap candidates on both document bases, with
chunk counts and token totals including overlap. No final chunk size is selected.
Standalone analysis entry point: `python -m scripts.analyze_food_tokens` (same
verified analysis implementation, also supports `--params` and `--tokenizer-file`).

## DVC and checks

```powershell
python -m dvc dag
python -m dvc repro
python -m pytest -q
python -m ruff check .
python -m compileall -q src databricks scripts
```

The preparation DAG contains food_import → food_prepare → food_split → food_analysis;
these verified stages are frozen during the current baseline phase.
Data/report paths are DVC outputs; do not `dvc add` the same source output separately.
Future baseline_rule/baseline_tfidf/proposed_v1/evaluate definitions are outside the
preparation DAG. The active downstream baseline_rule/baseline_tfidf/baseline_evaluate
stages are documented in BASELINES.md. `params.yaml`, source hashes, pinned revisions and `dvc.lock` provide
reproduction metadata. Network is needed for a fresh download; existing local source
and tokenizer snapshots support prepare/split/analysis offline. No DVC remote or
cloud deployment is configured automatically.
