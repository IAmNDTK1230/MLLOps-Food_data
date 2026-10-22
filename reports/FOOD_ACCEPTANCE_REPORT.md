# Vietnamese restaurant data acceptance

Date: 2026-10-13, Asia/Saigon. Status: **DATA PREPARATION PASS**.
Real VN-MSASar v3 import, preparation, canonical splitting, EDA and tokenizer analysis
were executed locally. No model was trained; no Qwen inference or embeddings ran.
Verified Phase 1 Databricks code/bundle, authentication and crawler were not modified.

## Source and discovered schema

- Dataset: [phamluan/vn_msa_sar_v3](https://huggingface.co/datasets/phamluan/vn_msa_sar_v3).
- Revision: `6f3f17b390dcf1bf411804e1bc8246bf88ea1bf4`.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution retained.
- Downloaded original test/train/validation Parquet files plus the dataset card.
- Actual columns: review_id, source, place_type, language, sentence, aspect (strings),
  sentiment and sarcasm (int64). Observed sentiments: -3, -2, -1, 0, 1, 2, 3.
- Published sentiment scale is signed ordinal -3..+3. The project's explicit three-class
  task collapses intensity by sign; original scores remain in original_sentiment.
- Missing restaurant IDs remain null. Rating is absent and not required or used.
- All nine aspects occur in the filtered subset; even rare room/amenities are preserved.

## Actual data counts

| Measure | Count |
|---|---:|
| Raw rows across original splits | 76,785 |
| Exact language=vi and place_type=restaurant | 5,869 |
| Filtered out | 70,916 |
| Valid sentence/aspect rows | 5,869 |
| Rejected rows | 0 |
| Duplicates removed | 0 |
| Unique review IDs | 1,530 |
| Globally unique normalized sentences | 5,316 |
| Unique review/sentence documents | 5,366 |
| Independent connected review/text groups | 1,481 |

Source platform counts: foody 5,649; other 220. The publisher describes `other` as
deliberately collected real sarcastic reviews. Retain this source distinction for
future stratified evaluation; the prepared experimental split is not the publisher's
original multilingual benchmark split. No claim of fuzzy near-duplicate protection
is made after creating this new split; exact normalized text and review IDs are protected.

| Sentiment | Rows | Percentage |
|---|---:|---:|
| positive | 3,714 | 63.28% |
| negative | 1,861 | 31.71% |
| neutral | 294 | 5.01% |

Majority/minority ratio: **12.63**. BEFORE statistics only. First proposed classical
strategy: training-derived `class_weight="balanced"`; no SMOTE and no AFTER results.
Text length in Unicode characters per sentence/aspect row: mean 63.28, median 52,
minimum 2, maximum 615. Exact values are in food_dataset_report.json.

| Aspect | Rows |
|---|---:|
| food | 2,413 |
| service | 1,031 |
| price | 878 |
| overall | 703 |
| ambiance | 483 |
| location | 193 |
| cleanliness | 139 |
| amenities | 20 |
| room | 9 |

## Shared canonical split

Seed 42; requested fractions 70/15/15. Group sizes make exact fractions approximate.
Every same review_id and identical NFC/whitespace-normalized text stays in one split,
including transitive overlaps between reviews. Source original_split is retained.

| Split | Rows | Actual % | Reviews | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|---:|
| train | 4,140 | 70.54% | 1,074 | 1,307 | 208 | 2,625 |
| validation | 842 | 14.35% | 229 | 270 | 43 | 529 |
| test | 887 | 15.11% | 227 | 284 | 43 | 560 |

Saved leakage checks: review IDs spanning splits **0**, normalized texts spanning
splits **0**. All future methods consume data/food/splits/reviews.parquet with its
checksum and metadata. Macro F1 is the primary comparison metric; definitions are
in food_experiment_plan.json. No baseline/proposed method results exist yet.

## RAG engineering evidence

Reference tokenizer only: Qwen/Qwen2.5-7B-Instruct, revision
`a09a35458c702b33eeacc393d103063234e8bc28`, Apache 2.0; tokenizer.json only,
special tokens disabled. No model weights downloaded. This is not a final model choice.

| Strategy | Documents | Total tokens | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| reconstructed review | 1,530 | 108,691 | 71.04 | 48 | 198.65 | 580 |
| source sentence | 5,366 | 110,175 | 20.53 | 17 | 48 | 187 |

Review reconstruction concatenates unique source sentences in stable lexical order;
original full review text/order is not available. Token totals differ between bases
because BPE tokenization depends on concatenation boundaries. Per-aspect sentiment
annotations and source/review/split attribution are preserved in JSONL documents.

chunking_comparison.csv contains intact review/sentence strategies and all nine
128/256/512 × 0/32/64 size/overlap candidates on each base (20 comparison rows total).
Examples: 128/0 review chunks 1,821; 256/0 1,558; 512/0 1,531. Sentence chunks at
128/0 are 5,370; 256/0 and 512/0 are 5,366. **No final chunk size selected.**

## Files created/changed

Created code:
- src/food/__init__.py
- src/food/config.py
- src/food/adapters.py
- src/food/data.py
- src/food/split.py
- src/food/analysis.py
- src/food/interfaces.py
- scripts/analyze_food_tokens.py
- tests/test_food_pipeline.py

Changed configuration/docs:
- src/ml/cli.py: food commands; obsolete product/model commands disabled.
- params.yaml: food-only source/revision/split/tokenizer/chunk parameters.
- dvc.yaml: four data-only stages.
- requirements-ml.txt: matplotlib/tokenizers and compatible dependency pins.
- README.md, docs/experiments/PHASE2.md, reports/README.md, reports/PHASE2_IMPLEMENTATION_REPORT.md.
- .gitignore: keep this acceptance document in version control.

Created execution artifacts:
- dvc.lock and DVC cache snapshots of all four executed stages.
- data/food/source/vn_msasar_v3/{original split files,README.md,provenance.json}.
- data/food/processed/reviews.parquet; data/food/rejected/rows.json.
- data/food/metadata/preparation.json and tokenizer/{tokenizer.json,provenance.json}.
- data/food/splits/{reviews.parquet,metadata.json}.
- data/food/rag/{review_documents.jsonl,sentence_documents.jsonl}.
- reports/food_label_mapping.json; food_dataset_report.json.
- reports/sentiment_distribution, aspect_distribution, text_length_distribution,
  sentiment_by_aspect: each CSV and PNG.
- reports/token_statistics.json; chunking_comparison.csv.
- reports/food_experiment_plan.json; qwen_output_schema.json; task2_slide_evidence.json.
- reports/FOOD_ACCEPTANCE_REPORT.md (this document).

## Validation executed

- python -m pytest -q: **148 passed**; one existing Starlette/AnyIO deprecation warning.
- python -m ruff check .: **PASS**.
- python -m compileall -q src databricks scripts: **PASS**.
- python -m pip check: **PASS**, no broken requirements.
- python -m dvc dag: **PASS**, only food stages.
- python -m dvc repro: **PASS**, all four real data stages executed; later split/analysis
  refresh passed and import/prepare were correctly skipped as unchanged.

Tests use temporary unit data, model doubles for historical infrastructure tests,
and no real estimator/Qwen fitting. Network downloads and pytest/DVC temp/cache
access required execution outside the restricted sandbox. The final reported checks
completed; earlier sandbox failures are not counted as acceptance passes.

## Commands and remaining work

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.cli food-data-import --dataset vn_msasar_v3
python -m src.cli food-prepare
python -m src.cli food-split
python -m src.cli food-analysis
# Reproduce the same tracked data workflow:
python -m dvc repro
```

The real primary dataset and tokenizer have already been downloaded here. No further
dataset download is needed to inspect the prepared reports or rerun prepare/split/
analysis locally. A fresh machine needs the pinned public downloads or manual files.
Model training/comparison, Qwen analysis, embeddings, final chunk-size selection and
external-dataset schema/license verification remain future work, as requested.
