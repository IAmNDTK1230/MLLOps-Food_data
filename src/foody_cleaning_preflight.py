"""Preservation/provenance gate only. Unresolved IDs stop all derived cleaning."""

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path

from scripts.preserve_foody_source import preserve

EXCLUDED = {
    ".git",
    ".venv",
    ".dvc",
    ".databricks",
    ".qwen-runtime",
    ".deps",
    ".lint-tools",
    ".pytest-tools",
    ".verify-tools",
    ".flow-tools",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "mlruns",
    "mlartifacts",
    "state",
    "logs",
    "models",
    "reports",
}
EXTENSIONS = {
    ".py",
    ".ipynb",
    ".sql",
    ".r",
    ".js",
    ".ts",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".json",
}
ID_QUERY = re.compile(r"RestaurantID(?:\.1)?|restaurant_id", re.IGNORECASE)
TRANSFORM_QUERY = re.compile(
    r"to_csv|read_csv|concat\(|merge\(|\.join\(|rename\(|CREATE\s+TABLE|INSERT\s+INTO",
    re.IGNORECASE,
)
NEW_AUDIT_FILES = {
    "src/foody_cleaning_preflight.py",
    "scripts/preserve_foody_source.py",
    "tests/test_foody_preservation.py",
}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_inventory(project):
    inventory, matches, transforms, notebooks, sql, skipped = [], [], [], [], [], []
    for folder, directories, names in os.walk(project):
        directories[:] = [name for name in directories if name not in EXCLUDED]
        if Path(folder).resolve() == (project / "data/foody").resolve():
            directories[:] = [name for name in directories if name not in {"audit", "audit_v2"}]
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(project).as_posix()
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                skipped.append(
                    {
                        "path": relative,
                        "reason": "Non-UTF8 text file; no lossy source decoding performed",
                    }
                )
                continue
            inventory.append({"path": relative, "sha256": checksum(path)})
            if path.suffix.lower() == ".ipynb":
                notebooks.append(relative)
            if path.suffix.lower() == ".sql":
                sql.append(relative)
            spans = []
            if path.suffix.lower() == ".py":
                try:
                    spans = [
                        (node.lineno, node.end_lineno, node.name)
                        for node in ast.walk(ast.parse(text))
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                except SyntaxError:
                    skipped.append(
                        {"path": relative, "reason": "AST unavailable; text search still performed"}
                    )
            # Newly authored gate/preservation code is inventoried but cannot be
            # evidence of how the pre-existing Foody export was generated.
            if relative in NEW_AUDIT_FILES:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                containing = [
                    (end - start, function)
                    for start, end, function in spans
                    if start <= number <= end
                ]
                function = min(containing)[1] if containing else "<module or non-Python source>"
                record = {
                    "path": relative,
                    "line": number,
                    "function": function,
                    "text": line[:800],
                }
                if ID_QUERY.search(line):
                    matches.append(record)
                if TRANSFORM_QUERY.search(line):
                    transforms.append(record)
    return {
        "scope": "All available first-party project text/code, SQL, notebooks, config, documentation and data-side code; generated reports, caches, binary datasets, models and third-party dependencies excluded.",
        "excluded_directory_names": sorted(EXCLUDED),
        "queries": [ID_QUERY.pattern, TRANSFORM_QUERY.pattern],
        "searched_files": sorted(inventory, key=lambda r: r["path"]),
        "id_matches": matches,
        "transformation_candidates": transforms,
        "notebook_files": notebooks,
        "sql_files": sql,
        "skipped_text_or_ast": skipped,
        "audit_only_files_excluded_from_historical_origin_evidence": sorted(NEW_AUDIT_FILES),
    }


def require_canonical_restaurant_id(decision):
    """Fail closed. Candidate statistics, ordering and Pandas aliases are not proof."""
    if (
        decision.get("resolution_status") != "RESOLVED_FROM_SOURCE_CODE"
        or decision.get("canonical_column") not in {"RestaurantID", "RestaurantID.1"}
        or not decision.get("evidence_file")
        or not decision.get("evidence_function")
        or not decision.get("exact_transformation_path")
    ):
        raise ValueError(
            "RestaurantID provenance unresolved: restaurant-specific processing must stop"
        )


def run(project, source, snapshot, reports):
    project, source, snapshot, reports = (
        Path(project).resolve(),
        Path(source),
        Path(snapshot),
        Path(reports),
    )
    if source.resolve().is_relative_to(reports.resolve()) or snapshot.resolve().is_relative_to(
        reports.resolve()
    ):
        raise ValueError("Reports must not contain raw source files")
    protected = {
        str(p): checksum(p)
        for folder in (
            project / "reports/foody",
            project / "reports/foody_v2",
            project / "data/foody/audit",
            project / "data/foody/audit_v2",
        )
        for p in folder.rglob("*")
        if p.is_file()
    }
    model_path = project / "models/tfidf_logreg.joblib"
    if model_path.exists():
        protected[str(model_path)] = checksum(model_path)
    expected = json.loads(
        (project / "data/foody/audit_v2/source_manifest.json").read_text(encoding="utf-8")
    )["source_sha256"]
    manifest = preserve(source, snapshot, expected)
    protected[str(source)] = manifest["sha256"]
    protected[str(snapshot)] = manifest["sha256"]
    protected[str(snapshot.with_suffix(".manifest.json"))] = checksum(
        snapshot.with_suffix(".manifest.json")
    )
    inventory = source_inventory(project)
    # Inspect structural ambiguity without assigning a business key or grouping
    # observations into restaurants. csv.reader retains both raw header positions.
    rows = list(csv.reader(io.StringIO(snapshot.read_bytes().decode("utf-8-sig"))))
    header, records = rows[0], rows[1:]
    positions = [n for n, name in enumerate(header) if name == "RestaurantID"]
    if len(positions) != 2:
        raise ValueError(
            "Expected the audited duplicate RestaurantID headers; inspect changed schema"
        )
    mismatches = sum(row[positions[0]] != row[positions[1]] for row in records)
    comments = [row[header.index("Comment")] for row in records]
    empty = sum(not text.strip() for text in comments)
    replacement = sum("\ufffd" in text for text in comments)
    nonempty = [text for text in comments if text.strip()]
    audit_schema = json.loads(
        (project / "reports/foody_v2/schema_report.json").read_text(encoding="utf-8")
    )
    decision = {
        "resolution_status": "UNRESOLVED",
        "outcome": "D",
        "canonical_column": None,
        "evidence_file": None,
        "evidence_function": None,
        "exact_transformation_path": None,
        "restaurant_specific_processing_allowed": False,
        "raw_header": header,
        "candidate_raw_column_positions_one_based": [p + 1 for p in positions],
        "disagreeing_rows": mismatches,
        "reason": "No available Foody scraper/exporter/database mapping or original notebook establishes either column's business meaning. Audits/readers/test fixtures cannot establish source provenance.",
        "known_path": [
            "Unknown external export creates two raw RestaurantID headers",
            "pandas.read_csv exposes the second header as RestaurantID.1",
            "existing Foody audits preserve/compare both candidates; no canonical mapping is produced",
        ],
        "classification_role": "unknown_provenance_source_metadata",
        "classification_derived_name_planned": "source_classification",
        "classification_rename_applied": False,
        "classification_original_values": sorted(
            {r[header.index("Classification")] for r in records}
        ),
        "classification_usage_allowed": ["source metadata only"],
        "classification_usage_disallowed": [
            "training target",
            "evaluation target",
            "sentiment truth",
            "RAG truth",
        ],
    }
    try:
        require_canonical_restaurant_id(decision)
    except ValueError as error:
        blocked_reason = str(error)
    else:
        raise AssertionError("This preflight has not established source provenance")
    processed = project / "data/foody/processed/reviews.parquet"
    metrics = {
        "execution_status": "BLOCKED",
        "blocked_reason": blocked_reason,
        "source_sha256": manifest["sha256"],
        "raw_rows": len(records),
        "empty_comments": empty,
        "replacement_affected_rows": replacement,
        "text_eligible_rows_before_id_validation": sum(
            bool(text.strip()) and "\ufffd" not in text for text in comments
        ),
        "text_eligibility_is_not_processed_ai_validity": True,
        "raw_extra_exact_duplicate_nonempty_comments": len(nonempty) - len(set(nonempty)),
        "processed_ai_valid_rows": None,
        "processed_deduplicated_rows": None,
        "normalized_restaurant_text_duplicates": None,
        "canonical_restaurant_count": None,
        "raw_candidate_id_counts_from_prior_audit": audit_schema["restaurants"]["candidates"],
        "missing_dates_source": audit_schema["date_audit"]["missing_timestamps"],
        "missing_ratings_source": audit_schema["missing_ratings"],
        "source_date_range": [
            audit_schema["date_audit"]["earliest"],
            audit_schema["date_audit"]["latest"],
        ],
        "source_rating_range": audit_schema["rating_range"],
        "demo_readiness": "BLOCKED",
        "rag_readiness": "BLOCKED",
        "readiness_reason": "Canonical RestaurantID unresolved; no processed dataset has passed validation",
        "cleaning_executed": False,
        "processed_output_created": False,
        "processed_output_exists": processed.exists(),
        "training_executed": False,
        "rag_or_streamlit_started": False,
        "primary_research_model": "models/tfidf_logreg.joblib (unchanged)",
        "next_step": "Obtain the original Foody scraper/export notebook, dataframe join/rename code or database mapping plus a restaurant business URL/ID example proving which raw column represents which concept. Then implement derived cleaning; enrichment remains deferred.",
    }
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "source_search_inventory.json", inventory)
    write_json(reports / "restaurant_id_decision.json", decision)
    write_json(reports / "cleaning_metrics.json", metrics)
    render_provenance(reports, inventory, decision)
    render_acceptance(reports, metrics, decision, manifest)
    if any(checksum(Path(name)) != sha for name, sha in protected.items()):
        raise ValueError("Source/snapshot/historical evidence/research model changed")
    write_json(
        reports / "preservation_checks.json",
        {
            "status": "PASS",
            "protected_files": protected,
            "processed_cleaning_not_executed": True,
            "source_and_snapshot_identical": True,
        },
    )
    return metrics


def render_provenance(reports, inventory, decision):
    contexts = {}
    for match in inventory["id_matches"]:
        key = (match["path"], match["function"])
        contexts.setdefault(key, []).append(match["line"])
    lines = [
        "# Restaurant ID provenance",
        "",
        "## Decision: D — provenance cannot be established",
        "",
        "**Neither RestaurantID nor RestaurantID.1 is canonical. Restaurant-specific cleaning, deduplication, grouping, enrichment and retrieval are stopped.**",
        "",
        decision["reason"],
        "",
        "## What is established",
        "",
        f"The raw CSV itself contains two headers spelled RestaurantID at positions {decision['candidate_raw_column_positions_one_based']} (one-based). They disagree on {decision['disagreeing_rows']:,} records.",
        "The .1 suffix is a Pandas disambiguation of the second raw header, not a source field name and not evidence that either field is accidental or canonical.",
        "The accidental Unnamed: 0 export index is a separate column. Neither RestaurantID may be treated as that index without source evidence.",
        "",
        "## Exact path established / missing",
        "",
        "1. **Missing upstream path:** original Foody scraper/database/export dataframe -> why two same-name ID columns were joined/concatenated/renamed. No such creator code or mapping is available here.",
        "2. **Observed input:** duplicate raw RestaurantID headers already exist in the supplied immutable CSV.",
        "3. **Verified reader path:** src/foody_v2_audit.py run -> csv.reader retains raw header; pandas.read_csv disambiguates to RestaurantID / RestaurantID.1.",
        "4. **Verified audit path:** restaurant_audit compares candidates, emits separate diagnostics and explicitly does not select a canonical business key.",
        "5. **Verified check path:** scripts/verify_foody_v2_audit.py main preserves both raw header positions and checks both against row-level audit metadata.",
        "",
        "## Candidate evidence",
        "",
        "| Candidate | Evidence | Canonical decision |",
        "|---|---|---|",
        "| First raw RestaurantID / Pandas RestaurantID | Supplied numeric values; first reader position. No source business-ID/URL mapping. | Unresolved |",
        "| Second raw RestaurantID / Pandas RestaurantID.1 | Supplied numeric values; suffix is reader-generated. No source business-ID/URL mapping. | Unresolved |",
        "Neither fewer duplicates, value ranges, candidate counts, first/last position nor agreement with an earlier audit proves provenance. A/B/C cannot be distinguished from available evidence.",
        "",
        "## Available components inspected",
        "",
        "- src/crawler/shopee_products.py ShopeeProductCrawler and src/crawler/shopee_reviews.py ShopeeReviewCrawler.crawl concern Shopee, not a Foody exporter. The review crawler is an unimplemented extension point.",
        "- src/storage/raw_writer.py write_jsonl/write_raw emits Shopee JSONL. src/processing/parquet_writer.py write_parquet writes existing product/review schemas; neither creates the Foody CSV or its ID mapping.",
        "- src/food/data.py read_source/import_dataset reads supplied research source files. src/food/adapters.py adapt_vn_msasar initializes an optional restaurant_id for VN-MSASar; it is unrelated to the Foody duplicate ID origin.",
        "- src/foody_audit.py run and src/foody_v2_audit.py run/restaurant_audit consume the supplied CSV and produce diagnostics only.",
        "- scripts/verify_foody_audit.py main and scripts/verify_foody_v2_audit.py main are consistency checks, not source-data generators.",
        "- Test fixture RestaurantID fields are synthetic test inputs; they cannot identify the real source business ID.",
        f"- Notebook files found: {inventory['notebook_files']}. SQL files found: {inventory['sql_files']}. scripts/databricks_bootstrap.sql creates Shopee Unity Catalog schemas/volume, not Foody business tables.",
        "",
        "## Every matching existing first-party file/function",
        "",
        "The inventory includes code, configuration and documentation matches. Matching text is not an exporter provenance claim. Exact matched lines, CSV/join/export candidates and searched file SHA256s are in source_search_inventory.json.",
        "",
        "| File | Function/context | Lines |",
        "|---|---|---|",
    ]
    lines += [
        f"| {path} | {function} | {', '.join(map(str, numbers))} |"
        for (path, function), numbers in sorted(contexts.items())
    ]
    lines += [
        "",
        "## Required next evidence",
        "",
        "Supply the original Foody scraper/export notebook or database loading/join/rename code and a known restaurant page/business ID mapping. Trace both columns from creation through merge/concat/export. If both encode different concepts, document both and identify the one used for restaurant-specific operations.",
        "No column selection, normalization or restaurant-specific deduplication has been performed.",
    ]
    (reports / "restaurant_id_provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_acceptance(reports, metrics, decision, manifest):
    lines = [
        "# Foody cleaning acceptance — BLOCKED before cleaning",
        "",
        "As of 2026-10-20 (Asia/Saigon).",
        "",
        "**Source preservation PASS. RestaurantID provenance UNRESOLVED (D). Cleaning NOT RUN.**",
        "",
        "The explicit instruction to stop restaurant-specific processing applies. src/foody/schema.py, cleaning.py and validation.py are deferred until canonical ID provenance is established. No processed reviews.parquet was created.",
        "",
        "## Preserved snapshot",
        "",
        f"Snapshot: `{manifest['snapshot_path']}`.",
        f"SHA256: `{manifest['sha256']}`; bytes: **{manifest['size_bytes']:,}**.",
        f"Snapshot timestamp: {manifest['snapshot_created_at_asia_ho_chi_minh']}.",
        "Source and versioned snapshot are byte-identical. The snapshot and manifest are write-once; reruns verify instead of overwrite. This preserves the current UTF-8 export, not the earlier unavailable legacy raw file.",
        "",
        "## Counts and source coverage",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Raw source rows | {metrics['raw_rows']:,} |",
        f"| Empty comments | {metrics['empty_comments']} |",
        f"| Replacement-affected rows | {metrics['replacement_affected_rows']} |",
        f"| Text-eligible rows before ID validation | {metrics['text_eligible_rows_before_id_validation']:,} |",
        f"| Extra exact nonempty Comment duplicates, raw/global | {metrics['raw_extra_exact_duplicate_nonempty_comments']} |",
        "| Processed AI-valid rows | NOT RUN / unavailable |",
        "| Deduplicated processed rows | NOT RUN / unavailable |",
        "| Canonical restaurant count | UNRESOLVED |",
        f"| Disagreeing raw RestaurantID rows | {decision['disagreeing_rows']:,} |",
        f"| Missing Review Time | {metrics['missing_dates_source']} |",
        f"| Missing Rating | {metrics['missing_ratings_source']} |",
        f"| Source date range | {metrics['source_date_range'][0]} to {metrics['source_date_range'][1]} |",
        f"| Source observed rating range | {metrics['source_rating_range']} |",
        "",
        "Each raw ID candidate had 818 distinct values in the prior audit; that statistic does not establish 818 verified business identities. Both fields remain unselected. Existing diagnostic counts are referenced only; no new restaurant-specific grouping or processing was run.",
        "The 15,638 text-eligible records are a text-only preflight count, not a processed or validated AI dataset. Processed counts are unavailable, not zero. The 940 duplicate count is global exact raw Comment duplication, not the future canonical deduplication result.",
        "",
        "## Classification handling",
        "",
        'classification_role = "unknown_provenance_source_metadata".',
        "Classification remains unchanged in raw source and snapshot, with original -1/0/1 codes. In a future derived dataset only, its field name will be source_classification. No derived rename has been executed while cleaning is blocked.",
        "It must not be a training target, evaluation target, sentiment truth or RAG truth. Unknown provenance alone does not prohibit a later demo/RAG; the unresolved restaurant identity currently blocks restaurant-specific processing.",
        "",
        "## Deferred derived plan (not implemented or executed)",
        "",
        "After evidence identifies the canonical source field, implement src/foody/schema.py, cleaning.py and validation.py. Preserve raw_review_time plus the requested canonical fields and valid_for_ai/duplicate_count metadata.",
        "- review_id: deterministic SHA256 of a versioned unambiguous serialization of canonical restaurant_id, raw user_id, raw Review Time and original Comment; never assume UserID is a review ID.",
        "- Text: NFC and surrounding/repeated whitespace normalization; preserve accents, case and semantic punctuation. Empty/U+FFFD records remain in audit/source and are ineligible for AI. No guessed characters.",
        "- Duplicates: document canonical restaurant_id + normalized text as the conservative observation key. Retain duplicate_count, contributing source/review IDs and conflicting metadata instead of discarding provenance or silently choosing ratings/labels.",
        "- Dates: preserve raw_review_time, recognize observed minute/seconds-plus-comma formats in derived parsing, leave missing dates null and do not invent dates.",
        "- Ratings: preserve raw_rating on the observed 0–10 scale, keep missing values null, and never derive sentiment from it.",
        "- Preserve source_classification values only as unknown-provenance metadata. source=foody; ingestion timestamp records real derived ingestion time, not review time.",
        "",
        "## Readiness and commands",
        "",
        f"demo_readiness = **{metrics['demo_readiness']}**.",
        f"rag_readiness = **{metrics['rag_readiness']}**.",
        "Neither is READY: no canonical ID or validated processed dataset exists. The prior READY_AFTER_CLEANING status was conditional and has not been fulfilled.",
        "There is **no runnable cleaning command yet**; inventing one would bypass the required provenance gate. Exact current runnable preflight:",
        "",
        "```powershell",
        "python -m src.foody_cleaning_preflight",
        "```",
        "",
        "Expected exit code 2 means the documented provenance blocker, not a cleaning execution. This command verifies snapshot/evidence and writes the blocked decision; it never selects an ID or creates processed data.",
        "",
        "## Next step for AI enrichment",
        "",
        metrics["next_step"],
        "After provenance resolution, implement/test derived cleaning and validate the accepted Parquet. Only then plan inference with existing models/tfidf_logreg.joblib to add predicted_sentiment, sentiment_confidence and sentiment_model_version. Keep VN-MSASar as the research benchmark. No Foody retraining, RAG or Streamlit was started.",
        "",
        "## Testing scope",
        "",
        "Snapshot immutability/idempotency, duplicate-header preservation and refusal to process unresolved IDs are tested now. Review-ID generation, NFC/whitespace cleaning, AI exclusion, canonical deduplication and processed rating/Classification/date validation remain deferred with their implementation because Step 4 requires resolved provenance. No such deferred tests or processing are claimed as passing.",
    ]
    (reports / "CLEANING_ACCEPTANCE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/foody/source/review.csv")
    parser.add_argument("--snapshot", default="data/foody/source/review_utf8_20261020.csv")
    parser.add_argument("--reports", default="reports/foody_cleaning")
    args = parser.parse_args()
    result = run(Path.cwd(), args.source, args.snapshot, args.reports)
    print(json.dumps(result, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
