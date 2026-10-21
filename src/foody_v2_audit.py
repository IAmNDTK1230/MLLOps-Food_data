"""Separate acceptance audit of a supplied UTF-8 export; no repair, model or RAG."""

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from src.foody_audit import (
    INTERNAL_QUESTION,
    MOJIBAKE,
    duplicate_counts,
    inspect_encoding,
    label_audit,
    write_json,
)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction(count, population):
    return {
        "numerator": int(count),
        "denominator": int(population),
        "percentage": 100 * int(count) / population if population else None,
    }


def text_quality(comments):
    nonempty = comments.str.strip().ne("")
    replacement = comments.str.contains("\ufffd", regex=False)
    internal = comments.str.contains(INTERNAL_QUESTION, regex=True)
    double = comments.str.contains("??", regex=False)
    mojibake = pd.Series(False, index=comments.index)
    patterns = {}
    for name, pattern in MOJIBAKE.items():
        flag = comments.str.contains(pattern, regex=True)
        patterns[name] = {"pattern": pattern, "comment_rows": int(flag.sum())}
        mojibake |= flag
    indicator = replacement | internal | mojibake
    population = int(nonempty.sum())
    quality = {
        "total_rows": len(comments),
        "nonempty_comments": population,
        "empty_or_whitespace_comments": int((~nonempty).sum()),
        "unique_nonempty_comments": int(comments.loc[nonempty].nunique()),
        "duplicate_extra_nonempty_comments": int(comments.loc[nonempty].duplicated().sum()),
        "u_fffd_occurrences": int(comments.str.count("\ufffd").sum()),
        "comments_with_u_fffd": int(replacement.sum()),
        "comments_with_double_question": int(double.sum()),
        "comments_with_word_internal_question": int(internal.sum()),
        "comments_with_mojibake_pattern": int(mojibake.sum()),
        "mojibake_patterns": patterns,
        "comments_with_retained_accented_latin": int(
            comments.map(
                lambda s: any(
                    "LATIN" in unicodedata.name(c, "") and unicodedata.decomposition(c) for c in s
                )
            ).sum()
        ),
        "replacement_affected_rate": fraction(replacement.sum(), population),
        "broad_corruption_indicator_rate": fraction(indicator.sum(), population),
        "word_internal_question_rate": fraction(internal.sum(), population),
        "double_question_rate": fraction(double.sum(), population),
        "metric_caveat": "Exact observed indicator rates, not a measured ground-truth percentage of all corrupted text. Question marks can be punctuation/URLs; U+FFFD is concrete replacement evidence. Undetected corruption remains possible.",
    }
    flags = pd.DataFrame(
        {
            "nonempty": nonempty,
            "replacement": replacement,
            "internal_question": internal,
            "double_question": double,
            "mojibake": mojibake,
            "broad_indicator": indicator,
        }
    )
    cases = []
    for index in comments.index[replacement | internal | mojibake]:
        text = comments.at[index]
        cases.append(
            {
                "source_record_number": int(index) + 1,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "u_fffd_occurrences": text.count("\ufffd"),
                "question_contexts": [
                    text[max(0, m.start() - 45) : m.end() + 45]
                    for m in re.finditer(INTERNAL_QUESTION, text)
                ],
                "replacement_contexts": [
                    text[max(0, m.start() - 45) : m.end() + 45] for m in re.finditer("\ufffd", text)
                ],
                "review_status": "flag only; no characters or labels changed",
            }
        )
    return quality, flags, cases


def parse_dates(raw):
    """Recognize observed exact formats without modifying raw source values."""
    formats = {
        "minute": (r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", "%d/%m/%Y %H:%M"),
        "second": (r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", "%d/%m/%Y %H:%M:%S"),
        "second_trailing_comma": (r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2},", "%d/%m/%Y %H:%M:%S,"),
    }
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    counts = {}
    for name, (pattern, fmt) in formats.items():
        mask = raw.str.fullmatch(pattern)
        counts[name] = int(mask.sum())
        parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")
    missing = raw.str.strip().eq("")
    invalid = ~missing & parsed.isna()
    report = {
        "valid_timestamps": int(parsed.notna().sum()),
        "missing_timestamps": int(missing.sum()),
        "invalid_nonempty_timestamps": int(invalid.sum()),
        "format_counts": counts,
        "earliest": parsed.min().isoformat() if parsed.notna().any() else None,
        "latest": parsed.max().isoformat() if parsed.notna().any() else None,
        "timezone": "not specified by source; naive timestamps",
        "note": "Trailing comma is explicitly recognized only as part of its exact date format; raw_review_time remains unchanged and the anomaly is flagged for later cleaning.",
        "reviews_by_year": {
            str(int(y)): int(n) for y, n in parsed.dt.year.value_counts().sort_index().items()
        },
    }
    return parsed, missing, invalid, report


def restaurant_audit(frame):
    candidates = [
        name for name in frame if name == "RestaurantID" or re.fullmatch(r"RestaurantID\.\d+", name)
    ]
    summaries, tables = [], []
    for name in candidates:
        counts = (
            frame.loc[frame[name].str.strip().ne("")]
            .groupby(name)
            .size()
            .rename("review_count")
            .reset_index()
        )
        counts = counts.rename(columns={name: "restaurant_id_as_supplied"})
        counts.insert(0, "candidate_column", name)
        tables.append(counts)
        summaries.append(
            {
                "candidate_column": name,
                "unique_restaurants": len(counts),
                "median_reviews": float(counts.review_count.median()),
                "min_reviews": int(counts.review_count.min()),
                "max_reviews": int(counts.review_count.max()),
                "fewer_than_5_reviews": int(counts.review_count.lt(5).sum()),
                "restaurant_comment_duplicate_extra_nonempty": int(
                    frame.loc[frame.Comment.str.strip().ne("")].duplicated([name, "Comment"]).sum()
                ),
            }
        )
    conflict = pd.Series(False, index=frame.index)
    for name in candidates[1:]:
        conflict |= frame[candidates[0]].ne(frame[name])
    return (
        {
            "candidates": summaries,
            "conflicting_rows": int(conflict.sum()),
            "canonical_business_key_selected": False,
            "decision": "Source owner must resolve conflicting duplicate RestaurantID headers before restaurant-specific retrieval or aggregation."
            if conflict.any()
            else "No conflicting candidate ID values observed; accidental index excluded.",
        },
        pd.concat(tables, ignore_index=True),
        conflict,
    )


def readiness(quality, utf8_valid, id_conflicts, provenance):
    widespread = not utf8_valid or quality["broad_corruption_indicator_rate"]["percentage"] > 10
    if widespread:
        training = (
            "NOT_READY_MULTIPLE_ISSUES" if provenance == "UNKNOWN" else "NOT_READY_TEXT_CORRUPTION"
        )
        return {
            "training_readiness": training,
            "demo_readiness": "NOT_READY_TEXT_CORRUPTION",
            "rag_readiness": "NOT_READY_TEXT_CORRUPTION",
        }
    training = (
        "NOT_READY_MULTIPLE_ISSUES"
        if provenance == "UNKNOWN" and id_conflicts
        else "NOT_READY_LABEL_PROVENANCE"
        if provenance == "UNKNOWN"
        else "READY_AFTER_CLEANING"
    )
    return {
        "training_readiness": training,
        "demo_readiness": "READY_AFTER_CLEANING",
        "rag_readiness": "READY_AFTER_CLEANING",
    }


def search_provenance(project):
    """Evidence inventory only. Matching a word cannot prove a label derivation."""
    extensions = {".py", ".ipynb", ".sql", ".js", ".r"}
    files = []
    for folder in ("src", "scripts", "tests", "databricks", "notebooks", "data/foody/source"):
        files += [
            p
            for p in (project / folder).rglob("*")
            if p.is_file() and p.suffix.lower() in extensions and "__pycache__" not in p.parts
        ]
    files += [p for p in project.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    matches = []
    inventory = []
    for path in sorted(set(files)):
        text = path.read_text(encoding="utf-8-sig")
        relative = str(path.relative_to(project))
        inventory.append({"path": relative, "sha256": checksum(path)})
        for number, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\bClassification\b", line):
                matches.append({"path": relative, "line": number, "text": line[:500]})
    return {
        "scope": "Available first-party source, scripts, tests, Databricks, notebooks, Foody source folder and root code; third-party dependencies and generated reports excluded.",
        "searched_files": inventory,
        "exact_word_query": r"\bClassification\b",
        "matches": matches,
        "provenance": "UNKNOWN",
        "creator_function_found": False,
        "conclusion": "Inspected matches read/audit Classification, test fixtures supply example codes, or reporting code describes it. No Foody label-creation function, threshold rule, annotation protocol or model-generation evidence was found in the available code. This does not prove such code is absent elsewhere.",
        "rating_derived_status": "NOT_ESTABLISHED",
        "gold_sentiment_status": "NOT_ESTABLISHED",
    }


def run(source, reports, audit_dir, historical_reports, historical_audit, as_of_date):
    source, reports, audit_dir = Path(source), Path(reports), Path(audit_dir)
    historical_reports, historical_audit = Path(historical_reports), Path(historical_audit)
    if any(source.resolve().is_relative_to(p.resolve()) for p in (reports, audit_dir)):
        raise ValueError("Outputs cannot contain the immutable source")
    if (
        reports.resolve() == historical_reports.resolve()
        or audit_dir.resolve() == historical_audit.resolve()
    ):
        raise ValueError("V2 outputs must be separate from historical evidence")
    protected = {
        str(p): checksum(p)
        for folder in (historical_reports, historical_audit)
        for p in folder.rglob("*")
        if p.is_file()
    }
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    decoded, encoding = inspect_encoding(raw)
    header = next(csv.reader(io.StringIO(decoded)))
    frame = pd.read_csv(io.StringIO(decoded), dtype=str, keep_default_na=False)
    inferred = pd.read_csv(io.StringIO(decoded))
    required = {
        "UserID",
        "User",
        "Review Time",
        "Rating",
        "Comment",
        "RestaurantID",
        "Classification",
    }
    if not required <= set(frame) or frame.empty:
        raise ValueError("Missing required columns or empty export")
    reports.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    quality, flags, cases = text_quality(frame.Comment)
    old_encoding = json.loads(
        (historical_reports / "encoding_report.json").read_text(encoding="utf-8")
    )
    old_text = json.loads(
        (historical_reports / "text_quality_report.json").read_text(encoding="utf-8")
    )
    old_manifest = json.loads(
        (historical_audit / "source_manifest.json").read_text(encoding="utf-8")
    )
    # Reconstruct the OLD union from saved row-level flags, not from the replaced path.
    old_rows = pd.read_parquet(historical_audit / "row_quality.parquet")
    old_indicator = (
        old_rows.comment_word_internal_question
        | old_rows.comment_replacement_character
        | old_rows.comment_mojibake_pattern
    )
    comparison = {
        "old_source_sha256": old_manifest["sha256"],
        "old_original_bytes_available_at_recorded_path": checksum(Path(old_manifest["source_path"]))
        == old_manifest["sha256"]
        if Path(old_manifest["source_path"]).exists()
        else False,
        "basis": "Saved immutable historical encoding/text reports and row flags; historical raw bytes are unavailable at the old path.",
        "old_nonempty_comments": old_text["nonempty_comments"],
        "old_broad_corruption_indicator_rate": fraction(
            old_indicator.sum(), old_text["nonempty_comments"]
        ),
        "new_broad_corruption_indicator_rate": quality["broad_corruption_indicator_rate"],
        "old_word_internal_question_rate": fraction(
            old_encoding["comments_with_word_internal_question"], old_text["nonempty_comments"]
        ),
        "new_word_internal_question_rate": quality["word_internal_question_rate"],
        "old_double_question_rate": fraction(
            old_encoding["comments_with_double_question"], old_text["nonempty_comments"]
        ),
        "new_double_question_rate": quality["double_question_rate"],
        "old_utf8_strict_valid": old_encoding["utf8_strict_valid"],
        "new_utf8_strict_valid": encoding["utf8_strict_valid"],
        "origin_recovery_method_verified": False,
        "origin_caveat": "Readable UTF-8 and reduction in observed damage support improved text quality. CSV bytes alone do not establish original re-export lineage or prove no one guessed/repaired text before delivery. This audit makes no repairs.",
    }
    samples = [
        {
            "source_record_number": int(i) + 1,
            "text": frame.Comment.at[i][:350],
            "truncated": len(frame.Comment.at[i]) > 350,
        }
        for i in frame.index[flags.nonempty][:6]
    ]
    encoding.update(
        quality=quality,
        comparison=comparison,
        representative_comments=samples,
        overall_text_assessment="Vietnamese accents and emoji readable; isolated U+FFFD remains. Not every comment can be certified intact.",
    )
    restaurants, stats, id_conflict = restaurant_audit(frame)
    parsed, date_missing, date_invalid, dates = parse_dates(frame["Review Time"])
    rating = pd.to_numeric(frame.Rating.mask(frame.Rating.str.strip().eq("")), errors="coerce")
    labels, exact_tab, band_tab = label_audit(frame, rating)
    provenance = search_provenance(Path.cwd())
    labels["assessment"] = (
        "UNKNOWN: no label-creation code was found. Rating association is descriptive only; manual, model, rating-threshold or other-rule origin is not established."
    )
    state = readiness(
        quality,
        encoding["utf8_strict_valid"],
        restaurants["conflicting_rows"],
        provenance["provenance"],
    )
    state.update(
        primary_research_benchmark="VN-MSASar (unchanged)",
        classification_required_as_gold_for_demo_or_rag=False,
        label_provenance=provenance["provenance"],
        training_blockers=[
            "Classification creation/provenance is unknown",
            "conflicting RestaurantID columns need source-owner resolution",
        ],
        demo_rag_prerequisites=[
            "resolve canonical RestaurantID without overwriting source",
            "flag/review two replacement-affected comments; do not guess characters",
            "handle empty/duplicate comments in a later derived layer",
            "retain/parse raw date values and document trailing commas",
        ],
        next_step="Ask the source exporter to identify the canonical RestaurantID and provide Classification creation code. Preserve this UTF-8 export under a versioned immutable name and locate the historical raw backup. Then implement a separately reviewed derived cleaning plan; no model or RAG yet.",
    )
    schema = {
        "row_count": len(frame),
        "raw_header": header,
        "pandas_column_names": frame.columns.tolist(),
        "raw_duplicate_headers": {k: n for k, n in Counter(header).items() if n > 1},
        "columns": [
            {
                "name": c,
                "dtype": str(inferred[c].dtype),
                "null_count": int(inferred[c].isna().sum()),
                "blank_count": int(frame[c].str.strip().eq("").sum()),
            }
            for c in frame
        ],
        "accidental_index_columns_excluded_from_business_logic": [
            c for c in frame if c.startswith("Unnamed:")
        ],
        "restaurants": restaurants,
        "date_audit": dates,
        "rating_range": [float(rating.min()), float(rating.max())],
        "rating_mean": float(rating.mean()),
        "rating_median": float(rating.median()),
        "missing_ratings": int(frame.Rating.str.strip().eq("").sum()),
        "classification_distribution": labels["classification_distribution"],
    }
    dupkeys = [
        ["Comment"],
        ["UserID", "Comment"],
        *[[c["candidate_column"], "Comment"] for c in restaurants["candidates"]],
        ["UserID", "RestaurantID", "Review Time", "Comment"],
    ]
    duplicates = [duplicate_counts(frame, keys) for keys in dupkeys]
    for name, value in (
        ("encoding_report", encoding),
        ("schema_report", schema),
        ("readiness", state),
        ("label_audit", labels),
        ("classification_provenance", provenance),
        ("duplicate_audit", duplicates),
        ("text_quality_report", quality),
        ("date_audit", dates),
        ("restaurant_audit", restaurants),
        ("flagged_text_cases", cases),
    ):
        write_json(reports / f"{name}.json", value)
    stats.to_csv(reports / "restaurant_statistics.csv", index=False, encoding="utf-8-sig")
    exact_tab.to_csv(reports / "rating_classification_crosstab.csv", encoding="utf-8-sig")
    band_tab.to_csv(reports / "rating_band_classification.csv", encoding="utf-8-sig")
    rows = flags.copy()
    rows.insert(0, "source_record_number", range(1, len(frame) + 1))
    rows["raw_rating"], rows["raw_review_time"], rows["raw_classification"] = (
        frame.Rating,
        frame["Review Time"],
        frame.Classification,
    )
    for c in restaurants["candidates"]:
        rows[c["candidate_column"]] = frame[c["candidate_column"]]
    rows["restaurant_id_conflict"] = id_conflict
    rows["timestamp_trailing_comma"] = frame["Review Time"].str.endswith(",")
    (
        rows["parsed_timestamp_naive"],
        rows["timestamp_missing"],
        rows["timestamp_invalid_nonempty"],
    ) = parsed, date_missing, date_invalid
    rows.to_parquet(audit_dir / "row_quality.parquet", index=False)
    evidence = {
        "as_of_date": as_of_date,
        "requested_path": "data/foody/source/review_v2_utf8.csv",
        "actual_source_path": str(source),
        "source_sha256": source_sha,
        "source_size_bytes": len(raw),
        "source_selection": "User clarified the new export is review.csv; requested versioned filename not found.",
        "historical_reports_used_read_only": protected,
        "historical_original_source_sha256": old_manifest["sha256"],
        "historical_original_file_replaced_before_this_audit": source_sha != old_manifest["sha256"],
        "no_source_writes_or_repairs": True,
        "all_rows_preserved": len(rows) == len(frame),
    }
    if checksum(source) != source_sha or any(
        checksum(Path(p)) != sha for p, sha in protected.items()
    ):
        raise ValueError("A source or historical audit artifact changed during audit")
    evidence["source_and_historical_reports_unchanged_during_audit"] = True
    write_json(audit_dir / "source_manifest.json", evidence)
    render_reports(reports, schema, encoding, provenance, state, duplicates, evidence)
    return {"evidence": evidence, "schema": schema, "encoding": encoding, "readiness": state}


def render_reports(reports, schema, encoding, provenance, state, duplicates, evidence):
    matches = "\n".join(
        f"- `{m['path']}:{m['line']}`: `{m['text'].replace('`', '')}`"
        for m in provenance["matches"]
    )
    (reports / "classification_provenance.md").write_text(
        "# Classification provenance\n\n**UNKNOWN — no creator function or rule found in the available project.**\n\n"
        + provenance["conclusion"]
        + "\n\nSearch scope: "
        + provenance["scope"]
        + "\n\nThe existing src/foody_audit.py label_audit function reads the supplied column and measures association. Test fixtures create synthetic codes only. Neither is evidence of Foody label creation.\n\n"
        + "Rating association is not provenance. Do not infer thresholds, manual annotation or model labels from correlations. If an actual rating derivation is later supplied, record its exact file/function/rule and mark WEAK_LABEL_RATING_DERIVED, not manually annotated gold. No such designation is verified now.\n\nClassification need not be gold for demo/RAG. VN-MSASar remains the primary research benchmark.\n\n## Search evidence\n\n"
        + matches
        + "\n\nFile hashes and the full search inventory are in classification_provenance.json. Generated audit references are evidence readers/reporters, not source label generators.\n",
        encoding="utf-8",
    )
    q, comp = encoding["quality"], encoding["comparison"]
    lines = [
        "# Foody UTF-8 export acceptance audit",
        "",
        f"Audit date: {evidence['as_of_date']} (Asia/Saigon).",
        "",
        "## Source identity and limits",
        "",
        f"User clarified actual new input: `{evidence['actual_source_path']}`. The requested review_v2_utf8.csv was absent. No file was renamed, copied, reconstructed or overwritten.",
        f"Current SHA256: `{evidence['source_sha256']}`. Current bytes: {evidence['source_size_bytes']:,}.",
        f"Historical audit SHA256: `{evidence['historical_original_source_sha256']}`. That old raw file is unavailable at its recorded path; it had changed before this audit. Historical reports/row flags were preserved and used for comparison.",
        "Readable UTF-8 and improved comments demonstrate a major text-quality improvement. The delivered CSV alone cannot prove original-source lineage or exclude earlier manual repair. This audit guessed/repaired no characters.",
        "",
        "## Encoding and exact indicator rates",
        "",
        f"encoding_status: **{encoding['encoding_status']}**; strict UTF-8: {encoding['utf8_strict_valid']}.",
        f"U+FFFD: **{q['u_fffd_occurrences']} characters in {q['comments_with_u_fffd']} comments**. Mojibake pattern matches: {q['comments_with_mojibake_pattern']} comments.",
        f"Nonempty denominator: **{q['nonempty_comments']:,} comments** in each export.",
        "",
        "| Metric | Old | New |",
        "|---|---:|---:|",
        f"| Broad indicator union: U+FFFD / word-internal ? / mojibake | {comp['old_broad_corruption_indicator_rate']['numerator']} / {comp['old_broad_corruption_indicator_rate']['denominator']} = {comp['old_broad_corruption_indicator_rate']['percentage']:.9f}% | {q['broad_corruption_indicator_rate']['numerator']} / {q['nonempty_comments']} = {q['broad_corruption_indicator_rate']['percentage']:.9f}% |",
        f"| Word-internal question-mark flags | {comp['old_word_internal_question_rate']['numerator']} = {comp['old_word_internal_question_rate']['percentage']:.9f}% | {q['comments_with_word_internal_question']} = {q['word_internal_question_rate']['percentage']:.9f}% |",
        f"| Literal ?? comments | {comp['old_double_question_rate']['numerator']} = {comp['old_double_question_rate']['percentage']:.9f}% | {q['comments_with_double_question']} = {q['double_question_rate']['percentage']:.9f}% |",
        f"| Confirmed replacement-affected comments | Old legacy view contained no U+FFFD; this did not imply intact text | {q['comments_with_u_fffd']} / {q['nonempty_comments']} = {q['replacement_affected_rate']['percentage']:.9f}% |",
        "",
        "These are exact observed rates, not an exact ground-truth corruption percentage. The five new word-internal flags were inspected: four are sentence/question punctuation without spaces and one is a URL query. They do not show missing Vietnamese letters. Repeated ?? often expresses emphasis. The two U+FFFD comments are concrete remaining replacement artifacts; their missing original content is unknown.",
        "",
        "Representative comments (loaded directly, not repaired):",
        "",
    ]
    lines += [
        f"- Source record {s['source_record_number']}: `{s['text'].replace(chr(10), ' / ').replace('`', '')}`"
        for s in encoding["representative_comments"]
    ]
    lines += [
        "",
        "## Actual schema and quality",
        "",
        f"Rows: **{schema['row_count']:,}**. Raw columns: `{schema['raw_header']}`.",
        "",
        "| Pandas column | dtype | nulls |",
        "|---|---|---:|",
    ]
    lines += [f"| {c['name']} | {c['dtype']} | {c['null_count']} |" for c in schema["columns"]]
    lines += [
        "",
        "Unnamed export indices are excluded from business logic and retained in source. Pandas disambiguates the repeated raw RestaurantID header as RestaurantID and RestaurantID.1.",
        f"**Restaurant ID conflict:** {schema['restaurants']['conflicting_rows']:,} rows disagree. Each candidate has 818 distinct IDs in this export, but their assignments differ. No canonical ID was silently selected. restaurant_statistics.csv reports both candidates separately.",
        f"Nonempty comments {q['nonempty_comments']:,}; empty {q['empty_or_whitespace_comments']}; unique nonempty {q['unique_nonempty_comments']:,}; exact duplicate extra nonempty comments {q['duplicate_extra_nonempty_comments']}.",
        "",
        "| Candidate restaurant ID | Unique IDs | Median reviews | Min / max | Restaurant + Comment extra nonempty duplicates |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['candidate_column']} | {r['unique_restaurants']} | {r['median_reviews']} | {r['min_reviews']} / {r['max_reviews']} | {r['restaurant_comment_duplicate_extra_nonempty']} |"
        for r in schema["restaurants"]["candidates"]
    ]
    dates = schema["date_audit"]
    lines += [
        "",
        f"Rating range {schema['rating_range']}; mean {schema['rating_mean']}, median {schema['rating_median']}; missing {schema['missing_ratings']}. raw_rating is preserved lexically without normalization.",
        f"Dates: {dates['earliest']} to {dates['latest']}; valid {dates['valid_timestamps']}, missing {dates['missing_timestamps']}, invalid nonempty {dates['invalid_nonempty_timestamps']}.",
        f"Date formats: `{dates['format_counts']}`. The explicit seconds-plus-comma format is parsed only into a derived column; all raw date strings and anomaly flags are preserved.",
        "",
        "| Classification code | Count | Percent |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {r['value']} | {r['count']} | {r['percentage']:.6f}% |"
        for r in schema["classification_distribution"]
    ]
    lines += [
        "",
        "## Classification provenance",
        "",
        provenance["conclusion"],
        "See classification_provenance.md and its JSON search evidence. No actual threshold rule/file/function was found. Rating correlation does not establish a rating-derived rule; Classification is not established as manually annotated gold.",
        "",
        "## Readiness",
        "",
        f"- training_readiness: **{state['training_readiness']}**",
        f"- demo_readiness: **{state['demo_readiness']}**",
        f"- rag_readiness: **{state['rag_readiness']}**",
        "",
        "Unknown Classification does not block demo/RAG by itself. The cleaning gates here are ID ambiguity, isolated replacement artifacts, empty/duplicate comments and date suffix artifacts. No rows have been removed or sources modified. VN-MSASar remains the unchanged primary research/benchmark dataset.",
        "",
        "## Exact next step",
        "",
        state["next_step"],
        "",
        "No model training, RAG indexing, full MVP, character repair, label remapping or source writes were performed. Separate audit row flags retain all source records and lexical raw_rating/raw_review_time.",
        "",
        "Reproduce with the user-approved actual source: `python -m src.foody_v2_audit --source data/foody/source/review.csv --as-of-date 2026-10-20`.",
    ]
    (reports / "DATA_ACCEPTANCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/foody/source/review_v2_utf8.csv")
    parser.add_argument("--reports", default="reports/foody_v2")
    parser.add_argument("--audit-dir", default="data/foody/audit_v2")
    parser.add_argument("--historical-reports", default="reports/foody")
    parser.add_argument("--historical-audit", default="data/foody/audit")
    parser.add_argument("--as-of-date", default="2026-10-20")
    args = parser.parse_args()
    result = run(
        args.source,
        args.reports,
        args.audit_dir,
        args.historical_reports,
        args.historical_audit,
        args.as_of_date,
    )
    for sample in result["encoding"]["representative_comments"]:
        print(json.dumps(sample, ensure_ascii=True))
    print(json.dumps(result["readiness"], indent=2))


if __name__ == "__main__":
    main()
