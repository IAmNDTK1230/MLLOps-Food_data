"""Read-only acceptance audit of the real Foody CSV. No cleaning or model training."""

import argparse
import codecs
import hashlib
import io
import json
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REQUIRED = {"UserID", "User", "Review Time", "Rating", "Comment", "RestaurantID", "Classification"}
INTERNAL_QUESTION = r"(?<=[^\W_])\?+(?=[^\W_])"
MOJIBAKE = {
    "utf8_vietnamese_as_western_bytes": r"(?:áº|á»|Ä‘|Ä\x91|Æ°|Æ¡)",
    "utf8_latin_as_western_bytes": r"(?:Ã[\u0080-\u00bf]|Â[\u0080-\u00bf])",
    "replacement_rendered_as_mojibake": r"ï¿½",
}


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def inspect_encoding(raw):
    """A strict UTF-8 failure is evidence about bytes, not Excel or alone text loss.

    CP1252 is a reversible inspection view when UTF-8 fails, not an asserted source
    encoding or a repair. Literal question marks are never replaced or guessed.
    """
    try:
        text = raw.decode("utf-8-sig", errors="strict")
        encoding, error = "utf-8-sig", None
    except UnicodeDecodeError as exc:
        error = {
            "start_byte": exc.start,
            "end_byte_exclusive": exc.end,
            "reason": exc.reason,
            "nearby_bytes_hex": raw[max(0, exc.start - 8) : exc.end + 8].hex(),
        }
        try:
            text, encoding = raw.decode("cp1252", errors="strict"), "cp1252"
        except UnicodeDecodeError:
            # Latin-1 is a byte-preserving diagnostic fallback, not a language claim.
            text, encoding = raw.decode("latin-1", errors="strict"), "latin-1"
    codec = "utf-8" if encoding == "utf-8-sig" else encoding
    roundtrip = text.encode(codec) == (
        raw.removeprefix(codecs.BOM_UTF8) if encoding == "utf-8-sig" else raw
    )
    return text, {
        "utf8_strict_valid": error is None,
        "encoding_status": "VALID_UTF8" if error is None else "INVALID_UTF8_LEGACY_VIEW_ONLY",
        "utf8_bom_present": raw.startswith(codecs.BOM_UTF8),
        "utf8_decode_error": error,
        "inspection_encoding": encoding,
        "inspection_encoding_is_verified_original": error is None,
        "inspection_view_byte_roundtrip": roundtrip,
        "raw_utf8_replacement_sequence_count": raw.count(b"\xef\xbf\xbd"),
        "inspection_view_u_fffd_count": text.count("\ufffd"),
        "replacement_count_if_forced_utf8_errors_replace_DIAGNOSTIC_ONLY": raw.decode(
            "utf-8", errors="replace"
        ).count("\ufffd"),
        "raw_literal_double_question_nonoverlapping_count": raw.count(b"??"),
        "first_48_raw_bytes_hex": raw[:48].hex(),
        "notes": "No Excel display used. Legacy view preserves bytes but does not identify original encoding or restore characters. No lossy decoding is used for CSV parsing.",
    }


def distribution(series):
    total = len(series)
    return [
        {
            "value": None if pd.isna(value) else str(value),
            "count": int(count),
            "percentage": 100 * int(count) / total if total else None,
        }
        for value, count in series.value_counts(dropna=False).items()
    ]


def duplicate_counts(frame, columns):
    """Report extra copies and all group members, including/excluding missing keys."""
    complete = frame.loc[
        frame[columns].notna().all(axis=1)
        & frame[columns].astype("string").apply(lambda c: c.str.strip().ne("")).all(axis=1)
    ]

    def counts(data):
        return {
            "rows_evaluated": len(data),
            "extra_rows_after_first": int(data.duplicated(columns).sum()),
            "rows_in_duplicate_groups": int(data.duplicated(columns, keep=False).sum()),
            "duplicate_groups": int(
                data.loc[data.duplicated(columns, keep=False), columns].drop_duplicates().shape[0]
            ),
        }

    return {
        "keys": columns,
        "all_rows_missing_keys_compared_as_equal": counts(frame),
        "complete_keys_only": counts(complete),
        "incomplete_key_rows": len(frame) - len(complete),
    }


def timestamp_audit(raw):
    missing = raw.str.strip().eq("")
    parsed = pd.to_datetime(raw.mask(missing), format="%d/%m/%Y %H:%M", errors="coerce")
    invalid = ~missing & parsed.isna()
    return (
        parsed,
        missing,
        invalid,
        {
            "parse_format": "%d/%m/%Y %H:%M",
            "day_first": True,
            "timezone": "not supplied by source; parsed as naive local timestamps",
            "valid_timestamps": int(parsed.notna().sum()),
            "missing_timestamps": int(missing.sum()),
            "invalid_nonempty_timestamps": int(invalid.sum()),
            "unparsed_total": int(parsed.isna().sum()),
            "earliest": parsed.min().isoformat() if parsed.notna().any() else None,
            "latest": parsed.max().isoformat() if parsed.notna().any() else None,
            "invalid_value_counts": distribution(raw.loc[invalid]),
            "reviews_by_year": {
                str(int(k)): int(v) for k, v in parsed.dt.year.value_counts().sort_index().items()
            },
        },
    )


def label_audit(raw, rating):
    labels = raw["Classification"]
    numeric = pd.to_numeric(labels, errors="coerce")
    crosstab = pd.crosstab(
        rating.fillna("MISSING_RATING"), labels.replace("", "MISSING_CLASSIFICATION"), dropna=False
    )
    eligible = rating.notna() & labels.str.strip().ne("")
    groups = pd.DataFrame({"rating": rating.loc[eligible], "label": labels.loc[eligible]}).groupby(
        "rating"
    )
    mixed_values = groups.label.nunique()
    mixed = mixed_values[mixed_values.gt(1)].index.tolist()
    agreement = (
        sum(group.label.value_counts().max() for _, group in groups) / eligible.sum()
        if eligible.any()
        else None
    )
    band_names = ["below_6", "6_to_below_7", "7_to_below_8.6", "8.6_and_above"]
    bands = (
        pd.cut(rating, [-np.inf, 6, 7, 8.6, np.inf], right=False, labels=band_names)
        .astype("string")
        .fillna("MISSING_RATING")
    )
    band_table = pd.crosstab(bands, labels.replace("", "MISSING_CLASSIFICATION"), dropna=False)
    rho = (
        rating.corr(numeric, method="spearman")
        if (eligible.sum() > 1 and numeric.nunique() > 1 and rating.nunique() > 1)
        else None
    )
    return (
        {
            "classification_distribution": distribution(labels.replace("", pd.NA)),
            "unique_nonempty_values": sorted(
                labels.loc[labels.str.strip().ne("")].unique().tolist()
            ),
            "label_provenance": "UNKNOWN",
            "manual_annotation_evidence": False,
            "assessment": "Rating association is observational. A rating-derived rule is plausible but unverified; manual annotation is not established. Do not use Classification as gold sentiment.",
            "semantics": "Original numeric codes preserved; their sentiment meaning is not asserted without provenance.",
            "rating_values_with_multiple_classifications": mixed,
            "rows_at_ambiguous_rating_values": int(rating.isin(mixed).sum()),
            "rating_band_classification_counts": [
                {
                    "rating_band": str(band),
                    "classification_counts": {str(k): int(v) for k, v in row.items()},
                }
                for band, row in band_table.iterrows()
            ],
            "majority_label_per_exact_rating_agreement": float(agreement)
            if agreement is not None
            else None,
            "numeric_code_vs_rating_spearman": float(rho)
            if rho is not None and pd.notna(rho)
            else None,
            "majority_mapping_note": "Descriptive frequency statistic only; no estimator was fitted or label mapping applied.",
            "missing_rating_classification_distribution": distribution(labels.loc[rating.isna()]),
            "missing_classification_rows": int(labels.str.strip().eq("").sum()),
            "provenance_evidence_available": "Only this CSV was supplied; it contains no annotation protocol or derivation code.",
        },
        crosstab,
        band_table,
    )


def make_report(result, reports):
    schema, enc, text = result["schema"], result["encoding"], result["text"]
    labels, dates, ratings = result["labels"], result["dates"], result["ratings"]
    lines = [
        "# Real Foody data acceptance and quality audit",
        "",
        f"Audit date: {result['as_of_date']} (Asia/Saigon).",
        "",
        f"**training_readiness: {result['training_readiness']}**",
        f"**demo_rag_readiness: {result['demo_rag_readiness']}**",
        "",
        result["decision_reason"],
        "",
        "## Immutable source and scope",
        "",
        f"Source: `{result['source_path']}`. Bytes: {result['source_size_bytes']:,}.",
        f"SHA256 before and after: `{result['source_sha256']}`. Source unchanged: {result['source_unchanged']}.",
        "No rows were removed, text repaired, ratings normalized, labels remapped, models trained or MVP built.",
        "",
        "## Actual schema",
        "",
        f"Rows: **{schema['row_count']:,}**. Columns: **{schema['column_count']}**.",
        "",
        "| Column | Pandas inferred dtype | Null count | Blank/whitespace count |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f"| {r['name']} | {r['pandas_inferred_dtype']} | {r['pandas_default_null_count']} | {r['raw_blank_or_whitespace_count']} |"
        for r in schema["columns"]
    ]
    lines += [
        "",
        "`Unnamed: 0` is retained for audit only. " + schema["exported_index_assessment"],
        "",
        "`UserID` observations: " + json.dumps(schema["user_id_observations"], ensure_ascii=False),
        "A unique/sequential UserID may be an assigned row identifier; it does not establish unique human reviewers or meaningful user-level deduplication.",
        "",
        "## Encoding evidence",
        "",
        f"encoding_status: **{enc['encoding_status']}**.",
        f"Strict UTF-8: {enc['utf8_strict_valid']}. Inspection view: {enc['inspection_encoding']}; byte round-trip: {enc['inspection_view_byte_roundtrip']}.",
        f"UTF-8 error evidence: `{json.dumps(enc['utf8_decode_error'])}`.",
        "A legacy encoding alone would not establish corruption. The following additional text-loss indicators are audited separately:",
        f"- Raw nonoverlapping `??` byte pairs: {enc['raw_literal_double_question_nonoverlapping_count']:,}.",
        f"- Comments containing `??`: {enc['comments_with_double_question']:,}.",
        f"- Comments with word-internal question-mark runs: {enc['comments_with_word_internal_question']:,}.",
        f"- U+FFFD in inspection view: {enc['inspection_view_u_fffd_count']}; raw UTF-8 replacement sequences: {enc['raw_utf8_replacement_sequence_count']}.",
        f"- Forced lossy UTF-8 decoding would insert {enc['replacement_count_if_forced_utf8_errors_replace_DIAGNOSTIC_ONLY']:,} replacement characters. These are diagnostic artifacts, not replacement characters observed in the parsed legacy view.",
        "Some Vietnamese accents survive; this does not establish intact Vietnamese text. The inspection codec is not verified original encoding. No missing characters were guessed.",
        "",
        "Representative loaded Comment fields (literal source punctuation retained):",
        "",
    ]
    for sample in enc["representative_comments"]:
        lines += [
            f"Source record {sample['source_record_number']}: `{sample['text'].replace(chr(10), ' / ').replace('`', '')}`",
            "",
        ]
    lines += [
        "Mojibake flags are heuristic, may have false positives and do not establish original spelling. Counts are in encoding_report.json.",
        "",
        "## Text and duplicates",
        "",
        f"Non-null comments: {text['non_null_comments']:,}. Empty/whitespace comments: {text['empty_or_whitespace_comments']:,}.",
        f"Nonempty unique exact comments: {text['unique_nonempty_comments']:,}; duplicate extra nonempty comments after first: {text['exact_duplicate_nonempty_comments_extra_rows']:,}.",
        f"Nonempty character length: mean {text['mean_characters']}, median {text['median_characters']}, min {text['min_characters']}, max {text['max_characters']}.",
        "Lengths/counts refer to the unmodified inspection view; no whitespace/accent normalization or duplicate deletion was performed.",
        "",
        "| Duplicate key | Extra rows (all / complete keys) | All group members (complete keys) |",
        "|---|---:|---:|",
    ]
    for item in result["duplicates"]:
        lines += [
            f"| {' + '.join(item['keys'])} | {item['all_rows_missing_keys_compared_as_equal']['extra_rows_after_first']} / {item['complete_keys_only']['extra_rows_after_first']} | {item['complete_keys_only']['rows_in_duplicate_groups']} |"
        ]
    lines += [
        "",
        "Extra rows means repeated occurrences after the first; group members includes every row in repeated groups. All-row counts treat missing keys as equal. Complete-key counts exclude missing/blank fields. Both are counts, not deletion recommendations.",
        "",
        "## Classification and rating relationship",
        "",
        "| Original Classification | Count | Percent of all rows |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {r['value']} | {r['count']:,} | {r['percentage']:.4f}% |"
        for r in labels["classification_distribution"]
    ]
    classes = labels["unique_nonempty_values"]
    lines += [
        "",
        "| Observed rating band | " + " | ".join(classes) + " |",
        "|---|" + "---:|" * len(classes),
    ]
    for row in labels["rating_band_classification_counts"]:
        lines += [
            "| "
            + row["rating_band"]
            + " | "
            + " | ".join(str(row["classification_counts"].get(code, 0)) for code in classes)
            + " |"
        ]
    lines += [
        "",
        f"Label provenance: **{labels['label_provenance']}**. {labels['assessment']}",
        f"Mixed Classification values occur at exact ratings {labels['rating_values_with_multiple_classifications']}, covering {labels['rows_at_ambiguous_rating_values']} rows.",
        f"Descriptive majority-label-per-exact-rating agreement: {labels['majority_label_per_exact_rating_agreement']}. Numeric-code/rating Spearman: {labels['numeric_code_vs_rating_spearman']}.",
        "This association does not prove who annotated the labels or how they were generated. Neither manual gold status nor a specific derivation rule is established. See rating_classification_crosstab.csv and rating_band_classification.csv.",
        "",
        "## Ratings and dates",
        "",
        f"Observed numeric rating range **{ratings['min']}–{ratings['max']}**; mean {ratings['mean']}, median {ratings['median']}. Missing {ratings['missing_rows']}; nonnumeric nonempty {ratings['nonnumeric_nonempty_rows']}; zero-valued {ratings['zero_rows']}.",
        "The file supports an observed 0–10 range, not an assumed 1–5 scale. The platform's official scale and whether zero is a sentinel remain unverified. Original lexical Rating values are retained as raw_rating in the row audit, without normalization.",
        f"Dates: {dates['valid_timestamps']:,} parsed, {dates['missing_timestamps']} missing, {dates['invalid_nonempty_timestamps']} invalid nonempty; total unparsed {dates['unparsed_total']}.",
        f"Range: **{dates['earliest']} through {dates['latest']}**. Format: `{dates['parse_format']}`; source timezone unspecified.",
        f"Year counts: `{json.dumps(dates['reviews_by_year'])}`. No invalid timestamp was silently dropped.",
        "",
        "## Restaurants",
        "",
        f"Unique nonmissing restaurant IDs: **{result['restaurants']['unique_restaurants']}**.",
        f"Median reviews per restaurant: {result['restaurants']['median_reviews_per_restaurant']}. Min/max: {result['restaurants']['min_reviews_per_restaurant']} / {result['restaurants']['max_reviews_per_restaurant']}.",
        f"Restaurants with 1–2 reviews: {result['restaurants']['restaurants_with_at_most_2_reviews']}; with fewer than 5: {result['restaurants']['restaurants_with_fewer_than_5_reviews']}.",
        "restaurant_statistics.csv retains each restaurant's count; restaurant_audit.json lists top restaurants and the full count summary.",
        "",
        "## Readiness and exact next step",
        "",
        result["decision_reason"],
        "A metadata-only demo can be considered after handling missing dates/ratings and confirming ID meaning. Full review-text RAG is not accepted while pervasive text-loss indicators remain. Unknown Classification provenance independently blocks supervised training.",
        "**Next step:** obtain a fresh UTF-8/UTF-8-BOM export directly from the original crawl/database before any lossy spreadsheet conversion, together with the Classification derivation code or annotation protocol and ID/rating definitions. Save it as a new versioned source (keep this file intact), then rerun this audit. Do not infer or fill missing Vietnamese characters. If the original text cannot be recovered, scope any demo explicitly to metadata and exclude damaged text from claims of Vietnamese retrieval quality.",
        "",
        "## Outputs and verification",
        "",
        "Reports: encoding_report.json, schema_report.json, label_audit.json, text_quality_report.json, duplicate_audit.json, rating_audit.json, date_audit.json, restaurant_audit.json, restaurant_statistics.csv, rating_distribution.csv, rating_classification_crosstab.csv, rating_band_classification.csv, review_counts_by_year.csv, audit_summary.json and this report.",
        "Derived audit evidence: data/foody/audit/row_quality.parquet (all source records retained, original raw_rating/raw_review_time, issue flags), source_manifest.json and execution_checks.json. These are audit artifacts, not cleaned/training data.",
        "No Foody data preparation, supervised training, inference, MVP, external upload or source repair was performed.",
    ]
    (reports / "data_acceptance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(source, reports, audit_dir, as_of_date):
    source, reports, audit_dir = Path(source), Path(reports), Path(audit_dir)
    if reports.resolve() == audit_dir.resolve():
        raise ValueError("Report and row-audit directories must be distinct")
    # Reject every output parent containing the raw source, preventing any
    # accidental overwrite through an output name such as restaurant_statistics.csv.
    if any(source.resolve().is_relative_to(p.resolve()) for p in (reports, audit_dir)):
        raise ValueError("Output directories must not contain the immutable source")
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    decoded, enc = inspect_encoding(raw)
    inferred = pd.read_csv(io.StringIO(decoded))
    frame = pd.read_csv(io.StringIO(decoded), dtype=str, keep_default_na=False)
    if not REQUIRED <= set(frame):
        raise ValueError(f"Required source columns missing: {sorted(REQUIRED - set(frame))}")
    if not len(frame):
        raise ValueError("Source has no rows")
    reports.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    comments = frame.Comment
    nonempty = comments.str.strip().ne("")
    lengths = comments.loc[nonempty].str.len()
    double = comments.str.contains("??", regex=False)
    internal = comments.str.contains(INTERNAL_QUESTION, regex=True)
    replacement = comments.str.contains("\ufffd", regex=False)
    mojibake = pd.Series(False, index=frame.index)
    enc["mojibake_patterns"] = {}
    for name, pattern in MOJIBAKE.items():
        flag = comments.str.contains(pattern, regex=True)
        mojibake |= flag
        enc["mojibake_patterns"][name] = {"pattern": pattern, "comment_rows": int(flag.sum())}
    accented = comments.map(
        lambda s: any(
            "LATIN" in unicodedata.name(c, "") and unicodedata.decomposition(c) for c in s
        )
    )
    enc.update(
        comments_with_double_question=int(double.sum()),
        comments_with_word_internal_question=int(internal.sum()),
        comments_with_u_fffd=int(replacement.sum()),
        comments_with_mojibake_pattern=int(mojibake.sum()),
        comments_with_retained_accented_latin_characters=int(accented.sum()),
        representative_comments=[
            {
                "source_record_number": int(i) + 1,
                "text": comments.at[i][:350],
                "truncated": len(comments.at[i]) > 350,
            }
            for i in comments.loc[nonempty].index[:6]
        ],
    )
    encoding_blocker = bool(
        replacement.any() or (not enc["utf8_strict_valid"] and internal.mean() > 0.1)
    )
    enc["substantial_text_loss_evidence"] = encoding_blocker
    enc["text_loss_assessment"] = (
        "Pervasive word-internal ASCII question marks plus failed strict UTF-8 indicate likely lossy legacy export; origin/characters cannot be proved or recovered from this CSV."
        if encoding_blocker
        else "UTF-8 validity and text-loss indicators reported separately; punctuation flags alone are not proof of corruption."
    )
    schema = {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "column_names": frame.columns.tolist(),
        "columns": [
            {
                "name": c,
                "pandas_inferred_dtype": str(inferred[c].dtype),
                "pandas_default_null_count": int(inferred[c].isna().sum()),
                "raw_empty_count": int(frame[c].eq("").sum()),
                "raw_blank_or_whitespace_count": int(frame[c].str.strip().eq("").sum()),
            }
            for c in frame
        ],
        "exported_index_assessment": "Unnamed: 0 absent.",
        "dtype_note": "Pandas default inference is reported; separate dtype=str / keep_default_na=False view preserves lexical raw values and distinguishes empty fields.",
    }
    if "Unnamed: 0" in frame:
        idx = pd.to_numeric(frame["Unnamed: 0"], errors="coerce")
        schema["exported_index_observations"] = {
            "min": float(idx.min()) if idx.notna().any() else None,
            "max": float(idx.max()) if idx.notna().any() else None,
            "unique": int(frame["Unnamed: 0"].nunique()),
            "duplicate_extra_rows": int(frame["Unnamed: 0"].duplicated().sum()),
        }
        schema["exported_index_assessment"] = (
            "An unnamed integer export-index column, with repeated local indices if present; treated as accidental exported index, not a business ID. It was not dropped or used for deduplication."
        )
    usernum = pd.to_numeric(frame.UserID, errors="coerce")
    schema["user_id_observations"] = {
        "unique_nonempty": int(frame.loc[frame.UserID.ne(""), "UserID"].nunique()),
        "equals_one_based_row_sequence": bool(
            np.array_equal(usernum.to_numpy(), np.arange(1, len(frame) + 1))
        ),
    }
    text = {
        "total_rows": len(frame),
        "non_null_comments": int(inferred.Comment.notna().sum()),
        "empty_or_whitespace_comments": int((~nonempty).sum()),
        "nonempty_comments": int(nonempty.sum()),
        "unique_nonempty_comments": int(comments.loc[nonempty].nunique()),
        "exact_duplicate_nonempty_comments_extra_rows": int(
            comments.loc[nonempty].duplicated().sum()
        ),
        "mean_characters": float(lengths.mean()) if len(lengths) else None,
        "median_characters": float(lengths.median()) if len(lengths) else None,
        "min_characters": int(lengths.min()) if len(lengths) else None,
        "max_characters": int(lengths.max()) if len(lengths) else None,
        "population": "Nonempty comments; exact decoded strings; no normalization; lengths include original whitespace/newlines.",
    }
    keys = [
        ["Comment"],
        ["UserID", "Comment"],
        ["RestaurantID", "Comment"],
        ["UserID", "RestaurantID", "Review Time", "Comment"],
    ]
    duplicates = [duplicate_counts(frame, k) for k in keys]
    parsed_rating = pd.to_numeric(
        frame.Rating.mask(frame.Rating.str.strip().eq("")), errors="coerce"
    )
    finite = parsed_rating.notna() & np.isfinite(parsed_rating)
    rating = parsed_rating.where(finite)
    ratings = {
        "min": float(rating.min()) if finite.any() else None,
        "max": float(rating.max()) if finite.any() else None,
        "mean": float(rating.mean()) if finite.any() else None,
        "median": float(rating.median()) if finite.any() else None,
        "finite_numeric_rows": int(finite.sum()),
        "missing_rows": int(frame.Rating.str.strip().eq("").sum()),
        "nonnumeric_nonempty_rows": int(
            (frame.Rating.str.strip().ne("") & parsed_rating.isna()).sum()
        ),
        "nonfinite_rows": int((parsed_rating.notna() & ~np.isfinite(parsed_rating)).sum()),
        "zero_rows": int(rating.eq(0).sum()),
        "raw_rating_preserved_in": "data/foody/audit/row_quality.parquet",
        "normalization_applied": False,
        "observed_range_is_not_official_scale_definition": True,
    }
    labels, crosstab, bands = label_audit(frame, rating)
    parsed_dates, date_missing, date_invalid, dates = timestamp_audit(frame["Review Time"])
    nonmissing_rest = frame.RestaurantID.str.strip().ne("")
    counts = (
        frame.loc[nonmissing_rest]
        .groupby("RestaurantID", sort=True)
        .size()
        .rename("review_count")
        .reset_index()
    )
    counts = counts.sort_values(
        ["review_count", "RestaurantID"], ascending=[False, True], kind="stable"
    )
    restaurants = {
        "unique_restaurants": len(counts),
        "missing_restaurant_id_rows": int((~nonmissing_rest).sum()),
        "median_reviews_per_restaurant": float(counts.review_count.median())
        if len(counts)
        else None,
        "min_reviews_per_restaurant": int(counts.review_count.min()) if len(counts) else None,
        "max_reviews_per_restaurant": int(counts.review_count.max()) if len(counts) else None,
        "restaurants_with_at_most_2_reviews": int(counts.review_count.le(2).sum()),
        "restaurants_with_fewer_than_5_reviews": int(counts.review_count.lt(5).sum()),
        "top_10_restaurants": counts.head(10).to_dict("records"),
        "restaurant_identity_provenance": "RestaurantID as supplied; no names/platform URL mapping supplied.",
    }
    summary = {
        "as_of_date": as_of_date,
        "source_path": str(source),
        "source_size_bytes": len(raw),
        "source_sha256": sha,
        "schema": schema,
        "encoding": enc,
        "text": text,
        "labels": labels,
        "ratings": ratings,
        "dates": dates,
        "restaurants": restaurants,
        "duplicates": duplicates,
        "training_readiness": "NOT_READY_MULTIPLE_ISSUES"
        if encoding_blocker
        else "NOT_READY_LABEL_PROVENANCE",
        "demo_rag_readiness": "NOT_READY_FOR_TEXT_RAG"
        if encoding_blocker
        else "READY_AFTER_CLEANING",
        "metadata_demo_readiness": "READY_AFTER_CLEANING",
        "decision_reason": "Supervised training is blocked by unknown Classification provenance"
        + (
            " and substantial text-loss/encoding evidence. A legacy inspection view is not a restored Vietnamese corpus."
            if encoding_blocker
            else ". Text can support a later demo after missing/duplicate handling; Classification is not accepted as gold."
        ),
        "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == sha,
    }
    if not summary["source_unchanged"]:
        raise ValueError("Raw source changed during audit")
    for name, value in (
        ("schema_report", schema),
        ("encoding_report", enc),
        ("label_audit", labels),
        ("text_quality_report", text),
        ("duplicate_audit", duplicates),
        ("rating_audit", ratings),
        ("date_audit", dates),
        ("restaurant_audit", restaurants),
        ("audit_summary", summary),
    ):
        write_json(reports / f"{name}.json", value)
    counts.to_csv(reports / "restaurant_statistics.csv", index=False, encoding="utf-8-sig")
    crosstab.to_csv(reports / "rating_classification_crosstab.csv", encoding="utf-8-sig")
    bands.to_csv(reports / "rating_band_classification.csv", encoding="utf-8-sig")
    pd.DataFrame(distribution(rating)).to_csv(
        reports / "rating_distribution.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"year": y, "reviews": n} for y, n in dates["reviews_by_year"].items()]).to_csv(
        reports / "review_counts_by_year.csv", index=False
    )
    row_audit = pd.DataFrame(
        {
            "source_record_number": np.arange(1, len(frame) + 1),
            "raw_rating": frame.Rating,
            "raw_review_time": frame["Review Time"],
            "raw_classification": frame.Classification,
            "restaurant_id_as_supplied": frame.RestaurantID,
            "comment_empty_or_whitespace": ~nonempty,
            "comment_double_question": double,
            "comment_word_internal_question": internal,
            "comment_replacement_character": replacement,
            "comment_mojibake_pattern": mojibake,
            "comment_characters_in_inspection_view": comments.str.len(),
            "timestamp_missing": date_missing,
            "timestamp_invalid_nonempty": date_invalid,
            "parsed_timestamp_naive": parsed_dates,
        }
    )
    if "Unnamed: 0" in frame:
        row_audit["raw_exported_index"] = frame["Unnamed: 0"]
    for n, key in enumerate(keys, start=1):
        row_audit[f"duplicate_key_{n}_all_rows_member"] = frame.duplicated(key, keep=False)
    row_audit.to_parquet(audit_dir / "row_quality.parquet", index=False)
    write_json(
        audit_dir / "source_manifest.json",
        {
            "source_path": str(source.resolve()),
            "sha256": sha,
            "size_bytes": len(raw),
            "inspection_encoding": enc["inspection_encoding"],
            "raw_source_unchanged": True,
            "row_count_preserved": len(row_audit) == len(frame),
        },
    )
    make_report(summary, reports)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/foody/source/review.csv")
    parser.add_argument("--reports", default="reports/foody")
    parser.add_argument("--audit-dir", default="data/foody/audit")
    parser.add_argument(
        "--as-of-date",
        default=datetime.now(UTC).astimezone(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat(),
    )
    args = parser.parse_args()
    result = run(args.source, args.reports, args.audit_dir, args.as_of_date)
    # Print representative loaded comments in UTF-8 consoles or escaped form if
    # the terminal cannot encode Vietnamese; neither affects file assessment.
    for sample in result["encoding"]["representative_comments"]:
        print(json.dumps(sample, ensure_ascii=True))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "source_sha256",
                    "source_unchanged",
                    "training_readiness",
                    "demo_rag_readiness",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
