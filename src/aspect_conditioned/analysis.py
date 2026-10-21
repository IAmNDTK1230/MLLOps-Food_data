import json

import pandas as pd

from src.ml.features import normalize_text
from src.proposed_v1.runner import score_predictions
from src.proposed_v2.analysis import csv_records, diagnostic_rows
from src.storage.manifest import atomic_json


def conflict_mask(frame):
    normalized = frame.text.map(normalize_text)
    counts = frame.groupby(normalized).sentiment.nunique()
    return normalized.isin(counts[counts.gt(1)].index)


def recover_classical(test, failures, recorded):
    """Read saved union-of-errors; absent rows were correct for BOTH old models.

    This is reconstruction of historical outputs, never model inference. Verify
    full metrics against the frozen test report before using subset scores.
    """
    keys = ["review_id", "text", "aspect", "true_label"]
    metadata = test.rename(columns={"sentiment": "true_label"}).copy()
    metadata["sample_index"] = test.index
    columns = keys + ["tfidf_prediction"]
    saved = failures[columns].drop_duplicates()
    if saved.duplicated(keys).any():
        raise ValueError("Ambiguous historical TF-IDF prediction identity")
    historical = metadata.merge(saved, on=keys, how="left", validate="many_to_one")
    historical["prediction"] = historical.tfidf_prediction.fillna(historical.true_label)
    measured, _ = score_predictions(historical.true_label, historical.prediction)
    for key, value in measured.items():
        if key in recorded and abs(recorded[key] - value) > 1e-12:
            raise ValueError("Reconstructed historical TF-IDF scores disagree")
    return historical.set_index("sample_index").loc[test.index].prediction.to_list()


def subset_metrics(truth, prediction):
    if len(truth) == 0:
        return {"sample_count": 0, "accuracy": None, "macro_f1": None, "neutral_f1": None}
    metrics, _ = score_predictions(truth, prediction)
    return {
        "sample_count": len(truth),
        **metrics,
        "macro_f1_note": "fixed three-label macro; absent labels contribute zero",
    }


def analyze(config, frame, root, protocol):
    qwen = pd.read_parquet(root / "qwen_predictions.parquet").set_index("sample_index")
    tfidf = pd.read_parquet(root / "tfidf_aspect_predictions.parquet").set_index("sample_index")
    test = frame.loc[frame.split.eq("test")]
    qwen = qwen.loc[test.index].rename_axis("sample_index")
    tfidf = tfidf.loc[test.index].rename_axis("sample_index")
    # Historical TEST artifacts are first parsed here, after all new inference.
    old = pd.read_parquet(config.reports_dir / "proposed_v1/qwen_predictions.parquet")
    index = "sample_index" if "sample_index" in old else "index"
    old = old.set_index(index).loc[test.index]
    if "text" in old and old.text.tolist() != test.text.tolist():
        raise ValueError("Historical V1 text mismatch")
    old_metrics = json.loads(
        (config.reports_dir / "baselines/tfidf_test_metrics.json").read_text(encoding="utf-8")
    )
    old_tfidf = recover_classical(
        test, pd.read_csv(config.reports_dir / "baselines/failure_cases.csv"), old_metrics
    )
    mask = conflict_mask(frame).loc[test.index]
    predictions = {
        "TF-IDF text-only": pd.Series(old_tfidf, index=test.index),
        "Qwen V1 text-only": old.prediction,
        "TF-IDF text+aspect": tfidf.prediction,
        "Qwen text+aspect": qwen.prediction,
    }
    rows = []
    for name, prediction in predictions.items():
        for subset, flag in (
            ("all_test", pd.Series(True, index=test.index)),
            ("conflicting_text", mask),
            ("non_conflicting_text", ~mask),
        ):
            rows.append(
                {
                    "model": name,
                    "input_protocol": "text+aspect" if "+aspect" in name else "text-only",
                    "subset": subset,
                    **subset_metrics(test.loc[flag].sentiment, prediction.loc[flag]),
                }
            )
    pd.DataFrame(rows).to_csv(root / "conflicting_text_comparison.csv", index=False)
    aspects = []
    for name, prediction in predictions.items():
        for aspect, group in test.groupby("aspect"):
            if aspect in {"room", "amenities"} and len(group) < 20:
                continue
            aspects.append(
                {
                    "model": name,
                    "aspect": aspect,
                    **subset_metrics(group.sentiment, prediction.loc[group.index]),
                }
            )
    pd.DataFrame(aspects).to_csv(root / "metrics_by_aspect.csv", index=False)
    flagged, cohorts, _, confusions = diagnostic_rows(qwen.reset_index())
    flagged["conflicting_text"] = flagged.sample_index.map(mask)
    flagged["aspect_interpretation_status"] = "not_manually_adjudicated"
    flagged["rare_aspect_support_under_30"] = flagged.aspect.map(test.aspect.value_counts()).lt(30)
    flagged.loc[flagged.error].to_csv(root / "failure_cases.csv", index=False)
    pair_rows = qwen.loc[mask].copy()
    pair_rows["tfidf_aspect_prediction"] = tfidf.prediction
    pair_rows["tfidf_text_prediction"] = pd.Series(old_tfidf, index=test.index)
    pair_rows["qwen_v1_prediction"] = old.prediction
    pair_rows.to_csv(root / "conflicting_text_cases.csv", index=True)
    # Validation failure cohorts are separate from post-hoc TEST diagnostics.
    validation_diagnostics = []
    candidates = []
    for name in protocol["settings"]["candidate_names"]:
        candidate = json.loads(
            (root / f"{name}_validation_metrics.json").read_text(encoding="utf-8")
        )
        candidates.append(candidate)
        val = pd.read_parquet(root / f"{name}_validation_predictions.parquet")
        failures, groups, _, changes = diagnostic_rows(val)
        failures.loc[failures.error].to_csv(
            root / f"{name}_validation_failure_cases.csv", index=False
        )
        validation_diagnostics.append({"prompt": name, "cohorts": groups, "confusions": changes})
    atomic_json(root / "validation_failure_analysis.json", validation_diagnostics)
    qmetrics = json.loads((root / "qwen_test_metrics.json").read_text(encoding="utf-8"))
    tmetrics = json.loads((root / "tfidf_aspect_test_metrics.json").read_text(encoding="utf-8"))
    subset_table = pd.DataFrame(rows)
    conflicting = subset_table.loc[subset_table.subset.eq("conflicting_text")].set_index("model")
    fair_deltas = {k: qmetrics[k] - tmetrics[k] for k in ("accuracy", "macro_f1", "neutral_f1")}
    conflict_deltas = {
        k: float(conflicting.loc["Qwen text+aspect", k] - conflicting.loc["TF-IDF text+aspect", k])
        for k in ("accuracy", "macro_f1", "neutral_f1")
    }
    historical_deltas = {
        "qwen_aspect_minus_v1_conflicting": {
            k: float(
                conflicting.loc["Qwen text+aspect", k] - conflicting.loc["Qwen V1 text-only", k]
            )
            for k in ("accuracy", "macro_f1")
        },
        "tfidf_aspect_minus_text_conflicting": {
            k: float(
                conflicting.loc["TF-IDF text+aspect", k] - conflicting.loc["TF-IDF text-only", k]
            )
            for k in ("accuracy", "macro_f1")
        },
    }
    analysis = {
        "fair_qwen_minus_tfidf_aspect": fair_deltas,
        "fair_conflicting_subset_deltas": conflict_deltas,
        "different_input_descriptive_deltas": historical_deltas,
        "global_conflicting_texts": int(
            frame.loc[conflict_mask(frame)].text.map(normalize_text).nunique()
        ),
        "test_conflicting_texts": int(test.loc[mask].text.map(normalize_text).nunique()),
        "test_conflicting_rows": int(mask.sum()),
        "cohorts": cohorts,
        "confusions": confusions,
        "neutral_predictions": int(qwen.prediction.eq("neutral").sum()),
        "correct_neutral": int(
            (qwen.prediction.eq("neutral") & test.sentiment.eq("neutral")).sum()
        ),
        "beats_aspect_tfidf_macro_f1": fair_deltas["macro_f1"] > 0,
        "limitations": [
            protocol["test_hygiene"],
            "small overlapping aspect/text clusters; no causal claim",
            "mixed/implicit/negation cohorts are lexical proxies; wrong-aspect errors require manual adjudication",
            "classical aspect token is additive; limited aspect-by-text interactions",
            "historical TF-IDF predictions reconstructed from union-of-errors CSV, full metrics verified; no old inference",
        ],
        "next_experiment": "Pre-register TRAIN/VALIDATION-only aspect-by-text feature interaction control, then an aspect-focused Qwen prompt ablation with matched examples; untouched licensed external confirmation before superiority claims.",
    }
    atomic_json(root / "failure_analysis.json", analysis)
    historical = csv_records(config.reports_dir / "proposed_v2/comparison.csv")
    for row in historical:
        row["input_protocol"] = "text-only"
    table = historical + [
        {"model": "TF-IDF text+aspect", **tmetrics},
        {"model": "Qwen text+aspect", **qmetrics},
    ]
    pd.DataFrame(table).to_csv(root / "comparison.csv", index=False)
    session = json.loads((root / "resource_session.json").read_text(encoding="utf-8"))
    resource = {
        "session": session,
        "test": {
            k: v
            for k, v in qmetrics.items()
            if k
            in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "mean_tokens_per_sample",
                "p95_input_tokens",
                "maximum_input_tokens",
                "mean_latency_ms",
                "p95_latency_ms",
                "evaluation_duration_seconds",
                "samples_per_second",
            )
        },
        "validation_input_tokens": sum(r["input_tokens"] for r in candidates),
        "validation_output_tokens": sum(r["output_tokens"] for r in candidates),
        "warmup_input_tokens_recorded": sum(
            r["warmup_usage"]["input_tokens"] or 0 for r in candidates
        ),
        "warmup_output_tokens_recorded": sum(
            r["warmup_usage"]["output_tokens"] or 0 for r in candidates
        ),
        "warmup_measurements_missing": sum(
            r["warmup_usage"]["input_tokens"] is None for r in candidates
        ),
    }
    if (root / "pretest_reporting_recovery.json").exists():
        resource["pretest_reporting_recovery"] = json.loads(
            (root / "pretest_reporting_recovery.json").read_text(encoding="utf-8")
        )
    atomic_json(root / "resource_report.json", resource)
    atomic_json(
        config.reports_dir / "task2_aspect_v2_evidence.json",
        {
            "test_hygiene": protocol["test_hygiene"],
            "slide_12": {
                "v2_a_failure": "text-only neutral overprediction persisted; macro F1 declined to 0.614487",
                "structural_ambiguity": {
                    "global_conflicting_texts": 81,
                    "test_conflicting_texts": 11,
                },
            },
            "slide_13": {
                "architecture": "frozen rows -> text+aspect -> Qwen strict JSON; TRAIN examples; validation selection; one exploratory test",
                "classical_control": "__aspect_X__ text -> TRAIN TF-IDF -> balanced Logistic Regression",
                "selected_prompt": qmetrics["selected_prompt"],
            },
            "slide_14": {
                "ablation": table,
                "fair_comparison": "text+aspect Qwen vs text+aspect TF-IDF",
                "different_input_warning": "text-only historical rows are descriptive, not apples-to-apples superiority",
                "analysis": analysis,
            },
        },
    )
