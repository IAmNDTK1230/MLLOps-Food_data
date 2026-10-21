import json

import matplotlib
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.baselines.metrics import LABELS
from src.storage.manifest import atomic_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def csv_records(path):
    frame = pd.read_csv(path)
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def abstention_plot(truth, predictions, path):
    labels = [*LABELS, "__invalid__"]
    matrix = confusion_matrix(truth, predictions.fillna("__invalid__"), labels=labels)
    fig, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
    )
    for i in range(4):
        for j in range(4):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diagnostic_rows(predictions):
    """Post-hoc only. Lexical flags are proxies, sarcasm is source gold."""
    rows = predictions.copy()
    rows["error"] = rows.true_label != rows.prediction
    flags = {
        "short_20orless_chars": rows.text.str.len().le(20),
        "negation_proxy": rows.text.str.contains(r"(?i)\b(?:không|chưa|chẳng|chả)\b", regex=True),
        "mixed_proxy": rows.text.str.contains(r"(?i)\b(?:nhưng|tuy|song|vừa)\b", regex=True),
        "sarcasm_gold": rows.sarcasm.eq(1),
        "implicit_word_absence_proxy": ~rows.text.str.contains(
            r"(?i)\b(?:ngon|tốt|tệ|dở|thích|chán|bẩn|sạch)\b", regex=True
        ),
    }
    cohorts = []
    for name, flag in flags.items():
        rows[name] = flag
        count, errors = int(flag.sum()), int((flag & rows.error).sum())
        cohorts.append(
            {
                "cohort": name,
                "samples": count,
                "errors": errors,
                "error_rate": errors / count if count else None,
            }
        )
    aspects = [
        {
            "aspect": name,
            "samples": len(group),
            "errors": int(group.error.sum()),
            "error_rate": float(group.error.mean()),
        }
        for name, group in rows.groupby("aspect")
    ]
    confusions = {
        f"{a}_to_{b}": int(((rows.true_label == a) & (rows.prediction == b)).sum())
        for a, b in (
            ("positive", "neutral"),
            ("negative", "neutral"),
            ("positive", "negative"),
            ("negative", "positive"),
        )
    }
    return rows, cohorts, aspects, confusions


def validation_analysis(config, frame, candidates, root):
    from src.proposed_v2.runner import historical_validation, join_predictions

    validation = frame.loc[frame.split == "validation"]
    records, metrics, report = historical_validation(config, validation)
    v1_metrics = json.loads(
        (config.reports_dir / "proposed_v1/v1_c_validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        {
            "prompt": "v1_c",
            **metrics,
            **{
                key: v1_metrics[key]
                for key in (
                    "json_validity_rate",
                    "mean_latency_ms",
                    "p95_latency_ms",
                    "input_tokens",
                    "output_tokens",
                )
            },
            "neutral_prediction_count": int(records.prediction.eq("neutral").sum()),
        }
    ]
    comparison = []
    v1_join = join_predictions(validation, records.rename_axis("sample_index").reset_index())
    _, cohorts, aspects, confusions = diagnostic_rows(v1_join)
    comparison.append(
        {"prompt": "v1_c", "split": "validation", "cohorts": cohorts, "confusions": confusions}
    )
    for candidate in candidates:
        name = candidate["prompt"]
        rows.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"warmup_usage", "resource_snapshot"}
            }
        )
        predictions = pd.read_parquet(root / f"{name}_validation_predictions.parquet")
        joined = join_predictions(validation, predictions)
        flagged, cohorts, aspects, confusions = diagnostic_rows(joined)
        flagged.loc[flagged.error].to_csv(
            root / f"{name}_validation_failure_cases.csv", index=False
        )
        pd.DataFrame(aspects).to_csv(root / f"{name}_validation_error_by_aspect.csv", index=False)
        comparison.append(
            {"prompt": name, "split": "validation", "cohorts": cohorts, "confusions": confusions}
        )
    pd.DataFrame(rows).to_csv(root / "v1_vs_v2_validation.csv", index=False)
    atomic_json(root / "validation_failure_analysis.json", comparison)
    pd.DataFrame([{"label": label, **report[label]} for label in LABELS]).to_csv(
        root / "v1_c_validation_classification_report.csv", index=False
    )
    axis = (
        pd.DataFrame(rows)
        .set_index("prompt")[["macro_f1", "neutral_precision", "neutral_recall", "neutral_f1"]]
        .plot.bar(
            figsize=(10, 6),
            ylim=(0, 1),
            rot=0,
            title="Validation only: V1-C vs frozen V2 candidates",
        )
    )
    axis.figure.tight_layout()
    axis.figure.savefig(root / "validation_comparison.png", dpi=160)
    plt.close(axis.figure)


def protocol_documents(root):
    aspect = """# Proposed V2-B / V3: Qwen(text + aspect) — protocol proposal only

This is a DISTINCT input/task, not directly equivalent to text-only baselines.
The existing dataset has 81 globally / 11 test conflicting normalized texts.
Keep original labels, review/text groups, seed and frozen membership. Supply ONLY
text plus the original aspect label; never scores, current labels or sarcasm.
Develop train-only aspect-matched contrasts, tune prompts on validation macro F1,
then freeze model/runtime/prompt/settings before an explicitly authorized evaluation.
Add an aspect-aware classical comparator if a fair same-input comparison is wanted.
Report macro/weighted/per-class scores and neutral calibration, contradictions,
latency/tokens and exact input differences. Do not call gains proof of a better
text-only model. Existing test outcomes have been inspected, so this reused test
is exploratory; untouched external confirmation is needed. No V2-B/V3 implementation
or inference is part of the present V2-A experiment.
"""
    external = """# Untouched external final evaluation recommendation — no download/evaluation

Candidate: VLSP2018 Restaurant, via the official VLSP resource process.
Official resources: https://vlsp.org.vn/resources
2018 dataset access: https://vlsp.org.vn/resources-vlsp2018
Task definition: https://vlsp.org.vn/node/82

The official resource pages require a signed data-user agreement/access request.
Verify applicable current permission, redistribution restrictions, source version,
license/agreement and attribution BEFORE obtaining files. A mirror's license tag
does not establish permission. No files have been downloaded or external evaluation
performed here, and no access email has been sent.

VLSP2018 is aspect-based: its original review/aspect polarity tuples do not imply
one global text label. Pre-register separate text-only and text+aspect contracts.
For text-only final confirmation, prefer independently annotated global review labels;
otherwise use an a-priori justified single-polarity subset and disclose selection,
coverage and domain shift. Never silently collapse contradictory aspects. For the
aspect-conditioned task preserve official entity/attribute labels and document any
taxonomy mapping without inspecting final gold labels.

Audit normalized/exact/near-duplicate overlap with all VN-MSASar partitions BEFORE
scoring: Foody sources can overlap. Have a custodian keep final labels untouched,
fix the exclusion/mapping protocol in advance, freeze all models/prompts, run ONE
external confirmation and publish all outcomes, including regressions. Do not tune
on this set or merge it into primary training. If official final labels are unavailable
or independence/permission cannot be established, procure another newly licensed,
untouched Vietnamese restaurant corpus rather than claim confirmation on reused data.
"""
    (root / "aspect_conditioned_protocol.md").write_text(aspect, encoding="utf-8")
    (root / "external_evaluation_recommendation.md").write_text(external, encoding="utf-8")


def final_analysis(config, frame, predictions, result, candidates, selected, protocol, root):
    # Historical TEST material is accessed here only AFTER frozen V2 test inference.
    v1 = json.loads(
        (config.reports_dir / "proposed_v1/qwen_test_metrics.json").read_text(encoding="utf-8")
    )
    old_comparison = csv_records(config.reports_dir / "proposed_v1/model_comparison.csv")
    v2 = {
        "model": "Qwen V2-A",
        **{
            key: result[key]
            for key in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "neutral_precision",
                "neutral_recall",
                "neutral_f1",
                "mean_latency_ms",
                "p95_latency_ms",
                "model_size_mb",
                "json_validity_rate",
            )
        },
    }
    comparison = old_comparison + [v2]
    pd.DataFrame(comparison).to_csv(root / "comparison.csv", index=False)
    tfidf = next(row for row in old_comparison if row["model"] == "TF-IDF + Logistic Regression")
    deltas = {
        "macro_f1_vs_v1": result["macro_f1"] - v1["macro_f1"],
        "macro_f1_vs_tfidf": result["macro_f1"] - tfidf["macro_f1"],
        "neutral_f1_vs_v1": result["neutral_f1"] - v1["neutral_f1"],
        "neutral_f1_vs_tfidf": result["neutral_f1"] - tfidf["neutral_f1"],
        "neutral_precision_vs_v1": result["neutral_precision"] - v1["neutral_precision"],
        "neutral_recall_vs_v1": result["neutral_recall"] - v1["neutral_recall"],
    }
    rows, cohorts, aspects, confusions = diagnostic_rows(predictions)
    conflicting = frame.groupby("text").sentiment.nunique().gt(1)
    rows["conflicting_gold_text"] = rows.text.isin(conflicting[conflicting].index)
    rows.loc[rows.error].to_csv(root / "failure_cases.csv", index=False)
    pd.DataFrame(aspects).to_csv(root / "error_by_aspect.csv", index=False)
    v1_predictions = pd.read_parquet(config.reports_dir / "proposed_v1/qwen_predictions.parquet")
    v1_predictions = (
        v1_predictions.set_index("sample_index")
        .loc[rows.sample_index]
        .rename_axis("sample_index")
        .reset_index()
    )
    if (
        v1_predictions.text.tolist() != rows.text.tolist()
        or v1_predictions.true_label.tolist() != rows.true_label.tolist()
    ):
        raise ValueError("V1/V2 TEST membership or labels differ")
    _, old_cohorts, _, old_confusions = diagnostic_rows(v1_predictions)
    confusion_changes = [
        {
            "confusion": name,
            "v1": old_confusions[name],
            "v2": value,
            "delta": value - old_confusions[name],
        }
        for name, value in confusions.items()
    ]
    pd.DataFrame(confusion_changes).to_csv(root / "confusion_changes.csv", index=False)
    patterns = []
    for before, after in zip(old_cohorts, cohorts, strict=True):
        patterns.append(
            {
                "cohort": after["cohort"],
                "samples": after["samples"],
                "v1_errors": before["errors"],
                "v2_errors": after["errors"],
                "v1_error_rate": before["error_rate"],
                "v2_error_rate": after["error_rate"],
            }
        )
    signals = {
        key: ("POSITIVE IMPROVEMENT SIGNAL" if value > 0 else "NO POSITIVE IMPROVEMENT SIGNAL")
        for key, value in deltas.items()
        if key.startswith(("macro_f1", "neutral_f1"))
    }
    summary = {
        "cohorts": patterns,
        "confusion_changes": confusion_changes,
        "deltas": deltas,
        "signals": signals,
        "improves_v1_primary_metric": deltas["macro_f1_vs_v1"] > 0,
        "surpasses_tfidf_primary_metric": deltas["macro_f1_vs_tfidf"] > 0,
        "v2_neutral_predictions": int(rows.prediction.eq("neutral").sum()),
        "v1_neutral_predictions": int(v1_predictions.prediction.eq("neutral").sum()),
        "conflicting_texts_global": int(conflicting.sum()),
        "conflicting_texts_test": int(rows.groupby("text").true_label.nunique().gt(1).sum()),
        "conflicting_text_row_errors": int((rows.error & rows.conflicting_gold_text).sum()),
        "phenomenon_warning": "Sarcasm is source gold; short/negation/mixed/implicit markers are lexical proxies, cohorts overlap",
        "test_hygiene": protocol["test_hygiene"],
        "external_confirmation_required": True,
    }
    atomic_json(root / "failure_analysis.json", summary)
    session = json.loads((root / "resource_session.json").read_text(encoding="utf-8"))
    resources = {
        "test": {
            key: result[key]
            for key in (
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
        "validation": [
            {
                key: candidate[key]
                for key in (
                    "prompt",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "mean_latency_ms",
                    "p95_latency_ms",
                    "evaluation_duration_seconds",
                )
            }
            for candidate in candidates
        ],
        "total_evaluation_input_tokens": result["input_tokens"]
        + sum(candidate["input_tokens"] for candidate in candidates),
        "total_evaluation_output_tokens": result["output_tokens"]
        + sum(candidate["output_tokens"] for candidate in candidates),
        "warmup_input_tokens": sum(
            candidate["warmup_usage"]["input_tokens"] for candidate in candidates
        ),
        "warmup_output_tokens": sum(
            candidate["warmup_usage"]["output_tokens"] for candidate in candidates
        ),
        "session": session,
    }
    atomic_json(root / "resource_report.json", resources)
    protocol_documents(root)
    evidence = {
        "slide_11": {
            "v1_vs_v2_deltas": deltas,
            "signals": signals,
            "validation": csv_records(root / "v1_vs_v2_validation.csv"),
        },
        "slide_12": {
            "historical_v1_test_confusions": old_confusions,
            "analysis_role": "historical motivation only; no test sentences used in example selection",
        },
        "slide_13": {
            "architecture": "text -> calibrated frozen prompt + TRAIN contrasts -> unchanged Qwen/llama.cpp -> strict JSON -> sentiment",
            "selected_prompt": selected,
            "example_manifest": "proposed_v2/train_example_manifest.json",
            "aspect_input": False,
            "fine_tuned": False,
        },
        "slide_14": {
            "comparison": comparison,
            "primary_metric": "macro_f1",
            "test_hygiene": protocol["test_hygiene"],
        },
        "failure_analysis": summary,
        "resources": resources,
        "next_experiment": "proposed_v2/aspect_conditioned_protocol.md",
        "external_evaluation": "proposed_v2/external_evaluation_recommendation.md",
    }
    atomic_json(config.reports_dir / "task2_proposed_v2_evidence.json", evidence)
    axis = (
        pd.DataFrame(comparison)
        .set_index("model")[["macro_f1", "neutral_f1", "accuracy"]]
        .plot.bar(
            figsize=(12, 6),
            ylim=(0, 1),
            rot=15,
            title="Exploratory reused-test comparison — external confirmation required",
        )
    )
    axis.figure.tight_layout()
    axis.figure.savefig(root / "comparison.png", dpi=160)
    plt.close(axis.figure)
