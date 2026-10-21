"""Refresh slide/failure evidence from saved predictions, never a model call."""

import json
from pathlib import Path

import pandas as pd

from src.food.split import load_splits
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def finalize_evidence(config):
    root = config.reports_dir / "baselines"
    state_path = root / "evaluation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "completed":
        raise ValueError("Cannot finalize evidence before completed evaluation")
    for filename, checksum in state["artifacts"].items():
        path = Path(filename)
        if not path.exists() or file_checksum(path) != checksum:
            raise ValueError("Saved evaluation artifacts changed; cannot refresh evidence")
    frame, split, preparation = load_splits(config)
    if (
        state["split_hash"] != split["split_file_sha256"]
        or state["dataset_hash"] != preparation["processed_sha256"]
    ):
        raise ValueError("Dataset/split changed after evaluation")
    weights = pd.read_csv(root / "class_weight_comparison.csv", keep_default_na=False)
    failures = pd.read_csv(root / "failure_cases.csv", keep_default_na=False)
    aspects = pd.read_csv(root / "error_by_aspect.csv", keep_default_na=False)
    test = frame.loc[frame.split == "test"]
    summaries = {}
    for name, prediction in (("rule", "rule_prediction"), ("tfidf", "tfidf_prediction")):
        errors = failures.loc[failures.true_label != failures[prediction]]
        summaries[name] = {
            "errors": len(errors),
            "positive_false_positives": int(
                ((errors[prediction] == "positive") & (errors.true_label != "positive")).sum()
            ),
            "positive_false_negatives": int(
                ((errors.true_label == "positive") & (errors[prediction] != "positive")).sum()
            ),
            "neutral_as_positive": int(
                ((errors.true_label == "neutral") & (errors[prediction] == "positive")).sum()
            ),
            "neutral_as_negative": int(
                ((errors.true_label == "neutral") & (errors[prediction] == "negative")).sum()
            ),
            "confusions": errors.groupby(["true_label", prediction])
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={prediction: "prediction"})
            .to_dict("records"),
            "examples": errors[
                ["review_id", "text", "aspect", "true_label", "rule_prediction", "tfidf_prediction"]
            ]
            .head(5)
            .to_dict("records"),
        }
    summary = {
        "evaluation_split": "test",
        "dataset_hash": state["dataset_hash"],
        "split_hash": state["split_hash"],
        "models": summaries,
        "aspect_errors": aspects.to_dict("records"),
        "globally_conflicting_normalized_texts": int(
            frame.groupby("text").sentiment.nunique().gt(1).sum()
        ),
        "test_conflicting_normalized_texts": int(
            test.groupby("text").sentiment.nunique().gt(1).sum()
        ),
        "aspects_without_test_support": sorted(set(frame.aspect) - set(test.aspect)),
        "interpretation": "Different aspect labels for identical text create ambiguity for the required text-only task; no labels/predictions changed.",
    }
    summary_path = root / "failure_analysis_summary.json"
    atomic_json(summary_path, summary)
    evidence_path = config.reports_dir / "task2_baseline_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    # Preserve literal 'None', which pandas' default CSV parser treats as NaN.
    evidence["slide_6_update"]["class_weight_comparison"] = weights.to_dict("records")
    evidence["slide_9"]["failure_analysis"] = summary
    evidence["target_protocol"] = {
        "features": ["text"],
        "label": "normalized sentiment",
        "aspect_feature_used": False,
        "overrides_preparatory_aspect_conditioned_plan": True,
    }
    atomic_json(evidence_path, evidence)

    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    json.loads(evidence_path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    state["artifacts"][str(evidence_path)] = file_checksum(evidence_path)
    state["artifacts"][str(summary_path)] = file_checksum(summary_path)
    state["reporting_refresh"] = (
        "Read saved predictions/CSV only; no fitting or additional test prediction"
    )
    atomic_json(state_path, state)
    return summary
