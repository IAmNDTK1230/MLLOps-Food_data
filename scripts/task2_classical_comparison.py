"""Separate Review-2 classical comparison; never rewrites accepted model artifacts."""

import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.baselines.tfidf_logreg import EXPERIMENTS, build_pipeline
from src.utils.hashing import file_checksum

OUT = Path("reports/task2")
SPLIT = Path("data/food/splits/reviews.parquet")
META = Path("data/food/splits/metadata.json")
LABELS = ["negative", "neutral", "positive"]
SEEDS = (42, 123, 2026)


def metrics(truth, predicted):
    per_class = classification_report(
        truth, predicted, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, labels=LABELS, average="macro"),
        "weighted_f1": f1_score(truth, predicted, labels=LABELS, average="weighted"),
        **{f"{label}_f1": per_class[label]["f1-score"] for label in LABELS},
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8"))
    if file_checksum(SPLIT) != meta["split_file_sha256"]:
        raise ValueError("Frozen split checksum changed")
    frame = pd.read_parquet(SPLIT)
    counts = frame.split.value_counts().to_dict()
    if counts != {"train": 4140, "validation": 842, "test": 887}:
        raise ValueError(f"Frozen split membership changed: {counts}")
    train = frame.loc[frame.split == "train"]
    validation = frame.loc[frame.split == "validation"]
    test = frame.loc[frame.split == "test"]

    rows = []
    for seed in SEEDS:
        model = build_pipeline(EXPERIMENTS[2], class_weight="balanced", seed=seed)
        model.fit(train.text, train.sentiment)
        for split_name, sample in (("validation", validation), ("test", test)):
            result = metrics(sample.sentiment, model.predict(sample.text))
            rows.append({"model": "TF-IDF + Logistic Regression", "seed": seed,
                         "split": split_name, **result})
    write_csv(OUT / "baseline_three_seed.csv", rows)

    # Additional classical comparison: same train-only TF-IDF vocabulary and frozen split.
    # Select C on validation only; evaluate test once after selection.
    svc_trials = []
    for c in (0.5, 1.0, 2.0):
        template = build_pipeline(EXPERIMENTS[2], class_weight="balanced", seed=42)
        svc = Pipeline([
            ("tfidf", template.named_steps["tfidf"]),
            ("classifier", LinearSVC(C=c, class_weight="balanced", random_state=42)),
        ])
        svc.fit(train.text, train.sentiment)
        val = metrics(validation.sentiment, svc.predict(validation.text))
        svc_trials.append((c, val, svc))
    selected_c, selected_val, selected_svc = max(
        svc_trials, key=lambda trial: (trial[1]["macro_f1"], -trial[0])
    )
    svc_test = metrics(test.sentiment, selected_svc.predict(test.text))
    accepted = json.loads(Path("reports/baselines/tfidf_test_metrics.json").read_text(encoding="utf-8"))
    rule = json.loads(Path("reports/baselines/rule_based_test_metrics.json").read_text(encoding="utf-8"))
    comparisons = [
        {"model": "Rule-based", "status": "frozen_baseline", "selected_c": "",
         "accuracy": rule["accuracy"], "macro_f1": rule["macro_f1"],
         "weighted_f1": rule["weighted_f1"]},
        {"model": "TF-IDF + Logistic Regression", "status": "frozen_baseline", "selected_c": 2.0,
         "accuracy": accepted["accuracy"], "macro_f1": accepted["macro_f1"],
         "weighted_f1": accepted["weighted_f1"]},
        {"model": "TF-IDF + LinearSVC", "status": "extra_exploratory_comparison",
         "selected_c": selected_c, "accuracy": svc_test["accuracy"],
         "macro_f1": svc_test["macro_f1"], "weighted_f1": svc_test["weighted_f1"]},
    ]
    write_csv(OUT / "classical_model_comparison.csv", comparisons)
    write_csv(OUT / "linearsvc_validation_selection.csv", [
        {"c": c, **val} for c, val, _ in svc_trials
    ])

    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("split")[["accuracy", "macro_f1", "weighted_f1",
                                           "negative_f1", "neutral_f1", "positive_f1"]].agg(
        ["mean", "std"]
    )
    summary = summary.fillna(0)
    lines = [
        "# Baseline stability on the frozen split", "",
        f"Split SHA256: `{meta['split_file_sha256']}`. Train/validation/test memberships are unchanged. "
        "Each seed trains the accepted TF-IDF + Logistic Regression configuration from scratch "
        "on the same 4,140 train rows; the accepted serialized model is not touched.", "",
        "`lbfgs` may be deterministic here. Identical scores are reported with standard deviation zero. "
        "The repeated test analysis is exploratory because the accepted test results were already viewed; "
        "it is not a new held-out model-selection claim.", "",
        "| Split | Accuracy mean ± sample SD | Macro F1 mean ± sample SD | Weighted F1 mean ± sample SD | Neutral F1 mean ± sample SD |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    def fmt(row, metric):
        return f"{row[(metric, 'mean')]:.6f} ± {row[(metric, 'std')]:.6f}"

    for split_name in ("validation", "test"):
        row = summary.loc[split_name]
        lines.append(f"| {split_name} | {fmt(row, 'accuracy')} | {fmt(row, 'macro_f1')} | "
                     f"{fmt(row, 'weighted_f1')} | {fmt(row, 'neutral_f1')} |")
    lines += ["", "Individual runs and per-class F1 are in `baseline_three_seed.csv`.", "",
              "LinearSVC is a separate classical comparison, not a replacement baseline. "
              f"C={selected_c} was selected on validation macro F1 "
              f"({selected_val['macro_f1']:.6f}); its single test macro F1 is "
              f"{svc_test['macro_f1']:.6f}. See `linearsvc_validation_selection.csv` and "
              "`classical_model_comparison.csv`. No test-based tuning was performed.", ""]
    (OUT / "BASELINE_STABILITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"3-seed test macro F1: {summary.loc['test', ('macro_f1', 'mean')]:.6f} ± "
          f"{summary.loc['test', ('macro_f1', 'std')]:.6f}")
    print(f"LinearSVC selected C={selected_c}; test macro F1={svc_test['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
