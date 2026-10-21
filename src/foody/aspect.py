"""VN-MSASar-only multi-label aspect baseline with a once-only test checkpoint."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from src.foody.schema import digest, normalize, save_json, save_parquet
from src.ml.config import TrackingConfig
from src.ml.mlflow_tracking import configure_tracking

ROOT = Path("reports/aspect_classifier")
MODEL = Path("models/aspect_tfidf.joblib")
SPLIT = Path("data/food/splits/reviews.parquet")
SPLIT_SHA = "860387be19c64220f91cc2b5e764aa3c99aa18daba9c776fde38e05da30a5ff9"


def grouped_targets(frame):
    rows = frame.assign(text=frame.text.map(normalize))
    if rows.groupby("text").split.nunique().gt(1).any():
        raise ValueError("Frozen normalized text crosses memberships")
    return rows.groupby(["split", "text"], sort=True).agg(
        aspects=("aspect", lambda x: sorted(set(x))),
        source_review_ids=("review_id", lambda x: sorted(set(x))),
        source_aspect_rows=("aspect", "size"),
    ).reset_index()


def scores(y, pred, classes):
    result = {
        "exact_match_ratio": float(accuracy_score(y, pred)),
        "average_predicted_labels_per_sample": float(pred.sum(axis=1).mean()),
    }
    for average in ("micro", "macro"):
        p, r, f, _ = precision_recall_fscore_support(
            y, pred, average=average, zero_division=0
        )
        result.update({f"{average}_precision": float(p), f"{average}_recall": float(r),
                       f"{average}_f1": float(f)})
    p, r, f, support = precision_recall_fscore_support(y, pred, zero_division=0)
    per_class = pd.DataFrame(
        {"aspect": classes, "precision": p, "recall": r, "f1": f, "support": support}
    )
    return result, per_class


def train():
    if digest(SPLIT) != SPLIT_SHA:
        raise ValueError("Frozen research split changed")
    ROOT.mkdir(parents=True, exist_ok=True)
    completed = ROOT / "test_metrics.json"
    if completed.exists():
        result = json.loads(completed.read_text(encoding="utf-8"))
        if digest(MODEL) != result["model_sha256"]:
            raise ValueError("Completed aspect model changed; no implicit retraining")
        return result
    selection_path = ROOT / "selection.json"
    mlflow = configure_tracking(TrackingConfig.from_env().model_copy(
        update={"experiment_name": "vn-msasar-multilabel-aspect-v1"}
    ))
    frame = pd.read_parquet(SPLIT)
    if frame.split.value_counts().to_dict() != {"train": 4140, "test": 887, "validation": 842}:
        raise ValueError("Unexpected frozen split memberships")
    groups = grouped_targets(frame)
    classes = sorted(set(frame.loc[frame.split == "train", "aspect"]))
    if set(frame.aspect) != set(classes):
        raise ValueError("Aspect absent from training vocabulary")
    encoder = MultiLabelBinarizer(classes=classes).fit([classes])
    train_rows = groups.loc[groups.split == "train"]
    val = groups.loc[groups.split == "validation"]
    y_train, y_val = encoder.transform(train_rows.aspects), encoder.transform(val.aspects)
    save_parquet(ROOT / "grouped_membership.parquet", groups)
    if not selection_path.exists():
        trials, fitted = [], {}
        # Fixed, small search decided before any test prediction. No test-based threshold.
        for c in (0.5, 2.0):
            model = Pipeline([
                ("tfidf", TfidfVectorizer(
                    preprocessor=normalize, lowercase=False, ngram_range=(1, 2),
                    min_df=2, max_features=20000, sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                )),
                ("classifier", OneVsRestClassifier(LogisticRegression(
                    C=c, class_weight="balanced", max_iter=1500, random_state=42,
                    solver="liblinear",
                ), n_jobs=1)),
            ])
            model.fit(train_rows.text, y_train)
            probabilities = model.predict_proba(val.text)
            fitted[c] = model
            for threshold in (0.3, 0.5, 0.7):
                metrics, _ = scores(y_val, probabilities >= threshold, classes)
                with mlflow.start_run(run_name=f"aspect_C{c}_threshold{threshold}") as run:
                    mlflow.log_params({
                        "C": c, "threshold": threshold, "split_sha256": SPLIT_SHA,
                        "input": "text_only", "labels": ",".join(classes),
                        "target": "union_of_annotated_aspects_per_normalized_text",
                        "selection": "validation_micro_f1_then_macro_f1_then_lower_C",
                        "train_groups": len(train_rows), "validation_groups": len(val),
                        "ngram_range": "1,2", "min_df": 2, "max_features": 20000,
                        "random_seed": 42, "class_weight": "balanced",
                    })
                    mlflow.log_metrics({f"validation_{k}": v for k, v in metrics.items()})
                    trials.append({"C": c, "threshold": threshold, "run_id": run.info.run_id,
                                   **metrics})
        pd.DataFrame(trials).to_csv(ROOT / "validation_comparison.csv", index=False)
        best = max(trials, key=lambda r: (r["micro_f1"], r["macro_f1"], -r["C"]))
        bundle = {"pipeline": fitted[best["C"]], "classes": classes,
                  "threshold": best["threshold"], "split_sha256": SPLIT_SHA,
                  "name": "vn-msasar-tfidf-ovr-aspect-v1"}
        joblib.dump(bundle, MODEL, compress=3)
        save_json(selection_path, {**best, "model_sha256": digest(MODEL),
                                  "classes": classes, "split_sha256": SPLIT_SHA})
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if digest(MODEL) != selection["model_sha256"]:
        raise ValueError("Frozen selected aspect model differs")
    bundle = joblib.load(MODEL)
    test = groups.loc[groups.split == "test"].copy()
    checkpoint = ROOT / "test_predictions.parquet"
    if not checkpoint.exists():
        # An interrupted evaluation cannot silently trigger a second test inference.
        with (ROOT / "test_evaluation_started.json").open("x", encoding="utf-8") as stream:
            json.dump({"model_sha256": digest(MODEL), "split_sha256": SPLIT_SHA}, stream)
        probabilities = bundle["pipeline"].predict_proba(test.text)
        for i, name in enumerate(classes):
            test[f"probability_{name}"] = probabilities[:, i]
            test[f"predicted_{name}"] = probabilities[:, i] >= bundle["threshold"]
        save_parquet(checkpoint, test)
    test = pd.read_parquet(checkpoint)
    y_test = encoder.transform(test.aspects)
    pred = test[[f"predicted_{name}" for name in classes]].to_numpy()
    metrics, per_class = scores(y_test, pred, classes)
    per_class.to_csv(ROOT / "test_per_aspect.csv", index=False)
    result = {
        **metrics, "model_sha256": digest(MODEL), "split_sha256": SPLIT_SHA,
        "classes": classes, "selected_C": selection["C"],
        "selected_threshold": selection["threshold"],
        "group_counts": groups.split.value_counts().to_dict(),
        "validation": {k: selection[k] for k in metrics},
        "run_id": selection["run_id"],
    }
    with mlflow.start_run(run_id=selection["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_param("model_sha256", digest(MODEL))
        mlflow.log_artifact(str(ROOT / "test_per_aspect.csv"))
        mlflow.log_artifact(str(selection_path))
        mlflow.set_tag("test_evaluation", "once_after_selection; checkpoint_reused_for_reporting")
    text = (
        "# Multi-label aspect classifier acceptance\n\n"
        "VN-MSASar v3 only. Frozen row memberships 4140/842/887; no resplit. "
        "Targets union all annotated aspects for each NFC/whitespace-normalized text "
        "within its existing membership. Group counts differ from aspect-row counts. "
        "No text crosses splits. Input is text only. Train-only vocabulary and estimator; "
        "validation-only C/threshold selection by micro F1, then macro F1, then lower C. "
        "No train+validation refit. Test prediction checkpoint created once.\n\n"
        "Unannotated aspects are treated as negatives for this benchmark; annotations may "
        "be incomplete. Rare room/amenities labels have zero test support: their zero F1 "
        "is included in all-nine-label macro F1, not evidence of measured competence. "
        "This experiment does not measure Foody accuracy.\n\n"
        f"```json\n{json.dumps(result, indent=2)}\n```\n\n"
        + per_class.to_markdown(index=False) + "\n"
    )
    (ROOT / "ASPECT_ACCEPTANCE_REPORT.md").write_text(text, encoding="utf-8")
    save_json(completed, result)
    return result
