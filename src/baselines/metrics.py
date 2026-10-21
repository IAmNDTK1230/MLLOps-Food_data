from time import perf_counter_ns

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ("negative", "neutral", "positive")


def measure_predictions(model, texts, warmup_texts):
    # Warm up on validation/train inputs only. Each evaluated sample is predicted once.
    for text in list(warmup_texts)[:3]:
        model.predict([text])
    predictions, latency = [], []
    for text in texts:
        start = perf_counter_ns()
        prediction = model.predict([text])[0]
        latency.append((perf_counter_ns() - start) / 1_000_000)
        predictions.append(str(prediction))
    return predictions, {
        "mean_latency_ms": float(np.mean(latency)),
        "p95_latency_ms": float(np.percentile(latency, 95)),
        "latency_protocol": "single-sample predict including normalization; 3 warmups on non-test inputs; perf_counter_ns",
    }


def scores(truth, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(
            precision_recall_fscore_support(
                truth, predictions, labels=LABELS, average="weighted", zero_division=0
            )[2]
        ),
    }


def class_report(truth, predictions):
    report = classification_report(
        truth, predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    return pd.DataFrame(
        [{"label": label, **report[label]} for label in (*LABELS, "macro avg", "weighted avg")]
    )


def confusion_plot(truth, predictions, path, title):
    matrix = confusion_matrix(truth, predictions, labels=LABELS)
    fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    for i in range(3):
        for j in range(3):
            axis.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return matrix.tolist()


def comparison_plot(frame, path):
    axis = frame.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]].plot.bar(
        figsize=(9, 5), rot=0, ylim=(0, 1), title="Text-only baselines — held-out test"
    )
    axis.set_ylabel("Measured score")
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)


def weight_plot(frame, path):
    pivot = frame.pivot(index="label", columns="class_weight", values="f1-score")
    axis = pivot.plot.bar(
        figsize=(8, 5), rot=0, ylim=(0, 1), title="Class weight comparison — validation F1"
    )
    axis.set_ylabel("Per-class F1")
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)
