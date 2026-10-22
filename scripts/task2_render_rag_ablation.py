"""Render the final Review-2 RAG evidence figure from saved comparison rows."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001  (backend must be selected first)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "task2" / "RAG_ABLATION_FINAL.csv"
TARGET = ROOT / "reports" / "task2" / "figures" / "rag_ablation_final.png"


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retrieval = [row for row in rows if row["Stage"] == "Retrieval"]
    final = next(row for row in rows if row["Variant"] == "V3 final Qwen 3B system")

    names = [row["Variant"] for row in retrieval]
    hit5 = [float(row["Hit@5"]) for row in retrieval]
    mean_ms = [float(row["Mean retrieval latency ms"]) for row in retrieval]
    colors = ["#2463a5"] * 5 + ["#d8782d"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.45, 1.45, 1]})
    y = list(range(len(names)))
    axes[0].barh(y, hit5, color=colors)
    axes[0].set_yticks(y, labels=names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.12)
    axes[0].set_xlabel("Known-positive Hit@5")
    axes[0].set_title("Retrieval: six labeled queries")
    for index, value in enumerate(hit5):
        axes[0].text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=9)

    axes[1].barh(y, mean_ms, color=colors)
    axes[1].set_yticks(y, labels=[""] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 380)
    axes[1].set_xlabel("Mean retrieval latency (ms)")
    axes[1].set_title("Latency: V2/V3 separate runs")
    for index, value in enumerate(mean_ms):
        axes[1].text(value + 5, index, f"{value:.0f}", va="center", fontsize=9)

    labels = ["Supported\ncomplete", "Supported\nincomplete", "Correct\ninsufficient", "Fallback", "Unsupported\naccepted"]
    keys = ["Supported complete", "Supported incomplete", "Correct insufficient", "Fallback count", "Unsupported accepted"]
    values = [int(final[key]) for key in keys]
    axes[2].bar(range(len(labels)), values, color=["#38875b", "#90b859", "#38875b", "#d8782d", "#b74d4d"])
    axes[2].set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylim(0, 3.7)
    axes[2].set_ylabel("Observed case/event count")
    axes[2].set_title("V3 final safety outcomes")
    for index, value in enumerate(values):
        axes[2].text(index, value + 0.08, str(value), ha="center", fontsize=9)

    fig.suptitle("RAG Review-2 component evidence — descriptive, not causal", fontsize=15)
    fig.text(0.5, 0.01, "Known positives are incomplete; safety counts share six cases and are not additive.",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
