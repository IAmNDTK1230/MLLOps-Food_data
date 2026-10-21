import pandas as pd

from src.storage.manifest import atomic_json


def analyze(config, frame, predictions, result, candidates, selection, resources):
    root = config.reports_dir / "proposed_v1"
    rows = predictions.copy()
    rows["error"] = rows.true_label != rows.prediction
    conflicting = frame.groupby("text").sentiment.nunique().gt(1)
    flags = {
        "sarcasm_gold": rows.sarcasm.eq(1),
        "mixed_marker_proxy": rows.text.str.contains(r"(?i)\b(nhưng|tuy|song|vừa)\b", regex=True),
        "negation_marker_proxy": rows.text.str.contains(
            r"(?i)\b(không|chưa|chẳng|chả)\b", regex=True
        ),
        "no_obvious_polarity_word_proxy": ~rows.text.str.contains(
            r"(?i)\b(ngon|tốt|tệ|dở|thích|chán|bẩn|sạch)\b", regex=True
        ),
        "neutral_gold": rows.true_label.eq("neutral"),
        "long_text_200plus_chars": rows.text.str.len().ge(200),
        "short_text_20orless_chars": rows.text.str.len().le(20),
        "conflicting_gold_text": rows.text.isin(conflicting[conflicting].index),
    }
    patterns = []
    for name, flag in flags.items():
        rows[name] = flag
        count, errors = int(flag.sum()), int((flag & rows.error).sum())
        patterns.append(
            {
                "pattern": name,
                "samples": count,
                "errors": errors,
                "error_rate": errors / count if count else None,
            }
        )
    rows.loc[rows.error].to_csv(root / "qwen_failure_cases.csv", index=False, encoding="utf-8")
    aspects = []
    for aspect, group in rows.groupby("aspect"):
        aspects.append(
            {
                "aspect": aspect,
                "samples": len(group),
                "errors": int(group.error.sum()),
                "error_rate": float(group.error.mean()),
            }
        )
    pd.DataFrame(aspects).to_csv(root / "qwen_error_by_aspect.csv", index=False)
    baseline = pd.read_csv(
        config.reports_dir / "baselines/baseline_comparison.csv", keep_default_na=False
    )
    baseline_rows = []
    for row, key in zip(baseline.to_dict("records"), ("rule_based", "tfidf"), strict=True):
        report = pd.read_csv(config.reports_dir / f"baselines/{key}_classification_report.csv")
        baseline_rows.append(
            {
                **row,
                "neutral_f1": float(report.loc[report.label == "neutral", "f1-score"].iloc[0]),
                "json_validity_rate": None,
            }
        )
    comparison = baseline_rows + [
        {
            "model": "Qwen Proposed V1",
            **{
                key: result[key]
                for key in (
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                    "neutral_f1",
                    "mean_latency_ms",
                    "p95_latency_ms",
                    "model_size_mb",
                    "json_validity_rate",
                )
            },
        }
    ]
    table = pd.DataFrame(comparison)
    table.to_csv(root / "model_comparison.csv", index=False)
    axis = table.set_index("model")[["accuracy", "macro_f1", "neutral_f1"]].plot.bar(
        figsize=(11, 6), ylim=(0, 1), rot=15, title="Frozen text-only test comparison"
    )
    axis.figure.tight_layout()
    axis.figure.savefig(root / "model_comparison.png", dpi=160)
    import matplotlib.pyplot as plt

    plt.close(axis.figure)
    deltas = {
        "macro_f1_vs_rule": result["macro_f1"] - baseline_rows[0]["macro_f1"],
        "macro_f1_vs_tfidf": result["macro_f1"] - baseline_rows[1]["macro_f1"],
        "neutral_f1_vs_tfidf": result["neutral_f1"] - baseline_rows[1]["neutral_f1"],
    }
    signals = {
        name: ("POSITIVE IMPROVEMENT SIGNAL" if value > 0 else "NO POSITIVE IMPROVEMENT SIGNAL")
        for name, value in deltas.items()
    }
    error_count = int(rows.error.sum())
    contradictory_errors = int((rows.error & rows.conflicting_gold_text).sum())
    neutral_errors = int((rows.error & rows.neutral_gold).sum())
    recommendation = {
        "implement_now": False,
        "first_step": "Investigate neutral and mixed/implicit sentiment using train/validation prompt examples only",
        "evidence": {
            "test_errors": error_count,
            "neutral_errors": neutral_errors,
            "conflicting_text_errors": contradictory_errors,
        },
        "aspect_conditioned_v2": "Define a separate aspect-conditioned task for contradictory same-text aspect labels; do not call it a text-only comparison",
        "larger_quantized_qwen": "Consider a separately frozen larger Qwen if contextual errors persist; benchmark memory/latency on this GPU first",
        "fine_tuning": "Not implemented; requires a separately authorized training protocol and preserved held-out discipline",
    }
    summary = {
        "patterns": patterns,
        "pattern_warning": "Only sarcasm/neutral labels are source ground truth. Mixed/negation/implicit-word flags are lexical proxies, not verified phenomenon annotations.",
        "conflicting_texts_global": int(conflicting.sum()),
        "conflicting_texts_test": int(rows.groupby("text").true_label.nunique().gt(1).sum()),
        "deltas": deltas,
        "improvement_signals": signals,
        "overall_vs_strongest_baseline": "POSITIVE IMPROVEMENT SIGNAL"
        if deltas["macro_f1_vs_tfidf"] > 0
        else "NO OVERALL IMPROVEMENT OVER TF-IDF",
        "v2_recommendation": recommendation,
    }
    atomic_json(root / "failure_analysis.json", summary)
    evidence = {
        "slide_10": {
            "architecture": "text -> frozen prompt/ChatML -> Qwen GGUF llama.cpp CUDA -> constrained JSON -> validation/parsing -> sentiment",
            "selection": selection,
            "input_features": ["text"],
            "fine_tuned": False,
        },
        "slide_11": {"comparison": comparison, "improvement": deltas, "signals": signals},
        "slide_12": summary,
        "slide_13": recommendation,
        "resources": resources,
    }
    atomic_json(config.reports_dir / "task2_proposed_v1_evidence.json", evidence)
