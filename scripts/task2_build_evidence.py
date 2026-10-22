"""Build slide evidence from frozen, already measured project artifacts."""

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUT = Path("reports/task2")
FIG = OUT / "figures"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=190, bbox_inches="tight")
    plt.close()


def bar(labels, values, title, ylabel, name, color="#2667a5"):
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=color)
    plt.title(title, fontsize=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=15, ha="right")
    for item, value in zip(bars, values, strict=True):
        plt.text(item.get_x() + item.get_width() / 2, value, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=10)
    plt.ylim(0, max(values) * 1.18 if values else 1)
    save(name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    food = read_json("reports/food_dataset_report.json")
    foody = read_json("reports/foody_intelligence/cleaning.json")
    final = read_json("reports/final/KEY_NUMBERS.json")
    token = read_json(OUT / "rag_token_statistics.json")
    accepted = read_csv("reports/final/MODEL_COMPARISON.csv")
    labels = ("negative", "neutral", "positive")
    counts = food["sentiment_counts"]
    ratio = food["imbalance_before"]["majority_minority_ratio"]
    write_csv(OUT / "dataset_sources.csv", [
        {"source": "VN-MSASar v3", "origin": food["source_url"],
         "access": "public offline pinned revision", "raw_rows": food["raw_rows"],
         "filtered_rows": food["valid_rows"], "format": "Parquet",
         "important_fields": "review_id; sentence; aspect; sentiment; place_type; language",
         "role": "research benchmark; provided labels"},
        {"source": "First-party Foody crawl", "origin": "data/foody/source/foody_reviews.csv",
         "access": "first-party local export; no public URL documented",
         "raw_rows": foody["raw_rows"], "filtered_rows": foody["valid_ai_rows"],
         "format": "CSV", "important_fields": "restaurant; review; rating; review time",
         "role": "real-world demo/RAG; no trusted gold sentiment/aspect labels"},
    ])
    (OUT / "DATASET_SUMMARY.md").write_text(
        "# Dataset sources and frozen split\n\n"
        f"VN-MSASar v3 has {food['raw_rows']:,} source rows, filtered to {food['valid_rows']:,} "
        "Vietnamese restaurant sentence/aspect rows and 1,530 unique reviews. "
        f"The frozen split is train {food['splits']['counts']['train']:,}, validation "
        f"{food['splits']['counts']['validation']:,}, test {food['splits']['counts']['test']:,}; "
        f"seed {food['splits']['random_seed']}, split SHA256 "
        f"`{food['splits']['split_file_sha256']}`. The connected-component split groups "
        "review IDs and normalized duplicate text; both reported cross-split counts are zero. "
        "Fuzzy near-duplicate leakage is not ruled out. Labels are supplied by the dataset.\n\n"
        f"The first-party Foody export has {foody['raw_rows']:,} raw rows, "
        f"{foody['valid_ai_rows']:,} valid AI rows, {foody['distinct_valid_review_ids']:,} "
        f"distinct valid review IDs, and {foody['restaurant_count']:,} restaurants. "
        "Foody has no trusted gold sentiment/aspect labels and is used for inference/demo/RAG, "
        "not held-out research scoring. The export has no documented public URL. "
        "See `dataset_sources.csv` and upstream source manifests.\n",
        encoding="utf-8",
    )
    distribution = [{"class": label, "count": counts[label],
                     "percent": food["sentiment_percentages"][label],
                     "majority_minority_ratio": ratio} for label in labels]
    write_csv(OUT / "class_distribution.csv", distribution)
    bar(labels, [counts[x] for x in labels],
        f"VN-MSASar sentiment classes (majority/minority {ratio:.2f}:1)",
        "Sentence/aspect rows", "dataset_class_distribution.png")
    plt.figure(figsize=(9, 5))
    plt.bar(labels, [counts[x] for x in labels], color="#2667a5")
    plt.title("Frozen VN-MSASar class distribution", fontsize=15)
    plt.ylabel("Rows", fontsize=12)
    for i, label in enumerate(labels):
        plt.text(i, counts[label], f"{counts[label]:,}\n({food['sentiment_percentages'][label]:.1f}%)",
                 ha="center", va="bottom")
    plt.ylim(0, max(counts.values()) * 1.2)
    save("class_distribution.png")
    weights = {label: 4140 / (3 * food["splits"]["class_distributions"]["train"][label])
               for label in labels}
    old = [row for row in read_csv("reports/baselines/class_weight_comparison.csv")
           if row["evaluation_split"] == "validation"]
    imbalance = []
    for row in old:
        imbalance.append({"strategy": row["class_weight"], "class": row["label"],
                          "train_rows": food["splits"]["class_distributions"]["train"][row["label"]],
                          "effective_weight": 1 if row["class_weight"] == "None"
                          else weights[row["label"]],
                          "validation_precision": row["precision"],
                          "validation_recall": row["recall"],
                          "validation_f1": row["f1-score"],
                          "validation_support": row["support"]})
    write_csv(OUT / "imbalance_comparison.csv", imbalance)
    x = np.arange(len(labels))
    none = {row["class"]: float(row["validation_f1"]) for row in imbalance
            if row["strategy"] == "None"}
    balanced = {row["class"]: float(row["validation_f1"]) for row in imbalance
                if row["strategy"] == "balanced"}
    plt.figure(figsize=(9, 5))
    plt.bar(x - .18, [none[v] for v in labels], .36, label="Unweighted")
    plt.bar(x + .18, [balanced[v] for v in labels], .36, label="Balanced weights")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Validation F1", fontsize=12)
    plt.title("Class weighting improves neutral recall/F1", fontsize=15)
    plt.legend()
    save("imbalance_effect.png")
    comparison = read_csv(OUT / "classical_model_comparison.csv")
    bar([row["model"] for row in comparison], [float(row["macro_f1"]) for row in comparison],
        "Text-only classical test macro F1", "Macro F1", "baseline_comparison.png")
    sentiment = [row for row in accepted if row["task"] == "sentiment"]
    bar([row["model"] for row in sentiment], [float(row["macro_f1"]) for row in sentiment],
        "Frozen test sentiment and exploratory V2-A", "Macro F1",
        "v1_v2_sentiment_comparison.png")
    aspects = [row for row in read_csv("reports/aspect_classifier/test_per_aspect.csv")
               if int(row["support"]) > 0]
    bar([row["aspect"] for row in aspects], [float(row["f1"]) for row in aspects],
        "Aspect classifier F1 (supported test classes)", "F1", "aspect_per_class_f1.png")
    retrieval = read_csv("reports/rag_v3/retrieval_policies.csv")
    bar([row["policy"].replace("RAG_V2_", "").replace("RAG_V3_", "")
         for row in retrieval], [float(row["known_mrr"]) for row in retrieval],
        "RAG known-positive MRR (six queries; incomplete judgments)", "Known-positive MRR",
        "rag_retrieval_ablation.png")
    rows = []
    for policy, short in (("RAG_V1_DENSE", "Dense only"),
                          ("RAG_V2_BM25", "BM25 only"), ("RAG_V2_RRF", "RRF"),
                          ("RAG_V2_RRF_EXPANDED", "RRF + expansion"),
                          ("RAG_V2_EVIDENCE_RERANK", "Evidence rerank"),
                          ("RAG_V3_ROUTED", "V3 routed retrieval")):
        source = retrieval if policy == "RAG_V3_ROUTED" else read_csv(
            "reports/rag_v2/retrieval_candidates.csv")
        metric = next(item for item in source if item.get("candidate", item.get("policy")) == policy)
        rows.append({"ablation": short,
                     "hit_at_5": metric.get("known_positive_hit_at_5", metric.get("known_hit_at_5")),
                     "hit_at_10": metric.get("known_positive_hit_at_10", metric.get("known_hit_at_10")),
                     "mrr": metric.get("known_positive_mrr", metric.get("known_mrr")),
                     "retrieval_latency_ms": metric["mean_latency_ms"],
                     "supported_answers": "", "unsupported_accepted": "", "fallback_rate": "",
                     "scope": "six queries; incomplete known positives"})
    generation = read_json("reports/rag_v3/generation_metrics.json")["qwen_3b"]["counts"]
    for stage in ("V3 + clause validation", "V3 + fallback"):
        rows.append({"ablation": stage, "hit_at_5": "", "hit_at_10": "", "mrr": "",
                     "retrieval_latency_ms": "", "supported_answers": generation["supported"],
                     "unsupported_accepted": generation["unsupported"],
                     "fallback_rate": generation["extractive_fallback"] / 6,
                     "scope": "same accepted end-to-end V3 outcome; stages not separately isolated"})
    write_csv(OUT / "RAG_ABLATION.csv", rows)
    bar([row["ablation"] for row in rows[:6]],
        [float(row["mrr"]) for row in rows[:6]],
        "Known-positive MRR by retrieval policy (six incomplete judgments)",
        "Known-positive MRR", "rag_retrieval_ablation.png")
    (OUT / "RAG_ABLATION.md").write_text(
        "# RAG ablation evidence\n\nRetrieval figures reuse saved six-query known-positive "
        "judgments. Known positives are incomplete; the values are not full relevance accuracy. "
        "The RRF + expansion and other V2 rows come from the V2 candidate run, while the "
        "V3 routed row comes from the V3 shared policy run; latency is not a strictly controlled "
        "head-to-head replay across these two files. The two generation-stage rows repeat the "
        "same accepted Qwen 3B V3 end-to-end result and cannot isolate clause validation "
        "from fallback. Blank cells mean inapplicable/unmeasured, never zero. In the measured "
        "retrieval comparison V3 routing tied BM25 at known-positive MRR 0.1667 and reduced "
        "latency relative to reranking; this tiny set cannot establish general superiority.\n",
        encoding="utf-8",
    )
    (OUT / "TRAINING_SCOPE.md").write_text(
        "# Training scope\n\nThe group trained the classical TF-IDF/Logistic Regression and "
        "aspect classifiers on frozen VN-MSASar training rows; the rule baseline is manually "
        "specified rather than fitted. The three LLM families Qwen, Llama, and DeepSeek "
        "are pretrained open-weight models used for local inference and benchmarking. "
        "No full LLM pretraining from scratch was performed. Any future LoRA/QLoRA adaptation "
        "would be fine-tuning, not from-scratch training. Foody has no trusted gold labels "
        "and was not used to train these benchmark classifiers.\n",
        encoding="utf-8",
    )
    (OUT / "CLEANING_FLOW.md").write_text(
        "# Cleaning and labeling flow\n\n```mermaid\nflowchart LR\n"
        "A[Raw source] --> B[Strict UTF-8 validation]\n"
        "B --> C[Missing text handling]\nC --> D[Unicode normalization]\n"
        "D --> E[Duplicate grouping]\nE --> F[Date/rating parsing where present]\n"
        "F --> G[Schema and domain validation]\nG --> H[Processed data]\n"
        "H --> I[Foody model enrichment for demo/RAG]\n```\n\n"
        "VN-MSASar labels are supplied by the public dataset. Review-ID and normalized-text "
        "groups define the frozen split, without removing the 5,869 sentence/aspect rows. "
        "Foody is a first-party crawl; its AI cleaning verifies text/date/rating fields and "
        "creates deduplicated observations. Sentiment/aspect enrichment is model inference, "
        "not manual gold labeling. No human annotation or inter-annotator Kappa is claimed.\n",
        encoding="utf-8",
    )
    baseline_refs = (
        ("Rule-based sentiment", "src/baselines/rule_based.py; no cited paper", 0.4312787685745347),
        ("TF-IDF + Logistic Regression",
         "src/baselines/tfidf_logreg.py; scikit-learn TF-IDF and LogisticRegression docs",
         final["research"]["sentiment_tfidf_test_macro_f1"]),
        ("TF-IDF + LinearSVC",
         "scripts/task2_classical_comparison.py; scikit-learn LinearSVC docs",
         0.6672098350839368),
    )
    paper = []
    for baseline, reference, score in baseline_refs:
        paper.append({
            "Baseline": baseline, "Reference": reference,
            "Paper Dataset": "N/A — no reconstructed classifier paper",
            "Paper Task": "N/A — no published matching protocol",
            "Paper Metric": "N/A", "Paper Result": "NOT AVAILABLE",
            "Group Dataset": "VN-MSASar v3; filtered Vietnamese restaurant text-only test (887)",
            "Group Result": f"macro F1 = {score:.6f}", "Absolute Gap": "",
            "Comparability": "NOT DIRECTLY COMPARABLE",
            "Reason": "Dataset, label/task, preprocessing and metric alignment cannot be verified "
                      "without a cited matching paper experiment",
        })
    write_csv(OUT / "paper_vs_group.csv", paper)
    (OUT / "PAPER_VS_GROUP.md").write_text(
        "# Paper vs Group — source audit\n\nThe three local baselines were **not reconstructed "
        "from a cited published classifier paper**. The rule lexicon was authored in "
        "`src/baselines/rule_based.py`; the two linear models were implemented with "
        "scikit-learn. Their scores are original group measurements, not replications "
        "of published scores. The [VN-MSASar v3 dataset card]("
        + food["source_url"] + ") is the dataset source, not a paper reporting these "
        "three baselines on the group's filtered split. The project README, experiment "
        "notes, reports and source contain no citation to a matching baseline paper "
        "or verified paper macro F1.\n\n"
        "| Baseline | Paper reconstruction | Group test macro F1 | Comparison |\n"
        "| --- | --- | ---: | --- |\n"
        "| Rule-based sentiment | No; manually authored domain lexicon | 0.431279 | "
        "NOT DIRECTLY COMPARABLE |\n"
        "| TF-IDF + Logistic Regression | No; local scikit-learn pipeline | 0.682854 | "
        "NOT DIRECTLY COMPARABLE |\n"
        "| TF-IDF + LinearSVC | No; extra validation-selected local comparison | "
        "0.667210 | NOT DIRECTLY COMPARABLE |\n\n"
        "Method references: [TfidfVectorizer](https://scikit-learn.org/stable/modules/"
        "generated/sklearn.feature_extraction.text.TfidfVectorizer.html), "
        "[LogisticRegression](https://scikit-learn.org/stable/modules/generated/"
        "sklearn.linear_model.LogisticRegression.html), and "
        "[LinearSVC](https://scikit-learn.org/stable/modules/generated/"
        "sklearn.svm.LinearSVC.html) describe algorithms/APIs; they supply no "
        "matching performance claim.\n\n"
        "**Dataset gap:** the public card covers a multilingual hotel/restaurant dataset, "
        "while the group filtered Vietnamese restaurant rows and fixed its own split. "
        "**Label/task gap:** source rows are aspect-level sentiment, while these three "
        "classifiers receive text only and collapse sentiment to three classes. "
        "**Preprocessing gap:** no cited paper specifies the same normalization, "
        "TF-IDF, grouping and train-only fitting. **Metric gap:** there is no published "
        "macro F1 on the exact 887-row held-out set. Therefore every numerical absolute "
        "paper-group gap remains blank in `paper_vs_group.csv`; no paper score is invented.\n",
        encoding="utf-8",
    )
    (OUT / "GAP_ANALYSIS.md").write_text(
        "# Evidence-based gap analysis\n\nNo directly comparable paper result exists, so a numeric "
        "paper-group gap is unavailable. Within the frozen group experiment, accepted TF-IDF "
        "macro F1 0.682854 exceeds Qwen V1 0.621789. Neutral F1 is 0.413043 versus 0.257778. "
        "The neutral class has only 294/5,869 rows and its validation F1 rises from 0.170213 "
        "to 0.471910 with balanced weights: measured evidence that weighting helps this "
        "class. Eighty-one normalized texts have conflicting aspect-level labels globally, "
        "including eleven in test: a text-only task has inherent ambiguity.\n\n"
        "Potential causes not isolated experimentally include domain differences, preprocessing, "
        "hyperparameters, pretrained model data, and task/input differences. These are hypotheses, "
        "not proven explanations of the measured gap. Aspect-conditioned V2-A is exploratory "
        "because its test set was previously inspected; RAG V3 addresses evidence grounding "
        "for the demo rather than sentiment benchmark macro F1.\n",
        encoding="utf-8",
    )
    (OUT / "TOKEN_NOTE.md").write_text(
        f"The {token['corpus']['document_count']:,} Foody reviews contain "
        f"{token['minilm_embedding']['total_tokens']:,} untruncated MiniLM content tokens "
        f"and {token['qwen2_5_3b_generation']['total_tokens']:,} Qwen2.5 content tokens. "
        "These are separate tokenizer vocabularies and must not be added as one billable total.\n",
        encoding="utf-8",
    )
    failures = read_json("reports/baselines/failure_analysis_summary.json")
    failure_rows = []
    for model in ("rule", "tfidf"):
        for item in failures["models"][model]["confusions"]:
            failure_rows.append({"source": model, "category":
                                 f"{item['true_label']} → {item['prediction']}",
                                 "count": item["count"], "denominator": 887,
                                 "percent": 100 * item["count"] / 887,
                                 "measurement": "exact test confusion"})
    for item in read_json("reports/proposed_v1/failure_analysis.json")["patterns"]:
        failure_rows.append({"source": "Qwen V1", "category": item["pattern"],
                             "count": item["errors"], "denominator": item["samples"],
                             "percent": 100 * item["error_rate"],
                             "measurement": "pattern subset; lexical proxies where named"})
    rag_failures = read_json("reports/rag_v3/generation_failure_analysis.json")
    for model, content in rag_failures["v3_rejected_attempts"].items():
        for category, count in content["categories"].items():
            failure_rows.append({"source": f"RAG V3 {model}", "category": category,
                                 "count": count, "denominator": content["rejected_attempt_count"],
                                 "percent": 100 * count / content["rejected_attempt_count"],
                                 "measurement": "rejected attempts; may repeat per question"})
    write_csv(OUT / "failure_counts.csv", failure_rows)
    (OUT / "FAILURE_TAXONOMY.md").write_text(
        "# Failure taxonomy\n\n`failure_counts.csv` contains all saved baseline confusion "
        "directions (887 test rows each), all existing Qwen V1 pattern subsets, and all "
        "retained V3 rejected-attempt categories. Pattern subsets overlap, so their counts "
        "must not be added. Mixed/negation/implicit-word flags are lexical proxies, not "
        "human phenomenon labels. The accepted TF-IDF model has 169 test errors; its largest "
        "directions are positive→negative (68) and negative→positive (47). Neutral has 43 "
        "test rows; 24 are misclassified by TF-IDF. Qwen V1 has 237 errors; 16/24 rows "
        "whose text has conflicting gold aspect labels are errors. RAG V3 rejected attempts "
        "include topic mismatch and unsupported clauses; final accepted Qwen 3B answers "
        "have zero validator-detected unsupported claims in the saved six-case set. "
        "The generator gate also correctly refused three insufficient-evidence cases. "
        "The small set and automated validator cannot establish a population hallucination rate.\n",
        encoding="utf-8",
    )
    (OUT / "MICROSERVICE_ARCHITECTURE.md").write_text(
        "# Actual service architecture and Docker status\n\n"
        "`docker-compose.yml` defines pipeline ingestion, optional Prefect server/scheduler, "
        "Streamlit, a separate FastAPI RAG backend, and a llama.cpp `qwen` model server. "
        "The Streamlit frontend and API both mount local data/model/index state; the qwen "
        "service serves a local GGUF. Compose currently points to Qwen 1.5B, whereas the "
        "accepted desktop RAG V3 demo uses Qwen 3B; container behavior has not been "
        "accepted as equivalent. Retrieval uses Chroma dense vectors plus BM25 and V3 "
        "routing/evidence gates. LLMOps monitoring writes local state. These are Compose "
        "services, not proof of an independently deployed production microservice estate. "
        "`Dockerfile.rag` excludes private data/model artifacts from the image; volumes provide "
        "them at runtime.\n\n`docker compose config --quiet`: PASS locally. `docker info`: "
        "FAILED because Docker Desktop Linux daemon pipe is absent. Image build/start and "
        "runtime connectivity: NOT VERIFIED locally. GitHub CI has a Docker build job, "
        "reported separately in `CICD_EVIDENCE.md`.\n",
        encoding="utf-8",
    )
    (OUT / "CICD_EVIDENCE.md").write_text(
        "# GitHub Actions evidence\n\nThe tracked `.github/workflows/ci.yml` runs for "
        "pushes to main/develop, pull requests to main, and manual dispatch. It executes "
        "Ruff, compileall, pytest with JUnit upload, app import, Compose syntax, and "
        "a `Dockerfile.rag` image build in Ubuntu jobs. The public repository API reports "
        "[CI run 39620310721](https://github.com/IAmNDTK1230/MLLOps-Food_data/actions/"
        "runs/39620310721) completed successfully at commit "
        "`d3a8393358233ae930ab1c4f277b1361ac40ad6c`. This verifies that commit, "
        "not these uncommitted Task-2 reports.\n\n`.github/workflows/release.yml` runs on "
        "`v*` tags or manual dispatch, validates a semver-style release tag, and publishes "
        "a GHCR image with the repository token. No release run or GHCR package is evidenced "
        "here; CD publication is NOT YET VERIFIED. No remote production deploy is claimed.\n",
        encoding="utf-8",
    )
    print("Built static Task-2 evidence and figures")


if __name__ == "__main__":
    main()
