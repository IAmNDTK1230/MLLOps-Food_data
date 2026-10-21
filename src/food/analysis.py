import json
import math

import numpy as np
import pandas as pd

from src.food.data import fetch, hub_metadata
from src.food.interfaces import EXPERIMENT_PLAN, StructuredReviewAnalysis
from src.food.split import load_splits
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def distributions(frame, column):
    counts = frame[column].value_counts().rename_axis(column).reset_index(name="count")
    counts["percentage"] = counts["count"] / len(frame) * 100
    return counts


def rag_documents(frame):
    """Reviews reconstructed from ordered source sentences, not claimed to be raw reviews."""
    reviews, sentences = [], []
    for review_id, rows in frame.groupby("review_id", sort=True):
        annotations = rows[
            ["text", "aspect", "sentiment", "original_sentiment", "sarcasm"]
        ].to_dict("records")
        common = {
            "review_id": review_id,
            "source": sorted(rows.source.unique().tolist()),
            "aspect": sorted(rows.aspect.unique().tolist()),
            "sentiment": sorted(rows.sentiment.unique().tolist()),
            "dataset_name": "vn_msasar_v3",
            "split": rows.split.iloc[0],
            "license": rows.license.iloc[0],
            "source_url": rows.source_url.iloc[0],
        }
        reviews.append(
            {**common, "text": " ".join(rows.text.drop_duplicates()), "annotations": annotations}
        )
        for text, group in rows.groupby("text", sort=True):
            sentences.append(
                {
                    **common,
                    "text": text,
                    "aspect": sorted(group.aspect.unique().tolist()),
                    "sentiment": sorted(group.sentiment.unique().tolist()),
                    "annotations": group[
                        ["aspect", "sentiment", "original_sentiment", "sarcasm"]
                    ].to_dict("records"),
                }
            )
    return reviews, sentences


def token_summary(lengths):
    return {
        "document_count": len(lengths),
        "token_count": int(sum(lengths)),
        "mean_tokens_per_document": float(np.mean(lengths)) if lengths else 0,
        "median": float(np.median(lengths)) if lengths else 0,
        "p95": float(np.percentile(lengths, 95)) if lengths else 0,
        "max": int(max(lengths, default=0)),
    }


def chunk_count(tokens, size, overlap):
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be nonnegative and smaller than chunk size")
    return 0 if tokens == 0 else 1 + math.ceil(max(0, tokens - size) / (size - overlap))


def token_analysis(reviews, sentences, encode, config, tokenizer_metadata):
    lengths = {
        "one_review_per_document": [len(encode(doc["text"])) for doc in reviews],
        "sentence_level": [len(encode(doc["text"])) for doc in sentences],
    }
    summary = {key: token_summary(values) for key, values in lengths.items()}
    rows = [
        {
            "method": key,
            "base_documents": key,
            "chunk_size": None,
            "overlap": 0,
            "chunk_count": len(values),
            "token_count_with_overlap": sum(values),
        }
        for key, values in lengths.items()
    ]
    for key, values in lengths.items():
        for size in config.chunk_sizes:
            for overlap in config.overlaps:
                counts = [chunk_count(value, size, overlap) for value in values]
                rows.append(
                    {
                        "method": "fixed_token",
                        "base_documents": key,
                        "chunk_size": size,
                        "overlap": overlap,
                        "chunk_count": sum(counts),
                        "token_count_with_overlap": sum(values)
                        + sum(max(0, count - 1) * overlap for count in counts),
                    }
                )
    return {
        "tokenizer": tokenizer_metadata,
        "statistics": summary,
        "selected_chunk_size": None,
        "status": "analysis_only_no_embeddings_no_model_inference",
        "review_reconstruction": "unique normalized source sentences in stable lexical order; original review ordering is unavailable",
        "sentence_definition": "dataset sentence column; no new sentence segmentation",
    }, pd.DataFrame(rows)


def load_tokenizer(config, local_path=None):
    from tokenizers import Tokenizer

    root = config.data_dir / "metadata/tokenizer"
    root.mkdir(parents=True, exist_ok=True)
    path, meta_path = root / "tokenizer.json", root / "provenance.json"
    if local_path:
        path = local_path
        metadata = {
            "repo": config.tokenizer_repo,
            "revision": "local_unverified",
            "requested_revision": config.tokenizer_revision,
        }
    elif path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            metadata["repo"] != config.tokenizer_repo
            or (
                metadata["requested_revision"] != config.tokenizer_revision
                and metadata["revision"] != config.tokenizer_revision
            )
            or metadata["sha256"] != file_checksum(path)
        ):
            raise ValueError(
                "Tokenizer cache/config mismatch; supply --tokenizer-file or refresh cache"
            )
        metadata["requested_revision"] = config.tokenizer_revision
    else:
        try:
            hub = hub_metadata(config.tokenizer_repo, config.tokenizer_revision, dataset=False)
            path.write_bytes(
                fetch(
                    f"https://huggingface.co/{config.tokenizer_repo}/resolve/{hub['sha']}/tokenizer.json"
                )
            )
            metadata = {
                "repo": config.tokenizer_repo,
                "revision": hub["sha"],
                "requested_revision": config.tokenizer_revision,
                "license": hub.get("cardData", {}).get("license"),
                "reference_only": True,
            }
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Tokenizer-only download failed ({type(error).__name__}); download tokenizer.json from https://huggingface.co/{config.tokenizer_repo} and rerun food-analysis --tokenizer-file <path>. No estimated token counts substituted."
            ) from None
    metadata.update(
        sha256=file_checksum(path), add_special_tokens=False, model_weights_downloaded=False
    )
    if not local_path:
        atomic_json(meta_path, metadata)
    tokenizer = Tokenizer.from_file(str(path))
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids, metadata


def charts(config, sentiment, aspects, lengths, by_aspect):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, data, x in (
        ("sentiment_distribution", sentiment, "sentiment"),
        ("aspect_distribution", aspects, "aspect"),
    ):
        fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
        axis.bar(data[x], data["count"])
        axis.set(title=name.replace("_", " "), ylabel="Sentence/aspect rows")
        axis.tick_params(axis="x", rotation=35)
        fig.savefig(config.reports_dir / f"{name}.png", dpi=160)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    axis.hist(lengths["characters"], bins=40)
    axis.set(
        title="Text length distribution",
        xlabel="Normalized Unicode characters",
        ylabel="Sentence/aspect rows",
    )
    fig.savefig(config.reports_dir / "text_length_distribution.png", dpi=160)
    plt.close(fig)
    pivot = by_aspect.pivot(index="aspect", columns="sentiment", values="count").fillna(0)
    axis = pivot.plot.bar(stacked=True, figsize=(10, 6), title="Sentiment by aspect")
    axis.set_ylabel("Sentence/aspect rows")
    axis.figure.tight_layout()
    axis.figure.savefig(config.reports_dir / "sentiment_by_aspect.png", dpi=160)
    plt.close(axis.figure)


def analyze(config, tokenizer_file=None, encoder=None, tokenizer_metadata=None):
    frame, split, preparation = load_splits(config)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    sentiment, aspects = distributions(frame, "sentiment"), distributions(frame, "aspect")
    lengths = frame[["review_id", "aspect", "split"]].copy()
    lengths["characters"] = frame.text.str.len()
    lengths["whitespace_words"] = frame.text.str.split().str.len()
    by_aspect = frame.groupby(["aspect", "sentiment"]).size().reset_index(name="count")
    by_aspect["percentage_within_aspect"] = (
        by_aspect["count"] / by_aspect.groupby("aspect")["count"].transform("sum") * 100
    )
    for name, data in (
        ("sentiment_distribution", sentiment),
        ("aspect_distribution", aspects),
        ("text_length_distribution", lengths),
        ("sentiment_by_aspect", by_aspect),
    ):
        data.to_csv(config.reports_dir / f"{name}.csv", index=False, encoding="utf-8")
    class_counts = frame.sentiment.value_counts()
    report = {
        **preparation,
        "unique_reviews": int(frame.review_id.nunique()),
        "sentence_aspect_pairs": len(frame),
        "unique_sentences": int(frame.text.nunique()),
        "sentiment_counts": class_counts.to_dict(),
        "sentiment_percentages": (frame.sentiment.value_counts(normalize=True) * 100).to_dict(),
        "aspect_counts": frame.aspect.value_counts().to_dict(),
        "source_counts": frame.source.value_counts().to_dict(),
        "text_length_characters": {
            "mean": float(lengths.characters.mean()),
            "median": float(lengths.characters.median()),
            "min": int(lengths.characters.min()),
            "max": int(lengths.characters.max()),
        },
        "imbalance_before": {
            "majority_minority_ratio": float(class_counts.max() / class_counts.min()),
            "counts": class_counts.to_dict(),
        },
        "imbalance_after": None,
        "splits": split,
        "source_other_caveat": "Publisher describes source=other as deliberately collected real sarcastic reviews; retain source for stratified future evaluation",
        "no_training_executed": True,
    }
    atomic_json(config.reports_dir / "food_dataset_report.json", report)
    charts(config, sentiment, aspects, lengths, by_aspect)
    reviews, sentences = rag_documents(frame)
    for name, documents in (("review_documents", reviews), ("sentence_documents", sentences)):
        path = config.data_dir / f"rag/{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents),
            encoding="utf-8",
        )
    atomic_json(config.reports_dir / "food_experiment_plan.json", EXPERIMENT_PLAN)
    atomic_json(
        config.reports_dir / "qwen_output_schema.json", StructuredReviewAnalysis.model_json_schema()
    )
    if encoder is None:
        encoder, tokenizer_metadata = load_tokenizer(config, tokenizer_file)
    tokens, comparison = token_analysis(reviews, sentences, encoder, config, tokenizer_metadata)
    tokens["processed_sha256"] = preparation["processed_sha256"]
    tokens["split_file_sha256"] = split["split_file_sha256"]
    atomic_json(config.reports_dir / "token_statistics.json", tokens)
    comparison.to_csv(config.reports_dir / "chunking_comparison.csv", index=False)
    slides = {
        "slide_3": {
            "source": preparation["source_url"],
            "revision": preparation["revision"],
            "license": preparation["license"],
            "raw_rows": preparation["raw_rows"],
            "valid_rows": len(frame),
            "schema": preparation["canonical_schema"],
        },
        "slide_4": {
            "workflow": "snapshot+SHA256 -> exact vi/restaurant filters -> NFC/whitespace -> reject empty/invalid labels -> sign-based gold sentiment -> exact duplicate removal",
            "counts": {
                key: preparation[key]
                for key in (
                    "raw_rows",
                    "filtered_vietnamese_restaurant_rows",
                    "valid_rows",
                    "rejected_rows",
                    "duplicates_removed",
                )
            },
            "label_mapping": "food_label_mapping.json",
        },
        "slide_5": {"splits": split, "imbalance_before": report["imbalance_before"]},
        "slide_6": {"proposal": EXPERIMENT_PLAN["imbalance"], "after_results": None},
        "rag": {
            "tokens": tokens,
            "comparison": "chunking_comparison.csv",
            "metadata_fields": [
                "review_id",
                "source",
                "aspect",
                "sentiment",
                "dataset_name",
                "split",
                "license",
                "source_url",
                "annotations",
            ],
            "embeddings_built": False,
        },
    }
    atomic_json(config.reports_dir / "task2_slide_evidence.json", slides)
    return {
        "valid_rows": len(frame),
        "splits": split["counts"],
        "reports_dir": str(config.reports_dir),
        "token_statistics": tokens,
    }
