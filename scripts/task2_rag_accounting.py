"""Read-only token and chunk workload analysis of the accepted Foody corpus."""

import csv
import json
from collections import Counter
from hashlib import sha256
from math import ceil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from src.rag.vector_store import read_documents

matplotlib.use("Agg")


OUT = Path("reports/task2")
DOCS = Path("data/foody/rag/documents.jsonl")
MANIFEST = Path("data/foody/vector_store/index_manifest.json")
QWEN = Path("models/task2_tokenizers/qwen2.5_3b/tokenizer.json")
QWEN_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
POLICIES = ((None, 0), (128, 0), (128, 32), (256, 0), (256, 32), (512, 0), (512, 64))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(lengths):
    data = np.asarray(lengths, dtype=np.int32)
    return {
        "document_count": len(data), "total_tokens": int(data.sum()),
        "mean_tokens_per_document": float(data.mean()),
        "median_tokens_per_document": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)), "max": int(data.max()),
    }


def policy_measure(lengths, limit, overlap):
    if limit is None:
        return len(lengths), 0, 0
    chunks = 0
    split = 0
    duplication = 0
    for length in lengths:
        if length <= limit:
            chunks += 1
            continue
        split += 1
        count = ceil((length - limit) / (limit - overlap)) + 1
        chunks += count
        for index in range(count):
            start = index * (limit - overlap)
            duplication += min(limit, length - start)
        duplication -= length
    return chunks, split, duplication


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual_hash = sha256(DOCS.read_bytes()).hexdigest()
    if actual_hash != manifest["documents_sha256"]:
        raise ValueError("Foody documents changed from the accepted vector-index source")
    if not QWEN.is_file():
        raise FileNotFoundError("Pinned Qwen2.5-3B tokenizer is absent")
    embed_candidates = list(Path("models/embeddings").glob(
        "models--Qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/snapshots/*/tokenizer.json"
    ))
    embed_candidates = [p for p in embed_candidates if manifest["model_revision"] in p.parts]
    if len(embed_candidates) != 1:
        raise FileNotFoundError("Pinned MiniLM tokenizer snapshot not found uniquely")
    tokenizer_paths = {"minilm_embedding": embed_candidates[0], "qwen2_5_3b_generation": QWEN}
    docs = list(read_documents(DOCS))
    if len(docs) != manifest["document_count"] or len(docs) != 32457:
        raise ValueError("Foody corpus count differs from accepted index")
    texts = [item["text"] for item in docs]
    results = {}
    histograms = {}
    for name, path in tokenizer_paths.items():
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        lengths = []
        for offset in range(0, len(texts), 512):
            lengths.extend(len(item.ids) for item in tokenizer.encode_batch(
                texts[offset:offset + 512], add_special_tokens=False
            ))
        results[name] = {**summary(lengths), "tokenizer_file": str(path),
                         "tokenizer_sha256": sha256(path.read_bytes()).hexdigest(),
                         "special_tokens_included": False, "truncation": "disabled"}
        histograms[name] = lengths
    results["corpus"] = {"document_count": len(docs), "sha256": actual_hash,
                         "embedding_model_revision": manifest["model_revision"],
                         "qwen_tokenizer_revision": QWEN_REVISION}
    (OUT / "rag_token_statistics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    distribution = []
    for name, lengths in histograms.items():
        for length, count in sorted(Counter(lengths).items()):
            distribution.append({"tokenizer": name, "tokens_per_review": length,
                                 "review_count": count})
    write_csv(OUT / "rag_token_distribution.csv", distribution)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, (name, lengths) in zip(axes, histograms.items(), strict=True):
        cap = int(np.percentile(lengths, 99))
        axis.hist([min(n, cap) for n in lengths], bins=35, color="#2870a6", edgecolor="white")
        axis.set_title(f"{name.replace('_', ' ')} (p99+ in last bin)", fontsize=12)
        axis.set_xlabel("Subword tokens per review", fontsize=11)
        axis.set_ylabel("Review documents", fontsize=11)
        axis.tick_params(labelsize=10)
    fig.suptitle("Foody RAG corpus: 32,457 distinct reviews", fontsize=14)
    fig.savefig(OUT / "rag_token_distribution.png", dpi=180)
    plt.close(fig)

    lengths = histograms["minilm_embedding"]
    workload = []
    chunks = []
    actual_seconds = float(manifest["embedding_seconds"])
    throughput = float(manifest["documents_per_second"])
    for limit, overlap in POLICIES:
        count, split, duplicate = policy_measure(lengths, limit, overlap)
        label = "one_review_one_document" if limit is None else f"max_{limit}_overlap_{overlap}"
        chunks.append({"policy": label, "max_tokens": "" if limit is None else limit,
                       "overlap_tokens": overlap, "review_count": len(lengths),
                       "chunk_count": count, "reviews_requiring_split": split,
                       "pct_reviews_requiring_split": split / len(lengths) * 100,
                       "mean_chunks_per_review": count / len(lengths),
                       "duplicated_tokens_from_overlap": duplicate,
                       "estimated_embedding_seconds": count / throughput,
                       "metadata_implication": "one review_id per vector" if limit is None else
                       "inherit review/restaurant metadata; add deterministic chunk_id"})
        workload.append({"policy": label, "chunk_count": count,
                         "reference_documents_per_second": throughput,
                         "reference_actual_seconds": actual_seconds if limit is None else "",
                         "estimated_seconds_at_reference_throughput": count / throughput,
                         "estimate_note": "reference measured build" if limit is None else
                         "estimate only; no alternate index built"})
    write_csv(OUT / "chunking_comparison.csv", chunks)
    write_csv(OUT / "embedding_workload.csv", workload)
    (OUT / "CHUNKING_DECISION.md").write_text(
        "# Foody RAG chunking decision\n\n"
        f"The accepted source has {len(lengths):,} distinct reviews and a matching Chroma index. "
        "One review remains one document; no reviews or restaurants are concatenated. "
        f"The MiniLM content-token maximum is {max(lengths):,}; "
        f"{sum(n > 128 for n in lengths):,} reviews ({sum(n > 128 for n in lengths) / len(lengths):.1%}) "
        "exceed the selected FastEmbed model's documented 128-token truncation limit. "
        f"{sum(n > 512 for n in lengths):,} reviews exceed 512 content tokens. "
        "The seven policies in `chunking_comparison.csv` are simulated with the exact local "
        "MiniLM tokenizer, splitting only within a review. They are not rebuilt indexes or "
        "measured retrieval improvements.\n\n"
        "The one-review policy preserves a direct review_id-to-vector correspondence, "
        "citation granularity, restaurant filters, and the accepted 32,457-ID integrity check. "
        "Splitting long reviews would require chunk IDs, inherited metadata, aggregation back "
        "to review IDs, and new retrieval relevance evaluation. This task does not modify "
        "the accepted index. The 128-token truncation is a material limitation: tail "
        "content of long reviews may not influence their embeddings even though the full "
        "review text remains available to BM25 and answer evidence. The one-review policy "
        "is the verified demo choice, not proven optimal; a separate chunked-index retrieval "
        "evaluation would be needed before changing it.\n\n"
        f"Actual initial embedding compute time was {actual_seconds:.2f} s at {throughput:.2f} "
        "documents/s on CPU ONNX. Alternate-policy times in `embedding_workload.csv` "
        "are proportional estimates using that observed throughput; they exclude index "
        "updates and any retrieval-quality changes. Token counts exclude special tokens "
        "and do not apply model input truncation.\n", encoding="utf-8"
    )
    (OUT / "RAG_METADATA_SCHEMA.md").write_text(
        "# Foody RAG metadata schema\n\n"
        "`data/foody/rag/documents.jsonl` contains one `id`, `text`, and `metadata` object "
        "per distinct valid `review_id`. `src/rag/vector_store.py::chroma_metadata` "
        "persists `review_id`, `restaurant_key`, `restaurant_name`, `review_timestamp`, "
        "`raw_rating`, `predicted_sentiment`, JSON-encoded `predicted_aspects`, "
        "JSON-encoded `detected_issues`, and `source=foody`. When a date exists it adds "
        "`timestamp_epoch`; each predicted aspect adds a Boolean `aspect_<name>` key.\n\n"
        "Restaurant and date fields constrain retrieval; sentiment, aspect, and issue fields "
        "support filters and evidence checks. Ratings are source attributes, never gold "
        "sentiment labels. The restaurant key is derived from its name, not an official ID. "
        "Chunk simulations preserve all metadata but would require a separate chunk ID.\n",
        encoding="utf-8"
    )
    print(json.dumps({name: row["total_tokens"] for name, row in results.items()
                      if "total_tokens" in row}))


if __name__ == "__main__":
    main()
