"""Freeze a tiny manually inspected known-positive retrieval set, with no label inference."""

import json
from pathlib import Path

from src.foody.schema import digest, save_json
from src.rag.vector_store import DOCUMENTS, read_documents

TARGET = Path("reports/rag/development_judgments.json")
CASES = [
    ("khách phàn nàn về phục vụ chậm", ["834282a9843b14", "650c44f4d8a4ba"]),
    ("đồ ăn không ngon", ["dd667028f270f1", "159c264cb68920"]),
    ("giá quá cao", ["d970727db5ec0e"]),
    ("không gian ồn", ["9e3b0c180702d209", "93981f2296a4ba87"]),
    ("nhân viên thái độ kém", ["862799ddfb017f21", "24cf1788733bd7ff"]),
    ("món ăn ngon", ["b27220feaa5d6238", "1841c0815446d251", "2c4ec894ee8e020d"]),
]


def create():
    source_sha = digest(DOCUMENTS)
    if TARGET.exists():
        saved = json.loads(TARGET.read_text(encoding="utf-8"))
        if saved["source_sha256"] != source_sha:
            raise ValueError("Frozen development judgments refer to another RAG source")
        return saved
    docs = list(read_documents())
    cases = []
    for query, prefixes in CASES:
        selected = []
        for prefix in prefixes:
            matches = [doc for doc in docs if doc["id"].startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous/missing inspected review prefix: {prefix}")
            selected.append(matches[0]["id"])
        cases.append({"query": query, "known_relevant_review_ids": selected})
    result = {
        "source_sha256": source_sha, "cases": cases,
        "judgment_method": "Single-reviewer manual reading of explicit Vietnamese review passages; no model predictions used as relevance labels",
        "completeness": "Incomplete positives only; unlabeled documents may also be relevant. Known-positive hit rates are not full-corpus recall or retrieval accuracy.",
        "selection": "Frozen before embedding candidate comparison; six fixed Vietnamese product questions",
    }
    save_json(TARGET, result)
    return result


if __name__ == "__main__":
    result = create()
    print(f"Frozen {len(result['cases'])} development queries from {DOCUMENTS}")
