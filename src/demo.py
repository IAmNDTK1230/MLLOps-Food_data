"""Read-only artifact checks for the frozen local presentation demo."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from src.foody.schema import SOURCE, digest

CONFIG = Path("config/demo.yaml")


def demo_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parquet_rows(path):
    return pq.read_metadata(path).num_rows


def demo_preflight(config_path=CONFIG):
    """Inspect frozen artifacts without inference, training, indexing, or writes."""
    config = demo_config(config_path)
    checks = {}
    details = {}

    def check(label, verify):
        try:
            details[label] = verify()
            checks[label] = "PASS"
        except (OSError, ValueError, KeyError, ImportError, sqlite3.Error) as error:
            checks[label] = "FAIL"
            details[label] = type(error).__name__ + ": " + str(error)[:180]

    def foody():
        enrichment = _json("reports/foody_intelligence/enrichment.json")
        source = SOURCE
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("Foody source CSV missing or empty")
        rows = _parquet_rows("data/foody/enriched/reviews.parquet")
        if rows != enrichment["rows_preserved"]:
            raise ValueError("Enriched row count differs from report")
        return {"reviews": rows, "ai_valid": enrichment["inference_observations"]}

    check("Foody dataset", foody)
    try:
        versions = _json("reports/foody_intelligence/versions.json")
    except (OSError, ValueError):
        versions = {}

    def model(path, expected):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Model checksum mismatch: {path}")
        return {"path": path, "sha256": actual}

    check("Sentiment model", lambda: model(config["sentiment_model"],
                                             versions["sentiment_model_sha256"]))
    check("Aspect model", lambda: model(config["aspect_model"],
                                          versions["aspect_model_sha256"]))

    def chroma():
        from src.rag.vector_store import open_collection

        collection, manifest = open_collection()
        if manifest["collection"] != config["vector_collection"]:
            raise ValueError("Active vector collection differs from demo config")
        if manifest["model_name"] != config["embedding_model"]:
            raise ValueError("Active embedding model differs from demo config")
        return {"collection": collection.name, "vector_count": collection.count()}

    check("Chroma", chroma)

    def bm25():
        manifest = _json("data/foody/lexical_index/index_manifest.json")
        database = Path("data/foody/lexical_index/reviews.sqlite")
        if not database.is_file():
            raise ValueError("BM25 database missing")
        with sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True) as db:
            count = db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        if count != manifest["document_count"]:
            raise ValueError("BM25 count differs from manifest")
        return {"document_count": count}

    check("BM25", bm25)

    def qwen():
        from src.rag.service import prompt

        provenance = _json(config["qwen_provenance"])
        qwen_config = _json_from_yaml(config["qwen_config"])
        if provenance["model"] != config["qwen_model"]:
            raise ValueError("Qwen model differs from active config")
        path = Path(qwen_config["model_dir"]) / qwen_config["model_file"]
        if digest(path) != provenance["model_sha256"]:
            raise ValueError("Qwen 3B checksum mismatch")
        runtime = Path(provenance["server_executable"])
        if not runtime.is_file():
            raise ValueError("llama.cpp server executable missing")
        _, prompt_version = prompt(config["prompt_path"])
        if prompt_version != config["prompt_version"] or config["rag_version"] != "RAG_V3":
            raise ValueError("Active RAG prompt/version differs from demo config")
        return {"model": provenance["model"], "sha256": provenance["model_sha256"],
                "runtime": str(runtime)}

    check("Qwen 3B", qwen)

    def marts():
        paths = ["restaurant_summary", "restaurant_monthly_metrics",
                 "restaurant_aspect_summary", "restaurant_issue_summary", "alerts"]
        rows = {name: _parquet_rows(f"data/foody/mart/{name}.parquet") for name in paths}
        if rows["restaurant_summary"] == 0:
            raise ValueError("Restaurant summary is empty")
        return {"restaurants": rows["restaurant_summary"], "tables": rows}

    check("Restaurant marts", marts)

    def llmops():
        path = Path(config["llmops_backend"])
        if not path.is_file():
            raise ValueError("LLMOps SQLite database missing")
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as db:
            db.execute("SELECT COUNT(*) FROM events").fetchone()
        return {"path": str(path)}

    check("LLMOps SQLite", llmops)
    check("Streamlit import", lambda: {"installed": True} if
          importlib.util.find_spec("streamlit") else _missing_streamlit())
    try:
        rag_manifest = _json("data/foody/vector_store/index_manifest.json")
    except (OSError, ValueError):
        rag_manifest = {}
    return {"checks": checks, "details": details,
            "counts": {"foody_reviews": details.get("Foody dataset", {}).get("reviews")
                       if isinstance(details.get("Foody dataset"), dict) else None,
                       "restaurants": details.get("Restaurant marts", {}).get("restaurants")
                       if isinstance(details.get("Restaurant marts"), dict) else None,
                       "rag_docs": rag_manifest.get("document_count"),
                       "vector_count": details.get("Chroma", {}).get("vector_count")
                       if isinstance(details.get("Chroma"), dict) else None},
            "active_models": {"sentiment": config["sentiment_model"],
                              "aspect": config["aspect_model"],
                              "embedding": config["embedding_model"],
                              "generator": config["qwen_model"]},
            "active_rag_version": config["rag_version"],
            "status": "PASS" if all(value == "PASS" for value in checks.values())
            else "FAIL"}


def _json_from_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _missing_streamlit():
    raise ImportError("Streamlit is not installed")


def print_preflight(result):
    width = max(len(label) for label in result["checks"])
    for label, status in result["checks"].items():
        print(f"{label:<{width}}  {status}")
        if status != "PASS":
            print(f"  {result['details'][label]}")
    counts = result["counts"]
    print(f"Foody reviews: {counts['foody_reviews']}; restaurants: {counts['restaurants']}; "
          f"RAG docs: {counts['rag_docs']}; vectors: {counts['vector_count']}")
    print("Active models: " + ", ".join(
        f"{name}={value}" for name, value in result["active_models"].items()))
    print("Active RAG version: " + result["active_rag_version"])
    print("Preflight: " + result["status"])
