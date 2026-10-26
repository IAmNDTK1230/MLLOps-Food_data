"""Content-free local operational telemetry."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

LOG = Path("state/llmops.sqlite")


def record(component, model_name, model_version, prompt_version, latency_ms, status,
           input_tokens=None, output_tokens=None, top_k=None, filters=None,
           prediction=None, confidence=None, path=LOG, embedding_model=None,
           retrieval_latency_ms=None, llm_latency_ms=None,
           end_to_end_latency_ms=None, schema_valid=None,
           retrieval_version=None, retrieval_method=None, reranker_model=None,
           candidate_count=None, final_top_k=None, rag_version=None,
           query_route=None, generation_model=None, generation_mode=None,
           evidence_count=None, support_gate_status=None, clause_support_status=None,
           validation_latency_ms=None, repair_latency_ms=None,
           generation_attempts=None, failure_category=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_filters = {key: value for key, value in (filters or {}).items() if key in {
        "restaurant_key", "date_from", "date_to", "sentiment", "aspect"
    } and value is not None}
    with sqlite3.connect(path, timeout=10) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, component TEXT NOT NULL,
            model_name TEXT, model_version TEXT, prompt_version TEXT,
            latency_ms REAL NOT NULL, input_tokens INTEGER, output_tokens INTEGER,
            status TEXT NOT NULL, top_k INTEGER, filters_json TEXT,
            prediction TEXT, confidence REAL
        )""")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
        for name, kind in (
            ("embedding_model", "TEXT"), ("retrieval_latency_ms", "REAL"),
            ("llm_latency_ms", "REAL"), ("end_to_end_latency_ms", "REAL"),
            ("schema_valid", "INTEGER"),
            ("retrieval_version", "TEXT"), ("retrieval_method", "TEXT"),
            ("reranker_model", "TEXT"), ("candidate_count", "INTEGER"),
            ("final_top_k", "INTEGER"),
            ("rag_version", "TEXT"), ("query_route", "TEXT"),
            ("generation_model", "TEXT"), ("generation_mode", "TEXT"),
            ("evidence_count", "INTEGER"), ("support_gate_status", "TEXT"),
            ("clause_support_status", "TEXT"),
            ("validation_latency_ms", "REAL"), ("repair_latency_ms", "REAL"),
            ("generation_attempts", "INTEGER"), ("failure_category", "TEXT"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE events ADD COLUMN {name} {kind}")
        connection.execute("""INSERT INTO events (
            timestamp, component, model_name, model_version, prompt_version, latency_ms,
            input_tokens, output_tokens, status, top_k, filters_json, prediction, confidence,
            embedding_model, retrieval_latency_ms, llm_latency_ms, end_to_end_latency_ms,
            schema_valid, retrieval_version, retrieval_method, reranker_model,
            candidate_count, final_top_k, rag_version, query_route, generation_model,
            generation_mode, evidence_count, support_gate_status, clause_support_status,
            validation_latency_ms, repair_latency_ms, generation_attempts, failure_category
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            datetime.now(UTC).isoformat(), component, model_name, model_version,
            prompt_version, float(latency_ms), input_tokens, output_tokens, status,
            top_k, json.dumps(safe_filters, sort_keys=True, default=str), prediction, confidence,
            embedding_model, retrieval_latency_ms, llm_latency_ms, end_to_end_latency_ms,
            schema_valid,
            retrieval_version, retrieval_method, reranker_model, candidate_count,
            final_top_k, rag_version, query_route, generation_model, generation_mode,
            evidence_count, support_gate_status, clause_support_status,
            validation_latency_ms, repair_latency_ms, generation_attempts, failure_category,
        ))


def summary(path=LOG):
    if not Path(path).exists():
        return []
    with sqlite3.connect(path) as connection:
        return [dict(zip(("component", "status", "request_count", "mean_latency_ms",
                          "input_tokens", "output_tokens"), row, strict=True))
                for row in connection.execute("""SELECT component, status, COUNT(*),
                    AVG(latency_ms), SUM(input_tokens), SUM(output_tokens)
                    FROM events GROUP BY component, status ORDER BY component, status""")]
