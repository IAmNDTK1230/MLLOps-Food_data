"""Persistent local SQLite FTS5/BM25 index over the frozen Foody documents."""

import json
import re
import sqlite3
from pathlib import Path
from time import perf_counter

from src.foody.schema import digest, save_json
from src.rag.schemas import RetrievalRequest, ReviewHit
from src.rag.vector_store import DOCUMENTS, chroma_metadata, epoch, read_documents

STORE = Path("data/foody/lexical_index")
DATABASE = STORE / "reviews.sqlite"
MANIFEST = STORE / "index_manifest.json"
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {"khách", "về", "có", "gì", "điều", "nào", "nhất", "quá", "không",
             "là", "của", "ở", "này", "cho", "tôi", "mình"}


def search_terms(query):
    terms = [term.lower() for term in TOKEN.findall(query)]
    return list(dict.fromkeys(term for term in terms if term and term not in STOPWORDS))


def match_expression(query):
    # Quoted tokens keep arbitrary user punctuation out of FTS5 syntax.
    raw = [term.lower() for term in TOKEN.findall(query)]
    phrases = [" ".join(raw[i:i + size]) for size in (3, 2)
               for i in range(len(raw) - size + 1)
               if any(term not in STOPWORDS for term in raw[i:i + size])]
    terms = list(dict.fromkeys([*phrases, *search_terms(query)]))
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def build_index(source=DOCUMENTS, database=DATABASE, manifest=MANIFEST):
    source, database, manifest = Path(source), Path(database), Path(manifest)
    source_hash = digest(source)
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info["source_sha256"] != source_hash or not database.exists():
            raise ValueError("Lexical index source or database differs from manifest")
        with sqlite3.connect(database) as connection:
            count = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        if count != info["document_count"]:
            raise ValueError("Incomplete lexical index")
        return info
    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_suffix(".building.sqlite")
    if temporary.exists():
        temporary.unlink()
    started = perf_counter()
    with sqlite3.connect(temporary) as connection:
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("CREATE VIRTUAL TABLE reviews_fts USING fts5(review_id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2')")
        connection.execute("""CREATE TABLE reviews (
            review_id TEXT PRIMARY KEY, restaurant_key TEXT, restaurant_id TEXT,
            sentiment TEXT, timestamp_epoch INTEGER, metadata_json TEXT NOT NULL
        )""")
        connection.execute("CREATE TABLE review_aspects (review_id TEXT, aspect TEXT, PRIMARY KEY(review_id, aspect))")
        connection.execute("CREATE INDEX reviews_filter ON reviews(restaurant_key, sentiment, timestamp_epoch)")
        connection.execute("CREATE INDEX reviews_id_filter ON reviews(restaurant_id, sentiment, timestamp_epoch)")
        connection.execute("CREATE INDEX aspects_filter ON review_aspects(aspect, review_id)")
        count = 0
        for doc in read_documents(source):
            metadata = chroma_metadata(doc)
            connection.execute("INSERT INTO reviews_fts(review_id, text) VALUES (?, ?)",
                               (doc["id"], doc["text"]))
            connection.execute("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)", (
                doc["id"], metadata.get("restaurant_key"), metadata.get("restaurant_id"),
                metadata["predicted_sentiment"],
                metadata.get("timestamp_epoch"), json.dumps(metadata, ensure_ascii=False),
            ))
            connection.executemany("INSERT INTO review_aspects VALUES (?, ?)",
                                   ((doc["id"], aspect) for aspect in doc["metadata"]["predicted_aspects"]))
            count += 1
        connection.execute("INSERT INTO reviews_fts(reviews_fts) VALUES ('optimize')")
    connection.close()
    temporary.replace(database)
    elapsed = perf_counter() - started
    result = {"backend": "SQLite FTS5 BM25", "source_sha256": source_hash,
              "document_count": count, "path": str(database.resolve()),
              "build_seconds": elapsed, "documents_per_second": count / elapsed,
              "tokenizer": "unicode61 remove_diacritics 2"}
    save_json(manifest, result)
    return result


class BM25Retriever:
    def __init__(self, database=DATABASE, manifest=MANIFEST, source=DOCUMENTS):
        self.database = Path(database)
        self.manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if self.manifest["source_sha256"] != digest(source):
            raise ValueError("Lexical index differs from current source documents")

    def search(self, request, query=None, top_n=None):
        if not isinstance(request, RetrievalRequest):
            request = RetrievalRequest.model_validate(request)
        started = perf_counter()
        expression = match_expression(query or request.query)
        if not expression:
            return [], (perf_counter() - started) * 1000
        clauses = ["reviews_fts MATCH ?"]
        values = [expression]
        if request.restaurant_key:
            clauses.append("reviews.restaurant_key = ?")
            values.append(request.restaurant_key)
        if request.restaurant_id:
            clauses.append("reviews.restaurant_id = ?")
            values.append(str(request.restaurant_id))
        if request.sentiment:
            clauses.append("reviews.sentiment = ?")
            values.append(request.sentiment)
        if request.aspect:
            clauses.append("EXISTS (SELECT 1 FROM review_aspects a WHERE a.review_id = reviews.review_id AND a.aspect = ?)")
            values.append(request.aspect)
        if request.date_from:
            clauses.append("reviews.timestamp_epoch >= ?")
            values.append(epoch(request.date_from.isoformat()))
        if request.date_to:
            from src.rag.retriever import where_filter

            end_clause = where_filter(request)
            # The shared filter chooses exclusive next-day upper bounds for date-only input.
            upper = end_clause["$and"][-1] if "$and" in end_clause else end_clause
            bound = upper["timestamp_epoch"]
            operator, value = next(iter(bound.items()))
            clauses.append("reviews.timestamp_epoch " + ("<" if operator == "$lt" else "<=") + " ?")
            values.append(value)
        sql = ("SELECT reviews.review_id, reviews_fts.text, reviews.metadata_json, "
               "bm25(reviews_fts) AS rank FROM reviews_fts JOIN reviews "
               "ON reviews.review_id = reviews_fts.review_id WHERE "
               + " AND ".join(clauses) + " ORDER BY rank LIMIT ?")
        values.append(top_n or request.top_k)
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(sql, values).fetchall()
        hits = [ReviewHit(review_id=review_id, restaurant_name=metadata["restaurant_name"],
                          text=text, score=-float(rank), metadata=metadata)
                for review_id, text, raw, rank in rows
                if (metadata := json.loads(raw))]
        return hits, (perf_counter() - started) * 1000
