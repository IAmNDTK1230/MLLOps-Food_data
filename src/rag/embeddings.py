"""Explicit multilingual embedding interface and version metadata."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    repo: str
    dimension: int
    query_prefix: str = ""
    passage_prefix: str = ""


CANDIDATES = (
    EmbeddingSpec("minishlab/potion-multilingual-128M",
                  "minishlab/potion-multilingual-128m-onnx", 256),
    EmbeddingSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                  "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q", 384),
)


class LocalEmbeddings:
    def __init__(self, spec, cache_dir="models/embeddings", local_files_only=False):
        from fastembed import TextEmbedding

        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(model_name=spec.name, cache_dir=str(self.cache_dir),
                                   local_files_only=local_files_only)
        self.device = "CPU/ONNX Runtime"

    def _embed(self, values, prefix, batch_size=32):
        started = perf_counter()
        vectors = np.asarray(list(self.model.embed(
            (prefix + value for value in values), batch_size=batch_size
        )), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.spec.dimension:
            raise ValueError("Embedding dimension differs from frozen configuration")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.maximum(norms, 1e-12)
        return vectors, perf_counter() - started

    def documents(self, texts, batch_size=32):
        return self._embed(texts, self.spec.passage_prefix, batch_size)

    def queries(self, texts):
        return self._embed(texts, self.spec.query_prefix, 8)
