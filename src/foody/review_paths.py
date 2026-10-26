"""Shared paths and version for the 2025–2026 review corpus."""

from pathlib import Path

SOURCE = Path("data/foody/source/data_2025_2026.csv")
CANONICAL = Path("data/foody/processed/reviews_2025_2026.parquet")
ENRICHED = Path("data/foody/processed/reviews_2025_2026_enriched.parquet")
DOCUMENTS = Path("data/foody/rag/reviews_2025_2026.jsonl")
MANIFEST = Path("data/foody/processed/reviews_2025_2026_manifest.json")
DATA_VERSION = "review_2025_2026_v1"
