"""Frozen demo configuration and read-only artifact acceptance."""

from pathlib import Path

import pytest

from src.demo import demo_config, demo_preflight


def test_active_demo_configuration_is_v3_3b():
    config = demo_config()
    assert config["rag_version"] == "RAG_V3"
    assert config["qwen_config"] == "qwen_v3_3b.yaml"
    assert config["qwen_provenance"] == "models/qwen_v3/provenance.json"
    assert config["top_k"] == 5


@pytest.mark.skipif(
    not all(Path(path).exists() for path in (
        "data/foody/source/foody_reviews.csv",
        "data/foody/vector_store/index_manifest.json",
        "data/foody/lexical_index/reviews.sqlite",
        "models/qwen_v3/qwen2.5-3b-instruct-q4_k_m.gguf",
        "state/llmops.sqlite",
    )),
    reason="Requires private local demo artifacts; CI uses synthetic tests",
)
def test_frozen_demo_artifacts_pass_read_only_preflight():
    result = demo_preflight()
    assert result["status"] == "PASS", result["details"]
    assert result["counts"] == {"foody_reviews": 33946, "restaurants": 2110,
                                "rag_docs": 32457, "vector_count": 32457}
