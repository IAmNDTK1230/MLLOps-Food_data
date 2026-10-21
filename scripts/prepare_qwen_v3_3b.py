"""Download and pin the official local Qwen2.5 3B Q4_K_M candidate."""

import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from src.foody.schema import digest, save_json

REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
REVISION = "cc1e68eea5f05f88f41a6de1fc73110178f23715"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
DESTINATION = Path("models/qwen_v3")


def main():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    sibling = next(item for item in info.siblings if item.rfilename == FILENAME)
    path = Path(hf_hub_download(REPO, filename=FILENAME, revision=REVISION,
                                local_dir=DESTINATION))
    existing = json.loads(Path("models/qwen_v1/provenance.json").read_text(
        encoding="utf-8",
    ))
    provenance = {
        "model": REPO, "revision": info.sha, "model_file": FILENAME,
        "model_sha256": digest(path), "model_size_bytes": path.stat().st_size,
        "quantization": "Q4_K_M", "parameter_count": 3090000000,
        "license": info.card_data.get("license") if info.card_data else None,
        "runtime": "llama.cpp", "runtime_tag": existing["runtime_tag"],
        "runtime_commit": existing["runtime_commit"],
        "server_executable": existing["server_executable"],
        "server_sha256": existing["server_sha256"],
        "source_url": f"https://huggingface.co/{REPO}/tree/{info.sha}",
        "lfs_sha256": getattr(getattr(sibling, "lfs", None), "sha256", None),
        "download_timestamp": datetime.now(UTC).isoformat(),
    }
    save_json(DESTINATION / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
