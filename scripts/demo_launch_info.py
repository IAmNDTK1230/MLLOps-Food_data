"""Machine-readable launcher details from the pinned demo configuration."""

import json
from pathlib import Path

import yaml

from src.demo import demo_config


def main():
    active = demo_config()
    qwen = yaml.safe_load(Path(active["qwen_config"]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(active["qwen_provenance"]).read_text(encoding="utf-8"))
    print(json.dumps({
        "executable": provenance["server_executable"],
        "model": str((Path(qwen["model_dir"]) / qwen["model_file"]).resolve()),
        "model_file": qwen["model_file"], "port": qwen["port"],
        "threads": qwen["threads"], "context_size": qwen["context_size"],
        "gpu_layers": qwen["gpu_layers"],
    }))


if __name__ == "__main__":
    main()
