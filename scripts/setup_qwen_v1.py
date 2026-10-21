"""Download pinned official model/runtime only. Never touches the review dataset."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def download(url, destination, expected=None):
    if destination.exists() and (not expected or file_checksum(destination) == expected):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "restaurant-qwen-v1"})
    digest, total = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            digest.update(block)
            total += len(block)
            if total % (64 * 1024 * 1024) == 0:
                print(f"{destination.name}: {total // (1024 * 1024)} MiB", flush=True)
    if expected and digest.hexdigest() != expected:
        raise ValueError("Download SHA-256 does not match publisher metadata")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    existing = Path(config["model_dir"]) / "provenance.json"
    if existing.exists():
        ready = json.loads(existing.read_text(encoding="utf-8"))
        if (
            ready["revision"] != config["model_revision"]
            or ready["runtime_tag"] != config["runtime_tag"]
            or file_checksum(Path(config["model_dir"]) / config["model_file"])
            != ready["model_sha256"]
            or file_checksum(Path(ready["server_executable"])) != ready["server_sha256"]
        ):
            raise ValueError(
                "Existing model/runtime differs from frozen config; no silent replacement"
            )
        print("Pinned Qwen model/runtime already verified; no downloads or provenance changes")
        return 0
    root = Path(config["runtime_dir"])
    root.mkdir(parents=True, exist_ok=True)
    for asset in config["runtime_assets"]:
        archive = root / asset["name"]
        download(
            f"https://github.com/ggml-org/llama.cpp/releases/download/{config['runtime_tag']}/{asset['name']}",
            archive,
            asset["sha256"],
        )
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                target = (root / name).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError("Runtime archive has an unsafe path")
            bundle.extractall(root)
    repo, revision = config["model_repo"], config["model_revision"]
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    entry = next(row for row in entries if row["path"] == config["model_file"])
    expected = entry["lfs"]["oid"].removeprefix("sha256:")
    model_root = Path(config["model_dir"])
    path = model_root / config["model_file"]
    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    download(f"{base}/{config['model_file']}", path, expected)
    for name in ("README.md", "LICENSE"):
        download(f"{base}/{name}", model_root / name)
    executable = next(root.rglob("llama-server.exe"))
    provenance = {
        "model": repo,
        "revision": revision,
        "model_sha256": file_checksum(path),
        "model_size_bytes": path.stat().st_size,
        "license": "apache-2.0",
        "runtime": config["runtime"],
        "runtime_tag": config["runtime_tag"],
        "runtime_commit": config["runtime_commit"],
        "server_executable": str(executable.resolve()),
        "server_sha256": file_checksum(executable),
        "runtime_assets": config["runtime_assets"],
        "download_timestamp": datetime.now(UTC).isoformat(),
        "source_url": f"https://huggingface.co/{repo}/tree/{revision}",
    }
    atomic_json(model_root / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
