"""OpenAI-compatible local llama.cpp gateway around the already pinned Qwen artifact."""

import json
import os
from pathlib import Path
from time import perf_counter

import requests
import yaml

from src.foody.schema import digest
from src.proposed_v1.runtime import LocalQwen
from src.rag.monitoring import record

CONFIG = Path("qwen_v1.yaml")
PROVENANCE = Path("models/qwen_v1/provenance.json")


class LLMGateway:
    def __init__(self, url=None, launch_local=True, monitor=True,
                 config_path=CONFIG, provenance_path=PROVENANCE):
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
        if digest(Path(self.config["model_dir"]) / self.config["model_file"]) != self.provenance["model_sha256"]:
            raise ValueError("Qwen model checksum differs from frozen provenance")
        self.url = (url or os.environ.get("QWEN_BASE_URL") or
                    f"http://127.0.0.1:{self.config['port']}").rstrip("/")
        self.launch_local = launch_local and not (url or os.environ.get("QWEN_BASE_URL"))
        self.monitor = monitor
        self.session = requests.Session()
        self.session.trust_env = False
        self.runtime = None

    @property
    def name(self):
        return self.provenance["model"]

    @property
    def version(self):
        return self.provenance["model_sha256"]

    def __enter__(self):
        try:
            live = self.session.get(self.url + "/health", timeout=2).status_code == 200
        except requests.RequestException:
            live = False
        if not live:
            if not self.launch_local:
                raise RuntimeError("Qwen endpoint is unavailable")
            self.runtime = LocalQwen(self.config, self.provenance)
            self.runtime.__enter__()
        return self

    def __exit__(self, *args):
        if self.runtime:
            self.runtime.__exit__(*args)
        self.session.close()

    def generate(self, *, system, user, response_schema, prompt_version,
                 max_new_tokens=256, component="rag_generation", audit_sink=None):
        start = perf_counter()
        try:
            response = self.session.post(self.url + "/v1/chat/completions", json={
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": 0, "seed": 42, "max_tokens": max_new_tokens,
                "stream": False, "cache_prompt": False,
                "response_format": {"type": "json_schema", "schema": response_schema},
            }, timeout=120)
            response.raise_for_status()
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            if audit_sink is not None:
                audit_sink.append({"raw_output": raw})
            usage = data.get("usage", {})
            value = json.loads(raw, object_pairs_hook=_unique_keys)
            status = "valid_json"
            return value, {"input_tokens": usage.get("prompt_tokens"),
                           "output_tokens": usage.get("completion_tokens"),
                           "latency_ms": (perf_counter() - start) * 1000,
                           "status": status}
        except (requests.RequestException, ValueError, KeyError, TypeError):
            status = "error_or_invalid_json"
            raise
        finally:
            if self.monitor:
                record(component, self.name, self.version, prompt_version,
                       (perf_counter() - start) * 1000, status,
                       input_tokens=usage.get("prompt_tokens") if "usage" in locals() else None,
                       output_tokens=usage.get("completion_tokens") if "usage" in locals() else None)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
