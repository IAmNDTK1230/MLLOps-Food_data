import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

import psutil
import requests
from pydantic import BaseModel, ConfigDict


class SentimentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sentiment: Literal["negative", "neutral", "positive"]


def parse_output(raw):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return SentimentOutput.model_validate(json.loads(raw, object_pairs_hook=unique_keys)).sentiment


def messages(instruction, text):
    # Deliberately no aspect, source, IDs, split, score, or current sample label.
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps({"text": text}, ensure_ascii=False)},
    ]


class LocalQwen:
    def __init__(self, config, provenance):
        self.config, self.provenance = config, provenance
        self.session = requests.Session()
        self.session.trust_env = False
        self.url = f"http://127.0.0.1:{config['port']}"
        self.peak_rss, self.cpu_samples, self.gpu_samples = 0, [], []
        self.stop = threading.Event()

    def __enter__(self):
        Path("logs").mkdir(exist_ok=True)
        self.log = Path(self.config.get("server_log_path", "logs/qwen_v1_server.log")).open(
            "a", encoding="utf-8",
        )
        model = (Path(self.config["model_dir"]) / self.config["model_file"]).resolve()
        command = [
            self.provenance["server_executable"],
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.config["port"]),
            "--n-gpu-layers",
            str(self.config["gpu_layers"]),
            "--threads",
            str(self.config["threads"]),
            "--threads-batch",
            str(self.config["threads"]),
            "--ctx-size",
            str(self.config["context_size"]),
            "--parallel",
            "1",
            "--batch-size",
            "512",
            "--ubatch-size",
            "256",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for _ in range(120):
                if self.process.poll() is not None:
                    raise RuntimeError("llama-server exited; inspect logs/qwen_v1_server.log")
                try:
                    if self.session.get(self.url + "/health", timeout=1).status_code == 200:
                        self.monitor = threading.Thread(target=self.sample_resources, daemon=True)
                        self.monitor.start()
                        return self
                except requests.RequestException:
                    pass
                time.sleep(1)
            raise RuntimeError("llama-server startup timeout")
        except Exception:
            self.__exit__(None, None, None)
            raise

    def sample_resources(self):
        process = psutil.Process(self.process.pid)
        process.cpu_percent()
        while not self.stop.wait(1):
            try:
                self.peak_rss = max(self.peak_rss, process.memory_info().rss)
                self.cpu_samples.append(process.cpu_percent())
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=3,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                self.gpu_samples.append(
                    [float(x.strip()) for x in output.splitlines()[0].split(",")]
                )
                # Keep telemetry even if downstream report assembly fails.
                with Path("logs/qwen_v1_resource_samples.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(
                        json.dumps(
                            {
                                "timestamp_unix": time.time(),
                                "server_pid": self.process.pid,
                                "rss_bytes": process.memory_info().rss,
                                "cpu_percent": self.cpu_samples[-1],
                                "gpu_utilization_percent": self.gpu_samples[-1][0],
                                "gpu_memory_mib": self.gpu_samples[-1][1],
                            }
                        )
                        + "\n"
                    )
            except (psutil.Error, OSError, subprocess.SubprocessError, ValueError):
                pass

    def predict(self, instruction, text):
        attempts, start = [], time.perf_counter()
        for attempt in range(self.config["max_attempts"]):
            retry = (
                "\nYour previous response was invalid. Return exactly one JSON sentiment object, with no extra keys."
                if attempt
                else ""
            )
            payload = {
                "messages": messages(instruction + retry, text),
                "temperature": self.config["temperature"],
                "seed": self.config["seed"],
                "max_tokens": self.config["max_new_tokens"],
                "stream": False,
                "cache_prompt": False,
                "response_format": {
                    "type": "json_schema",
                    "schema": SentimentOutput.model_json_schema(),
                },
            }
            response = self.session.post(
                self.url + "/v1/chat/completions", json=payload, timeout=120
            )
            response.raise_for_status()
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            try:
                prediction = parse_output(raw)
            except (ValueError, TypeError):
                prediction = None
            attempts.append(
                {"raw_output": raw, "valid": prediction is not None, "usage": body["usage"]}
            )
            if prediction is not None:
                break
        return {
            "prediction": prediction,
            "json_valid": prediction is not None,
            "first_attempt_valid": attempts[0]["valid"],
            "invalid_outputs": sum(not a["valid"] for a in attempts),
            "attempts": len(attempts),
            "raw_outputs": json.dumps(attempts, ensure_ascii=False),
            "input_tokens": sum(a["usage"]["prompt_tokens"] for a in attempts),
            "output_tokens": sum(a["usage"]["completion_tokens"] for a in attempts),
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

    def __exit__(self, *args):
        self.stop.set()
        if hasattr(self, "monitor"):
            self.monitor.join(timeout=5)
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if hasattr(self, "log"):
            self.log.close()
        self.session.close()
