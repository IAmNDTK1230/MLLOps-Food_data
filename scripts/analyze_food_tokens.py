"""Run measured food EDA/token/chunk analysis; no model weights or embeddings."""

import argparse
from pathlib import Path

from src.ml.cli import run_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--tokenizer-file", type=Path)
    args = parser.parse_args()
    args.command = "food-analysis"
    return run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
