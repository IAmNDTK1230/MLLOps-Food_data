import argparse
import json

from src.food.config import FoodConfig
from src.proposed_v1.reporting import complete_saved


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct complete saved Qwen reports, never predict"
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--config", default="qwen_v1.yaml")
    parser.add_argument("--probe-loaded-memory", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            complete_saved(FoodConfig.load(args.params), args.config, args.probe_loaded_memory),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
