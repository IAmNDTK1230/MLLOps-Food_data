"""Phase 2 restaurant preparation and explicit text-only baseline commands."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.food.config import FoodConfig

DISABLED = {
    "import-ml-data",
    "ml-prepare",
    "ml-split",
    "ml-train",
    "train-baseline",
    "ml-evaluate",
    "register-model",
}
BASELINES = {"baseline-rule", "baseline-tfidf", "baseline-evaluate", "run-baselines"}
PROPOSED = {"proposed-v1"}
PROPOSED_V2 = {"proposed-v2", "proposed-v2-prepare"}
ASPECT = {"proposed-v2-aspect", "proposed-v2-aspect-prepare"}


def add_commands(commands) -> set[str]:
    names = {"food-data-import", "food-prepare", "food-split", "food-analysis", "food-pipeline"}
    for name in sorted(names):
        sub = commands.add_parser(name, help="Vietnamese restaurant data preparation (no models)")
        sub.add_argument("--params", default="params.yaml")
        if name in {"food-data-import", "food-pipeline"}:
            sub.add_argument("--dataset", default="vn_msasar_v3", choices=["vn_msasar_v3"])
            sub.add_argument(
                "--input", help="Manual source file or directory of all original split files"
            )
        if name in {"food-analysis", "food-pipeline"}:
            sub.add_argument(
                "--tokenizer-file", type=Path, help="Offline tokenizer.json; no model weights"
            )
    for name in sorted(DISABLED):
        sub = commands.add_parser(name, help="Disabled: Phase 2 is food data preparation only")
        sub.add_argument("--params", default="params.yaml")
    for name in sorted(BASELINES):
        sub = commands.add_parser(name, help="Run text-only baselines on the frozen food split")
        sub.add_argument("--params", default="params.yaml")
    sub = commands.add_parser("proposed-v1", help="Run frozen text-only Qwen prompt experiments")
    sub.add_argument("--params", default="params.yaml")
    sub.add_argument("--qwen-config", default="qwen_v1.yaml")
    for name in sorted(PROPOSED_V2):
        sub = commands.add_parser(
            name, help="Calibrated text-only Qwen V2-A; frozen original experiments"
        )
        sub.add_argument("--params", default="params.yaml")
        sub.add_argument("--qwen-config", default="qwen_v2.yaml")
    for name in sorted(ASPECT):
        sub = commands.add_parser(
            name, help="Distinct text+aspect experiment with classical control"
        )
        sub.add_argument("--params", default="params.yaml")
        sub.add_argument("--qwen-config", default="qwen_aspect.yaml")
    return names | DISABLED | BASELINES | PROPOSED | PROPOSED_V2 | ASPECT


def run_command(args) -> int:
    if args.command in DISABLED:
        print(
            "Previous product-review ML commands are disabled. Use food-data-import, food-prepare, food-split and food-analysis. Training/evaluation/registration is not enabled in this phase."
        )
        return 1
    try:
        if args.command in ASPECT:
            from src.aspect_conditioned.runner import run

            result = run(
                FoodConfig.load(args.params),
                args.qwen_config,
                args.command == "proposed-v2-aspect-prepare",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0
        if args.command in PROPOSED_V2:
            from src.proposed_v2.runner import run

            result = run(
                FoodConfig.load(args.params),
                args.qwen_config,
                args.command == "proposed-v2-prepare",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0
        if args.command in PROPOSED:
            from src.proposed_v1.runner import run

            result = run(FoodConfig.load(args.params), args.qwen_config)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0
        if args.command in BASELINES:
            from src.baselines.runner import evaluate, run_all, run_rule, run_tfidf

            action = {
                "baseline-rule": run_rule,
                "baseline-tfidf": run_tfidf,
                "baseline-evaluate": evaluate,
                "run-baselines": run_all,
            }[args.command]
            result = action(FoodConfig.load(args.params))
            if args.command in {"baseline-evaluate", "run-baselines"}:
                from src.baselines.reporting import finalize_evidence

                finalize_evidence(FoodConfig.load(args.params))
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0
        from src.food.analysis import analyze
        from src.food.data import import_dataset, prepare
        from src.food.split import split_dataset

        config = FoodConfig.load(args.params)
        if args.command == "food-data-import":
            result = import_dataset(config, args.dataset, args.input)
        elif args.command == "food-prepare":
            result = prepare(config)
        elif args.command == "food-split":
            result = split_dataset(config)
        elif args.command == "food-analysis":
            result = analyze(config, args.tokenizer_file)
        else:
            import_dataset(config, args.dataset, args.input)
            prepare(config)
            split_dataset(config)
            result = analyze(config, args.tokenizer_file)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except ImportError:
        print(
            "Food preparation dependencies missing; install requirements.txt and requirements-ml.txt"
        )
    except ValidationError:
        print("Food configuration invalid; check the food section in params.yaml")
    except (ValueError, FileNotFoundError) as error:
        print(f"Food preparation stopped: {error}")
    except Exception as error:
        print(
            f"Food preparation failed ({type(error).__name__}); inspect local source/configuration"
        )
    return 1
