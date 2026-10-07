"""python -m endpoint_routing_strategy.data prepare|validate"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .pipeline import prepare_dataset
from .validation import validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate routing request datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--traffic", type=Path, default=Path("历史性能数据包/流量记录_历史性能.jsonl"))
    prepare.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiments/configs/data_preparation.json")
    prepare.add_argument("--output-dir", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            output = args.output_dir or Path(__file__).resolve().parents[1] / "experiments/output" / f"data_{datetime.now():%Y%m%d_%H%M%S_%f}"
            result = prepare_dataset(args.traffic, args.config, output)
            print(f"Dataset: {output.resolve()}")
        else:
            result = validate_dataset(args.dataset_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Data processing failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
