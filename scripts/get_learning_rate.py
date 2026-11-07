import argparse
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def get_learning_rate(config, model, split, method):
    choices = config
    for label, key in (("model", model), ("split", split), ("method", method)):
        if key not in choices:
            raise ValueError(f"Unknown {label}: {key}. Available: {', '.join(choices)}")
        choices = choices[key]
    if isinstance(choices, bool):
        raise ValueError(f"Invalid learning rate: {choices}")
    value = float(choices)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid learning rate: {choices}")
    for group, name in (("model", model), ("trainer", method)):
        if not (ROOT / "configs" / group / f"{name}.yaml").is_file():
            raise ValueError(f"Missing {group} config: {name}")
    return str(choices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?")
    parser.add_argument("split", nargs="?")
    parser.add_argument("method", nargs="?")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    with (ROOT / "scripts" / "paper_lrs.yaml").open() as file:
        config = yaml.safe_load(file)
    if args.list:
        if any((args.model, args.split, args.method)):
            parser.error("--list takes no positional arguments")
        print("model\tsplit\tmethod\tlr")
        for model, splits in config.items():
            for split, methods in splits.items():
                for method in methods:
                    print(model, split, method, get_learning_rate(config, model, split, method), sep="\t")
        return
    if not all((args.model, args.split, args.method)):
        parser.error("provide MODEL SPLIT METHOD, or --list")
    try:
        print(get_learning_rate(config, args.model, args.split, args.method))
    except (TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
