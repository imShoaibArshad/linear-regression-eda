"""Entry point: python run.py [--eda | --model | --all (default)]"""

import argparse
import sys

from src import explore, model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="California housing: EDA and a linear-regression baseline."
    )
    parser.add_argument("--eda", action="store_true", help="run exploratory data analysis")
    parser.add_argument("--model", action="store_true", help="train and evaluate the model")
    parser.add_argument("--all", action="store_true", help="run everything (default)")
    args = parser.parse_args()

    # Nothing selected -> run everything.
    run_eda = args.eda or args.all or not (args.eda or args.model)
    run_model = args.model or args.all or not (args.eda or args.model)

    if run_eda:
        explore.run()
        print()
    if run_model:
        model.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
