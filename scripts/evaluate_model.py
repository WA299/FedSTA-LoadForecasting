"""Evaluate one selected best checkpoint on the test split."""

import argparse

from src.training.engine import evaluate


def main():
    parser = argparse.ArgumentParser(description="Independent evaluation of best.pt on test data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-batches", type=int, help="Smoke evaluation only; not a formal test")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    evaluate(args.checkpoint, args.output_dir, args.device, args.max_batches)


if __name__ == "__main__":
    main()
