"""Pool one model's five client test results, including equal-client WAPE."""

import argparse
import json
from pathlib import Path

from src.training.engine import write_json
from src.training.metrics import aggregate_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True, help="One metrics.json per client")
    parser.add_argument("--output", default="results/g1/summary.json")
    args = parser.parse_args()
    summary = aggregate_results([json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metrics])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
