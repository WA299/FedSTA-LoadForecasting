"""Plan and execute the immutable, inference-only final TEST suite."""

import argparse
import json
from pathlib import Path

from src.training.final_evaluation import atomic_json, build_manifest, execute_manifest, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", default="configs/final_test_manifest.json")
    parser.add_argument("--output-dir", default="results/final_test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if args.plan_only:
        if manifest_path.exists() or (Path(args.output_dir) / "execution_state.json").exists():
            parser.error("A frozen manifest/ledger already exists; it must not be silently replaced")
        manifest = build_manifest()
        verify_manifest(manifest, Path.cwd())
        atomic_json(manifest_path, manifest)
        print(f"[Frozen] {len(manifest['entries'])} model/client combinations; {len(manifest['missing'])} missing; no source data read")
        print("[Manifest] " + str(manifest_path.resolve()))
        for method in sorted({e['method'] for e in manifest['entries']}):
            rows = [e for e in manifest['entries'] if e['method'] == method]
            print(f"  {method}: {len(rows)} clients; validation WAPE=" + ", ".join(f"{e['client_id']}:{e['validation_metrics']['wape_percent']:.4f}%" for e in rows))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        execute_manifest(manifest, output=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
