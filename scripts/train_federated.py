"""Train a G2 FedAvg/FedProx model with validation-only global selection."""

import argparse

from src.fl.common import MODEL_NAMES, read_g2_config
from src.fl.server import run_federated
from src.training.engine import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"))
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--config", default="configs/g2.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mu", type=float)
    parser.add_argument("--output-root")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--resume", help="latest_global.pt from the same run")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    overrides = dict(checkpoint["config"]) if checkpoint else {}
    for key, value in (("device", args.device), ("rounds", args.rounds),
                       ("local_epochs", args.local_epochs), ("mu", args.mu),
                       ("batch_size", args.batch_size), ("learning_rate", args.learning_rate),
                       ("weight_decay", args.weight_decay), ("seed", args.seed),
                       ("output_root", args.output_root)):
        if value is not None:
            overrides[key] = value
    config = read_g2_config(args.config, overrides)
    model_name = args.model or (checkpoint["model_name"] if checkpoint else "stfednet_gated")
    algorithm = args.algorithm or (checkpoint["algorithm"] if checkpoint else "fedavg")
    run = run_federated(config, model_name, algorithm, mu=args.mu,
                        resume=args.resume, max_train_batches=args.max_train_batches,
                        max_val_batches=args.max_val_batches)
    print("[G2] Federated validation-only run: " + str(run), flush=True)


if __name__ == "__main__":
    main()
