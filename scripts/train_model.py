"""PyCharm terminal: python -m scripts.train_model --model stfednet --client lv_39."""

import argparse

from src.models.forecasting import MODEL_NAMES
from src.training.engine import load_checkpoint, read_config, train


def main():
    parser = argparse.ArgumentParser(description="Local forecasting training; uses train and validation only")
    parser.add_argument("--config", default="configs/g1.yaml")
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--client", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", help="Path to latest.pt; defaults to the saved config/model/client")
    args = parser.parse_args()
    if args.resume:
        saved = load_checkpoint(args.resume)
        config = dict(saved["config"])
        model, client = args.model or saved["model_name"], args.client or saved["client_id"]
    else:
        config = read_config(args.config)
        model, client = args.model or "stfednet", args.client or "lv_39"
    for key in ("epochs", "batch_size", "learning_rate", "seed", "patience", "device",
                "max_train_batches", "max_val_batches"):
        value = getattr(args, key)
        if value is not None:
            if isinstance(value, (float, int)) and key != "seed" and value <= 0:
                parser.error(f"--{key.replace('_', '-')} must be positive")
            config[key] = value
    train(config, model, client, args.run_dir, args.resume)


if __name__ == "__main__":
    main()
