"""G3.1 research-only Poisson DP-SGD + FedAvg, validation-only selection."""

import argparse

from src.fl.dp_server import read_g31_config, run_dp_federated
from src.training.engine import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/g3_1.yaml")
    parser.add_argument("--model", choices=("stfednet_gated", "gru_attention"))
    parser.add_argument("--noise-multiplier", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--expected-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--gradient-method", choices=("auto", "loop", "torch_func"))
    parser.add_argument("--privacy-mode", choices=("research", "privacy_claim"))
    parser.add_argument("--max-dp-steps-per-epoch", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--resume", help="latest_dp_global.pt (full round or committed local step)")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    overrides = dict(checkpoint["config"]) if checkpoint else {}
    for key in ("noise_multiplier", "max_grad_norm", "delta", "rounds", "local_epochs",
                "expected_batch_size", "seed", "device", "gradient_method", "privacy_mode",
                "max_dp_steps_per_epoch", "max_val_batches", "output_root"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if overrides.get("noise_multiplier") is None:
        parser.error("--noise-multiplier is required for a new run (0 is an explicitly non-DP control)")
    config = read_g31_config(args.config, overrides)
    model_name = args.model or (checkpoint["model_name"] if checkpoint else "stfednet_gated")
    run = run_dp_federated(config, model_name, resume=args.resume)
    print("[G3.1] Research run (not a deployment privacy claim): " + str(run), flush=True)


if __name__ == "__main__":
    main()
