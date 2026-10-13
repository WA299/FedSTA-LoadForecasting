"""Scalar-only training diagnostics; never participate in model aggregation."""

DIAGNOSTIC_FIELDS = ("mean_huber_loss", "mean_prox_loss", "mean_total_loss",
                     "mean_grad_norm_before_clip", "clip_fraction")


def summarize_diagnostics(clients):
    """Losses use window weights; gradient/clip statistics use batch weights."""
    windows = sum(row["processed_windows"] for row in clients.values())
    batches = sum(row["batch_count"] for row in clients.values())
    if windows <= 0 or batches <= 0:
        raise ValueError("Diagnostics require positive window and batch counts")
    result = {"mean_" + key: sum(row[key] * row["processed_windows"]
                               for row in clients.values()) / windows
              for key in ("huber_loss", "prox_loss", "total_loss")}
    result["mean_grad_norm_before_clip"] = sum(
        row["grad_norm_before_clip"] * row["batch_count"] for row in clients.values()) / batches
    result["clip_fraction"] = sum(row["clipped_batch_count"] for row in clients.values()) / batches
    return result
