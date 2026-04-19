"""
Utility Functions — Metrics and Visualisation
==============================================
Provides:
  - calculate_sparsity  : fraction of gates below threshold
  - get_gate_stats      : diagnostic summary of gate values
  - plot_gate_distribution  : histogram of gate values
  - plot_training_history   : loss / accuracy / sparsity curves
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────
# 1. SPARSITY METRICS
# ──────────────────────────────────────────────────────────────

def calculate_sparsity(model, threshold: float = 0.01) -> float:
    """
    Return the fraction of gate values that fall below *threshold*.

    A gate value < threshold means that weight is effectively pruned
    (contributes near-zero signal to the next layer).

    Args:
        model     : Any model with PrunableLinear layers (has `get_gates()`).
        threshold : Pruning threshold (default 0.01).

    Returns:
        Float in [0.0, 1.0].  E.g. 0.72 → 72 % of weights pruned.
    """
    total_weights  = 0
    pruned_weights = 0

    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "get_gates"):
                gates           = module.get_gates()
                total_weights  += gates.numel()
                pruned_weights += (gates < threshold).sum().item()

    return pruned_weights / total_weights if total_weights > 0 else 0.0


def get_gate_stats(model) -> dict:
    """
    Return a diagnostic summary of gate values across all prunable layers.

    Returns a dict with keys:
        min, max, mean, std, small_count (gates < 0.1), total_count
    """
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "get_gates"):
                all_gates.append(module.get_gates().flatten())

    if not all_gates:
        return {"min": 0, "max": 0, "mean": 0, "std": 0,
                "small_count": 0, "total_count": 0}

    all_gates = torch.cat(all_gates)
    return {
        "min":         all_gates.min().item(),
        "max":         all_gates.max().item(),
        "mean":        all_gates.mean().item(),
        "std":         all_gates.std().item(),
        "small_count": (all_gates < 0.1).sum().item(),
        "total_count": all_gates.numel(),
    }


# ──────────────────────────────────────────────────────────────
# 2. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

def plot_gate_distribution(model, save_path: str = "gate_distribution.png"):
    """
    Histogram of all gate values across every prunable layer.

    A well-pruned model shows a **bimodal** distribution:
      - Large spike near 0   → pruned (dead) connections
      - Smaller cluster near 1 → active connections

    Args:
        model     : Model with `get_gates()` on PrunableLinear layers.
        save_path : File path for the saved PNG.
    """
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "get_gates"):
                gates = module.get_gates().cpu().numpy().flatten()
                all_gates.extend(gates)

    all_gates    = np.array(all_gates)
    sparsity_pct = (all_gates < 0.01).sum() / len(all_gates) * 100

    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    plt.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5,
                label="Prune threshold (0.01)")
    plt.title(
        f"Gate Value Distribution  |  Sparsity = {sparsity_pct:.1f}%",
        fontsize=13, fontweight="bold")
    plt.xlabel("Gate Value  (0 = pruned,  1 = active)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gate distribution plot saved → {save_path}")


def plot_training_history(history: dict, save_path: str = "training_history.png"):
    """
    Three-panel training summary: Loss | Accuracy | Sparsity over epochs.

    Args:
        history   : Dict with keys train_loss, test_loss, train_acc,
                    test_acc, sparsity (all lists of per-epoch values).
        save_path : File path for the saved PNG.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Loss ──────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["test_loss"],  label="Test Loss")
    axes[0].set_title("Loss vs. Epochs", fontsize=12)
    axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    # ── Accuracy ──────────────────────────────────────────────
    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["test_acc"],  label="Test Acc")
    axes[1].set_title("Accuracy vs. Epochs", fontsize=12)
    axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    # ── Sparsity ──────────────────────────────────────────────
    axes[2].plot(
        epochs,
        [s * 100 for s in history["sparsity"]],
        label="Sparsity", color="green",
    )
    axes[2].set_title("Sparsity vs. Epochs", fontsize=12)
    axes[2].set_xlabel("Epochs"); axes[2].set_ylabel("Sparsity (%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved → {save_path}")
