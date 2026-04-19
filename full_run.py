"""
Full Hyperparameter Sweep — Two-Phase Training, 3 Lambda Values
================================================================
Runs three independent experiments with λ ∈ {0.1, 1.0, 5.0}.

Each experiment uses:
  Phase 1 (10 warmup epochs, λ=0):  CE-only training — network converges.
  Phase 2 (15 prune  epochs, full λ): Sparsity penalty added — unimportant
    gates driven toward 0, important gates resist via strong CE gradient.

Saves to:
  - assets/history_lam_<λ>.png  — training curves + warmup/prune boundary
  - assets/gates_lam_<λ>.png    — gate distribution histogram
  - checkpoints/spnn_lam_<λ>.pth — model checkpoint

This script reproduces the results in report.md.

Usage:
    uv run full_run.py
    python full_run.py
"""

import os
import torch

from model   import SelfPruningNet
from dataset import get_dataloaders
from train   import run_training, plot_gate_distribution, plot_training_history
from utils   import calculate_sparsity


def run_experiments():
    # ── Config ────────────────────────────────────────────────
    EPOCHS         = 25   # 10 warmup + 15 prune
    WARMUP_EPOCHS  = 10
    BATCH_SIZE     = 128
    LEARNING_RATE  = 1e-3
    LAMBDA_VALUES  = [1e-6, 1e-5, 1e-4]   # raw L1 sum; two-phase makes these effective

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print(f"Training schedule: {WARMUP_EPOCHS} warmup + "
          f"{EPOCHS - WARMUP_EPOCHS} prune = {EPOCHS} total epochs\n")

    # ── Shared data loaders ───────────────────────────────────
    print("Loading CIFAR-10 …")
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    os.makedirs("assets",      exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    results = []

    for lam in LAMBDA_VALUES:
        print(f"\n{'=' * 55}")
        print(f"  Experiment: λ = {lam}  "
              f"(warmup {WARMUP_EPOCHS} → prune {EPOCHS - WARMUP_EPOCHS})")
        print(f"{'=' * 55}")

        # Fresh model for each lambda
        model = SelfPruningNet(
            input_dim=3072,
            hidden_dims=[1024, 512, 256],
            num_classes=10,
        ).to(device)

        history = run_training(
            model, train_loader, test_loader, device,
            num_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lambda_sparsity=lam,
            warmup_epochs=WARMUP_EPOCHS,
        )

        final_sparsity = calculate_sparsity(model)
        final_acc      = history["test_acc"][-1]
        results.append((lam, final_acc, final_sparsity * 100))

        print(f"\n  → Final Test Accuracy : {final_acc:.2f}%")
        print(f"  → Final Sparsity      : {final_sparsity * 100:.2f}%")

        # Save experiment artefacts
        exp = f"lam_{lam}"
        plot_training_history(history, lam,  save_path=f"assets/history_{exp}.png")
        plot_gate_distribution(model,   lam,  save_path=f"assets/gates_{exp}.png")
        torch.save(model.state_dict(),        f"checkpoints/spnn_{exp}.pth")

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print(f"  {'Lambda':<10} {'Test Accuracy':>15} {'Sparsity':>12}")
    print(f"  {'-' * 10} {'-' * 15} {'-' * 12}")
    for lam, acc, spar in results:
        print(f"  {lam:<10} {acc:>14.2f}% {spar:>11.2f}%")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    run_experiments()
