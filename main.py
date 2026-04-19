"""
Self-Pruning Neural Network — CLI Entry Point
=============================================
Runs a single experiment with the two-phase training schedule.

Usage:
    uv run main.py                                       # defaults
    uv run main.py --epochs 25 --warmup 10 --lambda_sparsity 1.0
    python main.py --no_cuda --epochs 25 --lambda_sparsity 5.0

Output:
    training_history.png   — loss / accuracy / sparsity curves
    gate_distribution.png  — histogram of gate values (should be bimodal)
    checkpoint/spnn_final.pth — saved model state dict
"""

import os
import torch
import argparse

from model   import SelfPruningNet
from dataset import get_dataloaders
from train   import run_training, plot_gate_distribution, plot_training_history, calculate_sparsity


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Self-Pruning Neural Network on CIFAR-10 (two-phase)."
    )
    p.add_argument("--batch_size",      type=int,   default=128,
                   help="Batch size                    (default: 128)")
    p.add_argument("--epochs",          type=int,   default=25,
                   help="Total epochs                  (default: 25)")
    p.add_argument("--warmup",          type=int,   default=10,
                   help="Warmup epochs (λ=0)           (default: 10)")
    p.add_argument("--lr",              type=float, default=1e-3,
                   help="Learning rate                 (default: 1e-3)")
    p.add_argument("--lambda_sparsity", type=float, default=1e-5,
                   help="Sparsity penalty λ            (default: 1e-5)")
    p.add_argument("--hidden_dims",     type=int,   nargs="+",
                   default=[1024, 512, 256],
                   help="Hidden layer dims             (default: 1024 512 256)")
    p.add_argument("--no_cuda",         action="store_true", default=False,
                   help="Disable CUDA")
    p.add_argument("--seed",            type=int,   default=42,
                   help="Random seed                   (default: 42)")
    p.add_argument("--save_dir",        type=str,   default="checkpoint",
                   help="Directory to save model       (default: checkpoint/)")
    return p.parse_args()


def main():
    args   = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu"
    )
    prune_epochs = args.epochs - args.warmup

    print(f"Device          : {device}")
    print(f"Epochs          : {args.epochs}  "
          f"({args.warmup} warmup + {prune_epochs} prune)")
    print(f"λ (sparsity)    : {args.lambda_sparsity}")
    print(f"Hidden dims     : {args.hidden_dims}")
    print()

    # ── Data ──────────────────────────────────────────────────
    print("Loading CIFAR-10 …")
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # ── Model ─────────────────────────────────────────────────
    print(f"Building SelfPruningNet {args.hidden_dims} …")
    model = SelfPruningNet(
        input_dim=3072,
        hidden_dims=args.hidden_dims,
        num_classes=10,
    ).to(device)

    param_info = model.count_parameters()
    print(f"  Total params  : {param_info['total_params']:,}")
    print(f"  Weight params : {param_info['weight_params']:,}")
    print(f"  Gate params   : {param_info['gate_params']:,}\n")

    # ── Training ──────────────────────────────────────────────
    print(f"Training with λ={args.lambda_sparsity} …")
    history = run_training(
        model, train_loader, test_loader, device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lambda_sparsity=args.lambda_sparsity,
        warmup_epochs=args.warmup,
    )

    # ── Results ───────────────────────────────────────────────
    final_sparsity = calculate_sparsity(model)
    final_acc      = history["test_acc"][-1]
    print(f"\nFinal Test Accuracy : {final_acc:.2f}%")
    print(f"Final Sparsity      : {final_sparsity * 100:.2f}%")

    # ── Plots ─────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_training_history(history, args.lambda_sparsity, "training_history.png")
    plot_gate_distribution(model, args.lambda_sparsity, "gate_distribution.png")

    # ── Save checkpoint ───────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "spnn_final.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved → {ckpt_path}")


if __name__ == "__main__":
    main()
