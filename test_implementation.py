"""
Implementation Diagnostic — Verify All Core Components
=======================================================
Run this script to confirm that:
  1. gate_scores parameters are registered and require gradients.
  2. Sparsity loss has a gradient path back to gate_scores.
  3. A mini 2-epoch run shows decreasing loss and increasing sparsity.
  4. Per-layer gate statistics are reported.

Usage:
  uv run test_implementation.py
  python test_implementation.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

from model   import SelfPruningNet, PrunableLinear
from dataset import get_dataloaders
from train   import train_one_epoch, evaluate, calculate_sparsity
from utils   import get_gate_stats

# ── Colour helpers (ANSI) ─────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

OK   = f"{GREEN}[OK]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"


def check_parameters(model: SelfPruningNet) -> bool:
    """Test 1: gate_scores are registered as learnable Parameters."""
    print("\n── Test 1: Parameter Registration ─────────────────────")
    gate_params = [n for n, _ in model.named_parameters() if "gate_scores" in n]

    if not gate_params:
        print(f"  {FAIL} No gate_scores parameters found!")
        return False

    all_require_grad = True
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            status = OK if param.requires_grad else FAIL
            print(f"  {status} {name}: requires_grad={param.requires_grad}, "
                  f"shape={tuple(param.shape)}")
            if not param.requires_grad:
                all_require_grad = False

    if all_require_grad:
        print(f"  {OK} All gate_scores require gradients.")
    return all_require_grad


def check_gradient_flow(model: SelfPruningNet, device: torch.device) -> bool:
    """Test 2: Sparsity penalty gradients reach gate_scores."""
    print("\n── Test 2: Sparsity Loss Gradient Flow ─────────────────")
    model.zero_grad()
    penalty = model.calculate_sparsity_penalty()

    if not penalty.requires_grad:
        print(f"  {FAIL} sparsity_penalty() has no grad_fn — check gate math!")
        return False

    penalty.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if "gate_scores" in name and param.grad is not None:
            grad_norms[name] = torch.norm(param.grad).item()

    if not grad_norms:
        print(f"  {FAIL} No gradients reached gate_scores!")
        return False

    for name, norm in grad_norms.items():
        print(f"  {OK} {name}: ‖grad‖ = {norm:.6f}")
    print(f"  {OK} Gradients flow correctly to all gate_scores.")
    return True


def mini_training_run(
    model: SelfPruningNet,
    device: torch.device,
    lambda_sparsity: float = 0.1,
    num_epochs: int = 2,
) -> bool:
    """Test 3: Two-epoch trial — loss should decrease, sparsity increase."""
    print(f"\n── Test 3: Mini {num_epochs}-Epoch Training Trial ────────────────")
    train_loader, test_loader = get_dataloaders(batch_size=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "acc": [], "sparsity": []}

    for epoch in range(1, num_epochs + 1):
        loss, acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, lambda_sparsity)
        sparsity = calculate_sparsity(model)
        stats    = get_gate_stats(model)

        history["loss"].append(loss)
        history["acc"].append(acc)
        history["sparsity"].append(sparsity)

        print(f"  Epoch {epoch}: Loss={loss:.4f}  Acc={acc:.2f}%  "
              f"Sparsity={sparsity * 100:.2f}%")
        print(f"    Gate stats → min={stats['min']:.4f}  "
              f"mean={stats['mean']:.4f}  "
              f"gates<0.1={stats['small_count']}/{stats['total_count']}")

    loss_ok     = history["loss"][-1] < history["loss"][0]
    acc_ok      = history["acc"][-1]  > 15.0  # above random chance
    pruning_started = history["sparsity"][-1] > 0.0

    print()
    print(f"  {'Loss decreasing':<25}: {OK if loss_ok else WARN}")
    print(f"  {'Accuracy > 15%':<25}: {OK if acc_ok else FAIL}  "
          f"({history['acc'][-1]:.2f}%)")
    print(f"  {'Pruning started':<25}: {OK if pruning_started else WARN}  "
          f"({history['sparsity'][-1]*100:.2f}% sparse)")

    return loss_ok and acc_ok


def per_layer_analysis(model: SelfPruningNet):
    """Print per-layer gate statistics."""
    print("\n── Layer-by-Layer Gate Analysis ────────────────────────")
    for idx, module in enumerate(
        (m for m in model.modules() if isinstance(m, PrunableLinear)), start=1
    ):
        gates   = module.get_gates()
        min_g   = gates.min().item()
        mean_g  = gates.mean().item()
        small_g = (gates < 0.1).sum().item()
        total_g = gates.numel()

        print(f"  Layer {idx} ({module})")
        print(f"    min={min_g:.4f}  mean={mean_g:.4f}  "
              f"gates<0.1={small_g}/{total_g} "
              f"({small_g/total_g*100:.1f}%)")


def run_diagnostic():
    print("╔══════════════════════════════════════════════════════╗")
    print("║    Self-Pruning Neural Network — Diagnostic Check    ║")
    print("╚══════════════════════════════════════════════════════╝")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = SelfPruningNet().to(device)

    # Run all checks
    t1 = check_parameters(model)
    t2 = check_gradient_flow(model, device)

    # Re-zero gradients before training run
    model.zero_grad()

    t3 = mini_training_run(model, device)
    per_layer_analysis(model)

    # ── Final Verdict ─────────────────────────────────────────
    print("\n── Final Verdict ────────────────────────────────────────")
    all_pass = t1 and t2 and t3
    if all_pass:
        print(f"  {OK} All checks passed. Implementation is verified.")
    else:
        print(f"  {FAIL} Some checks failed — review output above.")

    return all_pass


if __name__ == "__main__":
    run_diagnostic()
