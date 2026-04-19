"""
Self-Pruning Neural Network — Training & Evaluation
====================================================
Author: Krishnareddy Gari Ajay Kumar Reddy
Case Study: Tredence AI Engineering Internship

Two-Phase Training Strategy (fixes the 0% sparsity problem):
─────────────────────────────────────────────────────────────
  Phase 1 — WARMUP (epochs 1 .. warmup_epochs, λ=0):
    Train ONLY with CrossEntropy. The network converges and important
    connections develop large weight magnitudes. Unimportant connections
    stay small. Gate scores barely move during this phase.

  Phase 2 — PRUNE (epochs warmup+1 .. num_epochs, full λ):
    Add the L1 sparsity penalty. For gates attached to small (useless)
    weights, the CE gradient is near-zero, so sparsity wins →
    gate_score → -∞ → gate → 0 → connection pruned.
    For gates on large (useful) weights, CE fights back and the gate
    stays open. This creates the bimodal gate distribution.

Why warmup is necessary (Adam-specific):
    Adam normalises gradient magnitude, so in single-phase training the
    sparsity term cannot consistently dominate — CE noise inflates the
    second moment and drowns out the sparsity signal for useless gates.
    After warmup the CE gradient of an unimportant gate is near-zero AND
    consistent, so Adam gives a clean −lr step every batch → pruned.

Hybrid Loss:
    Total Loss = CrossEntropy(logits, labels)
               + λ · sum(sigmoid(gate_scores))   [Phase 2 only]

Usage (standalone):
    uv run train_single_file.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. CORE ARCHITECTURE & GATES
# ──────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """Custom linear layer with learnable sigmoid gates."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 0.0)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    def __repr__(self):
        return f"PrunableLinear(in={self.in_features}, out={self.out_features})"


class SelfPruningNet(nn.Module):
    """Feed-forward MLP for CIFAR-10 using PrunableLinear layers."""
    def __init__(self, input_dim=3072, hidden_dims=None, num_classes=10):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers.append(PrunableLinear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(PrunableLinear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))

    def calculate_sparsity_penalty(self):
        """Raw L1 sum of gate values."""
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total += torch.sigmoid(m.gate_scores).sum()
        return total

    def get_all_gates(self):
        return [m.get_gates() for m in self.modules() if isinstance(m, PrunableLinear)]


# ──────────────────────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128, data_dir: str = "./data"):
    """Download CIFAR-10 and return (train_loader, test_loader)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        )
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────
# 3. TRAINING & EVALUATION LOOPS
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    device, lambda_sparsity) -> tuple:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits  = model(images)
        ce_loss = criterion(logits, labels)

        if lambda_sparsity > 0:
            sp_loss = model.calculate_sparsity_penalty()
            loss    = ce_loss + lambda_sparsity * sp_loss
        else:
            loss    = ce_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = logits.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc":  f"{100. * correct / total:.1f}%",
        })

    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels  = images.to(device), labels.to(device)
            logits           = model(images)
            running_loss    += criterion(logits, labels).item()
            _, predicted     = logits.max(1)
            total           += labels.size(0)
            correct         += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def calculate_sparsity(model, threshold: float = 0.01) -> float:
    total, pruned = 0, 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates   = module.get_gates()
                total  += gates.numel()
                pruned += (gates < threshold).sum().item()
    return pruned / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

def plot_gate_distribution(model, lambda_val: float, save_path: str):
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().cpu().flatten().numpy())
    all_gates    = np.concatenate(all_gates)
    sparsity_pct = (all_gates < 0.01).sum() / len(all_gates) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_gates, bins=100, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5,
               label="Prune threshold (0.01)")
    ax.set_title(
        f"Gate Distribution  |  λ={lambda_val}  |  Sparsity={sparsity_pct:.1f}%",
        fontsize=13, fontweight="bold")
    ax.set_xlabel("Gate Value  (0 = pruned,  1 = active)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history: dict, lambda_val: float, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    warmup_end = None
    if "phase" in history:
        phases = history["phase"]
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                warmup_end = i + 1   # 1-indexed epoch where prune starts
                break

    def mark_transition(ax):
        if warmup_end is not None:
            ax.axvline(x=warmup_end, color="orange", linestyle=":",
                       linewidth=1.5, label="Prune starts")

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"],  label="Test")
    mark_transition(axes[0])
    axes[0].set_title(f"Loss  (λ={lambda_val})", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["test_acc"],  label="Test")
    mark_transition(axes[1])
    axes[1].set_title(f"Accuracy  (λ={lambda_val})", fontsize=12)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    axes[2].plot(epochs, [s * 100 for s in history["sparsity"]],
                 color="green", label="Sparsity")
    mark_transition(axes[2])
    axes[2].set_title(f"Sparsity  (λ={lambda_val})", fontsize=12)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Sparsity (%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 5. STANDALONE MAIN ORCHESTRATION
# ──────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    EPOCHS        = 25   # 10 warmup + 15 prune
    WARMUP_EPOCHS = 10
    BATCH_SIZE    = 128
    LR            = 1e-3
    LAMBDA_VALUES = [1e-6, 1e-5, 1e-4]

    print("Loading CIFAR-10 …")
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    results   = []
    os.makedirs("assets",      exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for lam in LAMBDA_VALUES:
        print(f"\n{'═' * 55}")
        print(f"  λ = {lam}  (warmup {WARMUP_EPOCHS} epochs → prune {EPOCHS - WARMUP_EPOCHS} epochs)")
        print(f"{'═' * 55}")

        model = SelfPruningNet().to(device)
        gate_params   = [p for n, p in model.named_parameters() if "gate" in n]
        weight_params = [p for n, p in model.named_parameters() if "gate" not in n]

        optimizer = optim.Adam([
            {"params": weight_params, "lr": LR},
            {"params": gate_params,   "lr": 0.05}
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        history = {
            "train_loss": [], "train_acc": [],
            "test_loss":  [], "test_acc":  [],
            "sparsity":   [], "phase":     [],
        }

        for epoch in range(1, EPOCHS + 1):
            lam_epoch   = 0.0 if epoch <= WARMUP_EPOCHS else lam
            phase_label = "warmup" if epoch <= WARMUP_EPOCHS else "prune"

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, lam_epoch)
            te_loss, te_acc = evaluate(model, test_loader, criterion, device)
            sparsity        = calculate_sparsity(model)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)
            history["sparsity"].append(sparsity)
            history["phase"].append(phase_label)

            if epoch % 5 == 0 or epoch == 1 or epoch == WARMUP_EPOCHS + 1:
                print(
                    f"  [{phase_label:6s}] Epoch {epoch:02d}/{EPOCHS} | "
                    f"Loss: {tr_loss:.4f} | "
                    f"Test Acc: {te_acc:.2f}% | "
                    f"Sparsity: {sparsity * 100:.2f}%"
                )

        final_acc      = history["test_acc"][-1]
        final_sparsity = history["sparsity"][-1]
        results.append((lam, final_acc, final_sparsity * 100))

        # Save outputs
        exp = f"lam_{lam}"
        plot_gate_distribution(model, lam, f"assets/gates_{exp}.png")
        plot_training_history(history, lam,  f"assets/history_{exp}.png")
        torch.save(model.state_dict(), f"checkpoints/spnn_{exp}.pth")

    # Results Table
    print(f"\n{'═' * 55}")
    print(f"  {'Lambda':<10} {'Test Accuracy':>15} {'Sparsity':>12}")
    print(f"  {'-' * 10} {'-' * 15} {'-' * 12}")
    for lam, acc, spar in results:
        print(f"  {lam:<10} {acc:>14.2f}% {spar:>11.2f}%")
    print(f"{'═' * 55}")
    print("\nDone. See assets/ for plots and checkpoints/ for saved models.")

if __name__ == "__main__":
    main()
