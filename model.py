"""
Self-Pruning Neural Network — Core Architecture
================================================
Author: Krishnareddy Gari Ajay Kumar Reddy
Case Study: Tredence AI Engineering Internship

Contains:
  - PrunableLinear : custom linear layer with learnable sigmoid gates
  - SelfPruningNet : MLP built from PrunableLinear layers for CIFAR-10
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# 1. PRUNABLE LINEAR LAYER
# ──────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom Linear layer with learnable sigmoid gates.

    For each weight w_ij, there is a corresponding gate score g_ij.
    The effective (pruned) weight is:

        pruned_weight = w_ij * sigmoid(g_ij)

    Training dynamics:
      - CrossEntropy pushes weights to learn good representations.
      - L1 sparsity penalty on sigmoid(g) drives gate values toward 0.
      - Gates near 0  → connection is effectively pruned.
      - Gates near 1  → connection is preserved.

    Initialization:
      - gate_scores are initialised to 0.0 → sigmoid(0)=0.5 (neutral start).
      - This avoids the "all gates already active" bias of high-init schemes.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias (same layout as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Gate scores — same shape as weight; learned during back-prop
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    # ----------------------------------------------------------
    def _reset_parameters(self) -> None:
        """Kaiming-uniform weights (matches nn.Linear), zero gate scores."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Neutral init: sigmoid(0) = 0.5 → not biased to open or closed
        nn.init.constant_(self.gate_scores, 0.0)

    # ----------------------------------------------------------
    def get_gates(self) -> torch.Tensor:
        """Return gate values in (0, 1). **Detached** — use for metrics only.

        For loss computation, call ``torch.sigmoid(module.gate_scores)`` directly
        so that gradients flow back to ``gate_scores``.
        """
        return torch.sigmoid(self.gate_scores).detach()

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sigmoid on gate_scores keeps gradient alive (not detached!)
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"PrunableLinear(in={self.in_features}, "
            f"out={self.out_features}, bias={self.bias is not None})"
        )


# ──────────────────────────────────────────────────────────────
# 2. SELF-PRUNING NETWORK
# ──────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward MLP for CIFAR-10 image classification using PrunableLinear.

    Architecture (defaults):
      Input  : 3 × 32 × 32 = 3072
      Hidden : 1024 → 512 → 256  (each followed by ReLU)
      Output : 10 class logits

    Training uses:
      Loss = CrossEntropy(logits, labels) + λ · sparsity_penalty()

    where sparsity_penalty() is the normalized mean of all gate values
    (always in (0, 1) regardless of model size, so λ is scale-stable).
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: list = None,
        num_classes: int = 10,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(PrunableLinear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(PrunableLinear(prev, num_classes))

        self.network = nn.Sequential(*layers)

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # [B, 3, 32, 32] → [B, 3072]
        return self.network(x)        # raw logits [B, 10]

    # ----------------------------------------------------------
    def calculate_sparsity_penalty(self) -> torch.Tensor:
        """
        Raw L1 penalty — sum of all gate values across all PrunableLinear layers.

        Uses ``torch.sigmoid(module.gate_scores)`` directly (not ``.get_gates()``)
        so the gradient flows back to gate_scores during backward().

        WHY RAW SUM (not normalized mean):
          Per-gate gradient = λ × sigmoid'(g).  With normalization ÷N (N≈3.9M),
          the sparsity gradient becomes λ×0.25/3.9M ≈ 6e-8 at λ=1, which is
          ~10,000× smaller than the typical CE gradient (≈1e-3).  Gates never
          move.  Using the raw sum keeps the per-gate gradient at λ×0.25,
          which is meaningful at λ ∈ [1e-4, 1e-2].

        Use λ values in range [1e-4, 1e-2] with this formulation.
        """
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)   # gradient-attached
                total += gates.sum()
        return total   # raw sum; use small λ (1e-4 to 1e-2)

    # ----------------------------------------------------------
    def get_all_gates(self) -> list:
        """Return list of detached gate tensors from every PrunableLinear."""
        return [
            m.get_gates()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        ]

    # ----------------------------------------------------------
    def count_parameters(self) -> dict:
        """Return total and gate parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        gate_count = sum(
            m.gate_scores.numel()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        )
        weight_count = sum(
            m.weight.numel()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        )
        return {
            "total_params": total,
            "weight_params": weight_count,
            "gate_params": gate_count,
        }


# Keep backward-compatible alias for old scripts
SelfPruningNN = SelfPruningNet
