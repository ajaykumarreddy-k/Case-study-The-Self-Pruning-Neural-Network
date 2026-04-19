import torch
import torch.nn as nn
from model import SelfPruningNet
import torch.optim as optim

model = SelfPruningNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for m in model.modules():
    if hasattr(m, 'gate_scores'):
        print("Initial gate mean:", m.gate_scores.mean().item())
        break

# Simulate 1 step of phase 2 exactly as in training loop
sp_loss = model.calculate_sparsity_penalty()
loss = sp_loss * 5.0
loss.backward()

for m in model.modules():
    if hasattr(m, 'gate_scores'):
        print("Gate grad mean:", m.gate_scores.grad.mean().item())
        break

optimizer.step()

for m in model.modules():
    if hasattr(m, 'gate_scores'):
        print("After 1 step gate mean:", m.gate_scores.mean().item())
        break
