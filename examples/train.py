"""One training step of an HProbZ head (Algorithm 1) on random data.

Usage: python examples/train.py
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without installing
from hprobz import HProbZHead, hprobz_nll  # noqa: E402

T, D, d_model = 12, 2, 128  # prediction steps, coordinates per step, feature size
encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8 * 2, d_model))  # any encoder with output (B, d_model)
head = HProbZHead(d_model, horizon=T, dim=D, n_b=1)
params = [*encoder.parameters(), *head.parameters()]
optimizer = torch.optim.AdamW(params)  # hyperparameters used in the paper: Table 7 (Appendix A.2)

past = torch.randn(32, 8, 2)     # a batch of inputs
future = torch.randn(32, T, D)   # the corresponding targets

loss = hprobz_nll(head(encoder(past)), future).mean()
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(params, 1.0)
optimizer.step()
print(f"loss = {loss.item():.4f}")
