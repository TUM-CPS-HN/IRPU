"""Split-conformal calibration of an HProbZ head (Eq. (6), Theorem 6) on random data.

Usage: python examples/conformal_calibration.py
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without installing
from hprobz import HProbZHead, conformal_quantile, hprobz_score, prediction_boxes  # noqa: E402

T, D, d_model = 12, 2, 128  # prediction steps, coordinates per step, feature size
encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8 * 2, d_model))
head = HProbZHead(d_model, horizon=T, dim=D, n_b=1)  # in practice, a trained head

cal_past, cal_future = torch.randn(500, 8, 2), torch.randn(500, T, D)    # calibration pairs, not used for training
test_past, test_future = torch.randn(100, 8, 2), torch.randn(100, T, D)  # test pairs

with torch.no_grad():
    q = conformal_quantile(hprobz_score(head(encoder(cal_past)), cal_future), alpha=0.1)
    out = head(encoder(test_past))
    covered = hprobz_score(out, test_future) <= q   # P(covered) >= 1 - alpha by Theorem 6
    centers, half_width = prediction_boxes(out, q)  # the 2^n_b boxes of the prediction set

print(f"q = {q.item():.4f}, empirical coverage = {covered.float().mean().item():.3f}")
print(f"box centers {tuple(centers.shape)}, half-widths {tuple(half_width.shape)}")
