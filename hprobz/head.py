"""HProbZ output head of Algorithm 1 (Appendix A.6)."""

import torch
import torch.nn as nn

__all__ = ["HProbZHead"]


class HProbZHead(nn.Module):
    """HProbZ output head of Algorithm 1.

    A two-layer MLP W_2 GELU(W_1 h) maps an encoder feature h of size d_model to,
    for every prediction step t and output coordinate j: the center c[t, j], the
    binary generator entries G_b[t, j, :], the bounded half-width a[t, j] and the
    noise scale sigma[t, j], where a = exp(clamp(log a, -4, 4)) and
    sigma = exp(clamp(log sigma, -4, 4)).

    Initialisation follows Algorithm 1: output-layer weights ~ N(0, 0.01^2); output
    bias 0 for c, b0 for G_b (symmetry breaking), init_log_a for log a and
    init_log_sigma for log sigma.

    Args:
        d_model: size of the encoder feature h.
        horizon: number of prediction steps T.
        dim: number of output coordinates per step D (2 for planar positions).
        n_b: number of binary generators; the mixture has 2^n_b modes.
        b0: initial output bias of the G_b entries.
        init_log_a: initial output bias of log a.
        init_log_sigma: initial output bias of log sigma.

    Returns (forward): a dict with "center", "a" and "sigma" of shape (..., T, D)
    and "G_b" of shape (..., T, D, n_b), for h of shape (..., d_model).
    """

    def __init__(self, d_model: int, horizon: int, dim: int = 2, n_b: int = 1,
                 b0: float = 0.0, init_log_a: float = 0.7,
                 init_log_sigma: float = 1.1):
        super().__init__()
        self.horizon, self.dim, self.n_b = horizon, dim, n_b
        width = dim * (n_b + 3)  # per step: [c | G_b | log a | log sigma]
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon * width),
        )
        with torch.no_grad():
            last = self.mlp[-1]
            nn.init.normal_(last.weight, std=0.01)
            bias = last.bias.view(horizon, width)
            bias[:, :dim] = 0.0
            bias[:, dim:dim * (n_b + 1)] = b0
            bias[:, dim * (n_b + 1):dim * (n_b + 2)] = init_log_a
            bias[:, dim * (n_b + 2):] = init_log_sigma

    def forward(self, h: torch.Tensor) -> dict:
        d, n_b = self.dim, self.n_b
        raw = self.mlp(h).view(*h.shape[:-1], self.horizon, d * (n_b + 3))
        log_a = raw[..., d * (n_b + 1):d * (n_b + 2)]
        log_sigma = raw[..., d * (n_b + 2):]
        return {
            "center": raw[..., :d],
            "G_b": raw[..., d:d * (n_b + 1)].reshape(*raw.shape[:-1], d, n_b),
            "a": log_a.clamp(-4.0, 4.0).exp(),
            "sigma": log_sigma.clamp(-4.0, 4.0).exp(),
        }
