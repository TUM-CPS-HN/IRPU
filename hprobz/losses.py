"""Training objectives of Algorithm 1.

For a target y with coordinates j, mode beta in {-1, 1}^{n_b} has center
mu_beta = c + G_b beta, and the training objective is

    l = -log sum_beta 2^{-n_b} prod_j f(y_j - mu_{beta, j}; a_j, sigma_j),

with f the density of Eq. (4). The Gaussian surrogate of Algorithm 1 replaces each
f by the Gaussian with the same variance.
"""

import math

import torch

from .density import log_uniform_gaussian
from .modes import binary_configurations

__all__ = ["mixture_nll", "hprobz_nll", "gaussian_surrogate_nll",
           "hprobz_gaussian_surrogate_nll"]


def mixture_nll(center: torch.Tensor, G_b: torch.Tensor, a: torch.Tensor,
                sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """-log sum_beta 2^{-n_b} prod_j f(y_j - (c + G_b beta)_j; a_j, sigma_j).

    The last axis indexes the coordinates j that share one binary assignment beta;
    all leading axes are independent.

    Args:
        center, a, sigma, y: tensors of shape (..., J).
        G_b: tensor of shape (..., J, n_b).

    Returns:
        A tensor of shape (...).
    """
    n_b = G_b.shape[-1]
    beta = binary_configurations(n_b, device=G_b.device, dtype=G_b.dtype)
    mode_center = center.unsqueeze(-2) + torch.einsum("...jn,kn->...kj", G_b, beta)
    log_f = log_uniform_gaussian(y.unsqueeze(-2) - mode_center,
                                 a.unsqueeze(-2), sigma.unsqueeze(-2))
    return n_b * math.log(2.0) - torch.logsumexp(log_f.sum(dim=-1), dim=-1)


def hprobz_nll(output: dict, target: torch.Tensor) -> torch.Tensor:
    """Per-sample loss l_i of Algorithm 1 for an HProbZHead output.

    One binary assignment beta is shared by all steps and coordinates of a sample.

    Args:
        output: dict returned by HProbZHead, for a batch of shape (...).
        target: tensor of shape (..., T, D).

    Returns:
        A tensor of shape (...); average it over the batch for the training loss.
    """
    return mixture_nll(output["center"].flatten(-2),
                       output["G_b"].flatten(-3, -2),
                       output["a"].flatten(-2),
                       output["sigma"].flatten(-2),
                       target.flatten(-2))


def gaussian_surrogate_nll(center: torch.Tensor, G_b: torch.Tensor, a: torch.Tensor,
                           sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """-log sum_beta 2^{-n_b} prod_j N(y_j; (c + G_b beta)_j, sigma_j^2 + a_j^2 / 3).

    The Gaussian surrogate of Algorithm 1: every factor f(.; a_j, sigma_j) is replaced
    by the Gaussian with the same variance. The loss depends on a and sigma only
    through sigma^2 + a^2 / 3, so it cannot separate them (Proposition 4).

    Args:
        center, a, sigma, y: tensors of shape (..., J); sigma must be positive.
        G_b: tensor of shape (..., J, n_b).

    Returns:
        A tensor of shape (...).
    """
    n_b = G_b.shape[-1]
    beta = binary_configurations(n_b, device=G_b.device, dtype=G_b.dtype)
    mode_center = center.unsqueeze(-2) + torch.einsum("...jn,kn->...kj", G_b, beta)
    var = (sigma ** 2 + a ** 2 / 3.0).unsqueeze(-2)
    log_n = -0.5 * (math.log(2.0 * math.pi) + torch.log(var)
                    + (y.unsqueeze(-2) - mode_center) ** 2 / var)
    return n_b * math.log(2.0) - torch.logsumexp(log_n.sum(dim=-1), dim=-1)


def hprobz_gaussian_surrogate_nll(output: dict, target: torch.Tensor) -> torch.Tensor:
    """Per-sample Gaussian-surrogate loss of Algorithm 1 for an HProbZHead output.

    Same arguments and result shape as hprobz_nll; one binary assignment beta is
    shared by all steps and coordinates of a sample.
    """
    return gaussian_surrogate_nll(output["center"].flatten(-2),
                                  output["G_b"].flatten(-3, -2),
                                  output["a"].flatten(-2),
                                  output["sigma"].flatten(-2),
                                  target.flatten(-2))
