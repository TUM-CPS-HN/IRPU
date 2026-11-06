"""Split-conformal calibration of an HProbZ head.

Implements the nonconformity score of Eq. (6) and the prediction set of Theorem 6.

For HProbZHead, ||G_d[j, :]||_1 = a_j and ||G_s[j, :]||_2 = sigma_j, so the score is

    s(x, y) = min_beta max_j (|y_j - (c + G_b beta)_j| - a_j) / sigma_j.

Let q be the ceil((1 - alpha)(n + 1))-th smallest score of n calibration pairs that are
exchangeable with the test pair and were not used for training. Then the set
{y : s(x, y) <= q}, the union over beta of the boxes
|y_j - (c + G_b beta)_j| <= a_j + q sigma_j, contains the test target with probability
at least 1 - alpha.
"""

import math

import torch

from .modes import binary_configurations

__all__ = ["nonconformity_score", "hprobz_score", "conformal_quantile",
           "prediction_boxes"]


def nonconformity_score(center: torch.Tensor, G_b: torch.Tensor, a: torch.Tensor,
                        sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """min_beta max_j (|y_j - (c + G_b beta)_j| - a_j) / sigma_j, the score of Eq. (6).

    The last axis indexes the coordinates j that share one binary assignment beta;
    all leading axes are independent.

    Args:
        center, a, sigma, y: tensors of shape (..., J); sigma must be positive.
        G_b: tensor of shape (..., J, n_b).

    Returns:
        A tensor of shape (...).
    """
    n_b = G_b.shape[-1]
    beta = binary_configurations(n_b, device=G_b.device, dtype=G_b.dtype)
    mode_center = center.unsqueeze(-2) + torch.einsum("...jn,kn->...kj", G_b, beta)
    excess = ((y.unsqueeze(-2) - mode_center).abs() - a.unsqueeze(-2)) / sigma.unsqueeze(-2)
    return excess.amax(dim=-1).amin(dim=-1)


def hprobz_score(output: dict, target: torch.Tensor) -> torch.Tensor:
    """Score of Eq. (6) for an HProbZHead output.

    The maximum runs over all steps and coordinates of a sample, which share one
    binary assignment beta, as in hprobz_nll.

    Args:
        output: dict returned by HProbZHead, for a batch of shape (...).
        target: tensor of shape (..., T, D).

    Returns:
        A tensor of shape (...).
    """
    return nonconformity_score(output["center"].flatten(-2),
                               output["G_b"].flatten(-3, -2),
                               output["a"].flatten(-2),
                               output["sigma"].flatten(-2),
                               target.flatten(-2))


def conformal_quantile(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """q_alpha of Theorem 6: the ceil((1 - alpha)(n + 1))-th smallest of n scores.

    Args:
        scores: tensor with the n calibration scores.
        alpha: miscoverage level in (0, 1).

    Returns:
        A 0-dim tensor; inf if ceil((1 - alpha)(n + 1)) > n, that is, if there are
        too few calibration scores for this alpha.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    scores = scores.flatten()
    n = scores.numel()
    k = math.ceil((1.0 - alpha) * (n + 1))
    if k > n:
        return torch.tensor(math.inf, dtype=scores.dtype, device=scores.device)
    return torch.kthvalue(scores, k).values


def prediction_boxes(output: dict, q) -> tuple:
    """The prediction set of Theorem 6 for an HProbZHead output, as 2^n_b boxes.

    Box beta is {z : |z - centers[beta]| <= half_width} elementwise, with
    centers[beta] = c + G_b beta and half_width = a + q * sigma. A target z lies in
    the union of the boxes exactly when hprobz_score(output, z) <= q. With q = gamma
    the union is the set S_gamma of Theorem 9.

    Args:
        output: dict returned by HProbZHead, for a batch of shape (...).
        q: threshold, e.g. from conformal_quantile.

    Returns:
        centers: tensor of shape (..., 2^n_b, T, D), in the order of
            binary_configurations(n_b).
        half_width: tensor of shape (..., T, D), shared by all boxes.
    """
    G_b = output["G_b"]
    beta = binary_configurations(G_b.shape[-1], device=G_b.device, dtype=G_b.dtype)
    centers = output["center"].unsqueeze(-3) + torch.einsum("...tdn,kn->...ktd", G_b, beta)
    return centers, output["a"] + q * output["sigma"]
