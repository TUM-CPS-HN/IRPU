"""Within-mode log-density of HProbZ, Eq. (4).

Within a mode, coordinate j is distributed as Uniform(-a_j, a_j) * N(0, sigma_j^2)
around the mode center, whose density is

    f(r; a, sigma) = [Phi((r + a) / sigma) - Phi((r - a) / sigma)] / (2 a).
"""

import math

import torch

__all__ = ["log_uniform_gaussian"]


def log_uniform_gaussian(r: torch.Tensor, a: torch.Tensor, sigma: torch.Tensor,
                         eps: float = 1e-4) -> torch.Tensor:
    """log f(r; a, sigma), the log-density of Uniform(-a, a) * N(0, sigma^2) at r.

    a and sigma are clamped below at eps. Because f is symmetric in r, the CDF
    difference is evaluated as Phi(u) - Phi(l) with u = (a - |r|) / sigma and
    l = -(a + |r|) / sigma, in log space and in float64, which keeps it accurate in
    both tails. For a < 1e-6 sigma the density is replaced by its limit
    N(r; 0, sigma^2). The result has the floating dtype of the inputs (the default
    dtype if none of them is floating).
    """
    out_dtype = torch.promote_types(torch.promote_types(r.dtype, a.dtype), sigma.dtype)
    if not out_dtype.is_floating_point:
        out_dtype = torch.get_default_dtype()
    work = torch.float64
    r, a, sigma = r.to(work).abs(), a.to(work).clamp(min=eps), sigma.to(work).clamp(min=eps)
    small = a < 1e-6 * sigma
    a_safe = torch.where(small, sigma, a)  # keeps the unselected branch finite
    log_u = torch.special.log_ndtr((a_safe - r) / sigma)
    log_l = torch.special.log_ndtr(-(a_safe + r) / sigma)
    exact = log_u + torch.log(-torch.expm1(log_l - log_u)) - torch.log(2.0 * a_safe)
    limit = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * (r / sigma) ** 2
    return torch.where(small, limit, exact).to(out_dtype)
