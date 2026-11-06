"""Enumeration of the 2^n_b binary assignments beta of the HProbZ modes."""

import torch

__all__ = ["binary_configurations"]


def binary_configurations(n_b: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """All 2^n_b binary assignments beta in {-1, 1}^n_b, as a (2^n_b, n_b) tensor."""
    index = torch.arange(2 ** n_b, device=device)
    bits = (index.unsqueeze(-1) >> torch.arange(n_b, device=device)) & 1
    return bits.to(dtype) * 2 - 1
