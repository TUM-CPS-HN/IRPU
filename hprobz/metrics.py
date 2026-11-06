"""Best-of-K displacement errors (minADE@K and minFDE@K), the accuracy metrics of Section 5."""

import torch

__all__ = ["min_ade_fde"]


def min_ade_fde(samples: torch.Tensor, target: torch.Tensor) -> tuple:
    """Best-of-K average and final displacement errors.

    ADE is the Euclidean error averaged over the T steps of a trajectory and FDE the
    Euclidean error at the last step; each is minimised over the K samples separately.

    Args:
        samples: tensor of shape (..., K, T, D), K predicted trajectories per target.
        target: tensor of shape (..., T, D).

    Returns:
        min_ade, min_fde: tensors of shape (...).
    """
    err = torch.linalg.vector_norm(samples - target.unsqueeze(-3), dim=-1)  # (..., K, T)
    return err.mean(dim=-1).amin(dim=-1), err[..., -1].amin(dim=-1)
