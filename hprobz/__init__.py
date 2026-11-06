"""HProbZ.

Reference implementation of the HProbZ output head, its training objective and its
split-conformal calibration.
"""

from .head import HProbZHead
from .modes import binary_configurations
from .density import log_uniform_gaussian
from .losses import (mixture_nll, hprobz_nll, gaussian_surrogate_nll,
                     hprobz_gaussian_surrogate_nll)
from .conformal import (nonconformity_score, hprobz_score, conformal_quantile,
                        prediction_boxes)
from .metrics import min_ade_fde

__all__ = ["HProbZHead", "binary_configurations", "log_uniform_gaussian",
           "mixture_nll", "hprobz_nll", "gaussian_surrogate_nll",
           "hprobz_gaussian_surrogate_nll", "nonconformity_score", "hprobz_score",
           "conformal_quantile", "prediction_boxes", "min_ade_fde"]
