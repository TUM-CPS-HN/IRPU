# HProbZ

Reference implementation of the HProbZ output head, its training objective and its split-conformal calibration.

## Repository structure

```text
hprobz/                        the Python package
├── head.py                    HProbZHead, the output head of Algorithm 1
├── modes.py                   the 2^n_b binary assignments beta
├── density.py                 the log-density of Eq. (4)
├── losses.py                  the training objective of Algorithm 1 and its Gaussian surrogate
├── conformal.py               the score of Eq. (6) and the prediction set of Theorem 6
└── metrics.py                 minADE@K and minFDE@K
examples/
├── train.py                   one training step of Algorithm 1
└── conformal_calibration.py   split-conformal calibration and the prediction boxes
theory/
└── relaxation_gap.py          the numerical check of Appendix B.6 (Figure 3)
```

All public functions are importable from `hprobz`:

| Name | What it computes |
|---|---|
| `HProbZHead` | The output head of Algorithm 1 (Appendix A.6): a two-layer MLP that maps an encoder feature to the centers `c`, binary generators `G_b`, bounded half-widths `a` and noise scales `sigma` at every prediction step. |
| `binary_configurations` | All 2^n_b binary assignments beta in {-1, 1}^n_b. |
| `log_uniform_gaussian` | The log-density of Uniform(-a, a) * N(0, sigma^2), Eq. (4), evaluated stably in log space. |
| `mixture_nll` | -log sum over beta of 2^(-n_b) prod_j f(y_j - mu_beta,j; a_j, sigma_j), enumerating all 2^n_b binary assignments beta. |
| `hprobz_nll` | The per-sample loss of Algorithm 1 for an `HProbZHead` output. |
| `gaussian_surrogate_nll`, `hprobz_gaussian_surrogate_nll` | The Gaussian surrogate of Algorithm 1 (the loss L_Gauss of Section 4), which replaces each factor f by a Gaussian of variance sigma_j^2 + a_j^2 / 3. |
| `nonconformity_score`, `hprobz_score` | The nonconformity score of Eq. (6). |
| `conformal_quantile` | The calibration quantile q_alpha of Theorem 6. |
| `prediction_boxes` | The prediction set of Theorem 6, a union of 2^n_b axis-aligned boxes. |
| `min_ade_fde` | minADE@K and minFDE@K of K sampled trajectories. |

Encoders, data loaders, baselines and experiment scripts are not included.

## Installation

```bash
pip install -e .              # the hprobz package; needs only PyTorch
pip install -e ".[theory]"    # also NumPy and Matplotlib, for theory/relaxation_gap.py
```

The package runs on CPU or CUDA; the log-density is evaluated in float64. The scripts in `examples/` also run from a plain checkout without installing the package, e.g. `python examples/train.py`.

## Usage

```python
import torch
from hprobz import HProbZHead, hprobz_nll

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
```

`hprobz_nll` uses one binary assignment for all steps and coordinates of a sample, as in Algorithm 1. For a separate assignment per step, apply `mixture_nll` to the per-step tensors and sum over the steps:

```python
from hprobz import mixture_nll

out = head(encoder(past))
loss = mixture_nll(out["center"], out["G_b"], out["a"], out["sigma"], future).sum(-1).mean()
```

For the Gaussian surrogate of Algorithm 1, use `hprobz_gaussian_surrogate_nll` in place of `hprobz_nll`.

The argument `b0` of `HProbZHead` sets the initial output bias of the `G_b` entries, the symmetry-breaking initialisation of Appendix D.25.

## Conformal calibration

Calibrate a trained head on pairs that were not used for training, then form the prediction set of Theorem 6:

```python
from hprobz import hprobz_score, conformal_quantile, prediction_boxes

cal_past, cal_future = torch.randn(500, 8, 2), torch.randn(500, T, D)    # calibration pairs
test_past, test_future = torch.randn(100, 8, 2), torch.randn(100, T, D)  # test pairs

with torch.no_grad():
    q = conformal_quantile(hprobz_score(head(encoder(cal_past)), cal_future), alpha=0.1)
    out = head(encoder(test_past))
    covered = hprobz_score(out, test_future) <= q   # P(covered) >= 1 - alpha by Theorem 6
    centers, half_width = prediction_boxes(out, q)  # the 2^n_b boxes of the prediction set
```

## Evaluation

`min_ade_fde(samples, target)` returns minADE@K and minFDE@K for sampled trajectories of shape `(..., K, T, D)` and targets of shape `(..., T, D)`; the two minima over the K samples are taken separately.

## License

MIT; see `LICENSE`.
