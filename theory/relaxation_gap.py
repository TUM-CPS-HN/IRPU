"""Numerical check of the Gaussian relaxation of the bounded factor (Appendix B.6, Figure 3).

Algorithm 2 replaces the Uniform([-1, 1]) prior of the bounded factor alpha by the
Gaussian relaxation alpha ~ N(0, 1/3). This script compares the two posterior
variances in one dimension (n_b = 0, n_d = 1):

    z_i - c_i = a * alpha + sigma * eta_i,   eta_i ~ N(0, 1),   i = 1, ..., k,

with alpha shared by the k observations. For each kappa = a / sigma in
{1.5, 2.0, 2.8, 3.7} and k in {1, 2, 4, 8}:
    - V_U: Var(alpha | obs) under the uniform prior, by grid integration over
      [-1, 1], averaged over 1000 Monte-Carlo draws of the true alpha and the noise;
    - V_G = 1 / (3 + k * kappa^2): the closed-form variance under the relaxation;
    - the relative gap (V_G - V_U) / V_U.

Writes relaxation_gap.json, relaxation_gap.png and relaxation_gap.pdf to
results/relaxation_gap next to this script.
"""
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"

OUT_DIR = Path(__file__).parent / "results" / "relaxation_gap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid for alpha posterior (true uniform prior)
ALPHA_GRID = np.linspace(-1.0, 1.0, 4001)  # fine grid
DALPHA = ALPHA_GRID[1] - ALPHA_GRID[0]

# Number of MC realisations per (kappa, k) cell
N_MC = 1000


def true_posterior_variance(observations, a, sigma):
    """Var(alpha | obs) under uniform([-1,1]) prior, exact via grid integration.

    Likelihood: each obs z_{t_i} - c_{t_i} = a*alpha + sigma*eta_i, eta_i ~ N(0,1).
    p(z|alpha) = prod_i N(z_i - c_i; a*alpha, sigma^2)
    p(alpha|z) propto  Uniform(-1,1) * p(z|alpha)
    """
    diffs = observations  # already z_i - c_i
    # log-likelihood at each grid alpha
    # ll(alpha) = sum_i -0.5 * (d_i - a*alpha)^2 / sigma^2
    a_grid = ALPHA_GRID  # (G,)
    pred = a * a_grid                    # (G,)
    res = diffs[:, None] - pred[None, :]  # (k, G)
    ll = -0.5 * np.sum(res ** 2, axis=0) / (sigma ** 2)  # (G,)
    ll -= ll.max()  # numeric stability
    p = np.exp(ll)
    p = p / p.sum()  # already includes uniform prior * dalpha cancellation
    mean = np.sum(p * a_grid)
    var = np.sum(p * (a_grid - mean) ** 2)
    return var


def relaxed_posterior_variance(k, a, sigma):
    """Var(alpha | obs) under N(0, 1/3) prior, closed form.

    Prior precision: 3
    Likelihood precision per obs: a^2 / sigma^2 = kappa^2
    Posterior precision: 3 + k * kappa^2
    Posterior variance: 1 / (3 + k * kappa^2)
    """
    kappa = a / sigma
    return 1.0 / (3.0 + k * kappa ** 2)


def run_sweep():
    rng = np.random.default_rng(42)
    KAPPAS = [1.5, 2.0, 2.8, 3.7]  # match paper's nuScenes constraint kappa sweep
    KS = [1, 2, 4, 8]              # number of revealed observations
    SIGMA = 1.0
    results = {"kappa": KAPPAS, "k": KS, "n_mc": N_MC, "data": []}

    print(f"Running relaxation-gap check: {len(KAPPAS)} kappas x {len(KS)} k-values x {N_MC} MC reps", flush=True)
    print()
    for kappa in KAPPAS:
        a = kappa * SIGMA
        for k in KS:
            true_vars = []
            relaxed_var = relaxed_posterior_variance(k, a, SIGMA)
            for _ in range(N_MC):
                # Sample true alpha~Uniform([-1,1]); generate observations
                alpha_true = rng.uniform(-1.0, 1.0)
                eta = rng.standard_normal(k)
                # observation residuals: z_i - c_i = a*alpha + sigma*eta_i
                obs = a * alpha_true + SIGMA * eta
                v = true_posterior_variance(obs, a, SIGMA)
                true_vars.append(v)
            true_vars = np.array(true_vars)
            mean_true = float(true_vars.mean())
            std_true = float(true_vars.std())
            gap_abs = relaxed_var - mean_true
            gap_rel = gap_abs / mean_true * 100
            print(f"  kappa={kappa:.2f}  k={k}: "
                  f"V_relaxed={relaxed_var:.5f}  V_true(MC)={mean_true:.5f}+/-{std_true:.5f}  "
                  f"gap={gap_rel:+.2f}%",
                  flush=True)
            results["data"].append({
                "kappa": kappa,
                "k": k,
                "V_relaxed": relaxed_var,
                "V_true_mean": mean_true,
                "V_true_std": std_true,
                "gap_abs": float(gap_abs),
                "gap_rel_pct": float(gap_rel),
            })

    out_json = OUT_DIR / "relaxation_gap.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    # Plot: gap vs k for each kappa, 0.7 of a 5.5 in text width.
    TEXT_W = 5.5
    fig, ax = plt.subplots(figsize=(TEXT_W * 0.7, 2.0))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, kappa in enumerate(KAPPAS):
        rows = [r for r in results["data"] if r["kappa"] == kappa]
        ks = [r["k"] for r in rows]
        gaps = [r["gap_rel_pct"] for r in rows]
        ax.plot(ks, gaps, "o-", label=fr"$\kappa={kappa}$", color=colors[i], lw=1.2, ms=3.5)
    ax.axhline(0, color="black", linestyle=":", lw=0.8)
    ax.axhline(10, color="red", linestyle="--", lw=0.8, alpha=0.6, label=r"$\pm 10\%$ band")
    ax.axhline(-10, color="red", linestyle="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Number of revealed observations $k$", fontsize=8)
    ax.set_ylabel("Relative gap (%)", fontsize=8)
    ax.set_xticks([1, 2, 4, 8])
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    # Legend at lower right (data is positive 1-9%, lower area is empty)
    ax.legend(loc="lower right", fontsize=6.5, ncol=3, framealpha=0.92,
              handlelength=1.2, handletextpad=0.3, columnspacing=0.8,
              borderpad=0.3, labelspacing=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.2)

    out_png = OUT_DIR / "relaxation_gap.png"
    out_pdf = OUT_DIR / "relaxation_gap.pdf"
    plt.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {out_png}\nSaved: {out_pdf}", flush=True)

    print("\n=== Summary table ===")
    print(f"{'kappa':<8}{'k':<6}{'V_relaxed':<12}{'V_true(MC)':<14}{'gap_rel(%)':<12}")
    for r in results["data"]:
        print(f"{r['kappa']:<8}{r['k']:<6}{r['V_relaxed']:<12.5f}{r['V_true_mean']:<14.5f}{r['gap_rel_pct']:<+12.2f}")

    max_abs_gap = max(abs(r["gap_rel_pct"]) for r in results["data"])
    print(f"\nMax |relative gap| across all (kappa, k): {max_abs_gap:.2f}%")


if __name__ == "__main__":
    run_sweep()
