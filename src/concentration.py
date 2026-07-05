"""Concentration metrics for the photonic fidelity kernel.

For the M-mode Gaussian product kernel  k = exp(-gamma * ||x-x'||^2)  with data drawn
iid x ~ N(0, sigma^2 I_M), the data-averaged kernel moments are exact:

    E[k]   = (1 + 4 gamma sigma^2)^{-M/2}
    E[k^2] = (1 + 8 gamma sigma^2)^{-M/2}
    Var[k] = (1 + 8 gamma sigma^2)^{-M/2} - (1 + 4 gamma sigma^2)^{-M}

Asymptotically Var[k] ~ (1 + 8 gamma sigma^2)^{-M/2}, i.e. exponential concentration in M
with per-mode rate  c(gamma) = (1/2) ln(1 + 8 gamma sigma^2)  (nats/mode). The required
shots to resolve two kernel entries separated by Delta scale as ~ 1/Var (Chebyshev).

These closed forms are validated against Monte-Carlo in tests/ and in the benchmark run.
"""
from __future__ import annotations
import numpy as np
from .gaussian import effective_bandwidth, product_kernel


# ----------------------------- exact (Gaussian product) -----------------------------

def exact_mean_var(gamma: float, M: int, sigma: float) -> tuple[float, float]:
    g = gamma * sigma**2
    Ek = (1.0 + 4.0 * g) ** (-M / 2.0)
    Ek2 = (1.0 + 8.0 * g) ** (-M / 2.0)
    return Ek, Ek2 - Ek**2


def exact_rate_per_mode(gamma: float, sigma: float) -> float:
    """Asymptotic per-mode concentration rate c (nats/mode): Var ~ exp(-c M)."""
    return 0.5 * np.log(1.0 + 8.0 * gamma * sigma**2)


def variance_grid_exact(r: float, eta: float, disp_scale: float, M_values, sigma: float):
    gamma = effective_bandwidth(r, eta, disp_scale)
    out = []
    for M in M_values:
        _, var = exact_mean_var(gamma, M, sigma)
        out.append(var)
    return gamma, np.array(out)


# ----------------------------- Monte-Carlo (general) -----------------------------

def mc_kernel_samples(r: float, eta: float, disp_scale: float, M: int, sigma: float,
                      n_pairs: int, rng) -> np.ndarray:
    X = rng.normal(0.0, sigma, size=(n_pairs, M))
    Xp = rng.normal(0.0, sigma, size=(n_pairs, M))
    gamma = effective_bandwidth(r, eta, disp_scale)
    d2 = np.sum((X - Xp) ** 2, axis=1)
    return np.exp(-gamma * d2)


def mc_mean_var(r: float, eta: float, disp_scale: float, M: int, sigma: float,
                n_pairs: int, rng, n_boot: int = 200):
    k = mc_kernel_samples(r, eta, disp_scale, M, sigma, n_pairs, rng)
    mean, var = k.mean(), k.var()
    # bootstrap CI on the variance
    boot = np.empty(n_boot)
    idx = np.arange(len(k))
    for b in range(n_boot):
        s = k[rng.choice(idx, size=len(k), replace=True)]
        boot[b] = s.var()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, var, (lo, hi)


# ----------------------------- decay-rate fit -----------------------------

def fit_decay_rate(M_values, variances):
    """Robust linear fit of ln(Var) vs M. Returns (slope, intercept, R2)."""
    M = np.asarray(M_values, float)
    y = np.log(np.asarray(variances, float))
    keep = np.isfinite(y)
    M, y = M[keep], y[keep]
    A = np.vstack([M, np.ones_like(M)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef
    yhat = A @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def shots_to_resolve(var: float, delta: float = 0.1, conf_factor: float = 9.0) -> float:
    """Chebyshev-style shots to resolve a kernel gap delta at the given confidence factor."""
    return conf_factor * var / max(delta**2, 1e-300)
