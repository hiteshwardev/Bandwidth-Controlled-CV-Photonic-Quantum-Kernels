"""GBS (Gaussian-boson-sampling) feature kernel under loss, via thewalrus.

Construction (non-Gaussian because of photon-number detection):
  * M modes, each squeezed by r, mixed by a fixed Haar-random interferometer U (seeded),
    with data encoded as per-mode displacements beta_m(x) = disp_scale * x_m.
  * Per-mode pure loss eta applied to the Gaussian state (covariance/mean rescaling).
  * Feature phi(x) = vector of photon-number-pattern probabilities up to a total-photon
    cutoff, computed with thewalrus.quantum.probabilities (hbar=2, xxpp ordering).
  * Kernel k(x,x') = cosine similarity <phi(x),phi(x')> / (||phi(x)|| ||phi(x')||) -- a
    valid PSD kernel on non-negative feature vectors, in [0,1], =1 for identical inputs.

The lossy photon-number probabilities are validated against qutip (1-2 modes) in
tests/test_validation.py before use.
"""
from __future__ import annotations
import numpy as np
from itertools import product as iproduct

from thewalrus.quantum import probabilities as tw_probabilities

HBAR = 2.0


def _haar_unitary(M: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diag(r) / np.abs(np.diag(r))
    return q * ph


def _interferometer_symplectic(U: np.ndarray) -> np.ndarray:
    """Symplectic (xxpp ordering) for a passive interferometer U (M x M unitary)."""
    X, Y = U.real, U.imag
    return np.block([[X, -Y], [Y, X]])


def gaussian_state_xxpp(x, r: float, U: np.ndarray, eta: float, disp_scale: float = 1.0):
    """Build (mu, cov) in xxpp ordering, hbar=2, for the lossy displaced GBS state."""
    M = U.shape[0]
    x = np.atleast_1d(x).astype(float)
    if x.size == 1:
        x = np.repeat(x, M)
    # squeezed vacuum covariance (xxpp): diag(e^{-2r}) on x-block, diag(e^{+2r}) on p-block
    cov = np.diag(np.concatenate([np.exp(-2 * r) * np.ones(M), np.exp(2 * r) * np.ones(M)]))
    S = _interferometer_symplectic(U)
    cov = S @ cov @ S.T
    # displacement before interferometer, in xxpp (x-quadrature only): mu0 = (2*disp*x, 0)
    mu0 = np.concatenate([2.0 * disp_scale * x, np.zeros(M)])
    mu = S @ mu0
    # pure loss
    cov = eta * cov + (1.0 - eta) * np.eye(2 * M)
    mu = np.sqrt(eta) * mu
    return mu, cov


def gbs_feature(x, r, U, eta, cutoff: int, disp_scale: float = 1.0) -> np.ndarray:
    mu, cov = gaussian_state_xxpp(x, r, U, eta, disp_scale)
    probs = tw_probabilities(mu, cov, cutoff, hbar=HBAR)
    return np.asarray(probs).ravel()


def gbs_kernel_value(x, xp, r, U, eta, cutoff: int, disp_scale: float = 1.0) -> float:
    f1 = gbs_feature(x, r, U, eta, cutoff, disp_scale)
    f2 = gbs_feature(xp, r, U, eta, cutoff, disp_scale)
    n1 = np.linalg.norm(f1); n2 = np.linalg.norm(f2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(f1, f2) / (n1 * n2))


def gbs_mean_var(r, M, eta, cutoff, sigma, n_pairs, seed_U, rng, disp_scale=1.0, n_boot=200):
    U = _haar_unitary(M, seed_U)
    ks = np.empty(n_pairs)
    for i in range(n_pairs):
        x = rng.normal(0.0, sigma, size=M)
        xp = rng.normal(0.0, sigma, size=M)
        ks[i] = gbs_kernel_value(x, xp, r, U, eta, cutoff, disp_scale)
    mean, var = ks.mean(), ks.var()
    boot = np.empty(n_boot)
    idx = np.arange(n_pairs)
    for b in range(n_boot):
        s = ks[rng.choice(idx, size=n_pairs, replace=True)]
        boot[b] = s.var()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, var, (lo, hi)
