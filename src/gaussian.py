"""Gaussian fidelity kernel under photon loss (exact, equal-covariance route).

Key structural fact used throughout: the data encoding changes ONLY the displacement
(mean), not the squeezing or the loss. Hence for any two data points the two Gaussian
states share the SAME covariance matrix V. For equal-covariance Gaussian states the
Uhlmann fidelity is exact and prefactor-free:

    F(rho1, rho2) = exp[ -(1/8) (mu2-mu1)^T V^{-1} (mu2-mu1) ]      (hbar=2, vacuum=I)

and we define the (squared) fidelity kernel  k = F^2  (the standard fidelity quantum
kernel, reducing to |<phi(x)|phi(x')>|^2 for pure states). For M independent modes the
state is a product, so the M-mode kernel is the product of single-mode kernels.

Validated against qutip Fock-space fidelity (pure coherent, pure squeezed, and a genuine
beamsplitter loss channel) in tests/test_validation.py.
"""
from __future__ import annotations
import numpy as np


def loss_covariance_diag(r: float, eta: float) -> tuple[float, float]:
    """Diagonal (v_x, v_p) of one mode: squeezed vacuum (squeezing r) after loss eta. Vacuum=I."""
    v_x = eta * np.exp(-2.0 * r) + (1.0 - eta)
    v_p = eta * np.exp(+2.0 * r) + (1.0 - eta)
    return v_x, v_p


def single_mode_kernel(dx: np.ndarray | float, r: float, eta: float, disp_scale: float = 1.0):
    """Squared-fidelity kernel k_1 for one mode as a function of data difference dx = x - x'.

    Closed form: k_1 = exp[ -eta * disp_scale^2 * dx^2 / v_x ],  v_x = eta e^{-2r} + (1-eta).
    Derivation: mu_x = 2*disp_scale*x, delta = mu_x(x)-mu_x(x') = 2*disp_scale*sqrt(eta)*dx after
    loss; F = exp[-(1/8) delta^2 / v_x]; k = F^2 = exp[-(1/4) delta^2 / v_x].
    """
    v_x, _ = loss_covariance_diag(r, eta)
    dx = np.asarray(dx, float)
    gamma = eta * disp_scale**2 / v_x          # effective RBF bandwidth (per mode)
    return np.exp(-gamma * dx**2)


def effective_bandwidth(r: float, eta: float, disp_scale: float = 1.0) -> float:
    """Per-mode RBF bandwidth gamma so that k_1 = exp(-gamma dx^2)."""
    v_x, _ = loss_covariance_diag(r, eta)
    return eta * disp_scale**2 / v_x


def product_kernel(x: np.ndarray, xp: np.ndarray, r: float, eta: float, disp_scale: float = 1.0) -> float:
    """M-mode product fidelity kernel for data vectors x, x' in R^M (one datum per mode)."""
    x = np.atleast_1d(x).astype(float); xp = np.atleast_1d(xp).astype(float)
    dx = x - xp
    gamma = effective_bandwidth(r, eta, disp_scale)
    return float(np.exp(-gamma * np.dot(dx, dx)))


# --- general equal-covariance fidelity (used by validation; not needed for the product route) ---

def equal_cov_fidelity(V: np.ndarray, mu1: np.ndarray, mu2: np.ndarray) -> float:
    """Exact Uhlmann fidelity for two Gaussian states with identical covariance V."""
    V = np.asarray(V, float); d = np.asarray(mu2, float) - np.asarray(mu1, float)
    return float(np.exp(-0.125 * d @ np.linalg.solve(V, d)))


def encode_single(x: float, r: float, eta: float, disp_scale: float = 1.0):
    """Return (V, mu) for one mode after loss; mu_x = 2*disp_scale*sqrt(eta)*x (post-loss mean)."""
    v_x, v_p = loss_covariance_diag(r, eta)
    V = np.diag([v_x, v_p])
    mu = np.array([2.0 * disp_scale * np.sqrt(eta) * x, 0.0])
    return V, mu
