r"""Non-Gaussian sector: squeezed single-photon encoding, for the concentration-vs-simulability
boundary (pre-registered H4).

Encoding state |chi(r)> = S(r)|1>  (the photon-subtracted squeezed vacuum, up to normalisation),
genuinely non-Gaussian with Wigner negativity. Data x enters as a displacement; the M-mode state
is a product. Two quantities, both exact:

  * fidelity kernel  k(x,x') = F(rho(x), rho(x'))^2 .  Using the loss-channel covariance property
    Lambda_eta(D(a) rho D(a)^dag) = D(sqrt(eta) a) Lambda_eta(rho) D(sqrt(eta) a)^dag and the
    displacement-invariance of fidelity, k depends only on delta = x - x':
        k(delta) = F( sigma, D(sqrt(eta) delta) sigma D(sqrt(eta) delta)^dag )^2,
    sigma = Lambda_eta(|chi><chi|). Computed once per (r,eta), then evaluated on a delta grid.
  * Wigner negativity volume N(sigma) = \int max(0, -W) dx dp  -- the classical-simulability resource
    (Chabaud et al. 2024). For an M-mode product, total negativity grows multiplicatively, so N>0
    per mode implies exponentially growing negativity (classically hard); N=0 implies simulable.

Single mode in Fock space (qutip), modest cutoff -> exact and cheap; M enters only through products.
"""
from __future__ import annotations
import numpy as np
import qutip as qt

N_FOCK = 40


def chi_state(r: float, N: int = N_FOCK):
    """Squeezed single photon S(r)|1> (normalised ket)."""
    return qt.squeeze(N, r) * qt.basis(N, 1)


def _loss(rho, eta: float, N: int = N_FOCK):
    """Pure-loss channel via beamsplitter with vacuum ancilla + partial trace."""
    if eta >= 1.0:
        return qt.ket2dm(rho) if rho.type == 'ket' else rho
    th = np.arccos(np.sqrt(eta))
    a1 = qt.tensor(qt.destroy(N), qt.qeye(N)); a2 = qt.tensor(qt.qeye(N), qt.destroy(N))
    BS = (th * (a1.dag() * a2 - a1 * a2.dag())).expm()
    ket = rho if rho.type == 'ket' else None
    full = qt.tensor(ket, qt.basis(N, 0)) if ket is not None else None
    out = BS * full
    return (out * out.dag()).ptrace(0)


def lossy_sigma(r: float, eta: float, N: int = N_FOCK):
    """sigma = Lambda_eta(|chi><chi|)."""
    return _loss(chi_state(r, N), eta, N)


def wigner_negativity(sigma, lim: float = 6.0, n: int = 161) -> float:
    """Wigner negativity volume N = integral of max(0, -W) over phase space (qutip default phase-space convention)."""
    xv = np.linspace(-lim, lim, n)
    W = qt.wigner(sigma, xv, xv)
    dx = xv[1] - xv[0]
    neg = np.clip(-W, 0.0, None)
    return float(neg.sum() * dx * dx)


def single_mode_kernel_nong(delta, sigma, eta: float, N: int = N_FOCK) -> float:
    """k(delta) = F(sigma, D(sqrt(eta) delta) sigma D^dag)^2 for the non-Gaussian encoding."""
    alpha = np.sqrt(eta) * float(delta)
    D = qt.displace(N, alpha)
    rho2 = D * sigma * D.dag()
    F = qt.fidelity(sigma, rho2)
    return F ** 2


def single_mode_moments(sigma, eta: float, sigma_data: float, N: int = N_FOCK,
                        grid_half: float = 6.0, n_grid: int = 241):
    """E[k], E[k^2] over delta ~ N(0, 2 sigma_data^2) for one mode (1-D quadrature on a delta grid)."""
    s2 = 2.0 * sigma_data ** 2
    deltas = np.linspace(-grid_half, grid_half, n_grid)
    pdf = np.exp(-deltas ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)
    kv = np.array([single_mode_kernel_nong(d, sigma, eta, N) for d in deltas])
    w = pdf / pdf.sum()
    Ek = float(np.sum(w * kv))
    Ek2 = float(np.sum(w * kv ** 2))
    return Ek, Ek2


def product_variance(Ek: float, Ek2: float, M: int) -> float:
    """Var of the M-mode product non-Gaussian kernel: (E[k^2])^M - (E[k])^{2M}."""
    return Ek2 ** M - Ek ** (2 * M)


def concentration_rate(Ek2: float) -> float:
    """Per-mode concentration rate (nats/mode): Var ~ (E[k^2])^M => rate = -ln(E[k^2])."""
    return -np.log(Ek2)
