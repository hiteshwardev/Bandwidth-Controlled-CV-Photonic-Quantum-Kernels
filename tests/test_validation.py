"""Validation tests: closed-form kernel and GBS probabilities vs independent ground truth.

Ground truth sources:
  * qutip Fock-space Uhlmann fidelity (with a genuine beamsplitter loss channel);
  * scipy 1-D quadrature for the data-averaged moments;
  * qutip photon-number distributions for thewalrus lossy probabilities.
These are the gate that justifies trusting the analytic results; keep cutoffs small for speed.
"""
import numpy as np
import pytest

qt = pytest.importorskip("qutip")
from scipy.integrate import quad
from src.gaussian import single_mode_kernel
from src.gbs import gbs_feature
from src.concentration import exact_mean_var

N = 30  # Fock cutoff for validation


def _cov_mean(rho):
    a = qt.destroy(N); x = a + a.dag(); p = -1j * (a - a.dag())
    rho = qt.ket2dm(rho) if rho.type == 'ket' else rho
    R = [x, p]; mu = np.array([qt.expect(Ri, rho).real for Ri in R]); V = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            V[i, j] = 0.5 * qt.expect(R[i] * R[j] + R[j] * R[i], rho).real - mu[i] * mu[j]
    return V, mu


def _k_qutip(rho1, rho2):
    f = qt.fidelity(qt.ket2dm(rho1) if rho1.type == 'ket' else rho1,
                    qt.ket2dm(rho2) if rho2.type == 'ket' else rho2)
    return f ** 2


def _lossy(psi_ket, eta):
    th = np.arccos(np.sqrt(eta))
    a1 = qt.tensor(qt.destroy(N), qt.qeye(N)); a2 = qt.tensor(qt.qeye(N), qt.destroy(N))
    BS = (th * (a1.dag() * a2 - a1 * a2.dag())).expm()
    out = BS * qt.tensor(psi_ket, qt.basis(N, 0))
    return (out * out.dag()).ptrace(0)


@pytest.mark.parametrize("r,x,xp", [(0.0, 0.3, -0.2), (0.5, 0.7, 0.0), (1.0, 0.0, 1.0)])
def test_kernel_vs_qutip_lossless(r, x, xp):
    s = qt.squeeze(N, r); v0 = qt.basis(N, 0)
    q1 = qt.displace(N, x) * s * v0; q2 = qt.displace(N, xp) * s * v0
    assert single_mode_kernel(x - xp, r, 1.0) == pytest.approx(_k_qutip(q1, q2), abs=1e-4)


@pytest.mark.parametrize("r,eta,x,xp", [(0.0, 0.5, 0.4, -0.3), (0.6, 0.5, 0.8, 0.2), (0.6, 0.3, 0.4, -0.3)])
def test_kernel_vs_qutip_lossy(r, eta, x, xp):
    s = qt.squeeze(N, r); v0 = qt.basis(N, 0)
    r1 = _lossy(qt.displace(N, x) * s * v0, eta)
    r2 = _lossy(qt.displace(N, xp) * s * v0, eta)
    assert single_mode_kernel(x - xp, r, eta) == pytest.approx(_k_qutip(r1, r2), abs=1e-3)


@pytest.mark.parametrize("r,beta,eta", [(0.3, 0.0, 1.0), (0.3, 0.5, 0.6), (0.6, 0.5, 0.6)])
def test_thewalrus_pn_vs_qutip(r, beta, eta):
    cutoff = 8
    U = np.array([[1.0 + 0j]])
    f = gbs_feature(beta, r, U, eta, cutoff, disp_scale=1.0)[:cutoff]
    # qutip lossy photon-number distribution
    v0 = qt.basis(N, 0)
    psi = qt.displace(N, beta) * qt.squeeze(N, r) * v0
    rho = _lossy(psi, eta)
    pq = np.real(rho.diag())[:cutoff]
    assert np.max(np.abs(f - pq)) < 1e-3


@pytest.mark.parametrize("gamma,sigma", [(0.8, 1.0), (4.0, 0.7), (2.0, 1.5)])
def test_moments_vs_quadrature(gamma, sigma):
    s2 = 2 * sigma ** 2
    pdf = lambda d: np.exp(-d * d / (2 * s2)) / np.sqrt(2 * np.pi * s2)
    I1, _ = quad(lambda d: np.exp(-gamma * d * d) * pdf(d), -np.inf, np.inf)
    I2, _ = quad(lambda d: np.exp(-2 * gamma * d * d) * pdf(d), -np.inf, np.inf)
    # M-mode product equals closed form
    for M in [5, 20]:
        Ek, var = exact_mean_var(gamma, M, sigma); Ek2 = var + Ek ** 2
        assert I1 ** M == pytest.approx(Ek, rel=1e-9)
        assert I2 ** M == pytest.approx(Ek2, rel=1e-9)
