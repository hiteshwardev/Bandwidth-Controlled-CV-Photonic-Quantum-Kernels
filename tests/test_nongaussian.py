"""Validation/unit tests for the non-Gaussian (squeezed single-photon) sector and H4 quantities."""
import numpy as np
import pytest

qt = pytest.importorskip("qutip")
from src.nongaussian import (chi_state, lossy_sigma, wigner_negativity,
                             single_mode_kernel_nong, _loss, single_mode_moments,
                             product_variance, concentration_rate, N_FOCK)


def test_chi_is_squeezed_single_photon_norm():
    chi = chi_state(0.5)
    assert abs(chi.norm() - 1.0) < 1e-9


def test_negativity_positive_lossless_and_vanishes_under_heavy_loss():
    sig1 = lossy_sigma(0.5, 1.0)
    sig0 = lossy_sigma(0.5, 0.2)
    assert wigner_negativity(sig1) > 0.05      # non-Gaussian, negative Wigner
    assert wigner_negativity(sig0) < 1e-3      # below single-photon loss threshold ~0.5


def test_negativity_threshold_near_half():
    # negativity should be ~0 at eta=0.4 and positive at eta=0.6 (threshold ~0.5)
    assert wigner_negativity(lossy_sigma(0.5, 0.4)) < 1e-3
    assert wigner_negativity(lossy_sigma(0.5, 0.6)) > 1e-3


def test_kernel_unit_at_zero_delta():
    sig = lossy_sigma(0.6, 0.7)
    assert single_mode_kernel_nong(0.0, sig, 0.7) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("r,eta,x,xp", [(0.6, 0.7, 0.3, -0.2), (0.6, 0.7, 0.8, 0.1)])
def test_displacement_shortcut_matches_direct(r, eta, x, xp):
    sig = lossy_sigma(r, eta)
    k_short = single_mode_kernel_nong(x - xp, sig, eta)
    chi = chi_state(r)
    rx = _loss(qt.displace(N_FOCK, x) * chi, eta)
    rxp = _loss(qt.displace(N_FOCK, xp) * chi, eta)
    k_direct = qt.fidelity(rx, rxp) ** 2
    assert k_short == pytest.approx(k_direct, abs=1e-4)


def test_product_variance_and_rate_positive():
    sig = lossy_sigma(0.5, 0.8)
    Ek, Ek2 = single_mode_moments(sig, 0.8, 0.8, n_grid=121)
    assert 0 < Ek <= 1 and 0 < Ek2 <= 1
    assert concentration_rate(Ek2) > 0
    for M in [1, 5, 20]:
        assert product_variance(Ek, Ek2, M) >= -1e-15


def test_concentration_persists_in_simulable_regime():
    # at eta=0.4 (simulable, N~0) the non-Gaussian kernel still concentrates (rate>0)
    sig = lossy_sigma(0.5, 0.4)
    assert wigner_negativity(sig) < 1e-3
    Ek, Ek2 = single_mode_moments(sig, 0.4, 0.8, n_grid=121)
    assert concentration_rate(Ek2) > 0
    assert product_variance(Ek, Ek2, 20) < product_variance(Ek, Ek2, 2)
