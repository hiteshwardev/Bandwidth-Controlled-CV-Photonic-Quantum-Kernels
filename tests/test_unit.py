"""Unit tests: closed-form kernel properties and limits."""
import numpy as np
import pytest
from src.gaussian import (loss_covariance_diag, single_mode_kernel, effective_bandwidth,
                          product_kernel, equal_cov_fidelity, encode_single)
from src.concentration import exact_mean_var, exact_rate_per_mode


def test_identical_inputs_give_unit_kernel():
    assert single_mode_kernel(0.0, r=0.5, eta=0.7) == pytest.approx(1.0)
    x = np.array([0.3, -0.2, 0.5])
    assert product_kernel(x, x, r=0.6, eta=0.8) == pytest.approx(1.0)


def test_kernel_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = rng.uniform(0, 1.5); eta = rng.uniform(0.1, 1.0)
        dx = rng.normal(0, 2)
        k = single_mode_kernel(dx, r, eta)
        assert 0.0 <= k <= 1.0


def test_lossless_vacuum_covariance_is_identity():
    v_x, v_p = loss_covariance_diag(r=0.0, eta=1.0)
    assert v_x == pytest.approx(1.0) and v_p == pytest.approx(1.0)


def test_squeezing_sets_covariance():
    r = 0.7
    v_x, v_p = loss_covariance_diag(r, eta=1.0)
    assert v_x == pytest.approx(np.exp(-2 * r))
    assert v_p == pytest.approx(np.exp(2 * r))
    assert v_x * v_p == pytest.approx(1.0)  # pure state: det V = 1


def test_high_loss_drives_covariance_to_vacuum():
    v_x, v_p = loss_covariance_diag(r=1.0, eta=1e-6)
    assert v_x == pytest.approx(1.0, abs=1e-4) and v_p == pytest.approx(1.0, abs=1e-4)


def test_bandwidth_monotone_in_loss():
    # gamma increases with eta (less loss -> larger bandwidth)
    g = [effective_bandwidth(r=0.7, eta=e) for e in [0.1, 0.3, 0.5, 0.7, 1.0]]
    assert all(g[i] < g[i + 1] for i in range(len(g) - 1))


def test_bandwidth_increases_with_squeezing_lossless():
    g = [effective_bandwidth(r=rr, eta=1.0) for rr in [0.0, 0.3, 0.6, 1.0]]
    assert all(g[i] < g[i + 1] for i in range(len(g) - 1))


def test_product_kernel_factorizes():
    r, eta = 0.5, 0.8
    x = np.array([0.4, -0.3]); xp = np.array([0.1, 0.2])
    prod = single_mode_kernel(x[0] - xp[0], r, eta) * single_mode_kernel(x[1] - xp[1], r, eta)
    assert product_kernel(x, xp, r, eta) == pytest.approx(prod)


def test_equal_cov_fidelity_matches_single_mode_kernel_squared():
    # k = F^2 ; equal_cov_fidelity returns F ; check consistency on one mode
    r, eta, x, xp = 0.5, 0.7, 0.6, -0.2
    V1, m1 = encode_single(x, r, eta); V2, m2 = encode_single(xp, r, eta)
    F = equal_cov_fidelity(V1, m1, m2)
    assert F ** 2 == pytest.approx(single_mode_kernel(x - xp, r, eta), rel=1e-9)


def test_exact_rate_positive_and_monotone_in_bandwidth():
    rates = [exact_rate_per_mode(g, sigma=1.0) for g in [0.1, 1.0, 4.0, 20.0]]
    assert all(r > 0 for r in rates)
    assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))


def test_exact_variance_nonnegative():
    for g in [0.1, 1.0, 4.0]:
        for M in [1, 5, 20, 50]:
            _, var = exact_mean_var(g, M, sigma=1.0)
            assert var >= -1e-15
