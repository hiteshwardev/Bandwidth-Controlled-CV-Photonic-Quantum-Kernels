"""Integration tests: kernel-matrix positive-semidefiniteness and cross-module consistency."""
import numpy as np
import pytest
from src.gaussian import product_kernel, effective_bandwidth
from src.gbs import _haar_unitary, gbs_kernel_value
from src.concentration import exact_mean_var, mc_mean_var


def _gram(kfun, data):
    n = len(data)
    K = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kfun(data[i], data[j])
    return K


def test_gaussian_product_kernel_matrix_is_psd():
    rng = np.random.default_rng(1)
    data = [rng.normal(0, 1, 3) for _ in range(12)]
    K = _gram(lambda a, b: product_kernel(a, b, r=0.6, eta=0.7), data)
    assert np.allclose(K, K.T, atol=1e-10)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > -1e-8


def test_gbs_kernel_matrix_is_psd():
    rng = np.random.default_rng(2)
    M = 2; U = _haar_unitary(M, seed=2026)
    data = [rng.normal(0, 0.8, M) for _ in range(8)]
    K = _gram(lambda a, b: gbs_kernel_value(a, b, 0.5, U, 0.7, cutoff=6), data)
    assert np.allclose(K, K.T, atol=1e-8)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > -1e-6


def test_gbs_kernel_diagonal_is_one():
    M = 2; U = _haar_unitary(M, seed=2026)
    x = np.array([0.3, -0.1])
    assert gbs_kernel_value(x, x, 0.5, U, 0.7, cutoff=6) == pytest.approx(1.0, abs=1e-9)


def test_exact_variance_agrees_with_mc_where_mc_reliable():
    # only test where Var is large enough that MC with 30k samples is reliable
    rng = np.random.default_rng(3)
    for r, eta, M in [(0.3, 1.0, 1), (0.3, 0.5, 5), (0.7, 0.5, 1)]:
        g = effective_bandwidth(r, eta, 1.0)
        _, ve = exact_mean_var(g, M, 1.0)
        _, vm, (lo, hi) = mc_mean_var(r, eta, 1.0, M, 1.0, n_pairs=30000, rng=rng, n_boot=100)
        assert lo <= ve <= hi or abs(ve - vm) / vm < 0.08
