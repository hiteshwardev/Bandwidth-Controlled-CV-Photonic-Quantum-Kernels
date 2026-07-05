"""Reproducibility tests: pin key computed numbers so regressions are caught.

Snapshot values are taken from the validated computation and must remain stable under
the fixed conventions/seeds. If physics code changes intentionally, update with a note.
"""
import numpy as np
import pytest
from src.gaussian import effective_bandwidth, single_mode_kernel
from src.concentration import exact_mean_var, exact_rate_per_mode


def test_effective_bandwidth_snapshot():
    assert effective_bandwidth(0.7, 1.0, 1.0) == pytest.approx(4.0551999, rel=1e-6)
    assert effective_bandwidth(0.7, 0.5, 1.0) == pytest.approx(0.8021839, rel=1e-6)
    assert effective_bandwidth(1.5, 1.0, 1.0) == pytest.approx(20.085536, rel=1e-6)


def test_rate_snapshot():
    g1 = effective_bandwidth(0.7, 1.0, 1.0)
    assert exact_rate_per_mode(g1, 1.0) == pytest.approx(1.7549003, rel=1e-6)
    g2 = effective_bandwidth(0.7, 0.1, 1.0)
    assert exact_rate_per_mode(g2, 1.0) == pytest.approx(0.3117, rel=1e-3)


def test_single_mode_kernel_snapshot():
    assert single_mode_kernel(1.0, 0.5, 1.0) == pytest.approx(np.exp(-np.exp(1.0)), rel=1e-9)
    assert single_mode_kernel(0.5, 0.0, 0.5) == pytest.approx(0.882496, rel=1e-5)


def test_variance_snapshot():
    g = effective_bandwidth(0.3, 1.0, 1.0)
    _, v5 = exact_mean_var(g, 5, 1.0)
    assert v5 == pytest.approx(1.0188e-3, rel=1e-3)
    _, v50 = exact_mean_var(g, 50, 1.0)
    assert v50 > 0 and v50 < 1e-20  # deep concentration at M=50


def test_seed_determinism_in_mc():
    from src.concentration import mc_kernel_samples
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    k1 = mc_kernel_samples(0.5, 0.7, 1.0, 4, 1.0, 500, rng1)
    k2 = mc_kernel_samples(0.5, 0.7, 1.0, 4, 1.0, 500, rng2)
    assert np.array_equal(k1, k2)
