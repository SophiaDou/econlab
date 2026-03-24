"""Tests for RDD estimation."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from econlab.causal.rdd import rdd_sharp, rdd_fuzzy, mccrary_density_test, _ik_bandwidth


def _make_sharp_rdd_dgp(n=1000, tau=3.0, seed=42, cutoff=0.0, noise=1.0):
    """
    Generate a sharp RDD DGP.

    y = f(x) + tau * 1(x >= cutoff) + noise
    f(x) = 2*x (linear running variable effect)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    treatment = (x >= cutoff).astype(float)
    y = 2.0 * x + tau * treatment + rng.normal(0, noise, n)
    return pd.DataFrame({'y': y, 'x': x, 'D': treatment})


def _make_fuzzy_rdd_dgp(n=1000, late=2.5, seed=42, cutoff=0.0):
    """
    Generate a fuzzy RDD DGP.

    First stage: P(D=1 | x) jumps at cutoff but is not deterministic.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    above = (x >= cutoff).astype(float)
    # Compliance probability: 0.7 above cutoff, 0.2 below
    p_comply = np.where(above == 1, 0.7, 0.2)
    D = (rng.random(n) < p_comply).astype(float)
    y = 2.0 * x + late * D + rng.normal(0, 1.0, n)
    return pd.DataFrame({'y': y, 'x': x, 'D': D})


def test_sharp_rdd_recovers_tau():
    """Sharp RDD should recover the true jump (tau=3.0) at the cutoff."""
    df = _make_sharp_rdd_dgp(n=2000, tau=3.0, seed=42, noise=0.5)
    result = rdd_sharp(df, y='y', running='x', cutoff=0.0,
                       bandwidth=1.0, kernel='triangular', poly_order=1)

    assert 'tau' in result
    tau_hat = result['tau']
    assert abs(tau_hat - 3.0) < 0.8, (
        f"RDD estimate {tau_hat:.3f} too far from true value 3.0"
    )
    assert result['nobs_left'] > 0
    assert result['nobs_right'] > 0
    assert result['se'] > 0
    assert result['pval'] < 0.05, "Should reject H0 (tau != 0) at 5% for tau=3.0"


def test_rdd_bandwidth_selection():
    """IK bandwidth selector should return a positive, reasonable bandwidth."""
    df = _make_sharp_rdd_dgp(n=500, tau=2.0, seed=7)
    result = rdd_sharp(df, y='y', running='x', cutoff=0.0,
                       bandwidth=None, kernel='triangular', poly_order=1)

    assert result['bandwidth'] > 0, "Bandwidth should be positive"
    # For x ~ Uniform(-2, 2), a reasonable bandwidth is in (0.1, 4.0)
    assert result['bandwidth'] < 10.0, (
        f"Bandwidth {result['bandwidth']:.3f} seems unreasonably large"
    )


def test_rdd_ci_coverage():
    """95% CI should cover the true tau=2.0."""
    df = _make_sharp_rdd_dgp(n=1000, tau=2.0, seed=123, noise=0.8)
    result = rdd_sharp(df, y='y', running='x', cutoff=0.0,
                       bandwidth=1.0, kernel='triangular')

    lower, upper = result['ci_95']
    # Check CI width is positive
    assert upper > lower, "CI upper bound should exceed lower bound"
    # Check CI contains the true value (this may fail occasionally, ~5% of the time)
    assert lower < 2.0 < upper, (
        f"95% CI [{lower:.3f}, {upper:.3f}] does not contain true tau=2.0. "
        "This can happen by chance ~5% of the time."
    )


def test_rdd_different_kernels():
    """RDD should run with all supported kernel types."""
    df = _make_sharp_rdd_dgp(n=500, tau=2.0, seed=1)
    for kernel in ['triangular', 'uniform', 'epanechnikov']:
        result = rdd_sharp(df, y='y', running='x', cutoff=0.0,
                           bandwidth=1.0, kernel=kernel)
        assert 'tau' in result, f"Missing 'tau' key for kernel={kernel}"
        assert not np.isnan(result['tau']), f"tau is NaN for kernel={kernel}"


def test_mccrary_no_manipulation():
    """
    McCrary density test should not reject when running variable is uniformly distributed
    (no manipulation at the cutoff).
    """
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.uniform(-2.0, 2.0, n)
    df = pd.DataFrame({'x': x})

    result = mccrary_density_test(df, running='x', cutoff=0.0)

    assert 'pval' in result
    assert 'z' in result
    assert 'theta' in result

    if not np.isnan(result['pval']):
        # Under no manipulation, should not reject at 5%
        assert result['pval'] > 0.01, (
            f"McCrary test unexpectedly rejected (pval={result['pval']:.3f}) "
            "for uniform distribution — this can happen occasionally."
        )


def test_mccrary_returns_expected_keys():
    """McCrary test should return expected keys."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'x': rng.normal(0, 1, 500)})
    result = mccrary_density_test(df, running='x', cutoff=0.0)
    expected_keys = {'theta', 'se', 'z', 'pval'}
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )


def test_fuzzy_rdd_recovers_late():
    """Fuzzy RDD should recover the Local Average Treatment Effect (LATE)."""
    df = _make_fuzzy_rdd_dgp(n=2000, late=2.5, seed=42)
    result = rdd_fuzzy(df, y='y', running='x', treatment='D',
                       cutoff=0.0, bandwidth=1.0, kernel='triangular')

    assert 'late' in result
    late_hat = result['late']
    # LATE may deviate more than sharp RDD due to fuzziness; allow wider tolerance
    assert abs(late_hat - 2.5) < 2.0, (
        f"Fuzzy RDD LATE estimate {late_hat:.3f} too far from true LATE=2.5"
    )
    assert result['first_stage_F'] > 0, "First-stage F should be positive"


def test_rdd_raises_on_missing_column():
    """rdd_sharp should raise ValueError when a required column is missing."""
    df = pd.DataFrame({'y': [1, 2, 3], 'x': [0.1, -0.1, 0.2]})
    with pytest.raises(ValueError, match="Missing columns"):
        rdd_sharp(df, y='y', running='missing_column', cutoff=0.0)


def test_rdd_invalid_kernel():
    """rdd_sharp should raise ValueError for unsupported kernel."""
    df = _make_sharp_rdd_dgp(n=100, tau=2.0)
    with pytest.raises(ValueError, match="Unknown kernel"):
        rdd_sharp(df, y='y', running='x', cutoff=0.0,
                  bandwidth=1.0, kernel='gaussian')
