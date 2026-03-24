"""Tests for the diagnostics module."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pytest
from econlab.diagnostics.tests import (
    vif,
    breusch_pagan,
    white_test,
    breusch_godfrey,
    ramsey_reset,
    durbin_watson,
    specification_summary,
)


def _make_ols_result(n=200, seed=42, heteroskedastic=False, autocorrelated=False,
                     nonlinear=False):
    """Helper to generate OLS result object for diagnostic tests."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    if heteroskedastic:
        errors = rng.normal(0, np.exp(0.5 * x1), n)
    elif autocorrelated:
        eps = rng.normal(0, 1, n)
        errors = np.zeros(n)
        errors[0] = eps[0]
        for i in range(1, n):
            errors[i] = 0.8 * errors[i-1] + eps[i]
    else:
        errors = rng.normal(0, 1, n)

    if nonlinear:
        y = 1.0 + 2.0 * x1 + 0.5 * x1**2 + errors
    else:
        y = 1.0 + 2.0 * x1 + 0.5 * x2 + errors

    df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
    X = sm.add_constant(df[['x1', 'x2']])
    res = sm.OLS(df['y'], X).fit()
    return res, df


def test_vif_no_multicollinearity():
    """Orthogonal regressors should have VIF close to 1."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        'x1': rng.normal(0, 1, n),
        'x2': rng.normal(0, 1, n),
        'x3': rng.normal(0, 1, n),
    })
    result = vif(df, X=['x1', 'x2', 'x3'])

    assert isinstance(result, pd.DataFrame)
    assert 'variable' in result.columns
    assert 'VIF' in result.columns
    assert len(result) == 3

    # VIF should be close to 1 for orthogonal variables
    for _, row in result.iterrows():
        assert row['VIF'] < 5.0, (
            f"VIF for {row['variable']} = {row['VIF']:.2f}, expected ~1 for orthogonal vars"
        )


def test_vif_high_multicollinearity():
    """Highly correlated variables should have high VIF."""
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.1, n)  # nearly collinear with x1
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    result = vif(df, X=['x1', 'x2'])

    assert len(result) == 2
    max_vif = result['VIF'].max()
    assert max_vif > 10.0, f"Expected VIF > 10 for collinear vars, got {max_vif:.2f}"


def test_breusch_pagan_homoskedastic():
    """Breusch-Pagan should not reject under homoskedasticity (most of the time)."""
    res, _ = _make_ols_result(n=300, seed=0, heteroskedastic=False)
    result = breusch_pagan(res)

    assert 'stat' in result
    assert 'pval' in result
    assert 'reject_h0' in result
    assert result['pval'] > 0.01, (
        f"BP test rejected homoskedasticity (pval={result['pval']:.3f}) for homoskedastic data. "
        "This can happen occasionally."
    )


def test_breusch_pagan_heteroskedastic():
    """Breusch-Pagan should reject under heteroskedasticity."""
    res, _ = _make_ols_result(n=500, seed=42, heteroskedastic=True)
    result = breusch_pagan(res)

    assert result['stat'] > 0
    assert result['reject_h0'] is True, (
        f"BP test failed to reject heteroskedasticity (pval={result['pval']:.4f})"
    )


def test_white_test_returns_correct_structure():
    """White test should return the expected result structure."""
    res, _ = _make_ols_result(n=200, seed=5)
    result = white_test(res)

    expected_keys = {'stat', 'pval', 'df', 'reject_h0'}
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )
    assert result['stat'] >= 0
    assert 0 <= result['pval'] <= 1


def test_breusch_godfrey_no_autocorrelation():
    """BG test should not reject for iid errors (most of the time)."""
    res, _ = _make_ols_result(n=300, seed=10, autocorrelated=False)
    result = breusch_godfrey(res, nlags=1)

    assert 'stat' in result
    assert 'pval' in result
    assert result['df'] == 1
    assert result['pval'] > 0.01, (
        f"BG test rejected no-autocorrelation (pval={result['pval']:.3f}) for iid errors."
    )


def test_breusch_godfrey_autocorrelated():
    """BG test should reject for autocorrelated errors."""
    res, _ = _make_ols_result(n=500, seed=0, autocorrelated=True)
    result = breusch_godfrey(res, nlags=1)

    assert result['reject_h0'] is True, (
        f"BG test failed to reject serial correlation (pval={result['pval']:.4f})"
    )


def test_ramsey_reset_correct_spec():
    """Ramsey RESET should not reject correct linear specification (most of the time)."""
    res, _ = _make_ols_result(n=300, seed=99, nonlinear=False)
    result = ramsey_reset(res, power=3)

    assert 'F_stat' in result
    assert 'pval' in result
    assert 'reject_h0' in result
    assert result['pval'] > 0.01, (
        f"RESET test rejected correct linear spec (pval={result['pval']:.3f}). "
        "This can happen occasionally."
    )


def test_ramsey_reset_nonlinear():
    """Ramsey RESET should reject for nonlinear functional form."""
    res, _ = _make_ols_result(n=500, seed=5, nonlinear=True)
    result = ramsey_reset(res, power=3)

    assert result['F_stat'] >= 0
    # Should reject (pval < 0.05) for misspecified model
    assert result['reject_h0'] is True, (
        f"RESET test failed to reject nonlinear model (pval={result['pval']:.4f})"
    )


def test_durbin_watson_no_autocorrelation():
    """Durbin-Watson statistic should be close to 2 for iid errors."""
    res, _ = _make_ols_result(n=300, seed=42, autocorrelated=False)
    dw = durbin_watson(res)

    assert isinstance(dw, float)
    assert 0 <= dw <= 4
    assert abs(dw - 2.0) < 1.0, (
        f"DW = {dw:.3f}, expected close to 2.0 for no autocorrelation"
    )


def test_specification_summary_returns_df():
    """specification_summary should return a DataFrame with expected columns."""
    res, _ = _make_ols_result(n=300, seed=42)
    result = specification_summary(res)

    assert isinstance(result, pd.DataFrame)
    expected_cols = {'test', 'statistic', 'p_value', 'reject_H0 (5%)', 'interpretation'}
    assert expected_cols.issubset(result.columns), (
        f"Missing columns: {expected_cols - set(result.columns)}"
    )
    assert len(result) >= 4, "Expected at least 4 diagnostic tests in summary"


def test_vif_missing_columns_raises():
    """vif should raise ValueError for missing columns."""
    df = pd.DataFrame({'x1': [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing columns"):
        vif(df, X=['x1', 'x_missing'])


def test_breusch_pagan_invalid_input_raises():
    """breusch_pagan should raise ValueError for non-result input."""
    with pytest.raises((ValueError, AttributeError)):
        breusch_pagan({'not': 'a result'})
