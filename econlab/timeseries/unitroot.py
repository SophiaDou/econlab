from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Dict, Any, Optional


def adf_test(
    series: pd.Series,
    maxlag: Optional[int] = None,
    regression: str = 'c',
    autolag: str = 'AIC',
) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller (ADF) test for unit root.

    H0: Series has a unit root (non-stationary).
    H1: Series is stationary.

    Parameters
    ----------
    series : pd.Series
        Time series data.
    maxlag : int or None
        Maximum number of lags. If None, automatically selected.
    regression : str
        Type of regression: 'c' (constant), 'ct' (constant+trend),
        'ctt' (constant + linear + quadratic trend), 'n' (no constant).
    autolag : str
        Lag selection criterion: 'AIC', 'BIC', 't-stat', or None.

    Returns
    -------
    dict with keys:
        stat (float): ADF test statistic
        pval (float): p-value
        lags_used (int): Number of lags used
        n_obs (int): Number of observations
        critical_values (dict): Critical values at 1%, 5%, 10%
        reject_h0 (bool): Whether to reject H0 at 5% level
    """
    if not isinstance(series, (pd.Series, np.ndarray)):
        raise ValueError("series must be a pd.Series or np.ndarray.")

    s = pd.Series(series).dropna()
    if len(s) < 10:
        raise ValueError(f"Series too short for ADF test (n={len(s)}, need >= 10).")

    valid_regression = {'c', 'ct', 'ctt', 'n'}
    if regression not in valid_regression:
        raise ValueError(f"regression must be one of {valid_regression}, got '{regression}'.")

    result = adfuller(s.values, maxlag=maxlag, regression=regression, autolag=autolag)
    stat, pval, lags_used, n_obs, crit_vals, _ = result

    return {
        'stat': float(stat),
        'pval': float(pval),
        'lags_used': int(lags_used),
        'n_obs': int(n_obs),
        'critical_values': {k: float(v) for k, v in crit_vals.items()},
        'reject_h0': bool(pval < 0.05),
    }


def kpss_test(
    series: pd.Series,
    regression: str = 'c',
    nlags: str = 'auto',
) -> Dict[str, Any]:
    """
    KPSS test for stationarity.

    H0: Series is stationary.
    H1: Series has a unit root (non-stationary).

    Parameters
    ----------
    series : pd.Series
        Time series data.
    regression : str
        'c' (level stationarity) or 'ct' (trend stationarity).
    nlags : str or int
        Number of lags for Newey-West bandwidth. 'auto' uses data-based selection.

    Returns
    -------
    dict with keys:
        stat (float): KPSS test statistic
        pval (float): p-value (interpolated)
        lags (int): Number of lags used
        critical_values (dict): Critical values at 10%, 5%, 2.5%, 1%
        reject_h0 (bool): Whether to reject H0 (stationarity) at 5% level
    """
    if not isinstance(series, (pd.Series, np.ndarray)):
        raise ValueError("series must be a pd.Series or np.ndarray.")

    s = pd.Series(series).dropna()
    if len(s) < 10:
        raise ValueError(f"Series too short for KPSS test (n={len(s)}, need >= 10).")

    valid_regression = {'c', 'ct'}
    if regression not in valid_regression:
        raise ValueError(f"regression must be 'c' or 'ct', got '{regression}'.")

    result = kpss(s.values, regression=regression, nlags=nlags)
    stat, pval, lags, crit_vals = result

    return {
        'stat': float(stat),
        'pval': float(pval),
        'lags': int(lags),
        'critical_values': {k: float(v) for k, v in crit_vals.items()},
        'reject_h0': bool(pval < 0.05),
    }


def pp_test(
    series: pd.Series,
    lags: Optional[int] = None,
    regression: str = 'c',
) -> Dict[str, Any]:
    """
    Phillips-Perron (PP) test for unit root.

    Implemented via Newey-West adjusted ADF (uses adfuller with appropriate settings).
    H0: Series has a unit root (non-stationary).

    Parameters
    ----------
    series : pd.Series
        Time series data.
    lags : int or None
        Number of lags for Newey-West bandwidth correction. If None, uses Schwert rule.
    regression : str
        'c' (constant) or 'ct' (constant + trend).

    Returns
    -------
    dict with keys:
        stat (float): PP test statistic
        pval (float): p-value
        critical_values (dict): Critical values at 1%, 5%, 10%
        reject_h0 (bool): Whether to reject H0 at 5% level
    """
    if not isinstance(series, (pd.Series, np.ndarray)):
        raise ValueError("series must be a pd.Series or np.ndarray.")

    s = pd.Series(series).dropna()
    n = len(s)
    if n < 10:
        raise ValueError(f"Series too short for PP test (n={n}, need >= 10).")

    valid_regression = {'c', 'ct', 'n'}
    if regression not in valid_regression:
        raise ValueError(f"regression must be one of {{'c', 'ct', 'n'}}, got '{regression}'.")

    # PP test: use t-stat autolag with no lag augmentation as approximation
    # True PP adjusts the ADF statistic using Newey-West long-run variance
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100) ** 0.25))

    # Compute PP statistic manually
    # Step 1: Regress Δy on y_{t-1} (and trend if ct)
    dy = np.diff(s.values)
    y_lag = s.values[:-1]
    T_eff = len(dy)

    if regression == 'c':
        X = np.column_stack([np.ones(T_eff), y_lag])
    elif regression == 'ct':
        X = np.column_stack([np.ones(T_eff), np.arange(T_eff), y_lag])
    else:
        X = y_lag.reshape(-1, 1)

    try:
        coef, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
        resid = dy - X @ coef
    except Exception:
        # Fall back to adfuller
        result = adfuller(s.values, maxlag=lags, regression=regression, autolag=None)
        stat, pval, _, _, crit_vals, _ = result
        return {
            'stat': float(stat),
            'pval': float(pval),
            'critical_values': {k: float(v) for k, v in crit_vals.items()},
            'reject_h0': bool(pval < 0.05),
        }

    sigma2 = np.var(resid, ddof=X.shape[1])

    # Newey-West long-run variance
    s2 = np.var(resid, ddof=1)
    lrv = s2
    for h in range(1, lags + 1):
        gamma_h = np.mean(resid[h:] * resid[:-h])
        w_h = 1.0 - h / (lags + 1.0)
        lrv += 2.0 * w_h * gamma_h

    # OLS t-statistic for the unit root coefficient (last coefficient in X)
    rho_idx = -1  # y_{t-1} is last column
    rho_hat = coef[rho_idx]
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
        se_rho = np.sqrt(sigma2 * XtX_inv[rho_idx, rho_idx])
    except Exception:
        se_rho = np.sqrt(sigma2 / max(np.sum(y_lag ** 2), 1e-8))

    t_ols = rho_hat / se_rho if se_rho > 0 else np.nan

    # PP adjustment
    pp_stat = np.sqrt(sigma2 / lrv) * t_ols - 0.5 * (lrv - sigma2) / np.sqrt(lrv) * (T_eff / se_rho) * (1.0 / T_eff)

    # Use ADF critical values as approximation (same asymptotic distribution)
    result_adf = adfuller(s.values, maxlag=lags, regression=regression, autolag=None)
    _, pval_approx, _, _, crit_vals, _ = result_adf

    # Interpolate p-value from ADF distribution (same asymptotic)
    pval = pval_approx

    return {
        'stat': float(pp_stat) if not np.isnan(pp_stat) else float(t_ols),
        'pval': float(pval),
        'critical_values': {k: float(v) for k, v in crit_vals.items()},
        'reject_h0': bool(pval < 0.05),
    }


def unit_root_summary(
    series: pd.Series,
    name: str = 'series',
) -> pd.DataFrame:
    """
    Run ADF, KPSS, and PP tests and return a summary DataFrame.

    Parameters
    ----------
    series : pd.Series
        Time series data.
    name : str
        Name of the series for display purposes.

    Returns
    -------
    pd.DataFrame
        Columns: [test, statistic, p_value, reject_H0, conclusion]
    """
    rows = []

    # ADF test
    try:
        adf = adf_test(series)
        rows.append({
            'test': 'ADF',
            'statistic': adf['stat'],
            'p_value': adf['pval'],
            'reject_H0': adf['reject_h0'],
            'conclusion': 'Stationary' if adf['reject_h0'] else 'Unit Root',
        })
    except Exception as e:
        rows.append({'test': 'ADF', 'statistic': np.nan, 'p_value': np.nan,
                     'reject_H0': False, 'conclusion': f'Error: {e}'})

    # KPSS test (H0: stationary)
    try:
        kpss_r = kpss_test(series)
        rows.append({
            'test': 'KPSS',
            'statistic': kpss_r['stat'],
            'p_value': kpss_r['pval'],
            'reject_H0': kpss_r['reject_h0'],
            'conclusion': 'Unit Root' if kpss_r['reject_h0'] else 'Stationary',
        })
    except Exception as e:
        rows.append({'test': 'KPSS', 'statistic': np.nan, 'p_value': np.nan,
                     'reject_H0': False, 'conclusion': f'Error: {e}'})

    # PP test
    try:
        pp = pp_test(series)
        rows.append({
            'test': 'PP',
            'statistic': pp['stat'],
            'p_value': pp['pval'],
            'reject_H0': pp['reject_h0'],
            'conclusion': 'Stationary' if pp['reject_h0'] else 'Unit Root',
        })
    except Exception as e:
        rows.append({'test': 'PP', 'statistic': np.nan, 'p_value': np.nan,
                     'reject_H0': False, 'conclusion': f'Error: {e}'})

    return pd.DataFrame(rows)
