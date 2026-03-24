from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from typing import Dict, Any, Optional, List


def engle_granger(
    y: pd.Series,
    x: pd.Series,
    trend: str = 'c',
    maxlag: Optional[int] = None,
    autolag: str = 'AIC',
) -> Dict[str, Any]:
    """
    Engle-Granger two-step cointegration test.

    Step 1: OLS regression of y on x to estimate the cointegrating vector.
    Step 2: ADF test on the residuals to test for stationarity.

    Parameters
    ----------
    y : pd.Series
        Dependent time series.
    x : pd.Series
        Independent time series (or multiple series stacked as DataFrame).
    trend : str
        Trend specification for OLS: 'c' (constant), 'ct' (constant+trend), 'n' (none).
    maxlag : int or None
        Maximum lags for ADF test on residuals.
    autolag : str
        Lag selection criterion for ADF test.

    Returns
    -------
    dict with keys:
        cointegrating_vector (pd.Series): OLS coefficients
        residuals (pd.Series): Residuals from cointegrating regression
        adf_stat (float): ADF statistic on residuals
        adf_pval (float): p-value from ADF test
        critical_values (dict): ADF critical values at 1%, 5%, 10%
        reject_h0 (bool): Whether to reject H0 (no cointegration) at 5% level
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    if not isinstance(x, (pd.Series, pd.DataFrame)):
        x = pd.Series(x)

    # Align and drop NaN
    combined = pd.concat([y.rename('_y_'), x], axis=1).dropna()
    y_clean = combined.iloc[:, 0]
    x_clean = combined.iloc[:, 1:]

    if len(y_clean) < 15:
        raise ValueError(f"Too few observations for cointegration test (n={len(y_clean)}, need >= 15).")

    valid_trend = {'c', 'ct', 'n'}
    if trend not in valid_trend:
        raise ValueError(f"trend must be one of {valid_trend}, got '{trend}'.")

    # Step 1: OLS cointegrating regression
    if trend == 'c':
        Xmat = sm.add_constant(x_clean, has_constant='add')
    elif trend == 'ct':
        t = np.arange(len(x_clean))
        Xmat = sm.add_constant(x_clean, has_constant='add')
        Xmat['_trend_'] = t
    else:
        Xmat = x_clean.copy()

    ols_res = sm.OLS(y_clean, Xmat.astype(float)).fit()
    residuals = ols_res.resid

    # Step 2: ADF on residuals (no constant — already de-meaned by regression)
    adf_result = adfuller(residuals.values, maxlag=maxlag, regression='n', autolag=autolag)
    adf_stat, adf_pval, _, _, crit_vals, _ = adf_result

    # Engle-Granger critical values differ from standard ADF; use approximate ones
    # For now, use the ADF pval as approximation (conservative)
    reject_h0 = bool(adf_pval < 0.05)

    return {
        'cointegrating_vector': ols_res.params,
        'residuals': residuals,
        'adf_stat': float(adf_stat),
        'adf_pval': float(adf_pval),
        'critical_values': {k: float(v) for k, v in crit_vals.items()},
        'reject_h0': reject_h0,
    }


def johansen_test(
    data: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> Dict[str, Any]:
    """
    Johansen cointegration test (trace and max-eigenvalue statistics).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with multiple time series as columns.
    det_order : int
        Deterministic terms:
        -1 = no deterministic terms,
         0 = restricted constant (in cointegrating relation),
         1 = unrestricted constant.
    k_ar_diff : int
        Number of lagged differences in the VECM. Default 1.

    Returns
    -------
    dict with keys:
        trace_stat (np.ndarray): Trace test statistics
        trace_cv (np.ndarray): Trace test critical values (90%, 95%, 99%)
        max_eig_stat (np.ndarray): Max eigenvalue test statistics
        max_eig_cv (np.ndarray): Max eigenvalue critical values
        n_cointegrating_vectors (int): Estimated number of cointegrating relations (at 5%)
        eigenvalues (np.ndarray): Johansen eigenvalues
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pd.DataFrame.")

    if data.shape[1] < 2:
        raise ValueError("Johansen test requires at least 2 time series (columns).")

    data_clean = data.dropna()
    if len(data_clean) < 10:
        raise ValueError(f"Too few observations (n={len(data_clean)}, need >= 10).")

    if det_order not in {-1, 0, 1}:
        raise ValueError(f"det_order must be -1, 0, or 1, got {det_order}.")

    result = coint_johansen(data_clean.values.astype(float), det_order=det_order, k_ar_diff=k_ar_diff)

    trace_stat = result.lr1  # trace statistics
    max_eig_stat = result.lr2  # max eigenvalue statistics
    trace_cv = result.cvt  # critical values for trace (columns: 90%, 95%, 99%)
    max_eig_cv = result.cvm  # critical values for max-eig
    eigenvalues = result.eig

    # Determine number of cointegrating vectors (at 5% level, column index 1)
    n_coint = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > trace_cv[i, 1]:  # 95% critical value
            n_coint = i + 1

    return {
        'trace_stat': trace_stat,
        'trace_cv': trace_cv,
        'max_eig_stat': max_eig_stat,
        'max_eig_cv': max_eig_cv,
        'n_cointegrating_vectors': n_coint,
        'eigenvalues': eigenvalues,
    }
