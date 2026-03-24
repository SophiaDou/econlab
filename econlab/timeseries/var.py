from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from typing import Dict, Any, List, Optional


def estimate_var(
    df: pd.DataFrame,
    variables: List[str],
    maxlags: Optional[int] = None,
    trend: str = 'c',
    ic: str = 'aic',
) -> Dict[str, Any]:
    """
    Vector Autoregression (VAR) estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data (rows = time periods).
    variables : list[str]
        Column names to include in the VAR.
    maxlags : int or None
        Maximum lag order to consider. If None, uses a rule of thumb.
    trend : str
        Deterministic terms: 'c' (constant), 'ct' (constant+trend), 'n' (none).
    ic : str
        Information criterion for lag selection: 'aic', 'bic', 'hqic', 'fpe'.

    Returns
    -------
    dict with keys:
        params: coefficient array
        aic (float): AIC
        bic (float): BIC
        hqic (float): HQIC
        lags_selected (int): Selected lag order
        nobs (int): Number of observations used
        irf: VARResults object (has .irf() and .fevd() methods)
        res: VARResults object
    """
    missing = set(variables) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    valid_trend = {'c', 'ct', 'nc', 'n'}
    if trend not in valid_trend:
        raise ValueError(f"trend must be one of {valid_trend}, got '{trend}'.")

    valid_ic = {'aic', 'bic', 'hqic', 'fpe'}
    if ic not in valid_ic:
        raise ValueError(f"ic must be one of {valid_ic}, got '{ic}'.")

    data_clean = df[variables].dropna()
    if len(data_clean) < 10:
        raise ValueError(f"Too few observations for VAR (n={len(data_clean)}, need >= 10).")

    model = VAR(data_clean.values.astype(float))

    if maxlags is None:
        maxlags = min(int(np.floor((len(data_clean) - 1) ** (1.0 / 3.0))), 12)

    lag_order_result = model.select_order(maxlags=maxlags, trend=trend)
    lags = getattr(lag_order_result, ic, None)
    if lags is None or lags <= 0:
        lags = 1

    res = model.fit(maxlags=lags, trend=trend)

    return {
        'params': res.params,
        'aic': float(res.aic),
        'bic': float(res.bic),
        'hqic': float(res.hqic),
        'lags_selected': int(lags),
        'nobs': int(res.nobs),
        'irf': res,  # Use res.irf(periods) to get IRFs
        'res': res,
    }


def granger_causality(
    df: pd.DataFrame,
    caused: str,
    causing: str,
    maxlag: int = 4,
    trend: str = 'c',
) -> Dict[str, Any]:
    """
    Granger causality test: does `causing` Granger-cause `caused`?

    Tests whether including lags of `causing` improves forecasting of `caused`.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data.
    caused : str
        The variable being predicted.
    causing : str
        The variable being tested as a Granger cause.
    maxlag : int
        Maximum number of lags to test. Default 4.
    trend : str
        Deterministic terms. Default 'c'.

    Returns
    -------
    dict with keys:
        results (list of dict): {lag, F, pval} for each lag 1..maxlag
        min_pval_lag (int): Lag with minimum p-value
        reject_h0 (bool): Whether to reject H0 (no Granger causality) at min p-value lag (5% level)
    """
    required = [caused, causing]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    data_clean = df[[caused, causing]].dropna()
    if len(data_clean) < 10:
        raise ValueError(f"Too few observations for Granger test (n={len(data_clean)}).")

    if maxlag < 1:
        raise ValueError(f"maxlag must be >= 1, got {maxlag}.")

    model = VAR(data_clean.values.astype(float))

    results_list = []
    for lag in range(1, maxlag + 1):
        try:
            res = model.fit(maxlags=lag, trend=trend)
            # Granger test: variable 1 (causing) -> variable 0 (caused)
            gc_result = res.test_causality(caused=0, causing=1, kind='f')
            F_stat = float(gc_result.test_statistic)
            pval = float(gc_result.pvalue)
        except Exception:
            F_stat = np.nan
            pval = np.nan
        results_list.append({'lag': lag, 'F': F_stat, 'pval': pval})

    valid_results = [r for r in results_list if not np.isnan(r['pval'])]
    if valid_results:
        min_pval_result = min(valid_results, key=lambda r: r['pval'])
        min_pval_lag = min_pval_result['lag']
        reject_h0 = min_pval_result['pval'] < 0.05
    else:
        min_pval_lag = 1
        reject_h0 = False

    return {
        'results': results_list,
        'min_pval_lag': min_pval_lag,
        'reject_h0': reject_h0,
    }


def var_irf(var_result: Any, periods: int = 10, orth: bool = True) -> Any:
    """
    Compute impulse response functions from a VAR result.

    Parameters
    ----------
    var_result : dict or VARResults
        Output of estimate_var() or a VARResults object directly.
    periods : int
        Number of periods for IRF. Default 10.
    orth : bool
        Whether to compute orthogonalized IRFs (Cholesky). Default True.

    Returns
    -------
    IRAnalysis object from statsmodels.
    """
    if isinstance(var_result, dict):
        res = var_result['res']
    else:
        res = var_result

    return res.irf(periods=periods)


def var_fevd(var_result: Any, periods: int = 10) -> Any:
    """
    Compute forecast error variance decomposition (FEVD) from a VAR result.

    Parameters
    ----------
    var_result : dict or VARResults
        Output of estimate_var() or a VARResults object.
    periods : int
        Number of periods for FEVD. Default 10.

    Returns
    -------
    FEVD object from statsmodels.
    """
    if isinstance(var_result, dict):
        res = var_result['res']
    else:
        res = var_result

    return res.fevd(periods=periods)
