from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    acorr_breusch_godfrey,
    linear_reset,
)
from statsmodels.stats.stattools import durbin_watson as _durbin_watson
from typing import Dict, Any, List


def vif(df: pd.DataFrame, X: List[str]) -> pd.DataFrame:
    """
    Variance Inflation Factors (VIF) for multicollinearity detection.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    X : list[str]
        Regressor column names.

    Returns
    -------
    pd.DataFrame
        Columns: [variable, VIF], sorted by VIF descending.
        VIF > 10 typically indicates severe multicollinearity.
    """
    missing = set(X) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[X].dropna().astype(float)
    if len(work) < len(X) + 2:
        raise ValueError("Too few observations to compute VIF.")

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xmat = sm.add_constant(work, has_constant='add')
    col_names = Xmat.columns.tolist()

    rows = []
    for i, var in enumerate(col_names):
        if var == 'const':
            continue
        vif_val = variance_inflation_factor(Xmat.values, i)
        rows.append({'variable': var, 'VIF': float(vif_val)})

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values('VIF', ascending=False).reset_index(drop=True)
    return result


def breusch_pagan(res: Any) -> Dict[str, Any]:
    """
    Breusch-Pagan test for heteroskedasticity.

    H0: Homoskedasticity (constant variance).
    H1: Heteroskedasticity.

    Parameters
    ----------
    res : statsmodels OLS result object
        Fitted OLS regression result.

    Returns
    -------
    dict with keys:
        stat (float): LM test statistic
        pval (float): p-value
        df (int): Degrees of freedom
        reject_h0 (bool): Whether to reject H0 at 5% level
    """
    if not hasattr(res, 'resid'):
        raise ValueError("res must be a statsmodels regression result object with 'resid' attribute.")

    lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(res.resid, res.model.exog)
    df = res.model.exog.shape[1] - 1

    return {
        'stat': float(lm_stat),
        'pval': float(lm_pval),
        'df': int(df),
        'reject_h0': bool(lm_pval < 0.05),
    }


def white_test(res: Any) -> Dict[str, Any]:
    """
    White's test for heteroskedasticity.

    More general than Breusch-Pagan; includes cross-products and squares.
    H0: Homoskedasticity.

    Parameters
    ----------
    res : statsmodels OLS result object
        Fitted OLS regression result.

    Returns
    -------
    dict with keys:
        stat (float): LM test statistic
        pval (float): p-value
        df (int): Degrees of freedom
        reject_h0 (bool): Whether to reject H0 at 5% level
    """
    if not hasattr(res, 'resid'):
        raise ValueError("res must be a statsmodels regression result object with 'resid' attribute.")

    lm_stat, lm_pval, f_stat, f_pval = het_white(res.resid, res.model.exog)
    k = res.model.exog.shape[1] - 1
    # White test df = k*(k+1)/2 (all cross products)
    df = k * (k + 3) // 2

    return {
        'stat': float(lm_stat),
        'pval': float(lm_pval),
        'df': int(df),
        'reject_h0': bool(lm_pval < 0.05),
    }


def breusch_godfrey(res: Any, nlags: int = 1) -> Dict[str, Any]:
    """
    Breusch-Godfrey LM test for serial correlation.

    H0: No serial correlation up to order nlags.

    Parameters
    ----------
    res : statsmodels OLS result object
        Fitted OLS regression result.
    nlags : int
        Maximum lag order to test. Default 1.

    Returns
    -------
    dict with keys:
        stat (float): LM test statistic
        pval (float): p-value
        df (int): Degrees of freedom (= nlags)
        reject_h0 (bool): Whether to reject H0 at 5% level
    """
    if not hasattr(res, 'resid'):
        raise ValueError("res must be a statsmodels regression result object with 'resid' attribute.")

    if nlags < 1:
        raise ValueError(f"nlags must be >= 1, got {nlags}.")

    lm_stat, lm_pval, f_stat, f_pval = acorr_breusch_godfrey(res, nlags=nlags)

    return {
        'stat': float(lm_stat),
        'pval': float(lm_pval),
        'df': int(nlags),
        'reject_h0': bool(lm_pval < 0.05),
    }


def ramsey_reset(res: Any, power: int = 3) -> Dict[str, Any]:
    """
    Ramsey RESET test for functional form misspecification.

    Tests whether non-linear combinations of fitted values improve the model.
    H0: Correct functional form.

    Parameters
    ----------
    res : statsmodels OLS result object
        Fitted OLS regression result.
    power : int
        Highest power of fitted values to include (2 through power). Default 3.

    Returns
    -------
    dict with keys:
        F_stat (float): F-statistic
        pval (float): p-value
        df_num (int): Numerator degrees of freedom
        df_denom (int): Denominator degrees of freedom
        reject_h0 (bool): Whether to reject H0 at 5% level
    """
    if not hasattr(res, 'resid'):
        raise ValueError("res must be a statsmodels regression result object with 'resid' attribute.")

    if power < 2:
        raise ValueError(f"power must be >= 2, got {power}.")

    reset_result = linear_reset(res, power=power, use_f=True)

    return {
        'F_stat': float(reset_result.statistic),
        'pval': float(reset_result.pvalue),
        'df_num': int(reset_result.df_num) if hasattr(reset_result, 'df_num') else power - 1,
        'df_denom': int(reset_result.df_denom) if hasattr(reset_result, 'df_denom') else int(res.df_resid),
        'reject_h0': bool(reset_result.pvalue < 0.05),
    }


def durbin_watson(res: Any) -> float:
    """
    Return the Durbin-Watson statistic for residual autocorrelation.

    A value near 2.0 indicates no autocorrelation.
    Values near 0 indicate positive autocorrelation.
    Values near 4 indicate negative autocorrelation.

    Parameters
    ----------
    res : statsmodels OLS result object
        Fitted OLS regression result.

    Returns
    -------
    float
        Durbin-Watson statistic.
    """
    if not hasattr(res, 'resid'):
        raise ValueError("res must be a statsmodels regression result object with 'resid' attribute.")

    return float(_durbin_watson(res.resid))


def specification_summary(res: Any) -> pd.DataFrame:
    """
    Run all diagnostic tests and return a summary table.

    Parameters
    ----------
    res : statsmodels OLS result object
        Fitted OLS regression result.

    Returns
    -------
    pd.DataFrame
        Columns: [test, statistic, p_value, reject_H0 (5%), interpretation]
    """
    rows = []

    # Breusch-Pagan
    try:
        bp = breusch_pagan(res)
        rows.append({
            'test': 'Breusch-Pagan (Heteroskedasticity)',
            'statistic': bp['stat'],
            'p_value': bp['pval'],
            'reject_H0 (5%)': bp['reject_h0'],
            'interpretation': 'Heteroskedastic errors' if bp['reject_h0'] else 'Homoskedastic errors',
        })
    except Exception as e:
        rows.append({'test': 'Breusch-Pagan', 'statistic': np.nan, 'p_value': np.nan,
                     'reject_H0 (5%)': False, 'interpretation': f'Error: {e}'})

    # White test
    try:
        wt = white_test(res)
        rows.append({
            'test': "White's Test (Heteroskedasticity)",
            'statistic': wt['stat'],
            'p_value': wt['pval'],
            'reject_H0 (5%)': wt['reject_h0'],
            'interpretation': 'Heteroskedastic errors' if wt['reject_h0'] else 'Homoskedastic errors',
        })
    except Exception as e:
        rows.append({'test': "White's Test", 'statistic': np.nan, 'p_value': np.nan,
                     'reject_H0 (5%)': False, 'interpretation': f'Error: {e}'})

    # Breusch-Godfrey
    try:
        bg = breusch_godfrey(res, nlags=1)
        rows.append({
            'test': 'Breusch-Godfrey (Serial Correlation, lag=1)',
            'statistic': bg['stat'],
            'p_value': bg['pval'],
            'reject_H0 (5%)': bg['reject_h0'],
            'interpretation': 'Serial correlation present' if bg['reject_h0'] else 'No serial correlation',
        })
    except Exception as e:
        rows.append({'test': 'Breusch-Godfrey', 'statistic': np.nan, 'p_value': np.nan,
                     'reject_H0 (5%)': False, 'interpretation': f'Error: {e}'})

    # Ramsey RESET
    try:
        rr = ramsey_reset(res)
        rows.append({
            'test': 'Ramsey RESET (Functional Form)',
            'statistic': rr['F_stat'],
            'p_value': rr['pval'],
            'reject_H0 (5%)': rr['reject_h0'],
            'interpretation': 'Functional form misspecification' if rr['reject_h0'] else 'Correct functional form',
        })
    except Exception as e:
        rows.append({'test': 'Ramsey RESET', 'statistic': np.nan, 'p_value': np.nan,
                     'reject_H0 (5%)': False, 'interpretation': f'Error: {e}'})

    # Durbin-Watson
    try:
        dw = durbin_watson(res)
        rows.append({
            'test': 'Durbin-Watson (Autocorrelation)',
            'statistic': dw,
            'p_value': np.nan,
            'reject_H0 (5%)': dw < 1.5 or dw > 2.5,
            'interpretation': f'DW={dw:.3f}: {"Positive" if dw < 1.5 else "Negative" if dw > 2.5 else "No"} autocorrelation',
        })
    except Exception as e:
        rows.append({'test': 'Durbin-Watson', 'statistic': np.nan, 'p_value': np.nan,
                     'reject_H0 (5%)': False, 'interpretation': f'Error: {e}'})

    return pd.DataFrame(rows)
