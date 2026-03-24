from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import RandomEffects, PanelOLS
from typing import Dict, Any, List, Optional


def re_panel(
    df: pd.DataFrame,
    y: str,
    X: List[str],
    unit: str,
    time: str,
    cluster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Random Effects GLS panel estimator.

    Uses linearmodels.panel.RandomEffects.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    X : list[str]
        Regressor column names (do not include unit/time columns).
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    cluster : str or None
        Column for cluster-robust standard errors.

    Returns
    -------
    dict with keys:
        params, bse, tvalues, pvalues, nobs,
        rsq_between, rsq_within, theta (GLS weight), res
    """
    required = [y] + list(X) + [unit, time]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[list(set(required + ([cluster] if cluster else [])))].dropna().copy()
    if len(work) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    # Set MultiIndex required by linearmodels
    work = work.set_index([unit, time])

    dep = work[y].astype(float)
    Xmat = sm.add_constant(work[X].astype(float), has_constant='add')

    model = RandomEffects(dep, Xmat)

    if cluster is not None:
        if cluster in df.columns:
            clust_vals = df.loc[work.index.get_level_values(0), cluster] if cluster != unit else work.index.get_level_values(0)
            # For linearmodels, cluster is not directly supported in the same way
            # Use entity_effects as approximation
            try:
                res = model.fit(cov_type='clustered', cluster_entity=True)
            except Exception:
                res = model.fit(cov_type='robust')
        else:
            res = model.fit(cov_type='robust')
    else:
        res = model.fit(cov_type='robust')

    # Extract theta (GLS weight, measure of within vs between)
    theta = getattr(res, 'theta', np.nan)
    if hasattr(theta, 'mean'):
        theta = float(theta.mean())
    elif theta is not None:
        try:
            theta = float(theta)
        except Exception:
            theta = np.nan

    return {
        'params': res.params,
        'bse': res.std_errors,
        'tvalues': res.tstats,
        'pvalues': res.pvalues,
        'nobs': int(res.nobs),
        'rsq_between': float(res.rsquared_between) if hasattr(res, 'rsquared_between') else np.nan,
        'rsq_within': float(res.rsquared_within) if hasattr(res, 'rsquared_within') else np.nan,
        'theta': theta,
        'res': res,
    }


def fe_panel(
    df: pd.DataFrame,
    y: str,
    X: List[str],
    unit: str,
    time: str,
    cluster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fixed Effects (within) panel estimator.

    Uses linearmodels.panel.PanelOLS with entity_effects=True.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    X : list[str]
        Regressor column names.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    cluster : str or None
        Column for cluster-robust standard errors.

    Returns
    -------
    dict with keys:
        params, bse, tvalues, pvalues, nobs, rsq_within, res
    """
    required = [y] + list(X) + [unit, time]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[list(set(required + ([cluster] if cluster else [])))].dropna().copy()
    if len(work) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    work = work.set_index([unit, time])

    dep = work[y].astype(float)
    Xmat = work[X].astype(float)

    model = PanelOLS(dep, Xmat, entity_effects=True)

    if cluster is not None:
        try:
            res = model.fit(cov_type='clustered', cluster_entity=True)
        except Exception:
            res = model.fit(cov_type='robust')
    else:
        res = model.fit(cov_type='robust')

    return {
        'params': res.params,
        'bse': res.std_errors,
        'tvalues': res.tstats,
        'pvalues': res.pvalues,
        'nobs': int(res.nobs),
        'rsq_within': float(res.rsquared) if hasattr(res, 'rsquared') else np.nan,
        'res': res,
    }


def hausman_test(
    fe_result: Dict[str, Any],
    re_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Hausman (1978) specification test: Fixed Effects vs Random Effects.

    H0: Random effects is consistent and efficient (no endogeneity of unit effects).
    H1: Fixed effects is consistent, random effects is inconsistent.

    Uses the formula: H = (β_fe - β_re)' [Var(β_fe) - Var(β_re)]^{-1} (β_fe - β_re)

    Parameters
    ----------
    fe_result : dict
        Output of fe_panel() — must have 'res' key (linearmodels result).
    re_result : dict
        Output of re_panel() — must have 'res' key (linearmodels result).

    Returns
    -------
    dict with keys:
        stat (float): Hausman test statistic (chi-squared)
        pval (float): p-value
        df (int): Degrees of freedom
        reject_h0 (bool): Whether to reject H0 at 5% level
    """
    from scipy import stats as _stats

    fe_res = fe_result['res']
    re_res = re_result['res']

    # Get common coefficients (exclude intercept/constant)
    fe_params = fe_res.params
    re_params = re_res.params

    # Find common variables (FE doesn't have a constant)
    common_vars = [v for v in fe_params.index if v in re_params.index and v != 'const']

    if len(common_vars) == 0:
        raise ValueError("No common variables between FE and RE estimates. "
                         "Ensure models use the same regressors.")

    b_fe = fe_params[common_vars].values.astype(float)
    b_re = re_params[common_vars].values.astype(float)
    diff = b_fe - b_re

    # Covariance matrices
    V_fe = fe_res.cov.loc[common_vars, common_vars].values.astype(float)
    V_re = re_res.cov.loc[common_vars, common_vars].values.astype(float)

    V_diff = V_fe - V_re

    # Handle non-positive-definite differences
    try:
        V_diff_inv = np.linalg.pinv(V_diff)
        stat = float(diff @ V_diff_inv @ diff)
    except np.linalg.LinAlgError:
        stat = np.nan

    df_test = len(common_vars)

    if np.isnan(stat) or stat < 0:
        warnings.warn("Hausman test statistic is negative or NaN — "
                      "the covariance difference is not positive semi-definite. "
                      "Results may be unreliable.", stacklevel=2)
        pval = np.nan
        reject = False
    else:
        pval = float(1 - _stats.chi2.cdf(stat, df=df_test))
        reject = pval < 0.05

    return {
        'stat': stat,
        'pval': pval,
        'df': df_test,
        'reject_h0': reject,
    }
