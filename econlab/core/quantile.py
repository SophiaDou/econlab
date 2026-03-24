from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from typing import List, Dict, Any, Tuple


def quantile_reg(
    df: pd.DataFrame,
    y: str,
    X: List[str],
    quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    add_const: bool = True,
) -> Dict[float, Dict[str, Any]]:
    """
    Quantile regression for distributional treatment effect analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    y : str
        Outcome variable column name.
    X : list[str]
        Regressor column names.
    quantiles : tuple[float]
        Quantiles to estimate. Default: (0.1, 0.25, 0.5, 0.75, 0.9).
    add_const : bool
        Whether to add an intercept. Default True.

    Returns
    -------
    dict mapping each quantile q -> {params, bse, tvalues, pvalues, nobs, res}

    Raises
    ------
    ValueError
        If columns are missing or data contains non-finite values.
    """
    all_cols = [y] + list(X)
    missing = set(all_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    sub = df[all_cols].dropna()
    if len(sub) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    if not np.isfinite(sub.values.astype(float)).all():
        raise ValueError("Non-finite (NaN/Inf) values detected in regression data.")

    for q in quantiles:
        if not (0.0 < q < 1.0):
            raise ValueError(f"All quantiles must be in (0, 1), got {q}.")

    dep = sub[y].values.astype(float)
    Xmat = sub[X].values.astype(float)
    if add_const:
        Xmat = sm.add_constant(Xmat, has_constant='add')
        col_names = ['const'] + list(X)
    else:
        col_names = list(X)

    results: Dict[float, Dict[str, Any]] = {}
    for q in quantiles:
        model = QuantReg(dep, Xmat)
        res = model.fit(q=q, max_iter=1000)
        results[q] = {
            'params': pd.Series(res.params, index=col_names),
            'bse': pd.Series(res.bse, index=col_names),
            'tvalues': pd.Series(res.tvalues, index=col_names),
            'pvalues': pd.Series(res.pvalues, index=col_names),
            'nobs': int(res.nobs),
            'res': res,
        }

    return results


def quantile_summary(results_dict: Dict[float, Dict[str, Any]]) -> pd.DataFrame:
    """
    Summarize quantile regression results across quantiles.

    Parameters
    ----------
    results_dict : dict
        Output of quantile_reg().

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with (variable, quantile) rows and
        columns: [coef, se, ci_lower_95, ci_upper_95].
    """
    rows = []
    for q, res in sorted(results_dict.items()):
        params = res['params']
        bse = res['bse']
        for var in params.index:
            coef = params[var]
            se = bse[var]
            rows.append({
                'quantile': q,
                'variable': var,
                'coef': coef,
                'se': se,
                'ci_lower_95': coef - 1.96 * se,
                'ci_upper_95': coef + 1.96 * se,
            })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.set_index(['variable', 'quantile']).sort_index()
    return df_out
