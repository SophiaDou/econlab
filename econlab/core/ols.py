from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Optional, Dict, Any

def ols(df: pd.DataFrame, y: str, X: List[str], add_const: bool = True,
        cluster: Optional[pd.Series] = None, robust: bool = True) -> Dict[str, Any]:
    """Simple OLS with robust/clustered SE support.
    Strictly validates inputs:
      - No NaN/Inf in y or X
      - If `cluster` is provided, length must match and contain no NaN"""
    # --- Input validation ---
    cols = [y] + list(X)
    if not set(cols).issubset(df.columns):
        missing = set(cols) - set(df.columns)
        raise ValueError(f"Missing columns in df: {missing}")

    # Check for NaN/Inf in y and X
    sub = df[cols]
    if not np.isfinite(sub.values).all():
        bad_where = ~np.isfinite(sub)
        bad_counts = bad_where.sum()
        raise ValueError(
            "Non-finite values detected in regression data. "
            f"Counts per column: {bad_counts[bad_counts > 0].to_dict()}"
        )
    # Cluster validation
    if cluster is not None:
        if len(cluster) != len(df):
            raise ValueError(
                f"`cluster` length {len(cluster)} does not match nobs {len(df)}"
            )
        if not np.isfinite(cluster.values).all():
            raise ValueError("`cluster` contains NaN/Inf values.")

    Xmat = df[X]
    if add_const:
        Xmat = sm.add_constant(Xmat)
    model = sm.OLS(df[y], Xmat)
    if cluster is not None:
        res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster})
    elif robust:
        res = model.fit(cov_type='HC1')
    else:
        res = model.fit()
    return {
        'params': res.params,
        'bse': res.bse,
        'tvalues': res.tvalues,
        'pvalues': res.pvalues,
        'nobs': int(res.nobs),
        'rsq': float(res.rsquared) if hasattr(res, 'rsquared') else np.nan,
        'res': res
    }