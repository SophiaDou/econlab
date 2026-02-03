from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Optional, Dict, Any


def ols(df: pd.DataFrame, y: str, X: List[str], add_const: bool = True,
        cluster: Optional[pd.Series] = None, robust: bool = True) -> Dict[str, Any]:
    """Simple OLS with robust/clustered SE support."""
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