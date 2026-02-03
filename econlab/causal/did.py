from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, Tuple, Dict, Any, List

def _coerce_numeric(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Coerce to numeric and drop rows with NA in model columns; return Xn, yn, mask."""
    Xn = X.apply(pd.to_numeric, errors='coerce')
    yn = pd.to_numeric(y, errors='coerce')
    mask = Xn.notna().all(axis=1) & yn.notna()
    return Xn.loc[mask], yn.loc[mask], mask

def twfe_did(df: pd.DataFrame, y: str, treatment: str, unit: str, time: str,
             controls: Optional[List[str]] = None,
             cluster: Optional[str] = None,
             add_fe: bool = True) -> Dict[str, Any]:
    """TWFE DID with unit/time FE; robust numeric casting & safe clustering."""
    work = df.copy()

    # Build regressors
    X_cols: List[str] = [treatment] + (controls or [])
    if add_fe:
        work = pd.get_dummies(work, columns=[unit, time], drop_first=True)
        fe_cols = [c for c in work.columns if c.startswith(unit + '_') or c.startswith(time + '_')]
        X_cols += fe_cols

    # Drop fully-missing columns first (e.g., a control entirely NaN)
    X = work[X_cols]
    X = X.loc[:, ~X.isna().all(axis=0)]
    yv = work[y]

    # Coerce & filter rows
    Xn, yn, mask = _coerce_numeric(X, yv)
    if Xn.shape[0] == 0:
        raise ValueError("No usable rows after dropping NaNs in regressors/outcome. "
                         "Check that your controls/FE are not entirely missing.")
    Xn = sm.add_constant(Xn, has_constant='add').astype(float)
    yn = yn.astype(float)

    model = sm.OLS(yn, Xn)

    # --- Safe cluster handling ---
    if cluster is not None:
        g_aligned = df.loc[mask, cluster]
        n_groups = pd.Series(g_aligned).dropna().nunique()
        if n_groups >= 2:
            res = model.fit(cov_type='cluster', cov_kwds={'groups': g_aligned})
        else:
            warnings.warn(f"[econlab.twfe_did] Only {n_groups} cluster found after filtering; "
                          f"falling back to HC1 robust SE.")
            res = model.fit(cov_type='HC1')
    else:
        res = model.fit(cov_type='HC1')

    return {
        'coef_D': float(res.params.get(treatment, np.nan)),
        'se_D': float(res.bse.get(treatment, np.nan)),
        'params': res.params,
        'bse': res.bse,
        'summary': res.summary().as_text(),
        'res': res
    }

def build_event_time(df: pd.DataFrame, unit: str, time: str, treat_start: str) -> pd.Series:
    tt = pd.to_numeric(df[time], errors='coerce')
    ts = pd.to_numeric(df[treat_start], errors='coerce')
    return tt - ts

def event_study(df: pd.DataFrame, y: str, unit: str, time: str, treat_start: str,
                window: Tuple[int, int] = (-5, 5), baseline: int = -1,
                controls: Optional[List[str]] = None,
                cluster: Optional[str] = None,
                add_fe: bool = True) -> Dict[str, Any]:
    """Event-study with robust numeric casting & safe clustering."""
    work = df.copy()
    work['k'] = build_event_time(work, unit, time, treat_start)

    lo, hi = window
    k_cols = []
    for kk in range(lo, hi + 1):
        if kk == baseline:
            continue
        col = f'k_{kk}'
        work[col] = (work['k'] == kk).astype(int)
        k_cols.append(col)

    X_cols: List[str] = k_cols + (controls or [])
    if add_fe:
        work = pd.get_dummies(work, columns=[unit, time], drop_first=True)
        fe_cols = [c for c in work.columns if c.startswith(unit + '_') or c.startswith(time + '_')]
        X_cols += fe_cols

    X = work[X_cols]
    X = X.loc[:, ~X.isna().all(axis=0)]
    yv = work[y]

    Xn, yn, mask = _coerce_numeric(X, yv)
    if Xn.shape[0] == 0:
        raise ValueError("No usable rows after dropping NaNs in regressors/outcome for event-study.")
    Xn = sm.add_constant(Xn, has_constant='add').astype(float)
    yn = yn.astype(float)

    model = sm.OLS(yn, Xn)
    if cluster is not None:
        g_aligned = df.loc[mask, cluster]
        n_groups = pd.Series(g_aligned).dropna().nunique()
        if n_groups >= 2:
            res = model.fit(cov_type='cluster', cov_kwds={'groups': g_aligned})
        else:
            warnings.warn(f"[econlab.event_study] Only {n_groups} cluster found; falling back to HC1.")
            res = model.fit(cov_type='HC1')
    else:
        res = model.fit(cov_type='HC1')

    ev_params = {c: res.params.get(c, np.nan) for c in k_cols}
    ev_se = {c: res.bse.get(c, np.nan) for c in k_cols}
    return {'event_params': ev_params, 'event_se': ev_se, 'res': res}

def parallel_trend_test(res_dict: Dict[str, Any], pre_k: List[int]) -> Dict[str, Any]:
    """Joint F-test that pre-period coefficients equal zero."""
    res = res_dict['res']
    names = [f'k_{k}' for k in pre_k]
    R = np.zeros((len(names), len(res.params)))
    param_names = list(res.params.index)
    for i, nm in enumerate(names):
        if nm in param_names:
            R[i, param_names.index(nm)] = 1.0
    ftest = res.f_test(R)
    return {'F': float(np.asarray(ftest.fvalue)), 'pval': float(np.asarray(ftest.pvalue))}