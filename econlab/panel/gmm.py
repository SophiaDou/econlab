from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional


def _build_diff_instruments(Y_panel: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Build instrument matrix for difference GMM (Arellano-Bond).

    For each time period t, instruments are lagged levels y_{t-2}, y_{t-3}, ...

    Parameters
    ----------
    Y_panel : np.ndarray
        Shape (n_units, n_times), the outcome in levels.
    max_lag : int
        Maximum lag order to use.

    Returns
    -------
    np.ndarray
        Stacked instrument matrix.
    """
    n, T = Y_panel.shape
    rows = []
    for i in range(n):
        for t in range(2, T):  # t starts at index 2 (3rd period) for diffs
            row_instruments = []
            for lag in range(2, t + 1):
                val = Y_panel[i, t - lag] if t - lag >= 0 else 0.0
                row_instruments.append(val if not np.isnan(val) else 0.0)
            rows.append(row_instruments)
    # Pad rows to same length
    max_len = max(len(r) for r in rows) if rows else 1
    Z_rows = np.array([r + [0.0] * (max_len - len(r)) for r in rows])
    return Z_rows


def diff_gmm(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    controls: Optional[List[str]] = None,
    lags: int = 1,
) -> Dict[str, Any]:
    """
    Arellano-Bond (1991) Difference GMM for dynamic panel models.

    Model: Δy_it = α·Δy_{i,t-1} + β·ΔX_it + Δε_it
    Instruments: lagged levels y_{i,t-2}, y_{i,t-3}, ... for the differenced equation.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    controls : list[str] or None
        Additional control variable column names.
    lags : int
        Number of lagged dependent variable lags to include. Default 1.

    Returns
    -------
    dict with keys:
        params (pd.Series): Parameter estimates (lag coefficients + controls)
        bse (pd.Series): Standard errors
        ar1_test (dict: stat, pval): Arellano-Bond AR(1) test
        ar2_test (dict: stat, pval): Arellano-Bond AR(2) test
        sargan (dict: stat, pval, df): Sargan overidentification test
        nobs (int): Number of observations used
        res: dict with additional details
    """
    required = [y, unit, time] + (controls or [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[required].dropna().copy()
    work[time] = pd.to_numeric(work[time], errors='coerce')
    work = work.sort_values([unit, time])

    units = work[unit].unique()
    times = sorted(work[time].unique())
    T = len(times)
    N = len(units)

    if T < 3:
        raise ValueError("Difference GMM requires at least 3 time periods.")

    # Map time to index
    time_map = {t: i for i, t in enumerate(times)}
    work['_t_idx'] = work[time].map(time_map)

    # Pivot to (N, T) arrays
    Y_level = np.full((N, T), np.nan)
    unit_map = {u: i for i, u in enumerate(units)}

    ctrl_levels = {c: np.full((N, T), np.nan) for c in (controls or [])}

    for _, row in work.iterrows():
        i = unit_map[row[unit]]
        t = int(row['_t_idx'])
        Y_level[i, t] = row[y]
        for c in (controls or []):
            ctrl_levels[c][i, t] = row[c]

    # First difference
    dY = np.diff(Y_level, axis=1)  # shape: (N, T-1)
    dCtrl = {c: np.diff(ctrl_levels[c], axis=1) for c in (controls or [])}

    # Build stacked differenced data for t = lags+1, ..., T-1 (0-indexed in diff space: t = lags, ..., T-2)
    rows_X = []
    rows_y = []
    rows_Z = []  # instruments
    row_units = []

    for i in range(N):
        for t in range(lags, T - 1):  # t in differenced space (t+1 in level space)
            # Dependent: Δy_{t+1} (in level time, period t+1, so diff index t)
            dy = dY[i, t]
            if np.isnan(dy):
                continue

            # Regressors: Δy_{t}, Δy_{t-1}, ..., plus ΔX
            x_row = []
            valid = True
            for lag_k in range(1, lags + 1):
                if t - lag_k < 0:
                    valid = False
                    break
                dx_lag = dY[i, t - lag_k + 1 - 1]  # Δy_{t-lag_k+1} in diff-space
                # Actually: Δy at diff-time (t - lag_k + 1)
                diff_idx = t - lag_k
                if diff_idx < 0:
                    valid = False
                    break
                dx_lag = dY[i, diff_idx]
                if np.isnan(dx_lag):
                    valid = False
                    break
                x_row.append(dx_lag)

            if not valid:
                continue

            for c in (controls or []):
                dc = dCtrl[c][i, t]
                if np.isnan(dc):
                    valid = False
                    break
                x_row.append(dc)

            if not valid or len(x_row) == 0:
                continue

            # Instruments: y_{i, t-1} and further lags in levels
            # t in diff-space corresponds to level time t+1 (0-indexed)
            # So we use y_{i, 0}, ..., y_{i, t-1} as instruments
            z_row = []
            for lag_k in range(2, t + 2):  # level lags >= 2
                level_t = t + 1 - lag_k  # level time index
                if level_t >= 0 and not np.isnan(Y_level[i, level_t]):
                    z_row.append(Y_level[i, level_t])

            if len(z_row) == 0:
                continue

            rows_X.append(x_row)
            rows_y.append(dy)
            rows_Z.append(z_row)
            row_units.append(i)

    if len(rows_y) == 0:
        raise ValueError("No valid observations for Difference GMM. "
                         "Check that the panel has enough time periods.")

    # Pad to equal length
    max_x = max(len(r) for r in rows_X)
    max_z = max(len(r) for r in rows_Z)
    X_mat = np.array([r + [0.0] * (max_x - len(r)) for r in rows_X])
    y_vec = np.array(rows_y)
    Z_mat = np.array([r + [0.0] * (max_z - len(r)) for r in rows_Z])

    n_obs = len(y_vec)
    n_params = X_mat.shape[1]
    n_instr = Z_mat.shape[1]

    if n_instr < n_params:
        warnings.warn("Fewer instruments than parameters — model may be underidentified.", stacklevel=2)

    # One-step GMM: A = (Z'Z)^{-1}
    ZZ = Z_mat.T @ Z_mat
    try:
        A = np.linalg.pinv(ZZ)
    except Exception:
        A = np.eye(n_instr) / n_obs

    # β = (X'ZAZ'X)^{-1} X'ZAZ'y
    XZ = X_mat.T @ Z_mat  # (n_params, n_instr)
    XZAZ = XZ @ A @ Z_mat.T  # (n_params, n_obs)
    try:
        beta = np.linalg.lstsq(XZAZ @ X_mat, XZAZ @ y_vec, rcond=None)[0]
    except Exception:
        beta = np.zeros(n_params)

    resid = y_vec - X_mat @ beta

    # Two-step GMM with optimal weighting matrix
    S = np.zeros((n_instr, n_instr))
    for i_obs in range(n_obs):
        z = Z_mat[i_obs]
        S += (resid[i_obs] ** 2) * np.outer(z, z)
    S = S / n_obs

    try:
        S_inv = np.linalg.pinv(S)
        XZS = XZ @ S_inv @ Z_mat.T
        beta2 = np.linalg.lstsq(XZS @ X_mat, XZS @ y_vec, rcond=None)[0]
        resid2 = y_vec - X_mat @ beta2

        # Covariance matrix
        Meat = np.zeros((n_instr, n_instr))
        for i_obs in range(n_obs):
            z = Z_mat[i_obs]
            Meat += (resid2[i_obs] ** 2) * np.outer(z, z)
        Meat /= n_obs

        Bread = XZ @ S_inv @ Z_mat.T @ X_mat / n_obs
        try:
            Bread_inv = np.linalg.pinv(Bread)
            Var = Bread_inv @ (XZ @ S_inv @ Meat @ S_inv @ XZ.T / n_obs) @ Bread_inv.T / n_obs
        except Exception:
            Var = np.eye(n_params) * (np.var(resid2) / n_obs)

        beta = beta2
        resid = resid2
    except Exception:
        Var = np.eye(n_params) * (np.var(resid) / n_obs + 1e-8)

    bse = np.sqrt(np.maximum(np.diag(Var), 0.0))

    param_names = [f'L{k}.{y}' for k in range(1, lags + 1)] + (controls or [])
    param_names = param_names[:n_params]

    params = pd.Series(beta, index=param_names)
    bse_series = pd.Series(bse, index=param_names)

    # AR(1) and AR(2) tests (Arellano-Bond)
    def _ar_test(resid_vec, lag):
        n_r = len(resid_vec)
        if n_r <= lag:
            return {'stat': np.nan, 'pval': np.nan}
        r1 = resid_vec[lag:]
        r2 = resid_vec[:-lag]
        min_len = min(len(r1), len(r2))
        cov = np.mean(r1[:min_len] * r2[:min_len])
        var1 = np.var(r1[:min_len])
        var2 = np.var(r2[:min_len])
        se_cov = np.sqrt(var1 * var2 / min_len + 1e-12)
        z = cov / se_cov if se_cov > 0 else np.nan
        pval = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
        return {'stat': float(z) if not np.isnan(z) else np.nan, 'pval': pval}

    ar1_test = _ar_test(resid, 1)
    ar2_test = _ar_test(resid, 2)

    # Sargan test
    J = float(n_obs * resid @ Z_mat @ np.linalg.pinv(Z_mat.T @ Z_mat) @ Z_mat.T @ resid / (resid @ resid + 1e-12))
    sargan_df = max(n_instr - n_params, 1)
    sargan_pval = float(1 - stats.chi2.cdf(abs(J), df=sargan_df))

    return {
        'params': params,
        'bse': bse_series,
        'ar1_test': ar1_test,
        'ar2_test': ar2_test,
        'sargan': {'stat': J, 'pval': sargan_pval, 'df': sargan_df},
        'nobs': n_obs,
        'res': {'X': X_mat, 'Z': Z_mat, 'y': y_vec, 'resid': resid},
    }


def sys_gmm(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    controls: Optional[List[str]] = None,
    lags: int = 1,
) -> Dict[str, Any]:
    """
    Blundell-Bond (1998) System GMM.

    Combines the differenced equation (with lagged level instruments) and
    the level equation (with lagged difference instruments).

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    controls : list[str] or None
        Additional control variable column names.
    lags : int
        Number of lagged dependent variable lags to include. Default 1.

    Returns
    -------
    dict with keys:
        params (pd.Series): Parameter estimates
        bse (pd.Series): Standard errors
        ar1_test (dict: stat, pval): AR(1) test
        ar2_test (dict: stat, pval): AR(2) test
        sargan (dict: stat, pval, df): Sargan/Hansen overidentification test
        hansen_j (dict: stat, pval, df): Hansen J test (same as sargan for sys GMM)
        nobs (int): Number of observations
        res: dict with additional details
    """
    # System GMM: run difference GMM and augment
    # For simplicity, we extend the difference GMM by also including level equations
    required = [y, unit, time] + (controls or [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[required].dropna().copy()
    work[time] = pd.to_numeric(work[time], errors='coerce')
    work = work.sort_values([unit, time])

    units = work[unit].unique()
    times = sorted(work[time].unique())
    T = len(times)
    N = len(units)

    if T < 3:
        raise ValueError("System GMM requires at least 3 time periods.")

    time_map = {t: i for i, t in enumerate(times)}
    work['_t_idx'] = work[time].map(time_map)

    Y_level = np.full((N, T), np.nan)
    unit_map = {u: i for i, u in enumerate(units)}
    ctrl_levels = {c: np.full((N, T), np.nan) for c in (controls or [])}

    for _, row in work.iterrows():
        i = unit_map[row[unit]]
        t = int(row['_t_idx'])
        Y_level[i, t] = row[y]
        for c in (controls or []):
            ctrl_levels[c][i, t] = row[c]

    dY = np.diff(Y_level, axis=1)
    dCtrl = {c: np.diff(ctrl_levels[c], axis=1) for c in (controls or [])}

    rows_X_diff = []
    rows_y_diff = []
    rows_Z_diff = []

    rows_X_lev = []
    rows_y_lev = []
    rows_Z_lev = []

    for i in range(N):
        for t in range(lags, T - 1):
            dy = dY[i, t]
            if np.isnan(dy):
                continue

            x_row = []
            valid = True
            for lag_k in range(1, lags + 1):
                diff_idx = t - lag_k
                if diff_idx < 0:
                    valid = False
                    break
                dx_lag = dY[i, diff_idx]
                if np.isnan(dx_lag):
                    valid = False
                    break
                x_row.append(dx_lag)

            if not valid:
                continue

            for c in (controls or []):
                dc = dCtrl[c][i, t]
                if np.isnan(dc):
                    valid = False
                    break
                x_row.append(dc)

            if not valid:
                continue

            z_diff = []
            for lag_k in range(2, t + 2):
                level_t = t + 1 - lag_k
                if level_t >= 0 and not np.isnan(Y_level[i, level_t]):
                    z_diff.append(Y_level[i, level_t])

            if len(z_diff) == 0:
                continue

            rows_X_diff.append(x_row)
            rows_y_diff.append(dy)
            rows_Z_diff.append(z_diff)

        # Level equation: y_{t+1} = α·y_t + ... with Δy_{t-1} as instrument
        for t in range(lags, T):
            y_lev = Y_level[i, t]
            if np.isnan(y_lev):
                continue

            x_row_l = []
            valid = True
            for lag_k in range(1, lags + 1):
                if t - lag_k < 0:
                    valid = False
                    break
                yl_lag = Y_level[i, t - lag_k]
                if np.isnan(yl_lag):
                    valid = False
                    break
                x_row_l.append(yl_lag)

            if not valid:
                continue

            for c in (controls or []):
                cl_val = ctrl_levels[c][i, t]
                if np.isnan(cl_val):
                    valid = False
                    break
                x_row_l.append(cl_val)

            if not valid:
                continue

            # Instrument: lagged differences
            z_lev = []
            if t >= 2 and not np.isnan(dY[i, t - 2]):
                z_lev.append(dY[i, t - 2])
            if t >= 1 and not np.isnan(dY[i, t - 1]):
                z_lev.append(dY[i, t - 1])

            if len(z_lev) == 0:
                continue

            rows_X_lev.append(x_row_l)
            rows_y_lev.append(y_lev)
            rows_Z_lev.append(z_lev)

    if len(rows_y_diff) == 0 and len(rows_y_lev) == 0:
        raise ValueError("No valid observations for System GMM.")

    def _pad(rows, max_len=None):
        if not rows:
            return np.zeros((0, max_len or 1))
        ml = max(len(r) for r in rows)
        if max_len is not None:
            ml = max(ml, max_len)
        return np.array([r + [0.0] * (ml - len(r)) for r in rows])

    max_x = max(
        max((len(r) for r in rows_X_diff), default=0),
        max((len(r) for r in rows_X_lev), default=0),
    )

    X_diff = _pad(rows_X_diff, max_x)
    y_diff = np.array(rows_y_diff) if rows_y_diff else np.zeros(0)
    Z_diff = _pad(rows_Z_diff)

    X_lev = _pad(rows_X_lev, max_x)
    y_lev_arr = np.array(rows_y_lev) if rows_y_lev else np.zeros(0)
    Z_lev = _pad(rows_Z_lev)

    # Stack system
    n_diff = len(y_diff)
    n_lev = len(y_lev_arr)
    n_obs = n_diff + n_lev

    if n_obs == 0:
        raise ValueError("No valid observations for System GMM.")

    # Build block-diagonal Z matrix
    n_z_diff = Z_diff.shape[1] if Z_diff.ndim > 1 and Z_diff.shape[0] > 0 else 0
    n_z_lev = Z_lev.shape[1] if Z_lev.ndim > 1 and Z_lev.shape[0] > 0 else 0

    Z_sys = np.zeros((n_obs, n_z_diff + n_z_lev))
    if n_diff > 0 and n_z_diff > 0:
        Z_sys[:n_diff, :n_z_diff] = Z_diff
    if n_lev > 0 and n_z_lev > 0:
        Z_sys[n_diff:, n_z_diff:n_z_diff + n_z_lev] = Z_lev

    X_sys = np.vstack([X_diff, X_lev]) if (n_diff > 0 and n_lev > 0) else (X_diff if n_diff > 0 else X_lev)
    y_sys = np.concatenate([y_diff, y_lev_arr]) if (n_diff > 0 and n_lev > 0) else (y_diff if n_diff > 0 else y_lev_arr)

    n_params = X_sys.shape[1]
    n_instr = Z_sys.shape[1]

    # One-step GMM
    ZZ = Z_sys.T @ Z_sys
    A = np.linalg.pinv(ZZ)
    XZ = X_sys.T @ Z_sys
    XZAZ = XZ @ A @ Z_sys.T
    try:
        beta = np.linalg.lstsq(XZAZ @ X_sys, XZAZ @ y_sys, rcond=None)[0]
    except Exception:
        beta = np.zeros(n_params)

    resid = y_sys - X_sys @ beta

    # Two-step
    S = np.zeros((n_instr, n_instr))
    for i_obs in range(n_obs):
        z = Z_sys[i_obs]
        S += (resid[i_obs] ** 2) * np.outer(z, z)
    S = S / n_obs + np.eye(n_instr) * 1e-8

    try:
        S_inv = np.linalg.pinv(S)
        XZS = XZ @ S_inv @ Z_sys.T
        beta2 = np.linalg.lstsq(XZS @ X_sys, XZS @ y_sys, rcond=None)[0]
        resid2 = y_sys - X_sys @ beta2

        Bread = XZ @ S_inv @ Z_sys.T @ X_sys / n_obs
        Bread_inv = np.linalg.pinv(Bread)
        Meat2 = np.zeros((n_instr, n_instr))
        for i_obs in range(n_obs):
            z = Z_sys[i_obs]
            Meat2 += (resid2[i_obs] ** 2) * np.outer(z, z)
        Meat2 /= n_obs
        Var = Bread_inv @ (XZ @ S_inv @ Meat2 @ S_inv @ XZ.T / n_obs) @ Bread_inv.T / n_obs
        beta = beta2
        resid = resid2
    except Exception:
        Var = np.eye(n_params) * (np.var(resid) / n_obs + 1e-8)

    bse = np.sqrt(np.maximum(np.diag(Var), 0.0))
    param_names = [f'L{k}.{y}' for k in range(1, lags + 1)] + (controls or [])
    param_names = param_names[:n_params]

    params = pd.Series(beta, index=param_names)
    bse_series = pd.Series(bse, index=param_names)

    def _ar_test(res_vec, lag):
        n_r = len(res_vec)
        if n_r <= lag:
            return {'stat': np.nan, 'pval': np.nan}
        r1 = res_vec[lag:]
        r2 = res_vec[:-lag]
        min_len = min(len(r1), len(r2))
        cov = np.mean(r1[:min_len] * r2[:min_len])
        se_cov = np.sqrt(np.var(r1[:min_len]) * np.var(r2[:min_len]) / min_len + 1e-12)
        z = cov / se_cov if se_cov > 0 else np.nan
        pval = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
        return {'stat': float(z) if not np.isnan(z) else np.nan, 'pval': pval}

    ar1_test = _ar_test(resid, 1)
    ar2_test = _ar_test(resid, 2)

    J = float(n_obs * resid @ Z_sys @ np.linalg.pinv(Z_sys.T @ Z_sys) @ Z_sys.T @ resid / (resid @ resid + 1e-12))
    sargan_df = max(n_instr - n_params, 1)
    sargan_pval = float(1 - stats.chi2.cdf(abs(J), df=sargan_df))
    hansen_j = {'stat': J, 'pval': sargan_pval, 'df': sargan_df}

    return {
        'params': params,
        'bse': bse_series,
        'ar1_test': ar1_test,
        'ar2_test': ar2_test,
        'sargan': {'stat': J, 'pval': sargan_pval, 'df': sargan_df},
        'hansen_j': hansen_j,
        'nobs': n_obs,
        'res': {'X': X_sys, 'Z': Z_sys, 'y': y_sys, 'resid': resid},
    }
