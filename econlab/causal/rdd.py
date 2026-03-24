from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Any, Optional, List


def _kernel_weights(x: np.ndarray, bandwidth: float, kernel: str) -> np.ndarray:
    """
    Compute kernel weights for local polynomial regression.

    Parameters
    ----------
    x : np.ndarray
        Distances from the cutoff (running variable - cutoff).
    bandwidth : float
        Bandwidth for the kernel.
    kernel : str
        Kernel type: 'triangular', 'uniform', or 'epanechnikov'.

    Returns
    -------
    np.ndarray
        Array of non-negative kernel weights.
    """
    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth}.")

    u = x / bandwidth
    if kernel == 'triangular':
        w = np.maximum(1.0 - np.abs(u), 0.0)
    elif kernel == 'uniform':
        w = (np.abs(u) <= 1.0).astype(float)
    elif kernel == 'epanechnikov':
        w = np.maximum(0.75 * (1.0 - u ** 2), 0.0)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'triangular', 'uniform', or 'epanechnikov'.")
    return w


def _ik_bandwidth(y: np.ndarray, running: np.ndarray, cutoff: float) -> float:
    """
    Imbens-Kalyanaraman (2012) MSE-optimal bandwidth selector (simplified version).

    Parameters
    ----------
    y : np.ndarray
        Outcome variable.
    running : np.ndarray
        Running variable.
    cutoff : float
        Cutoff value.

    Returns
    -------
    float
        Estimated optimal bandwidth.
    """
    n = len(y)
    r = running - cutoff

    # Split into left and right
    left_mask = r < 0
    right_mask = r >= 0

    if left_mask.sum() < 5 or right_mask.sum() < 5:
        # Fallback: use simple rule of thumb
        return float(np.std(r) * n ** (-1.0 / 5.0) * 2.0)

    # Estimate variance and second derivatives (local quadratic on each side)
    def _local_quad_fit(r_side, y_side):
        if len(r_side) < 4:
            return np.var(y_side), 0.0
        X = np.column_stack([np.ones(len(r_side)), r_side, r_side ** 2])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y_side, rcond=None)
            resid = y_side - X @ coef
            sigma2 = np.var(resid)
            m2 = coef[2]  # second derivative / 2
            return sigma2, m2
        except Exception:
            return np.var(y_side), 0.0

    sigma2_l, m2_l = _local_quad_fit(r[left_mask], y[left_mask])
    sigma2_r, m2_r = _local_quad_fit(r[right_mask], y[right_mask])

    # Bias and variance components
    m2 = (m2_l + m2_r) / 2.0
    sigma2 = (sigma2_l + sigma2_r) / 2.0
    f_x = 1.0 / (2.0 * np.std(r) + 1e-8)  # rough density estimate at cutoff

    # CK kernel constants for triangular kernel
    C_K = 3.4375  # 55 / 16

    if abs(m2) < 1e-10:
        m2 = 1e-10

    h = C_K * (sigma2 / (f_x * (m2 ** 2) * n)) ** (1.0 / 5.0)
    h = max(h, np.std(r) * 0.05)  # ensure minimum bandwidth
    return float(h)


def rdd_sharp(
    df: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    kernel: str = 'triangular',
    poly_order: int = 1,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sharp Regression Discontinuity Design.

    Fits local polynomial regression on each side of the cutoff and computes
    the discontinuity (treatment effect) at the cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    y : str
        Outcome variable column name.
    running : str
        Running variable column name.
    cutoff : float
        Cutoff value. Default 0.0.
    bandwidth : float or None
        Bandwidth for local regression. If None, uses IK bandwidth selector.
    kernel : str
        Kernel for local regression: 'triangular', 'uniform', or 'epanechnikov'.
    poly_order : int
        Polynomial order for local regression (1 = local linear). Default 1.
    controls : list[str] or None
        Additional covariates.
    cluster : str or None
        Column for cluster-robust standard errors.

    Returns
    -------
    dict with keys:
        tau (float): RDD treatment effect estimate
        se (float): Standard error
        pval (float): p-value
        bandwidth (float): Bandwidth used
        nobs_left (int): Observations left of cutoff
        nobs_right (int): Observations right of cutoff
        ci_95 (tuple): 95% confidence interval (lower, upper)
        res_left: OLS result for left side
        res_right: OLS result for right side
    """
    required = [y, running] + (controls or [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[required + ([cluster] if cluster else [])].dropna().copy()
    if len(work) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    r = work[running].values.astype(float) - cutoff
    yv = work[y].values.astype(float)

    if bandwidth is None:
        bandwidth = _ik_bandwidth(yv, r + cutoff, cutoff)

    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth}.")

    w = _kernel_weights(r, bandwidth, kernel)

    # Local polynomial on each side
    left_mask = (r < 0) & (w > 0)
    right_mask = (r >= 0) & (w > 0)

    def _build_poly(r_vals, poly_order):
        cols = [np.ones(len(r_vals))]
        for p in range(1, poly_order + 1):
            cols.append(r_vals ** p)
        return np.column_stack(cols)

    def _fit_side(r_vals, y_vals, w_vals, ctrl_vals=None, cluster_vals=None):
        X = _build_poly(r_vals, poly_order)
        if ctrl_vals is not None:
            X = np.column_stack([X, ctrl_vals])
        W = np.diag(w_vals)
        XtW = X.T @ W
        try:
            model = sm.WLS(y_vals, X, weights=w_vals)
            if cluster_vals is not None and len(np.unique(cluster_vals)) >= 2:
                res = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_vals})
            else:
                res = model.fit(cov_type='HC1')
            return res
        except Exception as e:
            raise ValueError(f"Local polynomial fitting failed: {e}")

    ctrl_left = work.loc[left_mask, controls].values if controls else None
    ctrl_right = work.loc[right_mask, controls].values if controls else None
    clust_left = work.loc[left_mask, cluster].values if cluster else None
    clust_right = work.loc[right_mask, cluster].values if cluster else None

    if left_mask.sum() < poly_order + 2:
        raise ValueError(f"Too few observations left of cutoff ({left_mask.sum()}) for poly_order={poly_order}.")
    if right_mask.sum() < poly_order + 2:
        raise ValueError(f"Too few observations right of cutoff ({right_mask.sum()}) for poly_order={poly_order}.")

    res_left = _fit_side(r[left_mask], yv[left_mask], w[left_mask], ctrl_left, clust_left)
    res_right = _fit_side(r[right_mask], yv[right_mask], w[right_mask], ctrl_right, clust_right)

    # Intercept = predicted value at cutoff (r=0)
    mu_left = float(res_left.params[0])
    mu_right = float(res_right.params[0])
    tau = mu_right - mu_left

    se_left = float(res_left.bse[0])
    se_right = float(res_right.bse[0])
    se = float(np.sqrt(se_left ** 2 + se_right ** 2))

    t_stat = tau / se if se > 0 else np.nan
    df_resid = left_mask.sum() + right_mask.sum() - 2 * (poly_order + 1)
    pval = float(2 * stats.t.sf(abs(t_stat), df=max(df_resid, 1)))
    ci_95 = (tau - 1.96 * se, tau + 1.96 * se)

    return {
        'tau': tau,
        'se': se,
        'pval': pval,
        'bandwidth': bandwidth,
        'nobs_left': int(left_mask.sum()),
        'nobs_right': int(right_mask.sum()),
        'ci_95': ci_95,
        'res_left': res_left,
        'res_right': res_right,
    }


def rdd_fuzzy(
    df: pd.DataFrame,
    y: str,
    running: str,
    treatment: str,
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    kernel: str = 'triangular',
    poly_order: int = 1,
) -> Dict[str, Any]:
    """
    Fuzzy RDD via IV: instrument = 1(running >= cutoff), endogenous = treatment.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    y : str
        Outcome variable column name.
    running : str
        Running variable column name.
    treatment : str
        Endogenous treatment indicator column.
    cutoff : float
        Cutoff value. Default 0.0.
    bandwidth : float or None
        Bandwidth. If None, uses IK selector.
    kernel : str
        Kernel type.
    poly_order : int
        Polynomial order. Default 1.

    Returns
    -------
    dict with keys:
        late (float): Local Average Treatment Effect
        se (float): Standard error
        pval (float): p-value
        first_stage_F (float): First-stage F-statistic
        bandwidth (float): Bandwidth used
        ci_95 (tuple): 95% CI
        res: IV result (reduced form / first stage dict)
    """
    required = [y, running, treatment]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[required].dropna().copy()
    if len(work) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    r = work[running].values.astype(float) - cutoff
    yv = work[y].values.astype(float)
    dv = work[treatment].values.astype(float)

    if bandwidth is None:
        bandwidth = _ik_bandwidth(yv, r + cutoff, cutoff)

    w = _kernel_weights(r, bandwidth, kernel)
    in_bw = w > 0

    if in_bw.sum() < 2 * (poly_order + 2):
        raise ValueError(f"Too few observations within bandwidth ({in_bw.sum()}).")

    r_bw = r[in_bw]
    y_bw = yv[in_bw]
    d_bw = dv[in_bw]
    w_bw = w[in_bw]

    # Instrument: above cutoff
    z_bw = (r_bw >= 0).astype(float)

    # Build polynomial features
    X_poly_cols = [np.ones(in_bw.sum())]
    for p in range(1, poly_order + 1):
        X_poly_cols.append(r_bw ** p)
    X_poly = np.column_stack(X_poly_cols)

    # First stage: D ~ Z + poly(r)
    X_fs = np.column_stack([z_bw, X_poly[:, 1:], np.ones(in_bw.sum())])
    fs_model = sm.WLS(d_bw, X_fs, weights=w_bw)
    fs_res = fs_model.fit(cov_type='HC1')
    first_stage_F = float(fs_res.fvalue) if hasattr(fs_res, 'fvalue') else np.nan

    # Reduced form: Y ~ Z + poly(r)
    rf_model = sm.WLS(y_bw, X_fs, weights=w_bw)
    rf_res = rf_model.fit(cov_type='HC1')

    # LATE = reduced form / first stage at Z coefficient
    rf_coef = float(rf_res.params[0])
    fs_coef = float(fs_res.params[0])

    if abs(fs_coef) < 1e-10:
        raise ValueError("First stage coefficient is near zero — instrument is too weak.")

    late = rf_coef / fs_coef
    # Delta method SE
    se_rf = float(rf_res.bse[0])
    se_fs = float(fs_res.bse[0])
    se = float(abs(late) * np.sqrt((se_rf / (rf_coef + 1e-10)) ** 2 + (se_fs / (fs_coef + 1e-10)) ** 2))

    t_stat = late / se if se > 0 else np.nan
    df_resid = in_bw.sum() - X_fs.shape[1]
    pval = float(2 * stats.t.sf(abs(t_stat), df=max(df_resid, 1)))
    ci_95 = (late - 1.96 * se, late + 1.96 * se)

    return {
        'late': late,
        'se': se,
        'pval': pval,
        'first_stage_F': first_stage_F,
        'bandwidth': bandwidth,
        'ci_95': ci_95,
        'res': {'rf': rf_res, 'fs': fs_res},
    }


def mccrary_density_test(
    df: pd.DataFrame,
    running: str,
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
) -> Dict[str, Any]:
    """
    McCrary (2008) density discontinuity test for manipulation at the cutoff.

    Tests whether there is a discontinuity in the density of the running variable
    at the cutoff, which would indicate sorting/manipulation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    running : str
        Running variable column name.
    cutoff : float
        Cutoff value. Default 0.0.
    bandwidth : float or None
        Bandwidth for kernel density estimation. If None, uses a rule-of-thumb.

    Returns
    -------
    dict with keys:
        theta (float): Log difference in density height at cutoff
        se (float): Standard error of theta
        z (float): z-statistic
        pval (float): p-value (two-sided)
    """
    if running not in df.columns:
        raise ValueError(f"Column '{running}' not found in df.")

    r = df[running].dropna().values.astype(float)
    n = len(r)
    if n < 20:
        raise ValueError("Need at least 20 observations for McCrary density test.")

    if bandwidth is None:
        bandwidth = 1.84 * np.std(r) * n ** (-1.0 / 5.0)

    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth}.")

    # Bin width (Scott's rule within bandwidth)
    h = 2.0 * bandwidth * n ** (-1.0 / 3.0) / (2 * bandwidth / bandwidth)
    h = max(h, (max(r) - min(r)) / 200)

    # Create bins
    left = r[(r >= cutoff - bandwidth) & (r < cutoff)]
    right = r[(r >= cutoff) & (r < cutoff + bandwidth)]

    if len(left) < 5 or len(right) < 5:
        return {'theta': np.nan, 'se': np.nan, 'z': np.nan, 'pval': np.nan}

    # Density estimate on each side using kernel density at the boundary
    # Use local linear density estimation (simplified)
    bins_left = np.arange(cutoff - bandwidth, cutoff, h)
    bins_right = np.arange(cutoff, cutoff + bandwidth + h, h)

    def _bin_counts(data, bins):
        counts = []
        centers = []
        for i in range(len(bins) - 1):
            cnt = np.sum((data >= bins[i]) & (data < bins[i+1]))
            counts.append(cnt)
            centers.append((bins[i] + bins[i+1]) / 2.0)
        return np.array(centers), np.array(counts, dtype=float)

    c_left, cnt_left = _bin_counts(r, bins_left)
    c_right, cnt_right = _bin_counts(r, bins_right)

    if len(c_left) < 2 or len(c_right) < 2:
        return {'theta': np.nan, 'se': np.nan, 'z': np.nan, 'pval': np.nan}

    # Normalize counts to density
    n_total = len(r)
    dens_left = cnt_left / (n_total * h)
    dens_right = cnt_right / (n_total * h)

    # Local linear fit on each side to estimate density at cutoff
    def _ll_at_boundary(centers, densities, boundary):
        dist = centers - boundary
        X = np.column_stack([np.ones(len(dist)), dist])
        w = np.maximum(1.0 - np.abs(dist / (bandwidth)), 0.0)
        pos_w = w > 0
        if pos_w.sum() < 2:
            return np.mean(densities), np.std(densities) / np.sqrt(len(densities))
        model = sm.WLS(densities[pos_w], X[pos_w], weights=w[pos_w])
        res = model.fit()
        return float(res.params[0]), float(res.bse[0])

    f_left, se_left = _ll_at_boundary(c_left, dens_left, cutoff)
    f_right, se_right = _ll_at_boundary(c_right, dens_right, cutoff)

    if f_left <= 0 or f_right <= 0:
        theta = f_right - f_left
        se_theta = np.sqrt(se_left ** 2 + se_right ** 2)
    else:
        theta = np.log(f_right) - np.log(f_left)
        se_theta = float(np.sqrt((se_left / (f_left + 1e-10)) ** 2 + (se_right / (f_right + 1e-10)) ** 2))

    z = theta / se_theta if se_theta > 0 else np.nan
    pval = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan

    return {
        'theta': float(theta),
        'se': float(se_theta),
        'z': float(z) if not np.isnan(z) else np.nan,
        'pval': pval,
    }
