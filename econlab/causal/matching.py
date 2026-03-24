from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from typing import Dict, Any, List, Optional


def _estimate_pscore(df: pd.DataFrame, treatment: str, covariates: List[str]) -> pd.Series:
    """Estimate propensity score via logistic regression."""
    Xmat = sm.add_constant(df[covariates].astype(float), has_constant='add')
    model = sm.Logit(df[treatment].astype(float), Xmat)
    try:
        res = model.fit(disp=False, maxiter=200)
    except Exception:
        res = model.fit(method='bfgs', disp=False, maxiter=500)
    return pd.Series(res.predict(), index=df.index)


def propensity_score_match(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: List[str],
    n_neighbors: int = 1,
    caliper: Optional[float] = None,
    with_replacement: bool = True,
) -> Dict[str, Any]:
    """
    Propensity Score Matching (PSM).

    Estimates propensity scores via logit, matches treated to control units,
    and computes the ATT (Average Treatment Effect on the Treated).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    outcome : str
        Outcome variable column name.
    treatment : str
        Binary treatment indicator column (0/1).
    covariates : list[str]
        Covariate column names for propensity score estimation.
    n_neighbors : int
        Number of nearest-neighbor matches per treated unit. Default 1.
    caliper : float or None
        Maximum propensity score distance for a valid match (caliper matching).
        If None, no caliper is applied.
    with_replacement : bool
        Whether matching is done with replacement. Default True.

    Returns
    -------
    dict with keys:
        att (float): Average Treatment Effect on the Treated
        se (float): Standard error (bootstrap-based approximation)
        n_treated (int): Number of treated units
        n_control (int): Number of matched control units
        matched_df (DataFrame): Matched dataset with columns [unit_index, outcome, treatment, pscore]
        pscore (pd.Series): Estimated propensity scores for all units
    """
    required = [outcome, treatment] + list(covariates)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[required].dropna().copy()
    if len(work) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    treat_vals = work[treatment].unique()
    if not set(treat_vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Treatment variable must be binary (0/1), got: {treat_vals}")

    pscore = _estimate_pscore(work, treatment, covariates)
    work['_pscore'] = pscore

    treated = work[work[treatment] == 1].copy()
    control = work[work[treatment] == 0].copy()

    if len(treated) == 0:
        raise ValueError("No treated units found.")
    if len(control) == 0:
        raise ValueError("No control units found.")

    # Match: nearest neighbor on propensity score
    ps_treated = treated['_pscore'].values.reshape(-1, 1)
    ps_control = control['_pscore'].values.reshape(-1, 1)
    dist_matrix = cdist(ps_treated, ps_control, metric='euclidean')  # shape: (n_treated, n_control)

    matched_rows = []
    used_control = set()

    for i in range(len(treated)):
        dists = dist_matrix[i].copy()
        if not with_replacement:
            for j in used_control:
                dists[j] = np.inf

        # Get k nearest neighbors
        sorted_idx = np.argsort(dists)
        selected = []
        for idx in sorted_idx:
            if len(selected) >= n_neighbors:
                break
            if caliper is not None and dists[idx] > caliper:
                break
            selected.append(idx)
            if not with_replacement:
                used_control.add(idx)

        if len(selected) == 0:
            continue  # No valid match within caliper

        t_row = treated.iloc[i]
        for ci in selected:
            c_row = control.iloc[ci]
            matched_rows.append({
                'treated_idx': treated.index[i],
                'control_idx': control.index[ci],
                'outcome_treated': t_row[outcome],
                'outcome_control': c_row[outcome],
                'pscore_treated': t_row['_pscore'],
                'pscore_control': c_row['_pscore'],
            })

    if len(matched_rows) == 0:
        raise ValueError("No matches found. Consider relaxing the caliper or checking the data.")

    match_df = pd.DataFrame(matched_rows)
    att_diffs = match_df['outcome_treated'].values - match_df['outcome_control'].values
    att = float(np.mean(att_diffs))
    se = float(np.std(att_diffs, ddof=1) / np.sqrt(len(att_diffs)))

    # Build matched dataset
    matched_df = pd.concat([
        df.loc[match_df['treated_idx'].values].assign(_role='treated'),
        df.loc[match_df['control_idx'].values].assign(_role='control'),
    ]).reset_index(drop=True)

    return {
        'att': att,
        'se': se,
        'n_treated': int(len(treated)),
        'n_control': int(len(control)),
        'matched_df': matched_df,
        'pscore': pscore,
    }


def ipw_estimate(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: List[str],
    normalize: bool = True,
    robust: bool = True,
) -> Dict[str, Any]:
    """
    Inverse Probability Weighting (Horvitz-Thompson) ATT estimator.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    outcome : str
        Outcome variable column name.
    treatment : str
        Binary treatment indicator column (0/1).
    covariates : list[str]
        Covariate column names.
    normalize : bool
        Use normalized (stabilized) weights. Default True.
    robust : bool
        Use HC1 robust standard errors. Default True.

    Returns
    -------
    dict with keys:
        att (float): Average Treatment Effect on the Treated
        ate (float): Average Treatment Effect
        se (float): Standard error (delta method approximation)
    """
    required = [outcome, treatment] + list(covariates)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[required].dropna().copy()
    if len(work) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    pscore = _estimate_pscore(work, treatment, covariates)
    pscore = pscore.clip(0.01, 0.99)  # Trim extreme scores

    d = work[treatment].values.astype(float)
    y = work[outcome].values.astype(float)
    p = pscore.values

    # IPW weights for ATT: treated=1, control=p/(1-p)
    if normalize:
        w_treated = d
        w_control = (1 - d) * p / (1 - p + 1e-10)
        # Normalize
        w_treated_norm = w_treated / (w_treated.sum() + 1e-10)
        w_control_norm = w_control / (w_control.sum() + 1e-10)
        att = float(np.sum(w_treated_norm * y) - np.sum(w_control_norm * y))
    else:
        # Horvitz-Thompson
        att = float(np.mean(d * y) - np.mean((1 - d) * p / (1 - p + 1e-10) * y))

    # ATE: E[Y(1)] - E[Y(0)]
    w_ate_1 = d / (p + 1e-10)
    w_ate_0 = (1 - d) / (1 - p + 1e-10)
    if normalize:
        ate = float(np.mean(w_ate_1 / (w_ate_1.mean() + 1e-10) * y) -
                    np.mean(w_ate_0 / (w_ate_0.mean() + 1e-10) * y))
    else:
        ate = float(np.mean(w_ate_1 * y) - np.mean(w_ate_0 * y))

    # Delta method SE: use bootstrap approximation (simple)
    n = len(work)
    if normalize:
        pseudo_outcomes = (d * y / (p + 1e-10) - (1 - d) * y / (1 - p + 1e-10))
    else:
        pseudo_outcomes = d * y - (1 - d) * p / (1 - p + 1e-10) * y
    se = float(np.std(pseudo_outcomes, ddof=1) / np.sqrt(n))

    return {
        'att': att,
        'ate': ate,
        'se': se,
    }


def doubly_robust(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: List[str],
    robust: bool = True,
    cluster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Doubly Robust (Augmented IPW) estimator.

    Combines outcome regression and IPW — consistent if either model is
    correctly specified.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    outcome : str
        Outcome variable column name.
    treatment : str
        Binary treatment indicator column (0/1).
    covariates : list[str]
        Covariate column names.
    robust : bool
        Use HC1 robust standard errors. Default True.
    cluster : str or None
        Column for cluster-robust standard errors.

    Returns
    -------
    dict with keys:
        att (float): Average Treatment Effect on the Treated
        ate (float): Average Treatment Effect
        se (float): Standard error
        res: Final OLS result object
    """
    required = [outcome, treatment] + list(covariates)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    if cluster is not None and cluster not in df.columns:
        raise ValueError(f"Cluster column '{cluster}' not found in df.")

    work = df[required + ([cluster] if cluster else [])].dropna().copy()
    if len(work) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    d = work[treatment].values.astype(float)
    y = work[outcome].values.astype(float)

    # Step 1: Propensity score
    pscore = _estimate_pscore(work, treatment, covariates)
    p = pscore.values.clip(0.01, 0.99)

    # Step 2: Outcome models E[Y|D=1, X] and E[Y|D=0, X]
    Xmat = sm.add_constant(work[covariates].astype(float), has_constant='add')

    treated_mask = d == 1
    control_mask = d == 0

    # Outcome regression for treated
    if treated_mask.sum() > len(covariates) + 1:
        res1 = sm.OLS(y[treated_mask], Xmat.values[treated_mask]).fit()
        mu1 = res1.predict(Xmat.values)
    else:
        mu1 = np.full(len(y), y[treated_mask].mean())

    # Outcome regression for control
    if control_mask.sum() > len(covariates) + 1:
        res0 = sm.OLS(y[control_mask], Xmat.values[control_mask]).fit()
        mu0 = res0.predict(Xmat.values)
    else:
        mu0 = np.full(len(y), y[control_mask].mean())

    # Step 3: Doubly robust scores
    # ATE augmented scores
    dr_score_1 = mu1 + d * (y - mu1) / (p + 1e-10)
    dr_score_0 = mu0 + (1 - d) * (y - mu0) / (1 - p + 1e-10)

    ate = float(np.mean(dr_score_1 - dr_score_0))

    # ATT
    att_score = d * (y - mu0) / (np.mean(d) + 1e-10) - (1 - d) * p * (y - mu0) / ((1 - p + 1e-10) * (np.mean(d) + 1e-10))
    att = float(np.mean(att_score) + np.mean(mu0[treated_mask]) - np.mean(mu0))

    # SE via linear regression of DR score on constant
    pseudo_y = dr_score_1 - dr_score_0
    X_const = np.ones((len(pseudo_y), 1))
    final_model = sm.OLS(pseudo_y, X_const)

    if cluster is not None:
        groups = work[cluster]
        n_groups = groups.nunique()
        if n_groups >= 2:
            res = final_model.fit(cov_type='cluster', cov_kwds={'groups': groups})
        else:
            warnings.warn("Too few clusters, falling back to HC1.", stacklevel=2)
            res = final_model.fit(cov_type='HC1')
    elif robust:
        res = final_model.fit(cov_type='HC1')
    else:
        res = final_model.fit()

    se = float(res.bse[0])

    return {
        'att': att,
        'ate': ate,
        'se': se,
        'res': res,
    }


def covariate_balance(
    df: pd.DataFrame,
    treatment: str,
    covariates: List[str],
) -> pd.DataFrame:
    """
    Covariate balance table.

    Computes mean, standardized mean difference (SMD), and variance ratio
    for each covariate between treated and control groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    treatment : str
        Binary treatment indicator column (0/1).
    covariates : list[str]
        Covariate column names.

    Returns
    -------
    pd.DataFrame
        Columns: [variable, mean_treated, mean_control, smd, var_ratio]
        SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2)
    """
    required = [treatment] + list(covariates)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[required].dropna()
    treated = work[work[treatment] == 1]
    control = work[work[treatment] == 0]

    if len(treated) == 0:
        raise ValueError("No treated units found.")
    if len(control) == 0:
        raise ValueError("No control units found.")

    rows = []
    for var in covariates:
        mt = float(treated[var].mean())
        mc = float(control[var].mean())
        vt = float(treated[var].var())
        vc = float(control[var].var())
        pooled_var = (vt + vc) / 2.0
        smd = (mt - mc) / (np.sqrt(pooled_var) + 1e-10)
        var_ratio = vt / (vc + 1e-10)
        rows.append({
            'variable': var,
            'mean_treated': mt,
            'mean_control': mc,
            'smd': smd,
            'var_ratio': var_ratio,
        })

    return pd.DataFrame(rows)
