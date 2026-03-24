from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, List, Optional


def synthetic_control(
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: str,
    pre_periods: List,
    predictor_vars: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Abadie-Diamond-Hainmueller Synthetic Control.

    Constructs a synthetic control unit as a weighted combination of donor
    units that best matches the treated unit in the pre-treatment period.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treated_unit : str
        Value in the `unit` column identifying the treated unit.
    pre_periods : list
        List of time values to use as the pre-treatment period.
    predictor_vars : list[str] or None
        Additional predictor variables to match on.
        If None, uses pre-period outcome values as predictors.

    Returns
    -------
    dict with keys:
        weights (pd.Series): Donor unit weights (indexed by unit name)
        synthetic (pd.Series): Synthetic outcome over all time periods
        gaps (pd.Series): Treated outcome minus synthetic outcome
        pre_rmspe (float): Pre-treatment RMSPE
        post_rmspe (float): Post-treatment RMSPE
    """
    required = [outcome, unit, time]
    if predictor_vars:
        required += predictor_vars
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df.copy()
    work[outcome] = pd.to_numeric(work[outcome], errors='coerce')
    work[time] = pd.to_numeric(work[time], errors='coerce') if pd.api.types.is_numeric_dtype(work[time]) else work[time]

    all_units = work[unit].unique()
    if treated_unit not in all_units:
        raise ValueError(f"treated_unit '{treated_unit}' not found in column '{unit}'.")

    donor_units = [u for u in all_units if u != treated_unit]
    if len(donor_units) == 0:
        raise ValueError("No donor units available (all units are the treated unit).")

    all_times = sorted(work[time].unique())
    post_periods = [t for t in all_times if t not in pre_periods]

    # Pivot to wide format: rows = time, columns = units
    pivot = work.pivot_table(index=time, columns=unit, values=outcome)

    if treated_unit not in pivot.columns:
        raise ValueError(f"No outcome data found for treated unit '{treated_unit}'.")

    donor_cols = [u for u in donor_units if u in pivot.columns]
    if len(donor_cols) == 0:
        raise ValueError("No donor units have outcome data.")

    # Build pre-period matrices
    pre_mask = pivot.index.isin(pre_periods)
    Y_pre_treated = pivot.loc[pre_mask, treated_unit].values.astype(float)
    Y_pre_donors = pivot.loc[pre_mask, donor_cols].values.astype(float)  # shape: (n_pre, n_donors)

    if predictor_vars:
        # Build additional predictors: average over pre-period
        pred_treated = []
        pred_donors = []
        for var in predictor_vars:
            var_pivot = work.pivot_table(index=time, columns=unit, values=var)
            var_pre = var_pivot.loc[pre_mask]
            if treated_unit in var_pivot.columns:
                pred_treated.append(var_pre[treated_unit].mean())
            else:
                pred_treated.append(np.nan)
            donor_means = var_pre[donor_cols].mean().values if all(u in var_pivot.columns for u in donor_cols) else np.zeros(len(donor_cols))
            pred_donors.append(donor_means)

        pred_treated = np.array(pred_treated)
        pred_donors = np.array(pred_donors)  # shape: (n_predictors, n_donors)

        # Stack with pre-period outcomes
        X_treated = np.concatenate([Y_pre_treated, pred_treated])
        X_donors = np.vstack([Y_pre_donors, pred_donors.T]).T  # shape: (n_donors, n_pre + n_pred)
        # Normalize each row
        scale = np.std(X_treated) + 1e-8
        X_treated_norm = X_treated / scale
        X_donors_norm = X_donors / scale
    else:
        X_treated_norm = Y_pre_treated / (np.std(Y_pre_treated) + 1e-8)
        X_donors_norm = Y_pre_donors / (np.std(Y_pre_treated) + 1e-8)

    n_donors = len(donor_cols)

    # Drop donors with any NaN
    valid_donors = ~np.isnan(X_donors_norm).any(axis=0)
    if not valid_donors.any():
        raise ValueError("All donor units have missing data in pre-period.")

    X_donors_clean = X_donors_norm[:, valid_donors]
    donor_cols_clean = [donor_cols[i] for i in range(n_donors) if valid_donors[i]]
    n_clean = len(donor_cols_clean)

    # Drop rows with NaN in treated
    valid_rows = ~np.isnan(X_treated_norm)
    X_treated_clean = X_treated_norm[valid_rows]
    X_donors_clean = X_donors_clean[valid_rows, :]

    def _objective(w):
        synth = X_donors_clean @ w
        return float(np.sum((X_treated_clean - synth) ** 2))

    def _gradient(w):
        synth = X_donors_clean @ w
        diff = synth - X_treated_clean
        return 2.0 * X_donors_clean.T @ diff

    # Constraints: weights sum to 1, weights >= 0
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_clean
    w0 = np.ones(n_clean) / n_clean

    result = minimize(
        _objective,
        w0,
        jac=_gradient,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9},
    )

    weights_clean = result.x
    weights_clean = np.maximum(weights_clean, 0.0)
    weights_clean /= weights_clean.sum() + 1e-12

    weights = pd.Series(0.0, index=donor_cols)
    for i, col in enumerate(donor_cols_clean):
        weights[col] = weights_clean[i]

    # Compute synthetic outcome over all time periods
    Y_donors_all = pivot[donor_cols].values  # shape: (n_times, n_donors)
    synthetic_vals = Y_donors_all @ weights.values  # shape: (n_times,)

    synthetic = pd.Series(synthetic_vals, index=pivot.index, name='synthetic')
    treated_outcome = pivot[treated_unit]
    gaps = treated_outcome - synthetic

    # RMSPE
    pre_gaps = gaps[gaps.index.isin(pre_periods)].dropna()
    post_gaps = gaps[gaps.index.isin(post_periods)].dropna()
    pre_rmspe = float(np.sqrt((pre_gaps ** 2).mean())) if len(pre_gaps) > 0 else np.nan
    post_rmspe = float(np.sqrt((post_gaps ** 2).mean())) if len(post_gaps) > 0 else np.nan

    return {
        'weights': weights,
        'synthetic': synthetic,
        'gaps': gaps,
        'pre_rmspe': pre_rmspe,
        'post_rmspe': post_rmspe,
    }
