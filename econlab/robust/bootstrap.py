from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable


def wild_cluster_bootstrap(
    model_func: Callable,
    df: pd.DataFrame,
    cluster: str,
    n_boot: int = 999,
    seed: Optional[int] = None,
    weight_type: str = 'rademacher',
) -> Dict[str, Any]:
    """
    Wild Cluster Bootstrap (Cameron, Gelbach & Miller 2008).

    Provides cluster-robust inference that is valid even with few clusters.
    Uses wild bootstrap weights applied at the cluster level.

    Parameters
    ----------
    model_func : callable
        A function that takes a DataFrame and returns a result dict with:
        - 'res': a statsmodels OLS result object (with .resid, .fittedvalues, .model.endog)
        - 'params': pd.Series of coefficients
    df : pd.DataFrame
        Input data.
    cluster : str
        Column name for the cluster variable.
    n_boot : int
        Number of bootstrap replications. Default 999.
    seed : int or None
        Random seed for reproducibility.
    weight_type : str
        Wild bootstrap weight distribution:
        - 'rademacher': +1 or -1 with equal probability (default)
        - 'mammen': continuous Mammen (1993) weights

    Returns
    -------
    dict with keys:
        boot_coefs (np.ndarray): Shape (n_boot, n_params) bootstrap coefficient draws
        boot_se (pd.Series): Bootstrap standard errors
        boot_pval (pd.Series): Bootstrap p-values (based on bootstrap t-distribution)
        boot_ci (pd.DataFrame): 95% confidence intervals [param, lower_2.5, upper_97.5]
        original_params (pd.Series): Original model coefficient estimates
    """
    if cluster not in df.columns:
        raise ValueError(f"Cluster column '{cluster}' not found in df.")

    valid_weights = {'rademacher', 'mammen'}
    if weight_type not in valid_weights:
        raise ValueError(f"weight_type must be one of {valid_weights}, got '{weight_type}'.")

    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}.")

    # Fit original model
    try:
        orig_result = model_func(df)
    except Exception as e:
        raise ValueError(f"model_func failed on original data: {e}")

    if 'res' not in orig_result:
        raise ValueError("model_func must return a dict with 'res' key (statsmodels result object).")

    orig_res = orig_result['res']
    orig_params = orig_result.get('params', orig_res.params)

    if not hasattr(orig_res, 'resid'):
        raise ValueError("The 'res' object must have a 'resid' attribute (statsmodels OLS result).")

    resid = orig_res.resid.values.astype(float)
    fitted = orig_res.fittedvalues.values.astype(float)
    y_orig = orig_res.model.endog.astype(float)
    n_obs = len(resid)
    n_params = len(orig_params)

    cluster_vals = df[cluster].values
    # Align cluster to residuals (handle possible index mismatch)
    if hasattr(orig_res, 'model') and hasattr(orig_res.model, 'data'):
        try:
            obs_idx = orig_res.model.data.row_labels
            if obs_idx is not None:
                cluster_aligned = pd.Series(cluster_vals, index=df.index).loc[obs_idx].values
            else:
                cluster_aligned = cluster_vals[:n_obs]
        except Exception:
            cluster_aligned = cluster_vals[:n_obs]
    else:
        cluster_aligned = cluster_vals[:n_obs]

    unique_clusters = np.unique(cluster_aligned)
    G = len(unique_clusters)

    if G < 2:
        warnings.warn(f"Only {G} cluster(s) found — wild cluster bootstrap may not be reliable.", stacklevel=2)

    rng = np.random.default_rng(seed)
    boot_coefs = np.full((n_boot, n_params), np.nan)

    # Pre-compute cluster memberships
    cluster_masks = {g: cluster_aligned == g for g in unique_clusters}

    for b in range(n_boot):
        # Draw wild weights at cluster level
        if weight_type == 'rademacher':
            cluster_weights = rng.choice([-1.0, 1.0], size=G, replace=True)
        else:  # mammen
            p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
            signs = rng.random(G)
            cluster_weights = np.where(
                signs < p,
                -(np.sqrt(5) - 1) / 2.0,
                (np.sqrt(5) + 1) / 2.0,
            )

        # Create bootstrap residuals: e*_it = e_it * w_g
        boot_resid = resid.copy()
        for gi, g in enumerate(unique_clusters):
            mask = cluster_masks[g]
            boot_resid[mask] = resid[mask] * cluster_weights[gi]

        # Bootstrap outcome: y* = ŷ + e*
        y_boot = fitted + boot_resid

        # Create bootstrap DataFrame
        df_boot = df.copy()

        # We need to replace the outcome in df_boot
        # Find the outcome column from the model
        try:
            endog_name = orig_res.model.endog_names
            if isinstance(endog_name, list):
                endog_name = endog_name[0]
        except Exception:
            endog_name = None

        if endog_name is not None and endog_name in df_boot.columns:
            # Align y_boot to df_boot index
            try:
                obs_idx = orig_res.model.data.row_labels
                if obs_idx is not None:
                    df_boot.loc[obs_idx, endog_name] = y_boot
                else:
                    df_boot.iloc[:n_obs, df_boot.columns.get_loc(endog_name)] = y_boot
            except Exception:
                df_boot[endog_name] = y_boot[:len(df_boot)]
        else:
            # Can't identify outcome column — skip this boot rep
            continue

        try:
            boot_result = model_func(df_boot)
            boot_params = boot_result.get('params', boot_result['res'].params)
            boot_coefs[b] = boot_params.values.astype(float)
        except Exception:
            # Leave as NaN for this rep
            continue

    # Remove failed reps
    valid_mask = ~np.isnan(boot_coefs).any(axis=1)
    boot_coefs_clean = boot_coefs[valid_mask]

    if len(boot_coefs_clean) == 0:
        warnings.warn("All bootstrap replications failed.", stacklevel=2)
        boot_coefs_clean = np.zeros((1, n_params))

    # Bootstrap SEs
    boot_se = pd.Series(np.std(boot_coefs_clean, axis=0, ddof=1), index=orig_params.index)

    # Bootstrap p-values (using bootstrap t-distribution)
    orig_se = orig_res.bse.values.astype(float)
    orig_t = orig_params.values / (orig_se + 1e-12)

    # Bootstrap t-stats: (β*_b - β) / se(β*)
    boot_t = (boot_coefs_clean - orig_params.values) / (boot_se.values + 1e-12)
    # p-value: proportion of |t*| > |t_orig|
    boot_pval_vals = np.mean(np.abs(boot_t) >= np.abs(orig_t), axis=0)
    boot_pval = pd.Series(boot_pval_vals, index=orig_params.index)

    # 95% CI
    ci_lower = np.percentile(boot_coefs_clean, 2.5, axis=0)
    ci_upper = np.percentile(boot_coefs_clean, 97.5, axis=0)
    boot_ci = pd.DataFrame({
        'param': orig_params.index,
        'lower_2.5': ci_lower,
        'upper_97.5': ci_upper,
    })

    return {
        'boot_coefs': boot_coefs_clean,
        'boot_se': boot_se,
        'boot_pval': boot_pval,
        'boot_ci': boot_ci,
        'original_params': orig_params,
    }
