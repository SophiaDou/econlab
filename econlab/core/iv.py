from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS as _IV2SLS
from typing import List, Optional, Dict, Any


def iv_2sls(
    df: pd.DataFrame,
    y: str,
    X_endog: List[str],
    X_exog: List[str],
    instruments: List[str],
    add_const: bool = True,
    robust: bool = True,
    cluster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Instrumental Variables (2SLS) estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    y : str
        Dependent variable column name.
    X_endog : list[str]
        Endogenous regressor column names.
    X_exog : list[str]
        Exogenous control column names (included instruments).
    instruments : list[str]
        Excluded instrument column names (IVs).
    add_const : bool
        Whether to add an intercept to the exogenous regressors.
    robust : bool
        Use heteroskedasticity-robust standard errors.
    cluster : str or None
        Column name for cluster-robust standard errors.

    Returns
    -------
    dict with keys:
        params, bse, tvalues, pvalues, nobs, rsq,
        first_stage (dict per endogenous var: F, p, partial_r2),
        sargan (dict: stat, pval, df) if overidentified else None,
        wu_hausman (dict: stat, pval),
        res (linearmodels result object)
    """
    # --- Input validation ---
    all_cols = [y] + list(X_endog) + list(X_exog) + list(instruments)
    missing = set(all_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    sub = df[all_cols]
    if not np.isfinite(sub.select_dtypes(include=[np.number]).values).all():
        raise ValueError("Non-finite (NaN/Inf) values detected in regression data. "
                         "Please clean your data before calling iv_2sls.")

    if len(X_endog) == 0:
        raise ValueError("X_endog must contain at least one endogenous variable.")
    if len(instruments) == 0:
        raise ValueError("instruments must contain at least one excluded instrument.")

    work = df[all_cols].dropna().copy()
    if len(work) == 0:
        raise ValueError("No observations remain after dropping NaN rows.")

    dep = work[y]
    endog = work[X_endog]
    exog = work[X_exog]
    instr = work[instruments]

    if add_const:
        exog = sm.add_constant(exog, has_constant='add')

    # --- Fit IV2SLS ---
    model = _IV2SLS(dep, exog, endog, instr)

    if cluster is not None:
        if cluster not in df.columns:
            raise ValueError(f"Cluster column '{cluster}' not found in df.")
        groups = work.index.map(lambda i: df.loc[i, cluster])
        # Re-align: need cluster for the rows in `work`
        orig_index = work.index
        groups = df.loc[orig_index, cluster]
        res = model.fit(cov_type='clustered', clusters=groups)
    elif robust:
        res = model.fit(cov_type='robust')
    else:
        res = model.fit(cov_type='unadjusted')

    # --- First stage ---
    first_stage: Dict[str, Dict[str, float]] = {}
    all_exog_instr = list(exog.columns) + list(instr.columns)
    first_stage_X = work[
        [c for c in all_exog_instr if c in work.columns]
    ]
    # Also add constant columns already in exog
    const_cols = [c for c in exog.columns if c not in work.columns]

    for endog_var in X_endog:
        fs_X = sm.add_constant(
            work[[c for c in X_exog] + list(instruments)],
            has_constant='add'
        )
        fs_model = sm.OLS(work[endog_var], fs_X)
        fs_res = fs_model.fit()

        # F-stat for excluded instruments
        n_instr = len(instruments)
        instr_names = [c for c in instruments if c in fs_res.params.index]
        if len(instr_names) == 0:
            first_stage[endog_var] = {'F': np.nan, 'p': np.nan, 'partial_r2': np.nan}
            continue

        R = np.zeros((len(instr_names), len(fs_res.params)))
        param_names = list(fs_res.params.index)
        for i, nm in enumerate(instr_names):
            R[i, param_names.index(nm)] = 1.0

        ftest = fs_res.f_test(R)
        F_stat = float(np.asarray(ftest.fvalue).ravel()[0])
        F_pval = float(np.asarray(ftest.pvalue).ravel()[0])

        # Partial R²: R² of restricted vs unrestricted first stage
        fs_res_restricted = sm.OLS(
            work[endog_var],
            sm.add_constant(work[X_exog], has_constant='add') if X_exog else pd.DataFrame(np.ones((len(work), 1)))
        ).fit()
        partial_r2 = max(0.0, fs_res.rsquared - fs_res_restricted.rsquared)

        if F_stat < 10:
            warnings.warn(
                f"[econlab.iv_2sls] Weak instrument warning for '{endog_var}': "
                f"first-stage F = {F_stat:.2f} (threshold: 10). "
                "Estimates may be unreliable.",
                stacklevel=2,
            )

        first_stage[endog_var] = {'F': F_stat, 'p': F_pval, 'partial_r2': partial_r2}

    # --- Sargan test (overidentification) ---
    n_endog = len(X_endog)
    n_instr_excl = len(instruments)
    sargan_result = None
    if n_instr_excl > n_endog:
        try:
            sargan = res.wooldridge_overid
            sargan_result = {
                'stat': float(sargan.stat),
                'pval': float(sargan.pval),
                'df': int(n_instr_excl - n_endog),
            }
        except Exception:
            # Fallback: manual Sargan
            try:
                fitted_endog = {}
                for ev in X_endog:
                    fs_X = sm.add_constant(
                        work[X_exog + list(instruments)], has_constant='add'
                    )
                    fitted_endog[ev] = sm.OLS(work[ev], fs_X).fit().fittedvalues

                Xhat = exog.copy()
                for ev in X_endog:
                    Xhat[ev] = fitted_endog[ev].values

                iv_resid = dep.values - Xhat.values @ np.linalg.lstsq(Xhat.values, dep.values, rcond=None)[0]
                Z_all = pd.concat([exog, instr], axis=1)
                nobs = len(iv_resid)
                ZZ_inv = np.linalg.pinv(Z_all.values.T @ Z_all.values)
                sargan_stat = nobs * float(iv_resid @ Z_all.values @ ZZ_inv @ Z_all.values.T @ iv_resid) / float(iv_resid @ iv_resid)
                from scipy import stats as _stats
                sargan_pval = 1 - _stats.chi2.cdf(abs(sargan_stat), df=n_instr_excl - n_endog)
                sargan_result = {
                    'stat': sargan_stat,
                    'pval': sargan_pval,
                    'df': n_instr_excl - n_endog,
                }
            except Exception:
                sargan_result = None

    # --- Wu-Hausman endogeneity test ---
    wu_hausman: Dict[str, float] = {'stat': np.nan, 'pval': np.nan}
    try:
        wh = res.wu_hausman
        wu_hausman = {'stat': float(wh.stat), 'pval': float(wh.pval)}
    except Exception:
        try:
            # Manual Wu-Hausman: include first-stage residuals in OLS
            fs_resids = {}
            for ev in X_endog:
                fs_X = sm.add_constant(
                    work[X_exog + list(instruments)], has_constant='add'
                )
                fs_fit = sm.OLS(work[ev], fs_X).fit()
                fs_resids[ev] = fs_fit.resid

            augmented = exog.copy()
            for ev in X_endog:
                augmented = pd.concat([augmented, work[ev].rename(ev)], axis=1)
            for ev, r in fs_resids.items():
                augmented[f'_v_{ev}'] = r.values

            ols_aug = sm.OLS(dep, augmented).fit(cov_type='HC1')
            v_names = [f'_v_{ev}' for ev in X_endog]
            R = np.zeros((len(v_names), len(ols_aug.params)))
            pnames = list(ols_aug.params.index)
            for i, nm in enumerate(v_names):
                if nm in pnames:
                    R[i, pnames.index(nm)] = 1.0
            ftest = ols_aug.f_test(R)
            wu_hausman = {
                'stat': float(np.asarray(ftest.fvalue).ravel()[0]),
                'pval': float(np.asarray(ftest.pvalue).ravel()[0]),
            }
        except Exception:
            pass

    return {
        'params': res.params,
        'bse': res.std_errors,
        'tvalues': res.tstats,
        'pvalues': res.pvalues,
        'nobs': int(res.nobs),
        'rsq': float(res.rsquared) if hasattr(res, 'rsquared') else np.nan,
        'first_stage': first_stage,
        'sargan': sargan_result,
        'wu_hausman': wu_hausman,
        'res': res,
    }
