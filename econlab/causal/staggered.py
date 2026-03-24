from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any, List, Optional


def sun_abraham(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    cohort: str,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sun & Abraham (2021) interaction-weighted estimator for staggered DiD.

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
    cohort : str
        Treatment cohort column — the period when unit first received treatment.
        Should be NaN for never-treated units.
    controls : list[str] or None
        Additional covariates.
    cluster : str or None
        Column for cluster-robust standard errors.

    Returns
    -------
    dict with keys:
        att_dynamic (dict: rel_time -> ATT(l)),
        cohort_atts (DataFrame with columns [cohort, rel_time, coef, se]),
        res (statsmodels result object)
    """
    required = [y, unit, time, cohort]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df.copy()
    work[cohort] = pd.to_numeric(work[cohort], errors='coerce')
    work[time] = pd.to_numeric(work[time], errors='coerce')

    cohort_vals = sorted(work[cohort].dropna().unique())
    if len(cohort_vals) == 0:
        raise ValueError("No treated units found (all cohort values are NaN).")

    time_vals = sorted(work[time].dropna().unique())
    t_min = min(time_vals)
    t_max = max(time_vals)

    # Relative time
    work['_rel_time'] = work[time] - work[cohort]

    # Create cohort × relative-time interactions
    interaction_cols: List[str] = []
    cohort_rel_pairs: List[tuple] = []

    for g in cohort_vals:
        g_mask = work[cohort] == g
        rel_times_in_data = work.loc[g_mask, '_rel_time'].dropna().unique()
        for l in sorted(rel_times_in_data):
            col = f'_D_g{int(g)}_l{int(l)}'
            work[col] = ((work[cohort] == g) & (work['_rel_time'] == l)).astype(float)
            interaction_cols.append(col)
            cohort_rel_pairs.append((g, l))

    # Remove baseline (l == -1) to avoid collinearity
    baseline_cols = [c for c, (g, l) in zip(interaction_cols, cohort_rel_pairs) if l == -1]
    reg_interaction_cols = [c for c in interaction_cols if c not in baseline_cols]
    reg_pairs = [(g, l) for c, (g, l) in zip(interaction_cols, cohort_rel_pairs) if c not in baseline_cols]

    # Unit and time FE via dummies
    X_cols = list(reg_interaction_cols) + (controls or [])
    work_fe = pd.get_dummies(work, columns=[unit, time], drop_first=True)
    fe_cols = [c for c in work_fe.columns
               if c.startswith(unit + '_') or c.startswith(time + '_')]
    X_cols_fe = X_cols + fe_cols

    # Drop NaN rows
    all_needed = [y] + X_cols_fe
    all_needed = [c for c in all_needed if c in work_fe.columns]
    sub = work_fe[all_needed].dropna()
    if len(sub) == 0:
        raise ValueError("No observations remain after dropping NaNs.")

    Xmat = sm.add_constant(sub[[c for c in X_cols_fe if c in sub.columns]], has_constant='add').astype(float)
    yvec = sub[y].astype(float)

    model = sm.OLS(yvec, Xmat)
    if cluster is not None and cluster in df.columns:
        groups = df.loc[sub.index, cluster]
        if groups.nunique() >= 2:
            res = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
        else:
            warnings.warn("Too few clusters, falling back to HC1.", stacklevel=2)
            res = model.fit(cov_type='HC1')
    else:
        res = model.fit(cov_type='HC1')

    # --- Aggregate to dynamic ATT(l) by weighting over cohorts ---
    # Weight for cohort g at rel_time l: share of cohort in the relevant sample
    cohort_sizes = {g: (work[cohort] == g).sum() for g in cohort_vals}
    total_treated = sum(cohort_sizes.values())

    # Collect all unique relative times
    all_rel_times = sorted(set(l for _, l in reg_pairs))

    att_dynamic: Dict[int, float] = {}
    for l in all_rel_times:
        coefs = []
        weights = []
        for col, (g, rl) in zip(reg_interaction_cols, reg_pairs):
            if rl == l and col in res.params.index:
                coefs.append(float(res.params[col]))
                weights.append(cohort_sizes.get(g, 0))
        if coefs:
            w_arr = np.array(weights, dtype=float)
            w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr
            att_dynamic[l] = float(np.dot(w_arr, coefs))

    # Build cohort_atts DataFrame
    rows = []
    for col, (g, l) in zip(reg_interaction_cols, reg_pairs):
        if col in res.params.index:
            rows.append({
                'cohort': g,
                'rel_time': l,
                'coef': float(res.params[col]),
                'se': float(res.bse[col]),
            })
    cohort_atts = pd.DataFrame(rows)

    return {
        'att_dynamic': att_dynamic,
        'cohort_atts': cohort_atts,
        'res': res,
    }


def callaway_santanna(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    cohort: str,
    controls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Callaway & Sant'Anna (2021) group-time ATTs.

    Uses not-yet-treated units as the comparison group (outcome regression approach).

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
    cohort : str
        Treatment cohort column — the first period of treatment (NaN = never treated).
    controls : list[str] or None
        Additional covariates for the outcome regression.

    Returns
    -------
    dict with keys:
        group_time (DataFrame: [cohort, time, att, se]),
        att_overall (float),
        att_dynamic (dict: rel_time -> ATT)
    """
    required = [y, unit, time, cohort]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df.copy()
    work[cohort] = pd.to_numeric(work[cohort], errors='coerce')
    work[time] = pd.to_numeric(work[time], errors='coerce')
    work[y] = pd.to_numeric(work[y], errors='coerce')

    cohort_vals = sorted(work[cohort].dropna().unique())
    time_vals = sorted(work[time].dropna().unique())

    if len(cohort_vals) == 0:
        raise ValueError("No treated units found.")

    records = []

    for g in cohort_vals:
        for t in time_vals:
            if t < g:
                continue  # pre-treatment for this cohort

            # Treated group: cohort == g
            treated_mask = work[cohort] == g
            # Control group: never treated OR not yet treated at time t
            control_mask = work[cohort].isna() | (work[cohort] > t)

            # Get base period (t-1 or first pre-period)
            if t > time_vals[0]:
                base_t = time_vals[time_vals.index(t) - 1]
            else:
                base_t = t

            # Treated outcomes at t and base
            t_now = work[treated_mask & (work[time] == t)][[y] + (controls or [])]
            c_now = work[control_mask & (work[time] == t)][[y] + (controls or [])]

            if len(t_now) == 0 or len(c_now) == 0:
                continue

            # Simple outcome regression approach:
            # E[Y(0)|treated] estimated via OLS on control group
            if controls and len(controls) > 0:
                Xc = sm.add_constant(c_now[controls], has_constant='add').astype(float)
                yc = c_now[y].astype(float)
                if len(Xc) > len(controls) + 1 and Xc.shape[1] > 1:
                    try:
                        or_res = sm.OLS(yc, Xc).fit()
                        Xt = sm.add_constant(t_now[controls], has_constant='add').astype(float)
                        y0_hat = or_res.predict(Xt)
                        att = float(t_now[y].mean() - y0_hat.mean())
                    except Exception:
                        att = float(t_now[y].mean() - c_now[y].mean())
                else:
                    att = float(t_now[y].mean() - c_now[y].mean())
            else:
                att = float(t_now[y].mean() - c_now[y].mean())

            # Bootstrap SE (simple)
            n_t = len(t_now)
            n_c = len(c_now)
            var_t = float(t_now[y].var()) / max(n_t, 1)
            var_c = float(c_now[y].var()) / max(n_c, 1)
            se = float(np.sqrt(var_t + var_c))

            records.append({
                'cohort': g,
                'time': t,
                'att': att,
                'se': se,
                'rel_time': t - g,
            })

    group_time = pd.DataFrame(records)

    if group_time.empty:
        return {
            'group_time': group_time,
            'att_overall': np.nan,
            'att_dynamic': {},
        }

    att_overall = float(group_time['att'].mean())

    # Aggregate dynamic ATT
    att_dynamic: Dict[int, float] = {}
    for rl, sub in group_time.groupby('rel_time'):
        att_dynamic[int(rl)] = float(sub['att'].mean())

    return {
        'group_time': group_time[['cohort', 'time', 'att', 'se']],
        'att_overall': att_overall,
        'att_dynamic': att_dynamic,
    }


def bacon_decomp(
    df: pd.DataFrame,
    y: str,
    treatment: str,
    unit: str,
    time: str,
) -> Dict[str, Any]:
    """
    Bacon (2021) decomposition of the TWFE estimator.

    Decomposes the TWFE 2x2 DID into weighted comparisons between:
    - 'Treated vs Untreated': always-treated or timing groups vs never-treated
    - 'Early vs Late': earlier-treated units as treated, later-treated as control (pre-period only)
    - 'Late vs Early': later-treated units as treated, earlier-treated as control (pre-period only)

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable.
    treatment : str
        Binary treatment indicator (0/1).
    unit : str
        Unit identifier.
    time : str
        Time variable.

    Returns
    -------
    dict with keys:
        twfe_beta (float): TWFE estimate
        table (DataFrame): columns [type, early_group, late_group, weight, beta_2x2]
    """
    required = [y, treatment, unit, time]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    work = df[[y, treatment, unit, time]].copy()
    work[treatment] = pd.to_numeric(work[treatment], errors='coerce')
    work[y] = pd.to_numeric(work[y], errors='coerce')
    work[time] = pd.to_numeric(work[time], errors='coerce')
    work = work.dropna()

    # Identify treatment timing for each unit
    def _first_treat(grp):
        treated_times = grp.loc[grp[treatment] == 1, time]
        if len(treated_times) == 0:
            return np.nan
        return treated_times.min()

    timing = work.groupby(unit).apply(_first_treat).reset_index()
    timing.columns = [unit, '_g']
    work = work.merge(timing, on=unit, how='left')

    never_units = work.loc[work['_g'].isna(), unit].unique()
    treat_units = work.loc[work['_g'].notna(), unit].unique()
    treat_timings = work.loc[work['_g'].notna()].groupby(unit)['_g'].first()

    unique_timings = sorted(treat_timings.unique())
    T_vals = sorted(work[time].unique())
    T = len(T_vals)
    N = work[unit].nunique()

    rows = []

    # --- Treated vs Never-treated 2x2 ---
    for g in unique_timings:
        g_units = treat_timings[treat_timings == g].index.tolist()
        # Pre: time < g, Post: time >= g
        pre_mask = work[time] < g
        post_mask = work[time] >= g

        never_pre = work[work[unit].isin(never_units) & pre_mask]
        never_post = work[work[unit].isin(never_units) & post_mask]
        treat_pre = work[work[unit].isin(g_units) & pre_mask]
        treat_post = work[work[unit].isin(g_units) & post_mask]

        if len(treat_pre) == 0 or len(treat_post) == 0 or len(never_pre) == 0 or len(never_post) == 0:
            continue

        beta = ((treat_post[y].mean() - treat_pre[y].mean()) -
                (never_post[y].mean() - never_pre[y].mean()))

        n_g = len(g_units)
        n_never = len(never_units)
        D_bar_g = post_mask.mean()
        s_g = n_g / N
        s_never = n_never / N
        weight = 2 * s_g * s_never * D_bar_g * (1 - D_bar_g)

        rows.append({
            'type': 'Treated vs Untreated',
            'early_group': g,
            'late_group': np.nan,
            'weight': weight,
            'beta_2x2': beta,
        })

    # --- Early vs Late / Late vs Early ---
    for i, g_early in enumerate(unique_timings):
        for g_late in unique_timings[i+1:]:
            early_units = treat_timings[treat_timings == g_early].index.tolist()
            late_units = treat_timings[treat_timings == g_late].index.tolist()

            # Early vs Late: use period before g_late as "clean" comparison
            # Early group treated from g_early, Late group not yet treated up to g_late
            pre_both = work[time] < g_early
            between = (work[time] >= g_early) & (work[time] < g_late)

            e_pre = work[work[unit].isin(early_units) & pre_both]
            e_mid = work[work[unit].isin(early_units) & between]
            l_pre = work[work[unit].isin(late_units) & pre_both]
            l_mid = work[work[unit].isin(late_units) & between]

            if len(e_pre) > 0 and len(e_mid) > 0 and len(l_pre) > 0 and len(l_mid) > 0:
                beta_el = ((e_mid[y].mean() - e_pre[y].mean()) -
                           (l_mid[y].mean() - l_pre[y].mean()))
                n_e = len(early_units)
                n_l = len(late_units)
                weight_el = (n_e * n_l) / (N ** 2)
                rows.append({
                    'type': 'Early vs Late',
                    'early_group': g_early,
                    'late_group': g_late,
                    'weight': weight_el,
                    'beta_2x2': beta_el,
                })

            # Late vs Early: after g_late, compare late (newly treated) vs early (already treated)
            post_late = work[time] >= g_late
            mid_period = (work[time] >= g_early) & (work[time] < g_late)

            l_mid2 = work[work[unit].isin(late_units) & mid_period]
            l_post = work[work[unit].isin(late_units) & post_late]
            e_mid2 = work[work[unit].isin(early_units) & mid_period]
            e_post = work[work[unit].isin(early_units) & post_late]

            if len(l_mid2) > 0 and len(l_post) > 0 and len(e_mid2) > 0 and len(e_post) > 0:
                beta_le = ((l_post[y].mean() - l_mid2[y].mean()) -
                           (e_post[y].mean() - e_mid2[y].mean()))
                n_e = len(early_units)
                n_l = len(late_units)
                weight_le = (n_e * n_l) / (N ** 2)
                rows.append({
                    'type': 'Late vs Early',
                    'early_group': g_early,
                    'late_group': g_late,
                    'weight': weight_le,
                    'beta_2x2': beta_le,
                })

    table = pd.DataFrame(rows)

    # Compute TWFE estimate as weighted average
    if not table.empty and table['weight'].sum() > 0:
        twfe_beta = float((table['weight'] * table['beta_2x2']).sum() / table['weight'].sum())
    else:
        twfe_beta = np.nan

    return {
        'twfe_beta': twfe_beta,
        'table': table,
    }
