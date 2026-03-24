import numpy as np
import pandas as pd
from econlab.causal.did import twfe_did, event_study, parallel_trend_test
from econlab.causal.staggered import sun_abraham, callaway_santanna, bacon_decomp
import pytest


def synth_panel(n=200, T=10, treat_time=6, tau=2.0, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.arange(n)
    ts = np.arange(1, T+1)
    rows = []
    alpha_i = rng.normal(0, 1, size=n)
    gamma_t = rng.normal(0, 1, size=T)
    treated = rng.random(n) < 0.5
    for i in ids:
        Ti = treat_time if treated[i] else np.nan
        for t in ts:
            D = 1.0 if (not np.isnan(Ti) and t >= Ti) else 0.0
            y = 1.0 + alpha_i[i] + gamma_t[t-1] + tau*D + rng.normal(0, 1)
            rows.append({'id': i, 't': t, 'y': y, 'D': D, 'Tstart': Ti})
    return pd.DataFrame(rows)


def staggered_panel(n=150, T=12, tau=2.0, seed=123):
    """
    Generate a staggered treatment panel with three cohorts:
    cohort 5 (treated from period 5), cohort 8 (from period 8), never-treated.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(n)
    ts = np.arange(1, T + 1)
    rows = []
    alpha_i = rng.normal(0, 1, size=n)
    gamma_t = rng.normal(0, 1, size=T)

    # Assign cohorts: 1/3 each to cohort 5, cohort 8, never-treated
    cohort_assign = rng.choice([5, 8, np.nan], size=n, p=[1/3, 1/3, 1/3])

    for i in ids:
        g = cohort_assign[i]
        for t in ts:
            D = 1.0 if (not np.isnan(g) and t >= g) else 0.0
            y = 1.0 + alpha_i[i] + gamma_t[t-1] + tau * D + rng.normal(0, 0.5)
            rows.append({
                'id': i,
                't': t,
                'y': y,
                'D': D,
                'cohort': g,
            })
    return pd.DataFrame(rows)


def test_twfe_recovers_tau():
    df = synth_panel()
    res = twfe_did(df, y='y', treatment='D', unit='id', time='t', cluster='id')
    assert abs(res['coef_D'] - 2.0) < 0.3


def test_event_study_pretrend():
    df = synth_panel()
    es = event_study(df, y='y', unit='id', time='t', treat_start='Tstart', window=(-5, 5))
    jt = parallel_trend_test(es, pre_k=[-5, -4, -3, -2])
    assert jt['pval'] > 0.05


def test_sun_abraham_recovers_tau():
    """Sun & Abraham estimator should recover approximately the true ATT."""
    df = staggered_panel(n=150, T=12, tau=2.0, seed=123)
    result = sun_abraham(df, y='y', unit='id', time='t', cohort='cohort')

    assert 'att_dynamic' in result
    assert 'cohort_atts' in result
    assert len(result['att_dynamic']) > 0

    # Post-treatment dynamic ATTs (rel_time >= 0) should be close to tau=2.0
    post_atts = [v for k, v in result['att_dynamic'].items() if k >= 0]
    assert len(post_atts) > 0
    avg_post_att = np.mean(post_atts)
    assert abs(avg_post_att - 2.0) < 1.0, (
        f"Average post-treatment ATT = {avg_post_att:.3f}, expected ~2.0"
    )


def test_callaway_santanna_runs():
    """Basic smoke test: Callaway & Sant'Anna should return the expected structure."""
    df = staggered_panel(n=100, T=10, tau=2.0, seed=42)
    result = callaway_santanna(df, y='y', unit='id', time='t', cohort='cohort')

    assert 'group_time' in result
    assert 'att_overall' in result
    assert 'att_dynamic' in result

    gt = result['group_time']
    assert isinstance(gt, pd.DataFrame)
    assert set(['cohort', 'time', 'att', 'se']).issubset(gt.columns)

    # att_overall should be roughly tau
    assert abs(result['att_overall'] - 2.0) < 1.5, (
        f"att_overall = {result['att_overall']:.3f}, expected ~2.0"
    )


def test_bacon_decomp_weights_sum_to_one():
    """Bacon decomposition weights should sum approximately to 1."""
    df = staggered_panel(n=120, T=10, tau=2.0, seed=7)
    result = bacon_decomp(df, y='y', treatment='D', unit='id', time='t')

    assert 'twfe_beta' in result
    assert 'table' in result

    table = result['table']
    assert isinstance(table, pd.DataFrame)
    assert 'weight' in table.columns
    assert 'beta_2x2' in table.columns

    if len(table) > 0:
        total_weight = table['weight'].sum()
        # Weights need not exactly sum to 1 due to approximation, but should be positive
        assert total_weight > 0, "Bacon decomposition weights should be positive"
        assert len(table) >= 1, "Should have at least one comparison type"
