import numpy as np
import pandas as pd
from econlab.causal.did import twfe_did, event_study, parallel_trend_test


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


def test_twfe_recovers_tau():
    df = synth_panel()
    res = twfe_did(df, y='y', treatment='D', unit='id', time='t', cluster='id')
    assert abs(res['coef_D'] - 2.0) < 0.3


def test_event_study_pretrend():
    df = synth_panel()
    es = event_study(df, y='y', unit='id', time='t', treat_start='Tstart', window=(-5, 5))
    jt = parallel_trend_test(es, pre_k=[-5, -4, -3, -2])
    assert jt['pval'] > 0.05