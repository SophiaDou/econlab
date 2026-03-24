# econlab — Econometrics Research Toolbox
# Top-level package init: exposes all public APIs.

from econlab import core, causal, panel, timeseries, diagnostics, robust, utils

# Core estimation
from econlab.core import ols, iv_2sls, quantile_reg

# Causal inference
from econlab.causal import (
    twfe_did,
    event_study,
    parallel_trend_test,
    rdd_sharp,
    rdd_fuzzy,
    mccrary_density_test,
    synthetic_control,
    propensity_score_match,
    ipw_estimate,
    doubly_robust,
    covariate_balance,
    sun_abraham,
    callaway_santanna,
    bacon_decomp,
)

# Panel data
from econlab.panel import fe_panel, re_panel, hausman_test, diff_gmm, sys_gmm

# Time series
from econlab.timeseries import (
    adf_test,
    kpss_test,
    pp_test,
    engle_granger,
    johansen_test,
    estimate_var,
    granger_causality,
)

# Diagnostics
from econlab.diagnostics import (
    vif,
    breusch_pagan,
    white_test,
    breusch_godfrey,
    ramsey_reset,
    durbin_watson,
    specification_summary,
)

# Robust inference
from econlab.robust import wild_cluster_bootstrap

# Utilities
from econlab.utils import (
    to_latex_table,
    to_word_table,
    regression_table,
    coef_plot,
    power_analysis,
    mde,
    sample_size,
)

__version__ = '0.2.0'

__all__ = [
    # submodules
    'core', 'causal', 'panel', 'timeseries', 'diagnostics', 'robust', 'utils',
    # core
    'ols', 'iv_2sls', 'quantile_reg',
    # causal
    'twfe_did', 'event_study', 'parallel_trend_test',
    'rdd_sharp', 'rdd_fuzzy', 'mccrary_density_test',
    'synthetic_control',
    'propensity_score_match', 'ipw_estimate', 'doubly_robust', 'covariate_balance',
    'sun_abraham', 'callaway_santanna', 'bacon_decomp',
    # panel
    'fe_panel', 're_panel', 'hausman_test', 'diff_gmm', 'sys_gmm',
    # timeseries
    'adf_test', 'kpss_test', 'pp_test', 'engle_granger', 'johansen_test',
    'estimate_var', 'granger_causality',
    # diagnostics
    'vif', 'breusch_pagan', 'white_test', 'breusch_godfrey',
    'ramsey_reset', 'durbin_watson', 'specification_summary',
    # robust
    'wild_cluster_bootstrap',
    # utils
    'to_latex_table', 'to_word_table', 'regression_table', 'coef_plot',
    'power_analysis', 'mde', 'sample_size',
]
