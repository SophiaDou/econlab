# causal submodule init
from econlab.causal.did import twfe_did, event_study, parallel_trend_test
from econlab.causal.rdd import rdd_sharp, rdd_fuzzy, mccrary_density_test
from econlab.causal.synthetic import synthetic_control
from econlab.causal.matching import (
    propensity_score_match,
    ipw_estimate,
    doubly_robust,
    covariate_balance,
)
from econlab.causal.staggered import sun_abraham, callaway_santanna, bacon_decomp

__all__ = [
    'twfe_did',
    'event_study',
    'parallel_trend_test',
    'rdd_sharp',
    'rdd_fuzzy',
    'mccrary_density_test',
    'synthetic_control',
    'propensity_score_match',
    'ipw_estimate',
    'doubly_robust',
    'covariate_balance',
    'sun_abraham',
    'callaway_santanna',
    'bacon_decomp',
]
