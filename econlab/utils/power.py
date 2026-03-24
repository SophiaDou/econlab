from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


def power_analysis(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
    """
    Compute statistical power for a given effect size (Cohen's d) and sample size.

    Uses the non-central t-distribution approximation via normal distribution.

    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d).
    n : int
        Total sample size.
    alpha : float
        Significance level. Default 0.05.
    two_tailed : bool
        Whether the test is two-tailed. Default True.

    Returns
    -------
    float
        Statistical power in [0, 1].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    # Non-centrality parameter
    ncp = abs(effect_size) * np.sqrt(n / 2)

    if two_tailed:
        power = stats.norm.sf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    else:
        power = stats.norm.sf(z_alpha - ncp)

    return float(np.clip(power, 0.0, 1.0))


def mde(
    n: int,
    power: float = 0.8,
    alpha: float = 0.05,
    sigma: float = 1.0,
    two_tailed: bool = True,
    icc: float = 0.0,
    cluster_size: int = 1,
) -> float:
    """
    Minimum Detectable Effect (MDE).

    Computes the smallest effect size detectable with the given sample size,
    power, and significance level. Supports clustered designs via ICC.

    Parameters
    ----------
    n : int
        Total sample size.
    power : float
        Desired statistical power. Default 0.8.
    alpha : float
        Significance level. Default 0.05.
    sigma : float
        Standard deviation of the outcome. Default 1.0.
    two_tailed : bool
        Whether the test is two-tailed. Default True.
    icc : float
        Intraclass correlation coefficient for clustered designs. Default 0.0.
    cluster_size : int
        Average cluster size. Default 1 (no clustering).

    Returns
    -------
    float
        MDE in outcome units (same scale as sigma).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if not (0 < power < 1):
        raise ValueError(f"power must be in (0, 1), got {power}.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if not (0 <= icc <= 1):
        raise ValueError(f"icc must be in [0, 1], got {icc}.")
    if cluster_size < 1:
        raise ValueError(f"cluster_size must be >= 1, got {cluster_size}.")

    # Design effect (DEFF)
    deff = 1.0 + (cluster_size - 1) * icc
    n_eff = n / deff

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # MDE formula: delta = (z_alpha + z_beta) * sigma * sqrt(2 / n_eff)
    mde_val = (z_alpha + z_beta) * sigma * np.sqrt(2.0 / max(n_eff, 1))
    return float(mde_val)


def sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    sigma: float = 1.0,
    two_tailed: bool = True,
    icc: float = 0.0,
    cluster_size: int = 1,
) -> int:
    """
    Required sample size for given effect size, power, and significance level.

    Supports clustered designs via ICC (design effect adjustment).

    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d) or unstandardized effect (in sigma units).
    power : float
        Desired statistical power. Default 0.8.
    alpha : float
        Significance level. Default 0.05.
    sigma : float
        Standard deviation of the outcome. Default 1.0.
        If effect_size is Cohen's d, set sigma=1.0.
    two_tailed : bool
        Whether the test is two-tailed. Default True.
    icc : float
        Intraclass correlation coefficient for clustered designs. Default 0.0.
    cluster_size : int
        Average cluster size. Default 1 (no clustering).

    Returns
    -------
    int
        Required total sample size (rounded up to nearest integer).
    """
    if effect_size == 0:
        raise ValueError("effect_size cannot be zero (would require infinite sample).")
    if not (0 < power < 1):
        raise ValueError(f"power must be in (0, 1), got {power}.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if not (0 <= icc <= 1):
        raise ValueError(f"icc must be in [0, 1], got {icc}.")
    if cluster_size < 1:
        raise ValueError(f"cluster_size must be >= 1, got {cluster_size}.")

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Cohen's d = effect_size / sigma (standardized effect)
    d = abs(effect_size) / sigma

    # Base sample size (per group, for two-sample t-test)
    n_per_group = 2.0 * ((z_alpha + z_beta) / d) ** 2
    n_total = 2.0 * n_per_group

    # Design effect adjustment
    deff = 1.0 + (cluster_size - 1) * icc
    n_total_adjusted = n_total * deff

    return int(math.ceil(n_total_adjusted))


def power_curve(
    n_range: Any,
    effect_size: float,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> pd.DataFrame:
    """
    Compute statistical power for a range of sample sizes.

    Parameters
    ----------
    n_range : array-like
        Range of sample sizes to evaluate.
    effect_size : float
        Standardized effect size (Cohen's d).
    alpha : float
        Significance level. Default 0.05.
    two_tailed : bool
        Whether the test is two-tailed. Default True.

    Returns
    -------
    pd.DataFrame
        Columns: [n, power]
    """
    n_arr = np.asarray(n_range, dtype=int)
    if len(n_arr) == 0:
        raise ValueError("n_range must be non-empty.")

    rows = []
    for n in n_arr:
        try:
            pwr = power_analysis(effect_size, int(n), alpha=alpha, two_tailed=two_tailed)
        except Exception:
            pwr = np.nan
        rows.append({'n': int(n), 'power': pwr})

    return pd.DataFrame(rows)
