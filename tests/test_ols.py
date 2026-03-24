import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from econlab.core.ols import ols
# -------------------------
# Fixtures: synthetic data
# -------------------------
@pytest.fixture(scope="module")
def sim_data():
    """
    Generate a reproducible dataset with cluster-level correlation so that
    robust/clustered SEs differ from non-robust SEs.

    Model:
        y = beta0 + beta1*x1 + beta2*x2 + u_g + e
        where u_g is a cluster shock and e ~ N(0,1)
    """
    rng = np.random.default_rng(123)
    n = 2000
    G = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    # Assign clusters and cluster-level shocks
    groups = rng.integers(0, G, size=n)
    u = rng.normal(0, 0.6, size=G)  # cluster shocks
    e = rng.normal(0, 1.0, size=n)  # idiosyncratic shocks

    beta0 = 1.5
    beta1 = 2.0
    beta2 = -3.0
    y = beta0 + beta1 * x1 + beta2 * x2 + u[groups] + e

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    cluster = pd.Series(groups, name="cluster")

    true_b = {"const": beta0, "x1": beta1, "x2": beta2}
    return df, cluster, true_b

# ---------------------------------------
# Structure and basic behavior tests
# ---------------------------------------
def test_return_structure_and_types(sim_data):
    df, cluster, _ = sim_data
    result = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=True)

    # Required keys
    for key in ["params", "bse", "tvalues", "pvalues", "nobs", "rsq", "res"]:
        assert key in result, f"Missing key: {key}"

    # Types
    assert isinstance(result["params"], pd.Series)
    assert isinstance(result["bse"], pd.Series)
    assert isinstance(result["tvalues"], pd.Series)
    assert isinstance(result["pvalues"], pd.Series)
    assert isinstance(result["nobs"], int)
    assert isinstance(result["rsq"], float)
    assert isinstance(result["res"], RegressionResultsWrapper)

    # Sanity checks
    assert result["nobs"] == len(df)
    assert 0.0 <= result["rsq"] <= 1.0
    assert result["res"].cov_type == "HC1"

def test_add_const_behavior(sim_data):
    df, _, true_b = sim_data

    # With constant
    r_with = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=False)
    assert "const" in r_with["params"].index
    # Coefficients should be close to the truth
    est = r_with["params"][["const", "x1", "x2"]]
    truth = pd.Series([true_b["const"], true_b["x1"], true_b["x2"]],
                      index=["const", "x1", "x2"])
    # With n=2000, this should be tight; loosen if needed
    assert np.allclose(est.values, truth.values, atol=0.15), f"Estimated: {est} vs True: {truth}"

    # Without constant
    r_wo = ols(df, y="y", X=["x1", "x2"], add_const=False, robust=False)
    assert "const" not in r_wo["params"].index

# ---------------------------------------
# Covariance type selection and effects
# ---------------------------------------
def test_covariance_modes_and_effects(sim_data):
    df, cluster, _ = sim_data

    # Non-robust (classical OLS)
    r_non = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=False)
    assert r_non["res"].cov_type == "nonrobust"

    # Robust HC1
    r_rob = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=True)
    assert r_rob["res"].cov_type == "HC1"

    # Cluster-robust
    r_clu = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=False, cluster=cluster)
    assert r_clu["res"].cov_type == "cluster"

    # With cluster-level correlation in errors, SEs should generally differ
    # between non-robust and robust/clustered.
    bse_non = r_non["bse"].values
    bse_rob = r_rob["bse"].values
    bse_clu = r_clu["bse"].values

    assert not np.allclose(bse_non, bse_rob), "Non-robust and robust SEs unexpectedly identical"
    assert not np.allclose(bse_non, bse_clu), "Non-robust and clustered SEs unexpectedly identical"


# ---------------------------------------
# Consistency with direct statsmodels API
# ---------------------------------------
def test_matches_statsmodels_parameters(sim_data):
    df, cluster, _ = sim_data
    X = sm.add_constant(df[["x1", "x2"]])

    # HC1 robust
    sm_res_hc1 = sm.OLS(df["y"], X).fit(cov_type="HC1")
    r_ours_hc1 = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=True)
    assert np.allclose(r_ours_hc1["params"].values, sm_res_hc1.params.values)
    # bse might differ in tiny float tolerances across versions, allow tight tolerance
    assert np.allclose(r_ours_hc1["bse"].values, sm_res_hc1.bse.values, rtol=1e-6, atol=1e-8)

    # Clustered
    sm_res_clu = sm.OLS(df["y"], X).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    r_ours_clu = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=False, cluster=cluster)
    assert np.allclose(r_ours_clu["params"].values, sm_res_clu.params.values)
    # bse equivalence under the same API call; tiny numeric diffs possible
    assert np.allclose(r_ours_clu["bse"].values, sm_res_clu.bse.values, rtol=1e-6, atol=1e-8)


# ---------------------------------------
# Error & edge-case handling
# ---------------------------------------
def test_cluster_length_mismatch_raises(sim_data):
    df, cluster, _ = sim_data
    bad_cluster = cluster.iloc[:-3]  # wrong length
    with pytest.raises(Exception):
        _ = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=False, cluster=bad_cluster)


def test_missing_values_raise(sim_data):
    df, cluster, _ = sim_data
    df_nan = df.copy()
    df_nan.loc[0, "y"] = np.nan
    with pytest.raises(Exception):
        _ = ols(df_nan, y="y", X=["x1", "x2"], add_const=True, robust=True)

    # Also try missing in X
    df_nan2 = df.copy()
    df_nan2.loc[1, "x1"] = np.nan
    with pytest.raises(Exception):
        _ = ols(df_nan2, y="y", X=["x1", "x2"], add_const=True, robust=True)


def test_has_inference_outputs(sim_data):
    df, _, _ = sim_data
    r = ols(df, y="y", X=["x1", "x2"], add_const=True, robust=True)
    # Ensure inference metrics are present and well-formed
    assert np.all(np.isfinite(r["tvalues"].values))
    assert np.all((r["pvalues"].values >= 0) & (r["pvalues"].values <= 1))
    # rsq in [0,1], already checked above, but specifically re-check here:
    assert 0.0 <= r["rsq"] <= 1.0