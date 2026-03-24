import json
import sys
import typer
import pandas as pd
import numpy as np

from econlab.causal.did import twfe_did
from econlab.core.ols import ols
from econlab.core.iv import iv_2sls
from econlab.causal.rdd import rdd_sharp
from econlab.panel.re import re_panel, fe_panel

app = typer.Typer(help="EconLab CLI — Econometrics Research Toolbox")


def _load_csv(path: str) -> pd.DataFrame:
    """Load CSV file and return DataFrame."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        typer.echo(f"Error: File not found: {path}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error reading CSV: {e}", err=True)
        raise typer.Exit(code=1)


def _format_result(d: dict) -> str:
    """Format a result dict for stdout output."""
    out = {}
    for k, v in d.items():
        if k == 'res':
            continue  # skip result objects
        if isinstance(v, pd.Series):
            out[k] = {str(idx): (float(val) if np.isfinite(val) else None)
                      for idx, val in v.items()}
        elif isinstance(v, (int, float)):
            out[k] = v if np.isfinite(v) else None
        elif isinstance(v, dict):
            out[k] = {str(kk): (float(vv) if isinstance(vv, float) and np.isfinite(vv) else vv)
                      for kk, vv in v.items()}
        else:
            try:
                out[k] = float(v) if np.isfinite(float(v)) else None
            except Exception:
                out[k] = str(v)
    return json.dumps(out, indent=2, default=str)


@app.command(name='ols')
def ols_cmd(
    csv: str = typer.Option(..., help="Path to input CSV file"),
    y: str = typer.Option(..., help="Dependent variable column name"),
    X: str = typer.Option(..., help="Comma-separated regressor column names"),
    robust: bool = typer.Option(True, help="Use HC1 robust standard errors"),
    cluster: str = typer.Option(None, help="Cluster column name (optional)"),
    add_const: bool = typer.Option(True, help="Add intercept"),
):
    """OLS regression with robust/clustered standard errors."""
    df = _load_csv(csv)
    x_cols = [c.strip() for c in X.split(',')]

    cluster_series = df[cluster] if cluster else None
    result = ols(df, y=y, X=x_cols, add_const=add_const, robust=robust, cluster=cluster_series)

    typer.echo(f"OLS Results — Dependent variable: {y}")
    typer.echo(f"N = {result['nobs']}, R² = {result['rsq']:.4f}")
    typer.echo("\nCoefficients:")
    for var in result['params'].index:
        coef = result['params'][var]
        se = result['bse'][var]
        pval = result['pvalues'][var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        typer.echo(f"  {var:20s}  {coef:10.4f}  ({se:.4f})  {stars}")


@app.command(name='twfe-did')
def twfe_did_cmd(
    csv: str = typer.Option(..., help="CSV path"),
    y: str = typer.Option(..., help="Dependent variable column"),
    treatment: str = typer.Option(..., help="Treatment indicator column (0/1)"),
    unit: str = typer.Option(..., help="Unit id column"),
    time: str = typer.Option(..., help="Time column"),
    cluster: str = typer.Option(None, help="Cluster column (optional)"),
):
    """Two-way Fixed Effects (TWFE) Difference-in-Differences."""
    df = _load_csv(csv)
    res = twfe_did(df, y=y, treatment=treatment, unit=unit, time=time, cluster=cluster)
    typer.echo(f"TWFE DiD Results")
    typer.echo(f"Treatment effect (coef_D): {res['coef_D']:.4f}")
    typer.echo(f"Standard error:            {res['se_D']:.4f}")


@app.command(name='iv')
def iv_cmd(
    csv: str = typer.Option(..., help="Path to input CSV file"),
    y: str = typer.Option(..., help="Dependent variable column name"),
    endog: str = typer.Option(..., help="Comma-separated endogenous regressor column names"),
    exog: str = typer.Option('', help="Comma-separated exogenous control column names"),
    instruments: str = typer.Option(..., help="Comma-separated excluded instrument column names"),
    robust: bool = typer.Option(True, help="Use heteroskedasticity-robust SEs"),
    cluster: str = typer.Option(None, help="Cluster column name (optional)"),
):
    """IV/2SLS estimation with first-stage diagnostics."""
    df = _load_csv(csv)
    endog_cols = [c.strip() for c in endog.split(',') if c.strip()]
    exog_cols = [c.strip() for c in exog.split(',') if c.strip()]
    instr_cols = [c.strip() for c in instruments.split(',') if c.strip()]

    result = iv_2sls(
        df, y=y, X_endog=endog_cols, X_exog=exog_cols,
        instruments=instr_cols, robust=robust, cluster=cluster
    )

    typer.echo(f"IV/2SLS Results — Dependent variable: {y}")
    typer.echo(f"N = {result['nobs']}")
    typer.echo("\nCoefficients:")
    for var in result['params'].index:
        coef = float(result['params'][var])
        se = float(result['bse'][var])
        pval = float(result['pvalues'][var])
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        typer.echo(f"  {var:20s}  {coef:10.4f}  ({se:.4f})  {stars}")

    typer.echo("\nFirst-Stage F-statistics:")
    for var, fs in result['first_stage'].items():
        typer.echo(f"  {var:20s}  F = {fs['F']:.2f}  (p = {fs['p']:.4f})")

    if result['sargan']:
        s = result['sargan']
        typer.echo(f"\nSargan overidentification: stat = {s['stat']:.4f}, p = {s['pval']:.4f}")

    wh = result['wu_hausman']
    typer.echo(f"Wu-Hausman endogeneity:    stat = {wh['stat']:.4f}, p = {wh['pval']:.4f}")


@app.command(name='rdd')
def rdd_cmd(
    csv: str = typer.Option(..., help="Path to input CSV file"),
    y: str = typer.Option(..., help="Outcome variable column"),
    running: str = typer.Option(..., help="Running variable column"),
    cutoff: float = typer.Option(0.0, help="Cutoff value (default 0.0)"),
    bandwidth: float = typer.Option(None, help="Bandwidth (default: IK selector)"),
    kernel: str = typer.Option('triangular', help="Kernel: triangular, uniform, epanechnikov"),
    poly_order: int = typer.Option(1, help="Polynomial order for local regression"),
):
    """Sharp Regression Discontinuity Design (RDD)."""
    df = _load_csv(csv)
    bw = bandwidth if bandwidth and bandwidth > 0 else None

    result = rdd_sharp(
        df, y=y, running=running, cutoff=cutoff,
        bandwidth=bw, kernel=kernel, poly_order=poly_order
    )

    typer.echo(f"Sharp RDD Results — Outcome: {y}, Running: {running}, Cutoff: {cutoff}")
    typer.echo(f"Bandwidth used:    {result['bandwidth']:.4f}")
    typer.echo(f"Treatment effect:  {result['tau']:.4f}  (SE: {result['se']:.4f})")
    typer.echo(f"p-value:           {result['pval']:.4f}")
    typer.echo(f"95% CI:            [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    typer.echo(f"N (left):          {result['nobs_left']}")
    typer.echo(f"N (right):         {result['nobs_right']}")


@app.command(name='fe')
def fe_cmd(
    csv: str = typer.Option(..., help="Path to input CSV file"),
    y: str = typer.Option(..., help="Outcome variable column"),
    X: str = typer.Option(..., help="Comma-separated regressor column names"),
    unit: str = typer.Option(..., help="Unit identifier column"),
    time: str = typer.Option(..., help="Time period column"),
    cluster: str = typer.Option(None, help="Cluster column name (optional)"),
):
    """Fixed Effects (within) panel estimator."""
    df = _load_csv(csv)
    x_cols = [c.strip() for c in X.split(',')]

    result = fe_panel(df, y=y, X=x_cols, unit=unit, time=time, cluster=cluster)

    typer.echo(f"Fixed Effects Panel Results — Outcome: {y}")
    typer.echo(f"N = {result['nobs']}, R² (within) = {result['rsq_within']:.4f}")
    typer.echo("\nCoefficients:")
    for var in result['params'].index:
        coef = float(result['params'][var])
        se = float(result['bse'][var])
        pval = float(result['pvalues'][var])
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        typer.echo(f"  {var:20s}  {coef:10.4f}  ({se:.4f})  {stars}")


@app.command(name='re')
def re_cmd(
    csv: str = typer.Option(..., help="Path to input CSV file"),
    y: str = typer.Option(..., help="Outcome variable column"),
    X: str = typer.Option(..., help="Comma-separated regressor column names"),
    unit: str = typer.Option(..., help="Unit identifier column"),
    time: str = typer.Option(..., help="Time period column"),
    cluster: str = typer.Option(None, help="Cluster column name (optional)"),
):
    """Random Effects GLS panel estimator."""
    df = _load_csv(csv)
    x_cols = [c.strip() for c in X.split(',')]

    result = re_panel(df, y=y, X=x_cols, unit=unit, time=time, cluster=cluster)

    typer.echo(f"Random Effects Panel Results — Outcome: {y}")
    typer.echo(f"N = {result['nobs']}")
    typer.echo("\nCoefficients:")
    for var in result['params'].index:
        coef = float(result['params'][var])
        se = float(result['bse'][var])
        pval = float(result['pvalues'][var])
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        typer.echo(f"  {var:20s}  {coef:10.4f}  ({se:.4f})  {stars}")


@app.command(name='var')
def var_cmd(
    csv: str = typer.Option(..., help="Path to input CSV file"),
    variables: str = typer.Option(..., help="Comma-separated variable column names"),
    maxlags: int = typer.Option(None, help="Maximum lag order (default: automatic)"),
    ic: str = typer.Option('aic', help="Information criterion: aic, bic, hqic, fpe"),
    granger_caused: str = typer.Option(None, help="Test Granger causality: caused variable"),
    granger_causing: str = typer.Option(None, help="Test Granger causality: causing variable"),
):
    """Vector Autoregression (VAR) estimation."""
    from econlab.timeseries.var import estimate_var, granger_causality

    df = _load_csv(csv)
    var_cols = [c.strip() for c in variables.split(',')]

    result = estimate_var(df, variables=var_cols, maxlags=maxlags, ic=ic)

    typer.echo(f"VAR Results — Variables: {', '.join(var_cols)}")
    typer.echo(f"Lags selected: {result['lags_selected']}, N = {result['nobs']}")
    typer.echo(f"AIC = {result['aic']:.4f}, BIC = {result['bic']:.4f}, HQIC = {result['hqic']:.4f}")

    if granger_caused and granger_causing:
        typer.echo(f"\nGranger Causality: {granger_causing} → {granger_caused}")
        gc = granger_causality(df, caused=granger_caused, causing=granger_causing)
        for r in gc['results']:
            stars = '**' if r['pval'] < 0.05 else ''
            typer.echo(f"  Lag {r['lag']}: F = {r['F']:.3f}, p = {r['pval']:.4f} {stars}")
        typer.echo(f"  Reject H0 (no Granger causality): {gc['reject_h0']}")


if __name__ == "__main__":
    app()
