import typer
import pandas as pd
from econlab.causal.did import twfe_did

app = typer.Typer(help="EconLab CLI")

@app.command()
def twfe_did_cmd(csv: str = typer.Option(..., help="CSV path"),
                 y: str = typer.Option(..., help="Dependent variable column"),
                 treatment: str = typer.Option(..., help="Treatment indicator column (0/1)"),
                 unit: str = typer.Option(..., help="Unit id column"),
                 time: str = typer.Option(..., help="Time column"),
                 cluster: str = typer.Option(None, help="Cluster column (optional)")):
    df = pd.read_csv(csv)
    res = twfe_did(df, y=y, treatment=treatment, unit=unit, time=time, cluster=cluster)
    typer.echo({'coef_D': res['coef_D'], 'se_D': res['se_D']})

if __name__ == "__main__":
    app()