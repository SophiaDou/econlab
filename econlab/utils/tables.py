from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def _stars(pval: float) -> str:
    """Return significance stars based on p-value."""
    if np.isnan(pval):
        return ''
    if pval < 0.01:
        return '***'
    if pval < 0.05:
        return '**'
    if pval < 0.1:
        return '*'
    return ''


def regression_table(
    results: List[Dict[str, Any]],
    model_names: Optional[List[str]] = None,
    var_labels: Optional[Dict[str, str]] = None,
    stars: bool = True,
    se_in_parentheses: bool = True,
    include_stats: Tuple[str, ...] = ('nobs', 'rsq', 'adj_rsq'),
    fmt: str = 'latex',
) -> str:
    """
    Stargazer-style regression table for multiple models.

    Parameters
    ----------
    results : list of dict
        Each dict is the output of ols(), iv_2sls(), etc.
        Must have keys: params, bse, pvalues, nobs.
        Optional keys: rsq, adj_rsq.
    model_names : list[str] or None
        Column headers for each model. Default: (1), (2), ...
    var_labels : dict or None
        Mapping from column names to display names.
    stars : bool
        Add significance stars (*** p<0.01, ** p<0.05, * p<0.1). Default True.
    se_in_parentheses : bool
        Show standard errors in parentheses below coefficients. Default True.
    include_stats : tuple
        Summary statistics to include: 'nobs', 'rsq', 'adj_rsq'.
    fmt : str
        Output format: 'latex', 'html', or 'text'. Default 'latex'.

    Returns
    -------
    str
        Formatted table string.
    """
    if not results:
        raise ValueError("results must be a non-empty list.")

    valid_fmt = {'latex', 'html', 'text'}
    if fmt not in valid_fmt:
        raise ValueError(f"fmt must be one of {valid_fmt}, got '{fmt}'.")

    n_models = len(results)
    if model_names is None:
        model_names = [f'({i+1})' for i in range(n_models)]
    if len(model_names) != n_models:
        raise ValueError(f"len(model_names)={len(model_names)} != len(results)={n_models}")

    var_labels = var_labels or {}

    # Collect all variable names across models
    all_vars = []
    seen = set()
    for res in results:
        params = res['params']
        for v in params.index:
            if v not in seen:
                all_vars.append(v)
                seen.add(v)

    def _fmt_num(x, decimals=3):
        if np.isnan(x):
            return ''
        return f'{x:.{decimals}f}'

    if fmt == 'latex':
        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(r'\begin{tabular}{l' + 'c' * n_models + '}')
        lines.append(r'\hline\hline')
        # Header
        header = ' & ' + ' & '.join(model_names) + r' \\'
        lines.append(header)
        lines.append(r'\hline')

        # Coefficients
        for var in all_vars:
            if var == 'const':
                display = 'Constant'
            else:
                display = var_labels.get(var, var)

            coef_row = display
            se_row = ''
            for res in results:
                params = res['params']
                bse = res.get('bse', pd.Series(dtype=float))
                pvalues = res.get('pvalues', pd.Series(dtype=float))

                if var in params.index:
                    coef = params[var]
                    se = bse[var] if var in bse.index else np.nan
                    pval = pvalues[var] if var in pvalues.index else np.nan
                    star = _stars(pval) if stars else ''
                    coef_row += f' & {_fmt_num(coef)}{star}'
                    if se_in_parentheses:
                        se_row += f' & ({_fmt_num(se)})'
                else:
                    coef_row += ' & '
                    if se_in_parentheses:
                        se_row += ' & '

            coef_row += r' \\'
            lines.append(coef_row)
            if se_in_parentheses:
                se_row += r' \\'
                lines.append(se_row)

        lines.append(r'\hline')

        # Summary statistics
        for stat in include_stats:
            if stat == 'nobs':
                row = 'N'
                for res in results:
                    n = res.get('nobs', np.nan)
                    row += f' & {int(n)}' if not np.isnan(n) else ' & '
                row += r' \\'
                lines.append(row)
            elif stat == 'rsq':
                row = r'$R^2$'
                for res in results:
                    r2 = res.get('rsq', np.nan)
                    row += f' & {_fmt_num(r2)}' if not np.isnan(r2) else ' & '
                row += r' \\'
                lines.append(row)
            elif stat == 'adj_rsq':
                row = r'Adj. $R^2$'
                for res in results:
                    r_obj = res.get('res')
                    adj_r2 = np.nan
                    if r_obj is not None and hasattr(r_obj, 'rsquared_adj'):
                        adj_r2 = float(r_obj.rsquared_adj)
                    row += f' & {_fmt_num(adj_r2)}' if not np.isnan(adj_r2) else ' & '
                row += r' \\'
                lines.append(row)

        lines.append(r'\hline\hline')
        if stars:
            lines.append(r'\multicolumn{' + str(n_models + 1) + r'}{l}{\footnotesize{*** p$<$0.01, ** p$<$0.05, * p$<$0.1}}\\')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')
        return '\n'.join(lines)

    elif fmt == 'html':
        lines = []
        lines.append('<table border="1" style="border-collapse:collapse;">')
        lines.append('<thead><tr><th>Variable</th>' +
                     ''.join(f'<th>{mn}</th>' for mn in model_names) + '</tr></thead>')
        lines.append('<tbody>')

        for var in all_vars:
            display = 'Constant' if var == 'const' else var_labels.get(var, var)
            row = f'<tr><td><strong>{display}</strong></td>'
            for res in results:
                params = res['params']
                bse = res.get('bse', pd.Series(dtype=float))
                pvalues = res.get('pvalues', pd.Series(dtype=float))
                if var in params.index:
                    coef = params[var]
                    pval = pvalues[var] if var in pvalues.index else np.nan
                    star = _stars(pval) if stars else ''
                    se = bse[var] if var in bse.index else np.nan
                    se_str = f'<br/>({_fmt_num(se)})' if se_in_parentheses else ''
                    row += f'<td>{_fmt_num(coef)}{star}{se_str}</td>'
                else:
                    row += '<td></td>'
            row += '</tr>'
            lines.append(row)

        lines.append('<tr><td colspan="' + str(n_models + 1) + '"><hr/></td></tr>')
        for stat in include_stats:
            if stat == 'nobs':
                row = '<tr><td>N</td>'
                for res in results:
                    n = res.get('nobs', np.nan)
                    row += f'<td>{int(n)}</td>' if not np.isnan(n) else '<td></td>'
                row += '</tr>'
                lines.append(row)
            elif stat == 'rsq':
                row = '<tr><td>R²</td>'
                for res in results:
                    r2 = res.get('rsq', np.nan)
                    row += f'<td>{_fmt_num(r2)}</td>' if not np.isnan(r2) else '<td></td>'
                row += '</tr>'
                lines.append(row)

        if stars:
            lines.append(f'<tr><td colspan="{n_models+1}"><small>*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1</small></td></tr>')
        lines.append('</tbody></table>')
        return '\n'.join(lines)

    else:  # text
        # Simple text table
        col_width = 14
        var_width = 25

        header = f"{'Variable':<{var_width}}" + ''.join(f'{mn:^{col_width}}' for mn in model_names)
        separator = '-' * len(header)
        lines = [separator, header, separator]

        for var in all_vars:
            display = 'Constant' if var == 'const' else var_labels.get(var, var)
            coef_line = f'{display:<{var_width}}'
            se_line = f'{"":<{var_width}}'
            for res in results:
                params = res['params']
                bse = res.get('bse', pd.Series(dtype=float))
                pvalues = res.get('pvalues', pd.Series(dtype=float))
                if var in params.index:
                    coef = params[var]
                    pval = pvalues[var] if var in pvalues.index else np.nan
                    star = _stars(pval) if stars else ''
                    se = bse[var] if var in bse.index else np.nan
                    coef_line += f'{_fmt_num(coef) + star:^{col_width}}'
                    se_line += f'{"(" + _fmt_num(se) + ")":^{col_width}}' if se_in_parentheses else ' ' * col_width
                else:
                    coef_line += ' ' * col_width
                    se_line += ' ' * col_width
            lines.append(coef_line)
            if se_in_parentheses:
                lines.append(se_line)

        lines.append(separator)
        for stat in include_stats:
            if stat == 'nobs':
                row = f'{"N":<{var_width}}'
                for res in results:
                    n = res.get('nobs', np.nan)
                    row += f'{int(n) if not np.isnan(n) else "":^{col_width}}'
                lines.append(row)
            elif stat == 'rsq':
                row = f'{"R-squared":<{var_width}}'
                for res in results:
                    r2 = res.get('rsq', np.nan)
                    row += f'{_fmt_num(r2) if not np.isnan(r2) else "":^{col_width}}'
                lines.append(row)

        lines.append(separator)
        if stars:
            lines.append('*** p<0.01, ** p<0.05, * p<0.1')

        return '\n'.join(lines)


def coef_plot(
    results_dict: Any,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Coefficient Plot',
    zero_line: bool = True,
) -> 'matplotlib.figure.Figure':
    """
    Plot coefficients with 95% confidence intervals.

    Parameters
    ----------
    results_dict : dict or list of dict
        Output of ols() or similar, or list of such dicts.
        Must contain 'params', 'bse' keys.
    var_names : list[str] or None
        Variables to plot. If None, plots all (excluding 'const').
    figsize : tuple
        Figure size. Default (8, 6).
    title : str
        Plot title. Default 'Coefficient Plot'.
    zero_line : bool
        Draw a vertical line at zero. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    if isinstance(results_dict, dict):
        results_list = [results_dict]
    else:
        results_list = results_dict

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for m_idx, res in enumerate(results_list):
        params = res['params']
        bse = res.get('bse', pd.Series(np.zeros(len(params)), index=params.index))

        plot_vars = var_names if var_names is not None else [v for v in params.index if v != 'const']
        plot_vars = [v for v in plot_vars if v in params.index]

        n_vars = len(plot_vars)
        y_positions = np.arange(n_vars) + m_idx * 0.25

        for j, var in enumerate(plot_vars):
            coef = float(params[var])
            se = float(bse[var]) if var in bse.index else 0.0
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se

            color = colors[m_idx % len(colors)]
            ax.errorbar(
                coef, y_positions[j],
                xerr=[[coef - ci_lo], [ci_hi - coef]],
                fmt='o', color=color,
                capsize=4, markersize=6,
                label=f'Model {m_idx+1}' if j == 0 and len(results_list) > 1 else None,
            )

    if zero_line:
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

    plot_vars_display = var_names if var_names is not None else [
        v for v in results_list[0]['params'].index if v != 'const'
    ]
    ax.set_yticks(np.arange(len(plot_vars_display)))
    ax.set_yticklabels(plot_vars_display)
    ax.set_title(title)
    ax.set_xlabel('Coefficient')

    if len(results_list) > 1:
        ax.legend()

    fig.tight_layout()
    return fig
