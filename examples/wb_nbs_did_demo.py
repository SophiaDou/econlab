"""
WB + NBS DID end-to-end demo (robust to CSV/Excel and CN encodings).

- Detects Excel files even if renamed as .csv (OOXML signature).
- Tries common CN encodings for CSV and auto-detects delimiters.
- Normalizes NBS columns to: prov, year, y, Tstart.
- Builds D = 1(year >= Tstart & Tstart notnull) and runs TWFE DID + event study.
- Saves event study figure to output/event_study_cn.png.
"""

import os
import io
import sys
import csv
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from econlab.causal.did import twfe_did, event_study, parallel_trend_test

# -----------------------------
# Config
# -----------------------------
WB_COUNTRIES = 'CHN;KOR;JPN;USA'
WB_INDICATOR = 'FP.CPI.TOTL.ZG'  # inflation yoy as example control
WB_RANGE = '2000:2023'
NBS_PATH = 'data/nbs_panel.csv'  # can actually be Excel even if extension says .csv

# -----------------------------
# Helpers
# -----------------------------
def is_ooxml_excel(path: str) -> bool:
    """Return True if file looks like Excel OOXML (xlsx) based on ZIP signature."""
    try:
        with open(path, 'rb') as f:
            sig = f.read(4)
        # OOXML (xlsx) is a ZIP file => starts with PK\x03\x04
        return sig.startswith(b'PK\x03\x04')
    except Exception:
        return False


def read_nbs_table(path: str) -> pd.DataFrame:
    """
    Robust reader for NBS data:
    1) If OOXML signature detected, read as Excel (needs openpyxl).
    2) Else try CSV with common CN encodings and auto-sniffed separators.
    3) As last resort, try read_excel() blindly (covers some edge cases).
    """
    # 1) Signature-based Excel read
    if is_ooxml_excel(path):
        try:
            import openpyxl  # ensure engine present
        except ImportError as e:
            raise RuntimeError(
                "This file looks like an Excel .xlsx (OOXML). Please install openpyxl:\n"
                "    pip install openpyxl"
            ) from e
        return pd.read_excel(path, engine='openpyxl')

    # 2) Try CSV with common encodings + auto-sep
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk', 'cp936', 'utf-16']
    for enc in encodings:
        try:
            # sep=None + engine='python' => delimiter sniffing
            return pd.read_csv(path, sep=None, engine='python', encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except Exception:
            continue

    # 3) Fallback: read_excel without signature (handles some misnamed files)
    try:
        return pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(
            "Failed to read NBS file with common CSV encodings/separators, "
            "and read_excel also failed. Please re-save as UTF-8 CSV or a genuine .xlsx file."
        ) from e


def normalize_nbs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns are exactly: prov, year, y, Tstart.
    Tries to auto-rename common Chinese headers, trims whitespace, coerces types.
    """
    # Strip whitespace in column names
    df = df.copy()
    # df.columns = [str(c).strip() for c in df.columns]
    df.columns = [
        str(c).replace('\ufeff', '').replace('\uFEFF', '').replace('\u3000', ' ').strip()
        for c in df.columns
    ]

    # Common mappings
    candidate_map = {
        '省份': 'prov',
        '地区': 'prov',
        '地区名称': 'prov',
        '省份代码': 'prov',

        '年份': 'year',
        '年度': 'year',
        '时间': 'year',

        '指标': 'y',
        '值': 'y',
        '因变量': 'y',
        '目标值': 'y',

        '政策开始年份': 'Tstart',
        '处理开始年份': 'Tstart',
        '开始年份': 'Tstart',
        '政策起始年': 'Tstart',
        '实施年份': 'Tstart',
    }

    # Try to map any non-English headers first
    mapped = {c: candidate_map[c] for c in df.columns if c in candidate_map}
    df = df.rename(columns=mapped)

    # If still missing, check for plausible English-ish variants
    english_guess = {
        'province': 'prov', 'prov_name': 'prov', 'prov_code': 'prov', 'prov': 'prov',
        'year': 'year', 'yr': 'year',
        'value': 'y', 'y_value': 'y', 'dep': 'y', 'outcome': 'y',
        'tstart': 'Tstart', 'treatment_start': 'Tstart', 'policy_start': 'Tstart'
    }
    mapped2 = {c: english_guess[c] for c in df.columns if c in english_guess}
    df = df.rename(columns=mapped2)

    # Required columns
    required = {'prov', 'year', 'y', 'Tstart'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['Tstart'] = pd.to_numeric(df['Tstart'], errors='coerce')  # may be NaN
    df = df.dropna(subset=['prov', 'year', 'y']).reset_index(drop=True)

    # Prefer plain int64 for year after dropping NaN
    df['year'] = df['year'].astype(int)
    return df

def get_wb_cn_cpi_yoy() -> pd.DataFrame:
    """
    Fetch CN CPI yoy from World Bank; returns columns: [year, cpi_yoy].
    If offline, returns empty DataFrame (merge will yield NaNs).
    """
    try:
        import requests
        BASE = "https://api.worldbank.org/v2"
        params = {"format": "json", "per_page": 20000, "date": WB_RANGE}
        url = f"{BASE}/country/{WB_COUNTRIES}/indicator/{WB_INDICATOR}"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()[1]
        rows = []
        for d in data:
            rows.append({
                'country': d['country']['id'],
                'year': int(d['date']),
                'cpi_yoy': d['value']
            })
        wb = pd.DataFrame(rows).dropna()
        wb_cn = wb[wb['country'] == 'CHN'][['year', 'cpi_yoy']]
        return wb_cn
    except Exception as e:
        warnings.warn(f"WB fetch failed (offline or API issue). Proceeding without control. Error: {e}")
        return pd.DataFrame(columns=['year', 'cpi_yoy'])


def make_event_study_figure(es_result: dict, out_path: str):
    """Plot event-study coefficients with 95% CI."""
    ev_params = es_result['event_params']
    ev_se = es_result['event_se']
    if not ev_params:
        warnings.warn("No event-time coefficients to plot.")
        return
    ks = sorted(int(k.split('_')[1]) for k in ev_params.keys())
    vals = [ev_params[f'k_{k}'] for k in ks]
    se = [ev_se[f'k_{k}'] for k in ks]
    ci_low = [v - 1.96*s for v, s in zip(vals, se)]
    ci_high = [v + 1.96*s for v, s in zip(vals, se)]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.axhline(0, color='gray', lw=1)
    plt.fill_between(ks, ci_low, ci_high, color='#d0e3ff', alpha=0.7, label='95% CI')
    plt.plot(ks, vals, marker='o', color='#1f77b4', label='Event-time coef')
    plt.axvline(-1, color='red', ls='--', label='Baseline (-1)')
    plt.title('Event Study: Dynamic Effects')
    plt.xlabel('Event time k')
    plt.ylabel('Effect on outcome (coef)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'Saved figure: {out_path}')

def ensure_ym(df, date_col='date', year_col='year', month_col='month', ym_col='ym'):
    """
    Create a monthly period column `ym` (datetime64[ns] at month start) in-place if not present.
    Tries in order:
      1) existing `ym` parseable to datetime,
      2) `date` parseable to datetime, normalized to month start,
      3) combination of `year` and `month`.
    Raises ValueError with a clear message if none work.
    """
    import pandas as pd
    if ym_col in df.columns:
        # Normalize if it exists but might be string/object
        df[ym_col] = pd.to_datetime(df[ym_col], errors='coerce')
        if df[ym_col].isna().all():
            raise ValueError("Existing 'ym' column cannot be parsed to datetime; "
                             "ensure it’s in 'YYYY-MM' or parseable format.")
        # normalize to month start
        df[ym_col] = df[ym_col].values.astype('datetime64[M]')
        return df

    # Try date column
    if date_col in df.columns:
        ser = pd.to_datetime(df[date_col], errors='coerce')
        if ser.notna().any():
            df[ym_col] = ser.values.astype('datetime64[M]')
            return df

    # Try year + month
    if year_col in df.columns and month_col in df.columns:
        # Be tolerant to strings/numbers
        yr = pd.to_numeric(df[year_col], errors='coerce')
        mo = pd.to_numeric(df[month_col], errors='coerce')
        mask = yr.notna() & mo.notna()
        if mask.any():
            # Build YYYY-MM-01
            tmp = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
            tmp.loc[mask] = pd.to_datetime(
                {'year': yr.loc[mask].astype(int),
                 'month': mo.loc[mask].astype(int),
                 'day': 1},
                errors='coerce'
            )
            if tmp.notna().any():
                df[ym_col] = tmp.values.astype('datetime64[M]')
                return df

    raise ValueError(
        "Could not create 'ym'. Provide one of: "
        "(a) parseable 'ym' column, (b) parseable 'date' column, or "
        "(c) numeric 'year' and 'month' columns."
    )

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Read NBS
    if not os.path.exists(NBS_PATH):
        raise SystemExit(
            f"Not found: {NBS_PATH}\n"
            "Please place your NBS panel at this path. It can be CSV or Excel (even if misnamed)."
        )
    nbs = read_nbs_table(NBS_PATH)

    print("Loaded from:", os.path.abspath(NBS_PATH))
    print("Columns seen:", list(nbs.columns))
    print([repr(c) for c in nbs.columns])

    print("Column check OK:", list(nbs.columns))
    print(nbs.head())
    nbs = normalize_nbs_columns(nbs)

    # 2) WB CPI (control)
    wb_cn = get_wb_cn_cpi_yoy()

    # 3) Merge & construct D
    panel = nbs.merge(wb_cn, on='year', how='left')  # cpi_yoy may be NaN if offline
    
    # Choose controls only if they have any non-missing values
    control_cols = []
    if 'cpi_yoy' in panel.columns and panel['cpi_yoy'].notna().any():
        control_cols = ['cpi_yoy']

    # Construct D
    Tstart_num = pd.to_numeric(panel['Tstart'], errors='coerce')
    panel['D'] = ((Tstart_num.notna()) & (panel['year'] >= Tstart_num)).astype(int)

    print("D unique:", panel['D'].unique())
    print(panel.groupby('prov')['year'].agg(['min','max','nunique']))
    print("Unique provinces in panel:", panel['prov'].nunique())
    if 'cpi_yoy' in panel.columns:
        print("Non-missing cpi_yoy:", int(panel['cpi_yoy'].notna().sum()))

    # --- Basic diagnostics before TWFE ---
    unit = 'prov'
    time = 'year'   # use annual; estimation below uses 'year'
    y    = 'y'
    treat= 'D'

    # Remove or comment out this line for annual data:
    # panel = ensure_ym(panel, date_col='date', year_col='year', month_col='month', ym_col='ym')

    print("Rows:", len(panel))
    print("Unique units:", panel[unit].nunique())
    print("Unique time periods:", panel[time].nunique())
    print("Non-missing outcome:", panel[y].notna().sum())
    print("Treatment variation:", panel[treat].dropna().unique())

    print(panel.groupby(unit)[y].apply(lambda s: s.notna().sum()).describe())
    print(panel.groupby(unit)[time].nunique().describe())

    # Optional NaN checks on columns that actually exist:
    for col in [y, 'cpi_yoy', 'year']:
        if col in panel:
            na_rate = panel[col].isna().mean()
            print(f"{col}: NaN rate = {na_rate:.3f}")

    # 4) TWFE DID
    # ---- TWFE DID ----
    res = twfe_did(
        panel,
        y='y', treatment='D', unit='prov', time='year',
        controls=(control_cols or None),
        cluster='prov'
    )
    print({'D_hat': res['coef_D'], 'se': res['se_D']})

    # 5) Event Study + parallel trend test
    es = event_study(
        panel,
        y='y', unit='prov', time='year', treat_start='Tstart',
        window=(-5, 5),
        controls=(control_cols or None),
        cluster='prov'
    )

    pre = [-5, -4, -3, -2]
    jt = parallel_trend_test(es, pre_k=pre)
    print({'parallel_trend_F': jt['F'], 'p': jt['pval']})

    # 6) Plot figure
    make_event_study_figure(es, out_path='output/event_study_cn.png')


if __name__ == '__main__':
    main()