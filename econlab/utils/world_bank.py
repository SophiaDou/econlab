from __future__ import annotations
import requests
import pandas as pd
from typing import Optional

BASE = "http://api.worldbank.org/v2"

def get_indicator(country: str, indicator: str, date: Optional[str] = None, per_page: int = 20000) -> pd.DataFrame:
    """Fetch indicator from World Bank API as long format [country,date,value]."""
    params = {"format": "json", "per_page": per_page}
    if date:
        params["date"] = date
    url = f"{BASE}/country/{country}/indicator/{indicator}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()[1]
    rows = []
    for d in data:
        rows.append({'country': d['country']['id'], 'date': int(d['date']), 'value': d['value']})
    return pd.DataFrame(rows).dropna()