from __future__ import annotations

from io import StringIO
from typing import List

import pandas as pd
import requests


def fetch_sp500_constituents(session: requests.Session, timeout: int = 20) -> List[str]:
    """Fetch S&P 500 constituents from a public GitHub dataset (CSV).

    Returns list of tickers (US style, e.g. AAPL).
    """
    url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    tickers = df['Symbol'].astype(str).str.strip().tolist()
    # Normalize: replace '.' with '-' to match common data sources
    return [t.replace('.', '-') for t in tickers if t]
