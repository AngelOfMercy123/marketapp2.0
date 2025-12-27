from __future__ import annotations

import io
import requests
import pandas as pd

STOOQ_URL = "https://stooq.com/q/d/l/"


def fetch_prices_stooq(stooq_symbol: str, timeout: int = 30) -> pd.DataFrame:
    """
    Returns DataFrame with columns: Date, Open, High, Low, Close, Volume
    stooq_symbol examples: aapl.us, spy.us
    """
    params = {"s": stooq_symbol, "i": "d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(STOOQ_URL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df = df.sort_values("Date").reset_index(drop=True)
    return df
