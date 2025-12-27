from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .storage import ensure_dirs


@dataclass(frozen=True)
class PriceFetchResult:
    ticker: str
    ok: bool
    message: str
    rows: int = 0


def _session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({'User-Agent': user_agent, 'Accept': '*/*'})
    return s


def fetch_daily_stooq(ticker: str, session: requests.Session, timeout: int = 20) -> pd.DataFrame:
    """Fetch daily OHLCV from Stooq (free, no key)."""
    # Stooq uses .us suffix for US tickers. We keep user tickers like AAPL and convert.
    sym = ticker.strip().lower()
    if not sym:
        return pd.DataFrame()
    if '.' not in sym:
        sym = f"{sym}.us"

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    # Stooq CSV: Date,Open,High,Low,Close,Volume
    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        return df
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def fetch_daily_yahoo(ticker: str, session: requests.Session, timeout: int = 20) -> pd.DataFrame:
    """Fallback daily fetch using Yahoo chart endpoint (may be blocked sometimes)."""
    import json

    sym = ticker.strip().upper()
    if not sym:
        return pd.DataFrame()

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=5y"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    result = (data.get('chart') or {}).get('result')
    if not result:
        return pd.DataFrame()
    res = result[0]
    ts = res.get('timestamp') or []
    ind = ((res.get('indicators') or {}).get('quote') or [{}])[0]

    rows = []
    for i, t in enumerate(ts):
        try:
            dt = datetime.utcfromtimestamp(int(t)).date()
            rows.append(
                {
                    'date': pd.to_datetime(dt),
                    'open': (ind.get('open') or [None])[i],
                    'high': (ind.get('high') or [None])[i],
                    'low': (ind.get('low') or [None])[i],
                    'close': (ind.get('close') or [None])[i],
                    'volume': (ind.get('volume') or [None])[i],
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['date', 'close']).sort_values('date')
    return df


def update_price_cache(
    ticker: str,
    out_dir: Path,
    user_agent: str,
    prefer: str = 'stooq',
    sleep_s: float = 0.5,
) -> PriceFetchResult:
    """Fetch and store daily prices as CSV in out_dir/<TICKER>.csv."""
    ensure_dirs(out_dir)
    t = ticker.strip().upper()
    if not t:
        return PriceFetchResult(ticker=ticker, ok=False, message='Empty ticker')

    s = _session(user_agent)
    df = pd.DataFrame()
    err = None

    try:
        if prefer == 'yahoo':
            df = fetch_daily_yahoo(t, s)
        else:
            df = fetch_daily_stooq(t, s)
    except Exception as e:
        err = e

    # Fallback try
    if df.empty:
        try:
            df = fetch_daily_yahoo(t, s)
        except Exception as e:
            err = err or e

    time.sleep(max(0.0, sleep_s))

    if df.empty:
        msg = f"No data (last error: {err})" if err else "No data"
        return PriceFetchResult(ticker=t, ok=False, message=msg, rows=0)

    out_path = out_dir / f"{t}.csv"
    df.to_csv(out_path, index=False)
    return PriceFetchResult(ticker=t, ok=True, message='OK', rows=len(df))


def load_price_cache(ticker: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{ticker.strip().upper()}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
    return df
