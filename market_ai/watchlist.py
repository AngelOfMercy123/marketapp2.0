from __future__ import annotations

from pathlib import Path

import pandas as pd

from .storage import safe_read_csv, safe_write_csv


DEFAULT_WATCHLIST_COLUMNS = ['ticker', 'notes']


def load_watchlist(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path, DEFAULT_WATCHLIST_COLUMNS)
    for c in DEFAULT_WATCHLIST_COLUMNS:
        if c not in df.columns:
            df[c] = None
    if not df.empty:
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        df = df.drop_duplicates(subset=['ticker']).reset_index(drop=True)
    return df[DEFAULT_WATCHLIST_COLUMNS]


def save_watchlist(df: pd.DataFrame, path: Path) -> None:
    if 'ticker' in df.columns:
        df = df.copy()
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        df = df.drop_duplicates(subset=['ticker']).reset_index(drop=True)
    safe_write_csv(df, path)


def set_watchlist_from_tickers(tickers: list[str]) -> pd.DataFrame:
    tickers = [t.upper().strip() for t in tickers if t.strip()]
    return pd.DataFrame({'ticker': tickers, 'notes': [''] * len(tickers)})
