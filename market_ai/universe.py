from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .storage import safe_read_csv, safe_write_csv


DEFAULT_UNIVERSE_COLUMNS = [
    'ticker',
    # "Structural" factors (manual / heuristic)
    'utility_score',         # 0-10: does it make life easier / higher demand
    'policy_score',          # 0-10: regulatory / country/state friendliness
    'leadership_score',      # 0-10: management quality signal
    'labor_exposure',        # 0-10: labour intensity / wage inflation sensitivity
    'sector',
    'country',

    # Demographic / workforce fields (aggregated)
    'avg_workforce_age',     # e.g., 35-55
    'pct_physical_labor',    # 0-100
    'pct_female',            # 0-100 (optional, descriptive)
    'pct_immigrant',         # 0-100 (visa/policy exposure proxy)
    'retirement_age',        # 50-70 typical for industry
    'workforce_stability',   # 0-10 retention / turnover inverse
]


def load_universe(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path, DEFAULT_UNIVERSE_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=DEFAULT_UNIVERSE_COLUMNS)
    # Ensure required columns exist
    for c in DEFAULT_UNIVERSE_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df = df.drop_duplicates(subset=['ticker']).reset_index(drop=True)
    return df[DEFAULT_UNIVERSE_COLUMNS]


def save_universe(df: pd.DataFrame, path: Path) -> None:
    if 'ticker' in df.columns:
        df = df.copy()
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        df = df.drop_duplicates(subset=['ticker']).reset_index(drop=True)
    safe_write_csv(df, path)


def upsert_tickers(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    tickers = [t.upper().strip() for t in tickers if t.strip()]
    existing = set(df['ticker'].astype(str).str.upper()) if 'ticker' in df.columns else set()
    add = [t for t in tickers if t not in existing]
    if not add:
        return df
    rows = []
    for t in add:
        row = {c: None for c in DEFAULT_UNIVERSE_COLUMNS}
        row['ticker'] = t
        # Sensible defaults to reduce blank work
        row['utility_score'] = 5
        row['policy_score'] = 5
        row['leadership_score'] = 5
        row['labor_exposure'] = 5
        row['avg_workforce_age'] = 40
        row['pct_physical_labor'] = 30
        row['pct_female'] = 50
        row['pct_immigrant'] = 20
        row['retirement_age'] = 65
        row['workforce_stability'] = 5
        rows.append(row)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
