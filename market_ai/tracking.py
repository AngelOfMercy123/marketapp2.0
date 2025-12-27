from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import safe_read_csv, safe_write_csv


PRED_COLS = [
    'timestamp_utc',
    'ticker',
    'horizon_days',
    'pred_prob_up',
    'pred_class',
    'model_tag',
    'close_at_pred',
    'target_close_date',
    'actual_close',
    'actual_class',
    'hit',
]


@dataclass
class PredictionRecord:
    timestamp_utc: str
    ticker: str
    horizon_days: int
    pred_prob_up: float
    pred_class: int
    model_tag: str
    close_at_pred: float
    target_close_date: str


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def append_prediction(log_path: Path, rec: PredictionRecord) -> None:
    df = safe_read_csv(log_path, PRED_COLS)
    row = {
        'timestamp_utc': rec.timestamp_utc,
        'ticker': rec.ticker,
        'horizon_days': rec.horizon_days,
        'pred_prob_up': float(rec.pred_prob_up),
        'pred_class': int(rec.pred_class),
        'model_tag': rec.model_tag,
        'close_at_pred': float(rec.close_at_pred),
        'target_close_date': rec.target_close_date,
        'actual_close': np.nan,
        'actual_class': np.nan,
        'hit': np.nan,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    safe_write_csv(log_path, df)


def reconcile_predictions(log_path: Path, price_lookup: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fill actual outcomes for rows that have reached their target date."""
    df = safe_read_csv(log_path, PRED_COLS)
    if df.empty:
        return df

    df2 = df.copy()
    df2['target_close_date'] = pd.to_datetime(df2['target_close_date'], errors='coerce')
    df2['timestamp_utc'] = pd.to_datetime(df2['timestamp_utc'], errors='coerce')

    now = pd.Timestamp.utcnow().tz_localize(None)

    for i, row in df2.iterrows():
        if pd.notna(row.get('hit')):
            continue
        tkr = str(row['ticker']).strip().upper()
        tgt = row['target_close_date']
        if pd.isna(tgt) or tgt > now:
            continue
        px = price_lookup.get(tkr)
        if px is None or px.empty:
            continue
        # Find closest close on/after target date
        px = px.sort_index()
        idx = px.index.searchsorted(tgt)
        if idx >= len(px.index):
            continue
        actual_close = float(px.iloc[idx]['Close'])
        close_at_pred = float(row['close_at_pred'])
        actual_class = int(actual_close > close_at_pred)
        pred_class = int(row['pred_class'])
        hit = int(actual_class == pred_class)

        df2.at[i, 'actual_close'] = actual_close
        df2.at[i, 'actual_class'] = actual_class
        df2.at[i, 'hit'] = hit

    df2['target_close_date'] = df2['target_close_date'].dt.strftime('%Y-%m-%d')
    df2['timestamp_utc'] = df2['timestamp_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    safe_write_csv(log_path, df2)
    return df2


def summarize_accuracy(df_preds: pd.DataFrame) -> dict[str, float]:
    done = df_preds.dropna(subset=['hit'])
    if done.empty:
        return {'n': 0, 'accuracy': float('nan')}
    return {'n': int(len(done)), 'accuracy': float(done['hit'].mean())}


def now_utc_iso() -> str:
    return _now_utc_iso()
