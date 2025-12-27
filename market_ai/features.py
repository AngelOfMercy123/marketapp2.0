from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder smoothing (EMA)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def make_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given OHLCV dataframe indexed by date, compute a compact feature set."""
    out = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    vol = df.get('volume')

    ret1 = close.pct_change()
    out['ret_1'] = ret1
    out['ret_5'] = close.pct_change(5)
    out['ret_20'] = close.pct_change(20)

    out['vol_5'] = ret1.rolling(5).std()
    out['vol_20'] = ret1.rolling(20).std()

    out['ma_5'] = close.rolling(5).mean() / close - 1
    out['ma_20'] = close.rolling(20).mean() / close - 1
    out['ma_50'] = close.rolling(50).mean() / close - 1

    out['range_pct'] = (high - low) / close.replace(0, np.nan)
    out['gap_pct'] = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)

    out['rsi_14'] = _rsi(close, 14)

    if vol is not None:
        out['vol_chg_5'] = vol.pct_change(5)
        out['obv'] = (np.sign(close.diff()).fillna(0) * vol).cumsum()
        out['obv_chg_20'] = out['obv'].pct_change(20)
    else:
        out['vol_chg_5'] = np.nan
        out['obv_chg_20'] = np.nan

    # Clean
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def make_target(df: pd.DataFrame, horizon: int = 5, kind: str = 'direction') -> pd.Series:
    """Target for supervised learning.

    kind:
      - 'direction': 1 if close(t+h) > close(t), else 0
      - 'return': forward return (continuous)
    """
    close = df['close']
    fwd = close.shift(-horizon) / close - 1
    if kind == 'return':
        return fwd
    return (fwd > 0).astype(int)


def align_features_target(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df = X.join(y.rename('target'), how='inner')
    df = df.dropna()
    y2 = df.pop('target')
    return df, y2
