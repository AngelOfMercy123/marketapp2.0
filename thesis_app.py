from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from tabulate import tabulate


# ----------------------------
# Data fetch (Stooq CSV)
# ----------------------------

def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Stooq as DataFrame indexed by date (UTC naive).
    Stooq uses lowercase tickers for US (e.g., aapl.us). For ETFs/stocks, ticker
    usually works without suffix on their endpoint, but we'll try both.
    """
    def _try(sym: str) -> pd.DataFrame:
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = df = pd.read_csv(io.StringIO(r.text))
        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # standardize columns
        cols = {c.lower(): c for c in df.columns}
        for required in ["close"]:
            if required not in cols:
                return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        return df

    # Try plain
    df = _try(ticker.lower())
    if not df.empty:
        return df

    # Try US suffix
    df = _try(f"{ticker.lower()}.us")
    return df


def ensure_history(df: pd.DataFrame, min_rows: int = 120) -> pd.DataFrame:
    if df.empty or len(df) < min_rows:
        raise RuntimeError(f"Not enough price history (got {len(df)})")
    return df


# ----------------------------
# Indicators
# ----------------------------

def last_trading_day_of_month(df: pd.DataFrame, year: int, month: int) -> pd.Timestamp | None:
    mask = (df.index.year == year) & (df.index.month == month)
    sub = df.loc[mask]
    if sub.empty:
        return None
    return sub.index.max()

def friday_close_series(df: pd.DataFrame) -> pd.Series:
    # Resample to weekly Friday close. If market closed Friday, it uses last available in that week.
    return df["close"].resample("W-FRI").last().dropna()

def rolling_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def pct_return(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return float("nan")
    return (series.iloc[-1] / series.iloc[-1 - periods]) - 1.0

def drawdown_from_20d_high(close: pd.Series) -> float:
    if len(close) < 20:
        return float("nan")
    last20 = close.iloc[-20:]
    hi = last20.max()
    if hi <= 0:
        return float("nan")
    return (close.iloc[-1] / hi) - 1.0  # negative if below high


# ----------------------------
# Thesis scoring model
# ----------------------------

@dataclass
class ManualInputs:
    ticker: str
    labor_exposure: int  # 0..10
    labor_shock: int     # 0/1
    catalyst: int        # -2..+2
    trend_flag: int      # 0/1 (manual confirmation)
    jan_score: int       # -2..+2
    volatile: bool       # widen stops
    earnings_in_days: int  # manual countdown for de-risk rule


@dataclass
class Computed:
    trend: float
    macro: float
    labor: float
    thesis_score: float
    rs_4w: float
    action: str
    reason: str


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sign(x: float) -> int:
    if math.isnan(x):
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def is_january(now: datetime) -> bool:
    return now.month == 1

def compute_macro(spy_close: pd.Series) -> float:
    ma20 = rolling_ma(spy_close, 20)
    ma50 = rolling_ma(spy_close, 50)
    spy_ma = 0
    if not np.isnan(ma20.iloc[-1]) and spy_close.iloc[-1] > ma20.iloc[-1]:
        spy_ma += 1
    if not np.isnan(ma50.iloc[-1]) and not np.isnan(ma20.iloc[-1]) and ma20.iloc[-1] > ma50.iloc[-1]:
        spy_ma += 1
    dd = drawdown_from_20d_high(spy_close)
    risk_off = 1 if (not np.isnan(dd) and dd < -0.04) else 0
    # Macro score 0..10
    return clamp(5 + 2*spy_ma - 3*risk_off, 0, 10)

def compute_trend(stock_daily: pd.DataFrame, stock_weekly: pd.Series,
                  spy_weekly: pd.Series, trend_flag: int) -> Tuple[float, float]:
    # RS: 4-week return (weekly) vs SPY
    stock_4w = pct_return(stock_weekly, 4)
    spy_4w = pct_return(spy_weekly, 4)
    rs = (stock_4w - spy_4w) if (not np.isnan(stock_4w) and not np.isnan(spy_4w)) else float("nan")

    close = stock_daily["close"]
    ma20 = rolling_ma(close, 20)
    ma50 = rolling_ma(close, 50)

    ma_score = 0
    if not np.isnan(ma20.iloc[-1]) and close.iloc[-1] > ma20.iloc[-1]:
        ma_score += 1
    if not np.isnan(ma50.iloc[-1]) and not np.isnan(ma20.iloc[-1]) and ma20.iloc[-1] > ma50.iloc[-1]:
        ma_score += 1

    mom_1w = pct_return(stock_weekly, 1)

    # Trend score 0..10 (price proxy + manual trend confirmation)
    trend = clamp(5 + 2*sign(rs) + 2*ma_score + 1*sign(mom_1w) + 1*trend_flag, 0, 10)
    return trend, rs

def compute_labor(labor_exposure: int, labor_shock: int) -> float:
    return clamp(labor_exposure * (0.6 + 0.4*int(bool(labor_shock))), 0, 10)

def compute_thesis_score(trend: float, macro: float, labor: float, catalyst: int, jan_score: int) -> float:
    # Convert catalyst -2..+2 -> roughly -10..+10 impact scale
    cat_term = (catalyst * 5.0)
    score = 0.45*trend + 0.25*macro - 0.20*labor + 0.10*cat_term
    # January overlay
    now = datetime.now(timezone.utc)
    if is_january(now):
        score += 1.5 * jan_score
    return clamp(score, 0, 10)

def action_logic(thesis_score: float, trend: float, rs_4w: float,
                 stock_daily: pd.DataFrame, earnings_in_days: int, volatile: bool) -> Tuple[str, str]:
    close = stock_daily["close"]
    ma20 = rolling_ma(close, 20)
    # entry conditions
    above_ma20 = (not np.isnan(ma20.iloc[-1]) and close.iloc[-1] > ma20.iloc[-1])

    # event-risk rule
    if earnings_in_days <= 7 and thesis_score < 7.5:
        return "SELL/DE-RISK", "Earnings within 7 trading days and score < 7.5"

    # buy/hold logic
    if thesis_score >= 6.8 and trend >= 6 and above_ma20:
        return "BUY", "Score strong + trend strong + above MA20"
    if thesis_score >= 6.0:
        return "HOLD", "Score >= 6.0"

    # trim/sell logic
    if thesis_score < 6.0:
        return "TRIM/SELL", "Score fell below 6.0"

    return "WATCH", "No clear signal"

# ----------------------------
# Run
# ----------------------------

def evaluate_watchlist(config_path: str) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    baseline = cfg.get("baseline", "SPY")
    items = [ManualInputs(**x) for x in cfg["tickers"]]

    # Fetch SPY
    spy_df = ensure_history(fetch_stooq_daily(baseline))
    spy_weekly = friday_close_series(spy_df)
    macro = compute_macro(spy_df["close"])

    rows = []
    for it in items:
        df = ensure_history(fetch_stooq_daily(it.ticker))
        w = friday_close_series(df)

        # align weekly indices
        common = w.index.intersection(spy_weekly.index)
        w2 = w.loc[common]
        spy2 = spy_weekly.loc[common]

        trend, rs_4w = compute_trend(df, w2, spy2, it.trend_flag)
        labor = compute_labor(it.labor_exposure, it.labor_shock)
        thesis = compute_thesis_score(trend, macro, labor, it.catalyst, it.jan_score)
        action, reason = action_logic(thesis, trend, rs_4w, df, it.earnings_in_days, it.volatile)

        vol_w = weekly_volatility(df["close"])
        bias, bucket, conf, ex_pct = predict_next_week(thesis_score, vol_w)


        rows.append({
            "Ticker": it.ticker,
            "Action": action,
            "ThesisScore": round(thesis, 2),
            "Trend": round(trend, 2),
            "Macro": round(macro, 2),
            "Labor": round(labor, 2),
            "Catalyst": it.catalyst,
            "JanScore": it.jan_score,
            "RS_4w_vs_SPY": None if np.isnan(rs_4w) else round(rs_4w * 100, 2),
            "Reason": reason
            "Prediction": f"{bias} ({bucket})",
            "ExpectedExcess_vs_SPY_%": round(ex_pct, 2),
            "Confidence": conf,
            "WeeklyVol_%": None if math.isnan(vol_w) else round(vol_w * 100, 2),

        })

    return pd.DataFrame(rows).sort_values(by="ThesisScore", ascending=False)


def main():
    df = evaluate_watchlist("watchlist.json")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()
