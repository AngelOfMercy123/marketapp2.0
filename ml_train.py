from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DATA_DIR = "data_prices"
BASELINE = "SPY"


def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    def _try(sym: str) -> pd.DataFrame:
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        text = r.text.strip()
        if not text or "Date" not in text.splitlines()[0]:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(text))
        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            return pd.DataFrame()
        return df

    t = ticker.lower().strip()
    df = _try(t)
    if not df.empty:
        return df
    return _try(f"{t}.us")


def cache_price(ticker: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker.upper()}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df
    df = fetch_stooq_daily(ticker)
    if df.empty or len(df) < 260:
        return pd.DataFrame()
    out = df.copy()
    out.reset_index().to_csv(path, index=False)
    return df


def friday_close_series(df: pd.DataFrame) -> pd.Series:
    return df["close"].resample("W-FRI").last().dropna()


def rolling_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=w).mean()


def weekly_vol_from_daily(close: pd.Series) -> pd.Series:
    # rolling 20d daily vol annualized to weekly ~sqrt(5)
    rets = close.pct_change()
    return rets.rolling(20, min_periods=20).std() * np.sqrt(5)


def build_features_for_ticker(
    ticker: str,
    baseline_weekly: pd.Series,
    labor_exposure: float,
    jan_score: float,
) -> pd.DataFrame:
    df = cache_price(ticker)
    if df.empty:
        return pd.DataFrame()

    w = friday_close_series(df)
    common = w.index.intersection(baseline_weekly.index)
    w = w.loc[common]
    b = baseline_weekly.loc[common]

    if len(w) < 80:
        return pd.DataFrame()

    # returns
    r1 = w.pct_change(1)
    r4 = (w / w.shift(4)) - 1.0
    b1 = b.pct_change(1)
    b4 = (b / b.shift(4)) - 1.0

    # relative strength (uses info up to this week)
    rs1 = r1 - b1
    rs4 = r4 - b4

    # daily features sampled on Fridays
    close = df["close"]
    ma20 = rolling_ma(close, 20)
    ma50 = rolling_ma(close, 50)
    volw = weekly_vol_from_daily(close)

    # align daily->weekly (take last value before/at Friday)
    ma20_w = ma20.reindex(w.index, method="ffill")
    ma50_w = ma50.reindex(w.index, method="ffill")
    volw_w = volw.reindex(w.index, method="ffill")

    above_ma20 = (w > ma20_w).astype(int)
    ma_stack = (ma20_w > ma50_w).astype(int)
    ma_gap = (ma20_w / ma50_w) - 1.0

    # month seasonality as numeric
    month = pd.Series(w.index.month, index=w.index).astype(int)
    is_jan = (month == 1).astype(int)

    # macro regime (baseline momentum)
    spy_above_ma20 = (b > b.rolling(20, min_periods=20).mean()).astype(int)
    spy_r4 = b4

    # ---- LABEL (next week outperformance) ----
    # label for week t uses NEXT week excess return (t+1)
    y = (rs1.shift(-1) > 0).astype(int)

    X = pd.DataFrame({
        "ticker": ticker,
        "date": w.index,
        "ret_1w": r1,
        "rs_1w": rs1,
        "rs_4w": rs4,
        "mom_4w": r4,
        "above_ma20": above_ma20,
        "ma_stack": ma_stack,
        "ma_gap": ma_gap,
        "vol_weekly": volw_w,
        "spy_rs_4w": spy_r4,
        "spy_above_ma20": spy_above_ma20,
        "labor_exposure": float(labor_exposure),
        "jan_score": float(jan_score),
        "is_january": is_jan,
        "month": month,
        "y": y,
    }).dropna()

    # remove last row (label uses next week)
    X = X.iloc[:-1].copy()
    return X


def main():
    uni = pd.read_csv("universe.csv")
    uni["ticker"] = uni["ticker"].astype(str).str.upper().str.strip()

    # baseline
    spy = cache_price(BASELINE)
    if spy.empty:
        raise RuntimeError("Failed to load SPY from Stooq.")
    b_weekly = friday_close_series(spy)

    all_rows = []
    for _, row in uni.iterrows():
        t = row["ticker"]
        le = row["labor_exposure"] if "labor_exposure" in row else 5
        js = row["jan_score"] if "jan_score" in row else 0
        feat = build_features_for_ticker(t, b_weekly, le, js)
        if not feat.empty:
            all_rows.append(feat)

    if not all_rows:
        raise RuntimeError("No training data built. Check tickers or connectivity.")

    data = pd.concat(all_rows, ignore_index=True)

    feature_cols = [
        "ret_1w","rs_1w","rs_4w","mom_4w",
        "above_ma20","ma_stack","ma_gap",
        "vol_weekly",
        "spy_rs_4w","spy_above_ma20",
        "labor_exposure",
        "jan_score","is_january","month"
    ]

    X = data[feature_cols].astype(float)
    y = data["y"].astype(int)

    # Time-series validation (no random shuffle)
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    accs, aucs = [], []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)

        accs.append(accuracy_score(yte, yhat))
        try:
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            pass

    print(f"CV Accuracy avg: {np.mean(accs):.3f}")
    if aucs:
        print(f"CV AUC avg:      {np.mean(aucs):.3f}")

    # Fit final model on all data
    pipe.fit(X, y)
    dump(pipe, "model.joblib")
    with open("feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print("Saved: model.joblib, feature_cols.json")
    print(f"Rows: {len(data)}, Tickers used: {data['ticker'].nunique()}")


if __name__ == "__main__":
    main()
