from __future__ import annotations

import io
import json
import math
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config
# ----------------------------
DATA_DIR = "data_prices"
MODEL_PATH = "model.joblib"
FEATURES_PATH = "feature_cols.json"


# ----------------------------
# Data fetch (Stooq CSV)
# ----------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
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

    df.reset_index().to_csv(path, index=False)
    return df


def ensure_history(df: pd.DataFrame, min_rows: int = 260) -> pd.DataFrame:
    if df.empty or len(df) < min_rows:
        raise RuntimeError(f"Not enough history (got {len(df)})")
    return df


def friday_close_series(df: pd.DataFrame) -> pd.Series:
    return df["close"].resample("W-FRI").last().dropna()


def rolling_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=w).mean()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def weekly_vol_from_daily(close: pd.Series) -> pd.Series:
    rets = close.pct_change()
    return rets.rolling(20, min_periods=20).std() * np.sqrt(5)


def is_january() -> bool:
    return datetime.now().month == 1


# ----------------------------
# Feature engineering for AI
# ----------------------------
def build_weekly_features(
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

    r1 = w.pct_change(1)
    r4 = (w / w.shift(4)) - 1.0
    b1 = b.pct_change(1)
    b4 = (b / b.shift(4)) - 1.0

    rs1 = r1 - b1
    rs4 = r4 - b4

    close = df["close"]
    ma20 = rolling_ma(close, 20)
    ma50 = rolling_ma(close, 50)
    volw = weekly_vol_from_daily(close)

    ma20_w = ma20.reindex(w.index, method="ffill")
    ma50_w = ma50.reindex(w.index, method="ffill")
    volw_w = volw.reindex(w.index, method="ffill")

    above_ma20 = (w > ma20_w).astype(int)
    ma_stack = (ma20_w > ma50_w).astype(int)
    ma_gap = (ma20_w / ma50_w) - 1.0

    month = pd.Series(w.index.month, index=w.index).astype(int)
    is_jan = (month == 1).astype(int)

    spy_above_ma20 = (b > b.rolling(20, min_periods=20).mean()).astype(int)
    spy_rs4 = b4

    # Label: next week outperformance vs SPY
    y = (rs1.shift(-1) > 0).astype(int)

    out = pd.DataFrame({
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
        "spy_rs_4w": spy_rs4,
        "spy_above_ma20": spy_above_ma20,
        "labor_exposure": float(labor_exposure),
        "jan_score": float(jan_score),
        "is_january": is_jan,
        "month": month,
        "y": y,
    }).dropna()

    # Drop last row (label uses next week)
    out = out.iloc[:-1].copy()
    return out


FEATURE_COLS = [
    "ret_1w", "rs_1w", "rs_4w", "mom_4w",
    "above_ma20", "ma_stack", "ma_gap",
    "vol_weekly",
    "spy_rs_4w", "spy_above_ma20",
    "labor_exposure",
    "jan_score", "is_january", "month",
]


def train_model(universe: pd.DataFrame, baseline: str) -> Tuple[Pipeline, dict]:
    spy = ensure_history(cache_price(baseline))
    b_weekly = friday_close_series(spy)

    all_rows = []
    for _, row in universe.iterrows():
        t = str(row["ticker"]).upper().strip()
        le = float(row["labor_exposure"]) if "labor_exposure" in universe.columns else 5.0
        js = float(row["jan_score"]) if "jan_score" in universe.columns else 0.0
        feat = build_weekly_features(t, b_weekly, le, js)
        if not feat.empty:
            all_rows.append(feat)

    if not all_rows:
        raise RuntimeError("No training data built (tickers missing data or not found).")

    data = pd.concat(all_rows, ignore_index=True)
    X = data[FEATURE_COLS].astype(float)
    y = data["y"].astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    accs, aucs = [], []
    for tr, te in tscv.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        yhat = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y.iloc[te], yhat))
        try:
            aucs.append(roc_auc_score(y.iloc[te], p))
        except Exception:
            pass

    metrics = {
        "rows": int(len(data)),
        "tickers_used": int(data["ticker"].nunique()),
        "cv_accuracy_avg": float(np.mean(accs)),
        "cv_auc_avg": float(np.mean(aucs)) if aucs else None,
    }

    # Fit final
    pipe.fit(X, y)

    dump(pipe, MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    return pipe, metrics


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None
    return load(MODEL_PATH)


def score_one_week(
    ticker: str,
    baseline: str,
    labor_exposure: float,
    jan_score: float,
    model: Pipeline | None,
) -> dict:
    spy = ensure_history(cache_price(baseline))
    b_weekly = friday_close_series(spy)

    df = ensure_history(cache_price(ticker))
    w = friday_close_series(df)
    common = w.index.intersection(b_weekly.index)
    w = w.loc[common]
    b = b_weekly.loc[common]

    if len(w) < 10:
        raise RuntimeError("Not enough weekly data.")

    # Build “current week” feature row (no lookahead)
    r1 = (w.iloc[-1] / w.iloc[-2]) - 1.0
    r4 = (w.iloc[-1] / w.iloc[-5]) - 1.0 if len(w) >= 5 else 0.0

    b1 = (b.iloc[-1] / b.iloc[-2]) - 1.0
    b4 = (b.iloc[-1] / b.iloc[-5]) - 1.0 if len(b) >= 5 else 0.0

    rs1 = r1 - b1
    rs4 = r4 - b4

    close = df["close"]
    ma20 = rolling_ma(close, 20)
    ma50 = rolling_ma(close, 50)
    volw = weekly_vol_from_daily(close)

    ma20_last = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else float("nan")
    ma50_last = float(ma50.iloc[-1]) if not np.isnan(ma50.iloc[-1]) else float("nan")
    vol_last = float(volw.iloc[-1]) if not np.isnan(volw.iloc[-1]) else 0.0

    above_ma20 = int(w.iloc[-1] > ma20_last) if not math.isnan(ma20_last) else 0
    ma_stack = int(ma20_last > ma50_last) if (not math.isnan(ma20_last) and not math.isnan(ma50_last)) else 0
    ma_gap = float((ma20_last / ma50_last) - 1.0) if (not math.isnan(ma20_last) and not math.isnan(ma50_last)) else 0.0

    spy_above_ma20 = int(b.iloc[-1] > b.rolling(20, min_periods=20).mean().iloc[-1]) if len(b) >= 20 else 0

    now = datetime.now()
    feat = pd.DataFrame([{
        "ret_1w": float(r1),
        "rs_1w": float(rs1),
        "rs_4w": float(rs4),
        "mom_4w": float(r4),
        "above_ma20": float(above_ma20),
        "ma_stack": float(ma_stack),
        "ma_gap": float(ma_gap),
        "vol_weekly": float(vol_last),
        "spy_rs_4w": float(b4),
        "spy_above_ma20": float(spy_above_ma20),
        "labor_exposure": float(labor_exposure),
        "jan_score": float(jan_score),
        "is_january": float(1 if now.month == 1 else 0),
        "month": float(now.month),
    }])[FEATURE_COLS].astype(float)

    ai_prob = None
    if model is not None:
        ai_prob = float(model.predict_proba(feat)[:, 1][0]) * 100.0

    return {
        "Ticker": ticker,
        "LastPrice": round(float(df["close"].iloc[-1]), 2),
        "RS_1w_vs_SPY_%": round(float(rs1) * 100.0, 2),
        "RS_4w_vs_SPY_%": round(float(rs4) * 100.0, 2),
        "WeeklyVol_%": round(float(vol_last) * 100.0, 2),
        "AI_Prob_Outperform_%": None if ai_prob is None else round(ai_prob, 1),
        "JanScore": int(jan_score),
    }


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Market AI (Option 2)", layout="wide")
st.title("🧠 Market AI (Option 2) — Next-week Outperformance Model")
st.caption("This uses end-of-day data and predicts probability of beating SPY next week. Not financial advice.")

baseline = st.sidebar.text_input("Baseline ticker", value="SPY").strip().upper()

st.header("1) Universe (the pool the AI learns from)")
st.write("Start with ~20–50 tickers. Expand later once it works.")

if os.path.exists("universe.csv"):
    universe = pd.read_csv("universe.csv")
else:
    universe = pd.DataFrame(columns=["ticker", "labor_exposure", "jan_score"])

universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
st.dataframe(universe, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Refresh cached prices"):
        st.cache_data.clear()
        st.success("Cache cleared.")

with col2:
    if st.button("🎓 Train AI model"):
        try:
            with st.spinner("Training (can take a few minutes depending on universe size)..."):
                model, metrics = train_model(universe, baseline)
            st.success("Model trained and saved.")
            st.json(metrics)
        except Exception as e:
            st.error(f"Training failed: {e}")

st.header("2) Watchlist scoring (AI picks best to watch)")
st.write("The model ranks tickers by probability of beating SPY next week.")

model = load_model()
if model is None:
    st.warning("No model found yet. Click **Train AI model** first.")
else:
    st.success("Model loaded.")

top_n = st.number_input("Show top N tickers", min_value=5, max_value=100, value=20, step=5)
max_vol = st.number_input("Max weekly volatility % (filter)", min_value=1.0, max_value=50.0, value=15.0, step=1.0)

if st.button("🔎 Score universe now"):
    rows = []
    errs = 0
    prog = st.progress(0)
    for i, r in universe.iterrows():
        t = str(r["ticker"]).upper().strip()
        le = float(r["labor_exposure"]) if "labor_exposure" in universe.columns else 5.0
        js = float(r["jan_score"]) if "jan_score" in universe.columns else 0.0
        try:
            row = score_one_week(t, baseline, le, js, model)
            if row["WeeklyVol_%"] <= float(max_vol):
                rows.append(row)
        except Exception:
            errs += 1
        prog.progress((i + 1) / max(1, len(universe)))

    out = pd.DataFrame(rows)
    if out.empty:
        st.warning("No results. Try fewer tickers or check data availability.")
    else:
        out = out.sort_values(by="AI_Prob_Outperform_%", ascending=False).head(int(top_n))
        st.subheader("Top AI picks to WATCH")
        st.dataframe(out, use_container_width=True)

        st.download_button(
            "⬇ Export CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="ai_watchlist.csv",
            mime="text/csv"
        )
        st.info(f"Done. Errors: {errs} (usually ticker missing data).")
