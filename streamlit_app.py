# app.py
from __future__ import annotations

import re
import csv
import io
import json
import math
import os
import time
import logging
import traceback
import threading
import random
import hashlib
import textwrap
from urllib.parse import quote_plus
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional, Tuple
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from joblib import dump, load
from asof import get_asof_date
from storage_sqlite import latest_price_date

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from streamlit_autorefresh import st_autorefresh
# Optional backfill modules (don’t crash app if missing)
try:
    from market_ai.backfill_ai import build_backfill_dataset, load_tickers_auto, train_meta_model
except Exception:
    build_backfill_dataset = None
    load_tickers_auto = None
    train_meta_model = None

try:
    from market_ai.backfill_meta_training import (
        build_relative_strength_backfill,
        extract_tickers_from_df,
        load_thesis_scores_from_scan_df,
    )
except Exception:
    build_relative_strength_backfill = None
    extract_tickers_from_df = None
    load_thesis_scores_from_scan_df = None

try:
    from backfill import CompanySpec, backfill_company
except Exception:
    CompanySpec = None
    backfill_company = None

# =========================
# Determinism + Safety utils
# =========================
SEED = 1337

def set_deterministic(seed: int = SEED) -> None:
    """Make results stable between restarts (as long as you don't retrain)."""
    random.seed(seed)
    np.random.seed(seed)

def ensure_ticker_col(df: pd.DataFrame, *, label: str = "DataFrame") -> pd.DataFrame:
    """
    Fix KeyError: 'ticker' by normalizing common column names to `ticker`.
    """
    if df is None or df.empty:
        raise ValueError(f"{label} is empty")

    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    candidates = ["ticker", "symbol", "tick", "code"]
    found = None
    for cand in candidates:
        if cand in lower_map:
            found = lower_map[cand]
            break

    if found is None:
        raise ValueError(
            f"{label} has no ticker column. Columns={cols}. "
            f"Expected one of {candidates}."
        )

    if found != "ticker":
        df = df.rename(columns={found: "ticker"})

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"].str.len() > 0].copy()
    return df

def normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any fetched price df into columns:
    date (datetime64[ns]), close (float), optional volume (float)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])

    out = df.copy()

    # common column names
    col_map = {c.lower(): c for c in out.columns.astype(str)}

    # date
    if "date" in col_map:
        date_col = col_map["date"]
    elif "datetime" in col_map:
        date_col = col_map["datetime"]
    elif "time" in col_map:
        date_col = col_map["time"]
    else:
        # if index is date-like
        out = out.reset_index()
        col_map = {c.lower(): c for c in out.columns.astype(str)}
        date_col = col_map.get("date", out.columns[0])

    out["date"] = pd.to_datetime(out[date_col], errors="coerce")

    # close
    if "close" in col_map:
        close_col = col_map["close"]
    elif "adj close" in col_map:
        close_col = col_map["adj close"]
    elif "adjclose" in col_map:
        close_col = col_map["adjclose"]
    else:
        # last resort: find a column named like "c" or something
        close_col = None
        for k in ["c", "price", "last"]:
            if k in col_map:
                close_col = col_map[k]
                break
        if close_col is None:
            raise ValueError(f"Price DF missing close column. Columns={list(df.columns)}")

    out["close"] = pd.to_numeric(out[close_col], errors="coerce")

    # volume optional
    if "volume" in col_map:
        out["volume"] = pd.to_numeric(out[col_map["volume"]], errors="coerce")
    else:
        out["volume"] = np.nan

    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out[["date", "close", "volume"]].reset_index(drop=True)
    return out

# =========================
# Intraday horizon settings
# =========================
DEFAULT_HORIZON_UNIT = "hours"   # "minutes" or "hours"
DEFAULT_HORIZON_VALUE = 4        # e.g. 30 minutes or 2 hours etc.
DEFAULT_BAR_INTERVAL = "1min"    # keep 1min for outcome scoring
RANDOM_SEED = 42

TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "")  # or put in .env
try:
    # streamlit secrets optional
    if not TWELVEDATA_KEY and hasattr(st, "secrets"):
        TWELVEDATA_KEY = st.secrets.get("TWELVEDATA_KEY", "")
except Exception:
    pass


# ==========================================================
# Paths / files
# ==========================================================
DATA_DIR = "data_prices"
INTRADAY_DIR = "data_intraday"
LOG_DIR = "logs"
NEWS_DIR = "data_news"
NEWS_LABELS_PATH = os.path.join(LOG_DIR, "news_labels.csv")
NEWS_MODEL_PATH = os.path.join(LOG_DIR, "news_model.joblib")
NEWS_MODEL_METRICS = os.path.join(LOG_DIR, "news_model_metrics.json")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MarketApp/1.0"

UNIVERSE_CSV = "universe.csv"
WATCHLIST_CSV = "watchlist.csv"
LISTINGS_CSV = "listings.csv"  # optional

MODEL_PATH = "model.joblib"
FEATURES_PATH = "feature_cols.json"

# =========================
# Model persistence (stable predictions)
# =========================
MODEL_META_PATH = "model_meta.json"

def _fingerprint_features(cols: list[str], horizon: int) -> str:
    raw = json.dumps({"cols": cols, "h": horizon}, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]

def load_saved_model() -> tuple[object | None, dict | None]:
    if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_META_PATH):
        try:
            model = load(MODEL_PATH)
            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return model, meta
        except Exception:
            return None, None
    return None, None

def save_model_with_meta(model: object, meta: dict) -> None:
    dump(model, MODEL_PATH)
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

# ==========================================================
# Ticker/company-name + news matching helpers
# ==========================================================

AMBIGUOUS_TICKERS = {"NOW", "ON", "IT", "AI", "CAR", "RUN", "SAVE", "LOVE", "LIFE", "PLAY"}

def _clean_str(x: str) -> str:
    return (x or "").strip()

def _norm_text(x: str) -> str:
    x = (x or "").lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()

# Compatibility alias: if _clean_str is ever missing, fall back to _norm_text
if "_clean_str" not in globals():
    _clean_str = _norm_text

TICKER_COL_CANDIDATES = [
    "ticker",
    "Ticker",
    "symbol",
    "Symbol",
    "SYMBOL",
    "ric",
    "RIC",
    "code",
    "Code",
    "asset",
    "Asset",
    "instrument",
    "Instrument",
    "tickers",
    "Tickers",
    "symbols",
    "Symbols",
]


def get_ticker_series(df: pd.DataFrame) -> pd.Series:
    """
    Robustly return the ticker column/series from a DataFrame, handling many common column names
    or ticker-in-index layouts.
    """
    for c in TICKER_COL_CANDIDATES:
        if c in df.columns:
            return df[c]

    def _norm_col(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    for c in df.columns:
        if _norm_col(c) in {"ticker", "tickers", "symbol", "symbols", "ric", "code", "asset", "instrument"}:
            return df[c]

    if df.index.name and _norm_col(df.index.name) in {"ticker", "tickers", "symbol", "symbols", "ric", "code"}:
        return df.index.to_series()

    raise KeyError(f"No ticker/symbol column found. Columns: {list(df.columns)} (index name={df.index.name})")

UNIVERSE_COLS = ["ticker", "name", "labor_exposure", "jan_score", "company_name", "aliases", "stooq_symbol"]

@st.cache_data(ttl=24 * 3600)
def load_universe_df(universe_csv: str) -> pd.DataFrame:
    if not os.path.exists(universe_csv):
        df = pd.DataFrame(columns=UNIVERSE_COLS)
        df.to_csv(universe_csv, index=False)

    df = pd.read_csv(universe_csv, dtype=str).fillna("")
    for c in UNIVERSE_COLS:
        if c not in df.columns:
            df[c] = ""

    df["ticker"] = df["ticker"].map(_clean_str).str.upper()
    df["name"] = df["name"].map(_clean_str)
    df["company_name"] = df["company_name"].map(_clean_str)
    df["aliases"] = df["aliases"].map(_clean_str)
    df["stooq_symbol"] = df["stooq_symbol"].map(_clean_str)

    # numeric fields stored as strings; normalize later in load_universe()
    df["labor_exposure"] = df["labor_exposure"].map(_clean_str)
    df["jan_score"] = df["jan_score"].map(_clean_str)

    df = df[df["ticker"] != ""].drop_duplicates("ticker").reset_index(drop=True)
    return df[UNIVERSE_COLS]

def save_universe_df(df: pd.DataFrame, universe_csv: str) -> None:
    df2 = df.copy()
    for c in UNIVERSE_COLS:
        if c not in df2.columns:
            df2[c] = ""
    df2["ticker"] = df2["ticker"].map(_clean_str).str.upper()
    df2["name"] = df2["name"].map(_clean_str)
    df2 = df2[df2["ticker"] != ""].drop_duplicates("ticker")
    df2.to_csv(universe_csv, index=False)

@st.cache_data(ttl=24 * 3600)
def fetch_sec_ticker_name_map() -> dict:
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        # IMPORTANT: change this to something real. SEC can block "generic" user agents.
        "User-Agent": "MarketApp (matty@example.com)"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        out = {}
        for _, item in data.items():
            t = str(item.get("ticker", "")).upper().strip()
            nm = str(item.get("title", "")).strip()
            if t and nm:
                out[t] = nm
        return out
    except Exception:
        return {}

def autofill_missing_company_names(universe_df: pd.DataFrame) -> pd.DataFrame:
    sec_map = fetch_sec_ticker_name_map()
    if not sec_map:
        return universe_df

    df = universe_df.copy()
    missing = df["name"].fillna("").str.strip() == ""
    df.loc[missing, "name"] = df.loc[missing, "ticker"].map(lambda t: sec_map.get(t, ""))
    return df

def ticker_display(t: str, name_map: dict) -> str:
    nm = (name_map.get(t) or "").strip()
    return f"{t} — {nm}" if nm else t

def build_company_aliases(ticker: str, name: str) -> list[str]:
    aliases = []
    if name:
        aliases.append(name)
        aliases.append(re.sub(r",?\s+inc\.?$", "", name, flags=re.I).strip())
        aliases.append(re.sub(r",?\s+corporation$", "", name, flags=re.I).strip())
        aliases.append(re.sub(r",?\s+plc$", "", name, flags=re.I).strip())
    aliases.append(f"NYSE:{ticker}")
    aliases.append(f"NASDAQ:{ticker}")
    aliases.append(f"${ticker}")

    aliases = [a.strip() for a in aliases if a and a.strip()]
    seen = set()
    out = []
    for a in aliases:
        k = a.lower()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out

def score_article_to_ticker(article_text: str, ticker: str, company_name: str) -> tuple[int, list[str]]:
    text = _norm_text(article_text)
    t = ticker.lower()
    reasons = []
    score = 0

    if re.search(rf"(\$|nyse:|nasdaq:|amex:)\s*{re.escape(t)}\b", text):
        score += 8
        reasons.append("strong:ticker_symbol")

    if re.search(rf"\b{re.escape(t)}\b", text):
        score += 2
        reasons.append("weak:ticker_word")

    aliases = build_company_aliases(ticker, company_name)
    for a in aliases:
        if a and _norm_text(a) in text:
            score += 6
            reasons.append(f"strong:alias:{a}")
            break

    if ticker in AMBIGUOUS_TICKERS:
        has_name_hit = any(r.startswith("strong:alias:") for r in reasons)
        has_strong_symbol = "strong:ticker_symbol" in reasons
        if not (has_name_hit or has_strong_symbol):
            score -= 6
            reasons.append("penalty:ambiguous_without_name")

    return score, reasons

def assign_articles_to_tickers(articles: list[dict], universe_df: pd.DataFrame) -> list[dict]:
    name_map = dict(zip(universe_df["ticker"], universe_df["name"]))
    tickers = universe_df["ticker"].tolist()

    out = []
    for a in articles:
        text = " ".join([str(a.get("title", "")), str(a.get("snippet", ""))])
        debug = {}
        scored = []
        for t in tickers:
            s, reasons = score_article_to_ticker(text, t, name_map.get(t, ""))
            debug[t] = {"score": s, "reasons": reasons}
            scored.append((t, s))

        matched = [t for (t, s) in scored if s >= 8]

        if not matched and scored:
            best_t, best_s = max(scored, key=lambda x: x[1])
            if best_s >= 6:
                matched = [best_t]

        a2 = dict(a)
        a2["matched_tickers"] = matched
        a2["match_debug"] = debug
        out.append(a2)

    return out

@st.cache_data(ttl=15 * 60)
def fetch_gdelt_news_for_query(query: str, maxrecords: int = 50) -> list[dict]:
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = f"?query={quote_plus(query)}&mode=ArtList&format=json&maxrecords={maxrecords}&sort=HybridRel"
    url = base + params

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        js = r.json()
        arts = []
        for it in js.get("articles", []) or []:
            arts.append({
                "title": it.get("title", ""),
                "url": it.get("url", ""),
                "source": it.get("sourceCountry", "") or it.get("source", ""),
                "published": it.get("seendate", ""),
                "snippet": it.get("snippet", ""),
            })
        return arts
    except Exception:
        return []


SCAN_LOG = os.path.join(LOG_DIR, "scan_history.csv")
WATCH_LOG = os.path.join(LOG_DIR, "watchlist_history.csv")
PRED_LOG = os.path.join(LOG_DIR, "predictions_log.csv")
BACKFILLED_PRED_LOG = os.path.join(LOG_DIR, "predictions_log_backfilled.csv")
BACKFILL_TRAINING_CSV = os.path.join(LOG_DIR, "backfill_training.csv")
TRAIN_METRICS = os.path.join(LOG_DIR, "train_metrics.json")
NEWS_LOG = os.path.join(LOG_DIR, "news_features.csv")

META_MODEL_PATH = "meta_model.joblib"
META_TRAIN_METRICS = os.path.join(LOG_DIR, "meta_train_metrics.json")

AUTO_STATE_PATH = os.path.join(LOG_DIR, "auto_state.json")

DEBUG_LOG_PATH = os.path.join(LOG_DIR, "app_debug.log")
ERRORS_CSV = os.path.join(LOG_DIR, "errors_log.csv")

INTRADAY_MODEL_PATH = "intraday_model.joblib"
INTRADAY_TRAIN_METRICS = os.path.join(LOG_DIR, "intraday_train_metrics.json")
INTRADAY_PRED_LOG = os.path.join(LOG_DIR, "intraday_predictions_log.csv")
STOOQ_CACHE_DIR = os.path.join(LOG_DIR, "stooq_cache")


# ==========================================================
# Intraday features
# ==========================================================
INTRA_FEATURE_COLS = [
    "ret_1", "ret_3", "ret_12",
    "rs_12",
    "ema20_gap",
    "rsi14",
    "vol_20",
    "vol_ratio_20_100",
]

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_intraday_dataset(sym_df: pd.DataFrame, base_df: pd.DataFrame, horizon_bars: int = 12) -> pd.DataFrame:
    if sym_df.empty or base_df.empty:
        return pd.DataFrame()
    if "close" not in sym_df.columns or "close" not in base_df.columns:
        return pd.DataFrame()

    s = sym_df[["close"]].rename(columns={"close": "s_close"}).copy()
    b = base_df[["close"]].rename(columns={"close": "b_close"}).copy()

    df = s.join(b, how="inner")
    if len(df) < 250:
        return pd.DataFrame()

    df["ret_1"] = df["s_close"].pct_change(1)
    df["ret_3"] = df["s_close"].pct_change(3)
    df["ret_12"] = df["s_close"].pct_change(12)

    s_ema20 = _ema(df["s_close"], 20)
    df["ema20_gap"] = (df["s_close"] / s_ema20) - 1.0

    df["rsi14"] = _rsi(df["s_close"], 14)

    df["vol_20"] = df["ret_1"].rolling(20, min_periods=20).std()

    if "volume" in sym_df.columns:
        v = sym_df["volume"].reindex(df.index).astype(float)
        v20 = v.rolling(20, min_periods=20).mean()
        v100 = v.rolling(100, min_periods=100).mean()
        df["vol_ratio_20_100"] = (v20 / v100.replace(0, np.nan))
    else:
        df["vol_ratio_20_100"] = 0.0

    s_h = (df["s_close"].shift(-horizon_bars) / df["s_close"]) - 1.0
    b_h = (df["b_close"].shift(-horizon_bars) / df["b_close"]) - 1.0
    rs_h = s_h - b_h
    df["rs_12"] = (df["ret_12"].fillna(0) - df["b_close"].pct_change(12).fillna(0))

    df["y"] = (rs_h > 0.0002).astype(int)

    out = df[INTRA_FEATURE_COLS + ["y"]].dropna().copy()
    if len(out) > horizon_bars:
        out = out.iloc[:-horizon_bars].copy()
    return out

def train_intraday_model_from_watchlist(
    baseline: str,
    interval: str,
    horizon_bars: int,
    td_key: str,
    min_rows: int = 2000,
) -> dict:
    wl = load_watchlist()
    if wl.empty:
        raise RuntimeError("Watchlist is empty. Add tickers to watchlist first.")
    if not td_key:
        raise RuntimeError("No Twelve Data API key set (sidebar).")

    tickers = wl["ticker"].astype(str).str.upper().tolist()
    base_df = cache_intraday(baseline, interval=interval, apikey=td_key, outputsize=5000)

    rows = []
    used = 0
    for t in tickers[:20]:
        sym_df = cache_intraday(t, interval=interval, apikey=td_key, outputsize=5000)
        ds = build_intraday_dataset(sym_df, base_df, horizon_bars=horizon_bars)
        if not ds.empty:
            rows.append(ds)
            used += 1

    if not rows:
        raise RuntimeError("No intraday training data built (rate limits / not enough bars).")

    data = pd.concat(rows, ignore_index=True)
    if len(data) < min_rows:
        raise RuntimeError(f"Not enough intraday rows yet ({len(data)}/{min_rows}). Let it collect more bars.")

    X = data[INTRA_FEATURE_COLS].astype(float)
    y = data["y"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(loss="log_loss", max_iter=4000, tol=1e-3, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    model.fit(X, y)
    dump(model, INTRADAY_MODEL_PATH)

    metrics = {"rows": int(len(data)), "tickers_used": int(used), "trained_at": now_stamp(), "interval": interval, "horizon_bars": int(horizon_bars)}
    with open(INTRADAY_TRAIN_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def load_intraday_model():
    if not os.path.exists(INTRADAY_MODEL_PATH):
        return None
    try:
        return load(INTRADAY_MODEL_PATH)
    except Exception:
        return None

def intraday_ai_prob_fast(symbol: str, base_df: pd.DataFrame, interval: str, horizon_bars: int, td_key: str):
    model = load_intraday_model()
    if model is None or base_df is None or base_df.empty:
        return None

    s_df = cache_intraday(symbol, interval=interval, apikey=td_key, outputsize=800).tail(600)
    ds = build_intraday_dataset(s_df, base_df, horizon_bars=horizon_bars)
    if ds.empty:
        return None

    Xlast = ds[INTRA_FEATURE_COLS].tail(1).astype(float)
    try:
        p = float(model.predict_proba(Xlast)[:, 1][0]) * 100.0
        return round(p, 1)
    except Exception:
        return None


# ==========================================================
# Meta model: P(your call is right | features)
# ==========================================================
META_FEATURES = [
    "ai_prob_frac",
    "thesis_score",
    "rs1",
    "rs4",
    "vol_weekly",
    "labor_exposure",
    "spy_above_ma200",
    "spy_vol_4w",
    "news_sent",
    "news_intensity",
]

auto_refresh = False

MIN_DAILY_ROWS_SCORE = 120
MIN_DAILY_ROWS_TRAIN_BASELINE = 260
MIN_WEEKLY_POINTS_SCORE = 15

AUTO_OUTCOME_CHECK_HOURS = 12
AUTO_META_TRAIN_HOURS = 24 * 7
AUTO_BASE_TRAIN_HOURS = 24 * 30


# ==========================================================
# HTTP session (thread-local) + retries
# ==========================================================
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_thread_local = threading.local()

def _get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(DEFAULT_HEADERS)
        _thread_local.session = s
    return s

def http_get(url: str, *, params: dict | None = None, timeout: int = 20, retries: int = 3) -> Optional[requests.Response]:
    backoff = 1.2
    for i in range(max(1, retries)):
        try:
            r = _get_session().get(url, params=params, timeout=timeout)
            dbg_event("http_get", url=url, status=r.status_code, attempt=i + 1)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (i + 1))
                continue
            if r.status_code == 403 and i < retries - 1:
                time.sleep(backoff * (i + 1))
                continue
            return r
        except Exception as e:
            dbg_event("http_get_exception", url=url, attempt=i + 1, err=str(e))
            try:
                logger.exception(f"http_get failed: {url} params={params}")
            except Exception:
                pass
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
                continue
            return None
    return None


# ==========================================================
# Seed tickers (auto-grow)
# ==========================================================
AUTO_SEED_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK.B","JPM","V",
    "MA","UNH","XOM","AVGO","LLY","PG","COST","HD","MRK","PEP",
    "KO","ABBV","WMT","BAC","ORCL","CSCO","ADBE","CRM","NFLX","TMO",
    "DG","DLTR","TGT","TJX","ROST","LOW","UPS","FDX","NKE","MCD",
    "SBUX","DIS","BKNG","ABNB","EBAY","ETSY","ULTA","LULU","BBY","KR",
    "CAT","DE","BA","GE","HON","RTX","LMT","MMM","UNP","CSX",
    "AMD","INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","ASML",
    "GS","MS","C","WFC","AXP","SCHW","BLK",
    "CVX","COP","SLB","EOG","OXY","LIN","NEM",
    "PFE","JNJ","MDT","ISRG","GILD","AMGN",
    "LEN","DHI","PHM","NVR","TOL",
    "SPY","QQQ","DIA","IWM","XLF","XLE","XLK","XLY","XLP","XLV"
]


# ==========================================================
# Ollama (optional local explainer)
# ==========================================================
def _ollama_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path

def ollama_version(base_url: str, timeout: int = 3) -> Optional[str]:
    try:
        r = requests.get(_ollama_url(base_url, "/api/version"), timeout=timeout)
        if r.status_code != 200:
            return None
        js = r.json()
        return str(js.get("version") or "")
    except Exception:
        return None

def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    timeout: int = 120,
) -> Optional[str]:
    try:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": float(temperature)},
        }
        r = requests.post(_ollama_url(base_url, "/api/chat"), json=payload, timeout=timeout)
        if r.status_code != 200:
            return None
        js = r.json()
        msg = js.get("message", {}) if isinstance(js, dict) else {}
        return msg.get("content")
    except Exception:
        return None

def build_ollama_explain_prompt(df: pd.DataFrame, baseline: str, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "No rows to explain."
    take = df.head(int(max_rows)).copy()
    cols = [
        "Ticker", "Action", "Confidence",
        "AI_Prob_Outperform_%", "AI_Prob_Calibrated_%", "ThesisLiteScore",
        "RS_1w_vs_SPY_%", "RS_4w_vs_SPY_%", "WeeklyVol_%",
        "NewsHeadlines_7d", "NewsSent_7d", "NewsIntensity_7d", "NewsLaborHits_7d",
        "LaborExposure", "LastPrice",
    ]
    cols = [c for c in cols if c in take.columns]
    take = take[cols]
    table = take.to_csv(index=False)
    return f"""
Baseline: {baseline}

You are explaining a market-dashboard output. Do NOT give financial advice.
Explain in plain English what the dashboard signals mean and why each ticker got its Action.

Rules:
- Keep it short and structured.
- For each ticker: 3 bullets = (1) main positives (2) main risks/red flags (3) what would invalidate / what to watch next week
- If AI_Prob_Calibrated_% exists, treat it as the "self-learned" calibrated probability.
- Mention news only if intensity/headlines are non-zero.
- No price targets. No "buy this". Just interpretation.

DATA (CSV):
{table}
""".strip()


# ==========================================================
# Helpers
# ==========================================================
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INTRADAY_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(NEWS_DIR, exist_ok=True)
    
def utcnow_floor():
    return datetime.utcnow().replace(microsecond=0)

def parse_dt_any(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        return dt.replace(tzinfo=None)
    except Exception:
        return None

def compute_target_ts(asof_ts: datetime, horizon_value: int, horizon_unit: str) -> datetime:
    unit = (horizon_unit or "").lower().strip()
    v = int(horizon_value)
    if unit.startswith("min"):
        return asof_ts + timedelta(minutes=v)
    if unit.startswith("hour"):
        return asof_ts + timedelta(hours=v)
    # safe fallback
    return asof_ts + timedelta(hours=v)

def init_horizon_state():
    if "horizon_unit" not in st.session_state:
        st.session_state["horizon_unit"] = DEFAULT_HORIZON_UNIT
    if "horizon_value" not in st.session_state:
        st.session_state["horizon_value"] = int(DEFAULT_HORIZON_VALUE)

def horizon_controls_ui():
    init_horizon_state()
    with st.expander("⏱ Prediction horizon (controls when outcomes become 'known')", expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.session_state["horizon_unit"] = st.selectbox(
                "Unit",
                ["minutes", "hours"],
                index=0 if st.session_state["horizon_unit"] == "minutes" else 1,
                key="horizon_unit_ui",
            )
        with c2:
            st.session_state["horizon_value"] = int(st.number_input(
                "Value",
                min_value=1,
                max_value=720 if st.session_state["horizon_unit"] == "minutes" else 72,
                value=int(st.session_state["horizon_value"]),
                step=1,
                key="horizon_value_ui",
            ))
        st.caption("Example: 30 minutes = fast labels. 4 hours = fewer labels but smoother noise.")

@st.cache_data(ttl=60, show_spinner=False)
def twelvedata_time_series(symbol: str, interval: str, outputsize: int, apikey: str) -> pd.DataFrame:
    if not apikey:
        raise ValueError("Missing TWELVEDATA_KEY (set env var TWELVEDATA_KEY or Streamlit secrets).")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": int(outputsize),
        "apikey": apikey,
        "format": "JSON",
        "order": "ASC",  # newer API supports it; harmless if ignored
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and data.get("status") == "error":
        raise ValueError(f"TwelveData error for {symbol}: {data.get('message', data)}")

    values = (data or {}).get("values", []) if isinstance(data, dict) else []
    if not values:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime")
    return df

def price_nearest(df: pd.DataFrame, target_ts: datetime, tolerance_minutes: int = 20) -> float:
    if df is None or df.empty:
        return float("nan")
    t = pd.to_datetime(target_ts)
    d = df.copy()
    d["__diff"] = (d["datetime"] - t).abs()
    row = d.sort_values("__diff").iloc[0]
    if pd.isna(row["__diff"]):
        return float("nan")
    if row["__diff"] > pd.Timedelta(minutes=int(tolerance_minutes)):
        return float("nan")
    return float(row["close"])

def ensure_pred_log_columns(pred: pd.DataFrame) -> pd.DataFrame:
    # Backward compatible: prefer *_ts, fall back to *_date
    if "asof_ts" not in pred.columns:
        pred["asof_ts"] = pred.get("asof_date", None)
    if "target_ts" not in pred.columns:
        pred["target_ts"] = pred.get("target_date", None)

    if "horizon_unit" not in pred.columns:
        pred["horizon_unit"] = ""
    if "horizon_value" not in pred.columns:
        pred["horizon_value"] = ""

    # normalize
    pred["asof_ts"] = pd.to_datetime(pred["asof_ts"], errors="coerce")
    pred["target_ts"] = pd.to_datetime(pred["target_ts"], errors="coerce")
    return pred


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def get_file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def sign(x: float) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def safe_read_csv(path: str, cols: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def safe_read_log_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_uploaded_csv(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(upload)
    except Exception:
        try:
            return pd.read_csv(upload, engine="python")
        except Exception:
            return pd.DataFrame()

def append_log(path: str, df: pd.DataFrame, fixed_cols: Optional[list[str]] = None):
    ensure_dirs()
    if df is None or df.empty:
        return
    df2 = df.copy()
    df2.insert(0, "timestamp", now_stamp())
    if fixed_cols is not None:
        for c in fixed_cols:
            if c not in df2.columns:
                df2[c] = np.nan
        df2 = df2[fixed_cols]
    write_header = not os.path.exists(path)
    df2.to_csv(path, mode="a", header=write_header, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")

def show_df(df: pd.DataFrame, *, key: str):
    colconf = {
        "AI_Prob_Outperform_%": st.column_config.NumberColumn(
            "AI_Prob_Outperform_%",
            help="Model probability of beating the baseline next week (not a guarantee).",
            format="%.1f",
        ),
        "AI_RT_%": st.column_config.NumberColumn(
            "AI_RT_%",
            help="Intraday model probability of beating the baseline over the next horizon bars (not a guarantee).",
            format="%.1f",
        ),
        "AI_Prob_Calibrated_%": st.column_config.NumberColumn(
            "AI_Prob_Calibrated_%",
            help="Self-learning calibrated probability (trained on your tracked outcomes).",
            format="%.1f",
        ),
        "ThesisLiteScore": st.column_config.NumberColumn(
            "ThesisLiteScore",
            help="0–10 thesis score = RS + trend − labor − excess vol (+ news + regime overlays).",
            format="%.2f",
        ),
        "RS_4w_vs_SPY_%": st.column_config.NumberColumn("RS_4w_vs_SPY_%", format="%.2f"),
        "RS_1w_vs_SPY_%": st.column_config.NumberColumn("RS_1w_vs_SPY_%", format="%.2f"),
        "WeeklyVol_%": st.column_config.NumberColumn("WeeklyVol_%", format="%.2f"),
        "AvgVol_20d": st.column_config.NumberColumn("AvgVol_20d", format="%d"),
        "NewsSent_7d": st.column_config.NumberColumn("NewsSent_7d", format="%.3f"),
        "NewsIntensity_7d": st.column_config.NumberColumn("NewsIntensity_7d", format="%.3f"),
        "Score": st.column_config.NumberColumn("Score", format="%.1f"),
        "Why": st.column_config.TextColumn("Why", width="large"),
    }
    colconf = {k: v for k, v in colconf.items() if k in df.columns}
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, column_config=colconf, key=key)
    except Exception:
        st.dataframe(df, use_container_width=True)

def _load_json(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) if f else {}
    except Exception:
        return {}

def _save_json(path: str, obj: dict) -> None:
    try:
        ensure_dirs()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except Exception:
        pass

def sync_watchlist_from_scan(scan_df: pd.DataFrame, *, mode: str = "merge", max_rows: int = 20) -> pd.DataFrame:
    wl_existing = load_watchlist()
    if scan_df is None or scan_df.empty:
        return wl_existing

    take = scan_df.head(int(max_rows)).copy()
    wl_new = take[["Ticker", "LaborExposure", "JanScore"]].copy()
    wl_new = wl_new.rename(columns={"Ticker": "ticker", "LaborExposure": "labor_exposure", "JanScore": "jan_score"})
    wl_new["catalyst"] = 0
    wl_new["labor_shock"] = 0
    wl_new["entry_price"] = np.nan
    wl_new["volatile"] = 0

    if "AI_Filter_%" in take.columns:
        wl_new["rank_ai"] = pd.to_numeric(take["AI_Filter_%"], errors="coerce").values
    elif "AI_Prob_Outperform_%" in take.columns:
        wl_new["rank_ai"] = pd.to_numeric(take["AI_Prob_Outperform_%"], errors="coerce").values

    if mode == "replace":
        out = wl_new.copy()
    else:
        out = pd.concat([wl_existing, wl_new], ignore_index=True)
        out = out.drop_duplicates(subset=["ticker"], keep="first")

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out = out[out["ticker"].str.len() > 0].sort_values("ticker")
    return out

def emit_rt_alerts(live_df: pd.DataFrame, *, threshold: float, cooldown_sec: int = 600):
    if live_df is None or live_df.empty or "AI_RT_%" not in live_df.columns:
        return

    key = "rt_alert_state"
    state = st.session_state.get(key, {})
    if not isinstance(state, dict):
        state = {}

    now = time.time()

    for _, r in live_df.iterrows():
        t = str(r.get("Ticker", "")).upper().strip()
        if not t:
            continue

        p = r.get("AI_RT_%", None)
        try:
            p = float(p)
        except Exception:
            continue

        if p >= float(threshold):
            last = float(state.get(t, 0.0))
            if (now - last) >= float(cooldown_sec):
                msg = f"⚡ {t}: AI_RT={p:.1f}% (≥ {threshold:.0f}%)"
                try:
                    st.toast(msg, icon="⚡")
                except Exception:
                    st.info(msg)
                state[t] = now

    st.session_state[key] = state


# ==========================================================
# Prediction target tuning (stronger labels)
# ==========================================================
OUTPERFORM_MIN = 0.002
OUTPERFORM_VOL_MULT = 0.25

def outperform_threshold(spy_vol_4w: float | None) -> float:
    try:
        v = float(spy_vol_4w)
        if np.isnan(v):
            v = 0.0
    except Exception:
        v = 0.0
    return max(float(OUTPERFORM_MIN), float(OUTPERFORM_VOL_MULT) * float(v))

def sample_weights_from_dates(dates: pd.Series, halflife_weeks: int = 78) -> np.ndarray:
    d = pd.to_datetime(dates, errors="coerce")
    d = d.dropna()
    if d.empty:
        return np.ones(len(dates), dtype=float)

    tmax = d.max()
    age_weeks = (tmax - pd.to_datetime(dates, errors="coerce")).dt.days / 7.0
    age_weeks = age_weeks.fillna(age_weeks.max())
    lam = math.log(2.0) / max(1.0, float(halflife_weeks))
    w = np.exp(-lam * age_weeks.astype(float))
    w = (w / np.nanmean(w)).clip(0.2, 5.0)
    return w.values.astype(float)

def _make_calibrator(estimator, cv):
    try:
        return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="sigmoid", cv=cv)

def get_model_feature_cols() -> list[str]:
    cols = _load_json(FEATURES_PATH)
    if isinstance(cols, list) and cols:
        return [str(c) for c in cols]
    return FEATURE_COLS


# ==========================================================
# Debug + logging helpers
# ==========================================================
def _setup_logger(debug: bool) -> logging.Logger:
    ensure_dirs()
    lg = logging.getLogger("market_ai")
    if getattr(lg, "_configured", False):
        lg.setLevel(logging.DEBUG if debug else logging.INFO)
        return lg

    lg.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(DEBUG_LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    lg.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    lg.addHandler(sh)

    lg._configured = True
    return lg

def _append_error_csv(where: str, exc: Exception):
    ensure_dirs()
    row = {
        "timestamp": now_stamp(),
        "where": where,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    df = pd.DataFrame([row])
    write_header = not os.path.exists(ERRORS_CSV)
    df.to_csv(ERRORS_CSV, mode="a", header=write_header, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")

def log_exception(lg: logging.Logger, where: str, exc: Exception, *, show_ui: bool, debug: bool):
    try:
        lg.exception(f"[{where}] {exc}")
    except Exception:
        pass
    try:
        _append_error_csv(where, exc)
    except Exception:
        pass
    if show_ui:
        st.error(f"❌ {where}: {exc}")
        if debug:
            st.exception(exc)

def dbg_event(msg: str, **kv):
    try:
        buff = st.session_state.get("_debug_events", [])
        if not isinstance(buff, list):
            buff = []
        item = {"t": now_stamp(), "msg": msg}
        if kv:
            item.update(kv)
        buff.append(item)
        st.session_state["_debug_events"] = buff[-250:]
    except Exception:
        pass

@contextmanager
def timed(label: str):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        dbg_event("timing", label=label, seconds=round(dt, 3))

def tail_text_file(path: str, n_lines: int = 200) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-int(n_lines):])
    except Exception:
        return ""


# ==========================================================
# Hands-off automation (on reruns)
# ==========================================================
def _get_auto_state() -> dict:
    stt = _load_json(AUTO_STATE_PATH)
    return stt if isinstance(stt, dict) else {}

def _last_run_dt(state: dict, key: str) -> Optional[datetime]:
    try:
        s = state.get(key)
        if not s:
            return None
        return datetime.fromisoformat(str(s))
    except Exception:
        return None

def _set_last_run(state: dict, key: str, dt: datetime):
    state[key] = dt.isoformat()
    _save_json(AUTO_STATE_PATH, state)

def _task_due(state: dict, key: str, every_hours: int) -> bool:
    last = _last_run_dt(state, key)
    if last is None:
        return True
    return (datetime.now() - last) >= timedelta(hours=int(every_hours))


# ==========================================================
# Split adjustment (heuristic)
# ==========================================================
SPLIT_FACTORS = [2, 3, 4, 5, 10]
SPLIT_TOL = 0.035

def _median_ratio(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    ratios = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        ratios.append(s / s.shift(1))
    return pd.concat(ratios, axis=1).median(axis=1)

def maybe_adjust_for_splits(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns:
        return df
    d = df.copy().sort_index()
    ratio = _median_ratio(d)
    big_move = (ratio <= 0.70) | (ratio >= 1.40)
    ratio = ratio.where(big_move)
    targets: list[float] = []
    for k in SPLIT_FACTORS:
        targets.append(float(k))
        targets.append(float(1.0 / k))
    snapped = pd.Series(np.nan, index=d.index, dtype=float)
    for i in range(len(d)):
        r = ratio.iloc[i]
        if pd.isna(r) or r <= 0:
            continue
        best = min(targets, key=lambda t: abs(r - t) / t)
        if abs(r - best) / best <= SPLIT_TOL:
            snapped.iloc[i] = best
    if snapped.notna().sum() == 0:
        return d
    adj = pd.Series(1.0, index=d.index, dtype=float)
    for i in range(len(d) - 1, 0, -1):
        m = snapped.iloc[i]
        adj.iloc[i - 1] = adj.iloc[i] * (float(m) if not pd.isna(m) else 1.0)
    for col in ["open", "high", "low", "close"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce") * adj
    if "volume" in d.columns:
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce") / adj
    return d


# ==========================================================
# Data fetch (Stooq daily)
# ==========================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    def _try(sym: str) -> pd.DataFrame:
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        try:
            r = http_get(url, timeout=20, retries=3)
            if r is None or r.status_code != 200:
                dbg_event("stooq_fetch_failed", symbol=sym, status=r.status_code if r else None)
                return pd.DataFrame()
            text = (r.text or "").strip()
            if not text:
                dbg_event("stooq_empty_response", symbol=sym)
                return pd.DataFrame()

            # Check if response looks like CSV
            if "Date" not in text.splitlines()[0]:
                dbg_event("stooq_invalid_format", symbol=sym, first_line=text.splitlines()[0][:100] if text else "")
                return pd.DataFrame()

            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            dbg_event("stooq_exception", symbol=sym, error=str(e))
            return pd.DataFrame()

        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        if df.empty:
            return pd.DataFrame()

        df = df.set_index("Date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            return pd.DataFrame()
        df = maybe_adjust_for_splits(df)
        dbg_event("stooq_fetch_success", symbol=sym, rows=len(df))
        return df

    t = ticker.lower().strip()
    # Try lowercase ticker first
    df = _try(t)
    if not df.empty:
        return df
    # Try with .us suffix
    df = _try(f"{t}.us")
    if not df.empty:
        return df

    # Log failure for debugging
    dbg_event("stooq_all_attempts_failed", ticker=ticker)
    return pd.DataFrame()

def cache_price(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    path = os.path.join(DATA_DIR, f"{ticker.upper()}.csv")
    cached_df: Optional[pd.DataFrame] = None
    if os.path.exists(path):
        try:
            tmp = pd.read_csv(path)
            tmp["Date"] = pd.to_datetime(tmp["Date"])
            tmp = tmp.set_index("Date").sort_index()
            tmp.columns = [c.lower() for c in tmp.columns]
            tmp = maybe_adjust_for_splits(tmp)
            cached_df = tmp
            if not force_refresh:
                return tmp
        except Exception:
            cached_df = None

    df = fetch_stooq_daily(ticker)
    if df.empty or len(df) < 60:
        return cached_df if cached_df is not None else pd.DataFrame()

    df = maybe_adjust_for_splits(df)
    df.reset_index().to_csv(path, index=False)
    return df

def require_history(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    if df.empty or len(df) < min_rows:
        raise RuntimeError(f"Not enough history (got {len(df)}, need {min_rows})")
    return df

def friday_close_series(df: pd.DataFrame) -> pd.Series:
    return df["close"].resample("W-FRI").last().dropna()

def rolling_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=w).mean()

def weekly_vol_from_daily(close: pd.Series) -> pd.Series:
    rets = close.pct_change()
    return rets.rolling(20, min_periods=20).std() * np.sqrt(5)

def avg_volume_20d(df: pd.DataFrame) -> float:
    if "volume" not in df.columns or len(df) < 25:
        return float("nan")
    v = pd.to_numeric(df["volume"], errors="coerce").tail(20).mean()
    try:
        return float(v)
    except Exception:
        return float("nan")

def rsi14(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr14(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not {"high", "low", "close"}.issubset(set(df.columns)):
        return pd.Series(index=df.index, dtype=float)
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ==========================================================
# Intraday provider (Twelve Data)
# ==========================================================
def get_td_key_from_env() -> str:
    return os.environ.get("TWELVE_DATA_API_KEY", "").strip()

@st.cache_data(ttl=20, show_spinner=False)
def fetch_td_time_series(symbol: str, interval: str, outputsize: int, apikey: str) -> pd.DataFrame:
    if not apikey:
        return pd.DataFrame()
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(outputsize),
        "apikey": apikey,
        "format": "JSON",
    }
    r = http_get(url, params=params, timeout=20, retries=2)
    if r is None or r.status_code != 200:
        return pd.DataFrame()
    try:
        js = r.json()
    except Exception:
        return pd.DataFrame()
    if "values" not in js:
        return pd.DataFrame()
    vals = js["values"]
    if not vals:
        return pd.DataFrame()
    df = pd.DataFrame(vals)
    if "datetime" not in df.columns:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"], how="any")

def cache_intraday(symbol: str, interval: str, apikey: str, outputsize: int = 500, min_fetch_secs: int = 25) -> pd.DataFrame:
    ensure_dirs()
    path = os.path.join(INTRADAY_DIR, f"{symbol.upper()}_{interval}.csv")

    def _load_disk() -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            old = pd.read_csv(path)
            if "datetime" not in old.columns:
                return pd.DataFrame()
            old["datetime"] = pd.to_datetime(old["datetime"], errors="coerce")
            old = old.dropna(subset=["datetime"]).set_index("datetime").sort_index()
            for c in ["open", "high", "low", "close", "volume"]:
                if c in old.columns:
                    old[c] = pd.to_numeric(old[c], errors="coerce")
            return old.dropna(subset=["close"])
        except Exception:
            return pd.DataFrame()

    try:
        if os.path.exists(path):
            age = time.time() - os.path.getmtime(path)
            if age < float(min_fetch_secs):
                return _load_disk()
    except Exception:
        pass

    if not apikey:
        return _load_disk()

    fresh = fetch_td_time_series(symbol, interval=interval, outputsize=int(outputsize), apikey=apikey)
    if fresh.empty:
        return _load_disk()

    fresh = fresh.copy()
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in fresh.columns]
    fresh = fresh[cols].dropna(subset=["close"])

    old = _load_disk()
    merged = pd.concat([old, fresh], axis=0) if not old.empty else fresh
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged = merged.tail(25000)

    merged.reset_index().to_csv(path, index=False)
    return merged

def intraday_signal(df: pd.DataFrame) -> dict:
    if df.empty or "close" not in df.columns:
        return {"Bias": "N/A", "Score": 0, "Why": "No intraday data."}
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    mom_10 = (close.iloc[-1] / close.iloc[-min(10, len(close))]) - 1.0
    rsi = rsi14(close, 14).iloc[-1] if len(close) >= 20 else np.nan
    trend_up = close.iloc[-1] > ema20.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]
    trend_dn = close.iloc[-1] < ema20.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]
    score = 5.0
    score += 1.5 if trend_up else (-1.5 if trend_dn else 0.0)
    score += 1.0 if mom_10 > 0 else (-1.0 if mom_10 < 0 else 0.0)
    if not np.isnan(rsi):
        if 45 <= rsi <= 65:
            score += 1.0
        elif rsi > 75:
            score -= 1.0
        elif rsi < 25:
            score -= 1.0
    score = clamp(score, 0, 10)
    if score >= 7:
        bias = "Long bias (only if your rules agree)"
    elif score <= 3:
        bias = "Short bias (only if your rules agree)"
    else:
        bias = "No clear edge / wait"
    why = f"Trend={'UP' if trend_up else ('DOWN' if trend_dn else 'MIXED')} | mom10={mom_10*100:.2f}% | RSI={'' if np.isnan(rsi) else f'{rsi:.0f}'} | score={score:.1f}/10"
    return {"Bias": bias, "Score": round(score, 1), "Why": why}


# ==========================================================
# News (RSS) – no lxml required
# ==========================================================
@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_rss_items(url: str, max_items: int = 12):
    try:
        r = http_get(url, timeout=15, retries=2)
        if r is None or r.status_code != 200:
            return []
        root = ET.fromstring(r.text)

        chan = root.find("channel")
        if chan is not None:
            items = []
            for it in chan.findall("item")[:max_items]:
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                pub = (it.findtext("pubDate") or "").strip()
                items.append({"title": title, "link": link, "pubDate": pub})
            return items

        ns = {"a": "http://www.w3.org/2005/Atom"}
        entries = root.findall("a:entry", ns)
        if entries:
            out = []
            for e in entries[:max_items]:
                title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
                link_el = e.find("a:link", ns)
                link = (link_el.get("href") if link_el is not None else "").strip()
                pub = (e.findtext("a:updated", default="", namespaces=ns) or "").strip()
                out.append({"title": title, "link": link, "pubDate": pub})
            return out

        return []
    except Exception:
        return []

def google_news_rss_query(q: str) -> str:
    q2 = requests.utils.quote(q)
    return f"https://news.google.com/rss/search?q={q2}&hl=en-GB&gl=GB&ceid=GB:en"

NEWS_POS = ["beats", "beat", "raises guidance", "upgrade", "strong", "record", "surge", "wins", "profit up"]
NEWS_NEG = ["misses", "miss", "cuts guidance", "downgrade", "layoffs", "strike", "lawsuit", "recall", "probe", "slump", "plunge"]
NEWS_LABOR = ["immigration", "migrant", "visa", "border", "ice", "union", "strike", "wage", "minimum wage", "labor", "hiring", "layoffs"]

def _safe_pub_dt(s: str) -> Optional[datetime]:
    try:
        dt = parsedate_to_datetime(s)
        return dt.replace(tzinfo=None) if dt is not None else None
    except Exception:
        return None

def _count_hits(text: str, words: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for w in words if w in t)

def feature_hash(x_row: np.ndarray) -> str:
    import hashlib

    b = np.asarray(x_row, dtype=np.float64).tobytes()
    return hashlib.sha256(b).hexdigest()[:12]

def parse_aliases(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split("|") if x.strip()]

def get_prices_by_ticker(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            dfp = cache_price(t)
            if dfp is None or dfp.empty:
                continue
            # Convert to standard format: Date, Close, volume columns
            tmp = dfp.reset_index()
            # Rename columns to match expected format
            rename_map = {}
            if "Date" not in tmp.columns and tmp.columns[0]:  # Index was reset, first col is date
                rename_map[tmp.columns[0]] = "Date"
            if "close" in tmp.columns:
                rename_map["close"] = "Close"
            if rename_map:
                tmp = tmp.rename(columns=rename_map)

            # Keep Date, Close, and volume (if exists)
            cols_to_keep = ["Date", "Close"]
            if "volume" in tmp.columns:
                cols_to_keep.append("volume")

            tmp = tmp[cols_to_keep].copy()
            tmp = tmp.sort_values("Date").reset_index(drop=True)
            out[t.upper()] = tmp
        except (KeyError, ValueError, AttributeError) as e:
            # Skip tickers with missing or malformed data
            continue
    return out

def _to_date(x):
    return pd.to_datetime(x).date()

def evaluate_predictions(pred_df: pd.DataFrame, prices_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, r in pred_df.iterrows():
        ticker = str(r.get("ticker", "")).upper()
        if not ticker or ticker not in prices_by_ticker:
            continue

        dfp = prices_by_ticker[ticker].copy()
        if dfp is None or dfp.empty or "Date" not in dfp.columns or "Close" not in dfp.columns:
            continue
        dfp["Date"] = pd.to_datetime(dfp["Date"]).dt.date
        dfp = dfp.sort_values("Date")

        try:
            asof = _to_date(r.get("asof_date"))
        except Exception:
            continue

        prob_up_raw = float(r.get("prob_up", r.get("ai_prob", np.nan)))
        if np.isnan(prob_up_raw):
            continue
        prob_up = prob_up_raw / 100.0 if prob_up_raw > 1 else prob_up_raw

        entry = r.get("entry_close", r.get("asof_close", np.nan))
        if np.isnan(entry):
            continue
        entry = float(entry)

        target_date_val = r.get("target_date", None)
        horizon = r.get("horizon", None)
        try:
            if pd.notna(target_date_val):
                target_date = _to_date(target_date_val)
                horizon_days = abs((target_date - asof).days)
            else:
                horizon_days = int(horizon) if horizon is not None else 7
                target_date = None
        except Exception:
            horizon_days = 7
            target_date = None

        dates = dfp["Date"].tolist()
        if asof not in set(dates):
            dates_le = [d for d in dates if d <= asof]
            if not dates_le:
                continue
            asof = max(dates_le)

        try:
            idx = dates.index(asof)
        except ValueError:
            continue

        target_idx = min(idx + horizon_days, len(dates) - 1)
        target_date = dates[target_idx] if target_date is None else target_date

        # Handle case where target_date might not exist in price data
        target_rows = dfp.loc[dfp["Date"] == target_date, "Close"]
        if target_rows.empty:
            # If exact date not found, use closest available date
            dates_ge = [d for d in dates if d >= target_date]
            if not dates_ge:
                continue
            target_date = min(dates_ge)
            target_rows = dfp.loc[dfp["Date"] == target_date, "Close"]
            if target_rows.empty:
                continue

        target_close = float(target_rows.iloc[0])

        actual_ret = (target_close / entry) - 1.0
        actual_up = 1 if actual_ret > 0 else 0

        pred_up = 1 if prob_up >= 0.5 else 0
        correct = pred_up == actual_up
        closeness = prob_up if actual_up == 1 else (1.0 - prob_up)
        strat_ret = actual_ret if pred_up == 1 else (-actual_ret)

        rows.append(
            {
                "Ticker": ticker,
                "As Of": str(asof),
                "Horizon(d)": horizon_days,
                "Entry": entry,
                "Target Date": str(target_date),
                "Target": target_close,
                "Price Move": "▲" if actual_up else "▼",
                "Actual %": actual_ret * 100.0,
                "Pred Move": "▲" if pred_up else "▼",
                "Prob Up": prob_up * 100.0,
                "Correct": "✅" if correct else "❌",
                "How Close": closeness * 100.0,
                "Strategy %": strat_ret * 100.0,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["How Close", "Correct"], ascending=[False, False]).reset_index(drop=True)
    return out

def style_scoreboard(df: pd.DataFrame):
    if df is None or df.empty:
        return df

    def color_arrows(val):
        if val == "▲":
            return "color: #1b5e20; font-weight: 700;"
        if val == "▼":
            return "color: #b71c1c; font-weight: 700;"
        return ""

    def color_correct(val):
        if val == "✅":
            return "background-color: #e8f5e9; font-weight: 700;"
        if val == "❌":
            return "background-color: #ffebee; font-weight: 700;"
        return ""

    def bar_closeness(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 70:
            return "background-color: #e8f5e9; font-weight: 700;"
        if x >= 55:
            return "background-color: #fffde7;"
        return "background-color: #ffebee;"

    sty = df.style
    if "Price Move" in df.columns:
        sty = sty.map(color_arrows, subset=["Price Move"])
    if "Pred Move" in df.columns:
        sty = sty.map(color_arrows, subset=["Pred Move"])
    if "Correct" in df.columns:
        sty = sty.map(color_correct, subset=["Correct"])
    if "How Close" in df.columns:
        sty = sty.map(bar_closeness, subset=["How Close"])

    fmt = {}
    for c in ["Actual %", "Prob Up", "How Close", "Strategy %"]:
        if c in df.columns:
            fmt[c] = "{:.2f}"
    if fmt:
        sty = sty.format(fmt)
    return sty

def add_day_move(scan_df: pd.DataFrame, prices_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scan_df = scan_df.copy()
    moves = []
    try:
        tickers_series = get_ticker_series(scan_df)
    except KeyError:
        # No ticker column found, return original dataframe
        return scan_df
    for t in tickers_series:
        ticker = str(t).upper()
        dfp = prices_by_ticker.get(ticker)
        if dfp is None or dfp.empty:
            moves.append((np.nan, np.nan, "", np.nan))
            continue
        dfp = dfp.sort_values("Date")
        last = float(dfp["Close"].iloc[-1])
        prev = float(dfp["Close"].iloc[-2]) if len(dfp) >= 2 else np.nan
        pct = ((last / prev) - 1.0) * 100.0 if prev == prev else np.nan
        arrow = "▲" if pct > 0 else "▼"
        moves.append((last, prev, arrow, pct))

    scan_df["Last Close"] = [m[0] for m in moves]
    scan_df["Prev Close"] = [m[1] for m in moves]
    scan_df["Day Move"] = [m[2] for m in moves]
    scan_df["Day %"] = [m[3] for m in moves]
    return scan_df

def build_google_news_query(ticker: str, company_name: str) -> str:
    t = str(ticker).upper().strip()
    nm = (company_name or "").strip()

    intent = "(stock OR shares OR earnings OR guidance OR forecast OR results)"

    if nm:
        if t in AMBIGUOUS_TICKERS:
            return f"\"{nm}\" ({t} OR ${t} OR \"NYSE:{t}\" OR \"NASDAQ:{t}\") {intent}"
        return f"({t} OR ${t} OR \"NYSE:{t}\" OR \"NASDAQ:{t}\" OR \"{nm}\") {intent}"

    return f"({t} OR ${t} OR \"NYSE:{t}\" OR \"NASDAQ:{t}\") {intent}"


def filter_items_by_company_match(items: list[dict], ticker: str, company_name: str) -> list[dict]:
    kept = []
    for it in items:
        txt = f"{it.get('title','')} {it.get('link','')}"
        score, reasons = score_article_to_ticker(txt, ticker, company_name)

        if ticker in AMBIGUOUS_TICKERS:
            if score >= 8:
                it2 = dict(it)
                it2["_match_score"] = score
                it2["_match_reasons"] = ",".join(reasons)
                kept.append(it2)
        else:
            if score >= 6:
                it2 = dict(it)
                it2["_match_score"] = score
                it2["_match_reasons"] = ",".join(reasons)
                kept.append(it2)

    return kept


@st.cache_resource(show_spinner=False)
def load_news_model_cached():
    if not os.path.exists(NEWS_MODEL_PATH):
        return None
    try:
        return load(NEWS_MODEL_PATH)
    except Exception:
        return None

def predict_good_probabilities(model, texts: list[str]) -> list[float]:
    if model is None or not hasattr(model, "predict_proba"):
        return [np.nan for _ in texts]
    try:
        probs = model.predict_proba(texts)
        classes = list(getattr(model, "classes_", []))
        idx = classes.index("good") if "good" in classes else 0
        return [float(p[idx]) if hasattr(p, "__len__") else float(p) for p in probs]
    except Exception:
        return [np.nan for _ in texts]

def score_news_items_with_model(items: list[dict], model=None, ticker: str | None = None) -> tuple[list[dict], dict]:
    if not items:
        return items, {"prob_good_mean": np.nan, "prob_good_max": np.nan, "scored": 0}
    model = model or load_news_model_cached()
    if model is None:
        return items, {"prob_good_mean": np.nan, "prob_good_max": np.nan, "scored": 0}

    labels = load_news_labels_df()
    label_map = {}
    if not labels.empty:
        labels["headline_id"] = labels["headline_id"].astype(str)
        label_map = labels.set_index("headline_id")["user_feedback"].fillna("").to_dict()

    texts = [f"{it.get('title','')} {it.get('snippet','')}".strip() for it in items]
    probs = predict_good_probabilities(model, texts)
    prob_arr = pd.to_numeric(pd.Series(probs), errors="coerce")
    prob_mean = float(prob_arr.mean()) if len(prob_arr.dropna()) else np.nan
    prob_max = float(prob_arr.max()) if len(prob_arr.dropna()) else np.nan

    out_items = []
    ticker_norm = str(ticker or "").upper().strip()
    for it, p in zip(items, probs):
        it2 = dict(it)
        hid = news_headline_id(ticker_norm or it.get("ticker", "") or it.get("symbol", "") or "", it.get("title", ""), it.get("link", ""))
        if hid:
            it2["headline_id"] = hid
            if hid in label_map:
                it2["user_feedback"] = label_map.get(hid, "")
        try:
            it2["_prob_good"] = float(p)
        except Exception:
            it2["_prob_good"] = np.nan
        out_items.append(it2)

    return out_items, {
        "prob_good_mean": prob_mean,
        "prob_good_max": prob_max,
        "scored": len(out_items),
    }

def train_news_classifier(min_rows: int = 20, test_size: float = 0.2) -> dict:
    df = load_news_labels_df()
    if df.empty:
        raise RuntimeError("news_labels.csv is empty. Fetch news and label some headlines first.")

    labeled = df.copy()
    labeled["user_feedback"] = labeled["user_feedback"].fillna("").astype(str).str.lower().str.strip()
    labeled = labeled[labeled["user_feedback"].isin(["good", "bad", "neutral"])]
    labeled["text"] = (labeled["title"].fillna("") + " " + labeled["snippet"].fillna("")).str.strip()
    labeled = labeled[labeled["text"].str.len() > 0]
    if len(labeled) < int(min_rows):
        raise RuntimeError(f"Need at least {min_rows} labeled headlines (good/bad/neutral). Have {len(labeled)}.")
    if labeled["user_feedback"].nunique() < 2:
        raise RuntimeError("Need labels across at least two classes to train the model.")

    ensure_dirs()
    strat = labeled["user_feedback"] if labeled["user_feedback"].nunique() > 1 else None
    tr, te = train_test_split(
        labeled,
        test_size=min(0.4, max(test_size, 0.2)),
        shuffle=True,
        stratify=strat,
        random_state=RANDOM_SEED,
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=2500, multi_class="auto")),
    ])
    pipe.fit(tr["text"], tr["user_feedback"])

    metrics = {
        "trained_at": now_stamp(),
        "rows_used": int(len(labeled)),
        "train_rows": int(len(tr)),
        "holdout_rows": int(len(te)),
        "class_counts": labeled["user_feedback"].value_counts(dropna=False).to_dict(),
    }

    holdout_eval = pd.DataFrame()
    if len(te):
        preds = pipe.predict(te["text"])
        probs = predict_good_probabilities(pipe, te["text"].tolist())
        holdout_eval = te.copy()
        holdout_eval["pred_label"] = preds
        holdout_eval["prob_good"] = probs
        holdout_eval["hit"] = (holdout_eval["pred_label"] == holdout_eval["user_feedback"]).astype(float)
        metrics["accuracy"] = float(accuracy_score(holdout_eval["user_feedback"], preds))

        if "good" in getattr(pipe, "classes_", []):
            bins = pd.cut(
                pd.to_numeric(holdout_eval["prob_good"], errors="coerce"),
                bins=np.linspace(0, 1, 6),
                include_lowest=True,
            )
            cal = (
                holdout_eval.groupby(bins, dropna=True)
                .agg(
                    count=("prob_good", "count"),
                    avg_prob=("prob_good", "mean"),
                    frac_good=("user_feedback", lambda x: np.mean(x == "good")),
                )
                .reset_index()
                .rename(columns={"prob_good": "prob_bin"})
            )
            metrics["calibration_bins"] = cal.to_dict(orient="records")

        per = []
        holdout_eval["logged_utc"] = pd.to_datetime(holdout_eval.get("logged_utc"), errors="coerce")
        for ticker, block in holdout_eval.groupby("ticker"):
            block = block.sort_values("logged_utc")
            rolling = block["hit"].rolling(30, min_periods=5).mean()
            per.append({
                "ticker": ticker,
                "rows": int(len(block)),
                "hit_rate": float(block["hit"].mean()),
                "rolling_hit": float(rolling.dropna().iloc[-1]) if len(rolling.dropna()) else float("nan"),
            })
        metrics["per_ticker"] = per

    dump(pipe, NEWS_MODEL_PATH)
    with open(NEWS_MODEL_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    try:
        load_news_model_cached.clear()
        get_news_features.clear()
    except Exception:
        pass

    return metrics

def load_news_model_metrics() -> dict:
    if not os.path.exists(NEWS_MODEL_METRICS):
        return {}
    try:
        with open(NEWS_MODEL_METRICS, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(ttl=60 * 10, show_spinner=False)
def get_news_features(ticker: str, company_name: str = "", days: int = 7, max_items: int = 30) -> dict:
    q = build_google_news_query(ticker, company_name)
    rss = google_news_rss_query(q)

    items = fetch_rss_items(rss, max_items=max_items)
    if not items:
        return {"headlines": 0, "sent": 0.0, "intensity": 0.0, "labor_hits": 0}

    cutoff = datetime.now() - timedelta(days=int(days))
    in_window = []
    for it in items:
        dt = _safe_pub_dt(it.get("pubDate", ""))
        if dt is None or dt >= cutoff:
            in_window.append(it)

    if not in_window:
        return {"headlines": 0, "sent": 0.0, "intensity": 0.0, "labor_hits": 0}

    matched = filter_items_by_company_match(in_window, str(ticker).upper().strip(), (company_name or "").strip())
    if not matched:
        return {"headlines": 0, "sent": 0.0, "intensity": 0.0, "labor_hits": 0}

    append_news_labels(ticker, matched)
    matched_scored, stats = score_news_items_with_model(matched, ticker=ticker)

    pos = neg = labor = 0
    for it in matched_scored:
        txt = f"{it.get('title','')} {it.get('link','')}"
        pos += _count_hits(txt, NEWS_POS)
        neg += _count_hits(txt, NEWS_NEG)
        labor += _count_hits(txt, NEWS_LABOR)

    denom = max(1, pos + neg)
    sent = clamp((pos - neg) / denom, -1.0, 1.0)
    intensity = clamp(len(matched) / 12.0, 0.0, 1.0)
    return {
        "headlines": len(matched_scored),
        "sent": float(sent),
        "intensity": float(intensity),
        "labor_hits": int(labor),
        "prob_good_mean": stats.get("prob_good_mean", np.nan),
        "prob_good_max": stats.get("prob_good_max", np.nan),
    }

def append_news_log(baseline: str, scored_df: pd.DataFrame):
    if scored_df is None or scored_df.empty:
        return
    ensure_dirs()
    spy = cache_price(baseline)
    if spy.empty:
        return
    bw = friday_close_series(spy)
    if bw.empty:
        return
    asof = str(bw.index[-1].date())

    rows = []
    for _, r in scored_df.iterrows():
        rows.append({
            "timestamp": now_stamp(),
            "asof_date": asof,
            "baseline": baseline,
            "ticker": r.get("Ticker", ""),
            "news_headlines_7d": r.get("NewsHeadlines_7d", np.nan),
            "news_sent_7d": r.get("NewsSent_7d", np.nan),
            "news_intensity_7d": r.get("NewsIntensity_7d", np.nan),
            "news_labor_hits_7d": r.get("NewsLaborHits_7d", np.nan),
            "news_prob_good_mean": r.get("NewsProbGood_Mean", np.nan),
            "news_prob_good_max": r.get("NewsProbGood_Max", np.nan),
        })

    df = pd.DataFrame(rows)
    write_header = not os.path.exists(NEWS_LOG)
    df.to_csv(NEWS_LOG, mode="a", header=write_header, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")


# ==========================================================
# Weekly model features
# ==========================================================
@dataclass
class BaselineContext:
    weekly: pd.Series
    ma20_w: pd.Series
    ma200_w: pd.Series
    vol4w_w: pd.Series

@st.cache_data(ttl=60 * 30, show_spinner=False)
def baseline_context(baseline: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, BaselineContext]:
    spy_df = cache_price(baseline, force_refresh=force_refresh)
    if spy_df.empty:
        raise RuntimeError(
            f"Failed baseline download: {baseline} (tried '{baseline.lower()}' and '{baseline.lower()}.us'). "
            f"Check your internet connection or try a different baseline ticker."
        )
    try:
        spy = require_history(spy_df, MIN_DAILY_ROWS_TRAIN_BASELINE)
    except RuntimeError as e:
        raise RuntimeError(
            f"Baseline {baseline} downloaded but insufficient history: {e}. "
            f"Try using a different baseline with more historical data."
        )
    bw = friday_close_series(spy)
    close = spy["close"]
    b_ma20 = rolling_ma(close, 20)
    b_ma200 = rolling_ma(close, 200)
    ma20_w = b_ma20.reindex(bw.index, method="ffill")
    ma200_w = b_ma200.reindex(bw.index, method="ffill")
    vol4w = bw.pct_change().rolling(4, min_periods=4).std()
    vol4w_w = vol4w.reindex(bw.index, method="ffill")
    return spy, BaselineContext(weekly=bw, ma20_w=ma20_w, ma200_w=ma200_w, vol4w_w=vol4w_w)

FEATURE_COLS = [
    "ret_1w","rs_1w","rs_4w","mom_4w","above_ma20","ma_stack","ma_gap","vol_weekly",
    "spy_rs_4w","spy_above_ma20","spy_above_ma200","spy_vol_4w",
    "ticker_above_ma200","ma20_200_gap","labor_exposure","jan_score","is_january","month",
    "rsi14","atrp14","vol_ratio",
    "rs_12w","mom_12w",
    "dd_52w","ema20_gap","bb_width_20",
]

def build_weekly_features(
    ticker: str,
    bctx: BaselineContext,
    labor_exposure: float,
    jan_score: float,
    force_refresh: bool = False,
) -> pd.DataFrame:
    df = cache_price(ticker, force_refresh=force_refresh)
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()

    close = df["close"]
    w = friday_close_series(df)
    common = w.index.intersection(bctx.weekly.index)
    w = w.loc[common]
    b = bctx.weekly.loc[common]
    if len(w) < 90:
        return pd.DataFrame()

    r1 = w.pct_change(1)
    r4 = (w / w.shift(4)) - 1.0
    b1 = b.pct_change(1)
    b4 = (b / b.shift(4)) - 1.0
    rs1 = r1 - b1
    rs4 = r4 - b4

    r12 = (w / w.shift(12)) - 1.0
    b12 = (b / b.shift(12)) - 1.0
    rs12 = r12 - b12
    mom12 = r12

    ema20_d = close.ewm(span=20, adjust=False).mean()
    ema20_gap_d = (close / ema20_d) - 1.0

    roll_max_252 = close.rolling(252, min_periods=252).max()
    dd_52w_d = (close / roll_max_252.replace(0, np.nan)) - 1.0

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_width_d = (4.0 * bb_std / bb_mid.replace(0, np.nan))

    rs12_w = rs12.reindex(w.index)
    mom12_w = mom12.reindex(w.index)
    ema20_gap_w = ema20_gap_d.reindex(w.index, method="ffill")
    dd_52w_w = dd_52w_d.reindex(w.index, method="ffill")
    bb_width_w = bb_width_d.reindex(w.index, method="ffill")

    ma20 = rolling_ma(close, 20)
    ma50 = rolling_ma(close, 50)
    ma200 = rolling_ma(close, 200)
    volw = weekly_vol_from_daily(close)

    rsi_d = rsi14(close, 14)
    atr_d = atr14(df, 14)
    atrp_d = atr_d / close.replace(0, np.nan)

    vol_ratio_d = pd.Series(index=df.index, dtype=float)
    if "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce")
        v20 = v.rolling(20, min_periods=20).mean()
        v100 = v.rolling(100, min_periods=100).mean()
        vol_ratio_d = v20 / v100.replace(0, np.nan)

    ma20_w = ma20.reindex(w.index, method="ffill")
    ma50_w = ma50.reindex(w.index, method="ffill")
    ma200_w = ma200.reindex(w.index, method="ffill")
    volw_w = volw.reindex(w.index, method="ffill")
    rsi_w = rsi_d.reindex(w.index, method="ffill")
    atrp_w = atrp_d.reindex(w.index, method="ffill")
    vol_ratio_w = vol_ratio_d.reindex(w.index, method="ffill")

    above_ma20 = (w > ma20_w).astype(int)
    ma_stack = (ma20_w > ma50_w).astype(int)
    ma_gap = (ma20_w / ma50_w) - 1.0

    ticker_above_ma200 = (w > ma200_w).astype(int)
    ma20_200_gap = (ma20_w / ma200_w) - 1.0

    month = pd.Series(w.index.month, index=w.index).astype(int)
    is_jan = (month == 1).astype(int)

    spy_above_ma20 = (b > bctx.ma20_w.loc[common]).astype(int)
    spy_above_ma200 = (b > bctx.ma200_w.loc[common]).astype(int)
    spy_vol_4w = bctx.vol4w_w.loc[common]
    spy_rs4 = b4

    next_rs1 = rs1.shift(-1)
    thr = bctx.vol4w_w.loc[common].apply(outperform_threshold)

    # stronger labels: drop "neutral" weeks where edge is tiny/noisy
    y = pd.Series(np.nan, index=common, dtype=float)
    y[next_rs1 >  thr] = 1
    y[next_rs1 < -thr] = 0


    out = pd.DataFrame(
        {
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
            "spy_above_ma200": spy_above_ma200,
            "spy_vol_4w": spy_vol_4w,
            "ticker_above_ma200": ticker_above_ma200,
            "ma20_200_gap": ma20_200_gap,
            "labor_exposure": float(labor_exposure),
            "jan_score": float(jan_score),
            "is_january": is_jan,
            "month": month,
            "rsi14": rsi_w,
            "atrp14": atrp_w,
            "vol_ratio": vol_ratio_w,
            "y": y,
            "rs_12w": rs12_w,
            "mom_12w": mom12_w,
            "dd_52w": dd_52w_w,
            "ema20_gap": ema20_gap_w,
            "bb_width_20": bb_width_w,
        }
    ).dropna()

    out = out.iloc[:-1].copy()
    return out

# =========================
# Features + dataset (new stable pipeline)
# =========================
FEATURE_COLS = [
    "ret_1", "ret_2", "ret_5",
    "vol_5", "vol_20",
    "mom_5_20", "mom_10_50",
    "vol_z_20",
]

def build_features_for_ticker(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: normalized df: date, close, volume
    Output: df with FEATURES + date + close
    """
    df = normalize_price_df(price_df)
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    df = df.copy()
    df["ret_1"] = np.log(df["close"]).diff(1)
    df["ret_2"] = np.log(df["close"]).diff(2)
    df["ret_5"] = np.log(df["close"]).diff(5)

    # rolling volatility of returns
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # momentum: moving average ratios
    ma5 = df["close"].rolling(5).mean()
    ma20 = df["close"].rolling(20).mean()
    ma10 = df["close"].rolling(10).mean()
    ma50 = df["close"].rolling(50).mean()

    df["mom_5_20"] = (ma5 / ma20) - 1.0
    df["mom_10_50"] = (ma10 / ma50) - 1.0

    # volume zscore (optional)
    if df["volume"].isna().all():
        df["vol_z_20"] = 0.0
    else:
        vol_mean = df["volume"].rolling(20).mean()
        vol_std = df["volume"].rolling(20).std()
        df["vol_z_20"] = (df["volume"] - vol_mean) / vol_std
        df["vol_z_20"] = df["vol_z_20"].fillna(0.0)

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df

def make_supervised_dataset(
    prices_by_ticker: dict[str, pd.DataFrame],
    horizon_days: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Builds a single dataset across tickers with NO lookahead.
    y = 1 if next-horizon return > 0 else 0
    Returns: X, y, index_df (ticker,date,close,next_close,real_ret)
    """
    rows = []
    for t, pdf in prices_by_ticker.items():
        feat = build_features_for_ticker(pdf)
        if feat.empty:
            continue
        missing_cols = [c for c in FEATURE_COLS if c not in feat.columns]
        if missing_cols:
            continue

        feat["ticker"] = str(t).upper().strip()
        feat["next_close"] = feat["close"].shift(-horizon_days)
        feat["real_ret"] = np.log(feat["next_close"] / feat["close"])
        feat["y_up"] = (feat["real_ret"] > 0).astype(int)

        feat = feat.dropna(subset=["next_close", "real_ret"]).copy()

        rows.append(feat[["ticker", "date", "close", "next_close", "real_ret", "y_up"] + FEATURE_COLS])

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    data = pd.concat(rows, ignore_index=True)
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

    X = data[FEATURE_COLS].copy()
    y = data["y_up"].astype(int).copy()
    idx = data[["ticker", "date", "close", "next_close", "real_ret"]].copy()
    return X, y, idx

# =========================
# Training + prediction (new stable API)
# =========================
def train_or_load_model(
    prices_by_ticker: dict[str, pd.DataFrame],
    horizon_days: int,
    force_retrain: bool,
) -> tuple[object, dict]:
    set_deterministic(SEED)

    # Try load
    model, meta = load_saved_model()
    expected_fp = _fingerprint_features(FEATURE_COLS, horizon_days)

    if (not force_retrain) and model is not None and meta is not None:
        if meta.get("feature_fingerprint") == expected_fp:
            return model, meta

    # Build dataset
    X, y, idx = make_supervised_dataset(prices_by_ticker, horizon_days=horizon_days)
    if X.empty:
        raise ValueError("Not enough data to train (need ~60+ bars per ticker).")

    # Simple strong baseline (stable)
    base = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=None
        ))
    ])

    # calibrated probabilities = better confidence
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X, y)

    meta = {
        "trained_utc": datetime.utcnow().isoformat(),
        "seed": SEED,
        "horizon_days": horizon_days,
        "feature_cols": FEATURE_COLS,
        "feature_fingerprint": expected_fp,
        "n_rows": int(len(X)),
        "tickers": int(idx["ticker"].nunique()),
        "date_min": str(idx["date"].min()),
        "date_max": str(idx["date"].max()),
    }

    save_model_with_meta(model, meta)
    return model, meta

def predict_latest(
    model: object,
    prices_by_ticker: dict[str, pd.DataFrame],
    horizon_days: int
) -> pd.DataFrame:
    """
    Returns a table with ticker, last_close, prob_up, predicted_dir
    """
    out_rows = []

    for t, pdf in prices_by_ticker.items():
        feat = build_features_for_ticker(pdf)
        if feat.empty:
            continue

        last = feat.iloc[-1]
        X_last = pd.DataFrame([{c: float(last[c]) for c in FEATURE_COLS}])
        prob_up = float(model.predict_proba(X_last)[0][1])

        out_rows.append({
            "ticker": str(t).upper().strip(),
            "asof": last["date"],
            "last_close": float(last["close"]),
            "prob_up": prob_up,
            "pred_dir": "UP" if prob_up >= 0.5 else "DOWN",
        })

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    df = df.sort_values("prob_up", ascending=False).reset_index(drop=True)
    return df

# =========================
# Prediction tracking (log + scoring)
# =========================
PRED_LOG_CSV = "predictions_log.csv"

def append_prediction_log(pred_df: pd.DataFrame, model_meta: dict) -> None:
    if pred_df is None or pred_df.empty:
        return

    log = pred_df.copy()
    log["logged_utc"] = datetime.utcnow().isoformat()
    log["model_trained_utc"] = model_meta.get("trained_utc", "")
    log["horizon_days"] = int(model_meta.get("horizon_days", 1))

    # outcome fields (to be filled later)
    if "actual_close" not in log.columns:
        log["actual_close"] = np.nan
    if "real_ret" not in log.columns:
        log["real_ret"] = np.nan
    if "correct_dir" not in log.columns:
        log["correct_dir"] = np.nan
    if "abs_error_ret" not in log.columns:
        log["abs_error_ret"] = np.nan

    header_needed = not os.path.exists(PRED_LOG_CSV)
    log.to_csv(PRED_LOG_CSV, mode="a", header=header_needed, index=False)

def update_log_outcomes(prices_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Fill outcomes for rows where enough time has passed.
    Uses prices_by_ticker to find the close after horizon_days.
    """
    if not os.path.exists(PRED_LOG_CSV):
        return pd.DataFrame()

    log = pd.read_csv(PRED_LOG_CSV)
    if log.empty:
        return log

    # parse dates
    log["asof"] = pd.to_datetime(log["asof"], errors="coerce")
    log["horizon_days"] = pd.to_numeric(log["horizon_days"], errors="coerce").fillna(1).astype(int)

    # only update rows missing actual_close
    need = log["actual_close"].isna()
    if not need.any():
        return log

    for i in log[need].index.tolist():
        t = str(log.at[i, "ticker"]).upper().strip()
        asof = log.at[i, "asof"]
        h = int(log.at[i, "horizon_days"])

        pdf = prices_by_ticker.get(t)
        if pdf is None:
            continue

        df = normalize_price_df(pdf)
        if df.empty:
            continue

        # find asof row (closest <= asof)
        df = df.sort_values("date").reset_index(drop=True)
        pos = df["date"].searchsorted(asof, side="right") - 1
        if pos < 0:
            continue

        future_pos = pos + h
        if future_pos >= len(df):
            continue  # not enough future yet

        last_close = float(df.at[pos, "close"])
        actual_close = float(df.at[future_pos, "close"])
        real_ret = float(np.log(actual_close / last_close))

        pred_dir = str(log.at[i, "pred_dir"])
        correct_dir = int((real_ret > 0 and pred_dir == "UP") or (real_ret <= 0 and pred_dir == "DOWN"))

        # if you later add predicted_return, compute abs error vs that;
        # for now we measure "distance from 0" as a rough closeness proxy
        abs_error_ret = float(abs(real_ret))

        log.at[i, "actual_close"] = actual_close
        log.at[i, "real_ret"] = real_ret
        log.at[i, "correct_dir"] = correct_dir
        log.at[i, "abs_error_ret"] = abs_error_ret

    log.to_csv(PRED_LOG_CSV, index=False)
    return log

def style_predictions(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def row_style(r):
        styles = [""] * len(r)
        # colour pred_dir
        if "pred_dir" in r.index:
            if r["pred_dir"] == "UP":
                styles[r.index.get_loc("pred_dir")] = "background-color: #0b3; color: white; font-weight: 700;"
            else:
                styles[r.index.get_loc("pred_dir")] = "background-color: #b30; color: white; font-weight: 700;"
        # colour correct_dir
        if "correct_dir" in r.index and pd.notna(r["correct_dir"]):
            if int(r["correct_dir"]) == 1:
                styles[r.index.get_loc("correct_dir")] = "background-color: #0a0; color: white; font-weight: 700;"
            else:
                styles[r.index.get_loc("correct_dir")] = "background-color: #a00; color: white; font-weight: 700;"
        return styles

    return (
        df.style
        .apply(row_style, axis=1)
        .format({
            "prob_up": "{:.2%}",
            "last_close": "{:.2f}",
            "actual_close": "{:.2f}",
            "real_ret": "{:.3%}",
            "abs_error_ret": "{:.3%}",
        }, na_rep="")
    )

def pct_change_over_window(series: pd.Series, window: int) -> Optional[float]:
    if series is None or len(series) <= window:
        return None
    try:
        end = float(series.iloc[-1])
        start = float(series.iloc[-window - 1])
        if start == 0 or np.isnan(start) or np.isnan(end):
            return None
        return (end / start) - 1.0
    except Exception:
        return None

def prepare_accuracy_summary(log_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if log_df is None or log_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = log_df.copy()
    df["logged_utc"] = pd.to_datetime(df.get("logged_utc") or df.get("logged_at"), errors="coerce")
    df["correct_dir"] = pd.to_numeric(df.get("correct_dir"), errors="coerce")
    df = df.dropna(subset=["logged_utc", "correct_dir"])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df.sort_values("logged_utc")
    df["rolling_hit"] = df["correct_dir"].rolling(40, min_periods=5).mean()
    df["cum_hit"] = df["correct_dir"].expanding().mean()
    timeline = df[["logged_utc", "rolling_hit", "cum_hit"]].rename(columns={"logged_utc": "timestamp"})

    by_ticker = (
        df.groupby("ticker")["correct_dir"]
        .agg(count="count", hit_rate="mean")
        .reset_index()
    )
    by_ticker["hit_rate"] = (by_ticker["hit_rate"] * 100.0).round(1)

    return timeline, by_ticker

def inject_css():
    """Single place for UI theme / colors.
    Green = up/good, Red = down/bad.
    """
    st.markdown(
        """
        <style>
        :root{
          --bg:#0b1220;
          --panel:#111b2e;
          --panel2:#0f172a;
          --text:#e5e7eb;
          --muted:#94a3b8;

          --green:#16a34a;
          --green2:#22c55e;
          --red:#dc2626;
          --red2:#ef4444;
          --amber:#f59e0b;

          --border:rgba(148,163,184,.22);
        }

        /* Make Streamlit a bit cleaner */
        .block-container { padding-top: 1.2rem; }
        [data-testid="stMetric"] { background: rgba(17,27,46,.45); border: 1px solid var(--border); border-radius: 14px; padding: .6rem .8rem; }

        .card{
          background: linear-gradient(180deg, rgba(17,27,46,.55), rgba(15,23,42,.35));
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 14px 16px;
          margin: 10px 0 14px 0;
          box-shadow: 0 10px 25px rgba(0,0,0,.18);
        }
        .card.upcard{ border-left: 7px solid var(--green2); }
        .card.downcard{ border-left: 7px solid var(--red2); }

        .card h3{ margin: 0 0 8px 0; font-weight: 800; letter-spacing: .2px; }
        .muted{ color: var(--muted); font-size: 0.92rem; }

        .rowline{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:6px; }

        .pill{
          display:inline-flex; align-items:center; gap:6px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid rgba(148,163,184,.25);
          font-weight: 800;
          font-size: 0.85rem;
          line-height: 1;
          white-space: nowrap;
          user-select: none;
        }
        .pill.up{ background: rgba(34,197,94,.14); border-color: rgba(34,197,94,.45); color: #d1fae5; }
        .pill.down{ background: rgba(239,68,68,.14); border-color: rgba(239,68,68,.45); color: #fee2e2; }
        .pill.ok{ background: rgba(34,197,94,.14); border-color: rgba(34,197,94,.45); color: #d1fae5; }
        .pill.bad{ background: rgba(239,68,68,.14); border-color: rgba(239,68,68,.45); color: #fee2e2; }
        .pill.neutral{ background: rgba(148,163,184,.10); border-color: rgba(148,163,184,.30); color: #e5e7eb; }

        .newslist{ margin: .3rem 0 0 0; padding-left: 1.1rem; }
        .newslist li{ margin: .25rem 0; color: var(--text); }
        .newslist a{ color: #93c5fd; text-decoration: none; }
        .newslist a:hover{ text-decoration: underline; }

        .tiny{ font-size: .82rem; color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )

def fmt_pct(x):
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return ""

def arrow_from_ret(r):
    if pd.isna(r):
        return "⏳"
    return "▲" if r > 0 else "▼"

def pill(label, kind):
    return f'<span class="pill {kind}">{label}</span>'

def compute_price_moves(price_df: pd.DataFrame) -> dict:
    df = normalize_price_df(price_df)
    if df.empty or len(df) < 6:
        return {"last": np.nan, "last_close": np.nan, "ret_1d": np.nan, "ret_5d": np.nan, "asof": None, "series": df}

    last = float(df["close"].iloc[-1])
    prev1 = float(df["close"].iloc[-2])
    prev5 = float(df["close"].iloc[-6])

    ret_1d = (last / prev1) - 1.0
    ret_5d = (last / prev5) - 1.0
    asof = df["date"].iloc[-1]
    return {"last": last, "last_close": last, "ret_1d": ret_1d, "ret_5d": ret_5d, "asof": asof, "series": df}

def render_price_chart(
    df_norm: pd.DataFrame,
    pred_points: Optional[pd.DataFrame] = None,
    *,
    height: int = 220,
    show_returns: bool = True,
    show_pred_table: bool = False,
):
    """
    Clean price chart:
      - Close + MA20 line (Altair if available; otherwise Streamlit line_chart)
      - Optional daily returns bar chart
      - Optional prediction markers table (since we don't have predicted prices)
    """
    if df_norm is None or df_norm.empty:
        st.info("No price history available for chart.")
        return

    df = df_norm.copy()
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df["ma20"] = df["close"].rolling(20, min_periods=5).mean()

    # Try Altair first (nicer than st.line_chart)
    try:
        import altair as alt  # type: ignore

        base = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("close:Q", title="Close"),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("close:Q", format=".2f")],
            )
        )
        ma = (
            alt.Chart(df)
            .mark_line(strokeDash=[6, 4])
            .encode(
                x="date:T",
                y=alt.Y("ma20:Q", title=""),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("ma20:Q", format=".2f")],
            )
        )
        st.altair_chart((base + ma).properties(height=height), use_container_width=True)
    except Exception:
        chart_df = df.set_index("date")[["close", "ma20"]].tail(240)
        st.line_chart(chart_df, height=height)

    if show_returns:
        # Daily returns (%)
        rets = df[["date", "close"]].copy()
        rets["ret_%"] = rets["close"].pct_change() * 100.0
        rets = rets.dropna().tail(60)
        if not rets.empty:
            st.bar_chart(rets.set_index("date")[["ret_%"]], height=110)

    # Optional: show pred points as a table (we only have timestamps/labels)
    if show_pred_table and pred_points is not None and not pred_points.empty:
        pp = pred_points.copy()
        if "date" in pp.columns:
            pp["date"] = pd.to_datetime(pp["date"], errors="coerce")
        cols = [c for c in ["date", "label", "horizon_unit", "horizon_value", "prob_up"] if c in pp.columns]
        st.caption("Prediction points (timestamps / confidence):")
        st.dataframe(pp[cols].sort_values("date", ascending=False), use_container_width=True, height=180)

def google_news_rss_url(query: str, days: int = 14) -> str:
    q = f'{query} when:{days}d'
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-GB&gl=GB&ceid=GB:en"

def fetch_url_text(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text

def parse_google_news_rss(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()

        dt = None
        try:
            dt = parsedate_to_datetime(pub)
        except Exception:
            dt = None

        clean_desc = re.sub(r"<.*?>", "", desc)
        clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

        items.append({
            "title": title,
            "link": link,
            "published": dt.isoformat() if dt else "",
            "snippet": clean_desc[:220],
        })
    return items

def cache_path_for_ticker(ticker: str) -> str:
    return os.path.join(NEWS_DIR, f"{ticker.upper().strip()}.json")

def load_cached_news(ticker: str, ttl_minutes: int = 60) -> list[dict] | None:
    path = cache_path_for_ticker(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ts = payload.get("cached_utc", "")
        cached = datetime.fromisoformat(ts) if ts else None
        if cached and (datetime.utcnow() - cached).total_seconds() < ttl_minutes * 60:
            return payload.get("items", [])
    except Exception:
        return None
    return None

def save_cached_news(ticker: str, items: list[dict]) -> None:
    path = cache_path_for_ticker(ticker)
    payload = {"cached_utc": datetime.utcnow().isoformat(), "items": items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


NEWS_LABEL_COLUMNS = [
    "headline_id",
    "logged_utc",
    "ticker",
    "title",
    "snippet",
    "link",
    "published",
    "user_feedback",
]

def news_headline_id(ticker: str, title: str, link: str = "") -> str:
    raw = f"{(ticker or '').upper().strip()}|{title or ''}|{link or ''}".strip().lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def load_news_labels_df() -> pd.DataFrame:
    df = safe_read_log_csv(NEWS_LABELS_PATH)
    if df is None or df.empty:
        return pd.DataFrame(columns=NEWS_LABEL_COLUMNS)
    out = df.copy()
    for c in NEWS_LABEL_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out

def append_news_labels(ticker: str, items: list[dict]) -> None:
    if not items:
        return
    ensure_dirs()
    existing = load_news_labels_df()
    seen = set(existing["headline_id"].astype(str)) if not existing.empty else set()
    rows = []
    for it in items:
        hid = news_headline_id(ticker, it.get("title", ""), it.get("link", ""))
        if hid in seen:
            continue
        rows.append({
            "headline_id": hid,
            "logged_utc": now_stamp(),
            "ticker": str(ticker).upper().strip(),
            "title": it.get("title", ""),
            "snippet": it.get("snippet", ""),
            "link": it.get("link", ""),
            "published": it.get("published", ""),
            "user_feedback": "",
        })
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    write_header = not os.path.exists(NEWS_LABELS_PATH)
    df_new.to_csv(
        NEWS_LABELS_PATH,
        mode="a",
        header=write_header,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )

def upsert_news_feedback(headline_id: str, feedback: str) -> bool:
    hid = str(headline_id or "").strip()
    if not hid:
        return False
    df = load_news_labels_df()
    if df.empty:
        return False
    df["headline_id"] = df["headline_id"].astype(str)
    if hid not in set(df["headline_id"].tolist()):
        return False
    df.loc[df["headline_id"] == hid, "user_feedback"] = str(feedback or "").strip().lower()
    df.to_csv(NEWS_LABELS_PATH, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    return True

def build_news_query(ticker: str, company_name: str | None) -> str:
    t = ticker.upper().strip()
    name = (company_name or "").strip()

    if t in AMBIGUOUS_TICKERS and name:
        return f'"{name}" stock OR "{t}" stock'
    if name:
        return f'"{t}" stock OR "{name}" stock'
    return f'"{t}" stock'

def fetch_news_for_ticker(ticker: str, company_name: str | None, days: int = 14, max_items: int | None = None) -> list[dict]:
    cached = load_cached_news(ticker, ttl_minutes=60)
    if cached is not None:
        items_cached = cached if max_items is None else cached[: int(max_items)]
        append_news_labels(ticker, items_cached)
        return items_cached

    query = build_news_query(ticker, company_name)
    url = google_news_rss_url(query, days=days)

    try:
        xml_text = fetch_url_text(url)
        items = parse_google_news_rss(xml_text)
    except Exception:
        items = []

    t = ticker.upper().strip()
    name = (company_name or "").lower()
    filtered = []
    for it in items:
        text = (it["title"] + " " + it["snippet"]).lower()
        hit = False
        if name and name in text:
            hit = True
        if f" {t.lower()} " in f" {text} ":
            hit = True
        if t in AMBIGUOUS_TICKERS and name:
            hit = hit
        if hit:
            filtered.append(it)

    if max_items is not None:
        try:
            filtered = filtered[: int(max_items)]
        except Exception:
            filtered = filtered

    append_news_labels(ticker, filtered)
    save_cached_news(ticker, filtered)
    return filtered

def summarize_accuracy(log_df: pd.DataFrame) -> dict:
    if log_df is None or log_df.empty:
        return {"n": 0, "acc": np.nan, "avg_abs_ret": np.nan}

    done = log_df[pd.notna(log_df.get("correct_dir"))].copy()
    if done.empty:
        return {"n": 0, "acc": np.nan, "avg_abs_ret": np.nan}

    acc = float(done["correct_dir"].mean())
    avg_abs_ret = float(pd.to_numeric(done["abs_error_ret"], errors="coerce").dropna().mean()) if "abs_error_ret" in done.columns else np.nan
    return {"n": int(len(done)), "acc": acc, "avg_abs_ret": avg_abs_ret}

def render_ticker_card(
    ticker: str,
    company_name: str,
    price_move: dict,
    pred_row: dict,
    last_outcome_row: Optional[dict],
    news_items: list,
    acc_row: Optional[dict] = None,
    *,
    show_chart: bool = True,
    show_news: bool = True,
    max_headlines: int = 4,
):
    """
    Overview card that actually explains what's going on:
      - Clear UP/DOWN + % move
      - AI direction + confidence
      - Last evaluated prediction result (correct/wrong)
      - Mini chart + top headlines
    """
    last_close = float(price_move.get("last_close") or (pred_row.get("last_close") or float("nan")))
    r1 = price_move.get("ret_1d")
    r5 = price_move.get("ret_5d")
    series = price_move.get("series")

    trend_cls = "upcard" if (r1 is not None and r1 >= 0) else "downcard"

    prob_up = pred_row.get("prob_up")
    pred_dir = pred_row.get("pred_dir")

    # Header pills (big + obvious)
    header_bits = []
    if pred_dir in ("UP", "DOWN") and prob_up is not None:
        header_bits.append(pill(f"AI: {pred_dir} ({prob_up:.0%})", "up" if pred_dir == "UP" else "down"))
    if r1 is not None:
        header_bits.append(pill(f"1D: {r1*100:+.2f}%", "up" if r1 >= 0 else "down"))
    if r5 is not None:
        header_bits.append(pill(f"5D: {r5*100:+.2f}%", "up" if r5 >= 0 else "down"))
    header_bits.append(pill(f"News: {len(news_items)}", "neutral"))

    # Last evaluated prediction (if any)
    eval_html = ""
    if last_outcome_row:
        correct = bool(last_outcome_row.get("correct_dir"))
        real_ret = last_outcome_row.get("real_ret")
        p_up = last_outcome_row.get("prob_up")
        conf = None
        if p_up is not None and pd.notna(p_up):
            conf = abs(float(p_up) - 0.5) * 2.0  # 0..1
        if real_ret is not None and pd.notna(real_ret):
            real_ret = float(real_ret) * 100.0

        tag = "ok" if correct else "bad"
        msg = "✅ Last eval: CORRECT" if correct else "❌ Last eval: WRONG"
        extra = []
        if real_ret is not None and not math.isnan(real_ret):
            extra.append(f"move {real_ret:+.2f}%")
        if conf is not None and not math.isnan(conf):
            extra.append(f"confidence {conf:.0%}")
        if extra:
            msg += " • " + ", ".join(extra)
        eval_html = pill(msg, tag)
    else:
        eval_html = pill("No evaluated predictions yet (run + wait for horizon).", "neutral")

    st.markdown(
        f"""
        <div class="card {trend_cls}">
          <h3>{ticker} <span class="muted">— {company_name}</span></h3>
          <div class="rowline">
            {''.join(header_bits)}
            {eval_html}
          </div>
          <div class="tiny" style="margin-top:6px;">
            Last close: <b>{last_close:,.2f}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mini chart
    if show_chart and isinstance(series, pd.DataFrame) and not series.empty:
        render_price_chart(series, height=140, show_returns=False, show_pred_table=False)

    # Top headlines (keep it short)
    if show_news and news_items:
        top = news_items[:max_headlines]
        st.markdown("<div class='muted'>Top headlines</div>", unsafe_allow_html=True)
        st.markdown("<ul class='newslist'>", unsafe_allow_html=True)
        for it in top:
            title = (it.get("title") or "").strip()
            link = (it.get("link") or "").strip()
            src = (it.get("source") or "").strip()
            when = (it.get("published") or "").strip()
            if title and link:
                st.markdown(
                    f"<li><a href='{link}' target='_blank'>{title}</a> <span class='tiny'>({src} • {when})</span></li>",
                    unsafe_allow_html=True,
                )
            elif title:
                st.markdown(f"<li>{title} <span class='tiny'>({src} • {when})</span></li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

    # Accuracy summary (optional)
    if acc_row:
        try:
            dir_acc = float(acc_row.get("direction_accuracy", float("nan")))
            avg_err = float(acc_row.get("avg_abs_pct_error", float("nan")))
        except Exception:
            dir_acc, avg_err = float("nan"), float("nan")
        if not math.isnan(dir_acc) or not math.isnan(avg_err):
            bits = []
            if not math.isnan(dir_acc):
                bits.append(pill(f"Dir accuracy: {dir_acc:.0f}%", "ok" if dir_acc >= 55 else "neutral"))
            if not math.isnan(avg_err):
                bits.append(pill(f"Avg abs move: {avg_err:.2f}%", "neutral"))
            st.markdown("".join(bits), unsafe_allow_html=True)

def render_quick_view_table(rows: list[dict]):
    """Compact scoreboard so you can instantly see what's up/down and what's working."""
    if not rows:
        return

    q = []
    for r in rows:
        t = r.get("ticker")
        name = r.get("company_name")
        pm = r.get("price_move") or {}
        pr = r.get("pred_row") or {}
        lo = r.get("last_outcome_row") or {}
        ar = r.get("acc_row") or {}

        r1 = pm.get("ret_1d")
        r5 = pm.get("ret_5d")

        correct = lo.get("correct_dir")
        real_ret = lo.get("real_ret")
        conf = None
        p_up = lo.get("prob_up", pr.get("prob_up"))
        if p_up is not None and pd.notna(p_up):
            conf = abs(float(p_up) - 0.5) * 2.0

        q.append(
            {
                "Ticker": t,
                "Name": name,
                "Last": pm.get("last_close", pr.get("last_close")),
                "1D %": (float(r1) * 100.0) if r1 is not None else None,
                "5D %": (float(r5) * 100.0) if r5 is not None else None,
                "AI Dir": pr.get("pred_dir"),
                "Prob Up": pr.get("prob_up"),
                "Last eval": ("✅" if bool(correct) else ("❌" if correct is not None else "—")),
                "Real move %": (float(real_ret) * 100.0) if real_ret is not None and pd.notna(real_ret) else None,
                "Confidence": conf,
                "Dir acc %": ar.get("direction_accuracy"),
            }
        )

    df = pd.DataFrame(q)

    if "Confidence" in df.columns:
        df = df.sort_values(["Confidence", "1D %"], ascending=[False, False], na_position="last")

    def _color_ret(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v > 0:
            return "color: #22c55e; font-weight: 800;"
        if v < 0:
            return "color: #ef4444; font-weight: 800;"
        return "color: #94a3b8;"

    def _color_eval(v):
        if v == "✅":
            return "background-color: rgba(34,197,94,.18);"
        if v == "❌":
            return "background-color: rgba(239,68,68,.18);"
        return ""

    try:
        sty = (
            df.style.format(
                {
                    "Last": "{:,.2f}",
                    "1D %": "{:+.2f}",
                    "5D %": "{:+.2f}",
                    "Prob Up": "{:.0%}",
                    "Real move %": "{:+.2f}",
                    "Confidence": "{:.0%}",
                    "Dir acc %": "{:.0f}",
                }
            )
            .applymap(_color_ret, subset=["1D %", "5D %", "Real move %"])
            .applymap(_color_eval, subset=["Last eval"])
        )
        st.dataframe(sty, use_container_width=True, height=320)
    except Exception:
        st.dataframe(df, use_container_width=True, height=320)

def render_news_list(news_items: list[dict], max_items: int = 6):
    if not news_items:
        st.caption("No matched news found for this ticker (try increasing days).")
        return
    for it in news_items[:max_items]:
        title = it.get("title", "")
        link = it.get("link", "")
        pub = it.get("published", "")
        snip = it.get("snippet", "")
        prob = it.get("_prob_good", np.nan)
        feedback = it.get("user_feedback", "")
        st.markdown(f"- [{title}]({link})")
        extras = []
        if prob == prob:
            extras.append(f"P(good)={float(prob):.2f}")
        if isinstance(feedback, str) and feedback.strip():
            extras.append(f"label={feedback.strip()}")
        suffix = f" • {' | '.join(extras)}" if extras else ""
        if pub:
            st.caption(f"{pub} — {snip}{suffix}")
        else:
            st.caption(f"{snip}{suffix}")

def render_dashboard(
    scan_df: pd.DataFrame,
    prices_by_ticker: dict[str, pd.DataFrame],
    pred_df: pd.DataFrame,
    log_df: pd.DataFrame,
    company_name_map: dict[str, str] | None = None,
):
    inject_css()

    scan_df = ensure_ticker_col(scan_df, label="scan_df")
    tickers = scan_df["ticker"].tolist()

    company_name_map = company_name_map or {}

    last_outcome = {}
    if log_df is not None and not log_df.empty:
        tmp = log_df.copy()
        tmp["logged_utc"] = pd.to_datetime(tmp.get("logged_utc"), errors="coerce")
        tmp = tmp.sort_values("logged_utc", ascending=False)
        for _, r in tmp.iterrows():
            t = str(r.get("ticker", "")).upper().strip()
            if t and t not in last_outcome:
                last_outcome[t] = r.to_dict()

    pred_lookup = {}
    if pred_df is not None and not pred_df.empty:
        for _, r in pred_df.iterrows():
            pred_lookup[str(r["ticker"]).upper().strip()] = r.to_dict()

    summ = summarize_accuracy(log_df if log_df is not None else pd.DataFrame())
    c1, c2, c3 = st.columns(3)
    c1.metric("Scored predictions", summ["n"])
    c2.metric("Direction accuracy", "" if pd.isna(summ["acc"]) else f"{summ['acc']*100:.1f}%")
    c3.metric("Avg |return| (closeness)", "" if pd.isna(summ["avg_abs_ret"]) else f"{summ['avg_abs_ret']*100:.2f}%")

    timeline, acc_table = prepare_accuracy_summary(log_df if log_df is not None else pd.DataFrame())
    acc_lookup = {}
    if acc_table is not None and not acc_table.empty:
        for _, r in acc_table.iterrows():
            acc_lookup[str(r.get("ticker", "")).upper().strip()] = {
                "direction_accuracy": r.get("hit_rate", np.nan),
                "avg_abs_pct_error": np.nan,
            }

    tracking_df = log_df.copy() if log_df is not None else pd.DataFrame()
    company_map = company_name_map

    tabs = st.tabs(["Overview", "Ticker detail", "Prediction log", "News"])
    with tabs[0]:
        st.markdown("### Watchlist overview")
        st.caption("Green = price up, Red = price down. Cards show: price move, AI direction/confidence, last evaluated result, mini chart, and headlines.")

        fcol1, fcol2, fcol3, fcol4 = st.columns([2, 1, 1, 1])
        ticker_filter = fcol1.text_input("Filter tickers (e.g. AAPL, MSFT)", value="").strip().upper()
        show_table = fcol2.checkbox("Quick table", value=True)
        show_charts = fcol3.checkbox("Mini charts", value=True)
        show_news = fcol4.checkbox("Headlines", value=True)

        rows = []
        for t in tickers:
            if ticker_filter and ticker_filter not in str(t).upper():
                continue

            name = company_map.get(t, "")
            pdf = prices_by_ticker.get(t)
            pm = compute_price_moves(pdf) if pdf is not None else {}
            pr = pred_lookup.get(t, {}) or {}

            # last evaluated prediction for this ticker (if any)
            last_eval = None
            try:
                tlog = tracking_df[tracking_df["ticker"].astype(str) == str(t)].copy()
                tlog = tlog[tlog["actual_close"].notna()]
                if not tlog.empty:
                    tlog = tlog.sort_values("asof").tail(1)
                    last_eval = tlog.iloc[0].to_dict()
            except Exception:
                last_eval = None

            acc_row = acc_lookup.get(str(t).upper().strip())

            news_items = []
            if show_news:
                news_items = fetch_news_for_ticker(ticker=t, company_name=name, max_items=8, days=10)

            rows.append(
                {
                    "ticker": t,
                    "company_name": name,
                    "price_move": pm,
                    "pred_row": pr,
                    "last_outcome_row": last_eval,
                    "news_items": news_items,
                    "acc_row": acc_row,
                }
            )

        if show_table:
            render_quick_view_table(rows)

        st.markdown("### Cards")
        for r in rows:
            render_ticker_card(
                ticker=r["ticker"],
                company_name=r["company_name"],
                price_move=r["price_move"],
                pred_row=r["pred_row"],
                last_outcome_row=r["last_outcome_row"],
                news_items=r["news_items"],
                acc_row=r["acc_row"],
                show_chart=show_charts,
                show_news=show_news,
                max_headlines=4,
            )
            st.divider()

    with tabs[1]:
        t = st.selectbox("Select ticker", tickers)
        name = company_name_map.get(t, "")
        pdf = prices_by_ticker.get(t)
        if pdf is None:
            st.warning("No prices for this ticker.")
        else:
            df_norm = normalize_price_df(pdf)
            st.subheader(f"{t} {name}".strip())
            st.caption("Price chart (close). Below: latest predictions and whether they scored correct yet.")
            pred_points = pd.DataFrame()
            if log_df is not None and not log_df.empty:
                lt = log_df[log_df["ticker"].astype(str).str.upper().str.strip() == t].copy()
                lt["asof"] = pd.to_datetime(lt["asof"], errors="coerce")
                pred_points = lt.rename(columns={"asof": "date"})[["date", "pred_dir", "prob_up"]].dropna(subset=["date"]).sort_values("date")
            render_price_chart(df_norm, pred_points=pred_points, height=260, show_returns=True, show_pred_table=True)

            st.divider()
            st.subheader("Matched news for this ticker")
            items = fetch_news_for_ticker(t, name, days=14)
            render_news_list(items, max_items=10)

    with tabs[2]:
        if log_df is None or log_df.empty:
            st.info("No prediction log yet. Click 'Log these predictions' after generating predictions.")
        else:
            show = log_df.copy()
            show["ticker"] = show["ticker"].astype(str).str.upper().str.strip()
            show["asof"] = pd.to_datetime(show["asof"], errors="coerce")
            show = show.sort_values("logged_utc", ascending=False).head(500)

            def status_row(r):
                if pd.isna(r.get("correct_dir")):
                    return "PENDING"
                return "CORRECT" if int(r["correct_dir"]) == 1 else "WRONG"

            show["status"] = show.apply(status_row, axis=1)
            st.dataframe(show[[
                "logged_utc","ticker","asof","pred_dir","prob_up","status","actual_close","real_ret","abs_error_ret"
            ]], use_container_width=True)

    with tabs[3]:
        st.caption("This tab shows news PER ticker. It is not mixed; it fetches RSS per company query.")
        t = st.selectbox("Ticker for news", tickers, key="news_ticker_select")
        name = company_name_map.get(t, "")
        days = st.slider("News lookback (days)", 1, 30, 14)
        items = fetch_news_for_ticker(t, name, days=days)
        render_news_list(items, max_items=15)

# =========================
# Original training functions (kept for compatibility)
# =========================
def train_model(
    universe: pd.DataFrame,
    baseline: str,
    force_refresh: bool = False,
    max_tickers: Optional[int] = None,
    max_workers: int = 8,
    fast_mode: bool = True,
    run_cv: bool = False,
    model_kind: str = "sgd",          # "sgd" | "logreg" | "hgb"
    calibrate_probs: bool = False,    # keep off by default (faster)
    halflife_weeks: int = 78,
) -> Tuple[object, dict]:
    _, bctx = baseline_context(baseline, force_refresh=force_refresh)

    rows = []
    uni = universe.copy()
    if max_tickers and int(max_tickers) > 0:
        uni = uni.head(int(max_tickers)).copy()

    def _one(row):
        t = str(row["ticker"]).upper().strip()
        le = float(row.get("labor_exposure", 5))
        js = float(row.get("jan_score", 0))
        return build_weekly_features(t, bctx, le, js, force_refresh=force_refresh)

    futures = []
    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        for _, r in uni.iterrows():
            futures.append(ex.submit(_one, r))
        for fut in as_completed(futures):
            try:
                feat = fut.result()
                if feat is not None and not feat.empty:
                    rows.append(feat)
            except Exception:
                continue

    if not rows:
        raise RuntimeError("No training data built (tickers missing data or baseline fetch failed).")

    data = pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)

    X = data[FEATURE_COLS].astype(float)
    y = data["y"].astype(int)

    weights = sample_weights_from_dates(data["date"], halflife_weeks=int(halflife_weeks))

    # choose model
    if model_kind == "hgb":
        base_est = HistGradientBoostingClassifier(max_depth=6, max_iter=250, learning_rate=0.08)
        pipe = Pipeline([("clf", base_est)])
    else:
        if model_kind == "logreg" or (not fast_mode):
            base_est = LogisticRegression(max_iter=2500, class_weight="balanced")
        else:
            base_est = SGDClassifier(loss="log_loss", max_iter=2500, tol=1e-3, class_weight="balanced", random_state=RANDOM_SEED)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_est)])

    def _fit_with_weights(p, X_, y_, w_):
        # Pipeline: pass sample_weight to final step
        try:
            p.fit(X_, y_, clf__sample_weight=w_)
            return True
        except TypeError:
            # estimator doesn't support sample_weight
            p.fit(X_, y_)
            return False

    accs, aucs = [], []
    if run_cv:
        tscv = TimeSeriesSplit(n_splits=3)
        for tr, te in tscv.split(X):
            _fit_with_weights(pipe, X.iloc[tr], y.iloc[tr], weights[tr])
            try:
                p = pipe.predict_proba(X.iloc[te])[:, 1]
            except Exception:
                # some pipelines might not have predict_proba
                p = pipe.decision_function(X.iloc[te])
                p = 1 / (1 + np.exp(-p))
            yhat = (p >= 0.5).astype(int)
            accs.append(accuracy_score(y.iloc[te], yhat))
            try:
                aucs.append(roc_auc_score(y.iloc[te], p))
            except Exception:
                pass

    # optional calibration (slow-ish)
    final_model = pipe
    if calibrate_probs and model_kind != "hgb":
        cv = TimeSeriesSplit(n_splits=3)
        try:
            cal = _make_calibrator(pipe, cv=cv)
            try:
                cal.fit(X, y, sample_weight=weights)
            except TypeError:
                cal.fit(X, y)
            final_model = cal
        except Exception:
            final_model = pipe
            _fit_with_weights(final_model, X, y, weights)
    else:
        _fit_with_weights(final_model, X, y, weights)

    dump(final_model, MODEL_PATH)

    metrics = {
        "rows": int(len(data)),
        "tickers_used": int(data["ticker"].nunique()),
        "cv_accuracy_avg": float(np.mean(accs)) if accs else None,
        "cv_auc_avg": float(np.mean(aucs)) if aucs else None,
        "trained_at": now_stamp(),
        "features": FEATURE_COLS,
        "model_kind": model_kind,
        "fast_mode": bool(fast_mode),
        "calibrated": bool(calibrate_probs),
        "max_workers": int(max_workers),
        "max_tickers": int(max_tickers) if max_tickers else None,
        "halflife_weeks": int(halflife_weeks),
    }

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(TRAIN_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return final_model, metrics


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load(MODEL_PATH)
    except Exception:
        return None


# ==========================================================
# Meta model (self-learning)
# ==========================================================
def load_meta_model():
    if not os.path.exists(META_MODEL_PATH):
        return None
    try:
        return load(META_MODEL_PATH)
    except Exception:
        return None

def train_meta_model_from_pred_log(min_rows: int = 80) -> dict:
    if not os.path.exists(PRED_LOG):
        raise RuntimeError("No predictions_log.csv found yet. Run Scan/Watchlist first.")
    pred = safe_read_log_csv(PRED_LOG)
    if pred is None or pred.empty:
        raise RuntimeError("predictions_log.csv is empty.")

    for c in ["outcome_known", "actual_outperform", "ai_prob"] + META_FEATURES:
        if c not in pred.columns:
            pred[c] = np.nan

    pred["outcome_known"] = pd.to_numeric(pred["outcome_known"], errors="coerce").fillna(0).astype(int)
    pred["actual_outperform"] = pd.to_numeric(pred["actual_outperform"], errors="coerce")

    ap = pd.to_numeric(pred.get("ai_prob", np.nan), errors="coerce")
    if "ai_prob_frac" not in pred.columns or pred["ai_prob_frac"].isna().all():
        pred["ai_prob_frac"] = ap.where(ap <= 1.5, ap / 100.0)

    done = pred[(pred["outcome_known"] == 1) & (pred["actual_outperform"].notna())].copy()
    if len(done) < int(min_rows):
        raise RuntimeError(f"Not enough known outcomes yet ({len(done)}/{min_rows}). Track more weeks.")

    X = done[META_FEATURES].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0).astype(float)
    y = done["actual_outperform"].astype(int)

    meta = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(loss="log_loss", max_iter=3000, tol=1e-3, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    meta.fit(X, y)
    dump(meta, META_MODEL_PATH)

    p = meta.predict_proba(X)[:, 1]
    yhat = (p >= 0.5).astype(int)
    acc = float(accuracy_score(y, yhat))
    try:
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = None

    metrics = {
        "rows": int(len(done)),
        "acc_in_sample": acc,
        "auc_in_sample": auc,
        "trained_at": now_stamp(),
        "features": META_FEATURES,
    }
    try:
        with open(META_TRAIN_METRICS, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    return metrics

def apply_meta_calibration(scored_df: pd.DataFrame) -> pd.DataFrame:
    meta = load_meta_model()
    if meta is None or scored_df is None or scored_df.empty:
        return scored_df
    df = scored_df.copy()
    if "AI_Prob_Calibrated_%" not in df.columns:
        df["AI_Prob_Calibrated_%"] = np.nan

    def _num(s, default=0.0):
        try:
            x = float(s)
            if np.isnan(x):
                return default
            return x
        except Exception:
            return default

    Xrows, idxs = [], []
    for idx, r in df.iterrows():
        ai = r.get("AI_Prob_Outperform_%", np.nan)
        if pd.isna(ai):
            continue
        ai_prob_frac = _num(ai, 0.0) / 100.0

        Xrows.append({
            "ai_prob_frac": ai_prob_frac,
            "thesis_score": _num(r.get("ThesisLiteScore", 0.0), 0.0),
            "rs1": _num(r.get("_rs1", 0.0), 0.0),
            "rs4": _num(r.get("_rs4", 0.0), 0.0),
            "vol_weekly": _num(r.get("_volw", 0.0), 0.0),
            "labor_exposure": _num(r.get("LaborExposure", 5.0), 5.0),
            "spy_above_ma200": _num(r.get("_spy_above_ma200", 1), 1.0),
            "spy_vol_4w": _num(r.get("_spy_vol_4w", 0.0), 0.0),
            "news_sent": _num(r.get("NewsSent_7d", 0.0), 0.0),
            "news_intensity": _num(r.get("NewsIntensity_7d", 0.0), 0.0),
        })
        idxs.append(idx)

    if not Xrows:
        return df

    X = pd.DataFrame(Xrows)[META_FEATURES].fillna(0.0).astype(float)
    try:
        p = meta.predict_proba(X)[:, 1] * 100.0
        for i, idx in enumerate(idxs):
            df.at[idx, "AI_Prob_Calibrated_%"] = round(float(p[i]), 1)
    except Exception:
        return df
    return df


# ==========================================================
# ThesisLite score + overlays
# ==========================================================
def thesis_lite_score(
    rs_4w: float,
    rs_1w: float,
    above_ma20: int,
    vol_weekly: float,
    labor_exposure: float,
    jan_score: float,
    news_sent: float = 0.0,
    news_intensity: float = 0.0,
    news_labor_hits: float = 0.0,
    news_prob_good: float | None = None,
    news_prob_best: float | None = None,
    spy_above_ma200: int = 1,
    spy_vol_4w: float = 0.0,

    # --- NEW (optional, backward compatible) ---
    rs_12w: float | None = None,
    mom_12w: float | None = None,
    dd_52w: float | None = None,          # negative when below 52w high
    ema20_gap: float | None = None,       # price/ema20 - 1
    bb_width_20: float | None = None,     # ~volatility structure
    rsi14_v: float | None = None,
    atrp14_v: float | None = None,
    vol_ratio_v: float | None = None,
    ticker_above_ma200: int | None = None,

    # manual overlays (watchlist)
    catalyst: float = 0.0,                # [-2..+2]
    labor_shock: int = 0,                 # 0/1
    volatile: int = 0,                    # 0/1
) -> float:
    # Helpers
    def _f(x, d=0.0):
        try:
            x = float(x)
            if np.isnan(x):
                return d
            return x
        except Exception:
            return d

    def _clip(x, lo, hi):
        return max(lo, min(hi, x))

    rs1 = _f(rs_1w)
    rs4 = _f(rs_4w)
    rs12 = _f(rs_12w, 0.0)
    mom12 = _f(mom_12w, 0.0)
    dd52 = _f(dd_52w, 0.0)
    egap = _f(ema20_gap, 0.0)
    bbw = _f(bb_width_20, 0.0)
    rsi = _f(rsi14_v, float("nan"))
    atrp = _f(atrp14_v, 0.0)
    vr = _f(vol_ratio_v, 1.0)

    score = 5.0

    # --- Core: reward sustained relative strength (magnitude matters) ---
    score += _clip(rs12 * 12.0, -2.0, 2.0)   # 12w RS is most important
    score += _clip(rs4  * 18.0, -1.6, 1.6)
    score += _clip(rs1  * 28.0, -1.2, 1.2)

    # --- Trend confirmation ---
    score += 0.9 * int(bool(above_ma20))
    if ticker_above_ma200 is not None:
        score += 0.6 * int(bool(ticker_above_ma200))

    # --- Momentum support (but not pure “chase”) ---
    score += _clip(mom12 * 3.5, -0.8, 0.8)

    # --- Structure: avoid stretched/unstable states ---
    # Drawdown: very deep dd can be either “value” or “broken”; treat extreme dd as risk
    # dd_52w is negative; penalize when below -25%
    score -= 1.2 * max(0.0, (-dd52 - 0.25) / 0.25)  # up to ~1.2 penalty

    # EMA gap: too far above EMA can be frothy; too far below can be weak
    score -= 0.8 * max(0.0, abs(egap) - 0.06) / 0.06  # penalize beyond ±6%

    # BB width: wide bands often mean noisy regime (harder to predict)
    score -= 0.9 * max(0.0, bbw - 0.18) / 0.18

    # RSI preference: “healthy” is ~45–65
    if not np.isnan(rsi):
        if 45 <= rsi <= 65:
            score += 0.6
        elif rsi > 75 or rsi < 25:
            score -= 0.8
        elif rsi > 70 or rsi < 30:
            score -= 0.4

    # ATR%: high ATR% = harder signal; penalize beyond ~3%
    score -= 0.8 * max(0.0, atrp - 0.03) / 0.03

    # Volume regime: rising volume can validate trend
    if vr >= 1.2:
        score += 0.25
    elif vr <= 0.8:
        score -= 0.15

    # --- “Labor exposure” (your thesis) ---
    le = _clip(_f(labor_exposure, 5.0), 0.0, 10.0)
    score -= 0.18 * le
    if int(labor_shock) == 1:
        score -= 0.6  # explicit shock flag = more uncertainty

    # --- Market regime overlay ---
    if int(spy_above_ma200) == 0:
        score -= 0.8  # risk-off: tighten
    sv = _f(spy_vol_4w, 0.0)
    score -= 1.0 * max(0.0, sv - 0.02) / 0.02  # vol > 2% weekly stdev penalized

    # --- News overlay (keep it modest) ---
    ns = _clip(_f(news_sent, 0.0), -1.0, 1.0)
    ni = _clip(_f(news_intensity, 0.0), 0.0, 1.0)
    score += 1.0 * ns * ni

    npg = _clip(_f(news_prob_good, 0.5), 0.0, 1.0)
    npb = _clip(_f(news_prob_best, npg), 0.0, 1.0)
    score += 1.1 * (npg - 0.5) * max(0.4, ni + 0.2)
    score += 0.6 * (npb - 0.5) * max(0.3, ni)

    # labor-themed news hits add risk (regardless of sign)
    if _f(news_labor_hits, 0.0) >= 3:
        score -= 0.25 * ni

    # --- Seasonality ---
    if datetime.now().month == 1:
        score += 0.7 * _clip(_f(jan_score, 0.0), -2.0, 2.0)

    # --- Manual “catalyst” overlay (watchlist) ---
    score += 0.35 * _clip(_f(catalyst, 0.0), -2.0, 2.0)

    # --- Volatile flag (watchlist) ---
    if int(volatile) == 1:
        score -= 0.35

    return clamp(score, 0, 10)


def action_from_scores(ai_prob: float | None, thesis_score: float) -> str:
    if ai_prob is None:
        if thesis_score >= 6.8:
            return "BUY"
        if thesis_score >= 6.0:
            return "HOLD"
        return "WATCH"
    if ai_prob >= 65 and thesis_score >= 6.6:
        return "BUY"
    if ai_prob >= 55 and thesis_score >= 6.0:
        return "HOLD"
    return "WATCH"

def confidence_label(ai_prob: float | None, thesis_score: float) -> str:
    if ai_prob is None:
        return "Medium" if thesis_score >= 6.5 else ("Low" if thesis_score < 6.0 else "Medium")
    if ai_prob >= 70 and thesis_score >= 6.8:
        return "High"
    if ai_prob >= 60 and thesis_score >= 6.3:
        return "Medium"
    return "Low"

def explain_why(
    ai_prob: float | None,
    thesis_score: float,
    rs1: float,
    rs4: float,
    above_ma20: int,
    vol_weekly: float,
    labor_exposure: float,
    jan_score: float,
    rsi_v: float,
    atrp_v: float,
    vol_ratio_v: float,
    baseline: str,
    news_heads: int = 0,
    news_sent: float = 0.0,
    news_intensity: float = 0.0,
    news_labor_hits: int = 0,
    news_prob_good: float | None = None,
    news_prob_best: float | None = None,
    spy_above_ma200: int = 1,
    spy_vol_4w: float = float("nan"),
    ai_prob_cal: float | None = None,
) -> str:
    parts = []
    parts.append(f"RS4w vs {baseline}: {'+' if rs4 >= 0 else ''}{rs4*100:.2f}%")
    parts.append(f"RS1w vs {baseline}: {'+' if rs1 >= 0 else ''}{rs1*100:.2f}%")
    parts.append("Trend: above MA20 ✅" if above_ma20 else "Trend: below MA20 ⚠️")

    if not (vol_weekly is None or np.isnan(vol_weekly)):
        parts.append(f"Weekly vol: {vol_weekly*100:.1f}%")
    if not np.isnan(spy_vol_4w):
        parts.append(f"Market vol4w: {spy_vol_4w*100:.2f}%")
    parts.append("Regime: risk-on ✅" if int(spy_above_ma200) == 1 else "Regime: risk-off ⚠️")

    if not np.isnan(rsi_v):
        parts.append(f"RSI14: {rsi_v:.0f}")
    if not np.isnan(atrp_v):
        parts.append(f"ATR%14: {atrp_v*100:.2f}%")
    if not np.isnan(vol_ratio_v):
        parts.append(f"Vol regime (20/100): {vol_ratio_v:.2f}x")

    if news_heads and news_intensity > 0:
        parts.append(f"News7d: {news_heads} | sent={news_sent:+.2f} | int={news_intensity:.2f} | laborHits={news_labor_hits}")
    if news_prob_good is not None and not np.isnan(news_prob_good):
        suffix = ""
        if news_prob_best is not None and not np.isnan(news_prob_best):
            suffix = f" (best {news_prob_best:.2f})"
        parts.append(f"News model: P(good)={news_prob_good:.2f}{suffix}")

    parts.append(f"Labor exposure: {labor_exposure:.1f}/10")
    if datetime.now().month == 1:
        parts.append(f"January effect: {jan_score:+.0f}")

    if ai_prob is not None:
        parts.append(f"AI: {ai_prob:.1f}% to beat {baseline} next week")
    if ai_prob_cal is not None:
        parts.append(f"AI(cal): {ai_prob_cal:.1f}% (self-learned)")

    parts.append(f"ThesisLiteScore: {thesis_score:.2f}/10")
    return " | ".join(parts)

def show_results_legend():
    st.markdown(
        """
**What the app is doing (simple)**  
- Pulls **end-of-day** price history for each ticker + baseline (e.g., SPY).  
- Builds weekly signals: **Relative Strength vs baseline**, trend (MA20), volatility, RSI/ATR.  
- If a model exists: **AI_Prob_Outperform_%** = estimated chance to beat baseline **next week**.  
- Combines signals into **ThesisLiteScore (0–10)** and an **Action**.

**Self-learning calibration (meta model)**  
- After you track outcomes, it can learn **AI_Prob_Calibrated_%** = probability your call is right, based on your own history.

**News overlay**  
- Google News RSS headline flow (last ~7 days) boosts/penalizes ThesisLiteScore for top candidates.

**Market regime overlay**  
- If baseline is **below MA200** (risk-off), ThesisLiteScore is penalized.  
- If baseline 4-week vol is high, ThesisLiteScore is penalized.

**Important**  
If training metrics are ~0.50 accuracy / ~0.50 AUC, the AI is basically coin-flip early on.
        """
    )


# ==========================================================
# Universe + watchlist management
# ==========================================================
def load_universe() -> pd.DataFrame:
    df = load_universe_df(UNIVERSE_CSV).copy()
    df["labor_exposure"] = pd.to_numeric(df["labor_exposure"], errors="coerce").fillna(5).clip(0, 10)
    df["jan_score"] = pd.to_numeric(df["jan_score"], errors="coerce").fillna(0).clip(-2, 2)
    df = df.drop_duplicates(subset=["ticker"]).sort_values("ticker")
    return df

def save_universe(df: pd.DataFrame):
    save_universe_df(df, UNIVERSE_CSV)

def load_watchlist() -> pd.DataFrame:
    cols = ["ticker","labor_exposure","jan_score","catalyst","labor_shock","entry_price","volatile"]
    df = safe_read_csv(WATCHLIST_CSV, cols)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"].str.len() > 0]
    df["labor_exposure"] = pd.to_numeric(df["labor_exposure"], errors="coerce").fillna(5).clip(0,10)
    df["jan_score"] = pd.to_numeric(df["jan_score"], errors="coerce").fillna(0).clip(-2,2)
    df["catalyst"] = pd.to_numeric(df["catalyst"], errors="coerce").fillna(0).clip(-2,2)
    df["labor_shock"] = pd.to_numeric(df["labor_shock"], errors="coerce").fillna(0).clip(0,1)
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["volatile"] = pd.to_numeric(df["volatile"], errors="coerce").fillna(0).clip(0,1)
    df = df.drop_duplicates(subset=["ticker"]).sort_values("ticker")
    return df

def save_watchlist(df: pd.DataFrame):
    df2 = df.copy()[["ticker","labor_exposure","jan_score","catalyst","labor_shock","entry_price","volatile"]]
    df2.to_csv(WATCHLIST_CSV, index=False)

def auto_grow_universe(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
    existing = set(df["ticker"].astype(str).str.upper().tolist())
    add = []
    for t in AUTO_SEED_TICKERS:
        if t.upper() not in existing:
            add.append(t.upper())
        if len(existing) + len(add) >= target_size:
            break
    if not add:
        return df
    add_df = pd.DataFrame({"ticker": add, "name": "", "labor_exposure": 5, "jan_score": 0, "company_name": "", "aliases": "", "stooq_symbol": ""})
    out = pd.concat([df, add_df], ignore_index=True)
    out = out.drop_duplicates(subset=["ticker"]).sort_values("ticker")
    return out

def _load_listings_symbols(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    candidates = ["Symbol", "symbol", "Ticker", "ticker", "ACT Symbol", "Security Symbol"]
    sym_col = None
    for c in candidates:
        if c in df.columns:
            sym_col = c
            break
    if sym_col is None:
        sym_col = df.columns[0]

    syms = df[sym_col].astype(str).str.upper().str.strip().tolist()
    out = []
    for s in syms:
        if not s or s == "NAN":
            continue
        if len(s) > 12:
            continue
        if any(ch in s for ch in [" ", "/", "\\"]):
            continue
        out.append(s)

    seen, final = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            final.append(s)
    return final

def auto_grow_universe_from_listings(df: pd.DataFrame, target_size: int, listings_path: str = LISTINGS_CSV) -> pd.DataFrame:
    out = df.copy()
    if len(out) >= int(target_size):
        return out

    existing = set(out["ticker"].astype(str).str.upper().tolist())
    add = []
    syms = _load_listings_symbols(listings_path)
    for s in syms:
        if s not in existing:
            add.append(s)
        if len(existing) + len(add) >= int(target_size):
            break

    if add:
        add_df = pd.DataFrame({"ticker": add, "name": "", "labor_exposure": 5, "jan_score": 0, "company_name": "", "aliases": "", "stooq_symbol": ""})
        out = pd.concat([out, add_df], ignore_index=True).drop_duplicates(subset=["ticker"]).sort_values("ticker")

    if len(out) < int(target_size):
        out = auto_grow_universe(out, int(target_size))
    return out


# ==========================================================
# Scoring (single ticker)
# ==========================================================
def score_one_week(
    ticker: str,
    baseline: str,
    labor_exposure: float,
    jan_score: float,
    model,
    *,
    force_refresh: bool = False,
    use_news: bool = False,
    news_days: int = 7,
    bctx: BaselineContext | None = None,
    catalyst: float = 0.0,
    labor_shock: int = 0,
    volatile: int = 0,
) -> dict:

    ticker = str(ticker).upper().strip()
    baseline = str(baseline).upper().strip()

    if bctx is None:
        _, bctx = baseline_context(baseline, force_refresh=force_refresh)

    df = cache_price(ticker, force_refresh=force_refresh)
    if df.empty or len(df) < MIN_DAILY_ROWS_SCORE:
        raise RuntimeError(f"Not enough daily history for {ticker} (got {len(df)}, need {MIN_DAILY_ROWS_SCORE})")

    w = friday_close_series(df)
    common = w.index.intersection(bctx.weekly.index)
    w = w.loc[common]
    b = bctx.weekly.loc[common]
    if len(w) < MIN_WEEKLY_POINTS_SCORE:
        raise RuntimeError("Not enough weekly data.")

    # weekly returns + relative strength
    r1 = (w.iloc[-1] / w.iloc[-2]) - 1.0
    r4 = (w.iloc[-1] / w.iloc[-5]) - 1.0 if len(w) >= 5 else 0.0
    b1 = (b.iloc[-1] / b.iloc[-2]) - 1.0
    b4 = (b.iloc[-1] / b.iloc[-5]) - 1.0 if len(b) >= 5 else 0.0
    rs1 = r1 - b1
    rs4 = r4 - b4

    close = df["close"]
    ma20 = rolling_ma(close, 20)
    ma50 = rolling_ma(close, 50) if len(df) >= 60 else pd.Series(index=df.index, dtype=float)
    ma200 = rolling_ma(close, 200) if len(df) >= 220 else pd.Series(index=df.index, dtype=float)
    volw = weekly_vol_from_daily(close)

    rsi_v = rsi14(close, 14).iloc[-1] if len(close) >= 20 else float("nan")
    atrp_v = (atr14(df, 14).iloc[-1] / close.iloc[-1]) if len(close) >= 20 else float("nan")

    vol_ratio_v = float("nan")
    if "volume" in df.columns and len(df) >= 120:
        v = pd.to_numeric(df["volume"], errors="coerce")
        denom = v.tail(100).mean()
        vr = (v.tail(20).mean() / denom) if (denom is not None and denom != 0 and not np.isnan(denom)) else np.nan
        try:
            vol_ratio_v = float(vr)
        except Exception:
            vol_ratio_v = float("nan")

    ma20_last = float(ma20.iloc[-1]) if (len(ma20) and not np.isnan(ma20.iloc[-1])) else float("nan")
    ma50_last = float(ma50.iloc[-1]) if (len(ma50) and not np.isnan(ma50.iloc[-1])) else float("nan")
    ma200_last = float(ma200.iloc[-1]) if (len(ma200) and not np.isnan(ma200.iloc[-1])) else float("nan")
    vol_last = float(volw.iloc[-1]) if (len(volw) and not np.isnan(volw.iloc[-1])) else float("nan")

    above_ma20 = int(w.iloc[-1] > ma20_last) if not math.isnan(ma20_last) else 0
    ma_stack = int(ma20_last > ma50_last) if (not math.isnan(ma20_last) and not math.isnan(ma50_last)) else 0
    ma_gap = float((ma20_last / ma50_last) - 1.0) if (not math.isnan(ma20_last) and not math.isnan(ma50_last)) else 0.0

    ticker_above_ma200 = int(w.iloc[-1] > ma200_last) if not math.isnan(ma200_last) else 0
    ma20_200_gap = float((ma20_last / ma200_last) - 1.0) if (not math.isnan(ma20_last) and not math.isnan(ma200_last)) else 0.0

    last_dt = common[-1]
    spy_above_ma20 = int(b.loc[last_dt] > bctx.ma20_w.loc[last_dt]) if last_dt in bctx.ma20_w.index else 0
    spy_above_ma200 = int(b.loc[last_dt] > bctx.ma200_w.loc[last_dt]) if last_dt in bctx.ma200_w.index else 0
    spy_vol_4w = float(bctx.vol4w_w.loc[last_dt]) if last_dt in bctx.vol4w_w.index else float("nan")

    # --- EXTRA STRUCTURE FEATURES (these make ThesisLiteScore much stronger) ---
    rs12_now = 0.0
    mom12_now = 0.0
    if len(w) >= 13:
        r12_now = (w.iloc[-1] / w.iloc[-13]) - 1.0
        b12_now = (b.iloc[-1] / b.iloc[-13]) - 1.0
        rs12_now = float(r12_now - b12_now)
        mom12_now = float(r12_now)

    ema20_gap_now = 0.0
    dd_52w_now = 0.0
    bb_width_now = 0.0

    if len(close) >= 25:
        ema20_d = close.ewm(span=20, adjust=False).mean()
        if not np.isnan(ema20_d.iloc[-1]) and ema20_d.iloc[-1] != 0:
            ema20_gap_now = float((close.iloc[-1] / ema20_d.iloc[-1]) - 1.0)

        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        if not np.isnan(bb_mid.iloc[-1]) and bb_mid.iloc[-1] != 0:
            bb_width_now = float(4.0 * bb_std.iloc[-1] / bb_mid.iloc[-1])

    if len(close) >= 260:
        roll_max_252 = close.rolling(252, min_periods=252).max()
        if not np.isnan(roll_max_252.iloc[-1]) and roll_max_252.iloc[-1] != 0:
            dd_52w_now = float((close.iloc[-1] / roll_max_252.iloc[-1]) - 1.0)

    # --- AI probability (only if model exists) ---
    ai_prob = None
    if model is not None:
        now = datetime.now()
        base_row = {
            "ret_1w": float(r1),
            "rs_1w": float(rs1),
            "rs_4w": float(rs4),
            "mom_4w": float(r4),
            "above_ma20": float(above_ma20),
            "ma_stack": float(ma_stack),
            "ma_gap": float(ma_gap),
            "vol_weekly": float(0.0 if math.isnan(vol_last) else vol_last),
            "spy_rs_4w": float(b4),
            "spy_above_ma20": float(spy_above_ma20),
            "spy_above_ma200": float(spy_above_ma200),
            "spy_vol_4w": float(0.0 if np.isnan(spy_vol_4w) else spy_vol_4w),
            "ticker_above_ma200": float(ticker_above_ma200),
            "ma20_200_gap": float(ma20_200_gap),
            "labor_exposure": float(labor_exposure),
            "jan_score": float(jan_score),
            "is_january": float(1 if now.month == 1 else 0),
            "month": float(now.month),
            "rsi14": float(0.0 if np.isnan(rsi_v) else rsi_v),
            "atrp14": float(0.0 if np.isnan(atrp_v) else atrp_v),
            "vol_ratio": float(0.0 if np.isnan(vol_ratio_v) else vol_ratio_v),

            # structure features (already in FEATURE_COLS)
            "rs_12w": float(rs12_now),
            "mom_12w": float(mom12_now),
            "dd_52w": float(dd_52w_now),
            "ema20_gap": float(ema20_gap_now),
            "bb_width_20": float(bb_width_now),
            "feature_hash": fh if "fh" in locals() else "",
        }

        feature_cols = get_model_feature_cols()
        feat = pd.DataFrame([base_row])
        for c in feature_cols:
            if c not in feat.columns:
                feat[c] = 0.0
        feat = feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

        try:
            fh = feature_hash(feat.to_numpy().ravel())
            ai_prob = float(model.predict_proba(feat)[:, 1][0]) * 100.0
        except Exception:
            ai_prob = None
            fh = ""

    # --- news ---
    news_heads = 0
    news_sent = 0.0
    news_int = 0.0
    news_labor = 0
    news_prob_mean = np.nan
    news_prob_max = np.nan
    if use_news:
        company_name = ""
        try:
            if "universe_df" in globals() and universe_df is not None and not universe_df.empty:
                hit = universe_df.loc[universe_df["ticker"] == ticker, "name"]
                company_name = str(hit.iloc[0]) if len(hit) else ""
        except Exception:
            company_name = ""

        nf = get_news_features(ticker, company_name=company_name, days=int(news_days), max_items=30)
        news_heads = int(nf["headlines"])
        news_sent = float(nf["sent"])
        news_int = float(nf["intensity"])
        news_labor = int(nf["labor_hits"])
        news_prob_mean = float(nf.get("prob_good_mean", np.nan))
        news_prob_max = float(nf.get("prob_good_max", np.nan))

    # --- STRONGER thesis score: PASS THE STRUCTURE FEATURES + WATCHLIST OVERLAYS ---
    thesis_score = thesis_lite_score(
        rs_4w=rs4,
        rs_1w=rs1,
        above_ma20=above_ma20,
        vol_weekly=vol_last,
        labor_exposure=labor_exposure,
        jan_score=jan_score,
        news_sent=news_sent,
        news_intensity=news_int,
        news_labor_hits=news_labor,
        news_prob_good=news_prob_mean,
        news_prob_best=news_prob_max,
        spy_above_ma200=spy_above_ma200,
        spy_vol_4w=spy_vol_4w,

        rs_12w=rs12_now,
        mom_12w=mom12_now,
        dd_52w=dd_52w_now,
        ema20_gap=ema20_gap_now,
        bb_width_20=bb_width_now,
        rsi14_v=rsi_v,
        atrp14_v=atrp_v,
        vol_ratio_v=vol_ratio_v,
        ticker_above_ma200=ticker_above_ma200,

        catalyst=catalyst,
        labor_shock=labor_shock,
        volatile=volatile,
    )

    action = action_from_scores(ai_prob, thesis_score)
    conf = confidence_label(ai_prob, thesis_score)

    avgv = avg_volume_20d(df)

    return {
        "Ticker": ticker,
        "Action": action,
        "Confidence": conf,
        "AI_Prob_Outperform_%": None if ai_prob is None else round(ai_prob, 1),
        "AI_Prob_Calibrated_%": np.nan,
        "ThesisLiteScore": round(thesis_score, 2),
        "RS_1w_vs_SPY_%": round(float(rs1) * 100.0, 2),
        "RS_4w_vs_SPY_%": round(float(rs4) * 100.0, 2),
        "WeeklyVol_%": None if math.isnan(vol_last) else round(float(vol_last) * 100.0, 2),
        "AvgVol_20d": None if math.isnan(avgv) else int(avgv),
        "LastPrice": round(float(df["close"].iloc[-1]), 2),
        "JanScore": int(jan_score),
        "LaborExposure": round(float(labor_exposure), 1),

        "NewsHeadlines_7d": int(news_heads) if use_news else np.nan,
        "NewsSent_7d": round(float(news_sent), 3) if use_news else np.nan,
        "NewsIntensity_7d": round(float(news_int), 3) if use_news else np.nan,
        "NewsLaborHits_7d": int(news_labor) if use_news else np.nan,
        "NewsProbGood_Mean": round(float(news_prob_mean), 3) if (use_news and not pd.isna(news_prob_mean)) else np.nan,
        "NewsProbGood_Max": round(float(news_prob_max), 3) if (use_news and not pd.isna(news_prob_max)) else np.nan,

        # debug internals used later
        "_rs1": float(rs1),
        "_rs4": float(rs4),
        "_rs12": float(rs12_now),
        "_mom12": float(mom12_now),
        "_dd52": float(dd_52w_now),
        "_ema20_gap": float(ema20_gap_now),
        "_bb_width": float(bb_width_now),

        "_volw": float(vol_last) if not math.isnan(vol_last) else np.nan,
        "_above_ma20": int(above_ma20),
        "_ma_stack": int(ma_stack),
        "_ma_gap": float(ma_gap),
        "_ticker_above_ma200": int(ticker_above_ma200),
        "_ma20_200_gap": float(ma20_200_gap),

        "_spy_above_ma20": int(spy_above_ma20),
        "_spy_above_ma200": int(spy_above_ma200),
        "_spy_vol_4w": float(spy_vol_4w) if not np.isnan(spy_vol_4w) else np.nan,

        "_rsi14": float(rsi_v) if not np.isnan(rsi_v) else np.nan,
        "_atrp14": float(atrp_v) if not np.isnan(atrp_v) else np.nan,
        "_vol_ratio": float(vol_ratio_v) if not np.isnan(vol_ratio_v) else np.nan,

        "_news_prob_mean": float(news_prob_mean) if not pd.isna(news_prob_mean) else np.nan,
        "_news_prob_max": float(news_prob_max) if not pd.isna(news_prob_max) else np.nan,

        "Why": "",
    }



# ==========================================================
# Prediction tracking
# ==========================================================
PRED_COLS = [
    "timestamp","source","baseline","ticker","asof_date","target_date",
    "action","ai_prob","thesis_score","pred_outperform",
    "outcome_known","actual_outperform","hit",
    "ai_prob_frac","rs1","rs4","vol_weekly","labor_exposure","spy_above_ma200","spy_vol_4w","news_sent","news_intensity",
    "news_prob_good_mean","news_prob_good_max",
]

def log_predictions(source: str, baseline: str, scored_df: pd.DataFrame):
    if scored_df is None or scored_df.empty:
        return
    spy = cache_price(baseline)
    if spy.empty:
        return
    b_weekly = friday_close_series(spy)
    if len(b_weekly) < 3:
        return
    asof = b_weekly.index[-1].date()
    target = (b_weekly.index[-1] + timedelta(days=7)).date()

    rows = []
    for _, r in scored_df.iterrows():
        ai = r.get("AI_Prob_Outperform_%", np.nan)
        ts = r.get("ThesisLiteScore", np.nan)
        action = r.get("Action", "")
        pred_out = 1 if str(action).upper() == "BUY" else 0

        rs1 = r.get("_rs1", np.nan)
        rs4 = r.get("_rs4", np.nan)
        volw = r.get("_volw", np.nan)
        spy_above = r.get("_spy_above_ma200", np.nan)
        spy_vol = r.get("_spy_vol_4w", np.nan)
        le = r.get("LaborExposure", np.nan)
        news_sent = r.get("NewsSent_7d", 0.0)
        news_int = r.get("NewsIntensity_7d", 0.0)
        news_prob_mean = r.get("NewsProbGood_Mean", np.nan)
        news_prob_max = r.get("NewsProbGood_Max", np.nan)

        try:
            ai_prob_frac = float(ai) / 100.0 if not pd.isna(ai) else np.nan
        except Exception:
            ai_prob_frac = np.nan

        rows.append({
            "timestamp": now_stamp(),
            "source": source,
            "baseline": baseline,
            "ticker": r["Ticker"],
            "asof_date": str(asof),
            "target_date": str(target),
            "action": action,
            "ai_prob": ai,
            "thesis_score": ts,
            "pred_outperform": pred_out,
            "outcome_known": 0,
            "actual_outperform": np.nan,
            "hit": np.nan,

            "ai_prob_frac": ai_prob_frac,
            "rs1": rs1,
            "rs4": rs4,
            "vol_weekly": volw,
            "labor_exposure": le,
            "spy_above_ma200": spy_above,
            "spy_vol_4w": spy_vol,
            "news_sent": news_sent if not pd.isna(news_sent) else 0.0,
            "news_intensity": news_int if not pd.isna(news_int) else 0.0,
            "news_prob_good_mean": news_prob_mean if not pd.isna(news_prob_mean) else np.nan,
            "news_prob_good_max": news_prob_max if not pd.isna(news_prob_max) else np.nan,
        })

    df = pd.DataFrame(rows)
    for c in PRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[PRED_COLS]

    write_header = not os.path.exists(PRED_LOG)
    df.to_csv(PRED_LOG, mode="a", header=write_header, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")

def update_prediction_outcomes(
    interval: str = DEFAULT_BAR_INTERVAL,
    tolerance_minutes: int = 20,
) -> pd.DataFrame:
    if not os.path.exists(PRED_LOG):
        return pd.DataFrame()

    pred = safe_read_log_csv(PRED_LOG)
    if pred.empty:
        return pred

    pred = ensure_pred_log_columns(pred)

    pred["outcome_known"] = pd.to_numeric(pred.get("outcome_known", 0), errors="coerce").fillna(0).astype(int)
    pred["hit"] = pd.to_numeric(pred.get("hit", np.nan), errors="coerce")

    now = utcnow_floor()
    # rows eligible to score
    todo = pred[
        (pred["outcome_known"] != 1)
        & pred["target_ts"].notna()
        & (pred["target_ts"] <= pd.Timestamp(now))
    ].copy()

    if todo.empty:
        return pred

    # Batch per symbol to reduce API calls
    # (we fetch enough recent 1min bars to cover the oldest target)
    sym_col = "ticker" if "ticker" in pred.columns else ("symbol" if "symbol" in pred.columns else None)
    if sym_col is None:
        raise ValueError("predictions_log.csv must contain a 'ticker' (or 'symbol') column to score outcomes.")

    for symbol, block in todo.groupby(sym_col):
        # determine how many bars to request
        oldest = block["asof_ts"].min()
        newest = block["target_ts"].max()
        if pd.isna(oldest) or pd.isna(newest):
            continue

        # Rough bars needed: minutes between now and oldest, capped
        mins_back = int(max(120, min(6000, (pd.Timestamp(now) - oldest).total_seconds() / 60.0 + 60)))
        dfbars = twelvedata_time_series(str(symbol), interval=interval, outputsize=mins_back, apikey=TWELVEDATA_KEY)

        for idx, row in block.iterrows():
            asof_ts = parse_dt_any(row.get("asof_ts"))
            target_ts = parse_dt_any(row.get("target_ts"))
            if asof_ts is None or target_ts is None:
                continue

            # use logged asof_price if you already store it; otherwise infer from nearest bar
            asof_price = pd.to_numeric(row.get("asof_price", np.nan), errors="coerce")
            if pd.isna(asof_price):
                asof_price = price_nearest(dfbars, asof_ts, tolerance_minutes=tolerance_minutes)

            target_price = price_nearest(dfbars, target_ts, tolerance_minutes=tolerance_minutes)

            if pd.isna(asof_price) or pd.isna(target_price):
                # can't score this one yet
                continue

            actual_up = 1 if target_price > asof_price else 0
            pred.loc[idx, "actual_outperform"] = int(actual_up)   # <-- (5) this is the missing column your report uses


            prob_up = pd.to_numeric(row.get("prob_up", row.get("pred_prob_up", np.nan)), errors="coerce")
            if not pd.isna(prob_up):
                pred_label = 1 if float(prob_up) >= 0.5 else 0
            else:
                pred_label = int(pd.to_numeric(row.get("pred_label", 1), errors="coerce") or 1)

            hit = 1 if pred_label == actual_up else 0
            ret = (target_price - asof_price) / asof_price if asof_price else 0.0

            pred.loc[idx, "outcome_known"] = 1
            pred.loc[idx, "hit"] = float(hit)
            pred.loc[idx, "actual_up"] = int(actual_up)
            pred.loc[idx, "asof_price_scored"] = float(asof_price)
            pred.loc[idx, "target_price"] = float(target_price)
            pred.loc[idx, "actual_return"] = float(ret)
            pred.loc[idx, "scored_at"] = now.isoformat(sep=" ")

    pred.to_csv(PRED_LOG, index=False)
    return pred


def _prob_to_frac(x) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        v = float(x)
        # accept either 0..1 or 0..100
        if v > 1.5:
            v = v / 100.0
        return float(clamp(v, 0.0, 1.0))
    except Exception:
        return float("nan")

def compute_prediction_report(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      done_df: cleaned outcomes-only df
      summary: dict metrics
      cal_table: calibration bins table
      weekly_table: per-target-week performance table
    """
    if pred is None or pred.empty:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    df = pred.copy()

    # ensure columns exist
    for c in ["outcome_known","actual_outperform","pred_outperform","ai_prob","ai_prob_frac","hit","target_date","asof_date","ticker","baseline"]:
        if c not in df.columns:
            df[c] = np.nan

    df["outcome_known"] = pd.to_numeric(df["outcome_known"], errors="coerce").fillna(0).astype(int)
    df["actual_outperform"] = pd.to_numeric(df["actual_outperform"], errors="coerce")
    df["pred_outperform"] = pd.to_numeric(df["pred_outperform"], errors="coerce").fillna(0).astype(int)

    done = df[(df["outcome_known"] == 1) & (df["actual_outperform"].notna())].copy()
    if done.empty:
        return done, {}, pd.DataFrame(), pd.DataFrame()

    done["y"] = done["actual_outperform"].astype(int)
    done["pred"] = done["pred_outperform"].astype(int)

    # probability (prefer ai_prob_frac if present)
    if "ai_prob_frac" in done.columns and done["ai_prob_frac"].notna().any():
        done["p"] = pd.to_numeric(done["ai_prob_frac"], errors="coerce").apply(_prob_to_frac)
    else:
        done["p"] = pd.to_numeric(done["ai_prob"], errors="coerce").apply(_prob_to_frac)

    done["hit_calc"] = (done["pred"] == done["y"]).astype(int)

    # confusion
    tp = int(((done["pred"] == 1) & (done["y"] == 1)).sum())
    tn = int(((done["pred"] == 0) & (done["y"] == 0)).sum())
    fp = int(((done["pred"] == 1) & (done["y"] == 0)).sum())
    fn = int(((done["pred"] == 0) & (done["y"] == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")  # recall on wins
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")  # recall on losses
    bal_acc = (tpr + tnr) / 2.0 if (not np.isnan(tpr) and not np.isnan(tnr)) else float("nan")

    acc = float(done["hit_calc"].mean())
    base_rate = float(done["y"].mean())
    buy_rate = float((done["pred"] == 1).mean())
    buy_precision = float(done.loc[done["pred"] == 1, "y"].mean()) if (done["pred"] == 1).any() else float("nan")
    watch_winrate = float(done.loc[done["pred"] == 0, "y"].mean()) if (done["pred"] == 0).any() else float("nan")

    # probabilistic scoring (only where p is available)
    p_df = done[done["p"].notna()].copy()
    auc = float("nan")
    brier = float("nan")
    logloss = float("nan")
    if not p_df.empty:
        yy = p_df["y"].astype(int).values
        pp = p_df["p"].astype(float).values
        brier = float(np.mean((pp - yy) ** 2))
        eps = 1e-15
        logloss = float(-np.mean(yy * np.log(pp + eps) + (1 - yy) * np.log(1 - pp + eps)))
        if len(np.unique(yy)) >= 2:
            try:
                auc = float(roc_auc_score(yy, pp))
            except Exception:
                auc = float("nan")

    # calibration bins
    bins = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]
    p_df["p_bin"] = pd.cut(p_df["p"], bins=bins, include_lowest=True)
    cal = (
        p_df.groupby("p_bin", dropna=True)
            .agg(
                n=("y", "size"),
                mean_p=("p", "mean"),
                win_rate=("y", "mean"),
                buy_rate=("pred", "mean"),
            )
            .reset_index()
    )

    # per target week performance
    done["target_date"] = pd.to_datetime(done["target_date"], errors="coerce")
    wk = (
        done.dropna(subset=["target_date"])
            .groupby("target_date")
            .agg(
                n=("y", "size"),
                hit_rate=("hit_calc", "mean"),
                win_rate=("y", "mean"),
                buy_rate=("pred", "mean"),
            )
            .reset_index()
            .sort_values("target_date")
    )

    summary = {
        "n_outcomes": int(len(done)),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "base_win_rate": base_rate,
        "buy_rate": buy_rate,
        "buy_precision": buy_precision,
        "watch_win_rate": watch_winrate,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "auc": auc,
        "brier": brier,
        "logloss": logloss,
    }

    return done, summary, cal, wk


# ==========================================================
# UI bootstrap
# ==========================================================
ensure_dirs()
st.set_page_config(page_title="Market AI Dashboard", layout="wide")
st.title("📊 Market AI Dashboard (Learning + Monitoring)")
st.caption("Daily (end-of-day) data by default. Optional live snapshots via Twelve Data. Not financial advice.")

# --- Load universe + auto-fill company names ---
universe_df = load_universe_df(UNIVERSE_CSV)

with st.sidebar:
    if st.button("Auto-fill missing company names (US SEC)"):
        universe_df = autofill_missing_company_names(universe_df)
        save_universe_df(universe_df, UNIVERSE_CSV)
        st.success("Updated universe.csv with company names where available.")

name_map = dict(zip(universe_df["ticker"], universe_df["name"]))

watch_tickers = st.sidebar.multiselect(
    "Watch tickers",
    options=universe_df["ticker"].tolist(),
    default=universe_df["ticker"].tolist()[:10],
    format_func=lambda t: ticker_display(t, name_map),
)

baseline = st.sidebar.text_input(
    "Baseline ticker",
    value="SPY",
    help="Everything is scored relative to this. The AI target is: beat this next week.",
).strip().upper()

st.sidebar.divider()
debug_mode = st.sidebar.toggle("🛠 Debug mode", value=False, help="Shows full stack traces in the UI + logs errors to logs/app_debug.log")
logger = _setup_logger(bool(debug_mode))
dbg_event("app_start", baseline=baseline)

st.sidebar.subheader("🤖 Hands-off mode (auto-run)")
hands_off = st.sidebar.toggle(
    "Enable hands-off mode",
    value=False,
    help="On each refresh/run, the app will: grow universe, update outcomes, retrain meta weekly, retrain base monthly.",
)
hands_off_target_universe = st.sidebar.number_input(
    "Hands-off target universe size",
    min_value=20, max_value=2000, value=120, step=10,
    help="If universe.csv is smaller than this, it will auto-fill (listings.csv first, then seed list).",
)
st.sidebar.caption(f"listings.csv: {'FOUND ✅' if os.path.exists(LISTINGS_CSV) else 'not found (optional)'}")

st.sidebar.divider()
st.sidebar.subheader("Filters (Scan)")
min_ai = st.sidebar.slider("Min AI Prob %", 0, 90, 55, 1, help="Filters scan results by AI probability.")
max_vol = st.sidebar.slider("Max weekly vol %", 1, 50, 18, 1, help="Filters out very volatile names.")
min_avg_vol = st.sidebar.number_input("Min avg volume (20d)", min_value=0, value=0, step=100000, help="Liquidity filter. 0 disables.")
relax_filters = st.sidebar.toggle("Relax filters if results too small", value=True, help="If strict filters return too few tickers, show unfiltered list instead.")

st.sidebar.divider()
if st.sidebar.button("Clear cache (fix stale data)"):
    st.cache_data.clear()
    st.success("Cache cleared. Refresh / rerun.")

st.sidebar.divider()
st.sidebar.subheader("Auto-refresh (monitoring-lite)")
auto_refresh_user = st.sidebar.toggle(
    "Enable auto-refresh",
    value=False,
    help="Re-runs auto-scan/live views on a timer.",
    disabled=hands_off,
)
refresh_mins = st.sidebar.number_input("Refresh every (minutes)", 1, 180, 10, 1, help="Longer is safer for free RSS/APIs.")
debounce = st.sidebar.toggle("Pause while interacting", value=True, help="Avoid refresh while you're clicking/typing.")

auto_refresh = bool(hands_off or auto_refresh_user)

refresh_count = None
if auto_refresh:
    refresh_count = st_autorefresh(interval=int(refresh_mins * 60 * 1000), key="auto_refresh", debounce=debounce)
st.sidebar.caption(f"Refresh: {'ON' if auto_refresh else 'OFF'} • count={refresh_count} • {now_stamp()}")

st.sidebar.divider()
st.sidebar.subheader("Live data (optional)")
td_key = st.sidebar.text_input(
    "Twelve Data API key",
    value=get_td_key_from_env(),
    type="password",
    help="Optional. Enables intraday tabs. Free plan has limits.",
).strip()
st.sidebar.caption("Free plan is limited credits; keep refresh slow & watch few tickers.")

st.sidebar.divider()
st.sidebar.subheader("🧠 Local LLM (Ollama) — optional")
ollama_on = st.sidebar.toggle("Enable Ollama", value=False)
ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
ollama_model = st.sidebar.text_input("Ollama model", value="llama3.1:8b")

ollama_ok = False
if ollama_on:
    v = ollama_version(ollama_url)
    if v:
        st.sidebar.success(f"Ollama OK (v{v})")
        ollama_ok = True
    else:
        st.sidebar.warning("Ollama not reachable. Is it running?")

tabs = st.tabs([
    "Train", "Scan Universe", "Watchlist", "AI Predictions (New)", "Live", "Day Trading (experimental)",
    "News", "Manage Universe", "Tracking", "Chat", "Logs / Files"
])

with st.expander("📌 What is this doing and what do results mean?", expanded=True):
    show_results_legend()


# ==========================================================
# Hands-off runner
# ==========================================================
def run_hands_off_tasks():
    if not hands_off:
        return
    state = _get_auto_state()

    # 1) Grow universe
    try:
        uni = load_universe()
        if len(uni) < int(hands_off_target_universe):
            uni2 = auto_grow_universe_from_listings(uni, int(hands_off_target_universe), listings_path=LISTINGS_CSV)
            if len(uni2) > len(uni):
                save_universe(uni2)
    except Exception:
        pass

    # 2) Update outcomes
    if _task_due(state, "update_outcomes", AUTO_OUTCOME_CHECK_HOURS):
        try:
            update_prediction_outcomes()
        except Exception:
            pass
        _set_last_run(state, "update_outcomes", datetime.now())

    # 3) Train meta weekly
    if _task_due(state, "train_meta", AUTO_META_TRAIN_HOURS):
        try:
            train_meta_model_from_pred_log(min_rows=80)
        except Exception:
            pass
        _set_last_run(state, "train_meta", datetime.now())

    # 4) Train base monthly (or if missing)
    base_missing = not os.path.exists(MODEL_PATH)
    if base_missing or _task_due(state, "train_base", AUTO_BASE_TRAIN_HOURS):
        try:
            uni = load_universe()
            train_model(uni, baseline, force_refresh=False)
        except Exception:
            pass
        _set_last_run(state, "train_base", datetime.now())

run_hands_off_tasks()


# ==========================================================
# TAB: Train
# ==========================================================
with tabs[0]:
    fast_mode = st.toggle("Fast training mode (recommended)", value=True)
    model_kind = st.selectbox("Model kind", ["sgd", "logreg", "hgb"], index=0)
    calibrate_probs = st.toggle("Calibrate probabilities (slower)", value=False)
    max_tickers = st.number_input("Max tickers to train on (0 = all)", 0, 2000, 400, 50)
    max_workers = st.slider("Parallel workers", 1, 32, 8, 1)
    run_cv = st.toggle("Run CV metrics (slower)", value=False)

    st.subheader("🎓 Train AI Model")
    uni = load_universe()
    show_df(uni, key="universe_table_train")

    force_refresh = st.toggle("Force refresh price cache during training", value=False)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train model", type="primary"):
            try:
                with st.spinner("Training..."):
                    _, metrics = train_model(
                        uni,
                        baseline,
                        force_refresh=force_refresh,
                        max_tickers=None if int(max_tickers) == 0 else int(max_tickers),
                        max_workers=int(max_workers),
                        fast_mode=bool(fast_mode),
                        run_cv=bool(run_cv),
                        model_kind=str(model_kind),
                        calibrate_probs=bool(calibrate_probs),
                    )
                st.success("Model trained and saved.")
                st.json(metrics)
            except Exception as e:
                log_exception(logger, "Training failed", e, show_ui=True, debug=debug_mode)

    with c2:
        if os.path.exists(TRAIN_METRICS):
            st.write("Last training metrics:")
            st.json(json.load(open(TRAIN_METRICS, "r", encoding="utf-8")))
        else:
            st.info("No training metrics found yet.")

    st.divider()
    st.subheader("🧠 Train Meta Model (self-learning calibration)")
    st.caption("Learns from your tracked outcomes in predictions_log.csv.")
    if st.button("Train meta model now", type="secondary"):
        try:
            with st.spinner("Training meta model..."):
                m = train_meta_model_from_pred_log(min_rows=80)
            st.success("Meta model trained.")
            st.json(m)
        except Exception as e:
            log_exception(logger, "Meta training failed", e, show_ui=True, debug=debug_mode)

    if os.path.exists(META_TRAIN_METRICS):
        st.write("Last meta training metrics:")
        st.json(json.load(open(META_TRAIN_METRICS, "r", encoding="utf-8")))


# ==========================================================
# TAB: Scan Universe
# ==========================================================
with tabs[1]:
    st.subheader("🔎 Scan Universe (auto-pick best tickers to watch)")
    model = load_model()
    if model is None:
        st.warning("No base model found yet. Train first for AI %. (Scan still works without AI %.)")

    uni = load_universe()

    st.markdown("### Auto-grow (optional)")
    cG1, cG2, cG3 = st.columns(3)
    with cG1:
        auto_grow_before_scan = st.toggle("Auto-grow universe before scan", value=True)
    with cG2:
        target_scan_universe = st.number_input(
            "Target universe size",
            min_value=10,
            max_value=2000,
            value=max(50, int(len(uni))),
            step=10,
            key="target_universe_size_scan",
        )
    with cG3:
        if st.button("Grow now"):
            uni2 = auto_grow_universe_from_listings(uni, int(target_scan_universe), listings_path=LISTINGS_CSV)
            save_universe(uni2)
            st.success(f"Universe: {len(uni)} → {len(uni2)}")
            st.rerun()

    if auto_grow_before_scan and len(uni) < int(target_scan_universe):
        uni2 = auto_grow_universe_from_listings(uni, int(target_scan_universe), listings_path=LISTINGS_CSV)
        save_universe(uni2)
        st.info(f"Auto-grew universe: {len(uni)} → {len(uni2)}")
        st.rerun()

    st.caption(f"Universe size: {len(uni)} tickers")
    top_n = st.number_input("Show top N", 5, 200, 30, 5)
    st.divider()

    cN1, cN2, cN3 = st.columns(3)
    with cN1:
        use_news_boost = st.toggle("Use news boost in scan ranking (top candidates only)", value=True)
    with cN2:
        news_days = st.number_input("News window (days)", 1, 14, 7, 1)
    with cN3:
        news_candidates = st.number_input("Max tickers to fetch news for", 5, 200, 60, 5)

    show_debug_cols = st.toggle("Show debug columns", value=False)
    auto_scan = st.toggle("Auto-run scan on each refresh", value=bool(hands_off), help="Hands-off mode enables this.")
    force_refresh = st.toggle("Force refresh price cache during scan", value=False)

    run_scan_now = st.button("Run scan", type="primary")
    should_scan = run_scan_now or (auto_refresh and auto_scan)

    if should_scan:
        rows = []
        errs = 0
        scored_ok = 0
        skipped = 0
        prog = st.progress(0)

        try:
            _, bctx = baseline_context(baseline, force_refresh=force_refresh)
        except Exception as e:
            log_exception(logger, f"Baseline fetch failed ({baseline})", e, show_ui=True, debug=debug_mode)
            bctx = None

        for i, r in uni.iterrows():
            t = str(r["ticker"]).upper().strip()
            le = float(r["labor_exposure"])
            js = float(r["jan_score"])
            try:
                row = score_one_week(t, baseline, le, js, model, force_refresh=force_refresh, use_news=False, bctx=bctx)
                rows.append(row)
                scored_ok += 1
            except Exception as e:
                skipped += 1
                errs += 1
                dbg_event("scan_skip", ticker=t, err=str(e))
                if debug_mode:
                    try:
                        logger.exception(f"scan_skip {t}")
                    except Exception:
                        pass
            prog.progress((i + 1) / max(1, len(uni)))

        out = pd.DataFrame(rows)
        st.info(f"Scan summary: scored={scored_ok} • skipped={skipped} • errors={errs} • min_daily_rows={MIN_DAILY_ROWS_SCORE}")

        if out.empty:
            st.warning("No results. Try fewer tickers or check data availability.")
        else:
            for col in ["NewsHeadlines_7d","NewsSent_7d","NewsIntensity_7d","NewsLaborHits_7d","NewsProbGood_Mean","NewsProbGood_Max"]:
                if col not in out.columns:
                    out[col] = np.nan

            out["AI_Prob_Outperform_%"] = pd.to_numeric(out["AI_Prob_Outperform_%"], errors="coerce")
            out["WeeklyVol_%"] = pd.to_numeric(out["WeeklyVol_%"], errors="coerce")
            out["AvgVol_20d"] = pd.to_numeric(out["AvgVol_20d"], errors="coerce")

            f = out.copy()
            f = f[(f["WeeklyVol_%"].fillna(999) <= float(max_vol))]
            if min_avg_vol > 0:
                f = f[(f["AvgVol_20d"].fillna(0) >= int(min_avg_vol))]
            if model is not None:
                f = f[(f["AI_Prob_Outperform_%"].fillna(0) >= float(min_ai))]

            if relax_filters and len(f) < max(3, int(top_n) // 4):
                f = out.copy()

            # news boost on top candidates only
            if use_news_boost and not f.empty:
                sort_cols_pre = ["AI_Prob_Outperform_%","ThesisLiteScore"] if model is not None else ["ThesisLiteScore"]
                f = f.sort_values(by=sort_cols_pre, ascending=False)
                cand_n = min(int(news_candidates), len(f))
                cand = f.head(cand_n).copy()

                spy_above_ma200 = int(cand["_spy_above_ma200"].dropna().iloc[0]) if "_spy_above_ma200" in cand.columns and cand["_spy_above_ma200"].notna().any() else 1
                spy_vol_4w = float(cand["_spy_vol_4w"].dropna().iloc[0]) if "_spy_vol_4w" in cand.columns and cand["_spy_vol_4w"].notna().any() else float("nan")

                for idx, row in cand.iterrows():
                    tkr = str(row["Ticker"]).upper().strip()
                    company_name = ""
                    try:
                        if "universe_df" in globals() and universe_df is not None and not universe_df.empty:
                            hit = universe_df.loc[universe_df["ticker"] == tkr, "name"]
                            company_name = str(hit.iloc[0]) if len(hit) else ""
                    except Exception:
                        company_name = ""

                    nf = get_news_features(tkr, company_name=company_name, days=int(news_days), max_items=30)

                    news_heads = int(nf["headlines"])
                    news_sent = float(nf["sent"])
                    news_int = float(nf["intensity"])
                    news_labor = int(nf["labor_hits"])
                    news_prob_mean = float(nf.get("prob_good_mean", np.nan))
                    news_prob_max = float(nf.get("prob_good_max", np.nan))

                    rs1 = float(row["_rs1"])
                    rs4 = float(row["_rs4"])
                    vol = float(row["_volw"]) if not pd.isna(row["_volw"]) else float("nan")
                    above_ma20 = int(row["_above_ma20"])
                    le = float(row["LaborExposure"])
                    js = float(row["JanScore"])

                    ai = row.get("AI_Prob_Outperform_%", np.nan)
                    ai_prob = None if pd.isna(ai) else float(ai)

                    ts = thesis_lite_score(
                        rs_4w=rs4,
                        rs_1w=rs1,
                        above_ma20=above_ma20,
                        vol_weekly=vol,
                        labor_exposure=le,
                        jan_score=js,
                        news_sent=news_sent,
                        news_intensity=news_int,
                        news_labor_hits=news_labor,
                        news_prob_good=news_prob_mean,
                        news_prob_best=news_prob_max,
                        spy_above_ma200=spy_above_ma200,
                        spy_vol_4w=spy_vol_4w,

                        # pull through structure features already computed in score_one_week
                        rs_12w=float(row.get("_rs12", 0.0)),
                        mom_12w=float(row.get("_mom12", 0.0)),
                        dd_52w=float(row.get("_dd52", 0.0)),
                        ema20_gap=float(row.get("_ema20_gap", 0.0)),
                        bb_width_20=float(row.get("_bb_width", 0.0)),
                        rsi14_v=float(row.get("_rsi14", np.nan)) if not pd.isna(row.get("_rsi14", np.nan)) else float("nan"),
                        atrp14_v=float(row.get("_atrp14", np.nan)) if not pd.isna(row.get("_atrp14", np.nan)) else float("nan"),
                        vol_ratio_v=float(row.get("_vol_ratio", np.nan)) if not pd.isna(row.get("_vol_ratio", np.nan)) else float("nan"),
                        ticker_above_ma200=int(row.get("_ticker_above_ma200", 0)),
                    )


                    cand.at[idx, "ThesisLiteScore"] = round(ts, 2)
                    cand.at[idx, "NewsHeadlines_7d"] = news_heads
                    cand.at[idx, "NewsSent_7d"] = round(news_sent, 3)
                    cand.at[idx, "NewsIntensity_7d"] = round(news_int, 3)
                    cand.at[idx, "NewsLaborHits_7d"] = news_labor
                    cand.at[idx, "NewsProbGood_Mean"] = round(news_prob_mean, 3) if not pd.isna(news_prob_mean) else np.nan
                    cand.at[idx, "NewsProbGood_Max"] = round(news_prob_max, 3) if not pd.isna(news_prob_max) else np.nan

                    cand.at[idx, "Action"] = action_from_scores(ai_prob, ts)
                    cand.at[idx, "Confidence"] = confidence_label(ai_prob, ts)

                f.update(cand)

            f = apply_meta_calibration(f)

            for idx, row in f.iterrows():
                ai = row.get("AI_Prob_Outperform_%", np.nan)
                ai_prob = None if pd.isna(ai) else float(ai)
                ai_cal = row.get("AI_Prob_Calibrated_%", np.nan)
                ai_prob_cal = None if pd.isna(ai_cal) else float(ai_cal)

                f.at[idx, "Why"] = explain_why(
                    ai_prob=ai_prob,
                    thesis_score=float(row.get("ThesisLiteScore", 0.0)),
                    rs1=float(row.get("_rs1", 0.0)),
                    rs4=float(row.get("_rs4", 0.0)),
                    above_ma20=int(row.get("_above_ma20", 0)),
                    vol_weekly=float(row.get("_volw", np.nan)) if not pd.isna(row.get("_volw", np.nan)) else float("nan"),
                    labor_exposure=float(row.get("LaborExposure", 5.0)),
                    jan_score=float(row.get("JanScore", 0.0)),
                    rsi_v=float(row.get("_rsi14", np.nan)) if not pd.isna(row.get("_rsi14", np.nan)) else float("nan"),
                    atrp_v=float(row.get("_atrp14", np.nan)) if not pd.isna(row.get("_atrp14", np.nan)) else float("nan"),
                    vol_ratio_v=float(row.get("_vol_ratio", np.nan)) if not pd.isna(row.get("_vol_ratio", np.nan)) else float("nan"),
                    baseline=baseline,
                    news_heads=int(row.get("NewsHeadlines_7d", 0) if not pd.isna(row.get("NewsHeadlines_7d", np.nan)) else 0),
                    news_sent=float(row.get("NewsSent_7d", 0.0) if not pd.isna(row.get("NewsSent_7d", np.nan)) else 0.0),
                    news_intensity=float(row.get("NewsIntensity_7d", 0.0) if not pd.isna(row.get("NewsIntensity_7d", np.nan)) else 0.0),
                    news_labor_hits=int(row.get("NewsLaborHits_7d", 0) if not pd.isna(row.get("NewsLaborHits_7d", np.nan)) else 0),
                    news_prob_good=float(row.get("NewsProbGood_Mean", np.nan)) if "NewsProbGood_Mean" in row else np.nan,
                    news_prob_best=float(row.get("NewsProbGood_Max", np.nan)) if "NewsProbGood_Max" in row else np.nan,
                    spy_above_ma200=int(row.get("_spy_above_ma200", 1)),
                    spy_vol_4w=float(row.get("_spy_vol_4w", np.nan)) if not pd.isna(row.get("_spy_vol_4w", np.nan)) else float("nan"),
                    ai_prob_cal=ai_prob_cal,
                )

            if "AI_Prob_Calibrated_%" in f.columns and f["AI_Prob_Calibrated_%"].notna().any():
                f["AI_Filter_%"] = pd.to_numeric(f["AI_Prob_Calibrated_%"], errors="coerce").fillna(f["AI_Prob_Outperform_%"])
            else:
                f["AI_Filter_%"] = pd.to_numeric(f["AI_Prob_Outperform_%"], errors="coerce")

            sort_cols = ["AI_Filter_%","AI_Prob_Outperform_%","ThesisLiteScore"] if model is not None else ["ThesisLiteScore"]
            f = f.sort_values(by=sort_cols, ascending=False).head(int(top_n))

            st.session_state["last_scan_df"] = f
            st.session_state["last_scan_time"] = now_stamp()
            st.session_state["last_scan_errs"] = errs

            display_cols = [c for c in f.columns if (show_debug_cols or not str(c).startswith("_"))]
            display_df = f[display_cols].copy()

            append_log(SCAN_LOG, display_df)
            log_predictions("scan", baseline, f)
            append_news_log(baseline, f)

    scan_df = st.session_state.get("last_scan_df", None)
    if isinstance(scan_df, pd.DataFrame) and not scan_df.empty:
        st.info(f"Last scan: {st.session_state.get('last_scan_time', '—')} • errors: {st.session_state.get('last_scan_errs', 0)}")
        # --- guarantee scan_df has a 'ticker' column using new safe helper ---
        try:
            scan_df = ensure_ticker_col(scan_df, label="scan_df")
            tickers = scan_df["ticker"].tolist()
            prices_dict = get_prices_by_ticker(tickers)
        except (ValueError, KeyError) as e:
            st.error(f"Unable to find ticker column in scan results: {e}")
            scan_df = scan_df.copy()
            scan_df["ticker"] = ""
            prices_dict = {}
        scan_df_display = add_day_move(scan_df, prices_dict)
        display_cols = [c for c in scan_df_display.columns if (show_debug_cols or not str(c).startswith("_"))]
        show_df(scan_df_display[display_cols], key="scan_results_table")

        st.download_button(
            "⬇ Export scan CSV",
            data=scan_df.to_csv(index=False).encode("utf-8"),
            file_name="scan_results.csv",
            mime="text/csv",
        )

        cA, cB, cC = st.columns(3)
        with cA:
            if st.button("💾 Save Top picks to watchlist.csv (merge)"):
                wl_updated = sync_watchlist_from_scan(scan_df, mode="merge", max_rows=int(min(50, len(scan_df))))
                save_watchlist(wl_updated)
                st.success(f"Watchlist updated (now {len(wl_updated)} tickers).")
        with cB:
            if st.button("🔁 Replace watchlist with Top picks"):
                wl_updated = sync_watchlist_from_scan(scan_df, mode="replace", max_rows=int(min(50, len(scan_df))))
                save_watchlist(wl_updated)
                st.success(f"Watchlist replaced (now {len(wl_updated)} tickers).")
        with cC:
            st.caption("Tip: if scan shows only 1, lower Min AI %, raise Max vol %, or enable Relax filters.")
    else:
        st.write("Run a scan to see results (or enable auto-refresh + auto-scan).")


# ==========================================================
# TAB: Watchlist
# ==========================================================
with tabs[2]:
    st.subheader("⭐ Watchlist (manual overrides)")
    st.write("Edit labor/catalyst/entry price per ticker and re-score quickly.")

    wl = load_watchlist()
    if wl.empty:
        st.info("No watchlist.csv yet. Use Scan tab → Save Top picks first.")
    else:
        edited = st.data_editor(wl, num_rows="dynamic", use_container_width=True)
        save_watchlist(edited)

        model = load_model()
        if model is None:
            st.warning("No base model found yet. Train first for AI %.")

        force_refresh = st.toggle("Force refresh price cache during watchlist score", value=False)
        use_news_watch = st.toggle("Use news overlay on watchlist scoring", value=True)
        news_days_w = st.number_input("Watchlist news window (days)", 1, 14, 7, 1)

        if st.button("Score watchlist", type="primary"):
            rows = []
            errs = 0
            prog = st.progress(0)

            try:
                _, bctx = baseline_context(baseline, force_refresh=force_refresh)
            except Exception:
                bctx = None

            for i, r in edited.iterrows():
                t = str(r["ticker"]).upper().strip()
                try:
                    row = score_one_week(
                        ticker=t,
                        baseline=baseline,
                        labor_exposure=float(r["labor_exposure"]),
                        jan_score=float(r["jan_score"]),
                        model=model,
                        force_refresh=force_refresh,
                        use_news=use_news_watch,
                        news_days=int(news_days_w),
                        bctx=bctx,

                        # ✅ apply watchlist overrides
                        catalyst=float(r.get("catalyst", 0)),
                        labor_shock=int(r.get("labor_shock", 0)),
                        volatile=int(r.get("volatile", 0)),
                    )


                    row["Catalyst"] = int(r.get("catalyst", 0))
                    row["LaborShock"] = int(r.get("labor_shock", 0))
                    row["EntryPrice"] = r.get("entry_price", np.nan)

                    try:
                        ep = float(r.get("entry_price"))
                        if not np.isnan(ep) and ep > 0:
                            row["PnL_%"] = round(((row["LastPrice"] / ep) - 1.0) * 100.0, 2)
                        else:
                            row["PnL_%"] = np.nan
                    except Exception:
                        row["PnL_%"] = np.nan

                    row["Why"] = ""
                    rows.append(row)
                except Exception:
                    errs += 1
                prog.progress((i + 1) / max(1, len(edited)))

            res = pd.DataFrame(rows)
            res = apply_meta_calibration(res)

            for idx, row in res.iterrows():
                ai = row.get("AI_Prob_Outperform_%", np.nan)
                ai_prob = None if pd.isna(ai) else float(ai)
                ai_cal = row.get("AI_Prob_Calibrated_%", np.nan)
                ai_prob_cal = None if pd.isna(ai_cal) else float(ai_cal)

                res.at[idx, "Why"] = explain_why(
                    ai_prob=ai_prob,
                    thesis_score=float(row.get("ThesisLiteScore", 0.0)),
                    rs1=float(row.get("_rs1", 0.0)),
                    rs4=float(row.get("_rs4", 0.0)),
                    above_ma20=int(row.get("_above_ma20", 0)),
                    vol_weekly=float(row.get("_volw", np.nan)) if not pd.isna(row.get("_volw", np.nan)) else float("nan"),
                    labor_exposure=float(row.get("LaborExposure", 5.0)),
                    jan_score=float(row.get("JanScore", 0.0)),
                    rsi_v=float(row.get("_rsi14", np.nan)) if not pd.isna(row.get("_rsi14", np.nan)) else float("nan"),
                    atrp_v=float(row.get("_atrp14", np.nan)) if not pd.isna(row.get("_atrp14", np.nan)) else float("nan"),
                    vol_ratio_v=float(row.get("_vol_ratio", np.nan)) if not pd.isna(row.get("_vol_ratio", np.nan)) else float("nan"),
                    baseline=baseline,
                    news_heads=int(row.get("NewsHeadlines_7d", 0) if not pd.isna(row.get("NewsHeadlines_7d", np.nan)) else 0),
                    news_sent=float(row.get("NewsSent_7d", 0.0) if not pd.isna(row.get("NewsSent_7d", np.nan)) else 0.0),
                    news_intensity=float(row.get("NewsIntensity_7d", 0.0) if not pd.isna(row.get("NewsIntensity_7d", np.nan)) else 0.0),
                    news_labor_hits=int(row.get("NewsLaborHits_7d", 0) if not pd.isna(row.get("NewsLaborHits_7d", np.nan)) else 0),
                    news_prob_good=float(row.get("NewsProbGood_Mean", np.nan)) if "NewsProbGood_Mean" in row else np.nan,
                    news_prob_best=float(row.get("NewsProbGood_Max", np.nan)) if "NewsProbGood_Max" in row else np.nan,
                    spy_above_ma200=int(row.get("_spy_above_ma200", 1)),
                    spy_vol_4w=float(row.get("_spy_vol_4w", np.nan)) if not pd.isna(row.get("_spy_vol_4w", np.nan)) else float("nan"),
                    ai_prob_cal=ai_prob_cal,
                )

            if "AI_Prob_Calibrated_%" in res.columns and res["AI_Prob_Calibrated_%"].notna().any():
                res["AI_Filter_%"] = pd.to_numeric(res["AI_Prob_Calibrated_%"], errors="coerce").fillna(res["AI_Prob_Outperform_%"])
            else:
                res["AI_Filter_%"] = pd.to_numeric(res["AI_Prob_Outperform_%"], errors="coerce")

            sort_cols = ["AI_Filter_%","AI_Prob_Outperform_%","ThesisLiteScore"] if model is not None else ["ThesisLiteScore"]
            res = res.sort_values(by=sort_cols, ascending=False)

            view = res[[c for c in res.columns if not str(c).startswith("_")]].copy()
            show_df(view, key="watchlist_scored_table")

            st.download_button(
                "⬇ Export watchlist CSV",
                data=res.to_csv(index=False).encode("utf-8"),
                file_name="watchlist_scored.csv",
                mime="text/csv",
            )

            append_log(WATCH_LOG, view)
            log_predictions("watchlist", baseline, res)
            append_news_log(baseline, res)
            st.info(f"Watchlist scoring done. Errors: {errs}")


# ==========================================================
# TAB: AI Predictions (New)
# ==========================================================
with tabs[3]:
    st.subheader("🤖 AI Predictions (Stable New Pipeline)")
    st.caption("Train a simple, deterministic LogisticRegression on price momentum & volume. Log predictions and track accuracy over time.")
    with st.expander("How to read this section (no financial advice)", expanded=True):
        st.markdown(
            """
            1. **Run predictions** → we score every ticker in your universe with `prob_up` (chance price is higher after the horizon).
            2. **Log them** → snapshot the table with **Log these predictions** so we can grade ourselves later.
            3. **Update outcomes** → once the horizon passes, click **Update outcomes** to fill in actual prices and accuracy.
            4. **Judge us visually** → rolling hit-rate and per-ticker accuracy charts live below so you can see if the model is working.
            """
        )

    # Load universe or watchlist to get tickers
    uni = load_universe()
    if uni.empty:
        st.info("Add tickers to your universe first (Manage Universe tab).")
    else:
        # Build price dict
        try:
            uni_safe = ensure_ticker_col(uni, label="universe")
            tickers = uni_safe["ticker"].tolist()

            with st.spinner(f"Loading price data for {len(tickers)} tickers..."):
                prices_dict = get_prices_by_ticker(tickers)

            # Ensure dict keys are uppercase
            prices_dict = {str(k).upper().strip(): v for k, v in prices_dict.items()}

            if not prices_dict:
                st.warning("No price data found for universe tickers. Add tickers or fetch prices first.")
            else:
                st.success(f"Loaded {len(prices_dict)} tickers with price data")

                horizon_days = st.sidebar.number_input("Prediction horizon (days)", min_value=1, max_value=20, value=1, step=1)
                force_retrain = st.sidebar.button("🔄 Force retrain model")

                try:
                    model, meta = train_or_load_model(prices_dict, horizon_days=horizon_days, force_retrain=force_retrain)
                    st.caption(f"✅ Model trained: {meta.get('trained_utc')} | rows={meta.get('n_rows')} | tickers={meta.get('tickers')}")

                    pred_df = predict_latest(model, prices_dict, horizon_days=horizon_days)
                    if pred_df.empty:
                        st.warning("No predictions generated (tickers may lack enough data).")
                    else:
                        scan_df = uni_safe.copy()
                        company_name_map: dict[str, str] = {}
                        for _, row in scan_df.iterrows():
                            nm = row.get("company_name") or row.get("name") or ""
                            tck = str(row.get("ticker", "")).upper().strip()
                            if tck and nm:
                                company_name_map[tck] = str(nm).strip()

                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("💾 Log these predictions", key="pred_log_btn_new"):
                                append_prediction_log(pred_df, meta)
                                st.success(f"Logged {len(pred_df)} predictions to {PRED_LOG_CSV}")
                        with c2:
                            st.caption("Predictions are logged with model meta so they can be graded later.")

                        with st.spinner("Refreshing outcomes so status is current..."):
                            log_df_to_show = update_log_outcomes(prices_dict)

                        render_dashboard(
                            scan_df=scan_df,
                            prices_by_ticker=prices_dict,
                            pred_df=pred_df,
                            log_df=log_df_to_show,
                            company_name_map=company_name_map,
                        )

                except Exception as e:
                    st.error(f"Prediction engine error: {e}")
                    with st.expander("See full error"):
                        st.exception(e)

        except (ValueError, KeyError) as e:
            st.error(f"Unable to process universe tickers: {e}")

# ==========================================================
# TAB: Live
# ==========================================================
with tabs[4]:
    st.subheader("🟢 Live monitoring (snapshot / near-real-time)")
    st.caption("Uses Twelve Data time_series. Needs a free API key. Not financial advice.")

    wl = load_watchlist()
    if wl.empty:
        st.info("Add a watchlist first (Scan tab → Save Top picks).")
    elif not td_key:
        st.warning("Add your Twelve Data API key in the sidebar to enable live + intraday AI.")
    else:
        max_live = st.slider(
            "How many tickers to watch live",
            1, min(20, len(wl)),
            min(5, len(wl)),
            1,
            key="live_max_live",
        )
        interval = st.selectbox("Live interval", ["1min", "5min", "15min"], index=1, key="live_interval")
        horizon_bars = st.number_input("AI horizon (bars ahead)", 3, 60, 12, 1, help="12 bars on 5min ≈ 60 minutes", key="live_horizon_bars")

        st.markdown("### 🔔 Alerts / filtering")
        min_ai_rt = st.slider("AI_RT_% threshold", 0, 95, 65, 1, key="live_min_ai_rt")
        only_show_alerts = st.toggle("Only show tickers >= threshold", value=False, key="live_only_alerts")
        enable_alerts = st.toggle("Enable alerts (toast/info)", value=True, key="live_enable_alerts")
        alert_cooldown_mins = st.number_input("Alert cooldown (minutes)", 1, 180, 10, 1, key="live_alert_cooldown")

        st.markdown("### 🤖 Intraday AI (optional)")
        use_intraday_ai = st.toggle("Enable intraday AI probabilities (AI_RT_%)", value=True, key="live_use_intraday_ai")

        with st.expander("🎓 Train intraday AI from watchlist (needs enough bars cached)", expanded=False):
            interval_ai = st.selectbox("Training interval", ["1min", "5min", "15min"], index=1, key="live_train_interval")
            horizon_bars_ai = st.number_input("Training horizon (bars ahead)", 3, 60, 12, 1, key="live_train_horizon")
            cRT1, cRT2 = st.columns(2)
            with cRT1:
                if st.button("Train intraday AI from watchlist", type="secondary", key="live_train_btn"):
                    try:
                        with st.spinner("Training intraday model (needs enough collected bars)..."):
                            m = train_intraday_model_from_watchlist(
                                baseline=baseline,
                                interval=interval_ai,
                                horizon_bars=int(horizon_bars_ai),
                                td_key=td_key,
                                min_rows=2000,
                            )
                        st.success("Intraday model trained.")
                        st.json(m)
                    except Exception as e:
                        st.warning(str(e))
            with cRT2:
                st.caption(
                    f"Intraday model: {'FOUND ✅' if os.path.exists(INTRADAY_MODEL_PATH) else 'not trained yet'} "
                    f"• interval={interval_ai} • horizon={int(horizon_bars_ai)} bars"
                )

        picks = wl["ticker"].astype(str).str.upper().tolist()[:int(max_live)]
        base_df = cache_intraday(baseline, interval=interval, apikey=td_key, outputsize=800).tail(600)
        model_intra = load_intraday_model() if (use_intraday_ai and os.path.exists(INTRADAY_MODEL_PATH)) else None

        rows = []
        for t in picks:
            df_i = cache_intraday(t, interval=interval, apikey=td_key, outputsize=240).tail(120)
            if df_i.empty:
                rows.append({"Ticker": t, "Last": np.nan, "Time": "", "AI_RT_%": None, "Bias": "N/A", "Score": 0, "Why": "No data / rate limit"})
                continue

            last = float(df_i["close"].iloc[-1])
            tm = str(df_i.index[-1])
            sig = intraday_signal(df_i)

            ai_rt = None
            if model_intra is not None:
                ai_rt = intraday_ai_prob_fast(
                    symbol=t,
                    base_df=base_df,
                    interval=interval,
                    horizon_bars=int(horizon_bars),
                    td_key=td_key,
                )

            rows.append({"Ticker": t, "Last": round(last, 2), "Time": tm, "AI_RT_%": ai_rt, **sig})

        live_df = pd.DataFrame(rows)

        if enable_alerts:
            emit_rt_alerts(live_df, threshold=float(min_ai_rt), cooldown_sec=int(alert_cooldown_mins * 60))

        df_show = live_df.copy()
        df_show["AI_RT_%"] = pd.to_numeric(df_show["AI_RT_%"], errors="coerce")
        if only_show_alerts:
            df_show = df_show[df_show["AI_RT_%"].fillna(-1) >= float(min_ai_rt)].copy()

        df_show = df_show.sort_values(by=["AI_RT_%","Score"], ascending=False)
        show_df(df_show, key="live_table")

        st.download_button(
            "⬇ Export live snapshot CSV",
            data=df_show.to_csv(index=False).encode("utf-8"),
            file_name="live_snapshot.csv",
            mime="text/csv",
        )


# ==========================================================
# TAB: Day Trading (experimental)
# ==========================================================
with tabs[5]:
    st.subheader("⚡ Day Trading (experimental)")
    st.caption("NOT financial advice. Intraday snapshot signals + optional intraday AI (AI_RT_%).")

    wl = load_watchlist()
    if wl.empty:
        st.info("Add a watchlist first (Scan tab → Save Top picks).")
    elif not td_key:
        st.warning("Add your Twelve Data API key in the sidebar to enable intraday/day trading views.")
    else:
        cA, cB, cC, cD = st.columns([2, 1, 1, 1])
        with cA:
            pick = st.selectbox("Ticker", wl["ticker"].astype(str).str.upper().tolist(), index=0, key="dt_pick")
        with cB:
            interval_dt = st.selectbox("Interval", ["1min", "5min", "15min"], index=1, key="dt_interval")
        with cC:
            bars = st.slider("Bars shown", 80, 1500, 420, 20, key="dt_bars")
        with cD:
            horizon_bars = st.number_input("AI horizon (bars)", 3, 60, 12, 1, key="dt_horizon_bars")

        df_i = cache_intraday(pick, interval=interval_dt, apikey=td_key, outputsize=max(int(bars), 800)).tail(int(bars))
        if df_i.empty:
            st.warning("No intraday data returned (rate limit / invalid symbol / key issue).")
        else:
            st.markdown("### Price")
            st.line_chart(df_i["close"])

            sig = intraday_signal(df_i)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Bias", sig.get("Bias", "N/A"))
            with c2:
                st.metric("Score (0–10)", sig.get("Score", 0))
            with c3:
                st.metric("Last", float(df_i["close"].iloc[-1]))

            st.caption(sig.get("Why", ""))

            st.markdown("### Intraday AI (AI_RT_%)")
            model_intra = load_intraday_model()
            if model_intra is None:
                st.info("Intraday model not trained yet. Use the Live tab → Train intraday AI from watchlist.")
            else:
                base_df = cache_intraday(baseline, interval=interval_dt, apikey=td_key, outputsize=1000).tail(800)
                ai_rt = intraday_ai_prob_fast(
                    symbol=pick,
                    base_df=base_df,
                    interval=interval_dt,
                    horizon_bars=int(horizon_bars),
                    td_key=td_key,
                )
                st.metric("AI_RT_% (prob beat baseline over next horizon)", "—" if ai_rt is None else f"{ai_rt:.1f}%")

        st.divider()
        st.markdown("### Watchlist intraday heatmap (quick)")
        max_rows = st.slider("Rows", 3, min(20, len(wl)), min(10, len(wl)), 1, key="dt_rows")
        only_show = st.toggle("Only show AI_RT_% >= threshold", value=False, key="dt_only_show")
        thr = st.slider("Threshold", 0, 95, 65, 1, key="dt_thr")

        base_df = cache_intraday(baseline, interval=interval_dt, apikey=td_key, outputsize=1000).tail(800)
        model_intra = load_intraday_model()

        rows = []
        for t in wl["ticker"].astype(str).str.upper().tolist()[:int(max_rows)]:
            d = cache_intraday(t, interval=interval_dt, apikey=td_key, outputsize=300).tail(160)
            if d.empty:
                rows.append({"Ticker": t, "AI_RT_%": None, "Bias": "N/A", "Score": 0, "Why": "No data / rate limit"})
                continue
            s = intraday_signal(d)
            ai_rt = None
            if model_intra is not None and not base_df.empty:
                ai_rt = intraday_ai_prob_fast(t, base_df, interval_dt, int(horizon_bars), td_key)
            rows.append({"Ticker": t, "AI_RT_%": ai_rt, **s})

        heat = pd.DataFrame(rows)
        heat["AI_RT_%"] = pd.to_numeric(heat["AI_RT_%"], errors="coerce")
        if only_show:
            heat = heat[heat["AI_RT_%"].fillna(-1) >= float(thr)].copy()
        heat = heat.sort_values(by=["AI_RT_%", "Score"], ascending=False)
        show_df(heat, key="dt_heat_table")


# ==========================================================
# TAB: News
# ==========================================================
with tabs[6]:
    st.subheader("📰 News (RSS)")
    st.caption("Google News RSS per ticker (company name + ticker) so headlines match the right company.")

    wl = load_watchlist()
    default_t = (wl["ticker"].iloc[0] if not wl.empty else baseline)
    tkr = st.text_input("Ticker for news", value=str(default_t)).strip().upper()
    days = st.slider("Window (days)", 1, 14, 7, 1, key="news_days_tab")
    max_items = st.slider("Max items", 5, 60, 30, 5, key="news_max_items_tab")

    company_name = ""
    try:
        if "universe_df" in globals() and universe_df is not None and not universe_df.empty:
            hit = universe_df.loc[universe_df["ticker"] == tkr, "name"]
            company_name = str(hit.iloc[0]) if len(hit) else ""
    except Exception:
        company_name = ""

    items_raw = fetch_news_for_ticker(tkr, company_name, days=int(days))
    items, model_stats = score_news_items_with_model(items_raw, ticker=tkr)
    text_blob = " ".join([f"{it.get('title','')} {it.get('snippet','')}" for it in items])
    pos_hits = _count_hits(text_blob, NEWS_POS)
    neg_hits = _count_hits(text_blob, NEWS_NEG)
    labor_hits = _count_hits(text_blob, NEWS_LABOR)
    denom = max(1, pos_hits + neg_hits)
    sent = clamp((pos_hits - neg_hits) / denom, -1.0, 1.0) if (pos_hits + neg_hits) else 0.0
    intensity = clamp(len(items) / 12.0, 0.0, 1.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Headlines", len(items))
    c2.metric("Sent (pos-neg)", f"{sent:+.2f}")
    c3.metric("Intensity", f"{intensity:.2f}")
    c4.metric("Labor hits", labor_hits)
    c5.metric(
        "News model P(good)",
        "—" if pd.isna(model_stats.get("prob_good_mean", np.nan)) else f"{model_stats['prob_good_mean']:.2f}",
    )

    st.markdown("### Headlines (matched to this ticker)")
    render_news_list(items, max_items=int(max_items))

    st.divider()
    st.markdown("### Label headlines (good / neutral / bad)")
    labels_df = load_news_labels_df()
    ticker_labels = labels_df[labels_df["ticker"] == tkr].copy() if not labels_df.empty else pd.DataFrame()
    if ticker_labels.empty:
        st.info("No logged headlines yet. Fetch news first, then label them here.")
    else:
        ticker_labels["logged_utc"] = pd.to_datetime(ticker_labels.get("logged_utc"), errors="coerce")
        ticker_labels = ticker_labels.sort_values("logged_utc", ascending=False)
        for _, row in ticker_labels.head(30).iterrows():
            cL1, cL2 = st.columns([5, 2])
            with cL1:
                st.markdown(f"**{row.get('title','')}**")
                st.caption(f"{row.get('published','')} — {row.get('snippet','')}")
                if row.get("link"):
                    st.caption(f"[Link]({row.get('link')}) • id={row.get('headline_id','')}")
            with cL2:
                options = ["", "good", "neutral", "bad"]
                current = str(row.get("user_feedback", "") or "").strip().lower()
                idx = options.index(current) if current in options else 0
                choice = st.radio(
                    "Feedback",
                    options,
                    index=idx,
                    horizontal=True,
                    key=f"label_{row.get('headline_id','')}",
                )
                if choice != current:
                    if upsert_news_feedback(row.get("headline_id", ""), choice):
                        st.toast(f"Saved label '{choice}'")
                        st.session_state["news_label_saved"] = now_stamp()
                        st.rerun()

    st.divider()
    st.markdown("### Train and monitor the news model")
    cT1, cT2 = st.columns([1, 1])
    with cT1:
        if st.button("Train news model", type="primary"):
            try:
                with st.spinner("Training news classifier..."):
                    metrics = train_news_classifier()
                st.success("News model trained and saved.")
                st.json(metrics)
            except Exception as e:
                st.warning(str(e))
    with cT2:
        met = load_news_model_metrics()
        if met:
            st.caption(f"Last trained: {met.get('trained_at', '—')} | holdout={met.get('holdout_rows', 0)}")
            st.metric("Holdout accuracy", "—" if met.get("accuracy") is None else f"{met['accuracy']*100:.1f}%")
            per = pd.DataFrame(met.get("per_ticker", []))
            if not per.empty:
                per["hit_rate"] = (per["hit_rate"] * 100).round(1)
                per["rolling_hit"] = (per["rolling_hit"] * 100).round(1)
                st.dataframe(per.sort_values(["hit_rate", "rows"], ascending=[False, False]), use_container_width=True)
            cal = pd.DataFrame(met.get("calibration_bins", []))
            if not cal.empty:
                st.caption("Calibration (holdout): predicted vs actual good-rate")
                st.dataframe(cal, use_container_width=True)
        else:
            st.info("Train the news model to view holdout accuracy and calibration.")

    st.divider()
    st.markdown("### Latest news overlay logged")
    if os.path.exists(NEWS_LOG):
        df = safe_read_log_csv(NEWS_LOG)
        if not df.empty:
            st.dataframe(df.tail(200), use_container_width=True)
            st.download_button(
                "⬇ Download news_features.csv",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="news_features.csv",
                mime="text/csv",
            )
        else:
            st.info("news_features.csv exists but is empty.")
    else:
        st.info("No news log yet (run Scan/Watchlist with news enabled).")

st.subheader("Latest news per watchlist ticker (no cross-matching)")
for t in watch_tickers:
    nm = name_map.get(t, "")
    st.markdown(f"### {ticker_display(t, name_map)}")
    items_w = fetch_news_for_ticker(t, nm, days=int(days))
    items_scored, _ = score_news_items_with_model(items_w, ticker=t)
    render_news_list(items_scored, max_items=8)


# ==========================================================
# TAB: Manage Universe
# ==========================================================
with tabs[7]:
    st.subheader("🧩 Manage Universe")
    st.caption("Universe = tickers the model trains/scans. You can edit, import, auto-grow, and export.")

    uni = load_universe()

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown("### Edit universe.csv")
    with c2:
        if st.button("Reload", key="uni_reload_btn"):
            st.rerun()
    with c3:
        if st.button("Auto-grow to sidebar target", key="uni_autogrow_btn"):
            uni2 = auto_grow_universe_from_listings(uni, int(hands_off_target_universe), listings_path=LISTINGS_CSV)
            save_universe(uni2)
            st.success(f"Universe: {len(uni)} → {len(uni2)}")
            st.rerun()

    edited_uni = st.data_editor(uni, num_rows="dynamic", use_container_width=True, key="uni_editor")
    if st.button("💾 Save universe.csv", type="primary", key="uni_save_btn"):
        save_universe(edited_uni)
        st.success("Saved universe.csv")
        st.rerun()

    st.download_button(
        "⬇ Export universe.csv",
        data=edited_uni.to_csv(index=False).encode("utf-8"),
        file_name="universe.csv",
        mime="text/csv",
        key="uni_export_btn",
    )

    st.divider()
    st.markdown("### Import / merge tickers")
    up = st.file_uploader("Upload CSV (must have a ticker column)", type=["csv"], key="uni_upload")
    if up is not None:
        try:
            dfu = pd.read_csv(up)
            # detect ticker column
            candidates = ["ticker", "Ticker", "symbol", "Symbol", "SYMBOL", "Security Symbol", "ACT Symbol"]
            col = None
            for c in candidates:
                if c in dfu.columns:
                    col = c
                    break
            if col is None:
                col = dfu.columns[0]
            syms = dfu[col].astype(str).str.upper().str.strip()
            syms = syms[syms.notna() & (syms.str.len() > 0)]
            syms = syms[~syms.isin(["NAN", "NONE", "NULL"])]
            syms = syms[~syms.str.contains(r"[ /\\]", regex=True)]
            syms = syms[syms.str.len() <= 12]
            add_df = pd.DataFrame({"ticker": syms.unique().tolist(), "labor_exposure": 5, "jan_score": 0})
            merged = pd.concat([edited_uni, add_df], ignore_index=True).drop_duplicates(subset=["ticker"]).sort_values("ticker")
            st.success(f"Loaded {len(add_df)} candidates. Merge preview below:")
            st.dataframe(merged.head(200), use_container_width=True)
            if st.button("✅ Merge into universe.csv", key="uni_merge_confirm"):
                save_universe(merged)
                st.success(f"Saved merged universe ({len(merged)} tickers).")
                st.rerun()
        except Exception as e:
            log_exception(logger, "Universe import failed", e, show_ui=True, debug=debug_mode)

    st.divider()
    st.markdown("### Quick add tickers (paste)")
    pasted = st.text_area("Paste tickers (comma/space/newline separated)", value="", height=90, key="uni_paste_box")
    if st.button("Add pasted tickers", key="uni_paste_add_btn"):
        toks = [x.strip().upper() for x in re.split(r"[,\s]+", pasted or "") if x.strip()]
        toks = [t for t in toks if (0 < len(t) <= 12) and (" " not in t) and ("/" not in t) and ("\\" not in t)]
        if toks:
            add_df = pd.DataFrame({"ticker": toks, "labor_exposure": 5, "jan_score": 0})
            merged = pd.concat([edited_uni, add_df], ignore_index=True).drop_duplicates(subset=["ticker"]).sort_values("ticker")
            save_universe(merged)
            st.success(f"Added {len(toks)} tickers. Universe now {len(merged)}.")
            st.rerun()
        else:
            st.info("No valid tickers found.")


# ==========================================================
# TAB: Tracking
# ==========================================================
with tabs[8]:
    st.subheader("🧾 Tracking (predictions_log.csv)")
    st.caption("Logs each scan/watchlist run. Use 'Update outcomes' to score hits after target week closes.")
    horizon_controls_ui()


    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🔄 Update outcomes now", type="primary", key="trk_update_outcomes"):
            try:
                with st.spinner("Updating outcomes..."):
                    dfp = update_prediction_outcomes()
                st.success(f"Outcome update finished. Rows: {0 if dfp is None else len(dfp)}")
            except Exception as e:
                log_exception(logger, "Update outcomes failed", e, show_ui=True, debug=debug_mode)

    with c2:
        if st.button("Train meta model (needs enough outcomes)", key="trk_train_meta"):
            try:
                with st.spinner("Training meta model..."):
                    m = train_meta_model_from_pred_log(min_rows=80)
                st.success("Meta model trained.")
                # st.json can fail on numpy types; fallback to json.dumps
                try:
                    st.json(m)
                except Exception:
                    st.code(json.dumps(m, default=str, indent=2))
            except Exception as e:
                st.warning(str(e))

    with c3:
        st.caption(
            f"Files: pred_log={'FOUND ✅' if os.path.exists(PRED_LOG) else 'missing'} • "
            f"meta={'FOUND ✅' if os.path.exists(META_MODEL_PATH) else 'missing'}"
        )

    def _ensure_col(df: pd.DataFrame, col: str, default):
        if col not in df.columns:
            df[col] = default

    pred_for_backfill = pd.DataFrame()
    if not os.path.exists(PRED_LOG):
        st.info("No predictions log yet. Run Scan/Watchlist to start logging.")
    else:
        pred = safe_read_log_csv(PRED_LOG)
        pred_for_backfill = pred.copy()
        if pred is None or pred.empty:
            st.info("predictions_log.csv is empty.")
        else:
            # Ensure expected columns exist BEFORE conversions (prevents scalar .fillna crashes)
            _ensure_col(pred, "outcome_known", 0)
            _ensure_col(pred, "hit", np.nan)
            _ensure_col(pred, "asof_date", pd.NaT)
            _ensure_col(pred, "target_date", pd.NaT)

            pred["outcome_known"] = (
                pd.to_numeric(pred["outcome_known"], errors="coerce")
                .fillna(0)
                .astype(int)
                .clip(0, 1)
            )
            pred["hit"] = pd.to_numeric(pred["hit"], errors="coerce")
            pred["asof_date"] = pd.to_datetime(pred.get("asof_date", None), errors="coerce")
            pred["target_date"] = pd.to_datetime(pred.get("target_date", None), errors="coerce")


            total = int(len(pred))
            known = int(pred["outcome_known"].sum())
            done = pred[(pred["outcome_known"] == 1) & (pred["hit"].notna())].copy()
            acc = float(done["hit"].mean()) if len(done) else float("nan")

            cA, cB, cC, cD = st.columns(4)
            cA.metric("Total preds", total)
            cB.metric("Known outcomes", known)
            cC.metric("Hit-rate (known)", "—" if np.isnan(acc) else f"{acc*100:.1f}%")
            cD.metric("Unknown", total - known)

            st.markdown("### Breakdown by source")
            if len(done) and "source" in done.columns:
                by = (
                    done.groupby("source", dropna=False)["hit"]
                    .agg(count="count", hit_rate="mean")
                    .reset_index()
                )
                by["hit_rate"] = (by["hit_rate"] * 100.0).round(1)
                st.dataframe(
                    by.sort_values(["count", "hit_rate"], ascending=[False, False]),
                    use_container_width=True,
                )
            else:
                st.info("No 'source' breakdown yet (need scored predictions + a source column).")

            st.markdown("### Rolling hit-rate (last 60 known)")
            if len(done) >= 10:
                d2 = done.sort_values("target_ts")
                d2["roll_hit"] = d2["hit"].rolling(60, min_periods=10).mean()
                chart = d2[["target_ts", "roll_hit"]].dropna().set_index("target_ts")
                st.line_chart(chart)
            else:
                st.info("Need at least 10 scored predictions to show a rolling hit-rate.")

            st.markdown("### Latest rows")
            st.dataframe(pred.tail(300), use_container_width=True)

            st.download_button(
                "⬇ Download predictions_log.csv",
                data=pred.to_csv(index=False).encode("utf-8"),
                file_name="predictions_log.csv",
                mime="text/csv",
                key="trk_dl_pred",
            )

            # Optional scoreboard if required columns exist
            required_cols = {"ticker", "asof_date", "ai_prob", "asof_close", "target_date"}
            if required_cols.issubset(set(pred.columns)):
                try:
                    pred_safe = ensure_ticker_col(pred.copy(), label="predictions")
                    prices_dict = get_prices_by_ticker(pred_safe["ticker"].unique().tolist())
                except (ValueError, KeyError):
                    prices_dict = {}
                score_df = evaluate_predictions(pred, prices_dict)
                st.subheader("Prediction Scoreboard (clear + colour-coded)")
                st.caption("Legend: ▲ up (green), ▼ down (red). ✅ correct / ❌ wrong. 'How Close' = model confidence in what actually happened (higher is better).")
                st.dataframe(style_scoreboard(score_df), use_container_width=True, hide_index=True)

    st.markdown("### 📰 News model monitoring")
    news_metrics = load_news_model_metrics()
    if news_metrics:
        cM1, cM2 = st.columns(2)
        with cM1:
            st.metric(
                "Holdout accuracy",
                "—" if news_metrics.get("accuracy") is None else f"{news_metrics['accuracy']*100:.1f}%",
                help="Computed on held-out labeled headlines.",
            )
        with cM2:
            st.caption(f"Last trained: {news_metrics.get('trained_at', '—')} • holdout_rows={news_metrics.get('holdout_rows', 0)}")
        per = pd.DataFrame(news_metrics.get("per_ticker", []))
        if not per.empty:
            per["hit_rate"] = (per["hit_rate"] * 100).round(1)
            per["rolling_hit"] = (per["rolling_hit"] * 100).round(1)
            st.dataframe(per.sort_values(["hit_rate", "rows"], ascending=[False, False]), use_container_width=True)
        cal = pd.DataFrame(news_metrics.get("calibration_bins", []))
        if not cal.empty:
            st.caption("Calibration bins (P(good) vs actual good-rate)")
            st.dataframe(cal, use_container_width=True)
    else:
        st.info("Train the news model to see held-out accuracy and calibration.")

    st.divider()
    st.markdown("### Backfill meta-training (relative-strength weekly picks)")
    st.caption(
        "Build historical weekly prediction/outcome rows using free Stooq prices. "
        "Ranks tickers by 4w + 1w relative momentum vs the baseline, logging top picks only."
    )

    scan_upload = st.file_uploader(
        "Optional scan_results.csv (uses ThesisLiteScore column for thesis_score)",
        type=["csv"],
        key="trk_backfill_scan",
    )
    tickers_upload = st.file_uploader(
        "Optional tickers CSV (override watchlist/universe/predictions_log tickers)",
        type=["csv"],
        key="trk_backfill_tickers",
    )

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        backfill_weeks = st.number_input("Weeks to replay (historical)", min_value=4, max_value=156, value=80, step=4)
        backfill_top_k = st.number_input("Top K tickers per week", min_value=1, max_value=40, value=10, step=1)
    with bc2:
        backfill_baseline = st.text_input("Baseline (Stooq symbol auto-mapped)", value=baseline or "SPY").strip().upper()
        backfill_source_tag = st.text_input("Source tag for rows", value="BACKFILL").strip() or "BACKFILL"
    with bc3:
        backfill_cache_dir = st.text_input("Price cache directory", value=STOOQ_CACHE_DIR)
        backfill_output_path = st.text_input(
            "Output CSV path", value=BACKFILLED_PRED_LOG, help="File to write the backfilled rows to."
        )

    def _pick_tickers_for_backfill() -> list[str]:
        tickers: list[str] = []
        tickers_df = read_uploaded_csv(tickers_upload)
        if not tickers_df.empty:
            tickers = extract_tickers_from_df(tickers_df)
        if not tickers:
            try:
                tickers = load_watchlist()["ticker"].tolist()
            except Exception:
                tickers = []
        if not tickers:
            try:
                tickers = load_universe()["ticker"].tolist()
            except Exception:
                tickers = []
        if not tickers and isinstance(pred_for_backfill, pd.DataFrame) and not pred_for_backfill.empty:
            if "ticker" in pred_for_backfill.columns:
                tickers = (
                    pred_for_backfill["ticker"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .tolist()
                )
        # De-dupe preserving order
        seen = set()
        uniq = []
        for t in tickers:
            t2 = str(t or "").strip().upper()
            if t2 and t2 not in seen:
                seen.add(t2)
                uniq.append(t2)
        return uniq

    if st.button("🏗️ Build historical predictions", type="primary", key="trk_run_relative_backfill"):
        if build_relative_strength_backfill is None or extract_tickers_from_df is None or load_thesis_scores_from_scan_df is None:
            st.warning("Backfill module not available. Please ensure market_ai/backfill_meta_training.py is present.")
            st.stop()
        tickers = _pick_tickers_for_backfill()
        if not tickers:
            st.warning("No tickers found. Add tickers to watchlist/universe or upload a CSV.")
        else:
            scan_df = read_uploaded_csv(scan_upload)
            thesis_scores = load_thesis_scores_from_scan_df(scan_df)
            os.makedirs(backfill_cache_dir or STOOQ_CACHE_DIR, exist_ok=True)
            output_path = backfill_output_path.strip() or BACKFILLED_PRED_LOG
            out_dir = os.path.dirname(output_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            try:
                with st.spinner("Downloading prices and replaying history..."):
                    bdf = build_relative_strength_backfill(
                        tickers=tickers,
                        baseline=backfill_baseline or baseline,
                        weeks=int(backfill_weeks),
                        top_k=int(backfill_top_k),
                        thesis_scores=thesis_scores,
                        cache_dir=backfill_cache_dir or STOOQ_CACHE_DIR,
                        source_tag=backfill_source_tag or "BACKFILL",
                    )

                if bdf.empty:
                    st.info("No backfilled rows generated (not enough price history?).")
                else:
                    bdf.to_csv(output_path, index=False)
                    st.session_state["rs_backfill_df"] = bdf

                    asof_min = pd.to_datetime(bdf["asof_date"]).min()
                    asof_max = pd.to_datetime(bdf["asof_date"]).max()
                    target_max = pd.to_datetime(bdf["target_date"]).max()

                    st.success(
                        f"Wrote {len(bdf)} rows for {len(bdf['ticker'].unique())} tickers "
                        f"({asof_min.date() if pd.notna(asof_min) else '?'} → "
                        f"{asof_max.date() if pd.notna(asof_max) else '?'} | "
                        f"last target {target_max.date() if pd.notna(target_max) else '?'}) "
                        f"to {output_path}."
                    )
                    st.caption("Top K per week only; all rows have outcome_known=1 for immediate meta-training.")
                    st.dataframe(bdf.tail(50), use_container_width=True)
                    st.download_button(
                        "⬇ Download backfilled CSV",
                        data=bdf.to_csv(index=False).encode("utf-8"),
                        file_name=os.path.basename(output_path),
                        mime="text/csv",
                        key="trk_dl_backfill",
                    )
            except Exception as e:
                log_exception(logger, "Backfill failed", e, show_ui=True, debug=debug_mode)

    st.divider()
    st.markdown("### Auto backfill + train meta-model (experimental)")
    st.caption(
        "Downloads historical weekly prices from Stooq for your tickers, builds a training dataset, "
        "and fits a calibrated logistic meta-model with time-series splits."
    )

    bb1, bb2, bb3 = st.columns(3)
    with bb1:
        auto_backfill_baseline = st.text_input("Baseline for training", value=baseline or "SPY").strip().upper()
        auto_backfill_weeks = st.number_input("Weeks of history", min_value=12, max_value=208, value=80, step=4)
    with bb2:
        auto_backfill_horizon = st.number_input("Horizon days", min_value=5, max_value=30, value=7, step=1)
        auto_backfill_top = st.number_input(
            "Max tickers (first N from universe/watchlist/scan)", min_value=0, max_value=2000, value=200, step=25
        )
    with bb3:
        auto_backfill_cache_dir = st.text_input("Cache dir", value=STOOQ_CACHE_DIR)
        auto_backfill_out_csv = st.text_input("Backfill CSV output", value=BACKFILL_TRAINING_CSV)
        auto_backfill_model_path = st.text_input("Model output path", value=META_MODEL_PATH)

    if st.button("🤖 Auto-backfill + Train", type="primary", key="trk_auto_backfill_train"):
        if build_backfill_dataset is None or load_tickers_auto is None or train_meta_model is None:
            st.warning("Backfill training module not available. Please ensure market_ai/backfill_ai.py is present.")
            st.stop()
        try:
            os.makedirs(auto_backfill_cache_dir or STOOQ_CACHE_DIR, exist_ok=True)
            out_dir = os.path.dirname(auto_backfill_out_csv) or "."
            os.makedirs(out_dir, exist_ok=True)

            with st.spinner("Building dataset and training meta-model..."):
                tickers_auto = load_tickers_auto()
                if auto_backfill_top > 0:
                    tickers_auto = tickers_auto[: int(auto_backfill_top)]

                if not tickers_auto:
                    st.warning("No tickers found in universe.csv/watchlist.csv/scan_results.csv.")
                    st.stop()

                dataset = build_backfill_dataset(
                    tickers=tickers_auto,
                    baseline=auto_backfill_baseline,
                    weeks=int(auto_backfill_weeks),
                    horizon_days=int(auto_backfill_horizon),
                    cache_dir=auto_backfill_cache_dir or STOOQ_CACHE_DIR,
                )
                dataset.to_csv(auto_backfill_out_csv, index=False)

                stats = train_meta_model(
                    dataset,
                    model_path=auto_backfill_model_path,
                )

            st.success(
                f"Backfill rows: {len(dataset)} | tickers: {len(dataset['ticker'].unique())} | "
                f"mean AUC: {stats['mean_auc']:.3f}"
            )
            st.dataframe(dataset.tail(50), use_container_width=True)
            st.download_button(
                "⬇ Download training CSV",
                data=dataset.to_csv(index=False).encode("utf-8"),
                file_name=os.path.basename(auto_backfill_out_csv),
                mime="text/csv",
                key="trk_dl_training_backfill",
            )
            st.caption(
                f"Model saved to {stats['model_path']} (rows={int(stats['rows'])}, last split AUC={stats['last_split_auc']:.3f})"
            )
        except Exception as e:
            log_exception(logger, "Auto backfill + train failed", e, show_ui=True, debug=debug_mode)

    st.divider()
    st.markdown("### Backfill (prices + news) to SQLite")
    st.caption("Fetches recent Stooq prices and GDELT news, stores in marketapp.db (WAL).")

    if backfill_company is None or CompanySpec is None:
        st.info("Backfill module not available (backfill.py missing).")
    else:
        if "universe_df" in globals():
            uni_df_for_backfill = universe_df
        else:
            uni_df_for_backfill = load_universe_df(UNIVERSE_CSV)

        if uni_df_for_backfill is None or uni_df_for_backfill.empty:
            st.info("Load your universe first to choose tickers.")
        else:
            tick = st.selectbox("Ticker", uni_df_for_backfill["ticker"].tolist(), key="bk_ticker_select")
            row = uni_df_for_backfill.loc[uni_df_for_backfill["ticker"] == tick].iloc[0]

            spec = CompanySpec(
                ticker=row["ticker"],
                name=row.get("company_name") or row.get("name") or row["ticker"],
                aliases=parse_aliases(row.get("aliases", "")),
                stooq_symbol=row.get("stooq_symbol") or f"{row['ticker'].lower()}.us",
            )

            news_days = st.slider("News backfill days (recent)", 7, 120, 90, key="bk_news_days")
            if st.button("Run backfill now", key="bk_run_now"):
                try:
                    with st.spinner("Backfilling prices + news..."):
                        res = backfill_company(spec, news_days=news_days)
                    st.success(
                        f"{res['ticker']}: +{res['prices_added']} price rows, +{res['news_added']} news rows (marketapp.db)."
                    )
                except Exception as e:
                    log_exception(logger, "Backfill prices+news failed", e, show_ui=True, debug=debug_mode)


# ==========================================================
# TAB: Chat
# ==========================================================
with tabs[9]:
    st.subheader("💬 Chat (optional local LLM via Ollama)")
    st.caption("This is an explainer only. No financial advice.")

    if not ollama_on or not ollama_ok:
        st.info("Enable Ollama in the sidebar to use this tab.")
    else:
        last_scan = st.session_state.get("last_scan_df", None)
        if isinstance(last_scan, pd.DataFrame) and not last_scan.empty:
            st.markdown("### Explain last scan (top rows)")
            n = st.slider("Rows to explain", 3, min(25, len(last_scan)), min(10, len(last_scan)), 1, key="chat_rows")
            if st.button("Explain scan with Ollama", type="primary", key="chat_explain_btn"):
                try:
                    prompt = build_ollama_explain_prompt(last_scan, baseline=baseline, max_rows=int(n))
                    system = "You are a market dashboard explainer. You DO NOT give financial advice. You interpret signals only."
                    with st.spinner("Thinking..."):
                        ans = ollama_chat(
                            base_url=ollama_url,
                            model=ollama_model,
                            system=system,
                            user=prompt,
                            temperature=0.2,
                            timeout=180,
                        )
                    st.write(ans if ans else "No response from Ollama.")
                except Exception as e:
                    st.warning(str(e))
        else:
            st.info("Run a scan first so there’s something to explain.")

        st.divider()
        st.markdown("### Free chat")
        if "ollama_chat_hist" not in st.session_state:
            st.session_state["ollama_chat_hist"] = []

        for msg in st.session_state["ollama_chat_hist"][-12:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_msg = st.chat_input("Ask about your dashboard / signals (no trading advice).")
        if user_msg:
            st.session_state["ollama_chat_hist"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)

            ctx = ""
            last_scan = st.session_state.get("last_scan_df", None)
            if isinstance(last_scan, pd.DataFrame) and not last_scan.empty:
                ctx = build_ollama_explain_prompt(last_scan, baseline=baseline, max_rows=8)

            system = "You explain market dashboard signals in plain English. No financial advice, no price targets."
            user = f"{ctx}\n\nUSER QUESTION:\n{user_msg}".strip()

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ans = ollama_chat(
                        base_url=ollama_url,
                        model=ollama_model,
                        system=system,
                        user=user,
                        temperature=0.2,
                        timeout=180,
                    )
                st.write(ans if ans else "No response from Ollama.")
            st.session_state["ollama_chat_hist"].append({"role": "assistant", "content": ans or "No response."})


# ==========================================================
# TAB: Logs / Files
# ==========================================================
with tabs[10]:
    st.subheader("🪵 Logs / Files")
    st.caption("Debug events, error logs, and CSV files created by the app.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Debug events (last ~250)")
        ev = st.session_state.get("_debug_events", [])
        if ev:
            st.dataframe(pd.DataFrame(ev).tail(250), use_container_width=True)
        else:
            st.info("No debug events yet.")
    with c2:
        st.markdown("### app_debug.log (tail)")
        tail = tail_text_file(DEBUG_LOG_PATH, n_lines=200)
        if tail:
            st.code(tail)
        else:
            st.info("No log yet.")
    with c3:
        st.markdown("### errors_log.csv (tail)")
        if os.path.exists(ERRORS_CSV):
            edf = safe_read_log_csv(ERRORS_CSV)
            st.dataframe(edf.tail(50), use_container_width=True)
        else:
            st.info("No errors_log.csv yet.")

    st.divider()
    st.markdown("### Download files")
    files = []
    for p in [UNIVERSE_CSV, WATCHLIST_CSV, MODEL_PATH, FEATURES_PATH, META_MODEL_PATH, PRED_LOG, NEWS_LOG, TRAIN_METRICS, META_TRAIN_METRICS, DEBUG_LOG_PATH, ERRORS_CSV]:
        if os.path.exists(p):
            files.append(p)

    if not files:
        st.info("No files to download yet.")
    else:
        for p in files:
            try:
                with open(p, "rb") as f:
                    data = f.read()
                st.download_button(
                    f"⬇ {p}",
                    data=data,
                    file_name=os.path.basename(p),
                    mime="application/octet-stream",
                    key=f"dl_{p}",
                )
            except Exception:
                st.write(p)

    st.divider()
    st.markdown("### Logs directory listing")
    try:
        if os.path.exists(LOG_DIR):
            items = []
            for name in sorted(os.listdir(LOG_DIR)):
                fp = os.path.join(LOG_DIR, name)
                if os.path.isfile(fp):
                    items.append({"file": name, "bytes": os.path.getsize(fp), "modified": datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M:%S")})
            st.dataframe(pd.DataFrame(items), use_container_width=True)
        else:
            st.info("logs/ directory not created yet.")
    except Exception as e:
        st.warning(str(e))
