"""Build historical weekly features/outcomes from Stooq and train a meta-model."""
from __future__ import annotations

import io
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_CACHE_DIR = "data_stooq_cache"


def to_stooq_symbol(ticker: str) -> str:
    """Convert ticker to Stooq symbol (adds .us default suffix)."""
    t = (ticker or "").strip()
    if not t:
        return ""
    if "." in t:
        return t.lower()

    u = t.upper()
    if u == "SPY":
        return "spy.us"
    if u == "QQQ":
        return "qqq.us"
    if u == "IWM":
        return "iwm.us"

    return f"{t.lower()}.us"


def stooq_download_daily(symbol: str, cache_dir: str = DEFAULT_CACHE_DIR, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """Download daily OHLC from Stooq with simple disk cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}.csv")

    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - mtime) < timedelta(hours=max_age_hours):
            try:
                df = pd.read_csv(cache_path)
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                return df
            except Exception:
                pass

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "Date" not in df.columns:
            return None
        df.to_csv(cache_path, index=False)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        return None


def last_friday(d: date) -> date:
    """Return the most recent Friday on/before date."""
    return d - timedelta(days=(d.weekday() - 4) % 7)


def close_on_or_before(df: pd.DataFrame, d: date) -> float:
    """Return closing price on/before date."""
    if df is None or df.empty:
        return float("nan")
    ts = pd.Timestamp(d)
    sub = df[df["Date"] <= ts]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1]["Close"])


def pct_return(start: float, end: float) -> float:
    if not (math.isfinite(start) and math.isfinite(end)) or start == 0:
        return float("nan")
    return (end / start - 1.0) * 100.0


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


@dataclass
class BackfillRow:
    ticker: str
    baseline: str
    asof_date: str
    target_date: str
    actual_outperform: float
    y_outperform: int
    rel_1w: float
    rel_4w: float
    rel_12w: float
    signal: float
    t_r1: float
    t_r4: float
    t_r12: float


def features_for_week(t_df: pd.DataFrame, b_df: pd.DataFrame, asof: date) -> Dict[str, float]:
    """Compute relative-strength features using prices up to as-of date."""
    d1 = asof - timedelta(days=7)
    d4 = asof - timedelta(days=28)
    d12 = asof - timedelta(days=84)

    t_now = close_on_or_before(t_df, asof)
    t_1w = close_on_or_before(t_df, d1)
    t_4w = close_on_or_before(t_df, d4)
    t_12w = close_on_or_before(t_df, d12)

    b_now = close_on_or_before(b_df, asof)
    b_1w = close_on_or_before(b_df, d1)
    b_4w = close_on_or_before(b_df, d4)
    b_12w = close_on_or_before(b_df, d12)

    t_r1 = pct_return(t_1w, t_now)
    t_r4 = pct_return(t_4w, t_now)
    t_r12 = pct_return(t_12w, t_now)

    b_r1 = pct_return(b_1w, b_now)
    b_r4 = pct_return(b_4w, b_now)
    b_r12 = pct_return(b_12w, b_now)

    rel1 = t_r1 - b_r1
    rel4 = t_r4 - b_r4
    rel12 = t_r12 - b_r12

    signal = 0.35 * rel1 + 0.45 * rel4 + 0.20 * rel12

    return {
        "rel_1w": rel1,
        "rel_4w": rel4,
        "rel_12w": rel12,
        "signal": signal,
        "t_r1": t_r1,
        "t_r4": t_r4,
        "t_r12": t_r12,
    }


def label_outperformance(t_df: pd.DataFrame, b_df: pd.DataFrame, asof: date, horizon_days: int = 7) -> Tuple[float, int]:
    """Label whether ticker beat baseline over the horizon."""
    target = asof + timedelta(days=horizon_days)

    t_start = close_on_or_before(t_df, asof)
    t_end = close_on_or_before(t_df, target)

    b_start = close_on_or_before(b_df, asof)
    b_end = close_on_or_before(b_df, target)

    t_ret = pct_return(t_start, t_end)
    b_ret = pct_return(b_start, b_end)
    actual_out = t_ret - b_ret

    if not math.isfinite(actual_out):
        return float("nan"), 0

    y = 1 if actual_out >= 0 else 0
    return actual_out, y


def load_tickers_auto(universe_csv: str = "universe.csv", watchlist_csv: str = "watchlist.csv", scan_csv: str = "scan_results.csv") -> List[str]:
    """Load tickers from multiple CSV sources (unique, uppercase)."""
    tickers = []
    seen = set()
    for path in [universe_csv, watchlist_csv, scan_csv]:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, engine="python")
        except Exception:
            continue
        for col in ["ticker", "Ticker", "symbol", "Symbol"]:
            if col in df.columns:
                for x in df[col].dropna().tolist():
                    t = str(x).strip().upper()
                    if t and t not in seen:
                        seen.add(t)
                        tickers.append(t)
                break
    return tickers


def build_backfill_dataset(
    tickers: Sequence[str],
    baseline: str = "SPY",
    weeks: int = 80,
    horizon_days: int = 7,
    cache_dir: str = DEFAULT_CACHE_DIR,
    rate_sleep: float = 0.0,
) -> pd.DataFrame:
    """Replay historical weeks to build features/outcomes."""
    b_sym = to_stooq_symbol(baseline)
    b_df = stooq_download_daily(b_sym, cache_dir=cache_dir)
    if b_df is None or b_df.empty:
        raise RuntimeError(f"Failed baseline download: {baseline} ({b_sym})")

    rows: List[BackfillRow] = []
    today = date.today()
    end_asof = last_friday(today)

    for t in tickers:
        sym = to_stooq_symbol(t)
        t_df = stooq_download_daily(sym, cache_dir=cache_dir)
        if t_df is None or t_df.empty:
            continue

        for i in range(1, weeks + 1):
            asof = end_asof - timedelta(days=7 * i)

            feats = features_for_week(t_df, b_df, asof)
            actual_out, y = label_outperformance(t_df, b_df, asof, horizon_days=horizon_days)
            if not math.isfinite(actual_out):
                continue

            rows.append(
                BackfillRow(
                    ticker=t.upper(),
                    baseline=baseline.upper(),
                    asof_date=asof.isoformat(),
                    target_date=(asof + timedelta(days=horizon_days)).isoformat(),
                    actual_outperform=float(actual_out),
                    y_outperform=int(y),
                    rel_1w=float(feats["rel_1w"]),
                    rel_4w=float(feats["rel_4w"]),
                    rel_12w=float(feats["rel_12w"]),
                    signal=float(feats["signal"]),
                    t_r1=float(feats["t_r1"]),
                    t_r4=float(feats["t_r4"]),
                    t_r12=float(feats["t_r12"]),
                )
            )

        if rate_sleep > 0:
            time.sleep(rate_sleep)

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        raise RuntimeError("Backfill produced 0 rows. Check tickers/provider coverage.")
    df = df.sort_values(["asof_date", "ticker"]).reset_index(drop=True)
    return df


def train_meta_model(df: pd.DataFrame, model_path: str = "meta_model.joblib", feature_cols: Optional[List[str]] = None) -> Dict[str, float]:
    """Train a calibrated logistic meta-model on backfilled features."""
    if feature_cols is None:
        feature_cols = ["rel_1w", "rel_4w", "rel_12w", "signal", "t_r1", "t_r4", "t_r12"]

    df2 = df.dropna(subset=feature_cols + ["y_outperform"]).copy()
    if len(df2) < 200:
        raise RuntimeError(f"Not enough rows to train reliably (need >=200, have {len(df2)}).")

    X = df2[feature_cols].values
    y = df2["y_outperform"].astype(int).values

    tscv = TimeSeriesSplit(n_splits=5)
    aucs: List[float] = []

    for train_idx, test_idx in tscv.split(X):
        base = LogisticRegression(max_iter=200, n_jobs=None)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        try:
            clf.fit(X[train_idx], y[train_idx])
            p = clf.predict_proba(X[test_idx])[:, 1]
            auc = roc_auc_score(y[test_idx], p)
            aucs.append(float(auc))
        except Exception:
            continue

    base = LogisticRegression(max_iter=200, n_jobs=None)
    try:
        final: LogisticRegression | CalibratedClassifierCV
        final = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        final.fit(X, y)
    except Exception:
        final = base
        final.fit(X, y)
    dump({"model": final, "feature_cols": feature_cols}, model_path)

    return {
        "rows": float(len(df2)),
        "mean_auc": float(sum(aucs) / len(aucs)) if aucs else float("nan"),
        "last_split_auc": float(aucs[-1]) if aucs else float("nan"),
        "model_path": model_path,
    }


def auto_backfill_and_train(
    baseline: str = "SPY",
    weeks: int = 80,
    top_limit: Optional[int] = None,
    horizon_days: int = 7,
    out_backfill_csv: str = "backfill_training.csv",
    model_path: str = "meta_model.joblib",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[str, float]:
    """Convenience: load tickers, build dataset, save CSV, and train meta-model."""
    tickers = load_tickers_auto()
    if top_limit is not None:
        tickers = tickers[:top_limit]

    df = build_backfill_dataset(
        tickers=tickers,
        baseline=baseline,
        weeks=weeks,
        horizon_days=horizon_days,
        cache_dir=cache_dir,
    )
    df.to_csv(out_backfill_csv, index=False)

    stats = train_meta_model(df, model_path=model_path)
    stats["tickers_used"] = float(len(set(df["ticker"].tolist())))
    stats["backfill_csv"] = out_backfill_csv
    stats["weeks"] = float(weeks)
    stats["baseline"] = baseline.upper()
    return stats
