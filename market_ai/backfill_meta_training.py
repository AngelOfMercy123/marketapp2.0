from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

DEFAULT_CACHE_DIR = "data_stooq_cache"
DEFAULT_MAX_AGE_HOURS = 24


@dataclass
class BackfillRow:
    timestamp: str
    source: str
    baseline: str
    ticker: str
    asof_date: str
    target_date: str
    action: str
    ai_prob: float
    thesis_score: float
    pred_outperform: float
    outcome_known: int
    actual_outperform: float
    hit: int


EXPECTED_COLS: list[str] = [
    "timestamp",
    "source",
    "baseline",
    "ticker",
    "asof_date",
    "target_date",
    "action",
    "ai_prob",
    "thesis_score",
    "pred_outperform",
    "outcome_known",
    "actual_outperform",
    "hit",
]


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def last_friday(d: date) -> date:
    return d - timedelta(days=(d.weekday() - 4) % 7)


def to_stooq_symbol(ticker: str) -> str:
    t = (ticker or "").strip()
    if not t:
        return ""
    if "." in t:
        return t.lower()
    u = t.upper()
    if u == "SPY":
        return "spy.us"
    return f"{t.lower()}.us"


def stooq_download_daily(
    symbol: str, cache_dir: str = DEFAULT_CACHE_DIR, max_age_hours: int = DEFAULT_MAX_AGE_HOURS
) -> Optional[pd.Series]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}.csv")

    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - mtime) < timedelta(hours=max_age_hours):
            try:
                df = pd.read_csv(cache_path)
                if {"Date", "Close"}.issubset(df.columns):
                    df["Date"] = pd.to_datetime(df["Date"])
                    return df.set_index("Date")["Close"].sort_index()
            except Exception:
                pass

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "Date" not in df.columns or "Close" not in df.columns:
            return None
        df.to_csv(cache_path, index=False)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")["Close"].sort_index()
    except Exception:
        return None


def close_on_or_before(close: pd.Series, d: date) -> float:
    if close is None or close.empty:
        return float("nan")
    ts = pd.Timestamp(d)
    idx = close.index[close.index <= ts]
    if len(idx) == 0:
        return float("nan")
    return float(close.loc[idx[-1]])


def pct_return(start: float, end: float) -> float:
    if not (math.isfinite(start) and math.isfinite(end)) or start == 0:
        return float("nan")
    return (end / start - 1.0) * 100.0


def safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_tickers(tickers: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        t2 = str(t or "").upper().strip()
        if not t2 or t2 in seen:
            continue
        seen.add(t2)
        out.append(t2)
    return out


def extract_tickers_from_df(df: pd.DataFrame, cols: Sequence[str] | None = None) -> list[str]:
    cols = cols or ["Ticker", "ticker", "symbol", "Symbol"]
    if df is None or df.empty:
        return []
    for c in cols:
        if c in df.columns:
            return normalize_tickers(df[c].dropna().tolist())
    return []


def load_thesis_scores_from_scan_df(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    if "Ticker" not in df.columns:
        return {}
    if "ThesisLiteScore" in df.columns:
        out = {}
        for _, r in df.iterrows():
            t = str(r["Ticker"]).strip().upper()
            out[t] = safe_float(r["ThesisLiteScore"], 5.0) * 10.0
        return out
    return {str(t).strip().upper(): 50.0 for t in df["Ticker"].dropna().unique().tolist()}


def compute_signal_relative_strength(close_t: pd.Series, close_b: pd.Series, asof: date) -> Tuple[float, float, float]:
    d_1w = asof - timedelta(days=7)
    d_4w = asof - timedelta(days=28)

    t_now = close_on_or_before(close_t, asof)
    t_1w = close_on_or_before(close_t, d_1w)
    t_4w = close_on_or_before(close_t, d_4w)

    b_now = close_on_or_before(close_b, asof)
    b_1w = close_on_or_before(close_b, d_1w)
    b_4w = close_on_or_before(close_b, d_4w)

    t_r1 = pct_return(t_1w, t_now)
    b_r1 = pct_return(b_1w, b_now)
    t_r4 = pct_return(t_4w, t_now)
    b_r4 = pct_return(b_4w, b_now)

    rel_1w = t_r1 - b_r1
    rel_4w = t_r4 - b_r4
    combined = 0.35 * rel_1w + 0.65 * rel_4w
    return rel_1w, rel_4w, combined


def build_relative_strength_backfill(
    tickers: Sequence[str],
    baseline: str,
    weeks: int,
    top_k: int,
    thesis_scores: Optional[Dict[str, float]] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
    source_tag: str = "BACKFILL",
) -> pd.DataFrame:
    tickers_norm = normalize_tickers(tickers)
    if not tickers_norm:
        raise ValueError("No tickers provided for backfill.")

    baseline_sym = to_stooq_symbol(baseline)
    close_b = stooq_download_daily(baseline_sym, cache_dir=cache_dir, max_age_hours=max_age_hours)
    if close_b is None or close_b.empty:
        raise RuntimeError(f"Could not download baseline prices from Stooq for {baseline} ({baseline_sym}).")

    closes: Dict[str, pd.Series] = {}
    for t in sorted(set(tickers_norm)):
        sym = to_stooq_symbol(t)
        s = stooq_download_daily(sym, cache_dir=cache_dir, max_age_hours=max_age_hours)
        if s is None or s.empty:
            continue
        closes[t.upper()] = s

    if not closes:
        raise RuntimeError("No ticker price series could be downloaded from Stooq. Check tickers/symbol mapping.")

    today = date.today()
    end_asof = last_friday(today)
    asofs = [end_asof - timedelta(days=7 * i) for i in range(1, weeks + 1)]

    rows: List[BackfillRow] = []
    thesis_scores = thesis_scores or {}

    for asof in asofs:
        target = asof + timedelta(days=7)
        scored = []
        for t, close_t in closes.items():
            _, _, combined = compute_signal_relative_strength(close_t, close_b, asof)
            if math.isfinite(combined):
                scored.append((t, combined))

        if not scored:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        picked = scored[:top_k]

        for t, combined in picked:
            pred_out = combined
            ai_prob = sigmoid(pred_out / 3.0)

            t_start = close_on_or_before(closes[t], asof)
            t_end = close_on_or_before(closes[t], target)
            b_start = close_on_or_before(close_b, asof)
            b_end = close_on_or_before(close_b, target)

            t_ret = pct_return(t_start, t_end)
            b_ret = pct_return(b_start, b_end)
            actual_out = t_ret - b_ret
            if not math.isfinite(actual_out):
                continue

            hit = int((pred_out >= 0) == (actual_out >= 0))
            th = thesis_scores.get(t.upper(), 50.0)
            action = "BUY" if ai_prob >= 0.60 else "WATCH"

            rows.append(
                BackfillRow(
                    timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    source=source_tag,
                    baseline=baseline.upper(),
                    ticker=t.upper(),
                    asof_date=asof.isoformat(),
                    target_date=target.isoformat(),
                    action=action,
                    ai_prob=float(ai_prob),
                    thesis_score=float(th),
                    pred_outperform=float(pred_out),
                    outcome_known=1,
                    actual_outperform=float(actual_out),
                    hit=hit,
                )
            )

    out_df = pd.DataFrame([r.__dict__ for r in rows])
    if out_df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)
    return out_df[EXPECTED_COLS].sort_values(["asof_date", "ticker"]).reset_index(drop=True)
