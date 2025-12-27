"""
Backfill training data by generating historical predictions and outcomes.
This creates synthetic training examples from past price data.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


def parse_yyyy_mm_dd(s: str) -> Optional[date]:
    """Parse YYYY-MM-DD string to date."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def today_utc_date() -> date:
    """Get today's date in UTC."""
    return datetime.utcnow().date()


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    """Stable sigmoid function."""
    if x >= 0:
        z = pow(2.718281828, -x)
        return 1.0 / (1.0 + z)
    z = pow(2.718281828, x)
    return z / (1.0 + z)


def ensure_dir(p: str) -> None:
    """Create directory if it doesn't exist."""
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def to_stooq_symbol(ticker: str, default_suffix: str = ".us") -> str:
    """
    Convert ticker to Stooq format.
    AAPL -> aapl.us
    BRK.B -> brk-b.us
    """
    t = (ticker or "").strip()
    if not t:
        return t
    t = t.replace(".", "-")
    t = t.lower()

    if t.endswith((".us", ".uk", ".de", ".fr", ".pl", ".jp", ".hk")):
        return t
    return t + default_suffix


@dataclass
class PriceSeries:
    """Wrapper for price dataframe."""
    df: pd.DataFrame  # indexed by date, has 'close' column


class StooqCache:
    """Cache for Stooq price data."""

    def __init__(self, cache_dir: str, polite_sleep_s: float = 0.25) -> None:
        self.cache_dir = cache_dir
        self.polite_sleep_s = polite_sleep_s
        ensure_dir(cache_dir)
        self._mem: Dict[str, PriceSeries] = {}

    def _cache_path(self, stooq_symbol: str) -> str:
        safe = stooq_symbol.replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe}.csv")

    def get_daily(self, ticker: str) -> Optional[PriceSeries]:
        """Get daily price series for ticker."""
        stooq_symbol = to_stooq_symbol(ticker)
        if not stooq_symbol:
            return None

        if stooq_symbol in self._mem:
            return self._mem[stooq_symbol]

        path = self._cache_path(stooq_symbol)
        df = None

        # Try disk cache first
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None

        # Fetch if needed
        if df is None or df.empty:
            url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
            try:
                time.sleep(self.polite_sleep_s)
                r = requests.get(url, timeout=20, headers={"User-Agent": "marketapp-backfill/1.0"})
                r.raise_for_status()
                text = r.text
                if "404 Not Found" in text or len(text) < 50:
                    return None
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(text)
                df = pd.read_csv(path)
            except Exception:
                return None

        # Normalize
        if "Date" not in df.columns:
            return None

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.set_index("Date")

        if "Close" not in df.columns:
            return None
        out = pd.DataFrame({"close": pd.to_numeric(df["Close"], errors="coerce")})
        out = out.dropna(subset=["close"])
        if out.empty:
            return None

        ps = PriceSeries(df=out)
        self._mem[stooq_symbol] = ps
        return ps

    @staticmethod
    def close_on_or_before(ps: PriceSeries, d: date) -> Optional[float]:
        """Get close price on or before date."""
        if ps is None or ps.df is None or ps.df.empty:
            return None
        idx = ps.df.index
        ts = pd.Timestamp(d)
        loc = idx.searchsorted(ts, side="right") - 1
        if loc < 0:
            return None
        return float(ps.df.iloc[loc]["close"])

    @staticmethod
    def window(ps: PriceSeries, end_d: date, lookback_days: int) -> Optional[pd.Series]:
        """Get price window ending at end_d."""
        if ps is None or ps.df is None or ps.df.empty:
            return None
        end_ts = pd.Timestamp(end_d)
        start_ts = end_ts - pd.Timedelta(days=lookback_days)
        w = ps.df.loc[(ps.df.index >= start_ts) & (ps.df.index <= end_ts), "close"]
        if w is None or w.empty or len(w) < 5:
            return None
        return w


def compute_retro_scores(
    cache: StooqCache,
    ticker: str,
    baseline: str,
    asof_d: date,
) -> Optional[Tuple[float, float, float, str, int]]:
    """
    Compute retrospective scores for a ticker.
    Returns: (ai_prob_0_100, thesis_score_0_100, combined_0_1, action, pred_outperform_0_1)
    """
    ps_t = cache.get_daily(ticker)
    ps_b = cache.get_daily(baseline)
    if ps_t is None or ps_b is None:
        return None

    # Get 1w and 4w windows
    w_t_7 = cache.window(ps_t, asof_d, 10)
    w_b_7 = cache.window(ps_b, asof_d, 10)
    w_t_30 = cache.window(ps_t, asof_d, 40)
    w_b_30 = cache.window(ps_b, asof_d, 40)
    if w_t_7 is None or w_b_7 is None or w_t_30 is None or w_b_30 is None:
        return None

    # Simple returns
    ret_t_1w = (w_t_7.iloc[-1] / w_t_7.iloc[0]) - 1.0
    ret_b_1w = (w_b_7.iloc[-1] / w_b_7.iloc[0]) - 1.0
    ret_t_4w = (w_t_30.iloc[-1] / w_t_30.iloc[0]) - 1.0
    ret_b_4w = (w_b_30.iloc[-1] / w_b_30.iloc[0]) - 1.0

    excess_1w = ret_t_1w - ret_b_1w
    excess_4w = ret_t_4w - ret_b_4w

    # Volatility proxy
    r_t = w_t_30.pct_change().dropna()
    vol_4w = float(r_t.std()) if len(r_t) > 5 else 0.0

    # Heuristic "AI prob"
    x = (6.0 * excess_1w) + (3.0 * excess_4w) - (2.0 * vol_4w)
    p = sigmoid(20.0 * x)
    ai_prob = clamp(p * 100.0, 0.0, 100.0)

    # Heuristic "thesis score"
    thesis_score = clamp(50.0 + (800.0 * excess_4w) - (200.0 * vol_4w), 0.0, 100.0)

    # Combined for decision
    combined = (0.6 * (thesis_score / 100.0)) + (0.4 * (ai_prob / 100.0))

    if combined >= 0.60:
        action = "BUY"
    elif combined >= 0.50:
        action = "WATCH"
    else:
        action = "AVOID"

    pred_outperform = 1 if combined >= 0.55 else 0
    return ai_prob, thesis_score, combined, action, pred_outperform


def compute_actual_outperform(
    cache: StooqCache,
    ticker: str,
    baseline: str,
    asof_d: date,
    target_d: date,
) -> Optional[float]:
    """Compute actual outperformance vs baseline."""
    ps_t = cache.get_daily(ticker)
    ps_b = cache.get_daily(baseline)
    if ps_t is None or ps_b is None:
        return None

    c0_t = cache.close_on_or_before(ps_t, asof_d)
    c1_t = cache.close_on_or_before(ps_t, target_d)
    c0_b = cache.close_on_or_before(ps_b, asof_d)
    c1_b = cache.close_on_or_before(ps_b, target_d)
    if None in (c0_t, c1_t, c0_b, c1_b):
        return None

    ret_t = (c1_t / c0_t) - 1.0
    ret_b = (c1_b / c0_b) - 1.0
    return float(ret_t - ret_b)


EXPECTED_COLS = [
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


def read_predictions_log(path: str) -> Tuple[List[str], List[dict]]:
    """Read predictions log CSV."""
    if not os.path.exists(path):
        return EXPECTED_COLS, []

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or EXPECTED_COLS
        rows = list(reader)
        return cols, rows


def write_predictions_log(path: str, cols: List[str], rows: List[dict]) -> None:
    """Write predictions log CSV."""
    colset = set(cols)
    out_cols = [c for c in EXPECTED_COLS if c in colset] + [c for c in cols if c not in EXPECTED_COLS]

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in out_cols})
    os.replace(tmp, path)


def row_key(r: dict) -> Tuple[str, str, str, str]:
    """Generate unique key for prediction row."""
    return (
        (r.get("source") or "").strip(),
        (r.get("ticker") or "").strip(),
        (r.get("asof_date") or "").strip(),
        (r.get("target_date") or "").strip(),
    )


def backfill_outcomes_inplace(
    rows: List[dict],
    cache: StooqCache,
    baseline: str,
    now_d: date,
) -> int:
    """Backfill outcomes for existing predictions."""
    changed = 0
    for r in rows:
        tkr = (r.get("ticker") or "").strip()
        if not tkr:
            continue

        asof_d = parse_yyyy_mm_dd(r.get("asof_date", ""))
        target_d = parse_yyyy_mm_dd(r.get("target_date", ""))
        if asof_d is None or target_d is None:
            continue

        outcome_known = str(r.get("outcome_known", "")).strip()
        if target_d > now_d:
            continue
        if outcome_known == "1":
            continue

        actual = compute_actual_outperform(cache, tkr, baseline, asof_d, target_d)
        if actual is None:
            continue

        r["actual_outperform"] = f"{actual:.6f}"
        r["outcome_known"] = "1"

        try:
            pred = int(float(str(r.get("pred_outperform", "0") or "0").strip()))
        except Exception:
            pred = 0
        actual_pos = 1 if actual > 0 else 0
        r["hit"] = "1" if pred == actual_pos else "0"

        changed += 1

    return changed


def generate_weekly_fridays(end_d: date, weeks_back: int) -> List[date]:
    """Generate list of Fridays going back N weeks."""
    d = end_d
    while d.weekday() != 4:  # Friday
        d -= timedelta(days=1)

    fridays = []
    for _ in range(weeks_back):
        fridays.append(d)
        d -= timedelta(days=7)
    fridays.reverse()
    return fridays


def generate_retro_rows(
    existing_keys: set,
    tickers: List[str],
    baseline: str,
    weeks_back: int,
    cache: StooqCache,
    now_d: date,
    source_tag: str = "retro",
) -> List[dict]:
    """Generate retrospective prediction rows."""
    fridays = generate_weekly_fridays(now_d, weeks_back)

    new_rows: List[dict] = []
    for asof_d in fridays:
        target_d = asof_d + timedelta(days=7)

        if target_d > now_d:
            continue

        for tkr in tickers:
            key = (source_tag, tkr, asof_d.isoformat(), target_d.isoformat())
            if key in existing_keys:
                continue

            scores = compute_retro_scores(cache, tkr, baseline, asof_d)
            if scores is None:
                continue
            ai_prob, thesis_score, _combined, action, pred_outperform = scores

            actual = compute_actual_outperform(cache, tkr, baseline, asof_d, target_d)
            if actual is None:
                continue

            actual_pos = 1 if actual > 0 else 0
            hit = 1 if pred_outperform == actual_pos else 0

            new_rows.append(
                {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": source_tag,
                    "baseline": baseline,
                    "ticker": tkr,
                    "asof_date": asof_d.isoformat(),
                    "target_date": target_d.isoformat(),
                    "action": action,
                    "ai_prob": f"{ai_prob:.2f}",
                    "thesis_score": f"{thesis_score:.2f}",
                    "pred_outperform": str(int(pred_outperform)),
                    "outcome_known": "1",
                    "actual_outperform": f"{actual:.6f}",
                    "hit": str(int(hit)),
                }
            )

    return new_rows


def load_tickers_from_csv(path: str, col: str = "ticker") -> List[str]:
    """Load tickers from CSV file."""
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if col in df.columns:
            tickers = [str(x).strip() for x in df[col].dropna().tolist()]
            tickers = [t for t in tickers if t]
            # De-dupe keeping order
            seen = set()
            out = []
            for t in tickers:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out
    except Exception:
        return []
    return []


def run_backfill(
    pred_log_path: str,
    cache_dir: str,
    baseline: str,
    weeks_back: int,
    max_tickers: int,
    watchlist_path: str,
    universe_path: str,
    source_tag: str = "retro",
    create_backup: bool = True,
) -> dict:
    """
    Run backfill operation.
    Returns dict with stats: updated, added, total, known
    """
    now_d = today_utc_date()

    cols, rows = read_predictions_log(pred_log_path)
    if not cols:
        cols = EXPECTED_COLS

    # Optional backup
    if create_backup and os.path.exists(pred_log_path):
        bak = pred_log_path + ".bak"
        if not os.path.exists(bak):
            with open(pred_log_path, "rb") as src, open(bak, "wb") as dst:
                dst.write(src.read())

    cache = StooqCache(cache_dir=cache_dir)

    # Backfill outcomes for existing rows
    updated = backfill_outcomes_inplace(rows, cache, baseline, now_d)

    # Choose tickers
    tickers = load_tickers_from_csv(watchlist_path, "ticker")
    if not tickers:
        tickers = load_tickers_from_csv(universe_path, "ticker")
    if not tickers:
        # Fallback: tickers already in log
        tickers = sorted({(r.get("ticker") or "").strip() for r in rows if (r.get("ticker") or "").strip()})

    if max_tickers > 0:
        tickers = tickers[:max_tickers]

    # Build existing key set
    existing_keys = set()
    for r in rows:
        existing_keys.add(row_key(r))

    # Generate retro rows
    new_rows = generate_retro_rows(
        existing_keys=existing_keys,
        tickers=tickers,
        baseline=baseline,
        weeks_back=weeks_back,
        cache=cache,
        now_d=now_d,
        source_tag=source_tag,
    )

    rows.extend(new_rows)

    # Ensure expected columns
    colset = set(cols)
    for c in EXPECTED_COLS:
        if c not in colset:
            cols.append(c)
            colset.add(c)

    write_predictions_log(pred_log_path, cols, rows)

    known = sum(1 for r in rows if str(r.get("outcome_known", "")).strip() == "1")

    return {
        "updated": updated,
        "added": len(new_rows),
        "total": len(rows),
        "known": known,
    }
