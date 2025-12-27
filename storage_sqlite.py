from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

DB_PATH = Path("marketapp.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS prices (
        ticker TEXT NOT NULL,
        date   TEXT NOT NULL,      -- YYYY-MM-DD
        open   REAL,
        high   REAL,
        low    REAL,
        close  REAL,
        volume REAL,
        source TEXT NOT NULL,
        PRIMARY KEY (ticker, date, source)
    );
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS news (
        ticker       TEXT NOT NULL,
        published_at TEXT NOT NULL, -- ISO8601
        title        TEXT NOT NULL,
        url          TEXT NOT NULL,
        source       TEXT NOT NULL,
        snippet      TEXT,
        match_score  INTEGER NOT NULL,
        PRIMARY KEY (ticker, url)
    );
    """
    )

    conn.commit()
    conn.close()


def upsert_prices(ticker: str, df: pd.DataFrame, source: str) -> int:
    """
    df columns expected: Date, Open, High, Low, Close, Volume (case-insensitive ok)
    """
    if df is None or df.empty:
        return 0

    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", "Date")
    o = cols.get("open", "Open")
    h = cols.get("high", "High")
    l = cols.get("low", "Low")
    c = cols.get("close", "Close")
    v = cols.get("volume", "Volume") if "volume" in cols else None

    rows = []
    for _, r in df.iterrows():
        d = str(r[date_col])[:10]
        rows.append(
            (
                ticker.upper(),
                d,
                float(r[o]) if pd.notna(r[o]) else None,
                float(r[h]) if pd.notna(r[h]) else None,
                float(r[l]) if pd.notna(r[l]) else None,
                float(r[c]) if pd.notna(r[c]) else None,
                float(r[v]) if (v and pd.notna(r[v])) else None,
                source,
            )
        )

    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """,
        rows,
    )
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def upsert_news(ticker: str, items: Iterable[Dict[str, Any]], source: str) -> int:
    rows = []
    for it in items:
        rows.append(
            (
                ticker.upper(),
                it["published_at"],
                it["title"],
                it["url"],
                source,
                it.get("snippet"),
                int(it.get("match_score", 0)),
            )
        )
    if not rows:
        return 0

    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO news (ticker, published_at, title, url, source, snippet, match_score)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """,
        rows,
    )
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def latest_price_date(ticker: str, source: str) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT MAX(date) FROM prices WHERE ticker=? AND source=?;
    """,
        (ticker.upper(), source),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def latest_news_time(ticker: str, source: str) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT MAX(published_at) FROM news WHERE ticker=? AND source=?;
    """,
        (ticker.upper(), source),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else None
