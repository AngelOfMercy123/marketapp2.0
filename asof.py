from __future__ import annotations

from storage_sqlite import latest_price_date


def get_asof_date(ticker: str) -> str | None:
    """Return latest stored daily price date (YYYY-MM-DD) for ticker/source=stooq."""
    return latest_price_date(ticker, source="stooq")
