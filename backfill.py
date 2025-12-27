from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from news_matcher import filter_and_tag_news
from sources_news_gdelt import fetch_news_gdelt
from sources_prices_stooq import fetch_prices_stooq
from storage_sqlite import init_db, upsert_news, upsert_prices


@dataclass
class CompanySpec:
    ticker: str
    name: str
    aliases: List[str]
    stooq_symbol: str  # e.g., "dg.us", "now.us"


def backfill_company(
    spec: CompanySpec,
    news_days: int = 90,
    price_source: str = "stooq",
    news_source: str = "gdelt",
) -> Dict[str, Any]:
    init_db()

    df = fetch_prices_stooq(spec.stooq_symbol)
    n_prices = upsert_prices(spec.ticker, df, source=price_source)

    raw_news = fetch_news_gdelt(
        ticker=spec.ticker,
        company_name=spec.name,
        aliases=spec.aliases,
        days=news_days,
    )
    news = filter_and_tag_news(
        raw_news,
        ticker=spec.ticker,
        company_name=spec.name,
        aliases=spec.aliases,
        min_score=60,
    )
    n_news = upsert_news(spec.ticker, news, source=news_source)

    return {"ticker": spec.ticker, "prices_added": n_prices, "news_added": n_news}
