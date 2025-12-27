from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import requests

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _dt_to_gdelt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_news_gdelt(
    ticker: str,
    company_name: str,
    aliases: List[str] | None = None,
    days: int = 90,
    timeout: int = 30,
    max_records: int = 250,
) -> List[Dict]:
    """
    Rolling-window fetch from GDELT (recent history).
    Returns list of dicts: {published_at, title, url, snippet, match_score}
    """
    aliases = aliases or []
    terms = [f"\"{company_name}\"", ticker]
    terms += [f"\"{a}\"" for a in aliases if a.strip()]
    query = "(" + " OR ".join(terms) + ")"

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": _dt_to_gdelt(start),
        "enddatetime": _dt_to_gdelt(end),
        "sort": "datedesc",
    }

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(GDELT_DOC_URL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    out = []
    for art in data.get("articles", []) or []:
        title = (art.get("title") or "").strip()
        url = (art.get("url") or "").strip()
        if not title or not url:
            continue
        published = (art.get("seendate") or "").strip()
        snippet = (art.get("snippet") or "").strip()

        out.append(
            {
                "published_at": published,
                "title": title,
                "url": url,
                "snippet": snippet,
                "match_score": 0,
            }
        )
    return out
