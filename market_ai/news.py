from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List
from urllib.parse import quote_plus

import requests
import xml.etree.ElementTree as ET


@dataclass
class NewsItem:
    title: str
    link: str
    pub_date: str
    source: str


def fetch_google_news_rss(query: str, session: requests.Session, max_items: int = 10, timeout: int = 20) -> List[NewsItem]:
    """Google News RSS is free and usually accessible.

    query can be ticker or phrase.
    """
    q = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-GB&gl=GB&ceid=GB:en"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    channel = root.find('channel')
    if channel is None:
        return []

    items: List[NewsItem] = []
    for it in channel.findall('item'):
        title = (it.findtext('title') or '').strip()
        link = (it.findtext('link') or '').strip()
        pub_date = (it.findtext('pubDate') or '').strip()
        source = (it.findtext('source') or 'Google News').strip()
        if title and link:
            items.append(NewsItem(title=title, link=link, pub_date=pub_date, source=source))
        if len(items) >= max_items:
            break

    # avoid hammering on auto-refresh
    time.sleep(0.2)
    return items
