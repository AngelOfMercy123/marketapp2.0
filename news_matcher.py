from __future__ import annotations

import re
from typing import Dict, List


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def score_article(title: str, snippet: str, ticker: str, company_name: str, aliases: List[str]) -> int:
    """Strict lexical score to avoid mismatches."""
    t = _norm(title)
    sn = _norm(snippet)
    blob = f"{t} {sn}".strip()

    ticker_u = ticker.upper()
    score = 0

    if re.search(rf"\b{re.escape(ticker_u.lower())}\b", blob):
        score += 60

    cname = _norm(company_name)
    if cname and cname in blob:
        score += 60

    for a in aliases:
        aa = _norm(a)
        if aa and aa in blob:
            score += 30

    if len(t) < 18:
        score -= 10

    return max(score, 0)


def filter_and_tag_news(
    items: List[Dict],
    ticker: str,
    company_name: str,
    aliases: List[str] | None = None,
    min_score: int = 60,
) -> List[Dict]:
    aliases = aliases or []
    out = []
    for it in items:
        s = score_article(it.get("title", ""), it.get("snippet", ""), ticker, company_name, aliases)
        if s >= min_score:
            it["match_score"] = s
            out.append(it)
    return out
