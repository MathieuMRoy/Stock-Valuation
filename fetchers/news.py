"""
News fetcher - combine recent market coverage with IR-style press releases.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
import yfinance as yf


def _parse_news_datetime(value: str | None) -> datetime | None:
    """Parse common news timestamp formats into timezone-aware UTC datetimes."""
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass

    try:
        parsed = parsedate_to_datetime(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _build_news_item(
    *,
    title: str | None,
    link: str | None,
    published_at: str | None,
    source: str | None,
    category: str,
) -> dict | None:
    clean_title = (title or "").strip()
    clean_link = (link or "").strip()
    if not clean_title or not clean_link:
        return None

    parsed_dt = _parse_news_datetime(published_at)
    return {
        "title": clean_title,
        "link": clean_link,
        "published_at": parsed_dt.isoformat() if parsed_dt else (published_at or ""),
        "source": (source or "").strip() or None,
        "category": category,
        "sort_key": parsed_dt.isoformat() if parsed_dt else "",
    }


def _sort_and_trim(items: list[dict], limit: int) -> list[dict]:
    deduped: dict[tuple[str, str], dict] = {}
    for item in items:
        if not item:
            continue
        key = ((item.get("title") or "").lower(), item.get("link") or "")
        existing = deduped.get(key)
        if existing is None or (item.get("sort_key") or "") > (existing.get("sort_key") or ""):
            deduped[key] = item

    ordered = sorted(
        deduped.values(),
        key=lambda item: ((item.get("sort_key") or ""), (item.get("title") or "")),
        reverse=True,
    )

    trimmed = []
    for item in ordered[:limit]:
        clean_item = dict(item)
        clean_item.pop("sort_key", None)
        trimmed.append(clean_item)
    return trimmed


def _fetch_google_news(query: str, limit: int, category: str) -> list[dict]:
    """Fetch news items from the Google News RSS endpoint."""
    try:
        encoded_query = quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        items: list[dict] = []
        for rss_item in root.findall(".//item")[:limit]:
            built = _build_news_item(
                title=getattr(rss_item.find("title"), "text", None),
                link=getattr(rss_item.find("link"), "text", None),
                published_at=getattr(rss_item.find("pubDate"), "text", None),
                source="Google News RSS",
                category=category,
            )
            if built:
                items.append(built)
        return items
    except Exception:
        return []


def _fetch_yfinance_news(ticker: str, limit: int) -> list[dict]:
    """Fetch richer market-news headlines via yfinance when available."""
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items: list[dict] = []
    for raw_item in raw_items[: max(limit * 2, limit)]:
        content = raw_item.get("content", raw_item) if isinstance(raw_item, dict) else {}
        provider = content.get("provider") or {}
        click = content.get("clickThroughUrl") or {}
        canonical = content.get("canonicalUrl") or {}
        built = _build_news_item(
            title=content.get("title"),
            link=click.get("url") or canonical.get("url"),
            published_at=content.get("pubDate") or content.get("displayTime"),
            source=provider.get("displayName") or "Yahoo Finance",
            category="market_news",
        )
        if built:
            items.append(built)
    return items[:limit]


def fetch_ir_press_releases(search_name: str, limit: int = 4) -> list[dict]:
    """
    Fetch IR-style press releases using a company-name-specific Google News RSS query.
    """
    query = f"\"{search_name}\" press release investor relations"
    return _sort_and_trim(_fetch_google_news(query, limit=limit * 2, category="press_release"), limit)


def fetch_recent_company_news(ticker: str, search_name: str, limit: int = 6) -> list[dict]:
    """
    Fetch recent broader company news, preferring market coverage and falling back to Google News.
    """
    market_news = _fetch_yfinance_news(ticker, limit=max(limit, 6))
    google_fallback = _fetch_google_news(f"\"{ticker}\" OR \"{search_name}\" stock", limit=limit, category="market_news")
    return _sort_and_trim(market_news + google_fallback, limit)
