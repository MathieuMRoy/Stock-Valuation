"""Earnings calendar fetchers and normalization helpers."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Any

import pandas as pd
import requests


NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings"
NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


def _nasdaq_time_label(raw_value: str | None) -> str:
    """Normalize Nasdaq's icon-style time labels."""
    normalized = str(raw_value or "").strip().lower()
    if "pre-market" in normalized:
        return "Pre-market"
    if "after-hours" in normalized:
        return "After close"
    if "not-supplied" in normalized:
        return "Not supplied"
    return str(raw_value or "Not supplied").replace("time-", "").replace("-", " ").title()


def _country_for_symbol(symbol: str) -> str:
    """Infer a light country/market label from the ticker format."""
    upper = str(symbol or "").upper()
    if upper.endswith(".TO") or upper.endswith(".V"):
        return "Canada"
    return "USA / Nasdaq"


def _parse_market_cap(raw_value: str | None) -> float:
    """Parse Nasdaq market-cap strings into numeric dollars."""
    text = str(raw_value or "").strip().upper()
    if not text or text == "N/A":
        return 0.0

    multiplier = 1.0
    if text.endswith("T"):
        multiplier = 1_000_000_000_000.0
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]

    cleaned = text.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return 0.0


def _nasdaq_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else {}
    rows = data.get("rows") if isinstance(data, dict) else []
    return rows if isinstance(rows, list) else []


def fetch_nasdaq_earnings_for_date(
    target_date: date,
    *,
    request_get: Callable[..., Any] = requests.get,
    timeout: int = 15,
) -> list[dict[str, object]]:
    """Fetch and normalize Nasdaq earnings calendar rows for one date."""
    response = request_get(
        NASDAQ_EARNINGS_URL,
        params={"date": target_date.isoformat()},
        headers=NASDAQ_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    events: list[dict[str, object]] = []
    for row in _nasdaq_rows(payload):
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue

        market_cap = str(row.get("marketCap") or "N/A").strip() or "N/A"
        events.append(
            {
                "Ticker": symbol,
                "Company": str(row.get("name") or symbol).strip(),
                "Country": _country_for_symbol(symbol),
                "Date": target_date,
                "Day": target_date.strftime("%A"),
                "Formatted": target_date.strftime("%b %d"),
                "DaysUntil": (target_date - date.today()).days,
                "Time": _nasdaq_time_label(str(row.get("time") or "")),
                "EPS Forecast": str(row.get("epsForecast") or "N/A").strip() or "N/A",
                "Estimates": str(row.get("noOfEsts") or "N/A").strip() or "N/A",
                "Fiscal Quarter": str(row.get("fiscalQuarterEnding") or "N/A").strip() or "N/A",
                "Market Cap": market_cap,
                "Market Cap Value": _parse_market_cap(market_cap),
                "Last Year EPS": str(row.get("lastYearEPS") or "N/A").strip() or "N/A",
                "Source": "Nasdaq",
            }
        )

    return events


def fetch_nasdaq_earnings_calendar(
    start_date: date,
    end_date: date,
    *,
    request_get: Callable[..., Any] = requests.get,
) -> pd.DataFrame:
    """Fetch Nasdaq earnings events across an inclusive date range."""
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    events: list[dict[str, object]] = []
    current = start_date
    while current <= end_date:
        try:
            events.extend(fetch_nasdaq_earnings_for_date(current, request_get=request_get))
        except Exception:
            # Keep the calendar useful even if one Nasdaq date fails or rate-limits.
            pass
        current += timedelta(days=1)

    if not events:
        return pd.DataFrame()

    def _patch_mc(ev: dict[str, Any]):
        try:
            import yfinance as yf
            val = getattr(yf.Ticker(ev["Ticker"]).fast_info, "market_cap", 0.0)
            if val and val > 0:
                ev["Market Cap Value"] = float(val)
        except Exception:
            pass

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(_patch_mc, ev) for ev in events]
        concurrent.futures.wait(futures, timeout=8.0)

    # Re-apply the formatting if needed, though calendar_view.py & report_export
    # mainly rely on 'Market Cap Value' directly anyway.

    return (
        pd.DataFrame(events)
        .drop_duplicates(["Ticker", "Date"])
        .sort_values(["Date", "Market Cap Value", "Ticker"], ascending=[True, False, True])
    )
