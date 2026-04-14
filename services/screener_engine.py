"""Reusable screening engine for the Streamlit screener view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

ProgressCallback = Callable[[int, int, str, str], None]
FINANCIAL_DATA_CACHE_VERSION = "2026-04-06-ai-news-v4"
MAX_REASONABLE_UPSIDE_PCT = 300.0
NON_OPERATING_QUOTE_TYPES = {"etf", "fund", "mutualfund", "closedendfund", "trust", "index", "commodity"}
UNSUITABLE_SECTOR_KEYWORDS = {"financial", "financial services", "banks", "insurance"}
CYCLICAL_SECTOR_KEYWORDS = {"energy", "basic materials", "materials"}


@dataclass(slots=True)
class ScreenerCandidate:
    """Normalized screener result row."""

    ticker: str
    price: float
    intrinsic: float
    upside: float
    bucket: str

    def as_row(self) -> dict:
        return {
            "bucket": self.bucket,
            "ticker": self.ticker,
            "price": self.price,
            "intrinsic": self.intrinsic,
            "upside": self.upside,
        }


def _find_matching_index(df: pd.DataFrame, search_terms: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    normalized_terms = ["".join(ch for ch in str(term).lower() if ch.isalnum()) for term in search_terms]
    for idx in df.index:
        normalized_idx = "".join(ch for ch in str(idx).lower() if ch.isalnum())
        if normalized_idx in normalized_terms:
            return idx
    for idx in df.index:
        normalized_idx = "".join(ch for ch in str(idx).lower() if ch.isalnum())
        if any(term in normalized_idx for term in normalized_terms):
            return idx
    return None


def _item_safe(df: pd.DataFrame, search_terms: list[str]) -> float:
    matched_idx = _find_matching_index(df, search_terms)
    if matched_idx is None:
        return 0.0
    value = df.loc[matched_idx]
    if isinstance(value, pd.Series):
        for item in value:
            if isinstance(item, (int, float)) and not pd.isna(item):
                return float(item)
        return 0.0
    return float(value)


def _ttm_or_latest(df: pd.DataFrame, search_terms: list[str]) -> float:
    matched_idx = _find_matching_index(df, search_terms)
    if matched_idx is None:
        return 0.0
    row = df.loc[matched_idx]
    values = [value for value in row if isinstance(value, (int, float)) and not pd.isna(value)]
    if not values:
        return 0.0
    if len(values) >= 4:
        return float(sum(values[:4]))
    if len(values) == 1:
        return float(values[0]) * 4
    return float(values[0])


def _debt_safe(df: pd.DataFrame) -> float:
    total_debt = _item_safe(df, ["TotalDebt", "Total Debt"])
    if total_debt > 0:
        return total_debt

    current_debt = _item_safe(
        df,
        [
            "CurrentDebt",
            "Current Debt",
            "CurrentDebtAndCapitalLeaseObligation",
            "Current Debt And Capital Lease Obligation",
        ],
    )
    long_term_debt = _item_safe(
        df,
        [
            "LongTermDebt",
            "Long Term Debt",
            "LongTermDebtAndCapitalLeaseObligation",
            "Long Term Debt And Capital Lease Obligation",
        ],
    )
    return current_debt + long_term_debt


def market_cap_ok(data: dict, minimum_market_cap: float) -> bool:
    """Check if a ticker clears the market-cap threshold."""
    market_cap = data.get("market_cap")
    if market_cap is None:
        return False
    try:
        return float(market_cap) >= float(minimum_market_cap)
    except Exception:
        return False


def _normalized_text(value: object) -> str:
    return str(value or "").strip().lower()


def _is_cyclical_sector(sector_name: str | None) -> bool:
    sector = _normalized_text(sector_name)
    return any(keyword in sector for keyword in CYCLICAL_SECTOR_KEYWORDS)


def _is_unsuitable_for_fast_dcf(data: dict) -> bool:
    sector = _normalized_text(data.get("sector"))
    industry = _normalized_text(data.get("industry"))
    quote_type = _normalized_text(data.get("quote_type"))
    long_name = _normalized_text(data.get("long_name"))

    if any(keyword in sector for keyword in UNSUITABLE_SECTOR_KEYWORDS):
        return True
    if any(keyword in industry for keyword in UNSUITABLE_SECTOR_KEYWORDS):
        return True
    if quote_type in NON_OPERATING_QUOTE_TYPES:
        return True
    if any(keyword in long_name for keyword in ("fund", "trust", "etf")):
        return True
    return False


def _conservative_growth_rate(data: dict, fallback_fcf_growth_pct: float) -> float:
    sector = _normalized_text(data.get("sector"))
    observed_growth = float(data.get("rev_growth", 0) or 0)
    fallback_growth = max(float(fallback_fcf_growth_pct or 0) / 100.0, 0.02)

    if _is_cyclical_sector(sector):
        lower_bound, upper_bound = 0.00, 0.05
    else:
        lower_bound, upper_bound = 0.01, 0.10

    base_growth = observed_growth if observed_growth > 0 else fallback_growth
    return max(lower_bound, min(base_growth, upper_bound))


def _sector_adjusted_wacc(data: dict, wacc_pct: float) -> float:
    sector = _normalized_text(data.get("sector"))
    base_wacc = max(float(wacc_pct or 0) / 100.0, 0.08)

    if _is_cyclical_sector(sector):
        base_wacc += 0.02
    elif "technology" in sector or "consumer cyclical" in sector:
        base_wacc += 0.01

    return min(base_wacc, 0.18)


def _normalized_fcf(data: dict, revenue: float, fcf: float, market_cap: float) -> float:
    if revenue <= 0 or fcf <= 0 or market_cap <= 0:
        return 0.0

    fcf_margin = fcf / revenue
    fcf_yield = fcf / market_cap

    if fcf_margin <= 0 or fcf_yield <= 0:
        return 0.0

    if _is_cyclical_sector(data.get("sector")):
        max_margin = 0.12
    else:
        max_margin = 0.18

    if fcf_yield > 0.25:
        return 0.0

    return revenue * min(fcf_margin, max_margin)


def quick_intrinsic_dcf(
    ticker: str,
    minimum_market_cap: float,
    fallback_fcf_growth_pct: float,
    wacc_pct: float,
    *,
    cache_version: str = FINANCIAL_DATA_CACHE_VERSION,
    data_fetcher=None,
    valuation_calculator=None,
) -> ScreenerCandidate | None:
    """Run a fast DCF screen for a single ticker."""
    if data_fetcher is None or valuation_calculator is None:
        from valuation import calculate_valuation
        from fetchers import get_financial_data_secure

        data_fetcher = data_fetcher or get_financial_data_secure
        valuation_calculator = valuation_calculator or calculate_valuation

    data = data_fetcher(ticker, cache_version=cache_version)
    price = float(data.get("price", 0) or 0)
    shares = float(data.get("shares_info", 0) or 0)
    if price <= 0 or shares <= 0:
        return None
    if not market_cap_ok(data, minimum_market_cap):
        return None
    if _is_unsuitable_for_fast_dcf(data):
        return None

    market_cap = float(data.get("market_cap", 0) or 0)
    if market_cap <= 0:
        market_cap = shares * price
    if market_cap <= 0:
        return None

    revenue = _ttm_or_latest(data["inc"], ["Revenue"])
    operating_cash_flow = _ttm_or_latest(data["cf"], ["OperatingCashFlow"])
    capex = abs(_item_safe(data["cf"], ["CapitalExpenditure"]))
    fcf = operating_cash_flow - capex
    cash = _item_safe(data["bs"], ["Cash"])
    debt = _debt_safe(data["bs"])
    normalized_fcf = _normalized_fcf(data, revenue, fcf, market_cap)
    if normalized_fcf <= 0:
        return None

    growth = _conservative_growth_rate(data, fallback_fcf_growth_pct)
    adjusted_wacc = _sector_adjusted_wacc(data, wacc_pct)

    dcf_value, _, _ = valuation_calculator(
        0,
        growth,
        0,
        adjusted_wacc,
        0,
        0,
        revenue,
        normalized_fcf,
        0,
        cash,
        debt,
        shares,
    )
    if dcf_value <= 0:
        return None

    upside_pct = (dcf_value / price - 1) * 100
    if upside_pct <= 0 or upside_pct > MAX_REASONABLE_UPSIDE_PCT:
        return None

    return ScreenerCandidate(
        ticker=ticker,
        price=price,
        intrinsic=dcf_value,
        upside=upside_pct,
        bucket=str(data.get("sector") or "Unknown"),
    )


def run_screener(
    minimum_market_cap: float,
    max_tickers_per_sector: int,
    fallback_fcf_growth_pct: float,
    wacc_pct: float,
    *,
    sectors: Iterable[tuple[str, str]] | None = None,
    progress_callback: ProgressCallback | None = None,
    ticker_fetcher=None,
    data_fetcher=None,
    valuation_calculator=None,
) -> list[ScreenerCandidate]:
    """Run the screener across configured Finviz sectors and geographies."""
    if sectors is None or ticker_fetcher is None:
        from fetchers import FINVIZ_SECTORS, finviz_fetch_tickers

        sectors = sectors or FINVIZ_SECTORS
        ticker_fetcher = ticker_fetcher or finviz_fetch_tickers

    results: list[ScreenerCandidate] = []
    sector_list = list(sectors)
    total_steps = max(len(sector_list) * 2, 1)
    current_step = 0

    for sector_name, sector_code in sector_list:
        for geography_name, geography_code in [("USA", "geo_usa"), ("Canada", "geo_canada")]:
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, sector_name, geography_name)

            tickers = ticker_fetcher(sector_code, geography_code, max_tickers_per_sector)
            for ticker in tickers:
                candidate = quick_intrinsic_dcf(
                    ticker,
                    minimum_market_cap,
                    fallback_fcf_growth_pct,
                    wacc_pct,
                    data_fetcher=data_fetcher,
                    valuation_calculator=valuation_calculator,
                )
                if candidate:
                    candidate.bucket = f"{sector_name} ({geography_name})"
                    results.append(candidate)

    return sorted(results, key=lambda row: row.upside, reverse=True)
