"""Reusable screening engine for the Streamlit screener view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

ProgressCallback = Callable[[int, int, str, str], None]
FINANCIAL_DATA_CACHE_VERSION = "2026-04-06-ai-news-v4"


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

    revenue = _ttm_or_latest(data["inc"], ["Revenue"])
    operating_cash_flow = _ttm_or_latest(data["cf"], ["OperatingCashFlow"])
    capex = abs(_item_safe(data["cf"], ["CapitalExpenditure"]))
    fcf = operating_cash_flow - capex
    cash = _item_safe(data["bs"], ["Cash"])
    debt = _debt_safe(data["bs"])

    growth = float(data.get("rev_growth", 0) or 0)
    if growth <= 0:
        growth = fallback_fcf_growth_pct / 100.0

    dcf_value, _, _ = valuation_calculator(
        0,
        growth,
        0,
        wacc_pct / 100.0,
        0,
        0,
        revenue,
        fcf,
        0,
        cash,
        debt,
        shares,
    )
    if dcf_value <= 0:
        return None

    return ScreenerCandidate(
        ticker=ticker,
        price=price,
        intrinsic=dcf_value,
        upside=(dcf_value / price - 1) * 100,
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
