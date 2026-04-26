"""Data-quality helpers shared by valuation and screening services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DataQualityReport:
    """Small status object used to avoid treating missing data as a real zero."""

    status: str
    blockers: list[str]
    warnings: list[str]

    @property
    def is_critical(self) -> bool:
        return self.status == "critical"

    @property
    def summary(self) -> str:
        if self.blockers:
            return "; ".join(self.blockers)
        if self.warnings:
            return "; ".join(self.warnings)
        return "Core market and statement inputs look usable."

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "summary": self.summary,
        }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def has_dataframe_data(value: Any) -> bool:
    return isinstance(value, pd.DataFrame) and not value.empty


def build_data_quality_report(
    *,
    price: Any,
    shares: Any,
    market_cap: Any,
    revenue_ttm: Any,
    eps_ttm: Any,
    fcf_ttm: Any,
    balance_sheet: Any,
    income_statement: Any,
    cash_flow: Any,
    fetcher_warnings: list[str] | None = None,
    fetcher_error: str | None = None,
) -> DataQualityReport:
    """Classify whether core valuation inputs are trustworthy enough to show normally."""
    blockers: list[str] = []
    warnings: list[str] = list(fetcher_warnings or [])

    if fetcher_error:
        warnings.append(f"Data fetcher reported: {fetcher_error}")
    if safe_float(price) <= 0:
        blockers.append("Current price is missing.")
    if safe_float(shares) <= 1 and safe_float(market_cap) <= 0:
        blockers.append("Share count and market cap are missing.")
    if safe_float(market_cap) <= 0:
        warnings.append("Market cap could not be verified.")
    if safe_float(revenue_ttm) <= 0:
        warnings.append("Revenue TTM is missing or zero.")
    if safe_float(eps_ttm) == 0:
        warnings.append("EPS TTM is unavailable or zero, so P/E may be unavailable.")
    if safe_float(fcf_ttm) <= 0:
        warnings.append("Free cash flow TTM is negative or unavailable.")

    if not has_dataframe_data(balance_sheet):
        warnings.append("Balance sheet statement is missing.")
    if not has_dataframe_data(income_statement):
        warnings.append("Income statement is missing.")
    if not has_dataframe_data(cash_flow):
        warnings.append("Cash-flow statement is missing.")

    deduped_warnings = list(dict.fromkeys(warnings))
    if blockers:
        status = "critical"
    elif deduped_warnings:
        status = "warning"
    else:
        status = "ok"

    return DataQualityReport(status=status, blockers=blockers, warnings=deduped_warnings)


def quality_label(status: str | None) -> str:
    if status == "ok":
        return "OK"
    if status == "critical":
        return "Incomplete"
    return "A surveiller"
