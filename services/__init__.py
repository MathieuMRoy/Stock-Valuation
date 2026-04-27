"""Domain services shared across Streamlit views."""

from .analyzer_service import (
    INVESTOR_OBJECTIVES,
    AnalyzerSnapshot,
    build_investor_objective_snapshot,
    business_model_hint,
    combined_risk_label,
    extract_next_earnings,
    is_financial_company,
    market_risk_label,
    prepare_analyzer_snapshot,
    profile_label,
    resolve_share_count,
    risk_label,
    statement_basis_label,
    valuation_label,
)
from .screener_engine import ScreenerCandidate, market_cap_ok, quick_intrinsic_dcf, run_screener
from .data_quality import DataQualityReport, build_data_quality_report, quality_label
from .earnings_calendar import fetch_nasdaq_earnings_calendar, fetch_nasdaq_earnings_for_date
from .sector_profiles import SectorProfile, get_sector_profile, sector_profile_summary
from .valuation_guardrails import blend_reasonable_intrinsic_values, is_reasonable_intrinsic_value, upside_pct
from .report_export import markdown_report_to_pdf_bytes

__all__ = [
    "AnalyzerSnapshot",
    "DataQualityReport",
    "INVESTOR_OBJECTIVES",
    "SectorProfile",
    "ScreenerCandidate",
    "blend_reasonable_intrinsic_values",
    "build_data_quality_report",
    "build_investor_objective_snapshot",
    "business_model_hint",
    "combined_risk_label",
    "extract_next_earnings",
    "fetch_nasdaq_earnings_calendar",
    "fetch_nasdaq_earnings_for_date",
    "get_sector_profile",
    "is_financial_company",
    "market_risk_label",
    "market_cap_ok",
    "markdown_report_to_pdf_bytes",
    "prepare_analyzer_snapshot",
    "profile_label",
    "quality_label",
    "quick_intrinsic_dcf",
    "resolve_share_count",
    "risk_label",
    "run_screener",
    "sector_profile_summary",
    "statement_basis_label",
    "is_reasonable_intrinsic_value",
    "upside_pct",
    "valuation_label",
]
