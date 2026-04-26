"""Domain services shared across Streamlit views."""

from .analyzer_service import (
    INVESTOR_OBJECTIVES,
    AnalyzerSnapshot,
    build_investor_objective_snapshot,
    business_model_hint,
    extract_next_earnings,
    is_financial_company,
    prepare_analyzer_snapshot,
    profile_label,
    resolve_share_count,
    risk_label,
    statement_basis_label,
    valuation_label,
)
from .screener_engine import ScreenerCandidate, market_cap_ok, quick_intrinsic_dcf, run_screener
from .data_quality import DataQualityReport, build_data_quality_report, quality_label
from .valuation_guardrails import blend_reasonable_intrinsic_values, is_reasonable_intrinsic_value, upside_pct

__all__ = [
    "AnalyzerSnapshot",
    "DataQualityReport",
    "INVESTOR_OBJECTIVES",
    "ScreenerCandidate",
    "blend_reasonable_intrinsic_values",
    "build_data_quality_report",
    "build_investor_objective_snapshot",
    "business_model_hint",
    "extract_next_earnings",
    "is_financial_company",
    "market_cap_ok",
    "prepare_analyzer_snapshot",
    "profile_label",
    "quality_label",
    "quick_intrinsic_dcf",
    "resolve_share_count",
    "risk_label",
    "run_screener",
    "statement_basis_label",
    "is_reasonable_intrinsic_value",
    "upside_pct",
    "valuation_label",
]
