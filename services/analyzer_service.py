"""Shared business logic for the stock analyzer view."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from data import get_benchmark_data
from .data_quality import build_data_quality_report, safe_float


FINANCIAL_DATA_CACHE_VERSION = "2026-04-06-ai-news-v4"


INVESTOR_OBJECTIVES = {
    "balanced": {
        "label": "Equilibre",
        "description": "mix upside, qualite financiere et valorisation",
        "decision_frame": "balanced mix of upside, quality and valuation",
    },
    "growth": {
        "label": "Croissance",
        "description": "priorite a la croissance, au rerating et aux catalyseurs",
        "decision_frame": "prioritize revenue growth, EPS acceleration and upside optionality",
    },
    "value": {
        "label": "Value",
        "description": "priorite a la valorisation et au potentiel de rerating",
        "decision_frame": "prioritize valuation support, cheaper multiples and mean reversion",
    },
    "defensive": {
        "label": "Defensif",
        "description": "priorite au bilan, a la resilience et a la stabilite",
        "decision_frame": "prioritize balance-sheet resilience, profitability quality and lower cyclicality",
    },
    "income": {
        "label": "Revenu",
        "description": "priorite au rendement, aux retours cash et a la stabilite",
        "decision_frame": "prioritize income, cash returns and dividend support when available",
    },
    "short_term": {
        "label": "Court terme",
        "description": "priorite au momentum, aux catalyseurs et au setup de marche",
        "decision_frame": "prioritize momentum, near-term catalysts and trading setup",
    },
}


@dataclass(slots=True)
class AnalyzerSnapshot:
    """Typed payload consumed by the Streamlit stock analyzer."""

    ticker: str
    data: dict
    bench_data: dict
    metrics: dict
    scores: dict
    bs: pd.DataFrame
    inc: pd.DataFrame
    cf: pd.DataFrame
    current_price: float
    shares: float
    market_cap: float
    piotroski: int | None
    altman_z: float | None
    revenue_ttm: float
    cfo_ttm: float
    capex_ttm: float
    fcf_ttm: float
    cash: float
    debt: float
    eps_ttm: float
    pe: float
    forward_pe: float
    ps: float
    sales_growth: float
    eps_growth: float
    quote_currency: str
    financial_currency: str
    is_financial: bool
    next_earnings: str
    revenue_basis: str
    eps_basis: str
    recent_news_count: int
    press_release_count: int
    manual_shares_applied: bool
    share_count_unavailable: bool
    share_count_estimated: bool
    data_quality: str
    data_quality_reasons: list[str]
    valuation_warnings: list[str]


def extract_next_earnings(calendar_data) -> str:
    """Extract the next earnings date from a Yahoo calendar payload."""
    if calendar_data is None:
        return "N/A"

    candidates: list[pd.Timestamp] = []

    def _collect(value):
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _collect(item)
            return
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            candidates.append(pd.Timestamp(parsed))

    if isinstance(calendar_data, pd.DataFrame):
        for col in calendar_data.columns:
            if "earnings" in str(col).lower():
                _collect(calendar_data[col].tolist())
        for idx in calendar_data.index:
            if "earnings" in str(idx).lower():
                row = calendar_data.loc[idx]
                if isinstance(row, pd.Series):
                    _collect(row.tolist())
                else:
                    _collect(row)
    elif isinstance(calendar_data, pd.Series):
        for label, value in calendar_data.items():
            if "earnings" in str(label).lower():
                _collect(value)
    elif isinstance(calendar_data, dict):
        for label, value in calendar_data.items():
            if "earnings" in str(label).lower():
                _collect(value)

    if not candidates:
        return "N/A"

    today = pd.Timestamp(date.today())
    future_candidates = sorted(ts for ts in candidates if ts.normalize() >= today)
    target = future_candidates[0] if future_candidates else sorted(candidates)[0]
    day_gap = (target.normalize() - today).days

    if day_gap == 0:
        return f"{target:%Y-%m-%d} (today)"
    if day_gap > 0:
        return f"{target:%Y-%m-%d} ({day_gap}d)"
    return f"{target:%Y-%m-%d}"


def is_financial_company(sector_name: str | None, benchmark_name: str | None) -> bool:
    """Detect banks and other financial institutions needing special heuristics."""
    text = " ".join([str(sector_name or ""), str(benchmark_name or "")]).lower()
    keywords = ["financial", "bank", "banks", "insurance", "capital markets", "asset management"]
    return any(keyword in text for keyword in keywords)


def risk_label(altman_z: float | None, piotroski: int | None, is_financial: bool = False) -> str:
    """Describe risk with a compact quality heuristic."""
    if is_financial:
        if (piotroski or 0) >= 7:
            return "Stable bank profile"
        if (piotroski or 0) <= 3:
            return "Watch bank fundamentals"
        return "Bank-specific risk view"
    if (altman_z or 0) >= 3 and (piotroski or 0) >= 7:
        return "Lower risk"
    if (altman_z or 0) < 1.8 or (piotroski or 0) <= 3:
        return "Higher risk"
    return "Balanced risk"


def valuation_label(gap_pct: float) -> str:
    """Convert valuation gap into a readable verdict."""
    if gap_pct >= 18:
        return "Undervalued setup"
    if gap_pct <= -18:
        return "Rich valuation"
    return "Fairly priced"


def profile_label(
    sales_growth: float,
    eps_growth: float,
    ps_ratio: float,
    peer_ps: float,
    is_financial: bool = False,
) -> str:
    """Suggest the dominant investor profile for the stock."""
    if is_financial:
        return "Regulated bank model"
    if sales_growth >= 0.18 and ps_ratio >= peer_ps:
        return "Growth-oriented profile"
    if eps_growth > 0.12 and ps_ratio <= peer_ps:
        return "Quality / value blend"
    return "Balanced core compounder"


def business_model_hint(sector_name: str | None, benchmark_name: str | None) -> str:
    """Provide a compact business-model descriptor for cross-sector comparisons."""
    text = " ".join([str(sector_name or ""), str(benchmark_name or "")]).lower()
    if "energy" in text or "oil" in text or "gas" in text:
        return "energy producer with commodity-price exposure"
    if "bank" in text or "financial" in text:
        return "financial institution driven by capital strength and credit quality"
    if "consumer app" in text or "platform" in text or "streaming" in text:
        return "consumer platform with growth-sensitive multiples"
    if "saas" in text or "cloud" in text or "software" in text or "technology" in text:
        return "software or platform business with valuation sensitivity to growth"
    if "pharma" in text or "biotech" in text:
        return "healthcare business influenced by product pipeline and regulation"
    return f"company exposed to the {sector_name or benchmark_name or 'broader market'} cycle"


def build_investor_objective_snapshot(objective_key: str) -> dict:
    """Build the comparison objective context passed into the AI chat."""
    preset = INVESTOR_OBJECTIVES.get(objective_key, INVESTOR_OBJECTIVES["balanced"])
    resolved_key = objective_key if objective_key in INVESTOR_OBJECTIVES else "balanced"
    return {
        "key": resolved_key,
        "label": preset["label"],
        "description": preset["description"],
        "decision_frame": preset["decision_frame"],
    }


def statement_basis_label(df_quarterly, df_annual, fallback_label: str) -> str:
    """Describe which statement basis currently drives a metric."""
    if hasattr(df_quarterly, "empty") and not df_quarterly.empty:
        return "Quarterly TTM"
    if hasattr(df_annual, "empty") and not df_annual.empty:
        return "Latest annual statement"
    return fallback_label


def resolve_share_count(
    *,
    price: float,
    raw_shares: float,
    reported_market_cap: float,
    manual_shares_millions: float = 0.0,
) -> tuple[float, float, bool, bool, bool]:
    """Resolve shares/market cap without silently using one fake share."""
    manual_applied = manual_shares_millions > 0
    if manual_applied:
        shares = manual_shares_millions * 1_000_000
        market_cap = shares * price if price > 0 else reported_market_cap
        return shares, market_cap, False, False, True

    if raw_shares > 1:
        market_cap = raw_shares * price if price > 0 else reported_market_cap
        if reported_market_cap > 0 and price > 0:
            market_cap = reported_market_cap
        return raw_shares, market_cap, False, False, False

    if reported_market_cap > 0 and price > 0:
        return reported_market_cap / price, reported_market_cap, False, True, False

    return 1.0, max(reported_market_cap, 0.0), True, False, False


def prepare_analyzer_snapshot(
    ticker: str,
    manual_shares_millions: float = 0.0,
    cache_version: str = FINANCIAL_DATA_CACHE_VERSION,
) -> AnalyzerSnapshot:
    """Load and normalize the business context needed by the stock analyzer UI."""
    from fetchers import get_debt_safe, get_financial_data_secure, get_item_safe, get_ttm_or_latest
    from scoring import calculate_altman_z, calculate_piotroski_score, score_out_of_10

    data = get_financial_data_secure(ticker, cache_version=cache_version)

    current_price = safe_float(data.get("price"))
    bs = data.get("bs", pd.DataFrame())
    inc = data.get("inc", pd.DataFrame())
    cf = data.get("cf", pd.DataFrame())
    piotroski = calculate_piotroski_score(bs, inc, cf)

    raw_shares = safe_float(data.get("shares_info"))
    reported_market_cap = safe_float(data.get("market_cap"))
    shares, market_cap, share_count_unavailable, share_count_estimated, manual_shares_applied = resolve_share_count(
        price=current_price,
        raw_shares=raw_shares,
        reported_market_cap=reported_market_cap,
        manual_shares_millions=manual_shares_millions,
    )
    altman_z = calculate_altman_z(bs, inc, market_cap) if current_price > 0 else 0.0

    revenue_ttm = safe_float(data.get("revenue_ttm"))
    if revenue_ttm == 0:
        revenue_ttm = get_ttm_or_latest(inc, ["TotalRevenue", "Revenue"])

    cfo_ttm = get_ttm_or_latest(cf, ["OperatingCashFlow", "Operating Cash Flow"])
    capex_ttm = abs(get_item_safe(cf, ["CapitalExpenditure", "PurchaseOfPPE"]))
    fcf_ttm = cfo_ttm - capex_ttm
    cash = get_item_safe(bs, ["CashAndCashEquivalents", "Cash"])
    debt = get_debt_safe(bs)

    eps_ttm = safe_float(data.get("trailing_eps"))
    if eps_ttm == 0:
        inc_q = data.get("inc_q")
        inc_a = data.get("inc_a")
        net_inc_ttm = (
            get_ttm_or_latest(inc_q, ["NetIncome", "Net Income Common Stockholders"])
            if hasattr(inc_q, "empty") and not inc_q.empty
            else 0
        )
        if net_inc_ttm == 0:
            net_inc_ttm = get_item_safe(inc_a, ["NetIncome", "Net Income Common Stockholders"])
        if net_inc_ttm == 0:
            net_inc_ttm = get_ttm_or_latest(inc, ["NetIncome", "Net Income Common Stockholders"])
        if shares > 0:
            eps_ttm = net_inc_ttm / shares

    pe = safe_float(data.get("pe_ratio"))
    if pe == 0 and eps_ttm > 0:
        pe = current_price / eps_ttm

    forward_pe = safe_float(data.get("forward_pe"))
    quote_currency = str(data.get("quote_currency") or "N/A")
    financial_currency = str(data.get("financial_currency") or quote_currency or "N/A")
    ps = market_cap / revenue_ttm if revenue_ttm > 0 else 0

    sales_growth = safe_float(data.get("rev_growth"))
    eps_growth = safe_float(data.get("eps_growth"))
    bench_data = get_benchmark_data(ticker, data.get("sector", "Default"))
    is_financial = is_financial_company(data.get("sector", "Default"), bench_data.get("name"))

    metrics = {
        "company_name": data.get("long_name", ticker),
        "ticker": ticker,
        "sector_name": data.get("sector", "Default"),
        "benchmark_name": bench_data.get("name"),
        "business_model_hint": business_model_hint(data.get("sector", "Default"), bench_data.get("name")),
        "price": current_price,
        "pe": pe,
        "forward_pe": forward_pe,
        "ps": ps,
        "sales_gr": sales_growth,
        "eps_gr": eps_growth,
        "dividend_yield": safe_float(data.get("dividend_yield")),
        "trailing_eps": eps_ttm,
        "quote_currency": quote_currency,
        "financial_currency": financial_currency,
        "net_cash": 0.0 if is_financial else cash - debt,
        "fcf_yield": 0.0 if is_financial else (fcf_ttm / market_cap if market_cap else 0),
        "rule_40": 0.0 if is_financial else sales_growth + ((fcf_ttm / revenue_ttm) if revenue_ttm else 0),
        "piotroski": piotroski,
        "altman_z": None if is_financial else altman_z,
        "is_financial": is_financial,
    }
    scores = score_out_of_10(metrics, bench_data)
    data_quality_report = build_data_quality_report(
        price=current_price,
        shares=shares,
        market_cap=market_cap,
        revenue_ttm=revenue_ttm,
        eps_ttm=eps_ttm,
        fcf_ttm=fcf_ttm,
        balance_sheet=bs,
        income_statement=inc,
        cash_flow=cf,
        fetcher_warnings=list(data.get("data_quality_reasons") or data.get("warnings") or []),
        fetcher_error=data.get("error"),
    )
    valuation_warnings: list[str] = []
    if share_count_estimated:
        valuation_warnings.append("Share count estimated from reported market cap and current price.")
    if share_count_unavailable:
        valuation_warnings.append("Share count unavailable; per-share values are not reliable until manually overridden.")
    if revenue_ttm <= 0:
        valuation_warnings.append("Revenue TTM unavailable; P/S valuation is disabled.")
    if eps_ttm <= 0:
        valuation_warnings.append("EPS TTM unavailable or negative; P/E valuation is disabled.")
    if fcf_ttm <= 0 and not is_financial:
        valuation_warnings.append("FCF TTM unavailable or negative; DCF valuation is disabled.")

    return AnalyzerSnapshot(
        ticker=ticker,
        data=data,
        bench_data=bench_data,
        metrics=metrics,
        scores=scores,
        bs=bs,
        inc=inc,
        cf=cf,
        current_price=current_price,
        shares=shares,
        market_cap=market_cap,
        piotroski=piotroski,
        altman_z=None if is_financial else altman_z,
        revenue_ttm=revenue_ttm,
        cfo_ttm=cfo_ttm,
        capex_ttm=capex_ttm,
        fcf_ttm=fcf_ttm,
        cash=cash,
        debt=debt,
        eps_ttm=eps_ttm,
        pe=pe,
        forward_pe=forward_pe,
        ps=ps,
        sales_growth=sales_growth,
        eps_growth=eps_growth,
        quote_currency=quote_currency,
        financial_currency=financial_currency,
        is_financial=is_financial,
        next_earnings=extract_next_earnings(data.get("calendar")),
        revenue_basis=statement_basis_label(data.get("inc_q"), data.get("inc_a"), "Yahoo info fallback"),
        eps_basis=(
            "Yahoo trailing EPS"
            if safe_float(data.get("trailing_eps"))
            else statement_basis_label(data.get("inc_q"), data.get("inc_a"), "Net income fallback")
        ),
        recent_news_count=len(data.get("news") or []),
        press_release_count=len(data.get("ir_news") or []),
        manual_shares_applied=manual_shares_applied,
        share_count_unavailable=share_count_unavailable,
        share_count_estimated=share_count_estimated,
        data_quality=data_quality_report.status,
        data_quality_reasons=[*data_quality_report.blockers, *data_quality_report.warnings],
        valuation_warnings=list(dict.fromkeys(valuation_warnings)),
    )
