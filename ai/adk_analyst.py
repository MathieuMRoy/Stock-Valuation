"""
Google ADK multi-agent chat analyst for the Streamlit stock analyzer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import uuid
from typing import Any

import pandas as pd
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import agent_tool
from google.genai import types

from data import TICKER_DB, get_benchmark_data
from fetchers import get_debt_safe, get_financial_data_secure, get_item_safe, get_ttm_or_latest
from fetchers.sec_edgar import get_sec_financials
from fetchers.short_interest import get_historical_short_interest
from fetchers.yahoo_finance import FINANCIAL_DATA_CACHE_VERSION
from scoring import score_out_of_10
from technical import add_indicators, bull_flag_score, fetch_price_history


MODEL_NAME = "gemini-2.5-flash"
APP_NAME = "stock_valuation_multi_agent"


def _run_coro(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, Exception] = {}

    def _worker():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            error["value"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    if error:
        raise error["value"]
    return result.get("value")


def _to_percent(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1.5:
        number *= 100
    return round(number, 2)


def _to_float(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _to_float(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _text_from_parts(parts: list[Any]) -> str:
    chunks = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def _resolve_api_key(api_key: str | None) -> str | None:
    if api_key and api_key.strip():
        return api_key.strip()
    return os.getenv("GOOGLE_API_KEY")


def _normalize_lookup(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _is_financial_company(sector_name: str | None, benchmark_name: str | None) -> bool:
    """Detect financial institutions that should avoid operating-company shortcuts."""
    text = " ".join([str(sector_name or ""), str(benchmark_name or "")]).lower()
    keywords = ["financial", "bank", "banks", "insurance", "capital markets", "asset management"]
    return any(keyword in text for keyword in keywords)


def _generate_aliases(company_name: str) -> set[str]:
    aliases = {company_name.strip()}
    base_name = company_name.split("(")[0].strip()
    if base_name:
        aliases.add(base_name)

    match = re.search(r"\(([^)]+)\)", company_name)
    if match:
        aliases.add(match.group(1).strip())

    trimmed = re.sub(
        r"\b(inc|corp|corporation|technologies|technology|platforms|holdings|group|company|co|ltd|limited|bank|pharma|usa)\b\.?",
        "",
        base_name,
        flags=re.IGNORECASE,
    )
    trimmed = " ".join(trimmed.split()).strip(" ,-")
    if trimmed:
        aliases.add(trimmed)

    return {alias for alias in aliases if alias}


def _build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for entry in TICKER_DB:
        if entry.startswith("---") or "Other" in entry or " - " not in entry:
            continue
        ticker, company_name = entry.split(" - ", 1)
        clean_ticker = ticker.strip().upper()
        alias_map[_normalize_lookup(clean_ticker)] = clean_ticker
        for alias in _generate_aliases(company_name):
            alias_map[_normalize_lookup(alias)] = clean_ticker
    return alias_map


TICKER_ALIAS_MAP = _build_alias_map()


def _build_company_snapshot(metrics: dict, scores: dict) -> dict[str, Any]:
    return {
        "ticker": metrics.get("ticker"),
        "price": _to_float(metrics.get("price")),
        "pe_ratio": _to_float(metrics.get("pe")),
        "ps_ratio": _to_float(metrics.get("ps")),
        "sales_growth_pct": _to_percent(metrics.get("sales_gr")),
        "eps_growth_pct": _to_percent(metrics.get("eps_gr")),
        "net_cash": _to_float(metrics.get("net_cash")),
        "fcf_yield_pct": _to_percent(metrics.get("fcf_yield")),
        "rule_of_40_pct": _to_percent(metrics.get("rule_40")),
        "health_score": _to_float(scores.get("health")),
        "growth_score": _to_float(scores.get("growth")),
        "valuation_score": _to_float(scores.get("valuation")),
    }


def _build_peer_snapshot(bench: dict) -> dict[str, Any]:
    return {
        "benchmark_name": bench.get("name"),
        "peer_group": bench.get("peers"),
        "peer_sales_growth_pct": _to_percent(bench.get("gr_sales")),
        "peer_eps_growth_pct": _to_percent(bench.get("gr_eps")),
        "peer_target_ps": _to_float(bench.get("ps")),
        "peer_target_pe": _to_float(bench.get("pe")),
        "wacc_pct": _to_float(bench.get("wacc")),
    }


def _build_technical_snapshot(tech: dict) -> dict[str, Any]:
    return {
        "technical_score_out_of_10": _to_float(tech.get("score")),
        "bull_flag_detected": bool(tech.get("is_bull_flag")),
    }


def _build_recent_news_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    news_items = []
    for item in (data.get("ir_news") or [])[:5]:
        news_items.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "published_at": item.get("pubDate"),
            }
        )

    calendar = data.get("calendar") or {}
    calendar_snapshot = {}
    if isinstance(calendar, dict):
        for key, value in calendar.items():
            calendar_snapshot[str(key)] = _json_safe(value)

    return {
        "company_name": data.get("long_name"),
        "recent_press_releases": news_items,
        "earnings_calendar": calendar_snapshot,
    }


def _dominant_rating_label(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    label_map = {
        "strongBuy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strongSell": "Strong Sell",
    }
    best_key = max(counts, key=lambda key: counts[key])
    if counts.get(best_key, 0) <= 0:
        return None
    return label_map.get(best_key, best_key)


def _build_analyst_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    current_price = float(data.get("price", 0) or 0)
    target_price = data.get("target_price")
    reco_summary = data.get("reco_summary")

    snapshot: dict[str, Any] = {
        "current_price": _to_float(current_price),
        "target_price": _to_float(target_price),
        "target_upside_pct": _to_percent(((float(target_price) - current_price) / current_price) if target_price and current_price > 0 else None),
    }

    if hasattr(reco_summary, "empty") and not reco_summary.empty:
        latest = reco_summary.iloc[0].to_dict()
        counts = {
            "strongBuy": int(latest.get("strongBuy", 0) or 0),
            "buy": int(latest.get("buy", 0) or 0),
            "hold": int(latest.get("hold", 0) or 0),
            "sell": int(latest.get("sell", 0) or 0),
            "strongSell": int(latest.get("strongSell", 0) or 0),
        }
        snapshot["ratings_period"] = latest.get("period")
        snapshot["ratings_breakdown"] = counts
        snapshot["dominant_rating"] = _dominant_rating_label(counts)

    calendar = data.get("calendar") or {}
    if isinstance(calendar, dict):
        snapshot["earnings_calendar"] = {str(key): _json_safe(value) for key, value in calendar.items()}

    return snapshot


def _build_insider_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    insiders = data.get("insiders")
    if not hasattr(insiders, "empty") or insiders.empty:
        return {"available": False, "recent_transactions": []}

    snapshot: dict[str, Any] = {"available": True}
    recent_transactions = []
    for _, row in insiders.head(8).iterrows():
        recent_transactions.append(
            {
                "date": _json_safe(row.get("Start Date")),
                "insider": row.get("Insider"),
                "position": row.get("Position"),
                "transaction": row.get("Transaction"),
                "shares": _to_int(row.get("Shares")),
                "value": _to_float(row.get("Value")),
                "ownership": row.get("Ownership"),
            }
        )

    tx_series = insiders["Transaction"].fillna("").astype(str).str.lower()
    value_series = pd.to_numeric(insiders["Value"], errors="coerce").fillna(0)
    snapshot["recent_transactions"] = recent_transactions
    snapshot["purchase_count"] = int(tx_series.str.contains("purchase").sum())
    snapshot["sale_count"] = int(tx_series.str.contains("sale").sum())
    snapshot["award_count"] = int(tx_series.str.contains("award").sum())
    snapshot["total_purchase_value"] = _to_float(value_series[tx_series.str.contains("purchase")].sum())
    snapshot["total_sale_value"] = _to_float(value_series[tx_series.str.contains("sale")].sum())
    return snapshot


def _build_short_interest_snapshot(ticker: str) -> dict[str, Any]:
    df = get_historical_short_interest(ticker)
    if df is None or df.empty:
        return {"ticker": ticker, "available": False}

    working = df.copy()
    for column in ["Short Interest", "Avg Daily Volume", "Days to Cover"]:
        if column in working.columns:
            working[column] = pd.to_numeric(
                working[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    working = working.sort_values("Date", ascending=False).reset_index(drop=True)
    latest = working.iloc[0]
    previous = working.iloc[1] if len(working) > 1 else None

    snapshot: dict[str, Any] = {
        "ticker": ticker,
        "available": True,
        "latest_settlement_date": _json_safe(latest.get("Date")),
        "latest_short_interest": _to_int(latest.get("Short Interest")),
        "latest_avg_daily_volume": _to_int(latest.get("Avg Daily Volume")),
        "latest_days_to_cover": _to_float(latest.get("Days to Cover")),
        "recent_history": [],
    }

    if previous is not None and pd.notna(previous.get("Short Interest")) and float(previous.get("Short Interest") or 0) > 0:
        latest_short = float(latest.get("Short Interest") or 0)
        previous_short = float(previous.get("Short Interest") or 0)
        snapshot["short_interest_change_pct"] = _to_percent((latest_short - previous_short) / previous_short)

    for _, row in working.head(6).iterrows():
        snapshot["recent_history"].append(
            {
                "date": _json_safe(row.get("Date")),
                "short_interest": _to_int(row.get("Short Interest")),
                "days_to_cover": _to_float(row.get("Days to Cover")),
            }
        )

    return snapshot


def _series_history(df: pd.DataFrame, metric_name: str, limit: int) -> list[dict[str, Any]]:
    if df is None or df.empty or metric_name not in df.index:
        return []
    row = df.loc[metric_name]
    history = []
    for period, value in row.items():
        if pd.notna(value):
            history.append({"period": _json_safe(period), "value": _to_float(value)})
    return history[-limit:]


def _build_sec_snapshot(ticker: str) -> dict[str, Any]:
    sec_data = get_sec_financials(ticker)
    if sec_data.get("error"):
        return {"ticker": ticker, "available": False, "error": sec_data.get("error")}

    annual_df = sec_data.get("inc_a", pd.DataFrame())
    quarterly_df = sec_data.get("inc_q", pd.DataFrame())
    snapshot: dict[str, Any] = {"ticker": ticker, "available": True, "source": "SEC EDGAR"}

    latest_annual = {}
    if hasattr(annual_df, "empty") and not annual_df.empty:
        latest_year = annual_df.columns[-1]
        snapshot["latest_annual_period"] = _json_safe(latest_year)
        for metric in ["Total Revenue", "Net Income", "Operating Income", "Gross Profit", "Cash From Operations", "Free Cash Flow", "EPS"]:
            if metric in annual_df.index:
                latest_annual[metric] = _to_float(annual_df.loc[metric, latest_year])
        snapshot["latest_annual_metrics"] = latest_annual
        snapshot["annual_revenue_history"] = _series_history(annual_df, "Total Revenue", limit=5)
        snapshot["annual_net_income_history"] = _series_history(annual_df, "Net Income", limit=5)

    latest_quarter = {}
    if hasattr(quarterly_df, "empty") and not quarterly_df.empty:
        latest_period = quarterly_df.columns[-1]
        snapshot["latest_quarter_period"] = _json_safe(latest_period)
        for metric in ["Total Revenue", "Net Income", "Operating Income", "Cash From Operations", "Free Cash Flow", "EPS"]:
            if metric in quarterly_df.index:
                latest_quarter[metric] = _to_float(quarterly_df.loc[metric, latest_period])
        snapshot["latest_quarter_metrics"] = latest_quarter
        snapshot["quarterly_revenue_history"] = _series_history(quarterly_df, "Total Revenue", limit=4)

    return snapshot


def _compute_stock_context(ticker: str) -> dict[str, Any]:
    data = get_financial_data_secure(ticker, cache_version=FINANCIAL_DATA_CACHE_VERSION)
    current_price = float(data.get("price", 0) or 0)
    if current_price <= 0:
        raise ValueError(f"Prix introuvable pour {ticker}.")

    shares = float(data.get("shares_info", 0) or 0)
    revenue_ttm = float(data.get("revenue_ttm", 0) or 0)

    inc = data.get("inc")
    cf = data.get("cf")
    bs = data.get("bs")

    if revenue_ttm == 0:
        revenue_ttm = get_ttm_or_latest(inc, ["TotalRevenue", "Revenue"])

    cfo_ttm = get_ttm_or_latest(cf, ["OperatingCashFlow", "Operating Cash Flow"])
    capex_ttm = abs(get_item_safe(cf, ["CapitalExpenditure", "PurchaseOfPPE"]))
    fcf_ttm = cfo_ttm - capex_ttm
    cash = get_item_safe(bs, ["CashAndCashEquivalents", "Cash"])
    debt = get_debt_safe(bs)

    eps_ttm = float(data.get("trailing_eps", 0) or 0)
    if eps_ttm == 0 and shares > 0:
        net_income_ttm = get_ttm_or_latest(inc, ["NetIncome", "Net Income Common Stockholders"])
        eps_ttm = net_income_ttm / shares if shares else 0

    pe = float(data.get("pe_ratio", 0) or 0)
    if pe == 0 and eps_ttm > 0:
        pe = current_price / eps_ttm

    market_cap = shares * current_price if shares > 0 else float(data.get("market_cap", 0) or 0)
    ps = market_cap / revenue_ttm if market_cap > 0 and revenue_ttm > 0 else 0

    benchmark = get_benchmark_data(ticker, data.get("sector", "Default"))
    is_financial = _is_financial_company(data.get("sector", "Default"), benchmark.get("name"))

    metrics = {
        "ticker": ticker.upper(),
        "price": current_price,
        "pe": pe,
        "ps": ps,
        "sales_gr": float(data.get("rev_growth", 0) or 0),
        "eps_gr": float(data.get("eps_growth", 0) or 0),
        "net_cash": 0.0 if is_financial else cash - debt,
        "fcf_yield": 0.0 if is_financial else ((fcf_ttm / market_cap) if market_cap else 0),
        "rule_40": 0.0 if is_financial else float(data.get("rev_growth", 0) or 0) + ((fcf_ttm / revenue_ttm) if revenue_ttm else 0),
        "is_financial": is_financial,
    }

    scores = score_out_of_10(metrics, benchmark)

    price_df = fetch_price_history(ticker, "1y")
    tech_df = add_indicators(price_df)
    technical = bull_flag_score(tech_df)

    return {
        "ticker": ticker.upper(),
        "company_name": data.get("long_name", ticker.upper()),
        "company": _build_company_snapshot(metrics, scores),
        "peer": _build_peer_snapshot(benchmark),
        "technical": _build_technical_snapshot(technical),
    }


def _resolve_requested_ticker(raw_target: str, current_ticker: str) -> str:
    cleaned = (raw_target or "").strip()
    if not cleaned:
        raise ValueError("Aucun ticker ou nom d'entreprise n'a ete fourni.")

    normalized = _normalize_lookup(cleaned)
    if normalized in TICKER_ALIAS_MAP:
        return TICKER_ALIAS_MAP[normalized]

    tokens = re.findall(r"[A-Za-z][A-Za-z\.\-]{0,9}", cleaned.upper())
    for token in tokens:
        token_normalized = _normalize_lookup(token)
        if token_normalized in TICKER_ALIAS_MAP:
            return TICKER_ALIAS_MAP[token_normalized]
        if 1 <= len(token) <= 6:
            return token

    substring_matches = [
        (alias, ticker)
        for alias, ticker in TICKER_ALIAS_MAP.items()
        if alias and alias in normalized
    ]
    if substring_matches:
        best_alias, best_ticker = max(substring_matches, key=lambda item: len(item[0]))
        if best_alias:
            return best_ticker

    fallback = cleaned.upper().replace("$", "").split()[0]
    fallback = fallback.rstrip(".,;:!?")
    if fallback == current_ticker.upper():
        return current_ticker.upper()
    return fallback


def _build_comparison_payload(current_company: dict[str, Any], current_peer: dict[str, Any], current_technical: dict[str, Any], other_context: dict[str, Any]) -> dict[str, Any]:
    other_company = other_context["company"]
    other_peer = other_context["peer"]
    other_technical = other_context["technical"]

    return {
        "current_stock": current_company,
        "other_stock": other_company,
        "current_benchmark": current_peer,
        "other_benchmark": other_peer,
        "current_technical": current_technical,
        "other_technical": other_technical,
        "comparison_highlights": {
            "sales_growth_gap_pct_points": _to_float(
                (current_company.get("sales_growth_pct") or 0) - (other_company.get("sales_growth_pct") or 0)
            ),
            "eps_growth_gap_pct_points": _to_float(
                (current_company.get("eps_growth_pct") or 0) - (other_company.get("eps_growth_pct") or 0)
            ),
            "pe_gap": _to_float((current_company.get("pe_ratio") or 0) - (other_company.get("pe_ratio") or 0)),
            "ps_gap": _to_float((current_company.get("ps_ratio") or 0) - (other_company.get("ps_ratio") or 0)),
            "fcf_yield_gap_pct_points": _to_float(
                (current_company.get("fcf_yield_pct") or 0) - (other_company.get("fcf_yield_pct") or 0)
            ),
            "valuation_score_gap": _to_float(
                (current_company.get("valuation_score") or 0) - (other_company.get("valuation_score") or 0)
            ),
            "growth_score_gap": _to_float(
                (current_company.get("growth_score") or 0) - (other_company.get("growth_score") or 0)
            ),
            "technical_score_gap": _to_float(
                (current_technical.get("technical_score_out_of_10") or 0)
                - (other_technical.get("technical_score_out_of_10") or 0)
            ),
        },
    }


def build_ai_chat_signature(metrics: dict, bench: dict, scores: dict, tech: dict) -> str:
    """
    Build a stable signature for the currently analyzed stock context.
    """

    payload = {
        "company": _build_company_snapshot(metrics, scores),
        "peer": _build_peer_snapshot(bench),
        "technical": _build_technical_snapshot(tech),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _build_agent(metrics: dict, bench: dict, scores: dict, tech: dict):
    company_snapshot = _build_company_snapshot(metrics, scores)
    peer_snapshot = _build_peer_snapshot(bench)
    technical_snapshot = _build_technical_snapshot(tech)
    current_ticker = str(metrics.get("ticker", "")).upper()
    current_data = get_financial_data_secure(current_ticker, cache_version=FINANCIAL_DATA_CACHE_VERSION)
    news_snapshot = _build_recent_news_snapshot(current_data)
    analyst_snapshot = _build_analyst_snapshot(current_data)
    insider_snapshot = _build_insider_snapshot(current_data)

    def get_company_snapshot() -> dict[str, Any]:
        """Returns the core company valuation snapshot for the current Streamlit analysis."""

        return company_snapshot

    def get_peer_snapshot() -> dict[str, Any]:
        """Returns the benchmark and peer assumptions used by the Streamlit app."""

        return peer_snapshot

    def get_technical_snapshot() -> dict[str, Any]:
        """Returns the technical signal summary already computed by the Streamlit app."""

        return technical_snapshot

    def get_full_snapshot() -> dict[str, Any]:
        """Returns the full stock-analysis context for the current Streamlit session."""

        return {
            "company": company_snapshot,
            "peer": peer_snapshot,
            "technical": technical_snapshot,
        }

    def get_recent_news_snapshot() -> dict[str, Any]:
        """Returns recent IR headlines and earnings-calendar catalysts for the selected stock."""

        return news_snapshot

    def get_analyst_market_snapshot() -> dict[str, Any]:
        """Returns target-price and Wall Street ratings information for the selected stock."""

        return analyst_snapshot

    def get_insider_activity_snapshot() -> dict[str, Any]:
        """Returns recent insider transactions and a simple buy/sell summary."""

        return insider_snapshot

    def get_short_interest_snapshot() -> dict[str, Any]:
        """Returns recent short-interest history and days-to-cover for the selected stock."""

        return _build_short_interest_snapshot(current_ticker)

    def get_sec_filing_snapshot() -> dict[str, Any]:
        """Returns summarized official SEC filing data for the selected stock."""

        return _build_sec_snapshot(current_ticker)

    def compare_against_other_stock(target: str) -> dict[str, Any]:
        """Compare the selected stock against another ticker or company name, for example DUOL or Duolingo."""

        resolved_ticker = _resolve_requested_ticker(target, current_ticker)
        if resolved_ticker == current_ticker:
            return {
                "error": f"Le ticker demande ({resolved_ticker}) est le meme que celui deja selectionne ({current_ticker})."
            }

        try:
            other_context = _compute_stock_context(resolved_ticker)
        except Exception as exc:
            return {
                "error": f"Impossible de recuperer les donnees pour {resolved_ticker}: {exc}"
            }

        return {
            "requested_target": target,
            "resolved_ticker": resolved_ticker,
            "resolved_company_name": other_context["company_name"],
            "comparison": _build_comparison_payload(
                company_snapshot,
                peer_snapshot,
                technical_snapshot,
                other_context,
            ),
        }

    fundamental_agent = LlmAgent(
        name="fundamental_agent",
        model=MODEL_NAME,
        description="Handles valuation, growth, profitability and balance-sheet questions.",
        instruction="""
You are the fundamental analyst for the currently selected stock.
Use `get_company_snapshot` and `get_peer_snapshot`.
Answer in French.
Focus on valuation, growth, profitability, free cash flow and relative positioning.
Quote relevant figures whenever possible.
""",
        tools=[get_company_snapshot, get_peer_snapshot],
    )

    technical_agent = LlmAgent(
        name="technical_agent",
        model=MODEL_NAME,
        description="Handles technical, momentum and trading-profile questions.",
        instruction="""
You are the technical analyst for the currently selected stock.
Use `get_company_snapshot` and `get_technical_snapshot`.
Answer in French.
Focus on trend, trading profile, momentum and risk level.
Keep the tone prudent and practical.
""",
        tools=[get_company_snapshot, get_technical_snapshot],
    )

    peer_agent = LlmAgent(
        name="peer_agent",
        model=MODEL_NAME,
        description="Handles peer and benchmark comparison questions.",
        instruction="""
You are the peer comparison analyst.
Use `get_company_snapshot` and `get_peer_snapshot`.
Answer in French.
Explain how the stock compares with its benchmark assumptions and peer group.
""",
        tools=[get_company_snapshot, get_peer_snapshot],
    )

    comparison_agent = LlmAgent(
        name="comparison_agent",
        model=MODEL_NAME,
        description="Handles direct comparisons between the selected stock and another stock requested by the user.",
        instruction="""
You are the direct stock-comparison analyst.
Answer in French.
Use `get_company_snapshot`, `get_peer_snapshot`, `get_technical_snapshot` and `compare_against_other_stock`.
When the user asks to compare the selected stock with another company or ticker, call `compare_against_other_stock`.
Then explain clearly which stock looks stronger on growth, valuation, quality, technical profile and risk/reward.
Do not invent data. If the comparison tool returns an error, explain it plainly.
""",
        tools=[get_company_snapshot, get_peer_snapshot, get_technical_snapshot, compare_against_other_stock],
    )

    news_agent = LlmAgent(
        name="news_agent",
        model=MODEL_NAME,
        description="Handles recent news, investor-relations headlines, earnings dates and near-term catalysts.",
        instruction="""
You are the recent-news and catalysts analyst.
Answer in French.
Use `get_recent_news_snapshot` and `get_analyst_market_snapshot`.
Summarize the most recent available headlines, earnings date information and likely near-term catalysts.
Be explicit about dates.
If the available feed looks like press releases or IR news rather than a full newswire, say so clearly.
""",
        tools=[get_recent_news_snapshot, get_analyst_market_snapshot],
    )

    market_signal_agent = LlmAgent(
        name="market_signal_agent",
        model=MODEL_NAME,
        description="Handles analyst ratings, target prices, insider activity and short-interest questions.",
        instruction="""
You are the market-signals analyst.
Answer in French.
Use `get_analyst_market_snapshot`, `get_insider_activity_snapshot` and `get_short_interest_snapshot`.
Explain what analyst sentiment, insider transactions and short-interest signals suggest, but avoid overclaiming.
If the data is incomplete, say so plainly.
""",
        tools=[get_analyst_market_snapshot, get_insider_activity_snapshot, get_short_interest_snapshot],
    )

    filings_agent = LlmAgent(
        name="filings_agent",
        model=MODEL_NAME,
        description="Handles official SEC filing questions and accounting-quality discussions.",
        instruction="""
You are the SEC filings and accounting-quality analyst.
Answer in French.
Use `get_sec_filing_snapshot` and `get_company_snapshot`.
Focus on official filed numbers, revenue trend, profitability trend, cash-flow quality and notable changes in recent quarters or years.
When the user asks for 'official' numbers, rely on the SEC snapshot first.
""",
        tools=[get_sec_filing_snapshot, get_company_snapshot],
    )

    risk_agent = LlmAgent(
        name="risk_agent",
        model=MODEL_NAME,
        description="Handles investor fit, downside risk and suitability questions.",
        instruction="""
You are the risk analyst.
Use `get_company_snapshot`, `get_peer_snapshot` and `get_technical_snapshot`.
Answer in French.
Explain the main downside risks, resilience factors and suitable investor profile.
""",
        tools=[get_company_snapshot, get_peer_snapshot, get_technical_snapshot],
    )

    return LlmAgent(
        name="stock_chat_supervisor",
        model=MODEL_NAME,
        description="Interactive French chat analyst for the currently selected stock.",
        instruction="""
You are an interactive multi-agent stock-analysis assistant for the currently selected stock.
You answer in French and you keep the conversation natural and concise unless the user asks for detail.

Available specialist agents:
- `fundamental_agent` for valuation and financial quality
- `technical_agent` for momentum and trading profile
- `peer_agent` for benchmark comparison
- `comparison_agent` for direct stock-vs-stock comparisons
- `news_agent` for recent news, catalysts and earnings dates
- `market_signal_agent` for analysts, insiders and short interest
- `filings_agent` for official SEC filings and accounting quality
- `risk_agent` for investor fit and downside

You also have direct tools for quick factual answers:
- `get_full_snapshot`
- `compare_against_other_stock`
- `get_recent_news_snapshot`
- `get_analyst_market_snapshot`
- `get_insider_activity_snapshot`
- `get_short_interest_snapshot`
- `get_sec_filing_snapshot`

Rules:
- For simple factual questions, you may use `get_full_snapshot`.
- If the user wants to compare the selected stock with another stock or company, call `comparison_agent` or `compare_against_other_stock`.
- If the user asks about recent news, latest updates, catalysts, press releases or the next earnings date, call `news_agent`.
- If the user asks about analysts, price targets, insider trades, short interest or market sentiment, call `market_signal_agent`.
- If the user asks for official filed numbers, SEC filings, accounting quality or multi-year reported trends, call `filings_agent`.
- For deeper questions, recommendations, risks, buy/sell opinions, or broad summaries, call one or more specialist agents and then synthesize.
- If the user asks whether the stock looks attractive, mention both upside and risk.
- If the user asks follow-up questions, use the conversation context.
- Do not invent data outside the provided tools.
- Remind the user this is educational analysis and not professional financial advice whenever the question sounds like an investment decision.
""",
        tools=[
            get_full_snapshot,
            compare_against_other_stock,
            get_recent_news_snapshot,
            get_analyst_market_snapshot,
            get_insider_activity_snapshot,
            get_short_interest_snapshot,
            get_sec_filing_snapshot,
            agent_tool.AgentTool(agent=fundamental_agent),
            agent_tool.AgentTool(agent=technical_agent),
            agent_tool.AgentTool(agent=peer_agent),
            agent_tool.AgentTool(agent=comparison_agent),
            agent_tool.AgentTool(agent=news_agent),
            agent_tool.AgentTool(agent=market_signal_agent),
            agent_tool.AgentTool(agent=filings_agent),
            agent_tool.AgentTool(agent=risk_agent),
        ],
        output_key="last_answer",
    )


def create_ai_chat_session(metrics: dict, bench: dict, scores: dict, tech: dict, api_key: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Create a reusable ADK chat session for the current Streamlit stock context.
    """

    resolved_api_key = _resolve_api_key(api_key)
    if not resolved_api_key:
        return None, "Google API key missing. Add it in the sidebar or in Streamlit secrets as GOOGLE_API_KEY."

    os.environ["GOOGLE_API_KEY"] = resolved_api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

    try:
        agent = _build_agent(metrics, bench, scores, tech)
        session_service = InMemorySessionService()
        user_id = "streamlit-user"
        session_id = str(uuid.uuid4())
        _run_coro(session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id))

        return {
            "runner": Runner(agent=agent, app_name=APP_NAME, session_service=session_service),
            "session_service": session_service,
            "user_id": user_id,
            "session_id": session_id,
        }, None
    except Exception as exc:
        return None, f"Echec de l'initialisation du chat multi-agents: {exc}"


def chat_with_ai_analyst(chat_context: dict[str, Any], user_message: str) -> tuple[str | None, str | None]:
    """
    Send a new user message to the ADK analyst chat session.
    """

    try:
        content = types.Content(role="user", parts=[types.Part(text=user_message)])
        final_answer = None
        events = chat_context["runner"].run(
            user_id=chat_context["user_id"],
            session_id=chat_context["session_id"],
            new_message=content,
        )

        for event in events:
            if event.is_final_response() and event.content:
                text = _text_from_parts(event.content.parts)
                if text:
                    final_answer = text

        if not final_answer:
            session = chat_context["session_service"].get_session(
                app_name=APP_NAME,
                user_id=chat_context["user_id"],
                session_id=chat_context["session_id"],
            )
            session = _run_coro(session)
            if session and getattr(session, "state", None):
                final_answer = session.state.get("last_answer")

        if not final_answer:
            return None, "Le chat multi-agents n'a pas retourne de reponse finale."
        return final_answer, None
    except Exception as exc:
        return None, f"Echec du chat multi-agents: {exc}"


def ai_analyst_report(metrics: dict, bench: dict, scores: dict, tech: dict, api_key: str) -> tuple[str | None, str | None]:
    """
    Backward-compatible one-shot report generation using the chat engine.
    """

    chat_context, error = create_ai_chat_session(metrics, bench, scores, tech, api_key)
    if error:
        return None, error
    return chat_with_ai_analyst(
        chat_context,
        f"Donne-moi une synthese executive de l'action {metrics.get('ticker', 'N/A')} avec forces, risques, verdict et avertissement.",
    )
