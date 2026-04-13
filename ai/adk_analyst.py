"""
Google ADK multi-agent chat analyst for the Streamlit stock analyzer.
"""

from __future__ import annotations

import asyncio
from datetime import date
import hashlib
import json
import os
import re
import threading
import uuid
from typing import Any, Callable

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
from scoring import calculate_altman_z, calculate_piotroski_score, score_out_of_10
from technical import add_indicators, bull_flag_score, fetch_price_history


MODEL_NAME = "gemini-2.5-flash"
APP_NAME = "stock_valuation_multi_agent"
SUPERVISOR_AGENT_NAME = "stock_chat_supervisor"
CHAT_ENGINE_VERSION = "2026-04-13-local-compare-v1"

AGENT_DISPLAY_NAMES = {
    SUPERVISOR_AGENT_NAME: "Superviseur",
    "fundamental_agent": "Fondamentaux",
    "technical_agent": "Technique",
    "peer_agent": "Pairs / secteur",
    "comparison_agent": "Comparaison",
    "news_agent": "Actualites",
    "market_signal_agent": "Signaux de marche",
    "filings_agent": "Filings SEC",
    "risk_agent": "Risque",
}


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


def _format_compact_number(value: Any, suffix: str = "") -> str:
    number = _to_float(value)
    if number is None:
        return "N/A"
    abs_value = abs(number)
    if abs_value >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B{suffix}"
    if abs_value >= 1_000_000:
        return f"{number / 1_000_000:.2f} M{suffix}"
    return f"{number:.2f}{suffix}"


def _format_pct(value: Any) -> str:
    number = _to_percent(value)
    return f"{number:.1f}%" if number is not None else "N/A"


def _format_ratio(value: Any) -> str:
    number = _to_float(value)
    return f"{number:.2f}x" if number is not None else "N/A"


def _extract_function_responses(event: Any) -> list[Any]:
    """Extract ADK function responses from an event when no final text is available."""
    if hasattr(event, "get_function_responses"):
        try:
            responses = event.get_function_responses()
            if responses:
                return list(responses)
        except Exception:
            pass

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    responses = []
    for part in parts:
        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            responses.append(function_response)
    return responses


def _comparison_tool_response_to_text(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("error"):
        return str(payload.get("error"))

    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        return None

    current = comparison.get("current_stock") or {}
    other = comparison.get("other_stock") or {}
    context = comparison.get("comparison_context") or {}
    objective = comparison.get("investor_objective") or {}

    current_name = current.get("company_name") or current.get("ticker") or "L'action actuelle"
    other_name = other.get("company_name") or other.get("ticker") or payload.get("resolved_company_name") or "l'autre action"
    current_ticker = current.get("ticker") or "N/A"
    other_ticker = other.get("ticker") or payload.get("resolved_ticker") or "N/A"
    objective_label = objective.get("label") or "Equilibre"

    lines = [
        f"Voici une comparaison rapide entre {current_name} ({current_ticker}) et {other_name} ({other_ticker}).",
    ]

    if context.get("is_cross_sector"):
        lines.append(
            f"C'est une comparaison intersectorielle: {current_name} est plutot {context.get('current_business_model') or current.get('business_model_hint') or 'dans son secteur'}, "
            f"alors que {other_name} est plutot {context.get('other_business_model') or other.get('business_model_hint') or 'dans un autre secteur'}."
        )

    lines.append(
        f"Croissance: {current_ticker} affiche { _format_pct(current.get('sales_growth_pct')) } de croissance des ventes et { _format_pct(current.get('eps_growth_pct')) } de croissance EPS, "
        f"contre { _format_pct(other.get('sales_growth_pct')) } et { _format_pct(other.get('eps_growth_pct')) } pour {other_ticker}."
    )
    lines.append(
        f"Valorisation: trailing P/E { _format_ratio(current.get('pe_ratio')) } vs { _format_ratio(other.get('pe_ratio')) }, "
        f"forward P/E { _format_ratio(current.get('forward_pe_ratio')) } vs { _format_ratio(other.get('forward_pe_ratio')) }, "
        f"et P/S { _format_ratio(current.get('ps_ratio')) } vs { _format_ratio(other.get('ps_ratio')) }."
    )
    lines.append(
        f"Qualite / bilan: net cash { _format_compact_number(current.get('net_cash'), '$') } vs { _format_compact_number(other.get('net_cash'), '$') }, "
        f"Piotroski {current.get('piotroski_score', 'N/A')} vs {other.get('piotroski_score', 'N/A')}."
    )
    lines.append(
        f"Sous l'angle {objective_label.lower()}, il faut surtout arbitrer entre croissance / rerating et profil defensif / cash generation selon le secteur."
    )

    return "\n\n".join(lines)


def _news_tool_response_to_text(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    market_news = payload.get("recent_market_news") or []
    press_releases = payload.get("recent_press_releases") or []
    company_name = payload.get("company_name") or "la societe"
    snippets = []
    for item in (market_news + press_releases)[:3]:
        title = item.get("title")
        published_at = item.get("published_at")
        source = item.get("source")
        if title:
            prefix = f"{published_at} - " if published_at else ""
            suffix = f" ({source})" if source else ""
            snippets.append(f"- {prefix}{title}{suffix}")
    if not snippets:
        return None
    return f"Voici les elements recents disponibles pour {company_name} :\n\n" + "\n".join(snippets)


def _extract_text_from_tool_payload(payload: Any, tool_name: str | None = None) -> str | None:
    if isinstance(payload, str):
        return payload.strip() or None
    if not isinstance(payload, dict):
        return None
    if payload.get("error"):
        return str(payload.get("error"))

    for key in ["summary", "analysis", "answer", "message", "text", "result"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if tool_name == "compare_against_other_stock" or "comparison" in payload:
        return _comparison_tool_response_to_text(payload)
    if tool_name == "get_recent_news_snapshot" or "recent_market_news" in payload or "recent_press_releases" in payload:
        return _news_tool_response_to_text(payload)

    return None


def _extract_text_from_event(event: Any) -> str | None:
    content = getattr(event, "content", None)
    if content and getattr(content, "parts", None):
        text = _text_from_parts(content.parts)
        if text:
            return text

    for function_response in _extract_function_responses(event):
        tool_name = getattr(function_response, "name", None)
        payload = getattr(function_response, "response", None)
        fallback_text = _extract_text_from_tool_payload(payload, tool_name=tool_name)
        if fallback_text:
            return fallback_text

    return None


def _extract_text_from_session_state(state: Any) -> str | None:
    """Recover a usable answer from session.state when ADK stores structured output there."""
    if not isinstance(state, dict):
        return None

    for key in ["last_answer", "answer", "final_answer", "response"]:
        value = state.get(key)
        text = _extract_text_from_tool_payload(value)
        if text:
            return text
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _extract_text_from_session_events(session: Any) -> str | None:
    """Walk backwards through the recorded session events to find the last useful answer."""
    events = getattr(session, "events", None) or []
    for event in reversed(events):
        if getattr(event, "author", None) == "user":
            continue
        text = _extract_text_from_event(event)
        if text:
            return text
    return None


def _looks_like_comparison_request(user_message: str) -> bool:
    message = (user_message or "").lower()
    markers = ["compare", "compar", "versus", "vs", "avec", "contre", "entre "]
    return any(marker in message for marker in markers)


def _extract_comparison_target_from_message(user_message: str, current_ticker: str) -> str | None:
    """Infer the non-current ticker/company mentioned in a comparison prompt."""
    current_upper = (current_ticker or "").upper()
    tokens = re.findall(r"[A-Za-z][A-Za-z\.\-]{0,12}", user_message or "")

    for token in tokens:
        normalized = _normalize_lookup(token)
        mapped = TICKER_ALIAS_MAP.get(normalized)
        if mapped and mapped != current_upper:
            return mapped
        token_upper = token.upper().rstrip(".,;:!?")
        if 1 < len(token_upper) <= 6 and token_upper != current_upper and token_upper.isascii():
            if token_upper in KNOWN_TICKERS:
                return token_upper

    normalized_message = _normalize_lookup(user_message or "")
    candidates = [
        (alias, ticker)
        for alias, ticker in TICKER_ALIAS_MAP.items()
        if len(alias) >= 4 and alias in normalized_message and ticker != current_upper
    ]
    if candidates:
        best_alias, best_ticker = max(candidates, key=lambda item: len(item[0]))
        if best_alias:
            return best_ticker

    return None


def _build_local_fallback_context(metrics: dict, bench: dict, scores: dict, tech: dict) -> dict[str, Any]:
    """Store a deterministic local stock context for non-LLM fallbacks."""
    current_ticker = str(metrics.get("ticker", "")).upper()
    current_data = get_financial_data_secure(current_ticker, cache_version=FINANCIAL_DATA_CACHE_VERSION)
    return {
        "ticker": current_ticker,
        "company_name": metrics.get("company_name") or current_data.get("long_name") or current_ticker,
        "company": _build_company_snapshot(metrics, scores),
        "peer": _build_peer_snapshot(bench),
        "technical": _build_technical_snapshot(tech),
        "sources": _build_source_refs(current_ticker, current_data),
    }


def _build_local_comparison_response(chat_context: dict[str, Any], user_message: str) -> tuple[str | None, dict[str, Any] | None]:
    """Generate a deterministic comparison answer when ADK fails to synthesize one."""
    fallback_context = chat_context.get("fallback_context") or {}
    current_ticker = fallback_context.get("ticker") or chat_context.get("current_ticker")
    if not current_ticker or not _looks_like_comparison_request(user_message):
        return None, None

    if not fallback_context:
        try:
            fallback_context = _compute_stock_context(current_ticker)
            chat_context["fallback_context"] = fallback_context
        except Exception:
            return None, None

    target_ticker = _extract_comparison_target_from_message(user_message, current_ticker)
    if not target_ticker:
        return None, None

    try:
        other_context = _compute_stock_context(target_ticker)
        payload = {
            "requested_target": target_ticker,
            "resolved_ticker": target_ticker,
            "resolved_company_name": other_context.get("company_name"),
            "comparison": _build_comparison_payload(
                fallback_context.get("company") or {},
                fallback_context.get("peer") or {},
                fallback_context.get("technical") or {},
                fallback_context.get("sources") or {},
                chat_context.get("investor_objective") or {"label": "Equilibre"},
                other_context,
            ),
        }
        text = _comparison_tool_response_to_text(payload)
        if not text:
            return None, None
        trace = _build_agent_trace(["comparison_agent", SUPERVISOR_AGENT_NAME], SUPERVISOR_AGENT_NAME)
        return text, trace
    except Exception as exc:
        return f"Je n'ai pas pu construire la comparaison locale avec {target_ticker}: {exc}", _build_agent_trace(["comparison_agent"], "comparison_agent")


def _agent_display_name(agent_name: str | None) -> str | None:
    if not agent_name:
        return None
    return AGENT_DISPLAY_NAMES.get(agent_name, agent_name.replace("_", " ").strip().title())


def _build_agent_trace(authors_seen: list[str], final_author: str | None = None) -> dict[str, Any] | None:
    """Build a compact agent-activity payload for the Streamlit chat UI."""
    ordered_authors: list[str] = []
    for author in authors_seen:
        clean_author = str(author or "").strip()
        if not clean_author or clean_author == "user":
            continue
        if clean_author not in ordered_authors:
            ordered_authors.append(clean_author)

    if final_author and final_author != "user" and final_author not in ordered_authors:
        ordered_authors.append(final_author)

    if not ordered_authors:
        return None

    specialist_agents = [author for author in ordered_authors if author != SUPERVISOR_AGENT_NAME]
    lead_agent = specialist_agents[-1] if specialist_agents else ordered_authors[-1]
    final_visible_author = final_author if final_author and final_author != "user" else None

    return {
        "authors": ordered_authors,
        "labels": [_agent_display_name(author) or author for author in ordered_authors],
        "specialist_agents": specialist_agents,
        "specialist_labels": [_agent_display_name(author) or author for author in specialist_agents],
        "lead_agent": lead_agent,
        "lead_agent_label": _agent_display_name(lead_agent) or lead_agent,
        "final_author": final_visible_author,
        "final_author_label": _agent_display_name(final_visible_author) if final_visible_author else None,
        "used_supervisor": SUPERVISOR_AGENT_NAME in ordered_authors,
    }


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


def _business_model_hint(sector_name: str | None, benchmark_name: str | None) -> str:
    """Give the model a compact business description to improve cross-sector comparisons."""
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
KNOWN_TICKERS = {
    entry.split(" - ", 1)[0].strip().upper()
    for entry in TICKER_DB
    if " - " in entry and not entry.startswith("---")
}


def _build_company_snapshot(metrics: dict, scores: dict) -> dict[str, Any]:
    return {
        "company_name": metrics.get("company_name"),
        "ticker": metrics.get("ticker"),
        "sector_name": metrics.get("sector_name"),
        "benchmark_name": metrics.get("benchmark_name"),
        "business_model_hint": metrics.get("business_model_hint"),
        "is_financial": bool(metrics.get("is_financial", False)),
        "price": _to_float(metrics.get("price")),
        "pe_ratio": _to_float(metrics.get("pe")),
        "forward_pe_ratio": _to_float(metrics.get("forward_pe")),
        "ps_ratio": _to_float(metrics.get("ps")),
        "sales_growth_pct": _to_percent(metrics.get("sales_gr")),
        "eps_growth_pct": _to_percent(metrics.get("eps_gr")),
        "dividend_yield_pct": _to_percent(metrics.get("dividend_yield")),
        "trailing_eps": _to_float(metrics.get("trailing_eps")),
        "quote_currency": metrics.get("quote_currency"),
        "financial_currency": metrics.get("financial_currency"),
        "net_cash": _to_float(metrics.get("net_cash")),
        "fcf_yield_pct": _to_percent(metrics.get("fcf_yield")),
        "rule_of_40_pct": _to_percent(metrics.get("rule_40")),
        "piotroski_score": _to_int(metrics.get("piotroski")),
        "altman_z_score": _to_float(metrics.get("altman_z")),
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


def _quote_page_url(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{ticker}"


def _sec_company_url(ticker: str) -> str:
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&owner=exclude&count=40"


def _compact_source_links(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    links = []
    for item in items[:limit]:
        if not item.get("title") or not item.get("link"):
            continue
        links.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "published_at": item.get("published_at") or item.get("pubDate"),
                "source": item.get("source"),
            }
        )
    return links


def _build_source_refs(ticker: str, data: dict[str, Any]) -> dict[str, Any]:
    """Provide URLs and source notes that the chat can cite explicitly."""
    market_links = _compact_source_links(data.get("news") or [], limit=4)
    press_links = _compact_source_links(data.get("ir_news") or [], limit=3)

    return {
        "as_of_date": date.today().isoformat(),
        "quote_page": {
            "label": f"{ticker} quote page",
            "url": _quote_page_url(ticker),
            "source": "Yahoo Finance",
        },
        "sec_filings_page": {
            "label": f"{ticker} SEC filings",
            "url": _sec_company_url(ticker),
            "source": "SEC EDGAR",
        },
        "market_news_links": market_links,
        "press_release_links": press_links,
        "valuation_notes": [
            "Trailing P/E prefers Yahoo trailingPE and falls back to current price divided by TTM EPS.",
            "Forward P/E uses Yahoo consensus when available.",
            "Revenue TTM prefers quarterly statements and falls back to the latest annual figure.",
        ],
    }


def _extract_earnings_context(calendar_data: Any) -> dict[str, Any]:
    """Normalize earnings dates so the AI can distinguish upcoming from already passed events."""
    if calendar_data is None:
        return {"next_earnings_date": None, "last_known_earnings_date": None, "all_detected_dates": []}

    candidates: list[pd.Timestamp] = []

    def _collect(value: Any):
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
                _collect(row.tolist() if isinstance(row, pd.Series) else row)
    elif isinstance(calendar_data, pd.Series):
        for label, value in calendar_data.items():
            if "earnings" in str(label).lower():
                _collect(value)
    elif isinstance(calendar_data, dict):
        for label, value in calendar_data.items():
            if "earnings" in str(label).lower():
                _collect(value)

    if not candidates:
        return {"next_earnings_date": None, "last_known_earnings_date": None, "all_detected_dates": []}

    today = pd.Timestamp(date.today())
    ordered = sorted(ts.normalize() for ts in candidates)
    future_candidates = [ts for ts in ordered if ts >= today]
    past_candidates = [ts for ts in ordered if ts < today]

    return {
        "next_earnings_date": future_candidates[0].strftime("%Y-%m-%d") if future_candidates else None,
        "last_known_earnings_date": past_candidates[-1].strftime("%Y-%m-%d") if past_candidates else None,
        "all_detected_dates": [ts.strftime("%Y-%m-%d") for ts in ordered[-4:]],
    }


def _build_recent_news_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    market_news_items = []
    for item in (data.get("news") or [])[:6]:
        market_news_items.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "published_at": item.get("published_at"),
                "source": item.get("source"),
            }
        )

    press_release_items = []
    for item in (data.get("ir_news") or [])[:5]:
        press_release_items.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "published_at": item.get("published_at") or item.get("pubDate"),
                "source": item.get("source"),
            }
        )

    earnings_context = _extract_earnings_context(data.get("calendar"))

    return {
        "as_of_date": date.today().isoformat(),
        "company_name": data.get("long_name"),
        "recent_market_news": market_news_items,
        "recent_press_releases": press_release_items,
        "earnings_context": earnings_context,
        "source_links": (market_news_items + press_release_items)[:6],
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
        inc_q = data.get("inc_q")
        inc_a = data.get("inc_a")
        net_income_ttm = (
            get_ttm_or_latest(inc_q, ["NetIncome", "Net Income Common Stockholders"])
            if hasattr(inc_q, "empty") and not inc_q.empty
            else 0.0
        )
        if net_income_ttm == 0:
            net_income_ttm = get_item_safe(inc_a, ["NetIncome", "Net Income Common Stockholders"])
        if net_income_ttm == 0:
            net_income_ttm = get_ttm_or_latest(inc, ["NetIncome", "Net Income Common Stockholders"])
        eps_ttm = net_income_ttm / shares if shares else 0

    pe = float(data.get("pe_ratio", 0) or 0)
    if pe == 0 and eps_ttm > 0:
        pe = current_price / eps_ttm

    market_cap = shares * current_price if shares > 0 else float(data.get("market_cap", 0) or 0)
    ps = market_cap / revenue_ttm if market_cap > 0 and revenue_ttm > 0 else 0

    benchmark = get_benchmark_data(ticker, data.get("sector", "Default"))
    is_financial = _is_financial_company(data.get("sector", "Default"), benchmark.get("name"))
    piotroski = calculate_piotroski_score(bs, inc, cf)
    altman_z = None if is_financial else calculate_altman_z(bs, inc, market_cap)

    metrics = {
        "company_name": data.get("long_name", ticker.upper()),
        "ticker": ticker.upper(),
        "sector_name": data.get("sector", "Default"),
        "benchmark_name": benchmark.get("name"),
        "business_model_hint": _business_model_hint(data.get("sector", "Default"), benchmark.get("name")),
        "price": current_price,
        "pe": pe,
        "forward_pe": float(data.get("forward_pe", 0) or 0),
        "ps": ps,
        "sales_gr": float(data.get("rev_growth", 0) or 0),
        "eps_gr": float(data.get("eps_growth", 0) or 0),
        "dividend_yield": float(data.get("dividend_yield", 0) or 0),
        "trailing_eps": eps_ttm,
        "quote_currency": data.get("quote_currency"),
        "financial_currency": data.get("financial_currency"),
        "net_cash": 0.0 if is_financial else cash - debt,
        "fcf_yield": 0.0 if is_financial else ((fcf_ttm / market_cap) if market_cap else 0),
        "rule_40": 0.0 if is_financial else float(data.get("rev_growth", 0) or 0) + ((fcf_ttm / revenue_ttm) if revenue_ttm else 0),
        "piotroski": piotroski,
        "altman_z": altman_z,
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
        "sources": _build_source_refs(ticker.upper(), data),
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


def _build_comparison_payload(
    current_company: dict[str, Any],
    current_peer: dict[str, Any],
    current_technical: dict[str, Any],
    current_sources: dict[str, Any],
    investor_objective: dict[str, Any],
    other_context: dict[str, Any],
) -> dict[str, Any]:
    other_company = other_context["company"]
    other_peer = other_context["peer"]
    other_technical = other_context["technical"]
    is_cross_sector = (current_company.get("benchmark_name") or "") != (other_company.get("benchmark_name") or "")
    current_name = current_company.get("company_name") or current_company.get("ticker")
    other_name = other_company.get("company_name") or other_company.get("ticker")

    return {
        "current_stock": current_company,
        "other_stock": other_company,
        "current_benchmark": current_peer,
        "other_benchmark": other_peer,
        "current_technical": current_technical,
        "other_technical": other_technical,
        "investor_objective": investor_objective,
        "comparison_context": {
            "is_cross_sector": is_cross_sector,
            "current_company_name": current_name,
            "other_company_name": other_name,
            "current_sector_name": current_company.get("sector_name"),
            "other_sector_name": other_company.get("sector_name"),
            "current_business_model": current_company.get("business_model_hint"),
            "other_business_model": other_company.get("business_model_hint"),
            "current_is_financial": current_company.get("is_financial"),
            "other_is_financial": other_company.get("is_financial"),
            "framing_note": (
                f"{current_name} and {other_name} are in different market buckets, so the answer should be conditional on investor objective."
                if is_cross_sector
                else f"{current_name} and {other_name} are close enough to compare on a more direct basis."
            ),
        },
        "source_refs": {
            "current_quote_page": current_sources.get("quote_page"),
            "other_quote_page": (other_context.get("sources") or {}).get("quote_page"),
            "current_recent_links": (current_sources.get("market_news_links") or [])[:2],
            "other_recent_links": ((other_context.get("sources") or {}).get("market_news_links") or [])[:2],
            "current_press_links": (current_sources.get("press_release_links") or [])[:2],
            "other_press_links": ((other_context.get("sources") or {}).get("press_release_links") or [])[:2],
        },
        "comparison_highlights": {
            "sales_growth_gap_pct_points": _to_float(
                (current_company.get("sales_growth_pct") or 0) - (other_company.get("sales_growth_pct") or 0)
            ),
            "eps_growth_gap_pct_points": _to_float(
                (current_company.get("eps_growth_pct") or 0) - (other_company.get("eps_growth_pct") or 0)
            ),
            "pe_gap": _to_float((current_company.get("pe_ratio") or 0) - (other_company.get("pe_ratio") or 0)),
            "forward_pe_gap": _to_float(
                (current_company.get("forward_pe_ratio") or 0) - (other_company.get("forward_pe_ratio") or 0)
            ),
            "ps_gap": _to_float((current_company.get("ps_ratio") or 0) - (other_company.get("ps_ratio") or 0)),
            "fcf_yield_gap_pct_points": _to_float(
                (current_company.get("fcf_yield_pct") or 0) - (other_company.get("fcf_yield_pct") or 0)
            ),
            "net_cash_gap": _to_float((current_company.get("net_cash") or 0) - (other_company.get("net_cash") or 0)),
            "piotroski_gap": _to_float(
                (current_company.get("piotroski_score") or 0) - (other_company.get("piotroski_score") or 0)
            ),
            "altman_gap": _to_float(
                (current_company.get("altman_z_score") or 0) - (other_company.get("altman_z_score") or 0)
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


def build_ai_chat_signature(metrics: dict, bench: dict, scores: dict, tech: dict, objective_snapshot: dict | None = None) -> str:
    """
    Build a stable signature for the currently analyzed stock context.
    """

    payload = {
        "chat_engine_version": CHAT_ENGINE_VERSION,
        "cache_version": FINANCIAL_DATA_CACHE_VERSION,
        "refresh_date": date.today().isoformat(),
        "investor_objective": objective_snapshot or {"key": "balanced"},
        "company": _build_company_snapshot(metrics, scores),
        "peer": _build_peer_snapshot(bench),
        "technical": _build_technical_snapshot(tech),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _build_agent(metrics: dict, bench: dict, scores: dict, tech: dict, objective_snapshot: dict | None = None):
    company_snapshot = _build_company_snapshot(metrics, scores)
    peer_snapshot = _build_peer_snapshot(bench)
    technical_snapshot = _build_technical_snapshot(tech)
    current_ticker = str(metrics.get("ticker", "")).upper()
    current_data = get_financial_data_secure(current_ticker, cache_version=FINANCIAL_DATA_CACHE_VERSION)
    news_snapshot = _build_recent_news_snapshot(current_data)
    analyst_snapshot = _build_analyst_snapshot(current_data)
    insider_snapshot = _build_insider_snapshot(current_data)
    source_snapshot = _build_source_refs(current_ticker, current_data)
    objective_snapshot = objective_snapshot or {
        "key": "balanced",
        "label": "Equilibre",
        "description": "mix upside, qualite financiere et valorisation",
        "decision_frame": "balanced mix of upside, quality and valuation",
    }

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
            "investor_objective": objective_snapshot,
            "sources": source_snapshot,
        }

    def get_investor_objective_snapshot() -> dict[str, Any]:
        """Returns the active investor objective selected in the Streamlit UI."""

        return objective_snapshot

    def get_context_sources() -> dict[str, Any]:
        """Returns source links and provenance notes for the selected stock."""

        return source_snapshot

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
                source_snapshot,
                objective_snapshot,
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
Use `get_company_snapshot`, `get_peer_snapshot`, `get_technical_snapshot`, `get_investor_objective_snapshot`, `get_context_sources` and `compare_against_other_stock`.
When the user asks to compare the selected stock with another company or ticker, call `compare_against_other_stock`.
In every comparison, identify what each company does and whether it is a same-sector or cross-sector comparison.
Use the active investor objective as the default framing unless the user explicitly asks for another angle.
When you quote a P/E ratio, say clearly whether it is trailing or forward if that matters.
If trailing and forward P/E tell different stories, mention that explicitly.
If it is cross-sector, do not give a single absolute winner without conditions. Explain the trade-off clearly:
- which stock looks stronger for growth and upside
- which stock looks stronger for balance-sheet resilience, cash generation or defensiveness
- which stock looks stronger for income or shareholder return if the active objective is income
- what sector-specific risk changes the conclusion
If one company is an energy producer, bank, insurer, or other cyclical/regulatory business, say so explicitly and explain why the valuation framework differs.
If the data suggests a split verdict, say for example: "meilleur pour croissance", "meilleur pour profil defensif", or "cela depend de ton objectif".
If a recent earnings release likely changed the trailing valuation multiple, mention that date and explain the shift briefly.
When balance-sheet metrics, Piotroski, Altman, net cash or benchmark context change the answer, mention them explicitly.
End with a short `Sources` section in markdown bullets when URLs are available.
Do not invent data. If the comparison tool returns an error, explain it plainly.
""",
        tools=[get_company_snapshot, get_peer_snapshot, get_technical_snapshot, get_investor_objective_snapshot, get_context_sources, compare_against_other_stock],
    )

    news_agent = LlmAgent(
        name="news_agent",
        model=MODEL_NAME,
        description="Handles recent news, investor-relations headlines, earnings dates and near-term catalysts.",
        instruction="""
You are the recent-news and catalysts analyst.
Answer in French.
Use `get_recent_news_snapshot`, `get_analyst_market_snapshot` and `get_context_sources`.
Summarize the most recent available headlines, earnings date information and likely near-term catalysts.
Be explicit about dates.
Distinguish clearly between broader market/newswire coverage and press-release-style headlines.
Lead with the freshest dated item available, not with an older earnings-calendar item.
Never describe a past earnings date as if it were still upcoming.
If the latest available official release is not very recent, say so plainly instead of implying the news is fresh.
Mention source names when possible.
End with a short `Sources` section using markdown links when URLs are available.
""",
        tools=[get_recent_news_snapshot, get_analyst_market_snapshot, get_context_sources],
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
        name=SUPERVISOR_AGENT_NAME,
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
- `get_investor_objective_snapshot`
- `get_context_sources`
- `compare_against_other_stock`
- `get_recent_news_snapshot`
- `get_analyst_market_snapshot`
- `get_insider_activity_snapshot`
- `get_short_interest_snapshot`
- `get_sec_filing_snapshot`

Rules:
- For simple factual questions, you may use `get_full_snapshot`.
- If the user wants to compare the selected stock with another stock or company, call `comparison_agent` or `compare_against_other_stock`.
- Use the active investor objective as the default frame for comparisons unless the user specifies another lens.
- If the user asks about recent news, latest updates, catalysts, press releases or the next earnings date, call `news_agent`.
- If the user asks about analysts, price targets, insider trades, short interest or market sentiment, call `market_signal_agent`.
- If the user asks for official filed numbers, SEC filings, accounting quality or multi-year reported trends, call `filings_agent`.
- For deeper questions, recommendations, risks, buy/sell opinions, or broad summaries, call one or more specialist agents and then synthesize.
- If the user asks whether the stock looks attractive, mention both upside and risk.
- For cross-sector comparisons, avoid declaring one stock universally "better" unless the objective is explicit. Prefer a conditional answer that separates growth/upside from defensiveness/resilience.
- In cross-sector comparisons, state clearly what each company is and why their sector changes the interpretation of valuation and risk metrics.
- If the answer relies on news, filings or a comparison payload with URLs, finish with a short `Sources` section in markdown bullets.
- If the user asks follow-up questions, use the conversation context.
- Do not invent data outside the provided tools.
- Remind the user this is educational analysis and not professional financial advice whenever the question sounds like an investment decision.
""",
        tools=[
            get_full_snapshot,
            get_investor_objective_snapshot,
            get_context_sources,
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


def create_ai_chat_session(
    metrics: dict,
    bench: dict,
    scores: dict,
    tech: dict,
    api_key: str,
    objective_snapshot: dict | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Create a reusable ADK chat session for the current Streamlit stock context.
    """

    resolved_api_key = _resolve_api_key(api_key)
    if not resolved_api_key:
        return None, "Google API key missing. Add it in the sidebar or in Streamlit secrets as GOOGLE_API_KEY."

    os.environ["GOOGLE_API_KEY"] = resolved_api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

    try:
        agent = _build_agent(metrics, bench, scores, tech, objective_snapshot=objective_snapshot)
        session_service = InMemorySessionService()
        user_id = "streamlit-user"
        session_id = str(uuid.uuid4())
        _run_coro(session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id))
        fallback_context = _build_local_fallback_context(metrics, bench, scores, tech)

        return {
            "runner": Runner(agent=agent, app_name=APP_NAME, session_service=session_service),
            "session_service": session_service,
            "user_id": user_id,
            "session_id": session_id,
            "current_ticker": fallback_context.get("ticker"),
            "investor_objective": objective_snapshot or {"label": "Equilibre", "description": "mix upside, qualite financiere et valorisation"},
            "fallback_context": fallback_context,
        }, None
    except Exception as exc:
        return None, f"Echec de l'initialisation du chat multi-agents: {exc}"


def chat_with_ai_analyst(
    chat_context: dict[str, Any],
    user_message: str,
    on_trace_update: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """
    Send a new user message to the ADK analyst chat session.
    """

    try:
        local_fallback_answer, local_trace = _build_local_comparison_response(chat_context, user_message)
        if local_fallback_answer:
            return local_fallback_answer, None, local_trace

        objective = chat_context.get("investor_objective") or {}
        objective_note = ""
        if objective:
            objective_note = (
                f"\n\n[Contexte interface: objectif d'investisseur actif = {objective.get('label', 'Equilibre')} - "
                f"{objective.get('description', 'mix upside, qualite financiere et valorisation')}. "
                "Utilise-le comme cadrage par defaut pour les comparaisons si l'utilisateur n'en precise pas un autre.]"
            )

        format_note = (
            "\n[Format souhaite: si tu cites de l'actualite, des filings ou une comparaison avec des URLs disponibles, "
            "termine par une section `Sources` avec 1 a 4 puces markdown.]"
        )

        content = types.Content(role="user", parts=[types.Part(text=user_message + objective_note + format_note)])
        final_answer = None
        final_author = None
        authors_seen: list[str] = []
        last_trace_signature = None

        def _emit_trace(trace_payload: dict[str, Any] | None):
            nonlocal last_trace_signature
            if not on_trace_update or not trace_payload:
                return
            signature = json.dumps(trace_payload, sort_keys=True, ensure_ascii=True)
            if signature == last_trace_signature:
                return
            try:
                on_trace_update(trace_payload)
            except Exception:
                return
            last_trace_signature = signature

        events = chat_context["runner"].run(
            user_id=chat_context["user_id"],
            session_id=chat_context["session_id"],
            new_message=content,
        )

        for event in events:
            event_author = getattr(event, "author", None)
            if event_author and event_author != "user":
                authors_seen.append(str(event_author))
                _emit_trace(_build_agent_trace(authors_seen, final_author))
            if event.is_final_response():
                if event_author and event_author != "user":
                    final_author = str(event_author)
                text = _extract_text_from_event(event)
                if text:
                    final_answer = text
                    _emit_trace(_build_agent_trace(authors_seen, final_author))

        if not final_answer:
            session = chat_context["session_service"].get_session(
                app_name=APP_NAME,
                user_id=chat_context["user_id"],
                session_id=chat_context["session_id"],
            )
            session = _run_coro(session)
            if session and getattr(session, "state", None):
                final_answer = _extract_text_from_session_state(session.state)
            if not final_answer and session:
                final_answer = _extract_text_from_session_events(session)

        trace_payload = _build_agent_trace(authors_seen, final_author)
        if not final_answer:
            local_fallback_answer, local_trace = _build_local_comparison_response(chat_context, user_message)
            if local_fallback_answer:
                return local_fallback_answer, None, local_trace or trace_payload
            return None, "Le chat multi-agents n'a pas retourne de reponse finale.", trace_payload
        return final_answer, None, trace_payload
    except Exception as exc:
        return None, f"Echec du chat multi-agents: {exc}", None


def ai_analyst_report(metrics: dict, bench: dict, scores: dict, tech: dict, api_key: str) -> tuple[str | None, str | None]:
    """
    Backward-compatible one-shot report generation using the chat engine.
    """

    chat_context, error = create_ai_chat_session(metrics, bench, scores, tech, api_key)
    if error:
        return None, error
    reply, error, _trace = chat_with_ai_analyst(
        chat_context,
        f"Donne-moi une synthese executive de l'action {metrics.get('ticker', 'N/A')} avec forces, risques, verdict et avertissement.",
    )
    return reply, error
