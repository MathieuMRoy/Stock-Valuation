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

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import agent_tool
from google.genai import types

from data import TICKER_DB, get_benchmark_data
from fetchers import get_financial_data_secure, get_item_safe, get_ttm_or_latest
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
    debt = get_item_safe(bs, ["LongTermDebt"]) + get_item_safe(bs, ["LeaseLiabilities", "TotalLiab"])

    eps_ttm = float(data.get("trailing_eps", 0) or 0)
    if eps_ttm == 0 and shares > 0:
        net_income_ttm = get_ttm_or_latest(inc, ["NetIncome", "Net Income Common Stockholders"])
        eps_ttm = net_income_ttm / shares if shares else 0

    pe = float(data.get("pe_ratio", 0) or 0)
    if pe == 0 and eps_ttm > 0:
        pe = current_price / eps_ttm

    market_cap = shares * current_price if shares > 0 else float(data.get("market_cap", 0) or 0)
    ps = market_cap / revenue_ttm if market_cap > 0 and revenue_ttm > 0 else 0

    metrics = {
        "ticker": ticker.upper(),
        "price": current_price,
        "pe": pe,
        "ps": ps,
        "sales_gr": float(data.get("rev_growth", 0) or 0),
        "eps_gr": float(data.get("eps_growth", 0) or 0),
        "net_cash": cash - debt,
        "fcf_yield": (fcf_ttm / market_cap) if market_cap else 0,
        "rule_40": float(data.get("rev_growth", 0) or 0) + ((fcf_ttm / revenue_ttm) if revenue_ttm else 0),
    }

    benchmark = get_benchmark_data(ticker, data.get("sector", "Default"))
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
- `risk_agent` for investor fit and downside

You also have `get_full_snapshot` for quick factual answers and `compare_against_other_stock` for comparing the selected stock with another ticker or company name.

Rules:
- For simple factual questions, you may use `get_full_snapshot`.
- If the user wants to compare the selected stock with another stock or company, call `comparison_agent` or `compare_against_other_stock`.
- For deeper questions, recommendations, risks, buy/sell opinions, or broad summaries, call one or more specialist agents and then synthesize.
- If the user asks whether the stock looks attractive, mention both upside and risk.
- If the user asks follow-up questions, use the conversation context.
- Do not invent data outside the provided tools.
- Remind the user this is educational analysis and not professional financial advice whenever the question sounds like an investment decision.
""",
        tools=[
            get_full_snapshot,
            compare_against_other_stock,
            agent_tool.AgentTool(agent=fundamental_agent),
            agent_tool.AgentTool(agent=technical_agent),
            agent_tool.AgentTool(agent=peer_agent),
            agent_tool.AgentTool(agent=comparison_agent),
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
