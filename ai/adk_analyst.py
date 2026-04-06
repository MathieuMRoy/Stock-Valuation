"""
Google ADK multi-agent chat analyst for the Streamlit stock analyzer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import agent_tool
from google.genai import types


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
- `risk_agent` for investor fit and downside

You also have `get_full_snapshot` for quick factual answers.

Rules:
- For simple factual questions, you may use `get_full_snapshot`.
- For deeper questions, recommendations, risks, buy/sell opinions, or broad summaries, call one or more specialist agents and then synthesize.
- If the user asks whether the stock looks attractive, mention both upside and risk.
- If the user asks follow-up questions, use the conversation context.
- Do not invent data outside the provided tools.
- Remind the user this is educational analysis and not professional financial advice whenever the question sounds like an investment decision.
""",
        tools=[
            get_full_snapshot,
            agent_tool.AgentTool(agent=fundamental_agent),
            agent_tool.AgentTool(agent=technical_agent),
            agent_tool.AgentTool(agent=peer_agent),
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
