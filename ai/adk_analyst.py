"""
Google ADK multi-agent analyst for the Streamlit stock analyzer.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"
APP_NAME = "stock_valuation_multi_agent"


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


def _build_agent(metrics: dict, bench: dict, scores: dict, tech: dict):
    company_snapshot = _build_company_snapshot(metrics, scores)
    peer_snapshot = _build_peer_snapshot(bench)
    technical_snapshot = _build_technical_snapshot(tech)

    def get_company_snapshot() -> dict[str, Any]:
        """Returns the core company valuation snapshot for the current Streamlit analysis."""

        return company_snapshot

    def get_peer_snapshot() -> dict[str, Any]:
        """Returns the benchmark and peer data used by the Streamlit app."""

        return peer_snapshot

    def get_technical_snapshot() -> dict[str, Any]:
        """Returns the technical signal summary already computed by the Streamlit app."""

        return technical_snapshot

    fundamental_agent = LlmAgent(
        name="fundamental_agent",
        model=MODEL_NAME,
        description="Analyzes valuation, growth, profitability and balance-sheet quality.",
        instruction="""
You are the fundamental analyst.
Use `get_company_snapshot` and `get_peer_snapshot`.
Write in French and produce:
- Lecture fondamentale
- Forces
- Points de vigilance
- Conclusion fondamentale
Mention the most relevant figures.
""",
        tools=[get_company_snapshot, get_peer_snapshot],
        output_key="fundamental_report",
    )

    technical_agent = LlmAgent(
        name="technical_agent",
        model=MODEL_NAME,
        description="Analyzes technical score, trading profile and short-term momentum.",
        instruction="""
You are the technical analyst.
Use `get_company_snapshot` and `get_technical_snapshot`.
Write in French and produce:
- Lecture technique
- Signaux favorables
- Signaux de risque
- Conclusion technique
Keep the tone prudent.
""",
        tools=[get_company_snapshot, get_technical_snapshot],
        output_key="technical_report",
    )

    peer_agent = LlmAgent(
        name="peer_agent",
        model=MODEL_NAME,
        description="Compares the stock to its benchmark and peer assumptions from the app.",
        instruction="""
You are the peer comparison analyst.
Use `get_company_snapshot` and `get_peer_snapshot`.
Write in French and produce:
- Comparaison au benchmark
- Avantages relatifs
- Faiblesses relatives
- Conclusion comparative
Explain whether the stock looks stronger or weaker than its benchmark assumptions.
""",
        tools=[get_company_snapshot, get_peer_snapshot],
        output_key="peer_report",
    )

    risk_agent = LlmAgent(
        name="risk_agent",
        model=MODEL_NAME,
        description="Assesses investor fit, downside risk and profile suitability.",
        instruction="""
You are the risk analyst.
Use `get_company_snapshot`, `get_peer_snapshot` and `get_technical_snapshot`.
Write in French and produce:
- Profil de risque
- Points de resilience
- Sources de fragilite
- Profil investisseur adapte
""",
        tools=[get_company_snapshot, get_peer_snapshot, get_technical_snapshot],
        output_key="risk_report",
    )

    analysis_team = ParallelAgent(
        name="analysis_team",
        description="Runs the specialized finance sub-agents in parallel.",
        sub_agents=[fundamental_agent, technical_agent, peer_agent, risk_agent],
    )

    decision_agent = LlmAgent(
        name="decision_agent",
        model=MODEL_NAME,
        description="Builds the final investment memo from the sub-agent outputs.",
        instruction="""
You are the final investment memo writer.

Fundamental analysis:
{fundamental_report?}

Technical analysis:
{technical_report?}

Peer comparison:
{peer_report?}

Risk analysis:
{risk_report?}

Write the final answer in French with this structure:
## Synthese executive
## Forces principales
## Risques principaux
## Verdict final
## Avertissement

The verdict must be one of: favorable, neutre, prudent.
Mention an investor profile: defensif, equilibre, offensif.
Finish by saying this is educational analysis and not professional financial advice.
""",
    )

    return SequentialAgent(
        name="stock_analysis_pipeline",
        description="Multi-agent finance analysis pipeline for the Streamlit app.",
        sub_agents=[analysis_team, decision_agent],
    )


def ai_analyst_report(metrics: dict, bench: dict, scores: dict, tech: dict, api_key: str) -> tuple[str | None, str | None]:
    """
    Generate a stock report using a Google ADK multi-agent pipeline.
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
        session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

        runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
        content = types.Content(
            role="user",
            parts=[types.Part(text=f"Analyse l'action {metrics.get('ticker', 'N/A')} a partir des donnees deja calculees par l'application.")],
        )

        final_answer = None
        events = runner.run(user_id=user_id, session_id=session_id, new_message=content)
        for event in events:
            if event.is_final_response() and event.content:
                final_answer = _text_from_parts(event.content.parts)

        if not final_answer:
            return None, "Le pipeline multi-agents n'a pas retourne de reponse finale."
        return final_answer, None
    except Exception as exc:
        return None, f"Echec de l'analyse multi-agents: {exc}"
