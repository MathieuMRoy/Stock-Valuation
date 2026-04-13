from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class StockContext:
    ticker: str | None = None
    company_name: str | None = None
    company: dict[str, Any] = field(default_factory=dict)
    peer: dict[str, Any] = field(default_factory=dict)
    technical: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fallback_context(cls, payload: dict[str, Any] | None) -> "StockContext":
        payload = payload or {}
        return cls(
            ticker=payload.get("ticker"),
            company_name=payload.get("company_name"),
            company=dict(payload.get("company") or {}),
            peer=dict(payload.get("peer") or {}),
            technical=dict(payload.get("technical") or {}),
            sources=dict(payload.get("sources") or {}),
        )

    def to_fallback_context(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "company": dict(self.company),
            "peer": dict(self.peer),
            "technical": dict(self.technical),
            "sources": dict(self.sources),
        }


@dataclass(slots=True)
class AgentTracePayload:
    authors: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    specialist_agents: list[str] = field(default_factory=list)
    specialist_labels: list[str] = field(default_factory=list)
    lead_agent: str | None = None
    lead_agent_label: str | None = None
    final_author: str | None = None
    final_author_label: str | None = None
    used_supervisor: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AgentTracePayload | None":
        if not payload:
            return None
        return cls(
            authors=list(payload.get("authors") or []),
            labels=list(payload.get("labels") or []),
            specialist_agents=list(payload.get("specialist_agents") or []),
            specialist_labels=list(payload.get("specialist_labels") or []),
            lead_agent=payload.get("lead_agent"),
            lead_agent_label=payload.get("lead_agent_label"),
            final_author=payload.get("final_author"),
            final_author_label=payload.get("final_author_label"),
            used_supervisor=bool(payload.get("used_supervisor", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentResponse:
    text: str | None = None
    error: str | None = None
    trace: AgentTracePayload | None = None

    @classmethod
    def from_handler_result(
        cls,
        result: tuple[str | None, dict[str, Any] | None] | None,
    ) -> "AgentResponse | None":
        if result is None:
            return None
        text, trace = result
        return cls(text=text, trace=AgentTracePayload.from_dict(trace))

    def as_specialist_tuple(self) -> tuple[str | None, dict[str, Any] | None]:
        return self.text, self.trace.to_dict() if self.trace else None

    def as_chat_tuple(self) -> tuple[str | None, str | None, dict[str, Any] | None]:
        return self.text, self.error, self.trace.to_dict() if self.trace else None


@dataclass(slots=True)
class ChatSessionContext:
    runner: Any
    session_service: Any
    user_id: str
    session_id: str
    current_ticker: str | None
    investor_objective: dict[str, Any]
    stock_context: StockContext

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ChatSessionContext":
        return cls(
            runner=payload["runner"],
            session_service=payload["session_service"],
            user_id=payload["user_id"],
            session_id=payload["session_id"],
            current_ticker=payload.get("current_ticker"),
            investor_objective=dict(payload.get("investor_objective") or {}),
            stock_context=StockContext.from_fallback_context(
                payload.get("stock_context") or payload.get("fallback_context")
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        fallback_context = self.stock_context.to_fallback_context()
        return {
            "runner": self.runner,
            "session_service": self.session_service,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "current_ticker": self.current_ticker,
            "investor_objective": dict(self.investor_objective),
            "stock_context": fallback_context,
            "fallback_context": fallback_context,
        }
