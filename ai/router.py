from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Callable, Mapping, Sequence

from .models import AgentResponse


HandlerResult = tuple[str | None, dict | None]
SpecialistHandler = Callable[[str], HandlerResult]


def normalize_intent_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def looks_like_comparison_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = ["compare", "compar", "versus", "vs", "avec", "contre", "entre "]
    return any(marker in message for marker in markers)


def looks_like_agent_meta_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = ["agent", "agents", "disponible", "disponibles", "capable", "capables", "peux tu faire", "que fais tu", "comment tu fonctionnes"]
    return any(marker in message for marker in markers)


def looks_like_peer_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = ["pair", "pairs", "peers", "comparables", "benchmark", "secteur", "sector"]
    return any(marker in message for marker in markers)


def looks_like_news_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = [
        "actualite",
        "news",
        "nouvelles",
        "headline",
        "catalyst",
        "catalyseur",
        "earnings",
        "publication",
        "resultat",
        "recent",
        "recente",
        "dernier",
        "update",
    ]
    return any(marker in message for marker in markers)


def looks_like_market_signal_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = ["analyst", "analyste", "target", "price target", "insider", "short interest", "sentiment", "rating"]
    return any(marker in message for marker in markers)


def looks_like_filing_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = ["sec", "filing", "10-k", "10-q", "officiel", "official", "reported", "reporte", "etats financiers"]
    return any(marker in message for marker in markers)


def looks_like_technical_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = ["technique", "technical", "chart", "momentum", "rsi", "bull flag", "setup", "trend", "tendance"]
    return any(marker in message for marker in markers)


def looks_like_risk_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = ["risque", "risk", "safe", "defensif", "profil", "downside", "reward", "robuste", "resilient"]
    return any(marker in message for marker in markers)


def looks_like_fundamental_request(user_message: str) -> bool:
    message = normalize_intent_text(user_message)
    markers = ["valorisation", "valuation", "cheap", "chere", "cher", "pe", "p/e", "ps", "p/s", "fcf", "growth", "croissance"]
    return any(marker in message for marker in markers)


@dataclass(frozen=True, slots=True)
class RouteRule:
    name: str
    predicate: Callable[[str], bool]


DEFAULT_ROUTE_RULES: tuple[RouteRule, ...] = (
    RouteRule("meta", looks_like_agent_meta_request),
    RouteRule("peer", looks_like_peer_request),
    RouteRule("comparison", looks_like_comparison_request),
    RouteRule("news", looks_like_news_request),
    RouteRule("market_signal", looks_like_market_signal_request),
    RouteRule("filings", looks_like_filing_request),
    RouteRule("technical", looks_like_technical_request),
    RouteRule("risk", looks_like_risk_request),
    RouteRule("fundamental", looks_like_fundamental_request),
)


class SpecialistRouter:
    def __init__(
        self,
        handlers: Mapping[str, SpecialistHandler],
        rules: Sequence[RouteRule] | None = None,
    ) -> None:
        self._handlers = dict(handlers)
        self._rules = tuple(rules or DEFAULT_ROUTE_RULES)

    def resolve(self, user_message: str) -> str | None:
        for rule in self._rules:
            if rule.predicate(user_message):
                return rule.name
        return None

    def route(self, user_message: str) -> AgentResponse | None:
        route_name = self.resolve(user_message)
        if not route_name:
            return None
        handler = self._handlers.get(route_name)
        if not handler:
            return None
        return AgentResponse.from_handler_result(handler(user_message))
