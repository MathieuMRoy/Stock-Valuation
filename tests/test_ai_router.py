import unittest

from ai.models import AgentResponse, ChatSessionContext, StockContext
from ai.router import SpecialistRouter, normalize_intent_text


class RouterIntentTests(unittest.TestCase):
    def test_normalize_intent_text_removes_accents(self):
        self.assertEqual(normalize_intent_text("actualité récente d'Apple"), "actualite recente d'apple")

    def test_routes_meta_questions_first(self):
        router = SpecialistRouter({"meta": lambda prompt: ("meta", None)})
        self.assertEqual(router.resolve("Qui sont les agents disponibles ?"), "meta")

    def test_routes_peer_questions_to_peer_handler(self):
        router = SpecialistRouter({"peer": lambda prompt: ("peer", None)})
        self.assertEqual(router.resolve("compare apple avec ses pairs"), "peer")

    def test_routes_direct_stock_comparisons(self):
        router = SpecialistRouter({"comparison": lambda prompt: ("comparison", None)})
        self.assertEqual(router.resolve("compare AAPL avec DUOL"), "comparison")

    def test_routes_recent_news_questions(self):
        router = SpecialistRouter({"news": lambda prompt: ("news", None)})
        self.assertEqual(router.resolve("Donne moi l'actualité récente d'Apple"), "news")

    def test_routes_market_signal_questions(self):
        router = SpecialistRouter({"market_signal": lambda prompt: ("signals", None)})
        self.assertEqual(router.resolve("Que disent les analystes sur Apple ?"), "market_signal")


class RouterHandlerTests(unittest.TestCase):
    def test_route_returns_agent_response_wrapper(self):
        router = SpecialistRouter({"fundamental": lambda prompt: ("reponse", {"authors": ["fundamental_agent"]})})
        response = router.route("cette action semble-t-elle chère ?")
        self.assertIsInstance(response, AgentResponse)
        self.assertEqual(response.text, "reponse")
        self.assertEqual(response.trace.to_dict()["authors"], ["fundamental_agent"])


class ChatModelRoundTripTests(unittest.TestCase):
    def test_chat_session_context_round_trip(self):
        payload = {
            "runner": object(),
            "session_service": object(),
            "user_id": "user-1",
            "session_id": "session-1",
            "current_ticker": "AAPL",
            "investor_objective": {"label": "Equilibre"},
            "fallback_context": {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "company": {"ticker": "AAPL"},
                "peer": {"benchmark_name": "Big Tech"},
                "technical": {"technical_score_out_of_10": 8.0},
                "sources": {"quote_page": {"url": "https://example.com"}},
            },
        }
        context = ChatSessionContext.from_mapping(payload)
        self.assertIsInstance(context.stock_context, StockContext)
        self.assertEqual(context.stock_context.company["ticker"], "AAPL")
        round_trip = context.to_mapping()
        self.assertEqual(round_trip["fallback_context"]["peer"]["benchmark_name"], "Big Tech")


if __name__ == "__main__":
    unittest.main()
