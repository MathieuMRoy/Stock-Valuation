import unittest

from ai.comparison_utils import (
    business_model_sentence,
    comparison_focus_from_message,
    describe_matchup,
    describe_valuation_tradeoff,
    objective_conclusion,
)
from ai.response_utils import compose_professional_response, dedupe_repetitive_sentences
from ai.technical_utils import compose_technical_response, technical_snapshot_is_sparse


class ResponseUtilsTests(unittest.TestCase):
    def test_dedupe_repetitive_sentences_removes_near_duplicates(self):
        text = (
            "**Nature du match-up**\n"
            "MSFT est une grande techno mature. MSFT est une grande techno mature. "
            "DUOL est une plateforme numerique plus oriente croissance."
        )
        cleaned = dedupe_repetitive_sentences(text)
        self.assertEqual(cleaned.count("MSFT est une grande techno mature."), 1)

    def test_compose_professional_response_adds_sources(self):
        response = compose_professional_response(
            "Ouverture",
            [("Section", "Contenu")],
            source_refs={"quote_page": {"label": "AAPL quote page", "url": "https://example.com/aapl"}},
        )
        self.assertIn("**Sources**", response)
        self.assertIn("[AAPL quote page](https://example.com/aapl)", response)


class ComparisonUtilsTests(unittest.TestCase):
    def test_comparison_focus_detects_solidite_angle(self):
        self.assertEqual(comparison_focus_from_message("laquelle est la plus solide niveau bilan ?"), "solidite")

    def test_describe_matchup_mentions_cross_sector_tradeoff(self):
        text = describe_matchup(
            {"ticker": "MSFT", "sector_name": "Technology", "benchmark_name": "Big Tech"},
            {"ticker": "DUOL", "sector_name": "Consumer", "benchmark_name": "Consumer Apps & Platforms"},
            True,
        )
        self.assertIn("intersectorielle", text)
        self.assertIn("croissance", text)

    def test_business_model_sentence_avoids_default_cycle_language(self):
        text = business_model_sentence(
            {
                "ticker": "MSFT",
                "company_name": "Microsoft",
                "sector_name": "Default",
                "benchmark_name": "Big Tech / GAFAM",
                "business_model_hint": "company exposed to the Default cycle",
            }
        )
        self.assertNotIn("Default cycle", text)
        self.assertIn("mega-cap technologique", text)

    def test_objective_conclusion_is_more_actionable_for_balanced_angle(self):
        text = objective_conclusion(
            "Equilibre",
            {"ticker": "MSFT", "growth_score": 7.0, "valuation_score": 5.5, "health_score": 8.0},
            {"ticker": "DUOL", "growth_score": 8.0, "valuation_score": 7.0, "health_score": 6.0},
            {"technical_score_out_of_10": 7.0},
            {"technical_score_out_of_10": 6.0},
        )
        self.assertIn("qualite et la resilience", text)
        self.assertIn("davantage de risque pour plus de croissance", text)

    def test_describe_valuation_tradeoff_handles_missing_pe_cleanly(self):
        text = describe_valuation_tradeoff(
            {"ticker": "MDA.TO", "pe_ratio": 57.15, "ps_ratio": 3.80},
            {"ticker": "RKLB", "pe_ratio": None, "ps_ratio": 81.26},
        )
        self.assertIn("P/E n'est pas vraiment exploitable pour RKLB", text)
        self.assertIn("MDA.TO", text)
        self.assertNotIn("RKLB semble moins cher sur le P/E", text)


class TechnicalUtilsTests(unittest.TestCase):
    def test_sparse_technical_snapshot_does_not_become_fake_bearish_signal(self):
        response = compose_technical_response(
            "DUOL",
            {"technical_score_out_of_10": 0.0, "bull_flag_detected": False},
        )

        self.assertTrue(technical_snapshot_is_sparse({"technical_score_out_of_10": 0.0}))
        self.assertIn("pas assez de donnees techniques fiables", response)
        self.assertNotIn("0.0/10", response)
        self.assertNotIn("reste fragile", response)

    def test_rich_technical_snapshot_mentions_trend_momentum_and_rsi(self):
        response = compose_technical_response(
            "DUOL",
            {
                "technical_score_out_of_10": 7.5,
                "bull_flag_detected": True,
                "latest_close": 100,
                "last_price_date": "2026-04-24",
                "sma20": 96,
                "sma50": 92,
                "sma200": 80,
                "distance_to_sma20_pct": 4.2,
                "distance_to_sma50_pct": 8.7,
                "distance_to_sma200_pct": 25.0,
                "rsi14": 62,
                "macd_status": "positif",
                "momentum_1m_pct": 12,
                "momentum_3m_pct": 18,
                "momentum_6m_pct": 22,
                "drawdown_from_52w_high_pct": -8,
                "atr_pct": 4.5,
                "volume_vs_20d_pct": 15,
            },
        )

        self.assertIn("setup constructif", response)
        self.assertIn("SMA50", response)
        self.assertIn("RSI 62.0", response)
        self.assertIn("1 mois +12.0%", response)


if __name__ == "__main__":
    unittest.main()
