import unittest

from ai.comparison_utils import comparison_focus_from_message, describe_matchup
from ai.response_utils import compose_professional_response, dedupe_repetitive_sentences


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
        self.assertIn("intersectoriel", text)
        self.assertIn("these de croissance", text)


if __name__ == "__main__":
    unittest.main()
