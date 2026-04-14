import unittest

import pandas as pd

from services import (
    build_investor_objective_snapshot,
    extract_next_earnings,
    is_financial_company,
    market_cap_ok,
    profile_label,
    risk_label,
    run_screener,
    valuation_label,
)


class AnalyzerServiceTests(unittest.TestCase):
    def test_extract_next_earnings_from_calendar_dict(self):
        calendar = {"Earnings Date": ["2030-05-10", "2030-08-10"]}
        self.assertTrue(extract_next_earnings(calendar).startswith("2030-05-10"))

    def test_is_financial_company_detects_banks(self):
        self.assertTrue(is_financial_company("Financial Services", "Canadian Banks"))
        self.assertFalse(is_financial_company("Technology", "Big Tech / GAFAM"))

    def test_labels_cover_core_paths(self):
        self.assertEqual(risk_label(3.5, 7), "Lower risk")
        self.assertEqual(risk_label(None, 7, is_financial=True), "Stable bank profile")
        self.assertEqual(valuation_label(25), "Undervalued setup")
        self.assertEqual(valuation_label(-25), "Rich valuation")
        self.assertEqual(profile_label(0.25, 0.18, 8.0, 6.0), "Growth-oriented profile")

    def test_objective_snapshot_falls_back_to_balanced(self):
        snapshot = build_investor_objective_snapshot("unknown")
        self.assertEqual(snapshot["key"], "balanced")
        self.assertEqual(snapshot["label"], "Equilibre")


class ScreenerEngineTests(unittest.TestCase):
    def test_market_cap_ok_handles_missing_values(self):
        self.assertTrue(market_cap_ok({"market_cap": 2_000_000_000}, 1_000_000_000))
        self.assertFalse(market_cap_ok({"market_cap": None}, 1_000_000_000))

    def test_run_screener_uses_engine_not_ui_loop(self):
        def fake_ticker_fetcher(sector_code, geography_code, max_tickers):
            self.assertEqual(max_tickers, 5)
            return ["AAA", "BBB"]

        def fake_data_fetcher(ticker, cache_version):
            if ticker == "BBB":
                return {
                    "price": 0,
                    "shares_info": 0,
                    "market_cap": 0,
                    "sector": "Technology",
                    "inc": pd.DataFrame(),
                    "cf": pd.DataFrame(),
                    "bs": pd.DataFrame(),
                    "rev_growth": 0,
                }

            columns = ["Q1", "Q2", "Q3", "Q4"]
            return {
                "price": 10.0,
                "shares_info": 100.0,
                "market_cap": 5_000_000_000,
                "sector": "Technology",
                "inc": pd.DataFrame([[50.0, 50.0, 50.0, 50.0]], index=["Revenue"], columns=columns),
                "cf": pd.DataFrame(
                    [[20.0, 20.0, 20.0, 20.0], [-5.0, -5.0, -5.0, -5.0]],
                    index=["OperatingCashFlow", "CapitalExpenditure"],
                    columns=columns,
                ),
                "bs": pd.DataFrame([[30.0], [0.0]], index=["Cash", "TotalDebt"], columns=["Latest"]),
                "rev_growth": 0.12,
            }

        def fake_valuation_calculator(*args):
            return (25.0, 0.0, 0.0)

        progress_calls = []

        results = run_screener(
            minimum_market_cap=1_000_000_000,
            max_tickers_per_sector=5,
            fallback_fcf_growth_pct=15.0,
            wacc_pct=10.0,
            sectors=[("Technology", "sec_technology")],
            progress_callback=lambda step, total, sector_name, geography_name: progress_calls.append(
                (step, total, sector_name, geography_name)
            ),
            ticker_fetcher=fake_ticker_fetcher,
            data_fetcher=fake_data_fetcher,
            valuation_calculator=fake_valuation_calculator,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].ticker, "AAA")
        self.assertEqual(results[0].bucket, "Technology (USA)")
        self.assertEqual(progress_calls[0], (1, 2, "Technology", "USA"))


if __name__ == "__main__":
    unittest.main()
