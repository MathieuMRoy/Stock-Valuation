import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

from services import (
    blend_reasonable_intrinsic_values,
    build_data_quality_report,
    build_investor_objective_snapshot,
    combined_risk_label,
    extract_next_earnings,
    get_sector_profile,
    is_financial_company,
    is_reasonable_intrinsic_value,
    markdown_report_to_pdf_bytes,
    market_risk_label,
    market_cap_ok,
    profile_label,
    quick_intrinsic_dcf,
    resolve_share_count,
    risk_label,
    run_screener,
    upside_pct,
    valuation_label,
)
from technical import add_indicators, summarize_technical_setup

_DCF_SPEC = spec_from_file_location("dcf_module", Path(__file__).resolve().parents[1] / "valuation" / "dcf.py")
_DCF_MODULE = module_from_spec(_DCF_SPEC)
assert _DCF_SPEC and _DCF_SPEC.loader
_DCF_SPEC.loader.exec_module(_DCF_MODULE)
calculate_valuation = _DCF_MODULE.calculate_valuation


class AnalyzerServiceTests(unittest.TestCase):
    def test_extract_next_earnings_from_calendar_dict(self):
        calendar = {"Earnings Date": ["2030-05-10", "2030-08-10"]}
        self.assertTrue(extract_next_earnings(calendar).startswith("2030-05-10"))

    def test_is_financial_company_detects_banks(self):
        self.assertTrue(is_financial_company("Financial Services", "Canadian Banks"))
        self.assertFalse(is_financial_company("Technology", "Big Tech / GAFAM"))

    def test_sector_profile_uses_energy_specific_caveats(self):
        profile = get_sector_profile("Energy", "Energy & Oil Majors")

        self.assertEqual(profile.key, "energy")
        self.assertIn("Commodity", profile.business_model.title())
        self.assertIn("cycle", profile.valuation_caveat.lower())

    def test_labels_cover_core_paths(self):
        self.assertEqual(risk_label(3.5, 7), "Lower financial risk")
        self.assertEqual(risk_label(None, 7, is_financial=True), "Stable bank profile")
        self.assertEqual(valuation_label(25), "Undervalued setup")
        self.assertEqual(valuation_label(-25), "Rich valuation")
        self.assertEqual(profile_label(0.25, 0.18, 8.0, 6.0), "Growth-oriented profile")

    def test_combined_risk_blocks_lower_label_after_big_drawdown(self):
        technical = {
            "drawdown_from_52w_high_pct": -80.0,
            "volatility_20d_pct": 45.0,
            "technical_score_out_of_10": 4.0,
            "momentum_3m_pct": -55.0,
        }

        self.assertEqual(market_risk_label(technical), "Severe market risk")
        self.assertEqual(
            combined_risk_label(4.2, 8, technical=technical),
            "Severe market risk",
        )

    def test_markdown_report_exports_valid_pdf_bytes(self):
        pdf_bytes = markdown_report_to_pdf_bytes(
            "\n".join(
                [
                    "# DUOL stock analysis",
                    "",
                    "## Snapshot",
                    "- Current price: 93.53 $",
                    "- Risk: Severe market risk",
                ]
            ),
            document_title="DUOL valuation report",
        )

        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 1_000)

    def test_objective_snapshot_falls_back_to_balanced(self):
        snapshot = build_investor_objective_snapshot("unknown")
        self.assertEqual(snapshot["key"], "balanced")
        self.assertEqual(snapshot["label"], "Equilibre")

    def test_resolve_share_count_estimates_from_market_cap_when_shares_missing(self):
        shares, market_cap, unavailable, estimated, manual = resolve_share_count(
            price=25.0,
            raw_shares=0.0,
            reported_market_cap=2_500_000_000.0,
        )

        self.assertEqual(shares, 100_000_000.0)
        self.assertEqual(market_cap, 2_500_000_000.0)
        self.assertFalse(unavailable)
        self.assertTrue(estimated)
        self.assertFalse(manual)

    def test_data_quality_report_marks_missing_core_inputs_critical(self):
        report = build_data_quality_report(
            price=0,
            shares=0,
            market_cap=0,
            revenue_ttm=0,
            eps_ttm=0,
            fcf_ttm=0,
            balance_sheet=pd.DataFrame(),
            income_statement=pd.DataFrame(),
            cash_flow=pd.DataFrame(),
        )

        self.assertEqual(report.status, "critical")
        self.assertIn("Current price is missing.", report.blockers)

    def test_valuation_guardrails_filter_unrealistic_intrinsic_values(self):
        self.assertTrue(is_reasonable_intrinsic_value(130, 100))
        self.assertFalse(is_reasonable_intrinsic_value(1000, 100))
        self.assertAlmostEqual(upside_pct(130, 100), 30.0)

        blended, warnings = blend_reasonable_intrinsic_values({"DCF": 1000, "P/S": 120, "P/E": 0}, 100)
        self.assertEqual(blended, 120)
        self.assertEqual(len(warnings), 2)

    def test_dcf_guardrail_blocks_terminal_spread_explosion(self):
        dcf, sales, earnings = calculate_valuation(
            0.1,
            0.1,
            0.1,
            0.031,
            5,
            20,
            1_000_000_000,
            100_000_000,
            1.0,
            0,
            0,
            100_000_000,
        )

        self.assertEqual(dcf, 0.0)
        self.assertGreater(sales, 0.0)
        self.assertGreater(earnings, 0.0)


class ScreenerEngineTests(unittest.TestCase):
    @staticmethod
    def _base_screening_payload(**overrides):
        columns = ["Q1", "Q2", "Q3", "Q4"]
        payload = {
            "price": 10.0,
            "shares_info": 100.0,
            "market_cap": 5_000_000_000,
            "sector": "Technology",
            "industry": "Software",
            "quote_type": "equity",
            "long_name": "Test Corp",
            "inc": pd.DataFrame([[50.0, 50.0, 50.0, 50.0]], index=["Revenue"], columns=columns),
            "cf": pd.DataFrame(
                [[20.0, 20.0, 20.0, 20.0], [-5.0, -5.0, -5.0, -5.0]],
                index=["OperatingCashFlow", "CapitalExpenditure"],
                columns=columns,
            ),
            "bs": pd.DataFrame([[30.0], [0.0]], index=["Cash", "TotalDebt"], columns=["Latest"]),
            "rev_growth": 0.12,
        }
        payload.update(overrides)
        return payload

    def test_market_cap_ok_handles_missing_values(self):
        self.assertTrue(market_cap_ok({"market_cap": 2_000_000_000}, 1_000_000_000))
        self.assertFalse(market_cap_ok({"market_cap": None}, 1_000_000_000))

    def test_quick_intrinsic_dcf_skips_financials_and_funds(self):
        def fake_valuation_calculator(*args):
            return (30.0, 0.0, 0.0)

        financial_candidate = quick_intrinsic_dcf(
            "BANK",
            minimum_market_cap=1_000_000_000,
            fallback_fcf_growth_pct=15.0,
            wacc_pct=10.0,
            data_fetcher=lambda ticker, cache_version: self._base_screening_payload(
                sector="Financial Services",
                industry="Banks",
            ),
            valuation_calculator=fake_valuation_calculator,
        )
        fund_candidate = quick_intrinsic_dcf(
            "FUND",
            minimum_market_cap=1_000_000_000,
            fallback_fcf_growth_pct=15.0,
            wacc_pct=10.0,
            data_fetcher=lambda ticker, cache_version: self._base_screening_payload(
                quote_type="etf",
                long_name="Test Income Fund",
            ),
            valuation_calculator=fake_valuation_calculator,
        )

        self.assertIsNone(financial_candidate)
        self.assertIsNone(fund_candidate)

    def test_quick_intrinsic_dcf_filters_absurd_upside(self):
        candidate = quick_intrinsic_dcf(
            "HYPE",
            minimum_market_cap=1_000_000_000,
            fallback_fcf_growth_pct=15.0,
            wacc_pct=10.0,
            data_fetcher=lambda ticker, cache_version: self._base_screening_payload(),
            valuation_calculator=lambda *args: (60.0, 0.0, 0.0),
        )

        self.assertIsNone(candidate)

    def test_quick_intrinsic_dcf_skips_critical_data_quality(self):
        candidate = quick_intrinsic_dcf(
            "BROKEN",
            minimum_market_cap=1_000_000_000,
            fallback_fcf_growth_pct=15.0,
            wacc_pct=10.0,
            data_fetcher=lambda ticker, cache_version: self._base_screening_payload(
                data_quality="critical",
                data_quality_reasons=["Current price missing."],
            ),
            valuation_calculator=lambda *args: (25.0, 0.0, 0.0),
        )

        self.assertIsNone(candidate)

    def test_run_screener_uses_engine_not_ui_loop(self):
        def fake_ticker_fetcher(sector_code, geography_code, max_tickers):
            self.assertEqual(max_tickers, 5)
            return ["AAA", "BBB"]

        def fake_data_fetcher(ticker, cache_version):
            if ticker == "BBB":
                return self._base_screening_payload(price=0, shares_info=0, market_cap=0)

            return self._base_screening_payload()

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


class TechnicalSummaryTests(unittest.TestCase):
    def test_summarize_technical_setup_adds_actionable_metrics(self):
        dates = pd.date_range("2025-01-01", periods=220, freq="B")
        close = pd.Series(range(100, 320), dtype="float64")
        price_frame = pd.DataFrame(
            {
                "Date": dates,
                "Open": close - 1,
                "High": close + 2,
                "Low": close - 2,
                "Close": close,
                "Volume": [1_000_000 + i * 1000 for i in range(220)],
            }
        )

        technical_frame = add_indicators(price_frame)
        summary = summarize_technical_setup(technical_frame, {"score": 7, "is_bull_flag": True})

        self.assertEqual(summary["data_quality"], "rich")
        self.assertGreater(summary["technical_score_out_of_10"], 7)
        self.assertTrue(summary["bull_flag_detected"])
        self.assertIn("support_60d", summary)
        self.assertIn("volatility_20d_pct", summary)


if __name__ == "__main__":
    unittest.main()
