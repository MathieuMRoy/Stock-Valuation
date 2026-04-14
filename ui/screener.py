"""Screener UI - AI-powered stock screener interface."""

import pandas as pd
import streamlit as st

from services import run_screener


def render_screener():
    """Render the AI Screener mode UI."""
    st.subheader("AI Screener (Top Upside)")

    min_market_cap = st.number_input("Min Market Cap (USD)", value=1_000_000_000, step=250_000_000)
    max_tickers_per_sector = st.slider("Max tickers per sector", 10, 80, 20, step=5)

    col_growth, col_wacc = st.columns(2)
    fallback_fcf_growth_pct = col_growth.number_input("FCF Growth % (fallback)", value=15.0, step=0.5)
    wacc_pct = col_wacc.number_input("WACC %", value=10.0, step=0.5)

    if st.button("Run Screener"):
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def _on_progress(step: int, total: int, sector_name: str, geography_name: str):
            progress_bar.progress(min(step / total, 1.0))
            progress_text.write(f"Scanning {sector_name} ({geography_name})...")

        results = run_screener(
            minimum_market_cap=min_market_cap,
            max_tickers_per_sector=max_tickers_per_sector,
            fallback_fcf_growth_pct=fallback_fcf_growth_pct,
            wacc_pct=wacc_pct,
            progress_callback=_on_progress,
        )

        progress_bar.progress(1.0)
        progress_text.write("Scan complete.")

        if results:
            dataframe = pd.DataFrame([row.as_row() for row in results])
            st.dataframe(
                dataframe[["bucket", "ticker", "price", "intrinsic", "upside"]].style.format(
                    {"price": "{:.2f}", "intrinsic": "{:.2f}", "upside": "{:.1f}%"}
                ),
                use_container_width=True,
            )
        else:
            st.error("No candidates met the current filters.")
