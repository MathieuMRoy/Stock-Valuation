"""
Screener UI - AI-powered stock screener interface
"""
import pandas as pd
import streamlit as st

from fetchers import get_financial_data_secure, get_item_safe, get_ttm_or_latest, finviz_fetch_tickers, FINVIZ_SECTORS
from valuation import calculate_valuation


def _mc_ok(data: dict, min_mc: float) -> bool:
    """Check if market cap meets minimum requirement"""
    mc = data.get("market_cap")
    if mc is None:
        return False
    try:
        return float(mc) >= float(min_mc)
    except:
        return False


def render_screener():
    """Render the AI Screener mode UI."""
    st.subheader("🔍 AI Screener (Top Upside)")
    
    min_market_cap = st.number_input("Min Market Cap (USD)", value=1_000_000_000, step=250_000_000)
    max_tickers_per_sector = st.slider("Max tickers per sector", 10, 80, 20, step=5)
    
    colA, colB = st.columns(2)
    scr_gr_fcf = colA.number_input("FCF Growth % (fallback)", value=15.0, step=0.5)
    scr_wacc = colB.number_input("WACC %", value=10.0, step=0.5)

    def intrinsic_dcf_quick(ticker: str) -> dict:
        """Quick DCF calculation for screening"""
        d = get_financial_data_secure(ticker)
        p = float(d.get("price", 0) or 0)
        s = float(d.get("shares_info", 0) or 0)
        if p <= 0 or s <= 0:
            return {"ok": False}
        if not _mc_ok(d, min_market_cap):
            return {"ok": False}
        
        rev = get_ttm_or_latest(d["inc"], ["Revenue"])
        cf = get_ttm_or_latest(d["cf"], ["OperatingCashFlow"])
        cap = abs(get_item_safe(d["cf"], ["CapitalExpenditure"]))
        fcf = cf - cap
        c = get_item_safe(d["bs"], ["Cash"])
        db = get_item_safe(d["bs"], ["LongTermDebt"])
        
        g = float(d.get("rev_growth", 0) or 0)
        if g <= 0:
            g = scr_gr_fcf / 100.0
        
        dcf, _, _ = calculate_valuation(0, g, 0, scr_wacc/100, 0, 0, rev, fcf, 0, c, db, s)
        if dcf <= 0:
            return {"ok": False}
        return {
            "ticker": ticker,
            "price": p,
            "intrinsic": dcf,
            "upside": (dcf/p - 1)*100,
            "ok": True,
            "bucket": d.get("sector")
        }

    if st.button("🚀 Run Screener"):
        res = []
        bar = st.progress(0)
        st_text = st.empty()
        sectors = FINVIZ_SECTORS
        total = len(sectors) * 2
        step = 0
        
        for (sec_name, sec_code) in sectors:
            for geo, geo_code in [("USA", "geo_usa"), ("Canada", "geo_canada")]:
                step += 1
                st_text.write(f"Scanning {sec_name} ({geo})...")
                ts = finviz_fetch_tickers(sec_code, geo_code, max_tickers_per_sector)
                for t in ts:
                    r = intrinsic_dcf_quick(t)
                    if r["ok"]:
                        r["bucket"] = f"{sec_name} ({geo})"
                        res.append(r)
                bar.progress(min(step/total, 1.0))
        
        bar.progress(1.0)
        st_text.write("Done!")
        
        if res:
            df = pd.DataFrame(res).sort_values("upside", ascending=False)
            st.dataframe(
                df[["bucket", "ticker", "price", "intrinsic", "upside"]]
                .style.format({"price": "{:.2f}", "intrinsic": "{:.2f}", "upside": "{:.1f}%"})
            )
        else:
            st.error("No results.")
