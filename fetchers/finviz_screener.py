"""
Finviz Screener - Fetch tickers from Finviz screener
"""
import requests
import streamlit as st


FINVIZ_SECTORS = [
    ("Technology", "sec_technology"),
    ("Healthcare", "sec_healthcare"),
    ("Financial", "sec_financial"),
    ("Energy", "sec_energy"),
    ("Consumer Cyclical", "sec_consumercyclical"),
    ("Industrials", "sec_industrials"),
]


@st.cache_data(ttl=3600)
def finviz_fetch_tickers(sector_filter: str, geo_filter: str, max_tickers: int = 60) -> list:
    """
    Fetch ticker symbols from Finviz screener.
    
    Args:
        sector_filter: Sector filter code (e.g., 'sec_technology')
        geo_filter: Geography filter (e.g., 'geo_usa')
        max_tickers: Maximum number of tickers to return
    
    Returns:
        List of ticker symbols
    """
    base = "https://finviz.com/screener.ashx"
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []
    r = 1
    f = f"{sector_filter},{geo_filter},cap_smallover"
    
    while len(tickers) < max_tickers and r <= 401:
        params = {"v": "111", "f": f, "r": str(r)}
        try:
            resp = requests.get(base, params=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                break
            html = resp.text
            found = []
            parts = html.split("quote.ashx?t=")
            for p in parts[1:]:
                t = ""
                for ch in p:
                    if ch.isalnum() or ch in ".-":
                        t += ch
                    else:
                        break
                if t and t not in found:
                    found.append(t)
            if not found:
                break
            for t in found:
                if t not in tickers:
                    tickers.append(t)
            r += 20
        except:
            break
    
    return tickers[:max_tickers]
