"""
StockAnalysis.com Fetcher - Get extended historical financials via scraping
"""
import requests
import pandas as pd
import io
import streamlit as st

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

@st.cache_data(ttl=3600*24) # Cache for 24 hours as historical data doesn't change often
def get_extended_history(ticker: str, version: int = 1) -> dict:
    """
    Fetch extended historical financials from StockAnalysis.com.
    Returns a dictionary with 'inc_a', 'inc_q', 'bs_a', 'bs_q', 'cf_a', 'cf_q'.
    """
    ticker_clean = ticker.lower().replace(".", "-") # Handle TSX tickers like RY.TO -> ry-to? (To convert properly later if needed)
    
    # StockAnalysis format handling for different exchanges might be needed. 
    # For now, assumes US stocks primarily or manual match.
    
    base_url = f"https://stockanalysis.com/stocks/{ticker_clean}/financials/"
    
    out = {
        "inc_a": pd.DataFrame(), "inc_q": pd.DataFrame(),
        "bs_a": pd.DataFrame(), "bs_q": pd.DataFrame(),
        "cf_a": pd.DataFrame(), "cf_q": pd.DataFrame(),
        "error": None
    }
    
    # 1. Fetch Annual
    try:
        r = requests.get(base_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            dfs = pd.read_html(io.StringIO(r.text))
            if len(dfs) > 0:
                df = dfs[0]
                
                # Handle MultiIndex columns (flatten)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Clean Dataframe
                # First column is usually metrics. Others are years.
                # Rename first col to "Metric"
                cols = list(df.columns)
                cols[0] = "Metric"
                df.columns = cols
                
                # Clean Metric Name (remove non-breaking spaces \xa0)
                if df["Metric"].dtype == object:
                    df["Metric"] = df["Metric"].astype(str).str.replace(r'\xa0', ' ', regex=True).str.strip()
                
                # Set Metric as Index
                df = df.set_index("Metric")
                
                # Convert columns (Years) to proper format if needed
                # StockAnalysis years are usually "2024", "2023"...
                # Ensure numeric data
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                out["inc_a"] = df
    except Exception as e:
        out["error"] = f"Annual fetch failed: {str(e)}"

    # 2. Fetch Quarterly Income
    try:
        url_q = f"{base_url}quarterly/"
        r_q = requests.get(url_q, headers=HEADERS, timeout=10)
        if r_q.status_code == 200:
            dfs_q = pd.read_html(io.StringIO(r_q.text))
            if len(dfs_q) > 0:
                df_q = dfs_q[0]
                if isinstance(df_q.columns, pd.MultiIndex): df_q.columns = df_q.columns.get_level_values(0)
                cols_q = list(df_q.columns)
                cols_q[0] = "Metric"
                df_q.columns = cols_q
                if df_q["Metric"].dtype == object:
                    df_q["Metric"] = df_q["Metric"].astype(str).str.replace(r'\xa0', ' ', regex=True).str.strip()
                df_q = df_q.set_index("Metric")
                for col in df_q.columns: df_q[col] = pd.to_numeric(df_q[col], errors="coerce")
                out["inc_q"] = df_q
    except Exception: pass

    # Helper function for other sheets
    def fetch_sheet(url_suffix):
        try:
            url = f"https://stockanalysis.com/stocks/{ticker_clean}/financials/{url_suffix}"
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                dfs = pd.read_html(io.StringIO(r.text))
                if len(dfs) > 0:
                    df = dfs[0]
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    cols = list(df.columns)
                    cols[0] = "Metric"
                    df.columns = cols
                    if df["Metric"].dtype == object:
                        df["Metric"] = df["Metric"].astype(str).str.replace(r'\xa0', ' ', regex=True).str.strip()
                    df = df.set_index("Metric")
                    for col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
        except: pass
        return pd.DataFrame()

    # 3. Fetch Balance Sheet (Annual & Quarterly)
    out["bs_a"] = fetch_sheet("balance-sheet/")
    out["bs_q"] = fetch_sheet("balance-sheet/quarterly/")

    # 4. Fetch Cash Flow (Annual & Quarterly)
    out["cf_a"] = fetch_sheet("cash-flow-statement/")
    out["cf_q"] = fetch_sheet("cash-flow-statement/quarterly/")

    return out

def get_available_metrics_sa(df: pd.DataFrame) -> list:
    """Get list of available metrics from the dataframe index"""
    if df is None or df.empty:
        return []
    return list(df.index)
