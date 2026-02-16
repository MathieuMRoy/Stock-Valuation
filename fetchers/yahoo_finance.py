"""
Yahoo Finance data fetcher - Robust functions for fetching financial data
"""
import pandas as pd
import yfinance as yf
import yfinance as yf
import streamlit as st
import requests
import re

from .news import fetch_ir_press_releases


def _safe_df(x) -> pd.DataFrame:
    """Safely convert to DataFrame"""
    if x is None:
        return pd.DataFrame()
    if hasattr(x, "empty"):
        return x if not x.empty else pd.DataFrame()
    return pd.DataFrame()



def _scrape_yahoo_price(ticker: str) -> float:
    """Fallback: Scrape price from Yahoo Finance HTML"""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return 0.0
            
        # Try finding regularMarketPrice in fin-streamer
        # <fin-streamer ... data-field="regularMarketPrice" data-value="412.34">
        # Note: Order of attributes varies, so we check simplistically
        
        # Regex for data-value associated with regularMarketPrice
        # We look for regularMarketPrice then data-value, or vice versa
        
        # Strategy 1: strict fin-streamer pattern
        # This is complex to regex robustly, so let's try a simpler approach finding the value near the keyword
        
        # Common pattern: "regularMarketPrice":{"raw":412.34, ...}
        match_json = re.search(r'"regularMarketPrice":\s*\{\s*"raw":\s*([\d\.]+)', r.text)
        if match_json:
            return float(match_json.group(1))
            
        # Strategy 2: fin-streamer data-value
        # Matches: data-field="regularMarketPrice" ... data-value="412.34"
        # We assume they are somewhat close
        
        # Just searching for the specific value often found in the header
        # <fin-streamer ... data-test="qsp-price" ... value="412.34">
        match_streamer = re.search(r'data-field="regularMarketPrice"[^>]*data-value="([\d\.]+)"', r.text)
        if match_streamer:
            return float(match_streamer.group(1))
            
        return 0.0
    except:
        return 0.0


def _robust_price(stock: yf.Ticker, ticker: str) -> float:
    """Get stock price with multiple fallback methods"""
    try:
        if hasattr(stock, 'fast_info'):
            return float(stock.fast_info['last_price'])
    except:
        pass
    try:
        hist = stock.history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except:
        pass
    try:
        info = stock.info or {}
        return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
    except:
        pass
    return _scrape_yahoo_price(ticker)


def _robust_shares(stock: yf.Ticker) -> float:
    """Get shares outstanding with fallback methods"""
    shares = 0.0
    try:
        shares = float(stock.info.get("sharesOutstanding", 0))
    except:
        pass
    if shares <= 0:
        try:
            if hasattr(stock, 'fast_info'):
                mcap = stock.fast_info['market_cap']
                price = stock.fast_info['last_price']
                if price > 0:
                    shares = mcap / price
        except:
            pass
    return shares


def get_growth_manual(df: pd.DataFrame, keys: list) -> float:
    """Calculate growth rate manually from financial data"""
    try:
        if df is None or df.empty:
            return 0.0
        row = None
        for key in keys:
            matches = df.index[df.index.astype(str).str.contains(key, case=False, regex=True)]
            if not matches.empty:
                row = df.loc[matches[0]]
                break
        if row is None:
            return 0.0
        vals = [v for v in row if isinstance(v, (int, float)) and not pd.isna(v)]
        if len(vals) >= 5 and vals[4] != 0:
            return float((vals[0] - vals[4]) / abs(vals[4]))
        elif len(vals) >= 2 and vals[1] != 0:
            return float((vals[0] - vals[1]) / abs(vals[1]))
        return 0.0
    except:
        return 0.0


def get_item_safe(df: pd.DataFrame, search_terms: list) -> float:
    """Safely get an item from a DataFrame by searching for matching index terms"""
    if df is None or df.empty:
        return 0.0
    for term in search_terms:
        matches = df.index[df.index.astype(str).str.contains(term, case=False, regex=True)]
        if not matches.empty:
            try:
                val = df.loc[matches[0]]
                if isinstance(val, pd.Series):
                    return float(val.iloc[0])
                return float(val)
            except:
                return 0.0
    return 0.0


def get_ttm_or_latest(df: pd.DataFrame, keys_list: list) -> float:
    """Get TTM (trailing twelve months) value or latest available"""
    if df is None or df.empty:
        return 0.0
    for key in keys_list:
        matches = df.index[df.index.astype(str).str.contains(key, case=False, regex=True)]
        if not matches.empty:
            row = df.loc[matches[0]]
            vals = [v for v in row if isinstance(v, (int, float)) and not pd.isna(v)]
            if not vals:
                return 0.0
            if len(vals) >= 4:
                return float(sum(vals[:4]))  # Sum 4 quarters
            if len(vals) == 1:
                return float(vals[0]) * 4  # Estimate annual if only 1 qtr
            return float(vals[0])
    return 0.0


@st.cache_data(ttl=3600)
def get_financial_data_secure(ticker: str) -> dict:
    """
    Fetch comprehensive financial data for a ticker.
    Returns a dictionary with all financial metrics, statements, and metadata.
    """
    out = {
        "bs": pd.DataFrame(), "inc": pd.DataFrame(), "cf": pd.DataFrame(),
        "reco_summary": None, "calendar": None, "target_price": None, "ir_news": [],
        "price": 0.0, "shares_info": 0.0, "sector": "Default",
        "rev_growth": 0.0, "eps_growth": 0.0, "trailing_eps": 0.0, "pe_ratio": 0.0, "revenue_ttm": 0.0,
        "long_name": ticker, "error": None, "market_cap": None, "insiders": pd.DataFrame()
    }
    try:
        stock = yf.Ticker(ticker)
        out["price"] = _robust_price(stock, ticker)
        out["shares_info"] = _robust_shares(stock)
        
        full_info = {}
        try:
            full_info = stock.info or {}
        except:
            pass

        out["sector"] = full_info.get("sector", "Default") or "Default"
        out["target_price"] = full_info.get("targetMeanPrice", None)
        out["long_name"] = full_info.get("longName", ticker) or ticker
        out["rev_growth"] = float(full_info.get("revenueGrowth", 0) or 0)
        out["eps_growth"] = float(full_info.get("earningsGrowth", 0) or 0)
        out["trailing_eps"] = float(full_info.get("trailingEps", 0) or 0)
        out["pe_ratio"] = float(full_info.get("trailingPE", 0) or 0)
        
        # Get Total Revenue from Info first
        out["revenue_ttm"] = float(full_info.get("totalRevenue", 0) or 0)

        if out["shares_info"] > 0 and out["price"] > 0:
            out["market_cap"] = out["shares_info"] * out["price"]
            out["shares_calc"] = out["shares_info"]
        else:
            out["market_cap"] = full_info.get("marketCap", None)
            if out["market_cap"] and out["price"] > 0:
                out["shares_calc"] = float(out["market_cap"]) / float(out["price"])

        bs_q = _safe_df(getattr(stock, "quarterly_balance_sheet", None))
        inc_q = _safe_df(getattr(stock, "quarterly_financials", None))
        cf_q = _safe_df(getattr(stock, "quarterly_cashflow", None))

        bs_a = _safe_df(getattr(stock, "balance_sheet", None))
        inc_a = _safe_df(getattr(stock, "financials", None))
        cf_a = _safe_df(getattr(stock, "cashflow", None))

        out["bs"] = bs_q if not bs_q.empty else bs_a
        out["inc"] = inc_q if not inc_q.empty else inc_a
        out["cf"] = cf_q if not cf_q.empty else cf_a
        
        # Expose raw dataframes for advanced plotting
        out["bs_a"], out["bs_q"] = bs_a, bs_q
        out["inc_a"], out["inc_q"] = inc_a, inc_q
        out["cf_a"], out["cf_q"] = cf_a, cf_q
        
        try:
            out["insiders"] = _safe_df(stock.insider_transactions)
        except:
            pass

        if out["rev_growth"] == 0:
            out["rev_growth"] = float(get_growth_manual(out["inc"], ["TotalRevenue", "Revenue"]) or 0)
        if out["eps_growth"] == 0:
            out["eps_growth"] = float(get_growth_manual(out["inc"], ["NetIncome", "Net Income Common Stockholders"]) or 0)

        try:
            out["reco_summary"] = getattr(stock, "recommendations_summary", None)
        except:
            pass
        try:
            out["calendar"] = getattr(stock, "calendar", None)
        except:
            pass
        out["ir_news"] = fetch_ir_press_releases(out["long_name"])
        
        return out
    except Exception as e:
        out["error"] = str(e)
        return out
