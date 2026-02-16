"""
Short Interest Fetcher - Get historical short interest data from Nasdaq.com Public API
"""
import requests
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600*24) # Cache for 24 hours
def get_historical_short_interest(ticker: str) -> pd.DataFrame:
    """
    Fetch historical short interest collection from Nasdaq API.
    
    Args:
        ticker: Stock symbol (e.g., AAPL)
        
    Returns:
        DataFrame with columns [Date, Short Interest, Avg Daily Volume, Days to Cover]
        sorted by date ascending.
    """
    url = f"https://api.nasdaq.com/api/quote/{ticker.upper()}/short-interest?assetclass=stocks"
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Origin": "https://www.nasdaq.com",
        "Referer": f"https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/short-interest"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
            
        data = r.json()
        
        # Try new format first: shortInterestTable.rows
        collection = data.get("data", {}).get("shortInterestTable", {}).get("rows", [])
        
        # Fallback to old format: shortInterestList.collection
        if not collection:
            collection = data.get("data", {}).get("shortInterestList", {}).get("collection", [])
        
        if not collection:
            return pd.DataFrame()
            
        df = pd.DataFrame(collection)
        
        # Handle both old and new field names
        rename_map = {}
        if "settlementDate" in df.columns:
            rename_map["settlementDate"] = "Date"
        # New format uses "interest", old uses "shortInterest"
        if "interest" in df.columns:
            rename_map["interest"] = "Short Interest"
        elif "shortInterest" in df.columns:
            rename_map["shortInterest"] = "Short Interest"
        if "avgDailyShareVolume" in df.columns:
            rename_map["avgDailyShareVolume"] = "Avg Daily Volume"
        if "daysToCover" in df.columns:
            rename_map["daysToCover"] = "Days to Cover"
            
        df = df.rename(columns=rename_map)
        
        # Convert date and sort
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date", ascending=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching short interest from Nasdaq: {e}")
        return pd.DataFrame()

