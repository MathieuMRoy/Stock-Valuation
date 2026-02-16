"""
Technical Indicators - Price history and technical analysis indicators
"""
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(ttl=1800)
def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch price history for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (e.g., "1y", "6mo", "1mo")
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        # Ensure Close and Volume exist
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                return pd.DataFrame()
        
        df = df.reset_index()
        # Convert numeric columns
        cols_to_numeric = ["Close", "High", "Low", "Volume"]
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        df.dropna(subset=["Close"], inplace=True)
        return df
    except:
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to price DataFrame.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    
    out = df.copy()
    
    # 1. Simple Moving Averages
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA100"] = out["Close"].rolling(100).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    
    # 2. Bollinger Bands (20, 2)
    std20 = out["Close"].rolling(20).std()
    out["BB_Upper"] = out["SMA20"] + (std20 * 2)
    out["BB_Lower"] = out["SMA20"] - (std20 * 2)
    
    # 3. RSI (Relative Strength Index)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))
    
    # 4. MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACDSignal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    
    # 5. ATR (Average True Range)
    if "High" in out.columns and "Low" in out.columns:
        high_low = out["High"] - out["Low"]
        high_close = (out["High"] - out["Close"].shift()).abs()
        low_close = (out["Low"] - out["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["ATR"] = tr.rolling(14).mean()
        
    # 6. Stochastic Oscillator
    if "High" in out.columns and "Low" in out.columns:
        low14 = out["Low"].rolling(14).min()
        high14 = out["High"].rolling(14).max()
        out["Stoch_K"] = 100 * ((out["Close"] - low14) / (high14 - low14))
        out["Stoch_D"] = out["Stoch_K"].rolling(3).mean()
        
    # 7. OBV (On-Balance Volume)
    if "Volume" in out.columns:
        out["OBV"] = (np.sign(out["Close"].diff()) * out["Volume"]).fillna(0).cumsum()

    return out
