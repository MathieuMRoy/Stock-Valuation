"""
Technical Patterns - Pattern recognition for technical analysis
"""
import pandas as pd


def bull_flag_score(df: pd.DataFrame) -> dict:
    """
    Calculate a bull flag pattern score based on technical indicators.
    
    Args:
        df: DataFrame with price data and indicators
    
    Returns:
        Dictionary with is_bull_flag, score (0-10), and notes
    """
    if df is None or df.empty or df.shape[0] < 80:
        return {"is_bull_flag": False, "score": 0.0, "notes": "Not enough data."}
    
    d = df.copy().dropna(subset=["Close"])
    last_close = d["Close"].iloc[-1]
    sma20 = d["SMA20"].iloc[-1] if "SMA20" in d.columns else last_close
    sma50 = d["SMA50"].iloc[-1] if "SMA50" in d.columns else last_close
    
    score = 5.0
    
    # Price above moving averages
    if last_close > sma20:
        score += 2
    if last_close > sma50:
        score += 1
    
    # RSI in healthy range
    if "RSI14" in d.columns:
        rsi = d["RSI14"].iloc[-1]
        if 40 < rsi < 70:
            score += 1
    
    return {
        "is_bull_flag": score > 7,
        "score": min(10.0, score),
        "notes": "Trend Analysis"
    }
