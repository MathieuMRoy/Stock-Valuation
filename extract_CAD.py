import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import streamlit as st
import yfinance as yf

# GESTION DES ERREURS D'IMPORT
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    st.warning("⚠️ 'matplotlib' n'est pas installé. Ajoute-le dans requirements.txt.")

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(page_title="Valuation Master Pro", page_icon="📱", layout="centered")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("🔑 Groq API Key", type="password", help="Gratuit sur console.groq.com. Commence par gsk_")
    
    st.divider()
    
    if st.button("🗑️ Reset Cache"):
        st.cache_data.clear()
        st.rerun()
    st.caption("À utiliser si les données semblent incorrectes.")

st.title("📱 Valuation Master Pro")
st.caption("Cash • Sales • Earnings • Health • Insiders • AI + Screener")

# =========================================================
# 0) SMART SEARCH DATABASE
# =========================================================
TICKER_DB = [
    "🔍 Other (Manual Entry)",
    "--- TECH US (MAGNIFICENT 7) ---",
    "AAPL - Apple Inc.",
    "MSFT - Microsoft Corp.",
    "NVDA - NVIDIA Corp.",
    "GOOG - Alphabet Inc. (Google)",
    "AMZN - Amazon.com",
    "META - Meta Platforms (Facebook)",
    "TSLA - Tesla Inc.",
    "--- POPULAR & CONSUMER ---",
    "DUOL - Duolingo",
    "UBER - Uber Technologies",
    "ABNB - Airbnb",
    "SPOT - Spotify",
    "NFLX - Netflix",
    "COST - Costco Wholesale",
    "LLY - Eli Lilly (Pharma)",
    "--- SPACE & DEFENSE ---",
    "MDA.TO - MDA Space (Canada)",
    "RKLB - Rocket Lab USA",
    "ASTS - AST SpaceMobile",
    "PLTR - Palantir Technologies",
    "LMT - Lockheed Martin",
    "PNG.V - Kraken Robotics",
    "IONQ - IonQ Inc",
    "--- CANADA (TSX) ---",
    "RY.TO - Royal Bank (RBC)",
    "TD.TO - TD Bank",
    "SHOP.TO - Shopify (CAD)",
    "CNR.TO - CN Rail",
    "ENB.TO - Enbridge",
    "VLE.TO - Valeura Energy",
    "ATD.TO - Alimentation Couche-Tard",
    "CSU.TO - Constellation Software",
    "--- CRYPTO & FINTECH ---",
    "COIN - Coinbase",
    "HOOD - Robinhood",
    "PYPL - PayPal",
    "SQ - Block (Square)",
    "MSTR - MicroStrategy",
]

# =========================================================
# 1) BENCHMARKS
# =========================================================
PEER_GROUPS = {
    "SPACE_TECH": {
        "tickers": ["MDA", "RKLB", "ASTS", "LUNR", "PL", "SPIR", "SPCE", "PNG", "IONQ"],
        "gr_sales": 20.0, "gr_fcf": 25.0, "gr_eps": 25.0, "ps": 6.0, "pe": 40.0, "p_fcf": 35.0, "wacc": 11.0,
        "name": "Space Tech & Robotics"
    },
    "CYBERSECURITY": {
        "tickers": ["PANW", "CRWD", "FTNT", "ZS", "OKTA", "NET", "CYBR"],
        "gr_sales": 22.0, "gr_fcf": 25.0, "gr_eps": 25.0, "ps": 9.0, "pe": 45.0, "p_fcf": 35.0, "wacc": 10.0,
        "name": "Cybersecurity & Network"
    },
    "SEMICONDUCTORS": {
        "tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "MU", "TXN"],
        "gr_sales": 18.0, "gr_fcf": 20.0, "gr_eps": 20.0, "ps": 8.0, "pe": 35.0, "p_fcf": 30.0, "wacc": 10.0,
        "name": "Semiconductors & AI"
    },
    "BIG_TECH": {
        "tickers": ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META"],
        "gr_sales": 12.0, "gr_fcf": 15.0, "gr_eps": 15.0, "ps": 6.5, "pe": 25.0, "p_fcf": 28.0, "wacc": 9.0,
        "name": "Big Tech / GAFAM"
    },
    "CONSUMER_APPS": {
        "tickers": ["DUOL", "UBER", "ABNB", "SPOT", "DASH", "BKNG", "PINS", "SNAP"],
        "gr_sales": 18.0, "gr_fcf": 25.0, "gr_eps": 25.0, "ps": 5.0, "pe": 30.0, "p_fcf": 25.0, "wacc": 10.0,
        "name": "Consumer Apps & Platforms"
    },
    "SAAS_CLOUD": {
        "tickers": ["CRM", "ADBE", "SNOW", "DDOG", "PLTR", "NOW", "SHOP", "WDAY", "MDB"],
        "gr_sales": 20.0, "gr_fcf": 22.0, "gr_eps": 25.0, "ps": 9.0, "pe": 40.0, "p_fcf": 35.0, "wacc": 10.0,
        "name": "SaaS & Enterprise Cloud"
    },
    "PHARMA_BIO": {
        "tickers": ["LLY", "NVO", "JNJ", "PFE", "MRK", "ABBV", "AMGN"],
        "gr_sales": 8.0, "gr_fcf": 10.0, "gr_eps": 12.0, "ps": 5.0, "pe": 25.0, "p_fcf": 22.0, "wacc": 8.5,
        "name": "Pharma & Biotech"
    },
    "FINANCE_US": {
        "tickers": ["JPM", "BAC", "V", "MA", "AXP", "GS", "MS"],
        "gr_sales": 6.0, "gr_fcf": 8.0, "gr_eps": 10.0, "ps": 3.0, "pe": 15.0, "p_fcf": 15.0, "wacc": 9.0,
        "name": "US Finance & Payments"
    },
    "ENERGY_OIL": {
        "tickers": ["XOM", "CVX", "SHEL", "TTE", "BP", "COP", "VLE", "SU", "CNQ"],
        "gr_sales": 3.0, "gr_fcf": 5.0, "gr_eps": 5.0, "ps": 1.5, "pe": 10.0, "p_fcf": 8.0, "wacc": 10.0,
        "name": "Energy & Oil Majors"
    },
    "AEROSPACE_DEF": {
        "tickers": ["LMT", "RTX", "BA", "GD", "NOC", "GE"],
        "gr_sales": 5.0, "gr_fcf": 8.0, "gr_eps": 8.0, "ps": 2.0, "pe": 18.0, "p_fcf": 18.0, "wacc": 8.5,
        "name": "Aerospace & Defense"
    },
    "STREAMING": {
        "tickers": ["NFLX", "DIS", "WBD", "PARA", "ROKU"],
        "gr_sales": 10.0, "gr_fcf": 15.0, "gr_eps": 18.0, "ps": 4.0, "pe": 25.0, "p_fcf": 20.0, "wacc": 9.0,
        "name": "Streaming & Media"
    },
    "EV_AUTO": {
        "tickers": ["TSLA", "RIVN", "LCID", "BYD", "F", "GM"],
        "gr_sales": 15.0, "gr_fcf": 12.0, "gr_eps": 15.0, "ps": 3.0, "pe": 30.0, "p_fcf": 25.0, "wacc": 11.0,
        "name": "Electric Vehicles"
    },
    "BANKS_CA": {
        "tickers": ["RY", "TD", "BMO", "BNS", "CM", "NA"],
        "gr_sales": 4.0, "gr_fcf": 5.0, "gr_eps": 6.0, "ps": 2.5, "pe": 11.0, "p_fcf": 12.0, "wacc": 8.0,
        "name": "Canadian Banks"
    }
}

SECTOR_BENCHMARKS = {
    "Technology": {"gr_sales": 12.0, "gr_fcf": 15.0, "gr_eps": 15.0, "ps": 5.0, "pe": 25.0, "p_fcf": 25.0, "wacc": 9.5},
    "Default": {"gr_sales": 7.0, "gr_fcf": 8.0, "gr_eps": 8.0, "ps": 2.5, "pe": 15.0, "p_fcf": 15.0, "wacc": 9.0}
}

def get_benchmark_data(ticker: str, sector_info: str) -> dict:
    ticker_clean = ticker.upper().split(".")[0]
    for _, data in PEER_GROUPS.items():
        clean_list = [t.upper().split(".")[0] for t in data.get("tickers", [])]
        if ticker_clean in clean_list:
            peers_list = [t for t in data.get("tickers", []) if t.upper().split(".")[0] != ticker_clean]
            peers_str = ", ".join(peers_list[:5])
            out = dict(data)
            out["source"] = "Comparables"
            out["peers"] = peers_str if peers_str else "Peers unavailable"
            return out

    bench = SECTOR_BENCHMARKS.get(sector_info, SECTOR_BENCHMARKS["Default"])
    out = dict(bench)
    out["source"] = "Sector"
    out["name"] = sector_info or "General"
    out["peers"] = "Sector Average"
    return out

# =========================================================
# 2) DATA HELPERS (ROBUSTES & CORRIGÉS)
# =========================================================
def _safe_df(x) -> pd.DataFrame:
    if x is None: return pd.DataFrame()
    if hasattr(x, "empty"): return x if not x.empty else pd.DataFrame()
    return pd.DataFrame()

def _robust_price(stock: yf.Ticker, ticker: str) -> float:
    try:
        if hasattr(stock, 'fast_info'):
            return float(stock.fast_info['last_price'])
    except: pass
    try:
        hist = stock.history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except: pass
    try:
        info = stock.info or {}
        return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
    except: pass
    return 0.0

def _robust_shares(stock: yf.Ticker) -> float:
    shares = 0.0
    try:
        shares = float(stock.info.get("sharesOutstanding", 0))
    except: pass
    if shares <= 0:
        try:
            if hasattr(stock, 'fast_info'):
                mcap = stock.fast_info['market_cap']
                price = stock.fast_info['last_price']
                if price > 0:
                    shares = mcap / price
        except: pass
    return shares

def _infer_is_quarterly(df: pd.DataFrame) -> bool:
    try:
        if df is None or df.empty: return False
        cols = list(df.columns)
        if len(cols) < 2: return False
        c0 = pd.to_datetime(cols[0])
        c1 = pd.to_datetime(cols[1])
        return abs((c0 - c1).days) < 160
    except: return False

def get_growth_manual(df: pd.DataFrame, keys: list) -> float:
    try:
        if df is None or df.empty: return 0.0
        row = None
        for key in keys:
            matches = df.index[df.index.astype(str).str.contains(key, case=False, regex=True)]
            if not matches.empty:
                row = df.loc[matches[0]]
                break
        if row is None: return 0.0
        vals = [v for v in row if isinstance(v, (int, float)) and not pd.isna(v)]
        if len(vals) >= 5 and vals[4] != 0: return float((vals[0] - vals[4]) / abs(vals[4]))
        elif len(vals) >= 2 and vals[1] != 0: return float((vals[0] - vals[1]) / abs(vals[1]))
        return 0.0
    except: return 0.0

def fetch_ir_press_releases(search_name: str) -> list:
    try:
        query = f"{search_name} press release investor relations"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall(".//item")[:3]:
            news_items.append({
                "title": (item.find("title").text or "").strip(),
                "link": (item.find("link").text or "").strip(),
                "pubDate": (item.find("pubDate").text or "")[:16]
            })
        return news_items
    except: return []

@st.cache_data(ttl=3600)
def get_financial_data_secure(ticker: str) -> dict:
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
        try: full_info = stock.info or {}
        except: pass

        out["sector"] = full_info.get("sector", "Default") or "Default"
        out["target_price"] = full_info.get("targetMeanPrice", None)
        out["long_name"] = full_info.get("longName", ticker) or ticker
        out["rev_growth"] = float(full_info.get("revenueGrowth", 0) or 0)
        out["eps_growth"] = float(full_info.get("earningsGrowth", 0) or 0)
        out["trailing_eps"] = float(full_info.get("trailingEps", 0) or 0)
        out["pe_ratio"] = float(full_info.get("trailingPE", 0) or 0)
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
        
        try: out["insiders"] = _safe_df(stock.insider_transactions)
        except: pass

        if out["rev_growth"] == 0:
            out["rev_growth"] = float(get_growth_manual(out["inc"], ["TotalRevenue", "Revenue"]) or 0)
        if out["eps_growth"] == 0:
            out["eps_growth"] = float(get_growth_manual(out["inc"], ["NetIncome", "Net Income Common Stockholders"]) or 0)

        try: out["reco_summary"] = getattr(stock, "recommendations_summary", None)
        except: pass
        try: out["calendar"] = getattr(stock, "calendar", None)
        except: pass
        out["ir_news"] = fetch_ir_press_releases(out["long_name"])
        
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

def get_item_safe(df: pd.DataFrame, search_terms: list) -> float:
    if df is None or df.empty: return 0.0
    for term in search_terms:
        matches = df.index[df.index.astype(str).str.contains(term, case=False, regex=True)]
        if not matches.empty:
            try:
                val = df.loc[matches[0]]
                if isinstance(val, pd.Series): return float(val.iloc[0])
                return float(val)
            except: return 0.0
    return 0.0

def get_ttm_or_latest(df: pd.DataFrame, keys_list: list) -> float:
    if df is None or df.empty: return 0.0
    is_q = len(df.columns) > 1 
    for key in keys_list:
        matches = df.index[df.index.astype(str).str.contains(key, case=False, regex=True)]
        if not matches.empty:
            row = df.loc[matches[0]]
            vals = [v for v in row if isinstance(v, (int, float)) and not pd.isna(v)]
            if not vals: return 0.0
            if len(vals) >= 4: return float(sum(vals[:4])) 
            if len(vals) == 1: return float(vals[0]) * 4
            return float(vals[0])
    return 0.0

def compute_asset_based_value(bs: pd.DataFrame, shares: float) -> dict:
    if bs is None or bs.empty or shares <= 0:
        return {"nav_ps": 0.0, "tnav_ps": 0.0, "notes": "Balance sheet unavailable."}
    total_assets = get_item_safe(bs, ["TotalAssets", "Total Assets"])
    total_liab = get_item_safe(bs, ["TotalLiab", "Total Liabilities"])
    goodwill = get_item_safe(bs, ["Goodwill"])
    intangibles = get_item_safe(bs, ["IntangibleAssets"])
    equity = total_assets - total_liab
    t_equity = (total_assets - goodwill - intangibles) - total_liab
    nav_ps = equity / shares if shares > 0 else 0.0
    tnav_ps = t_equity / shares if shares > 0 else 0.0
    notes = f"Assets={total_assets/1e9:.2f}B, Liab={total_liab/1e9:.2f}B"
    return {"nav_ps": float(nav_ps), "tnav_ps": float(tnav_ps), "notes": notes}

# Feature 1: Advanced Health Scores
def calculate_piotroski_score(bs, inc, cf):
    score = 0
    try:
        if bs.shape[1] < 2 or inc.shape[1] < 2: return None
        net_income = get_item_safe(inc, ["NetIncome", "Net Income"])
        total_assets = get_item_safe(bs, ["TotalAssets"])
        cfo = get_item_safe(cf, ["OperatingCashFlow", "Operating Cash Flow"])
        if cfo == 0: cfo = get_ttm_or_latest(cf, ["OperatingCashFlow"])
        roa = net_income / total_assets if total_assets else 0
        
        score += 1 if net_income > 0 else 0
        score += 1 if cfo > 0 else 0
        score += 1 if roa > 0 else 0
        score += 1 if cfo > net_income else 0

        curr_assets = get_item_safe(bs, ["CurrentAssets"])
        curr_liab = get_item_safe(bs, ["CurrentLiab", "Current Liabilities"])
        curr_ratio = curr_assets / curr_liab if curr_liab else 0
        score += 1 if curr_ratio > 1 else 0
        score += 1
        score += 1
        return min(score, 9)
    except: return 5

def calculate_altman_z(bs, inc, market_cap):
    try:
        total_assets = get_item_safe(bs, ["TotalAssets", "Total Assets", "Assets"])
        if total_assets <= 0: return 0
        
        curr_assets = get_item_safe(bs, ["CurrentAssets", "Current Assets"])
        curr_liab = get_item_safe(bs, ["CurrentLiab", "Current Liabilities"])
        working_cap = curr_assets - curr_liab
        retained_earnings = get_item_safe(bs, ["RetainedEarnings", "Retained Earnings", "Accumulated Deficit"])
        ebit = get_item_safe(inc, ["EBIT", "OperatingIncome", "Operating Income"])
        total_liab = get_item_safe(bs, ["TotalLiab", "Total Liabilities"])
        revenue = get_item_safe(inc, ["TotalRevenue", "Revenue"])

        A = working_cap / total_assets
        B = retained_earnings / total_assets
        C = ebit / total_assets
        D = market_cap / total_liab if total_liab > 0 else 0
        E = revenue / total_assets
        return 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    except: return 0

# =========================================================
# 3) VALUATION ENGINE
# =========================================================
def calculate_valuation(
    gr_sales, gr_fcf, gr_eps, wacc_val, ps_target, pe_target,
    revenue, fcf, eps, cash, debt, shares
):
    current_fcf = float(fcf or 0)
    safe_shares = max(shares, 1.0)
    
    if current_fcf <= 0 or wacc_val <= 0:
        price_dcf = 0.0
    else:
        fcf_projections = [current_fcf * (1 + gr_fcf) ** (i + 1) for i in range(5)]
        terminal_val = (fcf_projections[-1] * 1.03) / max((wacc_val - 0.03), 1e-6)
        pv_fcf = sum([val / ((1 + wacc_val) ** (i + 1)) for i, val in enumerate(fcf_projections)])
        price_dcf = ((pv_fcf + (terminal_val / ((1 + wacc_val) ** 5))) + cash - debt) / safe_shares

    if revenue <= 0: 
        price_sales = 0.0
    else: 
        future_market_cap = (revenue * ((1 + gr_sales) ** 5)) * ps_target
        discounted_mc = future_market_cap / ((1 + wacc_val) ** 5)
        price_sales = discounted_mc / safe_shares

    if eps <= 0: 
        price_earnings = 0.0
    else:
        eps_future = eps * ((1 + gr_eps) ** 5)
        price_earnings = (eps_future * pe_target) / ((1 + wacc_val) ** 5)

    return float(price_dcf), float(price_sales), float(price_earnings)

def solve_reverse_dcf(current_price, fcf, wacc, shares, cash, debt):
    if fcf <= 0 or current_price <= 0: return 0.0
    low, high = -0.50, 1.00
    for _ in range(30):
        mid = (low + high) / 2
        val, _, _ = calculate_valuation(0, mid, 0, wacc, 0, 0, 0, fcf, 0, cash, debt, shares)
        if val > current_price: high = mid
        else: low = mid
    return (low + high) / 2

def display_relative_analysis(current: float, benchmark: float, metric_name: str, group_name: str):
    if current <= 0 or benchmark <= 0:
        st.caption("Relative analysis unavailable.")
        return
    diff = ((current - benchmark) / benchmark) * 100
    if diff < -10:
        st.success(f"**Undervalued 🟢** (discount of {abs(diff):.0f}% vs {group_name})")
    elif diff > 10:
        st.error(f"**Overvalued 🔴** (premium of {diff:.0f}% vs {group_name})")
    else:
        st.warning(f"**Fair Value 🟡** (aligned vs {group_name})")
    st.write(f"Current {metric_name} **{current:.1f}x** vs Peer **{benchmark:.1f}x**")

# =========================================================
# 4) TECHNICALS
# =========================================================
@st.cache_data(ttl=1800)
def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
        if "Close" not in df.columns:
            if "Adj Close" in df.columns: df["Close"] = df["Adj Close"]
            else: return pd.DataFrame()
        df = df.reset_index()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        return df
    except: return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Close" not in df.columns: return pd.DataFrame()
    out = df.copy()
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACDSignal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    return out

def bull_flag_score(df: pd.DataFrame) -> dict:
    if df is None or df.empty or df.shape[0] < 80:
        return {"is_bull_flag": False, "score": 0.0, "notes": "Not enough data."}
    d = df.copy().dropna(subset=["Close"])
    last_close = d["Close"].iloc[-1]
    sma20 = d["SMA20"].iloc[-1] if "SMA20" in d.columns else last_close
    sma50 = d["SMA50"].iloc[-1] if "SMA50" in d.columns else last_close
    score = 5.0
    if last_close > sma20: score += 2
    if last_close > sma50: score += 1
    if "RSI14" in d.columns:
        rsi = d["RSI14"].iloc[-1]
        if 40 < rsi < 70: score += 1
    return {"is_bull_flag": score>7, "score": min(10.0, score), "notes": "Trend Analysis"}

def plot_technical_chart(df: pd.DataFrame, ticker: str):
    if not MATPLOTLIB_OK: return
    if df is None or df.empty:
        st.warning("No price history to chart.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Date"], df["Close"], linewidth=2, label="Close")
    if "SMA20" in df.columns: ax.plot(df["Date"], df["SMA20"], linewidth=1, label="SMA20", color="orange")
    if "SMA50" in df.columns: ax.plot(df["Date"], df["SMA50"], linewidth=1, label="SMA50", color="green")
    if "SMA200" in df.columns: ax.plot(df["Date"], df["SMA200"], linewidth=1, label="SMA200", color="red")
    ax.set_title(f"{ticker} — Price History")
    ax.legend()
    st.pyplot(fig)

# =========================================================
# 5) SCORING & RADAR
# =========================================================
def score_out_of_10(metrics: dict, bench: dict) -> dict:
    overall = 5.0
    if metrics['pe'] > 0 and metrics['pe'] < float(bench.get('pe', 20)): overall += 1
    if metrics['sales_gr'] > 0.10: overall += 1
    if metrics['net_cash'] > 0: overall += 1
    return {"overall": min(9.5, overall), "health": 7.0, "growth": 6.0, "valuation": 6.0, "sector": 5.0}

def plot_radar(scores: dict, tech_score: float = 5.0, dividend_score: float = 5.0):
    if not MATPLOTLIB_OK: return None
    labels = ["VALUE", "GROWTH", "HEALTH", "TECH", "DIV"]
    values = [scores.get("valuation", 5), scores.get("growth", 5), scores.get("health", 5), tech_score, dividend_score]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="gray", size=8)
    ax.grid(color='#AAAAAA', linestyle='--', linewidth=0.5)
    ax.plot(angles, values, linewidth=2, color="#4A90E2", linestyle='solid')
    ax.fill(angles, values, alpha=0.3, color="#4A90E2")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, weight='bold')
    plt.title("Scorecard Analysis (0-10)", size=12, weight='bold', pad=20)
    return fig

# =========================================================
# 6) AI ANALYST (GROQ)
# =========================================================
def ai_analyst_report(metrics: dict, bench: dict, scores: dict, tech: dict, api_key: str):
    if not GROQ_OK: return None, "Groq package missing."
    if not api_key: return None, "API Key missing."
    try:
        client = Groq(api_key=api_key)
        valuation_context = "Sous-évalué" if scores['valuation'] > 6 else "Sur-évalué"
        prompt = f"""
        Tu es Cameron Doerksen, Analyste Senior. Rédige un rapport de "Coverage" pour l'action {metrics['ticker']} (Données jan 2026).
        Prix: {metrics['price']}$. PE: {metrics['pe']}x. Croissance: {metrics['sales_gr']*100:.1f}%.
        Technique: {"Haussière" if tech['score'] > 6 else "Neutre"}.
        
        STRUCTURE DU RAPPORT:
        1. THÈSE D'INVESTISSEMENT (Note /10 et Recommandation: ACHAT/VENDRE)
        2. ANALYSE FONDAMENTALE (Forces/Faiblesses)
        3. VALORISATION ({valuation_context})
        4. STRATÉGIE TRADING (Prix d'entrée et Stop Loss précis)
        
        Ton analytique et professionnel. Français.
        """
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content, None
    except Exception as e: return None, str(e)

# =========================================================
# 9) MAIN LOGIC
# =========================================================
mode = st.sidebar.radio("Mode", ["Stock Analyzer", "AI Screener (Top Upside)"], index=0)

if mode == "Stock Analyzer":
    st.subheader("Search for a Company")
    choice = st.selectbox("Choose a popular stock:", TICKER_DB, index=2)
    ticker_final = "MSFT"
    if "Other" in choice:
        ticker_input = st.text_input("Ticker", "").upper()
        if ticker_input: ticker_final = ticker_input
    elif "-" in choice: ticker_final = choice.split("-")[0].strip()

    st.caption(f"Analyzing: **{ticker_final}**")
    data = get_financial_data_secure(ticker_final)
    current_price = float(data.get("price", 0) or 0)

    if current_price <= 0:
        st.error("Prix introuvable. Vérifiez le ticker.")
        st.stop()

    bs = data.get("bs", pd.DataFrame())
    inc = data.get("inc", pd.DataFrame())
    cf = data.get("cf", pd.DataFrame())
    piotroski = calculate_piotroski_score(bs, inc, cf)
    
    shares = float(data.get("shares_info", 0) or 0)
    st.sidebar.markdown("### 🔧 Data Override")
    manual_shares = st.sidebar.number_input("Manual Shares (Millions)", value=0.0, step=1.0)
    if manual_shares > 0: shares = manual_shares * 1_000_000
    if shares <= 1: shares = 1.0

    market_cap = shares * current_price
    altman_z = calculate_altman_z(bs, inc, market_cap)

    revenue_ttm = data.get("revenue_ttm", 0)
    if revenue_ttm == 0: revenue_ttm = get_ttm_or_latest(inc, ["TotalRevenue", "Revenue"])

    cfo_ttm = get_ttm_or_latest(cf, ["OperatingCashFlow", "Operating Cash Flow"])
    capex_ttm = abs(get_item_safe(cf, ["CapitalExpenditure", "PurchaseOfPPE"]))
    fcf_ttm = cfo_ttm - capex_ttm
    cash = get_item_safe(bs, ["CashAndCashEquivalents", "Cash"])
    debt = get_item_safe(bs, ["LongTermDebt"]) + get_item_safe(bs, ["LeaseLiabilities", "TotalLiab"])
    
    eps_ttm = data.get("trailing_eps", 0)
    if eps_ttm == 0:
        net_inc = get_ttm_or_latest(inc, ["NetIncome", "Net Income Common Stockholders"])
        if shares > 0: eps_ttm = net_inc / shares

    pe = data.get("pe_ratio", 0)
    if pe == 0 and eps_ttm > 0: pe = current_price / eps_ttm
        
    ps = market_cap / revenue_ttm if revenue_ttm > 0 else 0
    
    cur_sales_gr = data.get("rev_growth", 0)
    cur_eps_gr = data.get("eps_growth", 0)
    bench_data = get_benchmark_data(ticker_final, data.get("sector", "Default"))
    
    price_df = fetch_price_history(ticker_final, "1y")
    tech_df = add_indicators(price_df)
    tech = bull_flag_score(tech_df)

    metrics = {
        "ticker": ticker_final, "price": current_price, "pe": pe, "ps": ps, 
        "sales_gr": cur_sales_gr, "eps_gr": cur_eps_gr, "net_cash": cash-debt, 
        "fcf_yield": fcf_ttm/market_cap if market_cap else 0,
        "rule_40": cur_sales_gr + ((fcf_ttm/revenue_ttm) if revenue_ttm else 0)
    }
    scores = score_out_of_10(metrics, bench_data)

    with st.expander(f"💡 Help: {bench_data['name']} vs {ticker_final}", expanded=True):
        st.write(f"**Peers:** {bench_data.get('peers', 'N/A')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peer Sales Gr.", f"{bench_data['gr_sales']:.0f}%")
        c2.metric("Peer EPS Gr.", f"{bench_data.get('gr_eps', 0):.0f}%")
        c3.metric("Peer Target P/S", f"{bench_data['ps']}x")
        c4.metric("Peer Target P/E", f"{bench_data.get('pe', 20)}x")
        st.divider()
        # --- MODIFICATION ICI ---
        st.markdown(f"#### 📍 **{ticker_final}** Current Metrics (Actual)")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Actual Sales Gr.", f"{cur_sales_gr*100:.1f}%")
        c6.metric("Actual EPS Gr.", f"{cur_eps_gr*100:.1f}%")
        c7.metric("Actual P/S", f"{ps:.1f}x")
        c8.metric("Actual P/E", f"{pe:.1f}x")

    with st.expander("⚙️ Edit Assumptions", expanded=False):
        c1, c2, c3 = st.columns(3)
        gr_sales = c1.number_input("Sales Growth %", value=float(bench_data['gr_sales']))
        gr_fcf = c2.number_input("FCF Growth %", value=float(bench_data['gr_fcf']))
        wacc = c3.number_input("WACC %", value=float(bench_data['wacc']))
        c4, c5 = st.columns(2)
        target_pe = c4.number_input("Target P/E", value=float(bench_data.get('pe', 20)))
        target_ps = c5.number_input("Target P/S", value=float(bench_data['ps']))

    def run_calc(g_fac, m_fac, w_adj):
        return calculate_valuation(
            gr_sales/100*g_fac, gr_fcf/100*g_fac, 0.10, wacc/100 + w_adj, 
            target_ps*m_fac, target_pe*m_fac, 
            revenue_ttm, fcf_ttm, eps_ttm, cash, debt, shares
        )

    bear_res = run_calc(0.8, 0.8, 0.01)
    base_res = run_calc(1.0, 1.0, 0.0)
    bull_res = run_calc(1.2, 1.2, -0.01)

    st.metric("Current Price", f"{current_price:.2f} $")
    
    tabs = st.tabs(["💵 DCF (Cash)", "📈 Sales (P/S)", "💰 Earnings (P/E)", "🧱 Assets", "👥 Insiders", "📉 Tech", "📊 Scorecard", "🤖 AI Agent"])
    
    with tabs[0]:
        st.subheader("💵 Buy Price (DCF)")
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[0]:.2f} $", delta=f"{base_res[0]-current_price:.2f}")
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[0]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[0]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[0]:.2f} $")
        
        st.markdown("##### 📝 Investment Theses")
        st.error(f"**🐻 Bear:** FCF Growth slows to **{gr_fcf*0.8:.1f}%**, Exit Multiple contracts.")
        st.info(f"**🎯 Neutral:** Base case. FCF Growth **{gr_fcf:.1f}%**, WACC **{wacc:.1f}%**.")
        st.success(f"**🐂 Bull:** Execution perfect. FCF Growth accelerates to **{gr_fcf*1.2:.1f}%**.")

        st.markdown("##### Reverse DCF")
        implied_g = solve_reverse_dcf(current_price, fcf_ttm, wacc/100, shares, cash, debt)
        st.metric("Market Implied Growth", f"{implied_g*100:.1f}%")
        
        st.markdown("##### 🌡️ Sensitivity Matrix (Price vs Growth & WACC)")
        sens_wacc = [wacc-1, wacc-0.5, wacc, wacc+0.5, wacc+1]
        sens_growth = [gr_fcf-2, gr_fcf-1, gr_fcf, gr_fcf+1, gr_fcf+2]
        res_matrix = []
        for w in sens_wacc:
            row_vals = []
            for g in sens_growth:
                val, _, _ = calculate_valuation(0, g/100, 0, w/100, 0, 0, revenue_ttm, fcf_ttm, 0, cash, debt, shares)
                row_vals.append(val)
            res_matrix.append(row_vals)
        df_sens = pd.DataFrame(res_matrix, index=[f"WACC {w:.1f}%" for w in sens_wacc], columns=[f"Gr {g:.1f}%" for g in sens_growth])
        st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f} $"))

    with tabs[1]:
        st.subheader("📈 Buy Price (Sales)")
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[1]:.2f} $", delta=f"{base_res[1]-current_price:.2f}")
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[1]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[1]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[1]:.2f} $")
        
        st.markdown("##### 📝 Investment Theses")
        st.error(f"**🐻 Bear:** Market sentiment sours, P/S drops to **{target_ps*0.8:.1f}x**.")
        st.info(f"**🎯 Neutral:** Maintains valuation at **{target_ps:.1f}x** sales.")
        st.success(f"**🐂 Bull:** High demand, P/S expands to **{target_ps*1.2:.1f}x**.")
        
        st.write("")
        display_relative_analysis(ps, float(bench_data.get('ps', 3)), "P/S", bench_data['name'])

    with tabs[2]:
        st.subheader("💰 Buy Price (P/E)")
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[2]:.2f} $", delta=f"{base_res[2]-current_price:.2f}")
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[2]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[2]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[2]:.2f} $")
        
        st.markdown("##### 📝 Investment Theses")
        st.error(f"**🐻 Bear:** EPS growth misses targets, P/E contracts to **{target_pe*0.8:.1f}x**.")
        st.info(f"**🎯 Neutral:** Stable growth, P/E holds at **{target_pe:.1f}x**.")
        st.success(f"**🐂 Bull:** Growth beats expectations, P/E expands to **{target_pe*1.2:.1f}x**.")
        
        st.write("")
        display_relative_analysis(pe, float(bench_data.get('pe', 20)), "P/E", bench_data['name'])

    with tabs[3]:
        st.subheader("🧱 Asset Based Value")
        ab = compute_asset_based_value(bs, shares)
        c1, c2 = st.columns(2)
        c1.metric("NAV / Share", f"{ab['nav_ps']:.2f} $")
        c2.metric("Tangible NAV", f"{ab['tnav_ps']:.2f} $")
        st.caption(ab["notes"])

    with tabs[4]:
        st.subheader("👥 Insider Trading")
        if not data['insiders'].empty:
            st.dataframe(data['insiders'].head(10))
        else:
            st.info("No insider data found.")

    with tabs[5]:
        st.subheader("📉 Technical Analysis")
        c1, c2 = st.columns(2)
        c1.metric("Bull Flag Score", f"{tech['score']}/10")
        c2.metric("Pattern", "Bull Flag" if tech['is_bull_flag'] else "None")
        plot_technical_chart(tech_df, ticker_final)

    with tabs[6]:
        st.subheader("📊 Scorecard Pro")
        c1, c2, c3 = st.columns(3)
        c1.metric("Health", f"{scores['health']}/10")
        c2.metric("Growth", f"{scores['growth']}/10")
        c3.metric("Value", f"{scores['valuation']}/10")
        fig = plot_radar(scores, tech['score'])
        if fig: st.pyplot(fig)
        
        st.divider()
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.info(f"**Piotroski F-Score: {piotroski if piotroski else 'N/A'}/9**\n\nScore de santé financière (Profitabilité, Levier, Efficacité). 9 est excellent, <4 est risqué.")
        with c_exp2:
            z_col = "green" if altman_z > 3 else "red" if altman_z < 1.8 else "orange"
            st.warning(f"**Altman Z-Score: :{z_col}[{altman_z:.2f}]**\n\nRisque de faillite.\n* **> 3.0:** Sûr (Safe)\n* **< 1.8:** Détresse (Distress)")

    with tabs[7]:
        st.subheader("🤖 AI Analyst (Groq)")
        if st.button("✨ Generate Full Report"):
            with st.spinner("Analyzing..."):
                rep, err = ai_analyst_report(metrics, bench_data, scores, tech, api_key)
                if rep: st.markdown(rep)
                else: st.error(err)

else:
    # SCREENER LOGIC
    st.subheader("🔍 AI Screener (Top Upside)")
    min_market_cap = st.number_input("Min Market Cap (USD)", value=1_000_000_000, step=250_000_000)
    max_tickers_per_sector = st.slider("Max tickers per sector", 10, 80, 20, step=5)
    colA, colB = st.columns(2)
    scr_gr_fcf = colA.number_input("FCF Growth % (fallback)", value=15.0, step=0.5)
    scr_wacc = colB.number_input("WACC %", value=10.0, step=0.5)

    def intrinsic_dcf_quick(ticker: str) -> dict:
        d = get_financial_data_secure(ticker)
        p = float(d.get("price", 0) or 0)
        s = float(d.get("shares_info", 0) or 0)
        if p <= 0 or s <= 0: return {"ok": False}
        if not _mc_ok(d, min_market_cap): return {"ok": False}
        
        rev = get_ttm_or_latest(d["inc"], ["Revenue"])
        cf = get_ttm_or_latest(d["cf"], ["OperatingCashFlow"])
        cap = abs(get_item_safe(d["cf"], ["CapitalExpenditure"]))
        fcf = cf - cap
        c = get_item_safe(d["bs"], ["Cash"])
        db = get_item_safe(d["bs"], ["LongTermDebt"])
        
        g = float(d.get("rev_growth", 0) or 0)
        if g <= 0: g = scr_gr_fcf / 100.0
        
        dcf, _, _ = calculate_valuation(0, g, 0, scr_wacc/100, 0, 0, rev, fcf, 0, c, db, s)
        if dcf <= 0: return {"ok": False}
        return {"ticker": ticker, "price": p, "intrinsic": dcf, "upside": (dcf/p - 1)*100, "ok": True, "bucket": d.get("sector")}

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
            st.dataframe(df[["bucket", "ticker", "price", "intrinsic", "upside"]].style.format({"price": "{:.2f}", "intrinsic": "{:.2f}", "upside": "{:.1f}%"}))
        else: st.error("No results.")
