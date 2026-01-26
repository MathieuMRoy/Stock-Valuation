import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# --- MODIFICATION 1 : GROQ AU LIEU D'OPENAI ---
try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Valuation Master Pro", page_icon="📱", layout="centered")

with st.sidebar:
    st.header("⚙️ Configuration")
    # Ajout de la clé API pour Groq
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
# 2) DATA HELPERS (ROBUST & CORRIGÉ)
# =========================================================
def _safe_df(x) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame()
    if hasattr(x, "empty"):
        return x if not x.empty else pd.DataFrame()
    return pd.DataFrame()

# --- CORRECTION 2 : FONCTION PRIX/SHARES ROBUSTE ---
def _robust_price(stock: yf.Ticker, ticker: str) -> float:
    # 1. FastInfo (Méthode moderne)
    try:
        if hasattr(stock, 'fast_info'):
            return float(stock.fast_info['last_price'])
    except: pass
    
    # 2. History (Méthode lente mais sûre)
    try:
        hist = stock.history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except: pass
    
    # 3. Info (Méthode legacy)
    try:
        info = stock.info or {}
        return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
    except: pass
    
    return 0.0

def _robust_shares(stock: yf.Ticker) -> float:
    shares = 0.0
    # 1. Info
    try:
        shares = float(stock.info.get("sharesOutstanding", 0))
    except: pass
    
    # 2. FastInfo (Calcul inversé : MarketCap / Prix)
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
        if df is None or df.empty:
            return False
        cols = list(df.columns)
        if len(cols) < 2:
            return False
        c0 = pd.to_datetime(cols[0])
        c1 = pd.to_datetime(cols[1])
        delta_days = abs((c0 - c1).days)
        return delta_days < 160
    except Exception:
        return False

def get_growth_manual(df: pd.DataFrame, keys: list) -> float:
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
        if len(vals) >= 5:
            current = vals[0]
            last_year = vals[4]
            if last_year != 0:
                return float((current - last_year) / abs(last_year))
        elif len(vals) >= 2:
            current = vals[0]
            prev = vals[1]
            if prev != 0:
                return float((current - prev) / abs(prev))
        return 0.0
    except Exception:
        return 0.0

def fetch_ir_press_releases(search_name: str) -> list:
    try:
        query = f"{search_name} press release investor relations"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=6)
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall(".//item")[:3]:
            news_items.append({
                "title": (item.find("title").text or "").strip(),
                "link": (item.find("link").text or "").strip(),
                "pubDate": (item.find("pubDate").text or "")[:16]
            })
        return news_items
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_financial_data_secure(ticker: str) -> dict:
    out = {
        "bs": pd.DataFrame(), "inc": pd.DataFrame(), "cf": pd.DataFrame(),
        "reco_summary": None, "calendar": None, "target_price": None, "ir_news": [],
        "price": 0.0, "shares_calc": 0.0, "sector": "Default",
        "rev_growth": 0.0, "eps_growth": 0.0, "trailing_eps": 0.0,
        "shares_info": 0.0, "long_name": ticker, "error": None,
        "market_cap": None, "insiders": pd.DataFrame()
    }

    try:
        stock = yf.Ticker(ticker)

        # UTILISATION DES FONCTIONS ROBUSTES
        out["price"] = _robust_price(stock, ticker)
        out["shares_info"] = _robust_shares(stock)

        full_info = {}
        try: full_info = stock.info or {}
        except: full_info = {}

        out["sector"] = full_info.get("sector", "Default") or "Default"
        out["target_price"] = full_info.get("targetMeanPrice", None)
        out["long_name"] = full_info.get("longName", ticker) or ticker
        out["rev_growth"] = float(full_info.get("revenueGrowth", 0) or 0)
        out["eps_growth"] = float(full_info.get("earningsGrowth", 0) or 0)
        out["trailing_eps"] = float(full_info.get("trailingEps", 0) or 0)

        # Calcul Market Cap et Shares
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
        except: out["reco_summary"] = None

        try: out["calendar"] = getattr(stock, "calendar", None)
        except: out["calendar"] = None

        out["ir_news"] = fetch_ir_press_releases(out["long_name"])
        return out

    except Exception as e:
        out["error"] = str(e)
        return out

def get_item_safe(df: pd.DataFrame, search_terms: list) -> float:
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
            except Exception:
                return 0.0
    return 0.0

def get_ttm_or_latest(df: pd.DataFrame, keys_list: list) -> float:
    if df is None or df.empty:
        return 0.0
    is_q = _infer_is_quarterly(df)
    for key in keys_list:
        matches = df.index[df.index.astype(str).str.contains(key, case=False, regex=True)]
        if not matches.empty:
            row = df.loc[matches[0]]
            vals = [v for v in row if isinstance(v, (int, float)) and not pd.isna(v)]
            if not vals:
                return 0.0
            if is_q and len(vals) >= 4:
                return float(sum(vals[:4]))
            return float(vals[0])
    return 0.0

# Feature 1: Advanced Health Scores
def calculate_piotroski_score(bs, inc, cf):
    score = 0
    try:
        if bs.shape[1] < 2 or inc.shape[1] < 2:
            return None
            
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
    except:
        return 5

def calculate_altman_z(bs, inc, market_cap):
    try:
        total_assets = get_item_safe(bs, ["TotalAssets"])
        if total_assets <= 0: return 0

        curr_assets = get_item_safe(bs, ["CurrentAssets"])
        curr_liab = get_item_safe(bs, ["CurrentLiab", "Current Liabilities"])
        working_cap = curr_assets - curr_liab
        
        retained_earnings = get_item_safe(bs, ["RetainedEarnings", "Retained Earnings"])
        ebit = get_item_safe(inc, ["EBIT", "OperatingIncome", "Operating Income"])
        total_liab = get_item_safe(bs, ["TotalLiab", "Total Liabilities"])
        revenue = get_item_safe(inc, ["TotalRevenue", "Revenue"])

        A = working_cap / total_assets
        B = retained_earnings / total_assets
        C = ebit / total_assets
        D = market_cap / total_liab if total_liab > 0 else 0
        E = revenue / total_assets

        z_score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        return z_score
    except:
        return 0

# =========================================================
# 3) VALUATION ENGINE
# =========================================================
def calculate_valuation(
    gr_sales, gr_fcf, gr_eps, wacc_val, ps_target, pe_target,
    revenue, fcf, eps, cash, debt, shares
):
    current_fcf = float(fcf or 0)
    # Protection division par zéro ou shares=0
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
        price_sales = (((revenue * ((1 + gr_sales) ** 5)) * ps_target) / safe_shares) / (1.10 ** 5)

    if eps <= 0:
        price_earnings = 0.0
    else:
        eps_future = eps * ((1 + gr_eps) ** 5)
        price_earnings = (eps_future * pe_target) / (1.10 ** 5)

    return float(price_dcf), float(price_sales), float(price_earnings)

def solve_reverse_dcf(current_price, fcf, wacc, shares, cash, debt):
    if fcf <= 0 or current_price <= 0:
        return 0.0
    low = -0.50
    high = 1.00
    for _ in range(30):
        mid = (low + high) / 2
        val, _, _ = calculate_valuation(0, mid, 0, wacc, 0, 0, 0, fcf, 0, cash, debt, shares)
        if val > current_price:
            high = mid
        else:
            low = mid
    return (low + high) / 2

def display_relative_analysis(current: float, benchmark: float, metric_name: str, group_name: str):
    if current <= 0 or benchmark <= 0:
        st.caption("Relative analysis unavailable.")
        return

    diff = ((current - benchmark) / benchmark) * 100
    if diff < -10:
        box = st.success
        status = "Undervalued "
        msg = f"discount of {abs(diff):.0f}%"
    elif diff > 10:
        box = st.error
        status = "Overvalued "
        msg = f"premium of {diff:.0f}%"
    else:
        box = st.warning
        status = "Fair Value "
        msg = "aligned"

    box(
        f"** Relative Analysis:** Current {metric_name} **{current:.1f}x** vs Peer/Sector **{benchmark:.1f}x**.\n\n"
        f" **Verdict: {status}** ({msg} vs {group_name})."
    )

# =========================================================
# 4) TECHNICALS (ROBUST)
# =========================================================
@st.cache_data(ttl=1800)
def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                return pd.DataFrame()

        df = df.reset_index()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()

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
    if d.shape[0] < 80:
        return {"is_bull_flag": False, "score": 0.0, "notes": "Not enough clean data."}

    last60 = d.tail(60)
    first60 = last60["Close"].iloc[0]
    last_close = last60["Close"].iloc[-1]
    impulse = (last_close / first60 - 1) if first60 > 0 else 0

    flag = d.tail(15)
    flag_range = (flag["Close"].max() / flag["Close"].min() - 1) if flag["Close"].min() > 0 else 1
    x = np.arange(len(flag))
    y = flag["Close"].values
    slope = np.polyfit(x, y, 1)[0] if len(y) > 3 else 0

    sma20 = float(d["SMA20"].iloc[-1]) if "SMA20" in d.columns and pd.notna(d["SMA20"].iloc[-1]) else np.nan
    near_high = (last_close / d["Close"].tail(60).max()) if d["Close"].tail(60).max() > 0 else 0

    score = 0.0
    score += min(max((impulse - 0.10) / 0.20, 0), 1) * 4.0
    score += min(max((0.10 - flag_range) / 0.08, 0), 1) * 3.0
    score += (1.0 if slope <= 0 else 0.4) * 1.5
    if not np.isnan(sma20) and last_close > sma20:
        score += 1.5
    else:
        score += 0.3
    score += min(max((near_high - 0.92) / 0.06, 0), 1) * 1.0

    score = float(max(0, min(10, score)))
    is_flag = score >= 7.0
    notes = (
        f"Impulse(60d): {impulse*100:.1f}%, "
        f"FlagRange(15d): {flag_range*100:.1f}%, "
        f"Slope(15d): {'down/flat' if slope<=0 else 'up'}, "
        f"NearHigh(60d): {near_high*100:.1f}%"
    )
    return {"is_bull_flag": bool(is_flag), "score": score, "notes": notes}

def plot_technical_chart(df: pd.DataFrame, ticker: str):
    if df is None or df.empty:
        st.warning("No price history to chart.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Date"], df["Close"], linewidth=2, label="Close")
    if "SMA20" in df.columns:
        ax.plot(df["Date"], df["SMA20"], linewidth=1, label="SMA20", color="orange")
    if "SMA50" in df.columns:
        ax.plot(df["Date"], df["SMA50"], linewidth=1, label="SMA50", color="green")
    if "SMA200" in df.columns:
        ax.plot(df["Date"], df["SMA200"], linewidth=1, label="SMA200", color="red")
    ax.set_title(f"{ticker} — Price & Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# =========================================================
# 5) SCORING
# =========================================================
def clamp(x, lo=0, hi=10):
    return max(lo, min(hi, x))

def score_out_of_10(metrics: dict, bench: dict) -> dict:
    sales_gr = metrics.get("sales_gr", 0) * 100
    rule_40 = metrics.get("rule_40", sales_gr)

    growth_score = clamp((sales_gr / 5))
    growth_score = clamp(growth_score + (rule_40 - 20) / 10)

    net_cash = metrics.get("net_cash", 0)
    fcf_margin = metrics.get("fcf_margin", 0)
    fcf_yield = metrics.get("fcf_yield", 0)

    health_score = 5.0
    health_score += 1.5 if net_cash > 0 else -1.5
    health_score += fcf_margin / 10
    health_score += fcf_yield / 3
    health_score = clamp(health_score)

    pe = metrics.get("pe", 0)
    ps = metrics.get("ps", 0)
    pfcf = metrics.get("pfcf", 0)

    bench_pe = float(bench.get("pe", 20) or 20)
    bench_ps = float(bench.get("ps", 3) or 3)
    bench_pfcf = float(bench.get("p_fcf", 20) or 20)

    val_score = 5.0
    if pe > 0 and bench_pe > 0:
        val_score += (bench_pe - pe) / bench_pe * 4
    if ps > 0 and bench_ps > 0:
        val_score += (bench_ps - ps) / bench_ps * 3
    if pfcf > 0 and bench_pfcf > 0:
        val_score += (bench_pfcf - pfcf) / bench_pfcf * 3
    val_score = clamp(val_score)

    sector_score = clamp((float(bench.get("gr_sales", 7) or 7) / 3) - (float(bench.get("wacc", 9) or 9) - 8) / 2)

    overall = clamp(
        0.35 * growth_score +
        0.30 * health_score +
        0.25 * val_score +
        0.10 * sector_score
    )

    return {
        "sector": round(sector_score, 1),
        "growth": round(growth_score, 1),
        "health": round(health_score, 1),
        "valuation": round(val_score, 1),
        "overall": round(overall, 1),
    }

def plot_radar(scores: dict, tech_score: float = 5.0, dividend_score: float = 5.0):
    labels = ["VALUE", "GROWTH", "HEALTH", "TECH", "DIV"]
    values = [
        scores.get("valuation", 5.0),
        scores.get("growth", 5.0),
        scores.get("health", 5.0),
        tech_score,
        dividend_score,
    ]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.30)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticklabels([])
    return fig

# =========================================================
# 6) AI ANALYST (GROQ)
# =========================================================
def ai_analyst_report(metrics: dict, bench: dict, scores: dict, tech: dict, api_key: str):
    if not GROQ_OK:
        return None, "Le package groq n'est pas installé."
    
    if not api_key:
        return None, "Veuillez entrer votre clé API Groq dans la barre latérale."

    try:
        client = Groq(api_key=api_key)

        valuation_context = "Sous-évalué" if scores['valuation'] > 6 else "Sur-évalué" if scores['valuation'] < 4 else "Correctement valorisé"

        prompt = f"""
        Tu es Cameron Doerksen, Analyste Senior en Equity Research chez Goldman Sachs / NBCFM.
        Rédige un rapport de "Coverage" institutionnel complet sur l'action {metrics['ticker']} (Dernières données du 23 janvier 2026).

        CONCOURS MACROÉCONOMIQUE ACTUEL (Contexte Canada Defense) :
        - Le Canada a officiellement accepté la cible NATO de 3,5% du PIB d'ici 2035.
        - Modernisation du NORAD estimée à 38,6 milliards $ sur 20 ans.
        - Priorité gouvernementale sur la souveraineté Arctique et les drones sous-marins.

        DONNÉES FINANCIÈRES RÉELLES (Action {metrics['ticker']}) :
        - Prix actuel : {metrics['price']:.2f} $
        - Croissance Revenus : {metrics['sales_gr']*100:.1f}%
        - P/E Ratio : {metrics['pe']:.1f}x (Benchmark Peers : {bench.get('pe', 'N/A')}x)
        - P/S Ratio : {metrics['ps']:.1f}x
        - Trésorerie Nette : {metrics['net_cash']/1e6:.0f} M$
        - Santé Financière (Score): {scores['health']}/10
        - Tendance Technique : {"Haussière" if tech['score'] > 6 else "Baissière" if tech['score'] < 4 else "Neutre"}
        - Bull Flag Score : {tech['score']:.1f}/10
        
        TA MISSION :
        Rédige un rapport structuré d'au moins 800 mots en Markdown respectant scrupuleusement ce plan :

        1.  THÈSE D'INVESTISSEMENT & RATING (/10)
           - Attribue une note globale sévère mais juste.
           - Recommandation explicite : ACHAT FORT, ACHAT, CONSERVER ou VENDRE.
        
        2.  ANALYSE FONDAMENTALE & OPÉRATIONNELLE (SWOT)
           -  Forces : Détaille le Moat, les marges et la visibilité des revenus.
           -  Faiblesses & Risques : Détaille la dette, les risques de concurrence ou de régulation.
        
        3.  VALORISATION ET ANALYSE COMPARATIVE
           - Analyse si le prix actuel est justifié par rapport à l'industrie ({valuation_context}).
           - Le P/E actuel de {metrics['pe']:.1f} est-il cohérent avec la croissance affichée ?

        4.  PLAN DE TRADING & STRATÉGIE (POINTS D'ENTRÉE/SORTIE)
           Tu dois proposer des niveaux précis basés sur le prix actuel de {metrics['price']:.2f} $ :
           -  ZONE D'ACHAT IDÉALE (ENTRY) : [Donne une fourchette précise]
           -  OBJECTIF DE PRIX (TAKE PROFIT) : [Objectif réaliste à 12 mois]
           -  STOP LOSS (INVALIDATION) : [Niveau critique de sortie de sécurité]

        Ton ton doit être analytique, institutionnel et convaincant. Rédige en français.
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content, None

    except Exception as e:
        return None, f"Erreur Groq: {e}"

# =========================================================
# 7) ASSET-BASED VALUE
# =========================================================
def compute_asset_based_value(bs: pd.DataFrame, shares: float) -> dict:
    if bs is None or bs.empty or shares <= 0:
        return {"nav_ps": 0.0, "tnav_ps": 0.0, "notes": "Balance sheet unavailable."}

    total_assets = get_item_safe(bs, ["TotalAssets", "Total Assets"])
    total_liab = get_item_safe(bs, ["TotalLiab", "Total Liab", "Total Liabilities"])

    goodwill = get_item_safe(bs, ["Goodwill"])
    intangibles = get_item_safe(bs, ["IntangibleAssets", "Intangible Assets"])

    equity = total_assets - total_liab
    t_equity = (total_assets - goodwill - intangibles) - total_liab

    nav_ps = equity / shares if shares > 0 else 0.0
    tnav_ps = t_equity / shares if shares > 0 else 0.0

    notes = f"Assets={total_assets/1e9:.2f}B, Liab={total_liab/1e9:.2f}B, Goodwill={goodwill/1e9:.2f}B, Intangibles={intangibles/1e9:.2f}B"
    return {"nav_ps": float(nav_ps), "tnav_ps": float(tnav_ps), "notes": notes}

# =========================================================
# 8) FINVIZ SCREENER (TICKERS ONLY)
# =========================================================
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
        except Exception:
            break
    return tickers[:max_tickers]

def _mc_ok(data: dict, min_mc: float) -> bool:
    mc = data.get("market_cap")
    if mc is None:
        return False
    try:
        return float(mc) >= float(min_mc)
    except Exception:
        return False

# =========================================================
# 9) SIDEBAR MODE
# =========================================================
mode = st.sidebar.radio(
    "Mode",
    ["Stock Analyzer", "AI Screener (Top Upside)"],
    index=0
)

# =========================================================
# MODE A) STOCK ANALYZER
# =========================================================
if mode == "Stock Analyzer":
    st.subheader("Search for a Company")

    choice = st.selectbox("Choose a popular stock:", TICKER_DB, index=2)
    ticker_final = "MSFT"
    if "Other" in choice:
        ticker_input = st.text_input("Or type ticker here (e.g. AMD, GOOGL)", value="").upper()
        if ticker_input:
            ticker_final = ticker_input
    elif "-" in choice:
        ticker_final = choice.split("-")[0].strip()

    st.caption(f"Analyzing: **{ticker_final}**")
    
    # Feature 5: Export Button
    if st.sidebar.button(" Download Report (TXT)"):
         pass

    st.divider()

    data = get_financial_data_secure(ticker_final)

    if data.get("error"):
        st.warning(f" Data fetch warning: {data['error']}")

    current_price = float(data.get("price", 0) or 0)
    if current_price <= 0:
        st.error("Price data unavailable from Yahoo Finance on this network/PC.")
        st.stop()

    bs = data.get("bs", pd.DataFrame())
    inc = data.get("inc", pd.DataFrame())
    cf = data.get("cf", pd.DataFrame())
    
    # Advanced Health Scores
    piotroski = calculate_piotroski_score(bs, inc, cf)
    
    # --- CORRECTION 3 : OVERRIDE MANUEL DES ACTIONS ---
    shares = float(data.get("shares_calc", 0) or 0)
    if shares <= 1:
        shares = float(data.get("shares_info", 0) or 0)
    
    # Option manuelle si Yahoo échoue
    st.sidebar.markdown("### 🔧 Corrections Données")
    manual_shares = st.sidebar.number_input("Manual Shares (Millions)", value=0.0, step=1.0)
    if manual_shares > 0:
        shares = manual_shares * 1_000_000
        st.sidebar.success(f"Using manual shares: {shares:,.0f}")

    if shares <= 1:
        shares = 1.0
        st.warning("⚠️ Warning: Share count unavailable. Utilisez la case à gauche pour corriger.")

    market_cap = shares * current_price
    
    altman_z = calculate_altman_z(bs, inc, market_cap)

    revenue_ttm = get_ttm_or_latest(inc, ["TotalRevenue", "Revenue"])
    cfo_ttm = get_ttm_or_latest(cf, ["OperatingCashFlow", "Operating Cash Flow"])
    capex_ttm = abs(get_item_safe(cf, ["CapitalExpenditure", "PurchaseOfPPE"]))
    fcf_ttm = cfo_ttm - capex_ttm

    cash = get_item_safe(bs, ["CashAndCashEquivalents", "Cash"])
    debt = get_item_safe(bs, ["LongTermDebt"]) + get_item_safe(bs, ["LeaseLiabilities", "TotalLiab"])

    net_income = get_ttm_or_latest(inc, ["NetIncome", "Net Income Common Stockholders"])
    eps_ttm = float(data.get("trailing_eps", 0) or 0)
    if eps_ttm <= 0 and shares > 0:
        eps_ttm = net_income / shares

    ps_current = market_cap / revenue_ttm if revenue_ttm > 0 else 0.0
    pe_current = current_price / eps_ttm if eps_ttm > 0 else 0.0
    pfcf_current = market_cap / fcf_ttm if fcf_ttm > 0 else 0.0

    cur_sales_gr = float(data.get("rev_growth", 0) or 0)
    cur_eps_gr = float(data.get("eps_growth", 0) or 0)

    bench_data = get_benchmark_data(ticker_final, data.get("sector", "Default"))

    with st.expander(f" Help: {bench_data['name']} vs {ticker_final}", expanded=True):
        st.write(f"**Peers:** {bench_data.get('peers', 'N/A')}")
        st.markdown("###  Sector / Peer Averages")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peer Sales Gr.", f"{bench_data['gr_sales']:.0f}%")
        c2.metric("Peer EPS Gr.", f"{bench_data.get('gr_eps', 0):.0f}%")
        c3.metric("Peer Target P/S", f"{bench_data['ps']}x")
        c4.metric("Peer Target P/E", f"{bench_data.get('pe', 20)}x")

        st.divider()

        st.markdown(f"###  {ticker_final} Current Metrics (Actual)")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Actual Sales Gr.", f"{cur_sales_gr*100:.1f}%", delta_color="off")
        c6.metric("Actual EPS Gr.", f"{cur_eps_gr*100:.1f}%", delta_color="off")
        c7.metric("Actual P/S", f"{ps_current:.1f}x", delta_color="off")
        c8.metric("Actual P/E", f"{pe_current:.1f}x", delta_color="off")

    with st.expander(" Edit Assumptions (Neutral)", expanded=False):
        st.markdown("##### 1. Growth (5y CAGR)")
        c1, c2, c3 = st.columns(3)
        gr_sales_input = c1.number_input("Sales Growth (%)", value=float(bench_data["gr_sales"]), step=0.5, format="%.1f")
        gr_fcf_input = c2.number_input("FCF Growth (%)", value=float(bench_data["gr_fcf"]), step=0.5, format="%.1f")
        gr_eps_input = c3.number_input("EPS Growth (%)", value=float(bench_data.get("gr_eps", 10.0)), step=0.5, format="%.1f")

        st.markdown("##### 2. Exit Multiples & Risk")
        c4, c5, c6 = st.columns(3)
        target_ps = c4.number_input("Target P/S (x)", value=float(bench_data["ps"]), step=0.5)
        target_pe = c5.number_input("Target P/E (x)", value=float(bench_data.get("pe", 20.0)), step=0.5)
        wacc_input = c6.number_input("WACC / Discount (%)", value=float(bench_data["wacc"]), step=0.5, format="%.1f")

    def run_scenario(factor_growth: float, factor_mult: float, risk_adj: float):
        return calculate_valuation(
            (gr_sales_input / 100.0) * factor_growth,
            (gr_fcf_input / 100.0) * factor_growth,
            (gr_eps_input / 100.0) * factor_growth,
            (wacc_input / 100.0) + risk_adj,
            target_ps * factor_mult,
            target_pe * factor_mult,
            revenue_ttm, fcf_ttm, eps_ttm, cash, debt, shares
        )

    bear_res = run_scenario(0.8, 0.8, 0.01)
    base_res = run_scenario(1.0, 1.0, 0.0)
    bull_res = run_scenario(1.2, 1.2, -0.01)

    price_df = fetch_price_history(ticker_final, "1y")
    tech_df = add_indicators(price_df) if not price_df.empty else pd.DataFrame()
    tech = bull_flag_score(tech_df) if not tech_df.empty else {"is_bull_flag": False, "score": 0.0, "notes": "No data."}

    metrics = {
        "ticker": ticker_final,
        "sector": data.get("sector", "Default"),
        "price": current_price,
        "sales_gr": cur_sales_gr,
        "eps_gr": cur_eps_gr,
        "ps": ps_current,
        "pe": pe_current,
        "pfcf": pfcf_current,
        "net_cash": cash - debt,
        "fcf_margin": (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0.0,
        "fcf_yield": (fcf_ttm / market_cap) * 100 if market_cap > 0 else 0.0,
        "rule_40": gr_sales_input + ((fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0.0),
    }
    scores = score_out_of_10(metrics, bench_data)

    report_content = f"""
    VALUATION MASTER PRO - REPORT for {ticker_final}
    Date: {datetime.now().strftime("%Y-%m-%d")}
    Price: {current_price} $
    
    --- VALUATION SCENARIOS ---
    DCF (Base): {base_res[0]:.2f} $
    Sales (Base): {base_res[1]:.2f} $
    Earnings (Base): {base_res[2]:.2f} $
    
    --- SCORES ---
    Overall: {scores['overall']}/10
    Financial Health: {scores['health']}/10
    Technical: {tech['score']}/10
    """
    st.sidebar.download_button(" Download Report (TXT)", report_content, file_name=f"{ticker_final}_Analysis.txt")

    tabs = st.tabs([
        " DCF",
        " Sales",
        " Earnings",
        " Assets",
        " Insiders",
        " Tech",
        " Scorecard",
        " Analyst",
        " AI Analyst",
    ])

    # --- DCF ---
    with tabs[0]:
        st.subheader(" Buy Price (DCF)")
        st.caption("Discounted Cash Flow basé sur FCF projeté (5 ans) + terminal (3%) - dette + cash.")

        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        delta = base_res[0] - current_price
        c2.metric("Intrinsic (Neutral)", f"{base_res[0]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")

        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric(" Bear", f"{bear_res[0]:.2f} $")
        c_base.metric(" Neutral", f"{base_res[0]:.2f} $")
        c_bull.metric(" Bull", f"{bull_res[0]:.2f} $")
        
        st.divider()
        st.markdown("#####  Reverse DCF")
        implied_g = solve_reverse_dcf(current_price, fcf_ttm, wacc_input/100.0, shares, cash, debt)
        st.metric("Marché price une croissance FCF de :", f"{implied_g*100:.1f}%")
        
        st.divider()
        st.markdown("#####  Sensitivity Matrix (WACC vs Growth)")
        sens_wacc = [wacc_input - 1, wacc_input - 0.5, wacc_input, wacc_input + 0.5, wacc_input + 1]
        sens_growth = [gr_fcf_input - 2, gr_fcf_input - 1, gr_fcf_input, gr_fcf_input + 1, gr_fcf_input + 2]
        
        res_matrix = []
        for w in sens_wacc:
            row_vals = []
            for g in sens_growth:
                val, _, _ = calculate_valuation(0, g/100.0, 0, w/100.0, 0, 0, revenue_ttm, fcf_ttm, 0, cash, debt, shares)
                row_vals.append(val)
            res_matrix.append(row_vals)
            
        df_sens = pd.DataFrame(res_matrix, index=[f"WACC {w:.1f}%" for w in sens_wacc], columns=[f"Gr {g:.1f}%" for g in sens_growth])
        st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f} $"))

    # --- SALES ---
    with tabs[1]:
        st.subheader(" Buy Price (Sales)")
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        delta = base_res[1] - current_price
        c2.metric("Intrinsic (Neutral)", f"{base_res[1]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")
        st.write("")
        display_relative_analysis(ps_current, float(bench_data.get("ps", 3.0)), "P/S", bench_data["name"])

    # --- EARNINGS ---
    with tabs[2]:
        st.subheader(" Buy Price (P/E)")
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        delta = base_res[2] - current_price
        c2.metric("Intrinsic (Neutral)", f"{base_res[2]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")
        st.write("")
        display_relative_analysis(pe_current, float(bench_data.get("pe", 20.0)), "P/E", bench_data["name"])

    # --- ASSET-BASED ---
    with tabs[3]:
        st.subheader(" Asset-Based Value")
        ab = compute_asset_based_value(bs, shares)
        nav_ps = ab["nav_ps"]
        tnav_ps = ab["tnav_ps"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"{current_price:.2f} $")
        c2.metric("NAV / Share", f"{nav_ps:.2f} $")
        c3.metric("Tangible NAV", f"{tnav_ps:.2f} $")
        st.caption(ab["notes"])
        
    # --- INSIDERS ---
    with tabs[4]:
        st.subheader(" Insider Trading")
        insiders = data.get("insiders")
        if insiders is not None and not insiders.empty:
            st.caption("Transactions récentes des dirigeants (CEO, CFO, etc.)")
            cols_to_show = [c for c in ["Start Date", "Insider", "Position", "Shares", "Value", "Text"] if c in insiders.columns]
            st.dataframe(insiders[cols_to_show].head(15), use_container_width=True)
        else:
             st.info("Aucune donnée d'initiés récente disponible.")

    # --- TECHNICAL ---
    with tabs[5]:
        st.subheader(" Technical Analysis")
        if tech_df.empty:
            st.warning("No price history.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Bull flag?", " Yes" if tech["is_bull_flag"] else " No")
            c2.metric("Score /10", f"{tech['score']:.1f}")
            c3.metric("RSI(14)", f"{tech_df['RSI14'].iloc[-1]:.1f}")
            plot_technical_chart(tech_df, ticker_final)

    # --- SCORECARD ---
    with tabs[6]:
        st.subheader(" Scorecard Pro")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("**Piotroski F-Score (0-9)**")
            if piotroski is not None:
                color_p = "green" if piotroski >= 7 else "red" if piotroski <= 3 else "orange"
                st.markdown(f":{color_p}[**{piotroski}/9**]")
            else:
                st.write("N/A")
        with col_h2:
            st.markdown("**Altman Z-Score**")
            if altman_z > 0:
                color_z = "green" if altman_z > 3.0 else "red" if altman_z < 1.8 else "orange"
                st.markdown(f":{color_z}[**{altman_z:.2f}**]")
            else:
                st.write("N/A")

        st.divider()
        tech_score = float(tech.get("score", 0.0))
        dividend_score = min(metrics["fcf_yield"] * 2.0, 10.0)
        fig = plot_radar(scores, tech_score=tech_score, dividend_score=dividend_score)
        st.pyplot(fig)
        
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Secteur", scores["sector"])
        c2.metric("Growth", scores["growth"])
        c3.metric("Health", scores["health"])
        c4.metric("Value", scores["valuation"])
        c5.metric("Global", scores["overall"])

    # --- ANALYST & IR ---
    with tabs[7]:
        st.subheader(" Analyst & IR")
        reco = data.get("reco_summary")
        if reco is not None and hasattr(reco, "empty") and not reco.empty:
            try:
                rec = reco.iloc[0]
                buy = (rec.get("strongBuy", 0) or 0) + (rec.get("buy", 0) or 0)
                hold = rec.get("hold", 0) or 0
                sell = (rec.get("sell", 0) or 0) + (rec.get("strongSell", 0) or 0)
                c1, c2, c3 = st.columns(3)
                c1.metric(" BUY", int(buy))
                c2.metric(" HOLD", int(hold))
                c3.metric(" SELL", int(sell))
            except:
                st.write("N/A")
        st.write("---")
        if data.get("ir_news"):
            for item in data["ir_news"]:
                st.markdown(f"• [{item.get('title')}]({item['link']})")
                st.caption(f"_{item.get('pubDate')}_")

    # --- AI ANALYST (COVERAGE) ---
    with tabs[8]:
        st.subheader(" AI Analyst (Coverage)")
        st.caption("Génération d'un rapport institutionnel complet avec stratégie de trading.")
        
        if st.button(" Générer le Rapport Complet"):
            if not tech_df.empty:
                st.markdown("####  Contexte Technique")
                plot_technical_chart(tech_df, ticker_final)
            
            with st.spinner("L'analyste rédige le rapport (c'est un peu plus long)..."):
                report, err = ai_analyst_report(metrics, bench_data, scores, tech, api_key)
            
            if err:
                st.error(err)
            else:
                st.markdown("---")
                st.markdown(report)
                st.success("Analyse terminée.")

# =========================================================
# MODE B) AI SCREENER
# =========================================================
else:
    st.subheader(" AI Screener (Top Upside)")
    st.caption("Finviz (tickers) → Yahoo/yfinance (données) → Intrinsic (DCF) → Top 5 par secteur.")

    min_market_cap = st.number_input("Minimum Market Cap (USD)", value=1_000_000_000, step=250_000_000)
    max_tickers_per_sector = st.slider("Max tickers per sector", 10, 80, 20, step=5)

    colA, colB = st.columns(2)
    scr_gr_fcf = colA.number_input("FCF Growth % (fallback)", value=15.0, step=0.5)
    scr_wacc = colB.number_input("WACC %", value=10.0, step=0.5)

    def intrinsic_dcf_quick(ticker: str) -> dict:
        data = get_financial_data_secure(ticker)
        price = float(data.get("price", 0) or 0)
        if price <= 0: return {"ticker": ticker, "ok": False}

        bs = data.get("bs", pd.DataFrame())
        inc = data.get("inc", pd.DataFrame())
        cf = data.get("cf", pd.DataFrame())

        shares = float(data.get("shares_calc", 0) or 0)
        if shares <= 1: shares = float(data.get("shares_info", 0) or 0)
        if shares <= 1: return {"ticker": ticker, "ok": False}
        if not _mc_ok(data, min_market_cap): return {"ticker": ticker, "ok": False}

        revenue = get_ttm_or_latest(inc, ["TotalRevenue", "Revenue"])
        cfo = get_ttm_or_latest(cf, ["OperatingCashFlow", "Operating Cash Flow"])
        capex = abs(get_item_safe(cf, ["CapitalExpenditure", "PurchaseOfPPE"]))
        fcf = cfo - capex
        cash = get_item_safe(bs, ["CashAndCashEquivalents", "Cash"])
        debt = get_item_safe(bs, ["LongTermDebt"]) + get_item_safe(bs, ["LeaseLiabilities", "TotalLiab"])

        gr_fcf = float(data.get("rev_growth", 0) or 0)
        if gr_fcf <= 0: gr_fcf = scr_gr_fcf / 100.0

        dcf, _, _ = calculate_valuation(
            0, gr_fcf, 0, scr_wacc / 100.0, 0, 0,
            revenue, fcf, 0, cash, debt, shares
        )

        if dcf <= 0: return {"ticker": ticker, "ok": False}
        upside = (dcf / price - 1.0) * 100.0
        
        return {
            "ticker": ticker, "sector": data.get("sector", "Unknown"),
            "price": price, "intrinsic": dcf, "upside_pct": upside, "ok": True
        }

    if st.button(" Run Screener"):
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        sectors = FINVIZ_SECTORS
        total_steps = len(sectors) * 2
        step = 0
        
        for (sector_name, sector_code) in sectors:
            for geo_name, geo_code in [("USA", "geo_usa"), ("Canada", "geo_canada")]:
                step += 1
                status.write(f"Fetching {sector_name} ({geo_name})...")
                tickers = finviz_fetch_tickers(sector_code, geo_code, max_tickers=max_tickers_per_sector*2)
                tickers = tickers[:max_tickers_per_sector]
                
                for t in tickers:
                    r = intrinsic_dcf_quick(t)
                    if r.get("ok"):
                        r["bucket"] = f"{sector_name} ({geo_name})"
                        results.append(r)
                progress.progress(min(step / total_steps, 1.0))
        
        progress.progress(1.0)
        status.write("Done.")
        
        if results:
            df = pd.DataFrame(results).sort_values("upside_pct", ascending=False)
            out = []
            for bucket, g in df.groupby("bucket"):
                out.append(g.head(5))
            out_df = pd.concat(out).reset_index(drop=True)
            
            st.dataframe(
                out_df[["bucket", "ticker", "price", "intrinsic", "upside_pct"]].style.format({
                    "price": "{:.2f}", "intrinsic": "{:.2f}", "upside_pct": "{:.1f}%"
                }), use_container_width=True
            )
        else:
            st.error("No results found.")
