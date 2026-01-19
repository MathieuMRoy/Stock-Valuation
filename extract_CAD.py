import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go # AJOUTÉ : Pour le graphique interactif
from datetime import datetime     # AJOUTÉ : Pour gérer les dates des news

# --- CONFIGURATION ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")

# --- SIDEBAR : BOUTON RESET ---
with st.sidebar:
    if st.button("🗑️ Reset Cache"):
        st.cache_data.clear()
        st.rerun()
    st.caption("À utiliser si les données semblent incorrectes ou manquantes.")

st.title("📱 Valuation Master")
st.caption("3 Models: Cash • Sales • Earnings")

# --- 0. DATA: SMART SEARCH DATABASE ---
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
    "MSTR - MicroStrategy"
]

# --- 1. DATA: SECTOR BENCHMARKS ---
PEER_GROUPS = {
    "SPACE_TECH": {
        "tickers": ["MDA", "RKLB", "ASTS", "LUNR", "PL", "SPIR", "SPCE", "PNG"],
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

def get_benchmark_data(ticker, sector_info):
    ticker_clean = ticker.upper().split(".")[0]
    for group_key, data in PEER_GROUPS.items():
        clean_list = [t.upper() for t in data['tickers']]
        if ticker_clean in clean_list:
            peers_list = [t for t in data['tickers'] if t.upper() != ticker_clean]
            peers_str = ", ".join(peers_list[:5])
            return {**data, "source": "Comparables", "peers": peers_str}
    bench = SECTOR_BENCHMARKS.get("Default")
    return {**bench, "source": "Sector", "name": sector_info or "General", "peers": "Sector Average"}

# --- 2. DATA FUNCTIONS ---
def get_growth_manual(df, keys):
    try:
        if df is None or df.empty: return 0
        row = None
        for key in keys:
            for idx in df.index:
                if key.upper().replace(" ", "") in str(idx).upper().replace(" ", ""):
                    row = df.loc[idx]
                    break
            if row is not None: break
        if row is None: return 0
        vals = [v for v in row if pd.api.types.is_number(v)]
        if len(vals) >= 5:
            current = vals[0]
            last_year = vals[4]
            if last_year != 0: return (current - last_year) / abs(last_year)
        elif len(vals) >= 2:
             return (vals[0] - vals[1]) / abs(vals[1])
        return 0
    except: return 0

@st.cache_data(ttl=3600)
def get_financial_data_secure(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # 1. PRICE & HISTORY (For Chart)
        try:
            current_price = stock.fast_info['last_price']
            market_cap = stock.fast_info['market_cap']
            shares_calc = market_cap / current_price if current_price > 0 else 0
            # 6 months history
            history = stock.history(period="6mo")
        except:
            history = pd.DataFrame()
            current_price = 0
            shares_calc = 0 

        # 2. FINANCIALS
        bs = stock.quarterly_balance_sheet
        inc = stock.quarterly_financials
        cf = stock.quarterly_cashflow
        
        # 3. NEWS (NOUVEAU)
        try:
            news = stock.news
        except:
            news = []

        if bs is None or bs.empty: return None

        # 4. INFO
        try:
            full_info = stock.info
            sector = full_info.get('sector', 'Default')
            
            rev_growth = full_info.get('revenueGrowth', 0)
            if rev_growth is None or rev_growth == 0:
                rev_growth = get_growth_manual(inc, ["TotalRevenue", "Revenue"])
                
            eps_growth = full_info.get('earningsGrowth', 0)
            if eps_growth is None or eps_growth == 0:
                eps_growth = get_growth_manual(inc, ["NetIncome", "Net Income Common Stockholders"])

            trailing_eps = full_info.get('trailingEps', None)
            shares_info = full_info.get('sharesOutstanding', 0)
        except:
            sector = "Default"
            rev_growth = 0
            eps_growth = 0
            trailing_eps = None
            shares_info = 0
        
        return {
            "bs": bs, "inc": inc, "cf": cf, "history": history, "news": news,
            "price": current_price, "shares_calc": shares_calc,
            "sector": sector, "rev_growth": rev_growth, 
            "eps_growth": eps_growth, "trailing_eps": trailing_eps,
            "shares_info": shares_info
        }
    except: return None

def get_ttm_flexible(df, keys_list):
    if df is None or df.empty: return 0
    for key in keys_list:
        for idx in df.index:
            if key.upper().replace(" ", "") in str(idx).upper().replace(" ", ""):
                row = df.loc[idx]
                vals = [v for v in row if pd.api.types.is_number(v)]
                return sum(vals[:4]) if len(vals) >= 1 else 0
    return 0

def get_item_safe(df, search_terms):
    if df is None or df.empty: return 0
    for term in search_terms:
        for idx in df.index:
            if term.upper() in str(idx).upper().replace(" ", ""):
                return df.loc[idx].iloc[0]
    return 0

# --- CALCULATION ENGINE ---
def calculate_valuation(gr_sales, gr_fcf, gr_eps, wacc_val, ps_target, pe_target, revenue, fcf, eps, cash, debt, shares):
    current_fcf = fcf
    fcf_projections = [current_fcf * (1 + gr_fcf)**(i+1) for i in range(5)]
    terminal_val = (fcf_projections[-1] * 1.03) / (wacc_val - 0.03)
    pv_fcf = sum([val / ((1 + wacc_val)**(i+1)) for i, val in enumerate(fcf_projections)])
    price_dcf = ((pv_fcf + (terminal_val / ((1 + wacc_val)**5))) + cash - debt) / shares
    
    price_sales = (((revenue * ((1 + gr_sales)**5)) * ps_target) / shares) / (1.10**5)
    eps_future = eps * ((1 + gr_eps)**5)
    price_earnings = (eps_future * pe_target) / (1.10**5)
    
    return price_dcf, price_sales, price_earnings

def display_relative_analysis(current, benchmark, metric_name, group_name):
    if current <= 0:
        st.caption(f"Relative analysis unavailable.")
        return
    diff = ((current - benchmark) / benchmark) * 100
    if diff < -10: box = st.success; status = "Undervalued 🟢"; msg = f"discount of {abs(diff):.0f}%"
    elif diff > 10: box = st.error; status = "Overvalued 🔴"; msg = f"premium of {diff:.0f}%"
    else: box = st.warning; status = "Fair Value 🟡"; msg = "aligned"
    box(f"**🔍 Relative Analysis:** Current {metric_name} **{current:.1f}x** vs Peer/Sector **{benchmark}x**.\n\n"
        f"👉 **Verdict: {status}** ({msg} vs {group_name}).")

# --- 3. INTERFACE ---

st.subheader("Search for a Company")
col_search, col_manual = st.columns([2, 1])

# Smart Search
choice = st.selectbox("Choose a popular stock:", TICKER_DB, index=2)
ticker_final = "MSFT" 
if "Other" in choice:
    ticker_input = st.text_input("Or type ticker here (e.g. AMD, GOOGL)", value="").upper()
    if ticker_input: ticker_final = ticker_input
elif "-" in choice:
    ticker_final = choice.split("-")[0].strip()

st.caption(f"Analyzing: **{ticker_final}**")
st.divider()

# --- EXECUTION ---
if ticker_final:
    data = get_financial_data_secure(ticker_final)
    
    if data is None:
        st.error(f"Data not found for {ticker_final}. Check ticker or try again later.")
    else:
        # UNPACK
        bs = data['bs']; inc = data['inc']; cf = data['cf']
        current_price = data['price']
        
        # SHARES LOGIC
        shares = data['shares_calc']
        if shares <= 1: shares = data['shares_info']
        if shares <= 1: 
            shares = 1 
            st.warning("⚠️ Warning: Share count unavailable.")

        market_cap = shares * current_price

        # METRICS
        revenue_ttm = get_ttm_flexible(inc, ["TotalRevenue", "Revenue"])
        cfo_ttm = get_ttm_flexible(cf, ["OperatingCashFlow", "Operating Cash Flow"])
        capex_ttm = abs(get_item_safe(cf, ["CapitalExpenditure", "PurchaseOfPPE"]))
        fcf_ttm = cfo_ttm - capex_ttm
        cash = get_item_safe(bs, ["CashAndCashEquivalents", "Cash"])
        debt = get_item_safe(bs, ["LongTermDebt"]) + get_item_safe(bs, ["LeaseLiabilities"])
        
        # EPS
        net_income = get_ttm_flexible(inc, ["NetIncome", "Net Income Common Stockholders"])
        eps_ttm = data['trailing_eps']
        if eps_ttm is None:
            eps_ttm = net_income / shares if shares > 0 else 0

        # RATIOS & GROWTH
        ps_current = market_cap / revenue_ttm if revenue_ttm > 0 else 0
        pe_current = current_price / eps_ttm if eps_ttm > 0 else 0
        pfcf_current = market_cap / fcf_ttm if fcf_ttm > 0 else 0
        
        cur_sales_gr = data['rev_growth'] if data['rev_growth'] else 0
        cur_eps_gr = data['eps_growth'] if data['eps_growth'] else 0

        # BENCHMARKS
        bench_data = get_benchmark_data(ticker_final, data['sector'])
        
        # --- DISPLAY HELP ---
        with st.expander(f"💡 Help: {bench_data['name']} vs {ticker_final}", expanded=True):
            st.write(f"**Peers:** {bench_data['peers']}")
            
            # 1. SECTOR AVERAGES
            st.markdown("### 🏢 Sector / Peer Averages")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Peer Sales Gr.", f"{bench_data['gr_sales']:.0f}%")
            c2.metric("Peer EPS Gr.", f"{bench_data['gr_eps']:.0f}%")
            c3.metric("Peer Target P/S", f"{bench_data['ps']}x")
            c4.metric("Peer Target P/E", f"{bench_data.get('pe', 20)}x")

            st.divider()

            # 2. ACTUAL COMPANY METRICS
            st.markdown(f"### 📍 {ticker_final} Current Metrics (Actual)")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Actual Sales Gr.", f"{cur_sales_gr*100:.1f}%", delta_color="off")
            c6.metric("Actual EPS Gr.", f"{cur_eps_gr*100:.1f}%", delta_color="off")
            c7.metric("Actual P/S", f"{ps_current:.1f}x", delta_color="off")
            c8.metric("Actual P/E", f"{pe_current:.1f}x", delta_color="off")

        # --- INPUTS ---
        with st.expander("⚙️ Edit Assumptions (Neutral)", expanded=False):
            st.markdown("##### 1. Growth (5y CAGR)")
            c1, c2, c3 = st.columns(3)
            gr_sales_input = c1.number_input("Sales Growth (%)", value=bench_data['gr_sales'], step=0.5, format="%.1f")
            gr_fcf_input = c2.number_input("FCF Growth (%)", value=bench_data['gr_fcf'], step=0.5, format="%.1f")
            gr_eps_input = c3.number_input("EPS Growth (%)", value=bench_data.get('gr_eps', 10.0), step=0.5, format="%.1f")
            
            st.markdown("##### 2. Exit Multiples & Risk")
            c4, c5, c6 = st.columns(3)
            target_ps = c4.number_input("Target P/S (x)", value=bench_data['ps'], step=0.5)
            target_pe = c5.number_input("Target P/E (x)", value=float(bench_data.get('pe', 20.0)), step=0.5)
            wacc_input = c6.number_input("WACC / Discount (%)", value=bench_data['wacc'], step=0.5, format="%.1f")

        # --- SCENARIOS ---
        def run_scenario(factor_growth, factor_mult, risk_adj):
            return calculate_valuation(
                (gr_sales_input/100.0) * factor_growth, 
                (gr_fcf_input/100.0) * factor_growth, 
                (gr_eps_input/100.0) * factor_growth, 
                (wacc_input/100.0) + risk_adj, 
                target_ps * factor_mult, 
                target_pe * factor_mult, 
                revenue_ttm, fcf_ttm, eps_ttm, cash, debt, shares
            )

        bear_res = run_scenario(0.8, 0.8, 0.01)
        base_res = run_scenario(1.0, 1.0, 0.0)
        bull_res = run_scenario(1.2, 1.2, -0.01)

        # ==========================================
        # SMART ADVISOR
        # ==========================================
        st.divider()
        if eps_ttm <= 0:
            advice_msg = f"💡 **Analyst Tip for {ticker_final}:** This company is **unprofitable** (Negative Earnings). The best metric to use is likely **Price-to-Sales (P/S)** in the Sales tab."
        else:
            advice_msg = f"💡 **Analyst Tip for {ticker_final}:** This company is **profitable**. We recommend using **Price-to-Earnings (P/E)** or the **DCF (Cash Flow)** model for the most accurate valuation."
            
        st.info(advice_msg)

        # ==========================================
        # RESULTS TABS
        # ==========================================
        # AJOUT DE L'ONGLET NEWS
        tabs = st.tabs(["💵 DCF (Cash)", "📈 Sales (P/S)", "💰 Earnings (P/E)", "📊 Scorecard", "📰 News & Context"])

        # --- 1. DCF ---
        with tabs[0]:
            st.subheader("🏷️ Buy Price (DCF)")
            st.caption("ℹ️ **Discounted Cash Flow:** Based on future Free Cash Flow discounted to today. The 'true' intrinsic value.")
            
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[0] - current_price
            c2.metric("Intrinsic (Neutral)", f"{base_res[0]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal", help=f"Fair value based on DCF model.")
            
            st.write("")
            display_relative_analysis(pfcf_current, bench_data.get('p_fcf', 20.0), "P/FCF", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[0]:.2f} $", delta=f"{bear_res[0]-current_price:.1f}")
            c_base.metric("🎯 Neutral", f"{base_res[0]:.2f} $", delta=f"{base_res[0]-current_price:.1f}")
            c_bull.metric("🐂 Bull", f"{bull_res[0]:.2f} $", delta=f"{bull_res[0]-current_price:.1f}")

        # --- 2. SALES ---
        with tabs[1]:
            st.subheader("🏷️ Buy Price (Sales)")
            st.caption("ℹ️ **Price-to-Sales:** Values the company based on Revenue. Good for high-growth companies with no profits yet.")
            
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[1] - current_price
            c2.metric("Intrinsic (Neutral)", f"{base_res[1]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal", help=f"Fair value based on Price/Sales.")
            
            st.write("")
            display_relative_analysis(ps_current, bench_data['ps'], "P/S", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[1]:.2f} $")
            c_base.metric("🎯 Neutral", f"{base_res[1]:.2f} $")
            c_bull.metric("🐂 Bull", f"{bull_res[1]:.2f} $")

        # --- 3. EARNINGS ---
        with tabs[2]:
            st.subheader("🏷️ Buy Price (P/E)")
            st.caption("ℹ️ **Price-to-Earnings:** Values the company based on Profits. The standard for profitable companies.")
            
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[2] - current_price
            c2.metric("Intrinsic (Neutral)", f"{base_res[2]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal", help=f"Fair value based on Price/Earnings.")
            
            st.write("")
            display_relative_analysis(pe_current, bench_data.get('pe', 20), "P/E", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[2]:.2f} $")
            c_base.metric("🎯 Neutral", f"{base_res[2]:.2f} $")
            c_bull.metric("🐂 Bull", f"{bull_res[2]:.2f} $")

        # --- 4. SCORECARD ---
        with tabs[3]:
            # Scores
            fcf_margin = (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0
            fcf_yield = (fcf_ttm / market_cap) * 100 if market_cap > 0 else 0
            rule_40 = gr_sales_input + fcf_margin
            total_return = gr_eps_input + fcf_yield

            st.subheader("Current Fundamentals")
            r1, r2, r3 = st.columns(3)
            r1.metric("P/E (TTM)", f"{pe_current:.1f}x")
            r2.metric("P/FCF", f"{pfcf_current:.1f}x")
            net_pos = cash - debt
            color = "red" if net_pos < 0 else "green"
            r3.markdown(f"**Net Cash:** :{color}[{net_pos/1e6:.0f} M$]")
            
            st.divider()
            
            col_score1, col_score2 = st.columns(2)
            with col_score1:
                st.markdown("#### 🚀 Growth")
                st.caption("Rule of 40")
                if rule_40 >= 40: st.success(f"✅ {rule_40:.1f}")
                else: st.warning(f"⚠️ {rule_40:.1f}")
                with st.expander("Interpretation Guide"):
                    st.write(f"**Calc:** Growth {gr_sales_input:.1f}% + Margin {fcf_margin:.1f}%")

            with col_score2:
                st.markdown("#### 🛡️ Stability")
                st.caption("Total Return")
                if total_return >= 12: st.success(f"✅ {total_return:.1f}%")
                else: st.warning(f"⚠️ {total_return:.1f}%")
                with st.expander("Interpretation Guide"):
                    st.write(f"**Calc:** Yield {fcf_yield:.1f}% + Growth {gr_eps_input:.1f}%")

        # --- 5. INTERACTIVE CHART (NEWS) ---
        with tabs[4]:
            st.subheader("📰 Recent Context")
            st.caption("Interact with the chart to see news events.")

            if 'history' in data and not data['history'].empty:
                # 1. Prepare Base Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['history'].index, 
                    y=data['history']['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#0068C9', width=2)
                ))

                # 2. Map News to Dates
                if 'news' in data and data['news']:
                    news_dates = []
                    news_prices = []
                    news_titles = []
                    
                    for article in data['news']:
                        try:
                            # Convert Unix timestamp to Date
                            ts = article.get('providerPublishTime', 0)
                            dt = datetime.utcfromtimestamp(ts).date()
                            title = article.get('title', 'News')
                            
                            # Find matching (or closest) date in history
                            # Ensure chart index is timezone-naive for comparison
                            hist_dates = data['history'].index.date
                            
                            if dt in hist_dates:
                                # Get price on that day
                                price_on_day = data['history'].loc[data['history'].index.date == dt]['Close'].iloc[0]
                                
                                news_dates.append(data['history'].loc[data['history'].index.date == dt].index[0])
                                news_prices.append(price_on_day)
                                # Add simple HTML bold for title
                                news_titles.append(f"<b>{title}</b><br><i>Click list below for link</i>")
                        except:
                            continue

                    # 3. Add News Markers
                    if news_dates:
                        fig.add_trace(go.Scatter(
                            x=news_dates,
                            y=news_prices,
                            mode='markers',
                            name='News Event',
                            marker=dict(color='orange', size=10, symbol='circle'),
                            hovertext=news_titles,
                            hoverinfo="text"
                        ))

                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("Chart data unavailable.")

            st.divider()

            # News List Below
            if 'news' in data and data['news']:
                for article in data['news'][:7]:
                    with st.container():
                        st.markdown(f"**[{article['title']}]({article['link']})**")
                        try:
                            ts = article.get('providerPublishTime', 0)
                            date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                        except: date_str = "Recent"
                        st.caption(f"📅 {date_str} | 📢 {article.get('publisher', 'Unknown')}")
                        st.write("---")
            else:
                st.info("No recent news found.")
