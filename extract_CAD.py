import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")

# Bouton de secours discret dans la barre latérale
with st.sidebar:
    if st.button("🗑️ Reset Cache"):
        st.cache_data.clear()
        st.rerun()

st.title("📱 Valuation Master")
st.caption("3 Models: Cash • Sales • Earnings")

# --- 1. DATA: SECTOR BENCHMARKS ---
PEER_GROUPS = {
    "SPACE_TECH": {
        "tickers": ["MDA", "RKLB", "ASTS", "LUNR", "PL", "SPIR", "SPCE"],
        "gr_sales": 0.20, "gr_fcf": 0.25, "gr_eps": 0.25, "ps": 6.0, "pe": 40.0, "p_fcf": 35.0, "wacc": 0.11,
        "name": "Space Tech & Satellites"
    },
    "SEMICONDUCTORS": {
        "tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM"],
        "gr_sales": 0.18, "gr_fcf": 0.20, "gr_eps": 0.20, "ps": 8.0, "pe": 35.0, "p_fcf": 30.0, "wacc": 0.10,
        "name": "Semiconductors & AI"
    },
    "BIG_TECH": {
        "tickers": ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META"],
        "gr_sales": 0.12, "gr_fcf": 0.15, "gr_eps": 0.15, "ps": 6.5, "pe": 25.0, "p_fcf": 28.0, "wacc": 0.09,
        "name": "Big Tech / GAFAM"
    },
    "SAAS_CLOUD": {
        "tickers": ["CRM", "ADBE", "SNOW", "DDOG", "PLTR", "NOW", "SHOP", "DUOL"],
        "gr_sales": 0.20, "gr_fcf": 0.22, "gr_eps": 0.25, "ps": 10.0, "pe": 40.0, "p_fcf": 35.0, "wacc": 0.10,
        "name": "SaaS & Cloud Software"
    },
    "STREAMING": {
        "tickers": ["NFLX", "DIS", "WBD", "PARA", "SPOT"],
        "gr_sales": 0.10, "gr_fcf": 0.15, "gr_eps": 0.18, "ps": 4.0, "pe": 25.0, "p_fcf": 20.0, "wacc": 0.09,
        "name": "Streaming & Media"
    },
    "EV_AUTO": {
        "tickers": ["TSLA", "RIVN", "LCID", "BYD", "F", "GM"],
        "gr_sales": 0.15, "gr_fcf": 0.12, "gr_eps": 0.15, "ps": 3.0, "pe": 30.0, "p_fcf": 25.0, "wacc": 0.11,
        "name": "Electric Vehicles"
    },
    "BANKS_CA": {
        "tickers": ["RY", "TD", "BMO", "BNS", "CM", "NA"],
        "gr_sales": 0.04, "gr_fcf": 0.05, "gr_eps": 0.06, "ps": 2.5, "pe": 11.0, "p_fcf": 12.0, "wacc": 0.08,
        "name": "Canadian Banks"
    }
}

SECTOR_BENCHMARKS = {
    "Technology": {"gr_sales": 0.12, "gr_fcf": 0.15, "gr_eps": 0.15, "ps": 5.0, "pe": 25.0, "p_fcf": 25.0, "wacc": 0.095},
    "Communication Services": {"gr_sales": 0.08, "gr_fcf": 0.10, "gr_eps": 0.12, "ps": 3.5, "pe": 20.0, "p_fcf": 18.0, "wacc": 0.09},
    "Consumer Cyclical": {"gr_sales": 0.06, "gr_fcf": 0.08, "gr_eps": 0.10, "ps": 2.0, "pe": 18.0, "p_fcf": 15.0, "wacc": 0.10},
    "Healthcare": {"gr_sales": 0.05, "gr_fcf": 0.06, "gr_eps": 0.08, "ps": 4.0, "pe": 22.0, "p_fcf": 20.0, "wacc": 0.08},
    "Financial Services": {"gr_sales": 0.05, "gr_fcf": 0.05, "gr_eps": 0.06, "ps": 2.5, "pe": 12.0, "p_fcf": 12.0, "wacc": 0.09},
    "Energy": {"gr_sales": 0.03, "gr_fcf": 0.05, "gr_eps": 0.05, "ps": 1.5, "pe": 10.0, "p_fcf": 8.0, "wacc": 0.10},
    "Default": {"gr_sales": 0.07, "gr_fcf": 0.08, "gr_eps": 0.08, "ps": 2.5, "pe": 15.0, "p_fcf": 15.0, "wacc": 0.09}
}

def get_benchmark_data(ticker, sector_info):
    ticker_clean = ticker.upper().replace(".TO", "").replace("-B", "").replace(".UN", "")
    for group_key, data in PEER_GROUPS.items():
        if any(t in ticker_clean for t in data['tickers']):
            return {**data, "source": "Comparables", "peers": ", ".join(data['tickers'][:4])}
    bench = SECTOR_BENCHMARKS.get(sector_info, SECTOR_BENCHMARKS["Default"])
    return {**bench, "source": "Sector", "name": sector_info, "peers": "Sector Average"}

# --- 2. DATA FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # On essaie de récupérer le prix rapidement pour éviter les bugs
        try:
            current_price = stock.fast_info['last_price']
        except:
            hist = stock.history(period="1d")
            if hist.empty: return None
            current_price = hist['Close'].iloc[-1]

        bs = stock.quarterly_balance_sheet
        inc = stock.quarterly_financials
        cf = stock.quarterly_cashflow
        
        # Protection : Si bs est vide, c'est que Yahoo bloque
        if bs is None or bs.empty: return None

        # On récupère info, mais on ne plante pas si ça échoue
        try:
            info = stock.info
        except:
            info = {}
        
        # On injecte le prix manuellement pour être sûr
        info['currentPrice'] = current_price
        
        return {"bs": bs, "inc": inc, "cf": cf, "info": info, "stock_obj": stock}
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

def get_cash_safe(df):
    if df is None or df.empty: return 0
    keys = ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments", "Cash"]
    for key in keys:
        for idx in df.index:
            if key.upper().replace(" ","") in str(idx).upper().replace(" ",""): return df.loc[idx].iloc[0]
    return 0

def get_debt_safe(df):
    if df is None or df.empty: return 0
    lt_debt, lease = 0, 0
    for idx in df.index:
        if "LONGTERMDEBT" in str(idx).upper().replace(" ", ""): lt_debt = df.loc[idx].iloc[0]; break
    for idx in df.index:
        if "LEASE" in str(idx).upper() and "LIABILITIES" in str(idx).upper(): lease = df.loc[idx].iloc[0]; break
    return lt_debt + lease

# --- FIX DU BUG DES 3 MILLIARDS ---
def get_real_shares(info, stock_obj):
    # 1. Essai via Info (Méthode classique)
    shares = info.get('impliedSharesOutstanding', 0)
    if shares == 0: shares = info.get('sharesOutstanding', 0)
    
    if shares > 1000: # Si on a un chiffre cohérent
        return shares
        
    # 2. Essai via Fast Info (Méthode de secours)
    try:
        mcap = stock_obj.fast_info['market_cap']
        price = stock_obj.fast_info['last_price']
        if price > 0:
            return mcap / price
    except:
        pass
        
    return 1 # Si tout échoue

# --- CALCULATION ENGINE ---
def calculate_valuation(gr_sales, gr_fcf, gr_eps, wacc_val, ps_target, pe_target, revenue, fcf, eps, cash, debt, shares):
    # DCF
    current_fcf = fcf
    fcf_projections = [current_fcf * (1 + gr_fcf)**(i+1) for i in range(5)]
    terminal_val = (fcf_projections[-1] * 1.03) / (wacc_val - 0.03)
    pv_fcf = sum([val / ((1 + wacc_val)**(i+1)) for i, val in enumerate(fcf_projections)])
    price_dcf = ((pv_fcf + (terminal_val / ((1 + wacc_val)**5))) + cash - debt) / shares
    
    # Sales & Earnings
    price_sales = (((revenue * ((1 + gr_sales)**5)) * ps_target) / shares) / (1.10**5)
    eps_future = eps * ((1 + gr_eps)**5)
    price_earnings = (eps_future * pe_target) / (1.10**5)
    
    return price_dcf, price_sales, price_earnings

def display_relative_analysis(current, benchmark, metric_name, group_name):
    if current <= 0:
        st.caption(f"Relative analysis unavailable (negative or zero {metric_name}).")
        return
    diff = ((current - benchmark) / benchmark) * 100
    if diff < -10: box = st.success; status = "Undervalued 🟢"; msg = f"discount of {abs(diff):.0f}%"
    elif diff > 10: box = st.error; status = "Overvalued 🔴"; msg = f"premium of {diff:.0f}%"
    else: box = st.warning; status = "Fair Value 🟡"; msg = "aligned"
    box(f"**🔍 Relative Analysis:** Current {metric_name} **{current:.1f}x** vs Peer/Sector **{benchmark}x**.\n\n"
        f"👉 **Verdict: {status}** ({msg} vs {group_name}).")

# --- 3. INTERFACE ---

st.subheader("Search for a Company")
ticker_input = st.text_input("Symbol (Ticker)", help="Type any ticker here (e.g. VLE.TO, AMD, GOOGL)").upper().strip()

# --- EXECUTION ---
if not ticker_input:
    st.info("Please enter a symbol to start.")
else:
    ticker_final = ticker_input
    st.caption(f"Analyzing: **{ticker_final}**")
    st.divider()
    
    data = get_financial_data(ticker_final)
    
    if data is None:
        st.error(f"Data not found for {ticker_final}. Check ticker symbol.")
    else:
        bs = data['bs']; inc = data['inc']; cf = data['cf']; info = data['info']; stock_obj = data['stock_obj']
        
        # 1. EXTRACT DATA FIRST
        revenue_ttm = get_ttm_flexible(inc, ["TotalRevenue", "Total Revenue", "Revenue"])
        cfo_ttm = get_ttm_flexible(cf, ["OperatingCashFlow", "Operating Cash Flow"])
        capex_ttm = abs(get_ttm_flexible(cf, ["CapitalExpenditure", "Capital Expenditure"]))
        fcf_ttm = cfo_ttm - capex_ttm
        cash = get_cash_safe(bs); debt = get_debt_safe(bs)
        
        # FIX SHARES HERE
        shares = get_real_shares(info, stock_obj)
        
        current_price = info.get('currentPrice', 0); market_cap = shares * current_price
        
        eps_ttm = info.get('trailingEps')
        if eps_ttm is None:
            net_income = get_ttm_flexible(inc, ["NetIncome", "Net Income Common Stockholders"])
            eps_ttm = net_income / shares if shares > 0 else 0

        # Current Growth & Ratios
        curr_sales_gr = info.get('revenueGrowth', 0)
        curr_eps_gr = info.get('earningsGrowth', 0)
        ps_current = market_cap / revenue_ttm if revenue_ttm > 0 else 0
        pe_current = current_price / eps_ttm if eps_ttm > 0 else 0
        pfcf_current = market_cap / fcf_ttm if fcf_ttm > 0 else 0

        # 2. DATA PREP & BENCHMARKS
        raw_sector = info.get('sector', 'Default')
        bench_data = get_benchmark_data(ticker_final, raw_sector)
        
        # 3. HELP / BENCHMARK INFO
        with st.expander(f"💡 Help: {bench_data['name']} vs {ticker_final}", expanded=True):
            st.write(f"**Peers:** {bench_data['peers']}")
            
            st.markdown("### 🏢 Industry Averages (Benchmarks)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sales Gr.", f"{bench_data['gr_sales']*100:.0f}%")
            c2.metric("EPS Gr.", f"{bench_data['gr_eps']*100:.0f}%")
            c3.metric("Target P/S", f"{bench_data['ps']}x")
            c4.metric("Target P/E", f"{bench_data.get('pe', 20)}x")

            st.markdown(f"### 📍 Current {ticker_final} Metrics")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Sales Gr. (YoY)", f"{curr_sales_gr*100:.1f}%")
            c6.metric("EPS Gr. (YoY)", f"{curr_eps_gr*100:.1f}%")
            c7.metric("Current P/S", f"{ps_current:.1f}x")
            c8.metric("Current P/E", f"{pe_current:.1f}x")

        # 4. INPUTS
        with st.expander("⚙️ Edit Assumptions (Neutral)", expanded=False):
            st.markdown("##### 1. Growth (5y CAGR)")
            c1, c2, c3 = st.columns(3)
            gr_sales_input = c1.number_input("Sales Growth (%)", value=bench_data['gr_sales']*100, step=0.5, format="%.1f")
            gr_fcf_input = c2.number_input("FCF Growth (%)", value=bench_data['gr_fcf']*100, step=0.5, format="%.1f")
            gr_eps_input = c3.number_input("EPS Growth (%)", value=bench_data.get('gr_eps', 0.10)*100, step=0.5, format="%.1f")
            
            st.markdown("##### 2. Exit Multiples & Risk")
            c4, c5, c6 = st.columns(3)
            target_ps = c4.number_input("Target P/S (x)", value=bench_data['ps'], step=0.5)
            target_pe = c5.number_input("Target P/E (x)", value=float(bench_data.get('pe', 20.0)), step=0.5)
            wacc_input = c6.number_input("WACC (%)", value=bench_data['wacc']*100, step=0.5, format="%.1f")

        # 5. CALCULATE SCENARIOS
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
        # RESULTS TABS
        # ==========================================
        st.divider()
        tabs = st.tabs(["💵 DCF (Cash)", "📈 Sales (P/S)", "💰 Earnings (P/E)", "📊 Scorecard"])

        # --- 1. DCF ---
        with tabs[0]:
            st.subheader("🏷️ Buy Price (DCF)")
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[0] - current_price
            c2.metric("Intrinsic (Neutral)", f"{base_res[0]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")
            
            st.write("")
            display_relative_analysis(pfcf_current, bench_data.get('p_fcf', 20.0), "P/FCF", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[0]:.2f} $", delta=f"{bear_res[0]-current_price:.1f}")
            c_base.metric("🎯 Neutral", f"{base_res[0]:.2f} $", delta=f"{base_res[0]-current_price:.1f}")
            c_bull.metric("🐂 Bull", f"{bull_res[0]:.2f} $", delta=f"{bull_res[0]-current_price:.1f}")

            st.markdown("##### 📝 Investment Theses")
            st.error(f"**🐻 Bear (-20%):** FCF Growth slows to **{gr_fcf_input*0.8:.1f}%**. Market doubts cash flow sustainability.")
            st.info(f"**🎯 Neutral:** Base case. FCF Growth **{gr_fcf_input:.1f}%**, WACC **{wacc_input:.1f}%**.")
            st.success(f"**🐂 Bull (+20%):** Perfect execution. FCF Growth accelerates to **{gr_fcf_input*1.2:.1f}%**.")

        # --- 2. SALES ---
        with tabs[1]:
            st.subheader("🏷️ Buy Price (Sales)")
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[1] - current_price
            c2.metric("Intrinsic (Neutral)", f"{base_res[1]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")
            
            st.write("")
            display_relative_analysis(ps_current, bench_data['ps'], "P/S", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[1]:.2f} $")
            c_base.metric("🎯 Neutral", f"{base_res[1]:.2f} $")
            c_bull.metric("🐂 Bull", f"{bull_res[1]:.2f} $")

            st.markdown("##### 📝 Investment Theses")
            st.error(f"**🐻 Bear:** Multiple compression to **{target_ps*0.8:.1f}x** sales.")
            st.info(f"**🎯 Neutral:** Maintains historical multiple of **{target_ps:.1f}x**.")
            st.success(f"**🐂 Bull:** Market euphoria, multiple expands to **{target_ps*1.2:.1f}x**.")

        # --- 3. EARNINGS ---
        with tabs[2]:
            st.subheader("🏷️ Buy Price (P/E)")
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[2] - current_price
            c2.metric("Intrinsic (Neutral)", f"{base_res[2]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")
            
            st.write("")
            display_relative_analysis(pe_current, bench_data.get('pe', 20), "P/E", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[2]:.2f} $")
            c_base.metric("🎯 Neutral", f"{base_res[2]:.2f} $")
            c_bull.metric("🐂 Bull", f"{bull_res[2]:.2f} $")

            st.markdown("##### 📝 Investment Theses")
            st.error(f"**🐻 Bear:** EPS Growth **{gr_eps_input*0.8:.1f}%**, P/E drops to **{target_pe*0.8:.1f}x**.")
            st.info(f"**🎯 Neutral:** EPS Growth **{gr_eps_input:.1f}%**, Standard P/E of **{target_pe:.1f}x**.")
            st.success(f"**🐂 Bull:** Margin expansion (**{gr_eps_input*1.2:.1f}%**), Premium P/E of **{target_pe*1.2:.1f}x**.")

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
                with st.expander("Guide"):
                    st.write(f"**Calc:** Growth {gr_sales_input:.1f}% + Margin {fcf_margin:.1f}%")
                    st.markdown("""
                    * 🟢 **> 40: Excellent** (Efficient Hyper-growth)
                    * 🟡 **20 - 40: Average** (Watch closely)
                    * 🔴 **< 20: Weak** (Inefficient)
                    """)

            with col_score2:
                st.markdown("#### 🛡️ Stability")
                st.caption("Total Return")
                if total_return >= 12: st.success(f"✅ {total_return:.1f}%")
                else: st.warning(f"⚠️ {total_return:.1f}%")
                with st.expander("Guide"):
                    st.write(f"**Calc:** Yield {fcf_yield:.1f}% + Growth {gr_eps_input:.1f}%")
                    st.markdown("""
                    * 🟢 **> 12%: Excellent** (Beats Market)
                    * 🟡 **8 - 12%: Fair** (Market Average)
                    * 🔴 **< 8%: Weak** (Underperformance)
                    """)
