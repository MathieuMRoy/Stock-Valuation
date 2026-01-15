import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")
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
    "--- POPULAR & SPACE ---",
    "MDA.TO - MDA Space (Canada)",
    "RKLB - Rocket Lab USA",
    "ASTS - AST SpaceMobile",
    "PLTR - Palantir Technologies",
    "NFLX - Netflix",
    "SPOT - Spotify",
    "DUOL - Duolingo",
    "UBER - Uber Technologies",
    "ABNB - Airbnb",
    "--- CANADA (TSX) ---",
    "RY.TO - Royal Bank (RBC)",
    "TD.TO - TD Bank",
    "SHOP.TO - Shopify (CAD)",
    "CNR.TO - CN Rail",
    "ENB.TO - Enbridge",
    "BCE.TO - BCE Inc. (Bell)",
    "DOL.TO - Dollarama",
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
        bs, inc, cf, info = stock.quarterly_balance_sheet, stock.quarterly_financials, stock.quarterly_cashflow, stock.info
        return bs, inc, cf, info
    except: return None, None, None, None

def get_ttm_flexible(df, keys_list):
    if df is None or df.empty: return 0
    for key in keys_list:
        for idx in df.index:
            if key.upper().replace(" ", "") in str(idx).upper().replace(" ", ""):
                row = df.loc[idx]
                total = sum([val for val in row if pd.api.types.is_number(val)][:4])
                if total != 0: return total
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

def get_real_shares(info):
    shares = info.get('impliedSharesOutstanding', 0)
    if shares == 0 and info.get('marketCap', 0) > 0: shares = info.get('marketCap') / info.get('currentPrice', 1)
    if shares == 0: shares = info.get('sharesOutstanding', 0)
    return shares

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

# --- RELATIVE ANALYSIS HELPER ---
def display_relative_analysis(current, benchmark, metric_name, group_name):
    if current <= 0:
        st.caption(f"Relative analysis unavailable (negative or zero {metric_name}).")
        return

    diff = ((current - benchmark) / benchmark) * 100
    if diff < -10:
        box = st.success; status = "Undervalued 🟢"; msg = f"discount of {abs(diff):.0f}%"
    elif diff > 10:
        box = st.error; status = "Overvalued 🔴"; msg = f"premium of {diff:.0f}%"
    else:
        box = st.warning; status = "Fair Value 🟡"; msg = "aligned"
    
    box(f"**🔍 Relative Analysis:** Current {metric_name} **{current:.1f}x** vs Peer/Sector **{benchmark}x**.\n\n"
        f"👉 **Verdict: {status}** ({msg} vs {group_name}).")

# --- 3. INTERFACE ---

st.subheader("Search for a Company")
col_search, col_manual = st.columns([2, 1])

# Smart Search
choice = st.selectbox(
    "Choose a popular stock:", 
    TICKER_DB, 
    index=2 
)

ticker_final = "MSFT" 

if "Other" in choice:
    ticker_input = st.text_input("Or type ticker here (e.g. AMD, GOOGL)", value="").upper()
    if ticker_input:
        ticker_final = ticker_input
else:
    if "-" in choice:
        ticker_final = choice.split("-")[0].strip()

st.caption(f"Analyzing: **{ticker_final}**")
st.divider()

# --- EXECUTION ---
if ticker_final:
    bs, inc, cf, info = get_financial_data(ticker_final)
    
    if bs is None or inc.empty:
        st.error(f"Data not found for {ticker_final}. Check ticker symbol.")
    else:
        # DATA PREP
        raw_sector = info.get('sector', 'Default')
        bench_data = get_benchmark_data(ticker_final, raw_sector)
        
        # DATA EXTRACT
        revenue_ttm = get_ttm_flexible(inc, ["TotalRevenue", "Total Revenue", "Revenue"])
        cfo_ttm = get_ttm_flexible(cf, ["OperatingCashFlow", "Operating Cash Flow"])
        capex_ttm = abs(get_ttm_flexible(cf, ["CapitalExpenditure", "Capital Expenditure"]))
        fcf_ttm = cfo_ttm - capex_ttm
        cash = get_cash_safe(bs); debt = get_debt_safe(bs)
        shares = get_real_shares(info) if get_real_shares(info) > 0 else 1
        current_price = info.get('currentPrice', 0); market_cap = shares * current_price
        
        eps_ttm = info.get('trailingEps')
        if eps_ttm is None:
            net_income = get_ttm_flexible(inc, ["NetIncome", "Net Income Common Stockholders"])
            eps_ttm = net_income / shares if shares > 0 else 0

        # RATIOS & GROWTH (POUR COMPARAISON)
        ps_current = market_cap / revenue_ttm if revenue_ttm > 0 else 0
        pe_current = current_price / eps_ttm if eps_ttm > 0 else 0
        pfcf_current = market_cap / fcf_ttm if fcf_ttm > 0 else 0
        
        cur_sales_gr = info.get('revenueGrowth', 0)
        cur_eps_gr = info.get('earningsGrowth', 0)
        if cur_sales_gr is None: cur_sales_gr = 0
        if cur_eps_gr is None: cur_eps_gr = 0

        # HELP / BENCHMARK INFO (AVEC COMPARAISON - SANS COULEUR)
        with st.expander(f"💡 Help: {bench_data['name']} vs {ticker_final}", expanded=True):
            if bench_data['source'] == "Comparables": st.write(f"**Peers:** {bench_data['peers']}")
            else: st.write(f"**Sector:** {raw_sector}")
            
            c1, c2, c3, c4 = st.columns(4)
            # Affichage: Benchmark (Gros) vs Réel (Petit, sans couleur)
            c1.metric("Sales Gr.", f"{bench_data['gr_sales']*100:.0f}%", delta=f"{cur_sales_gr*100:.1f}% Actual", delta_color="off")
            c2.metric("FCF/EPS Gr.", f"{bench_data['gr_fcf']*100:.0f}%", delta=f"{cur_eps_gr*100:.1f}% Actual", delta_color="off")
            c3.metric("Target P/S", f"{bench_data['ps']}x", delta=f"{ps_current:.1f}x Actual", delta_color="off")
            c4.metric("Target P/E", f"{bench_data.get('pe', 20)}x", delta=f"{pe_current:.1f}x Actual", delta_color="off")

        # INPUTS (CONVERTED TO PERCENTAGE INPUTS)
        with st.expander("⚙️ Edit Assumptions (Neutral)", expanded=False):
            st.markdown("##### 1. Growth (5y CAGR)")
            c1, c2, c3 = st.columns(3)
            # On multiplie par 100 pour afficher en % (ex: 0.12 -> 12.0)
            gr_sales_input = c1.number_input("Sales Growth (%)", value=bench_data['gr_sales']*100, step=0.5, format="%.1f")
            gr_fcf_input = c2.number_input("FCF Growth (%)", value=bench_data['gr_fcf']*100, step=0.5, format="%.1f")
            gr_eps_input = c3.number_input("EPS Growth (%)", value=bench_data.get('gr_eps', 0.10)*100, step=0.5, format="%.1f")
            
            st.markdown("##### 2. Exit Multiples & Risk")
            c4, c5, c6 = st.columns(3)
            target_ps = c4.number_input("Target P/S (x)", value=bench_data['ps'], step=0.5)
            target_pe = c5.number_input("Target P/E (x)", value=float(bench_data.get('pe', 20.0)), step=0.5)
            # WACC en %
            wacc_input = c6.number_input("WACC / Discount (%)", value=bench_data['wacc']*100, step=0.5, format="%.1f")

        # CALCULATE SCENARIOS (DIVIDE INPUTS BY 100 TO GET DECIMALS)
        def run_scenario(factor_growth, factor_mult, risk_adj):
            # On divise par 100 ici pour remettre en décimale pour le calcul
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
            st.caption("ℹ️ **Discounted Cash Flow:** Based on future Free Cash Flow discounted to today. The 'true' intrinsic value.")
            
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[0] - current_price
            
            # TOOLTIP AJOUTÉ ICI
            c2.metric(
                "Intrinsic (Neutral)", 
                f"{base_res[0]:.2f} $", 
                delta=f"{delta:.2f} $", 
                delta_color="normal",
                help=f"It means {ticker_final} is supposed to be this price based on the DCF (Cash Flow) model."
            )
            
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
            st.caption("ℹ️ **Price-to-Sales:** Values the company based on Revenue. Good for high-growth companies with no profits yet.")
            
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[1] - current_price
            
            # TOOLTIP AJOUTÉ ICI
            c2.metric(
                "Intrinsic (Neutral)", 
                f"{base_res[1]:.2f} $", 
                delta=f"{delta:.2f} $", 
                delta_color="normal",
                help=f"It means {ticker_final} is supposed to be this price based on the Price/Sales model."
            )
            
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
            st.caption("ℹ️ **Price-to-Earnings:** Values the company based on Profits. The standard for profitable companies.")
            
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"{current_price:.2f} $")
            delta = base_res[2] - current_price
            
            # TOOLTIP AJOUTÉ ICI
            c2.metric(
                "Intrinsic (Neutral)", 
                f"{base_res[2]:.2f} $", 
                delta=f"{delta:.2f} $", 
                delta_color="normal",
                help=f"It means {ticker_final} is supposed to be this price based on the Price/Earnings model."
            )
            
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
            # On utilise les inputs en % directement (pas besoin de x100)
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
                with st.expander("Interpretation Guide"):
                    st.write(f"**Calc:** Yield {fcf_yield:.1f}% + Growth {gr_eps_input:.1f}%")
                    st.markdown("""
                    * 🟢 **> 12%: Excellent** (Beats Market)
                    * 🟡 **8 - 12%: Fair** (Market Average)
                    * 🔴 **< 8%: Weak** (Underperformance)
                    """)
