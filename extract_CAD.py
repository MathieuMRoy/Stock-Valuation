import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")

# --- SIDEBAR: CACHE RESET ---
with st.sidebar:
    st.header("🔧 Settings")
    if st.button("🗑️ Reset Cache (Fix Errors)"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Use this if data seems stuck or outdated.")

st.title("📱 Valuation Master")
st.caption("3 Models: Cash • Sales • Earnings")

# --- 0. DATA: SMART SEARCH DATABASE ---
TICKER_DB = [
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
    "SPACE_TECH": {"tickers": ["MDA", "RKLB", "ASTS", "LUNR", "PL"], "gr_sales": 0.20, "gr_fcf": 0.25, "gr_eps": 0.25, "ps": 6.0, "pe": 40.0, "p_fcf": 35.0, "wacc": 0.11, "name": "Space Tech"},
    "SEMICONDUCTORS": {"tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO"], "gr_sales": 0.18, "gr_fcf": 0.20, "gr_eps": 0.20, "ps": 8.0, "pe": 35.0, "p_fcf": 30.0, "wacc": 0.10, "name": "Semiconductors"},
    "BIG_TECH": {"tickers": ["AAPL", "MSFT", "GOOG", "AMZN", "META"], "gr_sales": 0.12, "gr_fcf": 0.15, "gr_eps": 0.15, "ps": 6.5, "pe": 25.0, "p_fcf": 28.0, "wacc": 0.09, "name": "Big Tech"},
    "SAAS_CLOUD": {"tickers": ["CRM", "ADBE", "SNOW", "DDOG", "PLTR"], "gr_sales": 0.20, "gr_fcf": 0.22, "gr_eps": 0.25, "ps": 10.0, "pe": 40.0, "p_fcf": 35.0, "wacc": 0.10, "name": "SaaS Cloud"},
    "STREAMING": {"tickers": ["NFLX", "DIS", "WBD", "SPOT"], "gr_sales": 0.10, "gr_fcf": 0.15, "gr_eps": 0.18, "ps": 4.0, "pe": 25.0, "p_fcf": 20.0, "wacc": 0.09, "name": "Streaming"},
    "EV_AUTO": {"tickers": ["TSLA", "RIVN", "BYD", "F", "GM"], "gr_sales": 0.15, "gr_fcf": 0.12, "gr_eps": 0.15, "ps": 3.0, "pe": 30.0, "p_fcf": 25.0, "wacc": 0.11, "name": "EV & Auto"},
    "BANKS_CA": {"tickers": ["RY", "TD", "BMO", "BNS", "CM"], "gr_sales": 0.04, "gr_fcf": 0.05, "gr_eps": 0.06, "ps": 2.5, "pe": 11.0, "p_fcf": 12.0, "wacc": 0.08, "name": "Canadian Banks"}
}

SECTOR_BENCHMARKS = {
    "Technology": {"gr_sales": 0.12, "gr_fcf": 0.15, "gr_eps": 0.15, "ps": 5.0, "pe": 25.0, "p_fcf": 25.0, "wacc": 0.095},
    "Default": {"gr_sales": 0.07, "gr_fcf": 0.08, "gr_eps": 0.08, "ps": 2.5, "pe": 15.0, "p_fcf": 15.0, "wacc": 0.09}
}

def get_benchmark_data(ticker, sector_info):
    ticker_clean = ticker.upper().replace(".TO", "").replace("-B", "").replace(".UN", "")
    for _, data in PEER_GROUPS.items():
        if any(t in ticker_clean for t in data['tickers']):
            return {**data, "source": "Comparables", "peers": ", ".join(data['tickers'][:4])}
    bench = SECTOR_BENCHMARKS.get("Default") # Simplification to avoid key errors
    return {**bench, "source": "Sector", "name": sector_info or "General", "peers": "Sector Avg"}

# --- 2. DATA FUNCTIONS (OPTIMIZED) ---
@st.cache_data(ttl=1800, show_spinner=False) # Cache for 30 mins only
def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Get Price FAST (No big download)
        try:
            current_price = stock.fast_info['last_price']
        except:
            # Fallback if fast_info fails
            hist = stock.history(period="1d")
            if hist.empty: return None
            current_price = hist['Close'].iloc[-1]

        # 2. Get Financials
        bs = stock.quarterly_balance_sheet
        inc = stock.quarterly_financials
        cf = stock.quarterly_cashflow
        
        if bs is None or bs.empty: return None

        # 3. Get Info (Only if needed, wrapped safely)
        try:
            # We construct a minimal info dict to avoid massive download if possible
            # But yfinance often forces it. We just catch errors.
            info = stock.info
        except:
            info = {}

        # Add price manually to info dict for compatibility
        info['currentPrice'] = current_price
        
        return {
            'bs': bs, 
            'inc': inc, 
            'cf': cf, 
            'info': info,
            'price': current_price
        }
    except Exception:
        return None

def get_ttm_flexible(df, keys_list):
    if df is None or df.empty: return 0
    for key in keys_list:
        for idx in df.index:
            if key.upper().replace(" ", "") in str(idx).upper().replace(" ", ""):
                row = df.loc[idx]
                # Sum last 4 quarters (TTM)
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

def get_real_shares(info_dict):
    shares = info_dict.get('impliedSharesOutstanding', 0)
    if shares == 0: shares = info_dict.get('sharesOutstanding', 0)
    return shares if shares > 0 else 1

# --- CALCULATION ---
def calculate_valuation(gr_sales, gr_fcf, gr_eps, wacc_val, ps_target, pe_target, revenue, fcf, eps, cash, debt, shares):
    # DCF
    current_fcf = fcf
    fcf_projections = [current_fcf * (1 + gr_fcf)**(i+1) for i in range(5)]
    # Terminal Value
    terminal_val = (fcf_projections[-1] * 1.03) / (wacc_val - 0.03)
    # Discount
    pv_fcf = sum([val / ((1 + wacc_val)**(i+1)) for i, val in enumerate(fcf_projections)])
    pv_tv = terminal_val / ((1 + wacc_val)**5)
    price_dcf = (pv_fcf + pv_tv + cash - debt) / shares
    
    # Sales & Earnings
    price_sales = (((revenue * ((1 + gr_sales)**5)) * ps_target) / shares) / (1.10**5)
    eps_future = eps * ((1 + gr_eps)**5)
    price_earnings = (eps_future * pe_target) / (1.10**5)
    
    return price_dcf, price_sales, price_earnings

def display_relative_analysis(current, benchmark, metric_name, group_name):
    if current <= 0:
        st.caption(f"Relative analysis unavailable.")
        return
    diff = ((current - benchmark) / benchmark) * 100
    if diff < -10: box = st.success; status = "Undervalued 🟢"; msg = f"-{abs(diff):.0f}%"
    elif diff > 10: box = st.error; status = "Overvalued 🔴"; msg = f"+{diff:.0f}%"
    else: box = st.warning; status = "Fair Value 🟡"; msg = "aligned"
    box(f"**Relative:** {current:.1f}x vs Peer {benchmark}x ({msg})")

# --- 3. INTERFACE ---

# Initialize Session State
if 'ticker_search' not in st.session_state:
    st.session_state.ticker_search = "MSFT"

def update_input_from_list():
    selection = st.session_state.preset_select
    if "-" in selection:
        st.session_state.ticker_search = selection.split("-")[0].strip()

st.subheader("Search for a Company")

# Dropdown
search_options = ["👇 Pick from list (or type below)..."] + TICKER_DB
st.selectbox("Popular Ideas:", options=search_options, key="preset_select", index=0, on_change=update_input_from_list, label_visibility="collapsed")

# Text Input
col_input, col_info = st.columns([3, 1])
with col_input:
    ticker_final = st.text_input("Symbol (Ticker)", key="ticker_search", help="Type any ticker here (e.g. VLE.TO, AMD)").upper().strip()

# --- EXECUTION ---
if not ticker_final:
    st.info("Please enter a symbol.")
else:
    # DATA LOADING WITH SPINNER
    with st.spinner(f"Fetching data for {ticker_final}..."):
        data = get_financial_data(ticker_final)
    
    if data is None:
        st.error(f"❌ Data not found for **{ticker_final}**.")
        st.caption("Try clicking 'Reset Cache' in the sidebar if you are sure the ticker exists.")
    else:
        # Unpack data
        bs, inc, cf, info = data['bs'], data['inc'], data['cf'], data['info']
        current_price = data['price']

        # EXTRACT METRICS
        revenue_ttm = get_ttm_flexible(inc, ["TotalRevenue", "Revenue"])
        cfo_ttm = get_ttm_flexible(cf, ["OperatingCashFlow"])
        capex_ttm = abs(get_item_safe(cf, ["CapitalExpenditure", "PurchaseOfPPE"]))
        fcf_ttm = cfo_ttm - capex_ttm
        cash = get_item_safe(bs, ["CashAndCashEquivalents", "Cash"])
        debt = get_item_safe(bs, ["LongTermDebt"]) + get_item_safe(bs, ["LeaseLiabilities"])
        shares = get_real_shares(info)
        market_cap = shares * current_price
        
        # EPS TTM
        net_income = get_ttm_flexible(inc, ["NetIncome", "NetIncomeCommonStockholders"])
        eps_ttm = info.get('trailingEps') or (net_income / shares if shares > 0 else 0)

        # RATIOS
        ps_current = market_cap / revenue_ttm if revenue_ttm > 0 else 0
        pe_current = current_price / eps_ttm if eps_ttm > 0 else 0
        pfcf_current = market_cap / fcf_ttm if fcf_ttm > 0 else 0
        
        # GROWTH RATES (Safe get)
        curr_sales_gr = info.get('revenueGrowth', 0)
        curr_eps_gr = info.get('earningsGrowth', 0)

        # BENCHMARKS
        raw_sector = info.get('sector', 'Default')
        bench_data = get_benchmark_data(ticker_final, raw_sector)
        
        # --- DISPLAY HELP ---
        with st.expander(f"💡 {bench_data['name']} vs {ticker_final}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Peer Sales Gr.", f"{bench_data['gr_sales']*100:.0f}%", delta=f"{curr_sales_gr*100:.1f}% Actual")
            c2.metric("Peer EPS Gr.", f"{bench_data['gr_eps']*100:.0f}%", delta=f"{curr_eps_gr*100:.1f}% Actual")
            c3.metric("Peer P/S", f"{bench_data['ps']}x", delta=f"{ps_current:.1f}x Actual", delta_color="inverse")
            c4.metric("Peer P/E", f"{bench_data.get('pe', 20)}x", delta=f"{pe_current:.1f}x Actual", delta_color="inverse")

        # --- INPUTS ---
        with st.expander("⚙️ Assumptions (Neutral)", expanded=False):
            c1, c2, c3 = st.columns(3)
            gr_sales_input = c1.number_input("Sales Gr (%)", value=bench_data['gr_sales']*100, step=0.5, format="%.1f")
            gr_fcf_input = c2.number_input("FCF Gr (%)", value=bench_data['gr_fcf']*100, step=0.5, format="%.1f")
            gr_eps_input = c3.number_input("EPS Gr (%)", value=bench_data.get('gr_eps', 0.10)*100, step=0.5, format="%.1f")
            
            c4, c5, c6 = st.columns(3)
            target_ps = c4.number_input("Target P/S", value=bench_data['ps'], step=0.5)
            target_pe = c5.number_input("Target P/E", value=float(bench_data.get('pe', 20.0)), step=0.5)
            wacc_input = c6.number_input("WACC (%)", value=bench_data['wacc']*100, step=0.5, format="%.1f")

        # --- RUN SCENARIOS ---
        def run_scenario(f_gr, f_mul, r_adj):
            return calculate_valuation(
                (gr_sales_input/100)*f_gr, (gr_fcf_input/100)*f_gr, (gr_eps_input/100)*f_gr, 
                (wacc_input/100)+r_adj, target_ps*f_mul, target_pe*f_mul, 
                revenue_ttm, fcf_ttm, eps_ttm, cash, debt, shares
            )

        bear = run_scenario(0.8, 0.8, 0.01)
        base = run_scenario(1.0, 1.0, 0.0)
        bull = run_scenario(1.2, 1.2, -0.01)

        # --- RESULTS ---
        st.divider()
        tabs = st.tabs(["💵 DCF", "📈 Sales", "💰 P/E", "📊 Scorecard"])

        with tabs[0]:
            st.subheader("🏷️ DCF Value")
            c1, c2 = st.columns(2)
            c1.metric("Price", f"{current_price:.2f} $")
            c2.metric("Intrinsic", f"{base[0]:.2f} $", delta=f"{base[0]-current_price:.2f} $")
            display_relative_analysis(pfcf_current, bench_data.get('p_fcf', 20), "P/FCF", "Peer")
            st.divider()
            b1, b2, b3 = st.columns(3)
            b1.metric("Bear", f"{bear[0]:.2f} $")
            b2.metric("Neutral", f"{base[0]:.2f} $")
            b3.metric("Bull", f"{bull[0]:.2f} $")
            st.info(f"**Neutral Thesis:** FCF grows at **{gr_fcf_input:.1f}%** with **{wacc_input:.1f}%** WACC.")

        with tabs[1]:
            st.subheader("🏷️ Sales Value")
            c1, c2 = st.columns(2)
            c1.metric("Price", f"{current_price:.2f} $")
            c2.metric("Intrinsic", f"{base[1]:.2f} $", delta=f"{base[1]-current_price:.2f} $")
            display_relative_analysis(ps_current, bench_data['ps'], "P/S", "Peer")
            st.divider()
            b1, b2, b3 = st.columns(3)
            b1.metric("Bear", f"{bear[1]:.2f} $")
            b2.metric("Neutral", f"{base[1]:.2f} $")
            b3.metric("Bull", f"{bull[1]:.2f} $")
            st.info(f"**Neutral Thesis:** Sales grow at **{gr_sales_input:.1f}%**, exit at **{target_ps}x** P/S.")

        with tabs[2]:
            st.subheader("🏷️ Earnings Value")
            c1, c2 = st.columns(2)
            c1.metric("Price", f"{current_price:.2f} $")
            c2.metric("Intrinsic", f"{base[2]:.2f} $", delta=f"{base[2]-current_price:.2f} $")
            display_relative_analysis(pe_current, bench_data.get('pe', 20), "P/E", "Peer")
            st.divider()
            b1, b2, b3 = st.columns(3)
            b1.metric("Bear", f"{bear[2]:.2f} $")
            b2.metric("Neutral", f"{base[2]:.2f} $")
            b3.metric("Bull", f"{bull[2]:.2f} $")
            st.info(f"**Neutral Thesis:** EPS grows at **{gr_eps_input:.1f}%**, exit at **{target_pe}x** P/E.")

        with tabs[3]:
            fcf_yld = (fcf_ttm/market_cap)*100 if market_cap > 0 else 0
            fcf_marg = (fcf_ttm/revenue_ttm)*100 if revenue_ttm > 0 else 0
            r40 = gr_sales_input + fcf_marg
            tot_ret = fcf_yld + gr_eps_input
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("#### 🚀 Rule of 40")
                if r40 >= 40: st.success(f"✅ {r40:.1f}")
                else: st.warning(f"⚠️ {r40:.1f}")
                st.caption(f"Growth {gr_sales_input:.1f}% + Margin {fcf_marg:.1f}%")
            with c2:
                st.write("#### 🛡️ Total Return")
                if tot_ret >= 12: st.success(f"✅ {tot_ret:.1f}%")
                else: st.warning(f"⚠️ {tot_ret:.1f}%")
                st.caption(f"Yield {fcf_yld:.1f}% + Growth {gr_eps_input:.1f}%")
