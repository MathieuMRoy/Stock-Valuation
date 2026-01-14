import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")
st.title("📱 Valuation Master")
st.caption("Scénarios • Thèses • Ratios Avancés")

# --- 0. DATA ---
PEER_GROUPS = {
    "SEMICONDUCTORS": {"tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM"], "gr_sales": 0.18, "gr_fcf": 0.20, "ps": 8.0, "wacc": 0.10, "name": "Semi-conducteurs & AI"},
    "BIG_TECH": {"tickers": ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META"], "gr_sales": 0.12, "gr_fcf": 0.15, "ps": 6.5, "wacc": 0.09, "name": "Big Tech / GAFAM"},
    "SAAS_CLOUD": {"tickers": ["CRM", "ADBE", "SNOW", "DDOG", "PLTR", "NOW", "SHOP", "DUOL"], "gr_sales": 0.20, "gr_fcf": 0.22, "ps": 10.0, "wacc": 0.10, "name": "Logiciel SaaS & Cloud"},
    "STREAMING": {"tickers": ["NFLX", "DIS", "WBD", "PARA", "SPOT"], "gr_sales": 0.10, "gr_fcf": 0.15, "ps": 4.0, "wacc": 0.09, "name": "Streaming & Média"},
    "EV_AUTO": {"tickers": ["TSLA", "RIVN", "LCID", "BYD", "F", "GM"], "gr_sales": 0.15, "gr_fcf": 0.12, "ps": 3.0, "wacc": 0.11, "name": "Véhicules Électriques"},
    "BANKS_CA": {"tickers": ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"], "gr_sales": 0.04, "gr_fcf": 0.05, "ps": 2.5, "wacc": 0.08, "name": "Banques Canadiennes"}
}

SECTOR_BENCHMARKS = {
    "Technology": {"gr_sales": 0.12, "gr_fcf": 0.15, "ps": 5.0, "wacc": 0.095},
    "Communication Services": {"gr_sales": 0.08, "gr_fcf": 0.10, "ps": 3.5, "wacc": 0.09},
    "Consumer Cyclical": {"gr_sales": 0.06, "gr_fcf": 0.08, "ps": 2.0, "wacc": 0.10},
    "Healthcare": {"gr_sales": 0.05, "gr_fcf": 0.06, "ps": 4.0, "wacc": 0.08},
    "Financial Services": {"gr_sales": 0.05, "gr_fcf": 0.05, "ps": 2.5, "wacc": 0.09},
    "Energy": {"gr_sales": 0.03, "gr_fcf": 0.05, "ps": 1.5, "wacc": 0.10},
    "Industrials": {"gr_sales": 0.04, "gr_fcf": 0.05, "ps": 1.8, "wacc": 0.09},
    "Utilities": {"gr_sales": 0.03, "gr_fcf": 0.03, "ps": 2.0, "wacc": 0.065},
    "Default": {"gr_sales": 0.07, "gr_fcf": 0.08, "ps": 2.5, "wacc": 0.09}
}

def get_benchmark_data(ticker, sector_info):
    ticker_clean = ticker.upper().replace(".TO", "")
    for group_key, data in PEER_GROUPS.items():
        if any(t in ticker_clean for t in data['tickers']):
            return {"gr_sales": data['gr_sales'], "gr_fcf": data['gr_fcf'], "ps": data['ps'], "wacc": data['wacc'], "source": "Comparables", "name": data['name'], "peers": ", ".join(data['tickers'][:4])}
    bench = SECTOR_BENCHMARKS.get(sector_info, SECTOR_BENCHMARKS["Default"])
    return {"gr_sales": bench['gr_sales'], "gr_fcf": bench['gr_fcf'], "ps": bench['ps'], "wacc": bench['wacc'], "source": "Secteur", "name": sector_info, "peers": "Moyenne du secteur"}

# --- 1. FONCTIONS DATA ---
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

def calculate_valuation(gr_sales, gr_fcf, wacc_val, ps_target, revenue, fcf, cash, debt, shares):
    # DCF
    current_fcf = fcf
    fcf_projections = [current_fcf * (1 + gr_fcf)**(i+1) for i in range(5)]
    terminal_val = (fcf_projections[-1] * 1.03) / (wacc_val - 0.03)
    pv_fcf = sum([val / ((1 + wacc_val)**(i+1)) for i, val in enumerate(fcf_projections)])
    price_dcf = ((pv_fcf + (terminal_val / ((1 + wacc_val)**5))) + cash - debt) / shares
    # Ventes
    price_sales = (((revenue * ((1 + gr_sales)**5)) * ps_target) / shares) / (1.10**5)
    return price_dcf, price_sales

# --- 2. INTERFACE ---
ticker = st.text_input("Symbole (Ticker)", value="NFLX").upper()

if ticker:
    bs, inc, cf, info = get_financial_data(ticker)
    
    if bs is None or inc.empty:
        st.error("Données introuvables.")
    else:
        # CONTEXTE
        raw_sector = info.get('sector', 'Default')
        bench_data = get_benchmark_data(ticker, raw_sector)
        
        with st.expander(f"💡 Aide : {bench_data['name']}", expanded=True):
            if bench_data['source'] == "Comparables": st.write(f"**Pairs :** {bench_data['peers']}")
            else: st.write(f"**Secteur :** {raw_sector}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Croiss. Ventes", f"{bench_data['gr_sales']*100:.0f}%")
            c2.metric("Croiss. FCF", f"{bench_data['gr_fcf']*100:.0f}%")
            c3.metric("P/S Cible", f"{bench_data['ps']}x")

        # INPUTS
        with st.expander("⚙️ Modifier Hypothèses (Neutral)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                gr_sales_input = st.number_input("Croiss. Ventes", value=bench_data['gr_sales'], step=0.01)
                gr_fcf_input = st.number_input("Croiss. FCF", value=bench_data['gr_fcf'], step=0.01)
            with c2:
                wacc = st.number_input("WACC", value=bench_data['wacc'], step=0.005)
                target_ps = st.number_input("P/S Cible", value=bench_data['ps'], step=0.5)
            
            # Tableau Spread
            st.divider()
            st.caption("Scénarios calculés (+/- 20%) :")
            st.table(pd.DataFrame({
                "Métrique": ["Croissance Ventes", "Croissance FCF", "P/S Cible"],
                "🐻 Bear": [f"{gr_sales_input*0.8:.1%}", f"{gr_fcf_input*0.8:.1%}", f"{target_ps*0.8:.1f}x"],
                "🎯 Neutral": [f"{gr_sales_input:.1%}", f"{gr_fcf_input:.1%}", f"{target_ps:.1f}x"],
                "🐂 Bull": [f"{gr_sales_input*1.2:.1%}", f"{gr_fcf_input*1.2:.1%}", f"{target_ps*1.2:.1f}x"]
            }))

        # CALCULS
        revenue_ttm = get_ttm_flexible(inc, ["TotalRevenue", "Total Revenue", "Revenue"])
        cfo_ttm = get_ttm_flexible(cf, ["OperatingCashFlow", "Operating Cash Flow"])
        capex_ttm = abs(get_ttm_flexible(cf, ["CapitalExpenditure", "Capital Expenditure"]))
        fcf_ttm = cfo_ttm - capex_ttm
        cash = get_cash_safe(bs); debt = get_debt_safe(bs)
        shares = get_real_shares(info) if get_real_shares(info) > 0 else 1
        current_price = info.get('currentPrice', 0); market_cap = shares * current_price

        # SCÉNARIOS
        bear_dcf, bear_sales = calculate_valuation(gr_sales_input*0.8, gr_fcf_input*0.8, wacc+0.01, target_ps*0.8, revenue_ttm, fcf_ttm, cash, debt, shares)
        base_dcf, base_sales = calculate_valuation(gr_sales_input, gr_fcf_input, wacc, target_ps, revenue_ttm, fcf_ttm, cash, debt, shares)
        bull_dcf, bull_sales = calculate_valuation(gr_sales_input*1.2, gr_fcf_input*1.2, wacc-0.01, target_ps*1.2, revenue_ttm, fcf_ttm, cash, debt, shares)

        # RATIOS
        eps = info.get('trailingEps', 0)
        pe_ratio = current_price / eps if eps > 0 else 0
        pfcf_ratio = market_cap / fcf_ttm if fcf_ttm > 0 else 0
        
        # RESULTATS
        st.divider()
        st.subheader("🏷️ Prix à Payer")
        c1, c2 = st.columns(2)
        c1.metric("Prix Actuel", f"{current_price:.2f} $")
        c2.metric("Intrinsèque (Neutre)", f"{base_dcf:.2f} $", delta=f"{base_dcf-current_price:.2f} $")

        # ONGLETS
        tab_dcf, tab_sales, tab_ratios = st.tabs(["💵 DCF (Cash)", "📈 Ventes (Growth)", "📊 Scorecard"])

        with tab_dcf:
            c1, c2, c3 = st.columns(3)
            c1.metric("🐻 Bear", f"{bear_dcf:.2f} $", delta=f"{bear_dcf-current_price:.1f}")
            c2.metric("🎯 Neutral", f"{base_dcf:.2f} $", delta=f"{base_dcf-current_price:.1f}")
            c3.metric("🐂 Bull", f"{bull_dcf:.2f} $", delta=f"{bull_dcf-current_price:.1f}")
            st.info("ℹ️ **DCF :** Pour les entreprises qui génèrent déjà des profits (Cash Flow).")

        with tab_sales:
            c1, c2, c3 = st.columns(3)
            c1.metric("🐻 Bear", f"{bear_sales:.2f} $", delta=f"{bear_sales-current_price:.1f}")
            c2.metric("🎯 Neutral", f"{base_sales:.2f} $", delta=f"{base_sales-current_price:.1f}")
            c3.metric("🐂 Bull", f"{bull_sales:.2f} $", delta=f"{bull_sales-current_price:.1f}")
            st.info("ℹ️ **Ventes :** Pour les entreprises en croissance qui ne font pas encore de profits.")

        with tab_ratios:
            # 1. RATIOS CLASSIQUES
            st.subheader("Fondamentaux")
            r1, r2, r3 = st.columns(3)
            r1.metric("P/E", f"{pe_ratio:.1f}x" if pe_ratio > 0 else "N/A")
            r2.metric("P/FCF", f"{pfcf_ratio:.1f}x" if pfcf_ratio > 0 else "N/A")
            net_pos = cash - debt
            color_net = "red" if net_pos < 0 else "green"
            r3.markdown(f"**Net Cash:** :{color_net}[{net_pos/1e6:.0f} M$]")

            st.divider()

            # 2. SCORING AVANCÉ (Rule of 40 vs Total Return)
            st.subheader("🏆 Scoring de Qualité")
            
            # Calculs
            fcf_margin = (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0
            fcf_yield = (fcf_ttm / market_cap) * 100 if market_cap > 0 else 0
            
            # Score 1 : Rule of 40 (Growth)
            rule_40 = (gr_sales_input * 100) + fcf_margin
            
            # Score 2 : Total Return (Stable)
            total_return = (gr_fcf_input * 100) + fcf_yield

            col_score1, col_score2 = st.columns(2)
            
            with col_score1:
                st.markdown("#### 🚀 Pour la Croissance")
                st.caption("Règle des 40 (SaaS / Tech)")
                if rule_40 >= 40: st.success(f"✅ Score : {rule_40:.1f}")
                elif rule_40 >= 20: st.warning(f"⚠️ Score : {rule_40:.1f}")
                else: st.error(f"❌ Score : {rule_40:.1f}")
                with st.expander("Explication"):
                    st.write(f"Croissance ({gr_sales_input*100:.1f}%) + Marge FCF ({fcf_margin:.1f}%)")
                    st.write("**Cible : > 40**")

            with col_score2:
                st.markdown("#### 🛡️ Pour la Stabilité")
                st.caption("Rendement Total (Dividende/FCF + Croissance)")
                if total_return >= 12: st.success(f"✅ Score : {total_return:.1f}%")
                elif total_return >= 8: st.warning(f"⚠️ Score : {total_return:.1f}%")
                else: st.error(f"❌ Score : {total_return:.1f}%")
                with st.expander("Explication"):
                    st.write(f"Rendement FCF ({fcf_yield:.1f}%) + Croissance ({gr_fcf_input*100:.1f}%)")
                    st.write("**Cible : > 10-12%**")
