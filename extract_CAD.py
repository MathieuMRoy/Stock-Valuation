import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")
st.title("📱 Valuation Master")
st.caption("Architecture Méthode > Scénario")

# --- 0. BENCHMARKS ---
SECTOR_BENCHMARKS = {
    "Technology": {"gr_sales": 0.15, "gr_fcf": 0.18, "ps": 6.0, "wacc": 0.095, "desc": "Tech / Logiciel"},
    "Communication Services": {"gr_sales": 0.10, "gr_fcf": 0.15, "ps": 4.0, "wacc": 0.09, "desc": "Média / Streaming"},
    "Consumer Cyclical": {"gr_sales": 0.08, "gr_fcf": 0.10, "ps": 2.5, "wacc": 0.10, "desc": "Conso / Auto"},
    "Healthcare": {"gr_sales": 0.06, "gr_fcf": 0.07, "ps": 4.0, "wacc": 0.08, "desc": "Santé / Pharma"},
    "Financial Services": {"gr_sales": 0.05, "gr_fcf": 0.05, "ps": 2.0, "wacc": 0.10, "desc": "Finance"},
    "Energy": {"gr_sales": 0.03, "gr_fcf": 0.05, "ps": 1.5, "wacc": 0.11, "desc": "Énergie"},
    "Industrials": {"gr_sales": 0.04, "gr_fcf": 0.05, "ps": 1.8, "wacc": 0.09, "desc": "Industrie"},
    "Utilities": {"gr_sales": 0.03, "gr_fcf": 0.03, "ps": 2.0, "wacc": 0.065, "desc": "Services Publics"},
    "Default": {"gr_sales": 0.08, "gr_fcf": 0.10, "ps": 3.0, "wacc": 0.09, "desc": "Moyenne"}
}

# --- 1. FONCTIONS DATA ---
@st.cache_data(ttl=3600)
def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        bs = stock.quarterly_balance_sheet
        inc = stock.quarterly_financials
        cf = stock.quarterly_cashflow
        info = stock.info
        return bs, inc, cf, info
    except:
        return None, None, None, None

def get_ttm_flexible(df, keys_list):
    if df is None or df.empty: return 0
    for key in keys_list:
        for idx in df.index:
            idx_clean = str(idx).upper().replace(" ", "")
            key_clean = key.upper().replace(" ", "")
            if key_clean in idx_clean:
                row = df.loc[idx]
                total = 0
                count = 0
                for val in row:
                    if pd.api.types.is_number(val):
                        total += val; count += 1
                    if count == 4: break
                if total != 0: return total
    return 0

def get_cash_safe(df):
    if df is None or df.empty: return 0
    keys = ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments", "Cash"]
    for key in keys:
        for idx in df.index:
            if key.upper().replace(" ","") in str(idx).upper().replace(" ",""):
                return df.loc[idx].iloc[0]
    return 0

def get_debt_safe(df):
    if df is None or df.empty: return 0
    lt_debt, lease = 0, 0
    for idx in df.index:
        s = str(idx).upper().replace(" ", "")
        if "LONGTERMDEBT" in s: lt_debt = df.loc[idx].iloc[0]; break
    for idx in df.index:
        s = str(idx).upper().replace(" ", "")
        if "LEASE" in s and "LIABILITIES" in s: lease = df.loc[idx].iloc[0]; break
    return lt_debt + lease

def get_real_shares(info):
    shares = 0
    if 'impliedSharesOutstanding' in info and info['impliedSharesOutstanding']:
        shares = info['impliedSharesOutstanding']
    if shares == 0:
        mcap = info.get('marketCap', 0); price = info.get('currentPrice', 0)
        if mcap > 0 and price > 0: shares = mcap / price
    if shares == 0: shares = info.get('sharesOutstanding', 0)
    return shares

def calculate_valuation(gr_sales, gr_fcf, wacc_val, ps_target, revenue, fcf, cash, debt, shares):
    # DCF
    current_fcf = fcf
    fcf_projections = []
    for i in range(5):
        current_fcf = current_fcf * (1 + gr_fcf)
        fcf_projections.append(current_fcf)
    
    terminal_val = (fcf_projections[-1] * (1 + 0.03)) / (wacc_val - 0.03)
    pv_fcf = sum([val / ((1 + wacc_val)**(i+1)) for i, val in enumerate(fcf_projections)])
    pv_terminal = terminal_val / ((1 + wacc_val)**5)
    equity_val = (pv_fcf + pv_terminal) + cash - debt
    price_dcf = equity_val / shares

    # Ventes
    rev_future = revenue * ((1 + gr_sales)**5)
    mcap_future = rev_future * ps_target
    price_sales = (mcap_future / shares) / ((1 + 0.10)**5)
    return price_dcf, price_sales

# --- 2. INTERFACE & LOGIQUE ---
ticker = st.text_input("Symbole (Ticker)", value="NFLX").upper()

if ticker:
    bs, inc, cf, info = get_financial_data(ticker)
    
    if bs is None or inc.empty:
        st.error("Données introuvables.")
    else:
        # SECTEUR
        sector = info.get('sector', 'Default')
        bench = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["Default"])
        st.info(f"🏢 **Secteur : {sector}**")

        # INPUTS
        with st.expander(f"⚙️ Modifier Hypothèses (Neutral)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                gr_sales_input = st.number_input("Croiss. Ventes (5 ans)", value=bench['gr_sales'], step=0.01, format="%.2f")
                gr_fcf_input = st.number_input("Croiss. FCF (5 ans)", value=bench['gr_fcf'], step=0.01, format="%.2f")
            with col2:
                wacc = st.number_input("CPMC (WACC)", value=bench['wacc'], step=0.005, format="%.3f")
                target_ps = st.number_input("Ratio P/S Cible", value=bench['ps'], step=0.5)

        # DATA
        revenue_ttm = get_ttm_flexible(inc, ["TotalRevenue", "Total Revenue", "Revenue"])
        cfo_ttm = get_ttm_flexible(cf, ["OperatingCashFlow", "Operating Cash Flow"])
        capex_ttm = abs(get_ttm_flexible(cf, ["CapitalExpenditure", "Capital Expenditure", "Purchase of PPE"]))
        fcf_ttm = cfo_ttm - capex_ttm
        cash = get_cash_safe(bs)
        debt = get_debt_safe(bs)
        shares = get_real_shares(info)
        if shares == 0: shares = 1
        current_price = info.get('currentPrice', 0)
        market_cap = shares * current_price

        # CALCUL SCENARIOS
        bear_dcf, bear_sales = calculate_valuation(gr_sales_input*0.8, gr_fcf_input*0.8, wacc+0.01, target_ps*0.8, revenue_ttm, fcf_ttm, cash, debt, shares)
        base_dcf, base_sales = calculate_valuation(gr_sales_input, gr_fcf_input, wacc, target_ps, revenue_ttm, fcf_ttm, cash, debt, shares)
        bull_dcf, bull_sales = calculate_valuation(gr_sales_input*1.2, gr_fcf_input*1.2, wacc-0.01, target_ps*1.2, revenue_ttm, fcf_ttm, cash, debt, shares)

        # Ratios (pour onglet 3)
        eps = info.get('trailingEps', 0)
        pe_ratio = current_price / eps if eps > 0 else 0
        pfcf_ratio = market_cap / fcf_ttm if fcf_ttm > 0 else 0
        
        # EBITDA
        ebitda_ttm = get_ttm_flexible(inc, ["EBITDA", "NormalizedEBITDA"])
        if ebitda_ttm == 0:
            op_inc = get_ttm_flexible(inc, ["OperatingIncome", "Operating Income", "EBIT"])
            da = get_ttm_flexible(cf, ["Depreciation", "DepreciationAndAmortization"])
            ebitda_ttm = op_inc + da
        ev = market_cap + debt - cash
        ev_ebitda = ev / ebitda_ttm if ebitda_ttm > 0 else 0

        # ==========================================
        # RESULTATS
        # ==========================================
        st.divider()
        
        # En-tête : Prix du Marché vs Valeur Intrinsèque Neutre
        st.subheader("🏷️ Prix à Payer Aujourd'hui")
        c_m1, c_m2 = st.columns(2)
        c_m1.metric("Prix Actuel", f"{current_price:.2f} $")
        delta_val = base_dcf - current_price
        c_m2.metric("Valeur Intrinsèque (Neutre)", f"{base_dcf:.2f} $", delta=f"{delta_val:.2f} $", delta_color="normal")

        st.write("")
        
        # --- STRUCTURE PAR MÉTHODE ---
        tab_dcf, tab_sales, tab_ratios = st.tabs(["💵 DCF (Cash)", "📈 Ventes (Growth)", "📊 Ratios"])

        # --- ONGLET 1 : DCF ---
        with tab_dcf:
            st.markdown("### Valorisation par Flux de Trésorerie")
            st.caption("Combien valent les bénéfices futurs (Cash Flow) ramenés à aujourd'hui ?")
            
            # Les 3 Scénarios côte à côte
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("🐻 Bear", f"{bear_dcf:.2f} $", delta=f"{bear_dcf-current_price:.1f}", delta_color="normal")
            col_d2.metric("🎯 Neutral", f"{base_dcf:.2f} $", delta=f"{base_dcf-current_price:.1f}", delta_color="normal")
            col_d3.metric("🐂 Bull", f"{bull_dcf:.2f} $", delta=f"{bull_dcf-current_price:.1f}", delta_color="normal")
            
            # Explication des Thèses DCF
            with st.expander("🔍 Voir les hypothèses DCF détaillées"):
                st.markdown(f"""
                - **🐻 Bear :** Croissance FCF ralentie à **{gr_fcf_input*0.8*100:.1f}%**, WACC augmenté à **{wacc+0.01:.1%}**.
                - **🎯 Neutral :** Croissance FCF de **{gr_fcf_input*100:.1f}%**, WACC de **{wacc:.1%}**.
                - **🐂 Bull :** Croissance FCF accélère à **{gr_fcf_input*1.2*100:.1f}%**, WACC réduit à **{wacc-0.01:.1%}**.
                """)

        # --- ONGLET 2 : VENTES ---
        with tab_sales:
            st.markdown("### Valorisation par Multiple des Ventes")
            st.caption("Quel prix pour la part de marché future (P/S Ratio) ?")
            
            # Les 3 Scénarios côte à côte
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("🐻 Bear", f"{bear_sales:.2f} $", delta=f"{bear_sales-current_price:.1f}", delta_color="normal")
            col_s2.metric("🎯 Neutral", f"{base_sales:.2f} $", delta=f"{base_sales-current_price:.1f}", delta_color="normal")
            col_s3.metric("🐂 Bull", f"{bull_sales:.2f} $", delta=f"{bull_sales-current_price:.1f}", delta_color="normal")

            # Explication des Thèses Ventes
            with st.expander("🔍 Voir les hypothèses Ventes détaillées"):
                st.markdown(f"""
                - **🐻 Bear :** Croiss. Ventes **{gr_sales_input*0.8*100:.1f}%**, on paiera **{target_ps*0.8:.1f}x** les ventes.
                - **🎯 Neutral :** Croiss. Ventes **{gr_sales_input*100:.1f}%**, on paiera **{target_ps:.1f}x** les ventes.
                - **🐂 Bull :** Croiss. Ventes **{gr_sales_input*1.2*100:.1f}%**, on paiera **{target_ps*1.2:.1f}x** les ventes.
                """)

        # --- ONGLET 3 : RATIOS ---
        with tab_ratios:
            st.markdown("### Santé & Ratios Actuels")
            
            r1, r2, r3 = st.columns(3)
            r1.metric("P/E (Bénéfices)", f"{pe_ratio:.1f}x" if pe_ratio > 0 else "N/A")
            r2.metric("P/FCF (Cash)", f"{pfcf_ratio:.1f}x" if pfcf_ratio > 0 else "N/A")
            r3.metric("EV/EBITDA", f"{ev_ebitda:.1f}x" if ev_ebitda > 0 else "N/A")
            
            st.divider()
            
            # Rule of 40 & Dette
            c_rule, c_net = st.columns(2)
            
            fcf_margin = (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0
            rule_40_score = (gr_sales_input * 100) + fcf_margin
            
            with c_rule:
                st.write("**Rule of 40**")
                if rule_40_score >= 40: st.success(f"✅ {rule_40_score:.1f}")
                else: st.warning(f"⚠️ {rule_40_score:.1f}")
                st.caption(f"Croissance Ventes {gr_sales_input*100:.0f}% + Marge FCF {fcf_margin:.0f}%")
            
            with c_net:
                st.write("**Position Nette (Cash - Dette)**")
                net = cash - debt
                color = "red" if net < 0 else "green"
                st.markdown(f":{color}[{net/1e6:.0f} M$]")
                if net < 0: st.caption("Dette Nette")
                else: st.caption("Cash Net")
