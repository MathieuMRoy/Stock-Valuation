import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")
st.title("📱 Valuation Master")
st.caption("Double Croissance • DCF vs Ventes • Ratios")

# --- 0. BENCHMARKS ---
SECTOR_BENCHMARKS = {
    "Technology": {"gr_sales": 0.15, "gr_fcf": 0.18, "ps": 6.0, "wacc": 0.095, "desc": "Levier opérationnel fort"},
    "Communication Services": {"gr_sales": 0.10, "gr_fcf": 0.15, "ps": 4.0, "wacc": 0.09, "desc": "Média/Streaming (Ex: Netflix)"},
    "Consumer Cyclical": {"gr_sales": 0.08, "gr_fcf": 0.10, "ps": 2.5, "wacc": 0.10, "desc": "Auto/Luxe"},
    "Healthcare": {"gr_sales": 0.06, "gr_fcf": 0.07, "ps": 4.0, "wacc": 0.08, "desc": "Stable"},
    "Financial Services": {"gr_sales": 0.05, "gr_fcf": 0.05, "ps": 2.0, "wacc": 0.10, "desc": "Croissance alignée"},
    "Energy": {"gr_sales": 0.03, "gr_fcf": 0.05, "ps": 1.5, "wacc": 0.11, "desc": "Volatile"},
    "Industrials": {"gr_sales": 0.04, "gr_fcf": 0.05, "ps": 1.8, "wacc": 0.09, "desc": "Cycle long"},
    "Utilities": {"gr_sales": 0.03, "gr_fcf": 0.03, "ps": 2.0, "wacc": 0.065, "desc": "Très stable"},
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
    """Cherche une valeur TTM en essayant plusieurs clés possibles"""
    if df is None or df.empty: return 0
    
    for key in keys_list:
        for idx in df.index:
            # On nettoie les espaces et la casse pour la comparaison
            idx_clean = str(idx).upper().replace(" ", "")
            key_clean = key.upper().replace(" ", "")
            
            if key_clean in idx_clean:
                row = df.loc[idx]
                total = 0
                count = 0
                for val in row:
                    if pd.api.types.is_number(val):
                        total += val
                        count += 1
                    if count == 4: break
                if total != 0: return total
    return 0

def get_cash_safe(df):
    if df is None or df.empty: return 0
    # Priorité aux liquidités immédiates
    val = get_ttm_flexible(df, ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments", "Cash"])
    # Note: get_ttm fait une somme, ici on veut la dernière valeur pour le Bilan
    # On refait une logique simple pour le bilan (dernière colonne)
    keys = ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments", "Cash"]
    for key in keys:
        for idx in df.index:
            if key.upper().replace(" ","") in str(idx).upper().replace(" ",""):
                return df.loc[idx].iloc[0]
    return 0

def get_debt_safe(df):
    if df is None or df.empty: return 0
    lt_debt = 0
    lease = 0
    
    for idx in df.index:
        s = str(idx).upper().replace(" ", "")
        if "LONGTERMDEBT" in s:
             lt_debt = df.loc[idx].iloc[0]; break
             
    for idx in df.index:
        s = str(idx).upper().replace(" ", "")
        if "LEASE" in s and "LIABILITIES" in s:
             lease = df.loc[idx].iloc[0]; break
             
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
        # Infos Secteur
        sector = info.get('sector', 'Default')
        bench = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["Default"])
        st.info(f"🏢 **Secteur : {sector}**")
        
        with st.expander(f"💡 Aide Hypothèses ({sector})", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Croiss. Ventes", f"{bench['gr_sales']*100:.0f}%")
            c2.metric("Croiss. FCF", f"{bench['gr_fcf']*100:.0f}%")
            c3.metric("P/S Cible", f"{bench['ps']}x")

        # Inputs
        st.subheader("⚙️ Vos Hypothèses (Neutral)")
        col1, col2 = st.columns(2)
        with col1:
            gr_sales_input = st.number_input("Croiss. Ventes (5 ans)", value=bench['gr_sales'], step=0.01, format="%.2f")
            gr_fcf_input = st.number_input("Croiss. FCF (5 ans)", value=bench['gr_fcf'], step=0.01, format="%.2f")
        with col2:
            wacc = st.number_input("CPMC (WACC)", value=bench['wacc'], step=0.005, format="%.3f")
            target_ps = st.number_input("Ratio P/S Cible", value=bench['ps'], step=0.5)

        # Calculs Data
        # Utilisation de get_ttm_flexible pour être sûr de trouver les revenus
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

        # --- CALCUL EBITDA ROBUSTE ---
        # On essaie de trouver l'EBITDA direct, sinon on le calcule
        ebitda_ttm = get_ttm_flexible(inc, ["EBITDA", "NormalizedEBITDA"])
        
        if ebitda_ttm == 0:
            # Plan B : Operating Income + D&A
            op_inc = get_ttm_flexible(inc, ["OperatingIncome", "Operating Income", "EBIT"])
            da = get_ttm_flexible(cf, ["Depreciation", "DepreciationAndAmortization", "Depreciation & Amortization"])
            ebitda_ttm = op_inc + da

        # Ratios
        eps = info.get('trailingEps', 0)
        pe_ratio = current_price / eps if eps > 0 else 0
        pfcf_ratio = market_cap / fcf_ttm if fcf_ttm > 0 else 0
        
        ev = market_cap + debt - cash
        ev_ebitda = ev / ebitda_ttm if ebitda_ttm > 0 else 0
        ps_current = market_cap / revenue_ttm if revenue_ttm > 0 else 0

        # Scénarios
        bear_dcf, bear_sales = calculate_valuation(gr_sales_input*0.8, gr_fcf_input*0.8, wacc+0.01, target_ps*0.8, revenue_ttm, fcf_ttm, cash, debt, shares)
        base_dcf, base_sales = calculate_valuation(gr_sales_input, gr_fcf_input, wacc, target_ps, revenue_ttm, fcf_ttm, cash, debt, shares)
        bull_dcf, bull_sales = calculate_valuation(gr_sales_input*1.2, gr_fcf_input*1.2, wacc-0.01, target_ps*1.2, revenue_ttm, fcf_ttm, cash, debt, shares)

        # --- AFFICHAGE ---
        st.divider()
        st.write(f"Prix Actuel : **{current_price:.2f} $**")
        
        tab1, tab2, tab3 = st.tabs(["💵 DCF (Cash)", "📈 Ventes (Growth)", "📊 Ratios"])

        with tab1:
            st.subheader("Modèle DCF")
            c1, c2, c3 = st.columns(3)
            c1.metric("🐻 Bear", f"{bear_dcf:.2f} $", delta=f"{bear_dcf-current_price:.1f}", delta_color="normal")
            c2.metric("🎯 Neutral", f"{base_dcf:.2f} $", delta=f"{base_dcf-current_price:.1f}", delta_color="normal")
            c3.metric("🐂 Bull", f"{bull_dcf:.2f} $", delta=f"{bull_dcf-current_price:.1f}", delta_color="normal")

        with tab2:
            st.subheader("Modèle Ventes")
            c1, c2, c3 = st.columns(3)
            c1.metric("🐻 Bear", f"{bear_sales:.2f} $", delta=f"{bear_sales-current_price:.1f}", delta_color="normal")
            c2.metric("🎯 Neutral", f"{base_sales:.2f} $", delta=f"{base_sales-current_price:.1f}", delta_color="normal")
            c3.metric("🐂 Bull", f"{bull_sales:.2f} $", delta=f"{bull_sales-current_price:.1f}", delta_color="normal")

        with tab3:
            st.subheader("Analyse Fondamentale")
            r1, r2, r3 = st.columns(3)
            r1.metric("P/E", f"{pe_ratio:.1f}x" if pe_ratio > 0 else "N/A")
            r2.metric("P/FCF", f"{pfcf_ratio:.1f}x" if pfcf_ratio > 0 else "N/A")
            r3.metric("EV/EBITDA", f"{ev_ebitda:.1f}x" if ev_ebitda > 0 else "N/A", help="Enterprise Value / EBITDA. Plus c'est bas, mieux c'est.")
            
            st.divider()
            fcf_margin = (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0
            rule_40_score = (gr_sales_input * 100) + fcf_margin
            
            c_rule, c_net = st.columns(2)
            with c_rule:
                st.write("**Rule of 40**")
                if rule_40_score >= 40: st.success(f"✅ {rule_40_score:.1f}")
                elif rule_40_score >= 20: st.warning(f"⚠️ {rule_40_score:.1f}")
                else: st.error(f"❌ {rule_40_score:.1f}")
            
            with c_net:
                # Clarification Position Nette
                st.write("**Position Nette (Cash - Dette)**")
                net = cash - debt
                color = "red" if net < 0 else "green"
                st.markdown(f":{color}[{net/1e6:.0f} M$]")
                if net < 0:
                    st.caption(f"Dette nette : {abs(net)/1e6:.0f} M$")
                else:
                    st.caption(f"Cash excédentaire : {net/1e6:.0f} M$")
