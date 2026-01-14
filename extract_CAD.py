import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")
st.title("📱 Valuation Master")
st.caption("Valuation • Ratios • Benchmarks Sectoriels")

# --- 0. DICTIONNAIRE DE BENCHMARKS (SUGGESTIONS) ---
SECTOR_BENCHMARKS = {
    "Technology": {"growth": 0.15, "ps": 6.0, "desc": "Croissance élevée, Marges fortes"},
    "Communication Services": {"growth": 0.10, "ps": 4.0, "desc": "Média/Télécom (Ex: Netflix/Google)"},
    "Consumer Cyclical": {"growth": 0.08, "ps": 2.5, "desc": "Auto/Luxe (Ex: Tesla/Amazon)"},
    "Healthcare": {"growth": 0.06, "ps": 4.0, "desc": "Défensif, R&D élevée"},
    "Financial Services": {"growth": 0.05, "ps": 2.0, "desc": "Banques/Assurances"},
    "Energy": {"growth": 0.03, "ps": 1.5, "desc": "Pétrole/Gaz (Volatile)"},
    "Industrials": {"growth": 0.04, "ps": 1.8, "desc": "Construction/Aérospatial"},
    "Utilities": {"growth": 0.03, "ps": 2.0, "desc": "Dividendes stables, Croissance faible"},
    "Basic Materials": {"growth": 0.03, "ps": 1.2, "desc": "Mines/Chimie"},
    "Real Estate": {"growth": 0.04, "ps": 5.0, "desc": "Immobilier (REITs)"},
    "Default": {"growth": 0.08, "ps": 3.0, "desc": "Moyenne du marché"}
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

def get_ttm(df, key):
    if df is None or df.empty: return 0
    total = 0
    for idx in df.index:
        if key.upper() in str(idx).upper():
            row = df.loc[idx]
            count = 0
            for val in row:
                if pd.api.types.is_number(val):
                    total += val
                    count += 1
                if count == 4: break
            return total
    return 0

def get_cash_safe(df):
    if df is None or df.empty: return 0
    keys = ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments", "Cash"]
    for key in keys:
        for idx in df.index:
            if key.upper() in str(idx).upper():
                val = df.loc[idx].iloc[0]
                if val > 0: return val
    return 0

def get_debt_safe(df):
    if df is None or df.empty: return 0
    lt_debt, lease = 0, 0
    for idx in df.index:
        if "LongTermDebt" in str(idx) or ("Long" in str(idx) and "Debt" in str(idx)):
             lt_debt = df.loc[idx].iloc[0]; break
    for idx in df.index:
        if "Lease" in str(idx) and "Liabilities" in str(idx):
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

# --- 2. INTERFACE & LOGIQUE ---
ticker = st.text_input("Symbole (Ticker)", value="NFLX").upper()

if ticker:
    bs, inc, cf, info = get_financial_data(ticker)
    
    if bs is None or inc.empty:
        st.error("Données introuvables.")
    else:
        # --- DÉTECTION SECTEUR & SUGGESTIONS ---
        sector = info.get('sector', 'Default')
        industry = info.get('industry', 'Inconnu')
        
        # Trouver le benchmark
        bench = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["Default"])
        
        # Affichage Infos Secteur
        st.info(f"🏢 **Secteur détecté : {sector}** ({industry})")
        
        with st.expander(f"💡 Aide à la décision pour {sector}", expanded=True):
            st.markdown(f"""
            Pour ce secteur (**{bench['desc']}**), les analystes utilisent souvent :
            * 📈 **Croissance :** env. **{bench['growth']*100:.0f}%**
            * 🏷️ **Ratio P/S :** env. **{bench['ps']}x**
            """)

        # --- INPUTS ---
        st.subheader("⚙️ Vos Hypothèses (Neutral)")
        col1, col2 = st.columns(2)
        with col1:
            growth_rate = st.number_input("Croissance (5 ans)", value=bench['growth'], step=0.01, format="%.2f")
            wacc = st.number_input("WACC", value=0.09, step=0.005, format="%.3f")
        with col2:
            target_ps = st.number_input("Ratio P/S Cible", value=bench['ps'], step=0.5)
            terminal_growth = st.number_input("Croissance Infinie", value=0.03, step=0.005, format="%.2f")

        # --- CALCULS FINANCIERS ---
        revenue_ttm = get_ttm(inc, "Total Revenue")
        cfo_ttm = get_ttm(cf, "Operating Cash Flow")
        capex_ttm = abs(get_ttm(cf, "Capital Expenditure"))
        fcf_ttm = cfo_ttm - capex_ttm
        cash = get_cash_safe(bs)
        debt = get_debt_safe(bs)
        shares = get_real_shares(info)
        if shares == 0: shares = 1
        current_price = info.get('currentPrice', 0)
        market_cap = shares * current_price
        
        # EBITDA (Calcul simplifié pour ratio)
        op_income = get_ttm(inc, "OperatingIncome")
        dep_amort = get_ttm(cf, "DepreciationAndAmortization")
        ebitda = op_income + dep_amort

        # --- CALCULS VALORISATION ---
        # 1. DCF
        current_fcf = fcf_ttm
        fcf_projections = []
        for i in range(5):
            current_fcf = current_fcf * (1 + growth_rate)
            fcf_projections.append(current_fcf)
        
        terminal_val = (fcf_projections[-1] * (1 + 0.03)) / (wacc - 0.03)
        pv_fcf = sum([val / ((1 + wacc)**(i+1)) for i, val in enumerate(fcf_projections)])
        pv_terminal = terminal_val / ((1 + wacc)**5)
        equity_val_dcf = (pv_fcf + pv_terminal) + cash - debt
        price_dcf = equity_val_dcf / shares

        # 2. VENTES
        rev_future = revenue_ttm * ((1 + growth_rate)**5)
        price_sales = ((rev_future * target_ps) / shares) / ((1 + 0.10)**5)

        # 3. RATIOS ACTUELS
        eps = info.get('trailingEps', 0)
        pe_ratio = current_price / eps if eps > 0 else 0
        pfcf_ratio = market_cap / fcf_ttm if fcf_ttm > 0 else 0
        ev = market_cap + debt - cash
        ev_ebitda = ev / ebitda if ebitda > 0 else 0
        ps_current = market_cap / revenue_ttm if revenue_ttm > 0 else 0

        # --- AFFICHAGE RÉSULTATS ---
        st.divider()
        st.subheader("🎯 Objectifs de Prix")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix Actuel", f"{current_price:.2f} $")
        c2.metric("Cible DCF", f"{price_dcf:.2f} $", delta=f"{price_dcf-current_price:.2f}")
        c3.metric("Cible Ventes", f"{price_sales:.2f} $", delta=f"{price_sales-current_price:.2f}")

        # --- NOUVELLE SECTION : RATIOS ---
        st.divider()
        st.subheader("📊 Ratios de Valorisation (Actuel)")
        
        r1, r2, r3, r4 = st.columns(4)
        
        # P/E
        r1.metric("P/E (Price/Earnings)", f"{pe_ratio:.1f}x" if pe_ratio > 0 else "N/A", help="Prix payé pour 1$ de bénéfice. Moyenne marché ~20x.")
        
        # P/FCF
        r2.metric("P/FCF (Free Cash Flow)", f"{pfcf_ratio:.1f}x" if pfcf_ratio > 0 else "N/A", help="Prix payé pour 1$ de vrai cash généré. <15x est souvent bon marché.")
        
        # EV/EBITDA
        r3.metric("EV / EBITDA", f"{ev_ebitda:.1f}x" if ev_ebitda > 0 else "N/A", help="Valorisation dette incluse. <10x est souvent bon marché.")
        
        # P/S
        r4.metric("P/S (Ventes)", f"{ps_current:.1f}x", help="Prix payé pour 1$ de ventes.")

        # --- RULE OF 40 ---
        fcf_margin = (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0
        rule_40_score = (growth_rate * 100) + fcf_margin
        
        st.write("") # Spacer
        if rule_40_score >= 40:
            st.success(f"✅ Rule of 40 Score: {rule_40_score:.1f} (Excellent)")
        else:
            st.warning(f"⚠️ Rule of 40 Score: {rule_40_score:.1f} (Moyen/Faible)")

        # Détails Financiers (Cachés)
        with st.expander("Voir Données Brutes"):
            st.write(f"**Revenus TTM:** {revenue_ttm/1e6:.0f} M$")
            st.write(f"**FCF TTM:** {fcf_ttm/1e6:.0f} M$")
            st.write(f"**EBITDA TTM:** {ebitda/1e6:.0f} M$")
            color = "red" if (cash-debt) < 0 else "green"
            st.markdown(f"**Position Nette:** :{color}[{(cash-debt)/1e6:.0f} M$]")
