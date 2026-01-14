import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Valuation Master", layout="centered")

st.title("📱 Valuation Master")
st.write("Analyse rapide DCF & Ventes")

# --- 1. ENTRÉES UTILISATEUR ---
ticker = st.text_input("Entrez le Ticker (ex: NFLX, DUOL)", value="NFLX").upper()

with st.expander("⚙️ Modifier les Hypothèses", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        growth_rate = st.number_input("Croissance (5 ans)", value=0.12, step=0.01, format="%.2f")
        wacc = st.number_input("WACC (Coût Capital)", value=0.095, step=0.005, format="%.3f")
    with col2:
        terminal_growth = st.number_input("Croissance Terminale", value=0.03, step=0.005, format="%.2f")
        target_ps = st.number_input("Ratio P/S Cible", value=6.0, step=0.5)

# --- FONCTIONS DE CALCUL (Ton Moteur V23) ---
def get_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # On force le téléchargement
        bs = stock.quarterly_balance_sheet
        inc = stock.quarterly_financials
        cf = stock.quarterly_cashflow
        info = stock.info
        return bs, inc, cf, info
    except:
        return None, None, None, None

def get_ttm(df, key):
    if df is None or df.empty: return 0
    # Recherche flexible
    for idx in df.index:
        if key.upper() in str(idx).upper():
            return df.loc[idx].iloc[:4].sum() # Somme des 4 derniers trimestres
    return 0

def get_latest(df, key):
    if df is None or df.empty: return 0
    for idx in df.index:
        if key.upper() in str(idx).upper():
            return df.loc[idx].iloc[0]
    return 0

# --- BOUTON D'ANALYSE ---
if st.button("Lancer l'Analyse 🚀"):
    with st.spinner('Récupération des données...'):
        bs, inc, cf, info = get_data(ticker)

    if bs is None or inc.empty:
        st.error(f"Impossible de trouver des données pour {ticker}")
    else:
        # --- CALCULS ---
        # 1. Données TTM
        revenue = get_ttm(inc, "Total Revenue")
        cfo = get_ttm(cf, "Operating Cash Flow")
        capex = abs(get_ttm(cf, "Capital Expenditure"))
        fcf = cfo - capex
        
        # 2. Bilan & Actions
        cash = get_latest(bs, "Cash")
        debt = get_latest(bs, "Long Term Debt") + get_latest(bs, "Lease")
        
        shares = info.get('impliedSharesOutstanding', 0)
        if shares == 0: shares = info.get('sharesOutstanding', 1)
        
        current_price = info.get('currentPrice', 0)
        
        # 3. DCF
        fcf_future = []
        current_fcf = fcf
        for i in range(5):
            current_fcf = current_fcf * (1 + growth_rate)
            fcf_future.append(current_fcf)
            
        terminal_value = (fcf_future[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        
        pv_flows = sum([f / ((1 + wacc)**(i+1)) for i, f in enumerate(fcf_future)])
        pv_terminal = terminal_value / ((1 + wacc)**5)
        
        ev = pv_flows + pv_terminal
        equity_value = ev + cash - debt
        target_price_dcf = equity_value / shares

        # 4. REVENUE MODEL
        revenue_future = revenue * ((1 + growth_rate)**5)
        market_cap_future = revenue_future * target_ps
        price_future = market_cap_future / shares
        target_price_sales = price_future / ((1 + 0.10)**5) # Actualisé à 10%

        # --- AFFICHAGE ---
        st.success(f"Analyse terminée pour {ticker}")
        
        # Affichage des Prix Cibles
        col1, col2, col3 = st.columns(3)
        col1.metric("Prix Actuel", f"{current_price:.2f} $")
        col2.metric("Cible DCF", f"{target_price_dcf:.2f} $", delta=f"{target_price_dcf-current_price:.2f}")
        col3.metric("Cible Ventes", f"{target_price_sales:.2f} $", delta=f"{target_price_sales-current_price:.2f}")

        # Détails
        st.divider()
        st.subheader("📊 Détails Financiers (TTM)")
        st.write(f"**Chiffre d'Affaires :** {revenue/1e9:.2f} Md$")
        st.write(f"**Free Cash Flow :** {fcf/1e6:.0f} M$")
        st.write(f"**Cash Net :** {(cash-debt)/1e6:.0f} M$")
