import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION DE LA PAGE (Mode Mobile) ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")

# --- TITRE ---
st.title("📱 Valuation Master")
st.caption("DCF • Ventes • Rule of 40 • Scénarios")

# --- 1. ENTRÉES UTILISATEUR (INPUTS) ---
ticker = st.text_input("Symbole (Ticker)", value="DUOL").upper()

with st.expander("⚙️ Modifier les Hypothèses (Cas de Base)", expanded=False):
    st.subheader("Hypothèses Générales")
    col1, col2 = st.columns(2)
    with col1:
        growth_rate = st.number_input("Croissance (5 ans)", value=0.20, step=0.01, format="%.2f")
        wacc = st.number_input("WACC", value=0.09, step=0.005, format="%.3f")
    with col2:
        terminal_growth = st.number_input("Croissance Infinie", value=0.03, step=0.005, format="%.2f")
    
    st.subheader("Spécifique Modèle Ventes")
    col3, col4 = st.columns(2)
    with col3:
        target_ps = st.number_input("Ratio P/S Cible", value=6.0, step=0.5)
    with col4:
        discount_rate_sales = st.number_input("Taux Actualisation", value=0.10, step=0.01, format="%.2f")

# --- 2. FONCTIONS (DATA & CALCULS) ---
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

def get_latest_bs_val(df, key):
    if df is None or df.empty: return 0
    for idx in df.index:
        if key.upper() in str(idx).upper():
            if "TOTAL" in key.upper() and "OTHER" in str(idx).upper(): continue
            return df.loc[idx].iloc[0]
    return 0

def get_real_shares(info):
    shares = 0
    if 'impliedSharesOutstanding' in info and info['impliedSharesOutstanding']:
        shares = info['impliedSharesOutstanding']
    if shares == 0:
        mcap = info.get('marketCap', 0)
        price = info.get('currentPrice', 0)
        if mcap > 0 and price > 0: shares = mcap / price
    if shares == 0: shares = info.get('sharesOutstanding', 0)
    return shares

# --- FONCTION DE VALORISATION GÉNÉRIQUE ---
def calculate_valuation(growth, wacc_val, ps_target, revenue, fcf, cash, debt, shares):
    # DCF
    current_fcf = fcf
    fcf_projections = []
    for i in range(5):
        current_fcf = current_fcf * (1 + growth)
        fcf_projections.append(current_fcf)
    
    terminal_val = (fcf_projections[-1] * (1 + 0.03)) / (wacc_val - 0.03)
    pv_fcf = sum([val / ((1 + wacc_val)**(i+1)) for i, val in enumerate(fcf_projections)])
    pv_terminal = terminal_val / ((1 + wacc_val)**5)
    equity_val = (pv_fcf + pv_terminal) + cash - debt
    price_dcf = equity_val / shares

    # Ventes
    rev_future = revenue * ((1 + growth)**5)
    mcap_future = rev_future * ps_target
    price_sales = (mcap_future / shares) / ((1 + 0.10)**5)
    
    return price_dcf, price_sales

# --- 3. EXÉCUTION ---
if ticker:
    bs, inc, cf, info = get_financial_data(ticker)
    
    if bs is None or inc.empty:
        st.error(f"Erreur données pour {ticker}")
    else:
        # CHIFFRES CLÉS
        revenue_ttm = get_ttm(inc, "Total Revenue")
        cfo_ttm = get_ttm(cf, "Operating Cash Flow")
        capex_ttm = abs(get_ttm(cf, "Capital Expenditure"))
        fcf_ttm = cfo_ttm - capex_ttm
        cash = get_latest_bs_val(bs, "Cash")
        debt = get_latest_bs_val(bs, "Long Term Debt") + get_latest_bs_val(bs, "Lease")
        shares = get_real_shares(info)
        if shares == 0: shares = 1
        current_price = info.get('currentPrice', 0)

        # --- RULE OF 40 ---
        fcf_margin = (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0
        rule_40_score = (growth_rate * 100) + fcf_margin
        
        # Affichage Rule of 40
        st.divider()
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            st.subheader("Rule of 40 (SaaS)")
            st.caption(f"Croissance ({growth_rate*100:.0f}%) + Marge FCF ({fcf_margin:.0f}%)")
        with col_r2:
            if rule_40_score >= 40:
                st.success(f"✅ {rule_40_score:.1f}")
            elif rule_40_score >= 20:
                st.warning(f"⚠️ {rule_40_score:.1f}")
            else:
                st.error(f"❌ {rule_40_score:.1f}")

        # --- BULL & BEAR SCENARIOS ---
        # On définit les scénarios automatiquement basés sur les inputs de l'utilisateur
        
        # Bear: Croissance -20% relatif, P/S -20%
        bear_growth = growth_rate * 0.8
        bear_ps = target_ps * 0.8
        bear_dcf, bear_sales = calculate_valuation(bear_growth, wacc + 0.01, bear_ps, revenue_ttm, fcf_ttm, cash, debt, shares)
        
        # Base: Tes inputs
        base_dcf, base_sales = calculate_valuation(growth_rate, wacc, target_ps, revenue_ttm, fcf_ttm, cash, debt, shares)
        
        # Bull: Croissance +20% relatif, P/S +20%
        bull_growth = growth_rate * 1.2
        bull_ps = target_ps * 1.2
        bull_dcf, bull_sales = calculate_valuation(bull_growth, wacc - 0.01, bull_ps, revenue_ttm, fcf_ttm, cash, debt, shares)

        # --- AFFICHAGE RÉSULTATS ---
        st.divider()
        st.subheader("🎯 Objectifs de Prix")
        st.write(f"Prix Actuel : **{current_price:.2f} $**")

        # Onglets pour switch entre DCF et Ventes
        tab1, tab2 = st.tabs(["Flux de Trésorerie (DCF)", "Multiple des Ventes"])

        with tab1:
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("🐻 Bear", f"{bear_dcf:.2f} $", delta=f"{bear_dcf-current_price:.1f}", delta_color="normal")
            col_b2.metric("🎯 Base", f"{base_dcf:.2f} $", delta=f"{base_dcf-current_price:.1f}", delta_color="normal")
            col_b3.metric("bull Bull", f"{bull_dcf:.2f} $", delta=f"{bull_dcf-current_price:.1f}", delta_color="normal")
            st.caption("Basé sur le Free Cash Flow généré.")

        with tab2:
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("🐻 Bear", f"{bear_sales:.2f} $", delta=f"{bear_sales-current_price:.1f}", delta_color="normal")
            col_s2.metric("🎯 Base", f"{base_sales:.2f} $", delta=f"{base_sales-current_price:.1f}", delta_color="normal")
            col_s3.metric("bull Bull", f"{bull_sales:.2f} $", delta=f"{bull_sales-current_price:.1f}", delta_color="normal")
            st.caption("Basé sur la croissance du Chiffre d'Affaires.")

        # Détails Financiers
        with st.expander("📊 Voir Données Brutes (TTM)"):
            st.write(f"**Revenus:** {revenue_ttm/1e6:.0f} M$")
            st.write(f"**FCF:** {fcf_ttm/1e6:.0f} M$")
            st.write(f"**Cash Net:** {(cash-debt)/1e6:.0f} M$")
            st.write(f"**Actions:** {shares/1e6:.1f} M")
