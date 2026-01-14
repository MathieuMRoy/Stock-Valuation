import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")
st.title("📱 Valuation Master")
st.caption("Comparables • Pédagogie • Scénarios")

# --- 0. DATA : COMPARABLES & SECTEURS ---

# 1. Groupes de Comparables (Base de données manuelle pour la rapidité)
PEER_GROUPS = {
    "SEMICONDUCTORS": {
        "tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM"],
        "gr_sales": 0.18, "gr_fcf": 0.20, "ps": 8.0, "wacc": 0.10,
        "name": "Semi-conducteurs & AI"
    },
    "BIG_TECH": {
        "tickers": ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META"],
        "gr_sales": 0.12, "gr_fcf": 0.15, "ps": 6.5, "wacc": 0.09,
        "name": "Big Tech / GAFAM"
    },
    "SAAS_CLOUD": {
        "tickers": ["CRM", "ADBE", "SNOW", "DDOG", "PLTR", "NOW", "SHOP", "DUOL"],
        "gr_sales": 0.20, "gr_fcf": 0.22, "ps": 10.0, "wacc": 0.10,
        "name": "Logiciel SaaS & Cloud (Hyper-croissance)"
    },
    "STREAMING": {
        "tickers": ["NFLX", "DIS", "WBD", "PARA", "SPOT"],
        "gr_sales": 0.10, "gr_fcf": 0.15, "ps": 4.0, "wacc": 0.09,
        "name": "Streaming & Média"
    },
    "EV_AUTO": {
        "tickers": ["TSLA", "RIVN", "LCID", "BYD", "F", "GM"],
        "gr_sales": 0.15, "gr_fcf": 0.12, "ps": 3.0, "wacc": 0.11,
        "name": "Véhicules Électriques & Auto"
    },
    "BANKS_CA": {
        "tickers": ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"],
        "gr_sales": 0.04, "gr_fcf": 0.05, "ps": 2.5, "wacc": 0.08,
        "name": "Banques Canadiennes"
    }
}

# 2. Benchmarks Sectoriels (Fallback)
SECTOR_BENCHMARKS = {
    "Technology": {"gr_sales": 0.12, "gr_fcf": 0.15, "ps": 5.0, "wacc": 0.095},
    "Communication Services": {"gr_sales": 0.08, "gr_fcf": 0.10, "ps": 3.5, "wacc": 0.09},
    "Consumer Cyclical": {"gr_sales": 0.06, "gr_fcf": 0.08, "ps": 2.0, "wacc": 0.10},
    "Healthcare": {"gr_sales": 0.05, "gr_fcf": 0.06, "ps": 4.0, "wacc": 0.08},
    "Financial Services": {"gr_sales": 0.05, "gr_fcf": 0.05, "ps": 2.5, "wacc": 0.09},
    "Energy": {"gr_sales": 0.03, "gr_fcf": 0.05, "ps": 1.5, "wacc": 0.10},
    "Industrials": {"gr_sales": 0.04, "gr_fcf": 0.05, "ps": 1.8, "wacc": 0.09},
    "Default": {"gr_sales": 0.07, "gr_fcf": 0.08, "ps": 2.5, "wacc": 0.09}
}

def get_benchmark_data(ticker, sector_info):
    """Trouve les meilleures données : soit par Groupe de Pairs, soit par Secteur"""
    ticker_clean = ticker.upper().replace(".TO", "") # Nettoyage pour recherche
    
    # 1. Recherche dans les groupes spécifiques
    for group_key, data in PEER_GROUPS.items():
        # On vérifie si le ticker est dans la liste ou contient le ticker (ex: GOOG vs GOOGL)
        if any(t in ticker_clean for t in data['tickers']):
            return {
                "gr_sales": data['gr_sales'], "gr_fcf": data['gr_fcf'],
                "ps": data['ps'], "wacc": data['wacc'],
                "source": "Comparables", "name": data['name'], "peers": ", ".join(data['tickers'][:4])
            }
            
    # 2. Fallback Secteur
    bench = SECTOR_BENCHMARKS.get(sector_info, SECTOR_BENCHMARKS["Default"])
    return {
        "gr_sales": bench['gr_sales'], "gr_fcf": bench['gr_fcf'],
        "ps": bench['ps'], "wacc": bench['wacc'],
        "source": "Secteur", "name": sector_info, "peers": "Moyenne du secteur"
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
        # --- LOGIQUE INTELLIGENTE DE COMPARABLES ---
        raw_sector = info.get('sector', 'Default')
        bench_data = get_benchmark_data(ticker, raw_sector)
        
        # --- AIDE DYNAMIQUE ---
        # On change le titre selon si on a trouvé des vrais concurrents ou juste le secteur
        help_title = f"💡 Aide : {bench_data['name']}"
        
        with st.expander(help_title, expanded=True):
            if bench_data['source'] == "Comparables":
                st.write(f"**Comparables identifiés :** {bench_data['peers']}")
            else:
                st.write(f"**Secteur :** {raw_sector} (Données moyennes)")
                
            c1, c2, c3 = st.columns(3)
            c1.metric("Croiss. Ventes", f"{bench_data['gr_sales']*100:.0f}%", help="Moyenne historique des comparables")
            c2.metric("Croiss. FCF", f"{bench_data['gr_fcf']*100:.0f}%", help="Souvent plus élevé que les ventes pour les leaders")
            c3.metric("P/S Cible", f"{bench_data['ps']}x")

        # --- INPUTS ---
        st.subheader("⚙️ Vos Hypothèses (Neutral)")
        col1, col2 = st.columns(2)
        with col1:
            gr_sales_input = st.number_input("Croiss. Ventes (5 ans)", value=bench_data['gr_sales'], step=0.01, format="%.2f")
            gr_fcf_input = st.number_input("Croiss. FCF (5 ans)", value=bench_data['gr_fcf'], step=0.01, format="%.2f")
        with col2:
            wacc = st.number_input("CPMC (WACC)", value=bench_data['wacc'], step=0.005, format="%.3f")
            target_ps = st.number_input("Ratio P/S Cible", value=bench_data['ps'], step=0.5)

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

        # Ratios
        eps = info.get('trailingEps', 0)
        pe_ratio = current_price / eps if eps > 0 else 0
        pfcf_ratio = market_cap / fcf_ttm if fcf_ttm > 0 else 0
        
        ebitda_ttm = get_ttm_flexible(inc, ["EBITDA", "NormalizedEBITDA"])
        if ebitda_ttm == 0:
            op_inc = get_ttm_flexible(inc, ["OperatingIncome", "Operating Income", "EBIT"])
            da = get_ttm_flexible(cf, ["Depreciation", "DepreciationAndAmortization"])
            ebitda_ttm = op_inc + da
        ev = market_cap + debt - cash
        ev_ebitda = ev / ebitda_ttm if ebitda_ttm > 0 else 0

        # ==========================================
        # RESULTATS AVEC PÉDAGOGIE
        # ==========================================
        st.divider()
        st.subheader("🏷️ Prix à Payer Aujourd'hui")
        
        c_m1, c_m2 = st.columns(2)
        c_m1.metric("Prix Actuel", f"{current_price:.2f} $")
        
        delta_val = base_dcf - current_price
        c_m2.metric("Valeur Intrinsèque (Neutre)", f"{base_dcf:.2f} $", delta=f"{delta_val:.2f} $", delta_color="normal")
        
        # --- EXPLICATION PÉDAGOGIQUE ---
        st.info(
            "ℹ️ **Comprendre ce chiffre :** \n\n"
            "Ce prix est calculé selon la méthode **DCF (Discounted Cash Flow)**. \n"
            "Il représente la valeur totale de tout l'argent (Cash Flow) que l'entreprise va générer "
            "dans le futur, ramenée à sa valeur d'aujourd'hui, plus sa trésorerie nette."
        )

        st.write("")
        
        # ONGLETS MÉTHODES
        tab_dcf, tab_sales, tab_ratios = st.tabs(["💵 DCF (Cash)", "📈 Ventes (Growth)", "📊 Ratios"])

        # --- ONGLET 1 : DCF ---
        with tab_dcf:
            st.markdown("### Valorisation par Flux de Trésorerie")
            
            # Scénarios
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("🐻 Bear", f"{bear_dcf:.2f} $", delta=f"{bear_dcf-current_price:.1f}", delta_color="normal")
            col_d2.metric("🎯 Neutral", f"{base_dcf:.2f} $", delta=f"{base_dcf-current_price:.1f}", delta_color="normal")
            col_d3.metric("🐂 Bull", f"{bull_dcf:.2f} $", delta=f"{bull_dcf-current_price:.1f}", delta_color="normal")
            
            with st.expander("🔍 Comprendre les scénarios DCF"):
                st.markdown(f"""
                Le moteur de ce modèle est la **capacité à générer du Cash**.
                - **🐻 Bear :** Croissance décevante de **{gr_fcf_input*0.8*100:.1f}%**.
                - **🎯 Neutral :** Votre hypothèse de base (**{gr_fcf_input*100:.1f}%**).
                - **🐂 Bull :** L'entreprise surperforme (**{gr_fcf_input*1.2*100:.1f}%**).
                """)

        # --- ONGLET 2 : VENTES ---
        with tab_sales:
            st.markdown("### Valorisation par Multiple des Ventes")
            
            # Scénarios
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("🐻 Bear", f"{bear_sales:.2f} $", delta=f"{bear_sales-current_price:.1f}", delta_color="normal")
            col_s2.metric("🎯 Neutral", f"{base_sales:.2f} $", delta=f"{base_sales-current_price:.1f}", delta_color="normal")
            col_s3.metric("🐂 Bull", f"{bull_sales:.2f} $", delta=f"{bull_sales-current_price:.1f}", delta_color="normal")

            with st.expander("🔍 Comprendre les scénarios Ventes"):
                st.markdown(f"""
                Le moteur de ce modèle est la **popularité future (Ratio P/S)**.
                - **🐻 Bear :** Le marché paie seulement **{target_ps*0.8:.1f}x** les revenus.
                - **🎯 Neutral :** Le marché paie **{target_ps:.1f}x** les revenus.
                - **🐂 Bull :** L'euphorie monte à **{target_ps*1.2:.1f}x** les revenus.
                """)

        # --- ONGLET 3 : RATIOS ---
        with tab_ratios:
            st.markdown("### Indicateurs Fondamentaux")
            r1, r2, r3 = st.columns(3)
            r1.metric("P/E", f"{pe_ratio:.1f}x" if pe_ratio > 0 else "N/A", help="Années de bénéfices pour rembourser le prix")
            r2.metric("P/FCF", f"{pfcf_ratio:.1f}x" if pfcf_ratio > 0 else "N/A", help="Le ratio roi pour le cash flow")
            r3.metric("EV/EBITDA", f"{ev_ebitda:.1f}x" if ev_ebitda > 0 else "N/A", help="Neutre vis-à-vis de la dette")
            
            st.divider()
            
            c_rule, c_net = st.columns(2)
            fcf_margin = (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0
            rule_40_score = (gr_sales_input * 100) + fcf_margin
            
            with c_rule:
                st.write("**Rule of 40**")
                if rule_40_score >= 40: st.success(f"✅ {rule_40_score:.1f}")
                else: st.warning(f"⚠️ {rule_40_score:.1f}")
                st.caption("Croissance + Marge FCF")
            
            with c_net:
                st.write("**Position Nette**")
                net = cash - debt
                color = "red" if net < 0 else "green"
                st.markdown(f":{color}[{net/1e6:.0f} M$]")
                if net < 0: st.caption("Dette Nette")
                else: st.caption("Cash Net")
