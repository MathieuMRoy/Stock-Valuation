import streamlit as st
import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Valuation Master", page_icon="📱", layout="centered")
st.title("📱 Valuation Master")
st.caption("3 Modèles : Cash • Ventes • Bénéfices")

# --- 0. DATA : BASE DE DONNÉES INTELLIGENTE (SEARCH) ---
# Format : "TICKER - Nom de la compagnie"
TICKER_DB = [
    "🔍 Autre (Entrer manuellement)",
    "--- TECH US (MAGNIFICENT 7) ---",
    "AAPL - Apple Inc.",
    "MSFT - Microsoft Corp.",
    "NVDA - NVIDIA Corp.",
    "GOOG - Alphabet Inc. (Google)",
    "AMZN - Amazon.com",
    "META - Meta Platforms (Facebook)",
    "TSLA - Tesla Inc.",
    "--- POPULAIRE & SPACE ---",
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

# --- 1. DATA : BENCHMARKS ---
PEER_GROUPS = {
    "SPACE_TECH": {
        "tickers": ["MDA", "RKLB", "ASTS", "LUNR", "PL", "SPIR", "SPCE"],
        "gr_sales": 0.20, "gr_fcf": 0.25, "gr_eps": 0.25, "ps": 6.0, "pe": 40.0, "p_fcf": 35.0, "wacc": 0.11,
        "name": "Space Tech & Satellites"
    },
    "SEMICONDUCTORS": {
        "tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM"],
        "gr_sales": 0.18, "gr_fcf": 0.20, "gr_eps": 0.20, "ps": 8.0, "pe": 35.0, "p_fcf": 30.0, "wacc": 0.10,
        "name": "Semi-conducteurs & AI"
    },
    "BIG_TECH": {
        "tickers": ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META"],
        "gr_sales": 0.12, "gr_fcf": 0.15, "gr_eps": 0.15, "ps": 6.5, "pe": 25.0, "p_fcf": 28.0, "wacc": 0.09,
        "name": "Big Tech / GAFAM"
    },
    "SAAS_CLOUD": {
        "tickers": ["CRM", "ADBE", "SNOW", "DDOG", "PLTR", "NOW", "SHOP", "DUOL"],
        "gr_sales": 0.20, "gr_fcf": 0.22, "gr_eps": 0.25, "ps": 10.0, "pe": 40.0, "p_fcf": 35.0, "wacc": 0.10,
        "name": "Logiciel SaaS & Cloud"
    },
    "STREAMING": {
        "tickers": ["NFLX", "DIS", "WBD", "PARA", "SPOT"],
        "gr_sales": 0.10, "gr_fcf": 0.15, "gr_eps": 0.18, "ps": 4.0, "pe": 25.0, "p_fcf": 20.0, "wacc": 0.09,
        "name": "Streaming & Média"
    },
    "EV_AUTO": {
        "tickers": ["TSLA", "RIVN", "LCID", "BYD", "F", "GM"],
        "gr_sales": 0.15, "gr_fcf": 0.12, "gr_eps": 0.15, "ps": 3.0, "pe": 30.0, "p_fcf": 25.0, "wacc": 0.11,
        "name": "Véhicules Électriques"
    },
    "BANKS_CA": {
        "tickers": ["RY", "TD", "BMO", "BNS", "CM", "NA"],
        "gr_sales": 0.04, "gr_fcf": 0.05, "gr_eps": 0.06, "ps": 2.5, "pe": 11.0, "p_fcf": 12.0, "wacc": 0.08,
        "name": "Banques Canadiennes"
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
    return {**bench, "source": "Secteur", "name": sector_info, "peers": "Moyenne du secteur"}

# --- 2. FONCTIONS DATA ---
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

def calculate_valuation(gr_sales, gr_fcf, gr_eps, wacc_val, ps_target, pe_target, revenue, fcf, eps, cash, debt, shares):
    # DCF
    current_fcf = fcf
    fcf_projections = [current_fcf * (1 + gr_fcf)**(i+1) for i in range(5)]
    terminal_val = (fcf_projections[-1] * 1.03) / (wacc_val - 0.03)
    pv_fcf = sum([val / ((1 + wacc_val)**(i+1)) for i, val in enumerate(fcf_projections)])
    price_dcf = ((pv_fcf + (terminal_val / ((1 + wacc_val)**5))) + cash - debt) / shares
    
    # Ventes & Earnings
    price_sales = (((revenue * ((1 + gr_sales)**5)) * ps_target) / shares) / (1.10**5)
    eps_future = eps * ((1 + gr_eps)**5)
    price_earnings = (eps_future * pe_target) / (1.10**5)
    
    return price_dcf, price_sales, price_earnings

def display_relative_analysis(current, benchmark, metric_name, group_name):
    if current <= 0:
        st.caption(f"Analyse relative impossible (ratio {metric_name} négatif ou nul).")
        return
    diff = ((current - benchmark) / benchmark) * 100
    if diff < -10: box = st.success; status = "Sous-évalué 🟢"; msg = f"décote de {abs(diff):.0f}%"
    elif diff > 10: box = st.error; status = "Surévalué 🔴"; msg = f"prime de {diff:.0f}%"
    else: box = st.warning; status = "Juste Valeur 🟡"; msg = "aligné"
    box(f"**🔍 Analyse Relative :** {metric_name} actuel **{current:.1f}x** vs Indice **{benchmark}x**.\n\n"
        f"👉 **Verdict : {status}** ({msg} vs {group_name}).")

# --- 3. INTERFACE ---

# >>> NOUVEAU SYSTÈME DE RECHERCHE <<<
st.subheader("Rechercher une entreprise")
col_search, col_manual = st.columns([2, 1])

# On utilise une selectbox pour la recherche rapide
choice = st.selectbox(
    "Choisir une entreprise populaire :", 
    TICKER_DB, 
    index=2 # Met Microsoft par défaut (plus user friendly)
)

ticker_final = "MSFT" # Valeur par défaut de sécurité

if "Autre" in choice:
    # Mode Manuel
    ticker_input = st.text_input("Ou tapez le symbole ici (ex: AMD, GOOGL)", value="").upper()
    if ticker_input:
        ticker_final = ticker_input
else:
    # Mode Liste (Extraction du ticker avant le tiret)
    # Ex: "MSFT - Microsoft" -> "MSFT"
    if "-" in choice:
        ticker_final = choice.split("-")[0].strip()

# On affiche le ticker sélectionné pour confirmer
st.caption(f"Analyse en cours pour : **{ticker_final}**")
st.divider()

# --- EXÉCUTION DE L'ANALYSE ---
if ticker_final:
    bs, inc, cf, info = get_financial_data(ticker_final)
    
    if bs is None or inc.empty:
        st.error(f"Données introuvables pour {ticker_final}. Vérifiez le symbole.")
    else:
        # DATA PREP
        raw_sector = info.get('sector', 'Default')
        bench_data = get_benchmark_data(ticker_final, raw_sector)
        
        # AIDE
        with st.expander(f"💡 Aide : {bench_data['name']}", expanded=True):
            if bench_data['source'] == "Comparables": st.write(f"**Pairs :** {bench_data['peers']}")
            else: st.write(f"**Secteur :** {raw_sector}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Croiss. Ventes", f"{bench_data['gr_sales']*100:.0f}%")
            c2.metric("Croiss. FCF", f"{bench_data['gr_fcf']*100:.0f}%")
            c3.metric("P/S Cible", f"{bench_data['ps']}x")
            c4.metric("P/E Cible", f"{bench_data.get('pe', 20)}x")

        # INPUTS
        with st.expander("⚙️ Modifier Hypothèses (Neutral)", expanded=False):
            st.markdown("##### 1. Croissance (Annuelle 5 ans)")
            c1, c2, c3 = st.columns(3)
            gr_sales_input = c1.number_input("Ventes", value=bench_data['gr_sales'], step=0.01)
            gr_fcf_input = c2.number_input("FCF", value=bench_data['gr_fcf'], step=0.01)
            gr_eps_input = c3.number_input("EPS (Bénéf.)", value=bench_data.get('gr_eps', 0.10), step=0.01)
            
            st.markdown("##### 2. Multiples de Sortie & Risque")
            c4, c5, c6 = st.columns(3)
            target_ps = c4.number_input("P/S Cible", value=bench_data['ps'], step=0.5)
            target_pe = c5.number_input("P/E Cible", value=float(bench_data.get('pe', 20.0)), step=0.5)
            wacc = c6.number_input("WACC", value=bench_data['wacc'], step=0.005)

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

        # RATIOS ACTUELS
        ps_current = market_cap / revenue_ttm if revenue_ttm > 0 else 0
        pe_current = current_price / eps_ttm if eps_ttm > 0 else 0
        pfcf_current = market_cap / fcf_ttm if fcf_ttm > 0 else 0

        # CALCULS SCENARIOS
        def run_scenario(factor_growth, factor_mult, risk_adj):
            return calculate_valuation(
                gr_sales_input * factor_growth, gr_fcf_input * factor_growth, gr_eps_input * factor_growth, 
                wacc + risk_adj, target_ps * factor_mult, target_pe * factor_mult, 
                revenue_ttm, fcf_ttm, eps_ttm, cash, debt, shares
            )

        bear_res = run_scenario(0.8, 0.8, 0.01)
        base_res = run_scenario(1.0, 1.0, 0.0)
        bull_res = run_scenario(1.2, 1.2, -0.01)

        # ONGLETS
        tabs = st.tabs(["💵 DCF (Cash)", "📈 Ventes (P/S)", "💰 Bénéfices (P/E)", "📊 Scorecard"])

        # --- 1. DCF ---
        with tabs[0]:
            st.subheader("🏷️ Prix à Payer (DCF)")
            c1, c2 = st.columns(2)
            c1.metric("Prix Actuel", f"{current_price:.2f} $")
            delta = base_res[0] - current_price
            c2.metric("Intrinsèque (Neutre)", f"{base_res[0]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")
            
            st.write("")
            display_relative_analysis(pfcf_current, bench_data.get('p_fcf', 20.0), "P/FCF", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[0]:.2f} $", delta=f"{bear_res[0]-current_price:.1f}")
            c_base.metric("🎯 Neutral", f"{base_res[0]:.2f} $", delta=f"{base_res[0]-current_price:.1f}")
            c_bull.metric("🐂 Bull", f"{bull_res[0]:.2f} $", delta=f"{bull_res[0]-current_price:.1f}")

            st.markdown("##### 📝 Thèses d'Investissement")
            st.error(f"**🐻 Bear (-20%) :** Croissance FCF ralentie à **{gr_fcf_input*0.8:.1%}**.")
            st.info(f"**🎯 Neutral :** Croissance FCF de **{gr_fcf_input:.1%}**, WACC **{wacc:.1%}**.")
            st.success(f"**🐂 Bull (+20%) :** Croissance FCF accélérée à **{gr_fcf_input*1.2:.1%}**.")

        # --- 2. VENTES ---
        with tabs[1]:
            st.subheader("🏷️ Prix à Payer (Ventes)")
            c1, c2 = st.columns(2)
            c1.metric("Prix Actuel", f"{current_price:.2f} $")
            delta = base_res[1] - current_price
            c2.metric("Intrinsèque (Neutre)", f"{base_res[1]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")
            
            st.write("")
            display_relative_analysis(ps_current, bench_data['ps'], "P/S", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[1]:.2f} $")
            c_base.metric("🎯 Neutral", f"{base_res[1]:.2f} $")
            c_bull.metric("🐂 Bull", f"{bull_res[1]:.2f} $")

            st.markdown("##### 📝 Thèses")
            st.error(f"**🐻 Bear :** Multiple P/S de **{target_ps*0.8:.1f}x**.")
            st.info(f"**🎯 Neutral :** Multiple P/S de **{target_ps:.1f}x**.")
            st.success(f"**🐂 Bull :** Multiple P/S de **{target_ps*1.2:.1f}x**.")

        # --- 3. EARNINGS ---
        with tabs[2]:
            st.subheader("🏷️ Prix à Payer (P/E)")
            c1, c2 = st.columns(2)
            c1.metric("Prix Actuel", f"{current_price:.2f} $")
            delta = base_res[2] - current_price
            c2.metric("Intrinsèque (Neutre)", f"{base_res[2]:.2f} $", delta=f"{delta:.2f} $", delta_color="normal")
            
            st.write("")
            display_relative_analysis(pe_current, bench_data.get('pe', 20), "P/E", bench_data['name'])
            st.divider()
            
            c_bear, c_base, c_bull = st.columns(3)
            c_bear.metric("🐻 Bear", f"{bear_res[2]:.2f} $")
            c_base.metric("🎯 Neutral", f"{base_res[2]:.2f} $")
            c_bull.metric("🐂 Bull", f"{bull_res[2]:.2f} $")

            st.markdown("##### 📝 Thèses")
            st.error(f"**🐻 Bear :** Croissance EPS **{gr_eps_input*0.8:.1%}**, P/E **{target_pe*0.8:.1f}x**.")
            st.info(f"**🎯 Neutral :** Croissance EPS **{gr_eps_input:.1%}**, P/E **{target_pe:.1f}x**.")
            st.success(f"**🐂 Bull :** Croissance EPS **{gr_eps_input*1.2:.1%}**, P/E **{target_pe*1.2:.1f}x**.")

        # --- 4. SCORECARD ---
        with tabs[3]:
            st.subheader("Ratios Actuels")
            r1, r2, r3 = st.columns(3)
            r1.metric("P/E (TTM)", f"{pe_current:.1f}x")
            r2.metric("P/FCF", f"{pfcf_current:.1f}x")
            net_pos = cash - debt
            color = "red" if net_pos < 0 else "green"
            r3.markdown(f"**Net Cash:** :{color}[{net_pos/1e6:.0f} M$]")
            
            st.divider()
            fcf_margin = (fcf_ttm / revenue_ttm) * 100 if revenue_ttm > 0 else 0
            fcf_yield = (fcf_ttm / market_cap) * 100 if market_cap > 0 else 0
            rule_40 = (gr_sales_input * 100) + fcf_margin
            total_return = (gr_eps_input * 100) + fcf_yield

            col_score1, col_score2 = st.columns(2)
            with col_score1:
                st.markdown("#### 🚀 Croissance")
                st.caption("Rule of 40")
                if rule_40 >= 40: st.success(f"✅ {rule_40:.1f}")
                else: st.warning(f"⚠️ {rule_40:.1f}")
                with st.expander("Guide"):
                    st.write("Croissance + Marge FCF")
                    st.write("Cible > 40")

            with col_score2:
                st.markdown("#### 🛡️ Stabilité")
                st.caption("Rendement Total")
                if total_return >= 12: st.success(f"✅ {total_return:.1f}%")
                else: st.warning(f"⚠️ {total_return:.1f}%")
                with st.expander("Guide"):
                    st.write("Yield FCF + Croissance")
                    st.write("Cible > 12%")
