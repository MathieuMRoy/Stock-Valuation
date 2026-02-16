"""
Stock Analyzer UI - Main stock analysis interface
"""
import pandas as pd
import streamlit as st

from data import TICKER_DB, get_benchmark_data
from fetchers import get_financial_data_secure, get_item_safe, get_ttm_or_latest
from fetchers.stock_analysis import get_extended_history
from valuation import calculate_valuation, solve_reverse_dcf, display_relative_analysis, compute_asset_based_value
from technical import fetch_price_history, add_indicators, bull_flag_score, plot_technical_chart, plot_fundamental_overlay
from scoring import calculate_piotroski_score, calculate_altman_z, score_out_of_10, plot_radar
from ai import ai_analyst_report


def render_stock_analyzer(api_key: str):
    """
    Render the Stock Analyzer mode UI.
    
    Args:
        api_key: Groq API key for AI analysis
    """
    st.subheader("Search for a Company")
    choice = st.selectbox("Choose a popular stock:", TICKER_DB, index=2)
    
    ticker_final = "MSFT"
    if "Other" in choice:
        ticker_input = st.text_input("Ticker", "").upper()
        if ticker_input:
            ticker_final = ticker_input
    elif "-" in choice:
        ticker_final = choice.split("-")[0].strip()

    st.caption(f"Analyzing: **{ticker_final}**")
    data = get_financial_data_secure(ticker_final)
    
    if data.get("error"):
        st.warning(f"Data fetch warning: {data['error']}")
    
    current_price = float(data.get("price", 0) or 0)

    if current_price <= 0:
        st.error("Prix introuvable. Vérifiez le ticker.")
        st.stop()

    bs = data.get("bs", pd.DataFrame())
    inc = data.get("inc", pd.DataFrame())
    cf = data.get("cf", pd.DataFrame())
    piotroski = calculate_piotroski_score(bs, inc, cf)
    
    # Shares override
    shares = float(data.get("shares_info", 0) or 0)
    st.sidebar.markdown("### 🔧 Data Override")
    manual_shares = st.sidebar.number_input("Manual Shares (Millions)", value=0.0, step=1.0)
    if manual_shares > 0:
        shares = manual_shares * 1_000_000
        st.sidebar.success(f"Using manual shares: {shares:,.0f}")
    if shares <= 1:
        shares = 1.0
        st.warning("⚠️ Share count unavailable. Enter manually in Sidebar.")

    market_cap = shares * current_price
    altman_z = calculate_altman_z(bs, inc, market_cap)

    # Revenue
    revenue_ttm = data.get("revenue_ttm", 0)
    if revenue_ttm == 0:
        revenue_ttm = get_ttm_or_latest(inc, ["TotalRevenue", "Revenue"])

    cfo_ttm = get_ttm_or_latest(cf, ["OperatingCashFlow", "Operating Cash Flow"])
    capex_ttm = abs(get_item_safe(cf, ["CapitalExpenditure", "PurchaseOfPPE"]))
    fcf_ttm = cfo_ttm - capex_ttm
    cash = get_item_safe(bs, ["CashAndCashEquivalents", "Cash"])
    debt = get_item_safe(bs, ["LongTermDebt"]) + get_item_safe(bs, ["LeaseLiabilities", "TotalLiab"])
    
    # EPS and P/E
    eps_ttm = data.get("trailing_eps", 0)
    if eps_ttm == 0:
        net_inc_ttm = get_ttm_or_latest(inc, ["NetIncome", "Net Income Common Stockholders"])
        if shares > 0:
            eps_ttm = net_inc_ttm / shares

    pe = data.get("pe_ratio", 0)
    if pe == 0 and eps_ttm > 0:
        pe = current_price / eps_ttm
        
    ps = market_cap / revenue_ttm if revenue_ttm > 0 else 0
    
    cur_sales_gr = data.get("rev_growth", 0)
    cur_eps_gr = data.get("eps_growth", 0)
    bench_data = get_benchmark_data(ticker_final, data.get("sector", "Default"))
    
    price_df = fetch_price_history(ticker_final, "1y")
    tech_df = add_indicators(price_df)
    tech = bull_flag_score(tech_df)

    metrics = {
        "ticker": ticker_final, "price": current_price, "pe": pe, "ps": ps, 
        "sales_gr": cur_sales_gr, "eps_gr": cur_eps_gr, "net_cash": cash-debt, 
        "fcf_yield": fcf_ttm/market_cap if market_cap else 0,
        "rule_40": cur_sales_gr + ((fcf_ttm/revenue_ttm) if revenue_ttm else 0)
    }
    scores = score_out_of_10(metrics, bench_data)

    # Help section
    with st.expander(f"💡 Help: {bench_data['name']} vs {ticker_final}", expanded=True):
        st.write(f"**Peers:** {bench_data.get('peers', 'N/A')}")
        st.markdown("### 🏢 Sector / Peer Averages")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peer Sales Gr.", f"{bench_data['gr_sales']:.0f}%")
        c2.metric("Peer EPS Gr.", f"{bench_data.get('gr_eps', 0):.0f}%")
        c3.metric("Peer Target P/S", f"{bench_data['ps']}x")
        c4.metric("Peer Target P/E", f"{bench_data.get('pe', 20)}x")

        st.divider()

        st.markdown(f"### 📍 {ticker_final} Current Metrics (Actual)")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Actual Sales Gr.", f"{cur_sales_gr*100:.1f}%", delta_color="off")
        c6.metric("Actual EPS Gr.", f"{cur_eps_gr*100:.1f}%", delta_color="off")
        c7.metric("Actual P/S", f"{ps:.1f}x", delta_color="off")
        c8.metric("Actual P/E", f"{pe:.1f}x", delta_color="off")

    # Assumptions
    with st.expander("⚙️ Edit Assumptions", expanded=False):
        c1, c2, c3 = st.columns(3)
        gr_sales = c1.number_input("Sales Growth %", value=float(bench_data['gr_sales']))
        gr_fcf = c2.number_input("FCF Growth %", value=float(bench_data['gr_fcf']))
        wacc = c3.number_input("WACC %", value=float(bench_data['wacc']))
        c4, c5 = st.columns(2)
        target_pe = c4.number_input("Target P/E", value=float(bench_data.get('pe', 20)))
        target_ps = c5.number_input("Target P/S", value=float(bench_data['ps']))

    # Calculation scenarios
    def run_calc(g_fac, m_fac, w_adj):
        return calculate_valuation(
            gr_sales/100*g_fac, gr_fcf/100*g_fac, 0.10, wacc/100 + w_adj, 
            target_ps*m_fac, target_pe*m_fac, 
            revenue_ttm, fcf_ttm, eps_ttm, cash, debt, shares
        )

    bear_res = run_calc(0.8, 0.8, 0.01)
    base_res = run_calc(1.0, 1.0, 0.0)
    bull_res = run_calc(1.2, 1.2, -0.01)

    # Display tabs
    st.metric("Current Price", f"{current_price:.2f} $")
    
    # Define Tabs (Reordered: Fundamentals First, Analyst Added)
    t_fund, t_dcf, t_sales, t_pe, t_assets, t_analyst, t_ins, t_tech, t_score, t_ai = st.tabs([
        "📊 Fundamentals", 
        "💵 DCF (Cash)", 
        "📈 Sales (P/S)", 
        "💰 Earnings (P/E)", 
        "🧱 Assets", 
        "🎯 Analystes",
        "👥 Insiders", 
        "📉 Tech", 
        "📊 Scorecard", 
        "🤖 AI Agent"
    ])
    
    # --- TAB 1: FUNDAMENTALS (Moved from Pos 6) ---
    with t_fund:
        st.subheader("📊 Extended Financial Dashboard")
        st.caption("Source: StockAnalysis.com")
        
        # Display Frequency - FORCE ANNUAL
        st.caption("Frequency: Annual")
        freq_dashboard = "Annual"
        
        # Fetch Extended Data ("version=3" to ensure cache bust if needed, or stick to 2)
        with st.spinner("Fetching extended historical data..."):
            sa_data = get_extended_history(ticker_final, version=2)
            
        if sa_data.get("error"):
            st.error(f"Could not load extended data: {sa_data['error']}")
        else:
            # 1. Prepare Helper Function for Data Processing
            def process_financial_metrics(inc_df, bs_df, cf_df):
                if inc_df.empty and bs_df.empty and cf_df.empty:
                    return pd.DataFrame()
                    
                # Transpose
                def clean_transpose(df):
                    if df.empty: return pd.DataFrame()
                    df_t = df.T
                    if "TTM" in df_t.index: df_t = df_t.rename(index={"TTM": "Last 12M"})
                    return df_t

                df_inc = clean_transpose(inc_df)
                df_bs = clean_transpose(bs_df)
                df_cf = clean_transpose(cf_df)
                
                # Merge
                df_all = df_inc.join(df_bs, lsuffix="_inc", rsuffix="_bs", how="outer")
                df_all = df_all.join(df_cf, rsuffix="_cf", how="outer")
                
                # Sort
                try:
                    df_all.index = df_all.index.astype(str)
                    
                    # Extract 4-digit year for proper sorting
                    df_all["_sort_val"] = pd.to_numeric(df_all.index.str.extract(r'(\d{4})')[0], errors='coerce')
                    
                    # Sort Ascending (Oldest -> Newest)
                    df_all = df_all.sort_values("_sort_val", ascending=True)
                    df_all = df_all.drop(columns=["_sort_val"])
                except Exception as e:
                    pass

                # Helper col finder
                def get_col(df, candidates):
                    for c in df.columns:
                        for cand in candidates:
                            if cand.lower().replace(" ", "") == str(c).lower().replace(" ", ""):
                                return c
                    return None
                    
                # --- Calculations ---
                
                # FCF
                col_ocf = get_col(df_all, ["Operating Cash Flow", "Cash From Operations"])
                col_capex = get_col(df_all, ["Capital Expenditures", "CapEx"])
                if col_ocf and col_capex:
                     df_all["Free Cash Flow"] = df_all[col_ocf] - df_all[col_capex].abs()

                # Margins
                col_rev = get_col(df_all, ["Total Revenue", "Revenue"])
                col_gp = get_col(df_all, ["Gross Profit"])
                col_op = get_col(df_all, ["Operating Income"])
                col_ni = get_col(df_all, ["Net Income", "Net Income Common Stockholders"])
                
                if col_rev:
                    if col_gp: df_all["Gross Margin %"] = (df_all[col_gp] / df_all[col_rev]) * 100
                    if col_op: df_all["Operating Margin %"] = (df_all[col_op] / df_all[col_rev]) * 100
                    if col_ni: df_all["Net Margin %"] = (df_all[col_ni] / df_all[col_rev]) * 100

                # ROIC Proxy
                col_equity = get_col(df_all, ["Total Stockholders' Equity", "Total Equity", "Shareholders' Equity"])
                col_debt = get_col(df_all, ["Total Debt"])
                col_cash = get_col(df_all, ["Cash & Equivalents", "Cash and Equivalents", "Total Cash & Equivalents"])
                
                if col_op and col_equity and col_debt and col_cash:
                    nopat = df_all[col_op] * 0.79
                    invested_capital = df_all[col_equity] + df_all[col_debt] - df_all[col_cash]
                    df_all["ROIC %"] = (nopat / invested_capital) * 100
                
                # Shares
                # (We assign this to specific column name to find it later easily)
                col_shares = get_col(df_all, ["Shares Outstanding (Basic)", "Shares Outstanding", "Weighted Average Shares"])
                if col_shares:
                    df_all["Shares Outstanding"] = df_all[col_shares]

                # Liquidity Ratios
                col_ca = get_col(df_all, ["Total Current Assets", "Current Assets"])
                col_cl = get_col(df_all, ["Total Current Liabilities", "Current Liabilities"])
                col_inv = get_col(df_all, ["Inventory", "Inventories"])
                col_ta = get_col(df_all, ["Total Assets"])
                col_opex = get_col(df_all, ["Total Operating Expenses", "Operating Expenses"])
                
                if not col_opex:
                     c_cogs = get_col(df_all, ["Cost of Revenue", "COGS"])
                     c_sgna = get_col(df_all, ["Selling, General and Administrative", "SG&A", "Operating Expenses"])
                     if c_cogs and c_sgna:
                         df_all["Total_OpEx_Calc"] = df_all[c_cogs] + df_all[c_sgna]
                         col_opex = "Total_OpEx_Calc"
                
                if col_ca and col_cl:
                    df_all["Current Ratio (Fonds de roulement)"] = df_all[col_ca] / df_all[col_cl]
                    df_all["Net Working Capital (FRN)"] = df_all[col_ca] - df_all[col_cl]
                    if col_ta:
                        df_all["FRN / Total Assets"] = (df_all["Net Working Capital (FRN)"] / df_all[col_ta]) * 100
                    if col_inv:
                        df_all["Quick Ratio (Trésorerie)"] = (df_all[col_ca] - df_all[col_inv]) / df_all[col_cl]
                    if col_cash:
                        df_all["Cash Ratio (Liquidité immédiate)"] = df_all[col_cash] / df_all[col_cl]
                    if col_opex:
                        daily_opex = df_all[col_opex] / 365
                        df_all["Defensive Interval (Days)"] = df_all[col_ca] / daily_opex
                        
                # Clean up Infs and NaNs
                import numpy as np
                df_all = df_all.replace([np.inf, -np.inf], np.nan)
                
                return df_all

            # 2. Process Datasets (Annual Only)
            df_annual = process_financial_metrics(
                sa_data.get("inc_a", pd.DataFrame()),
                sa_data.get("bs_a", pd.DataFrame()),
                sa_data.get("cf_a", pd.DataFrame())
            )
            
            df_main = df_annual
            
            if df_main.empty:
                st.warning("No data found.")
            else:
                pass

                # Helper Col Finder
                def get_col(df, candidates):
                    for c in df.columns:
                         for cand in candidates:
                             if cand.lower().replace(" ", "") == str(c).lower().replace(" ", ""):
                                 return c
                    return None

                # Plot Grid
                metrics_grid = [
                    {"label": "Revenue", "match": ["Revenue", "Total Revenue"], "color": "#ffaa00"}, 
                    {"label": "Net Income", "match": ["Net Income", "Profit"], "color": "#00aaff"}, 
                    {"label": "EBITDA", "match": ["EBITDA"], "color": "#aa00ff"}, 
                    {"label": "EPS", "match": ["EPS (Basic)", "Earnings Per Share"], "color": "#ffff00"}, 
                    {"label": "Free Cash Flow", "match": ["Free Cash Flow"], "color": "#00ff00"},
                    {"label": "Dividends Paid", "match": ["Dividends Paid"], "color": "#ff00aa"},
                ]
                
                cols = st.columns(3)
                for i, m in enumerate(metrics_grid):
                    col = cols[i % 3]
                    found_col = get_col(df_main, m["match"])
                    with col:
                        if found_col:
                            st.markdown(f"**{m['label']}**")
                            st.bar_chart(df_main[found_col], color=m["color"], height=200)
                        else:
                            st.info(f"{m['label']} data not available")

                # Restore Balance Sheet Health
                st.markdown("---")
                st.markdown("### 💰 Balance Sheet Health")
                c_bal1, c_bal2 = st.columns(2)
                
                col_cash = get_col(df_main, ["Cash & Equivalents", "Cash and Equivalents", "Total Cash & Equivalents"])
                col_debt = get_col(df_main, ["Total Debt"])
                
                with c_bal1:
                     st.markdown("**Cash vs Total Debt**")
                     if col_cash and col_debt:
                         st.bar_chart(df_main[[col_cash, col_debt]], color=["#00ff00", "#ff0000"], height=250)
                     else:
                         st.info("Balance sheet data missing for Cash/Debt comparison")

                # Restore Advanced Metrics
                st.markdown("---")
                c_adv_title, c_adv_freq = st.columns([3, 1])
                with c_adv_title:
                    st.markdown("### 📈 Advanced Metrics (Margins, Efficiency, Liquidity)")
                with c_adv_freq:
                    st.caption("Annual")
                
                df_adv = df_annual
                
                if df_adv.empty:
                    st.warning("No data for Advanced Metrics.")
                else:
                    dynamic_options = [
                        "Gross Margin %", "Operating Margin %", "Net Margin %", 
                        "ROIC %", "Shares Outstanding",
                        "Current Ratio (Fonds de roulement)", "Quick Ratio (Trésorerie)", "Cash Ratio (Liquidité immédiate)",
                        "Net Working Capital (FRN)", "FRN / Total Assets", "Defensive Interval (Days)"
                    ]
                    
                    metrics_descriptions = {
                        "Current Ratio (Fonds de roulement)": "Actif CT / Passif CT. Capacité à payer les dettes courantes. (>1 visé)",
                        "Quick Ratio (Trésorerie)": "(Actif CT - Stocks) / Passif CT. Capacité à payer sans vendre les stocks (plus prudent).",
                        "Cash Ratio (Liquidité immédiate)": "Cash / Passif CT. Cash disponible immédiatement.",
                        "Net Working Capital (FRN)": "Actif CT - Passif CT. Marge de sécurité en dollars.",
                        "FRN / Total Assets": "Part de l'actif total qui représente la liquidité nette.",
                        "Defensive Interval (Days)": "Actif CT / (Dépenses/Jour). Jours de survie sans aucun revenu.",
                        "Gross Margin %": "Marge Brute (Pricing Power).",
                        "Operating Margin %": "Marge Opérationnelle (Efficacité).",
                        "Net Margin %": "Marge Nette (Profitabilité finale).",
                        "ROIC %": "Retour sur Capital Investi (Qualité du business).",
                        "Shares Outstanding": "Nombre d'actions (Baisse = Rachats = Positif)."
                    }
                    
                    available_dynamic = [opt for opt in dynamic_options if opt in df_adv.columns]
                    
                    selected_dynamic = st.multiselect("Select additional metrics to visualize:", available_dynamic, default=["Operating Margin %", "Shares Outstanding"] if "Shares Outstanding" in available_dynamic else [])
                    
                    if selected_dynamic:
                        dyn_cols = st.columns(2)
                        for i, metric_name in enumerate(selected_dynamic):
                            col = dyn_cols[i % 2]
                            with col:
                                st.markdown(f"**{metric_name}**")
                                color = "#00aaff"
                                if "Margin" in metric_name: color = "#ffaa00"
                                if "ROIC" in metric_name: color = "#aa00ff"
                                if "Shares" in metric_name: color = "#ff00aa" 
                                if "Ratio" in metric_name: color = "#00ffaa"
                                if "FRN" in metric_name: color = "#00ffaa"
                                if "Defensive" in metric_name: color = "#55aaff"
                                
                                st.bar_chart(df_adv[metric_name], color=color, height=250)
                                desc = metrics_descriptions.get(metric_name)
                                if desc: st.caption(desc)

    with t_dcf:
        st.subheader("💵 Buy Price (DCF)")
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[0]:.2f} $", delta=f"{base_res[0]-current_price:.2f}")
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[0]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[0]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[0]:.2f} $")
        
        st.markdown("##### Reverse DCF")
        implied_g = solve_reverse_dcf(current_price, fcf_ttm, wacc/100, shares, cash, debt)
        st.metric("Market Implied Growth", f"{implied_g*100:.1f}%")
        
        # Sensitivity matrix
        st.markdown("##### 🌡️ Sensitivity Matrix (Price vs Growth & WACC)")
        sens_wacc = [wacc-1, wacc-0.5, wacc, wacc+0.5, wacc+1]
        sens_growth = [gr_fcf-2, gr_fcf-1, gr_fcf, gr_fcf+1, gr_fcf+2]
        res_matrix = []
        for w in sens_wacc:
            row_vals = []
            for g in sens_growth:
                val, _, _ = calculate_valuation(0, g/100, 0, w/100, 0, 0, revenue_ttm, fcf_ttm, 0, cash, debt, shares)
                row_vals.append(val)
            res_matrix.append(row_vals)
        df_sens = pd.DataFrame(res_matrix, index=[f"WACC {w:.1f}%" for w in sens_wacc], columns=[f"Gr {g:.1f}%" for g in sens_growth])
        st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f} $"))

    with t_sales:
        st.subheader("📈 Buy Price (Sales)")
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[1]:.2f} $", delta=f"{base_res[1]-current_price:.2f}")
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[1]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[1]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[1]:.2f} $")
        st.write("")
        display_relative_analysis(ps, float(bench_data.get('ps', 3)), "P/S", bench_data['name'])

    with t_pe:
        st.subheader("💰 Buy Price (P/E)")
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[2]:.2f} $", delta=f"{base_res[2]-current_price:.2f}")
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[2]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[2]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[2]:.2f} $")
        st.write("")
        display_relative_analysis(pe, float(bench_data.get('pe', 20)), "P/E", bench_data['name'])

    with t_assets:
        st.subheader("🧱 Asset Based Value")
        ab = compute_asset_based_value(bs, shares)
        c1, c2 = st.columns(2)
        c1.metric("NAV / Share", f"{ab['nav_ps']:.2f} $")
        c2.metric("Tangible NAV", f"{ab['tnav_ps']:.2f} $")
        st.caption(ab["notes"])

    with t_analyst:
        st.subheader("🎯 Analyst Ratings & Price Targets")
        
        # We need fresh info for analysts
        try:
            import yfinance as yf
            from datetime import timedelta, datetime
            
            ticker_obj = yf.Ticker(ticker_final)
            info_an = ticker_obj.info
            
            # --- Consensul & Ratings ---
            rec_key = info_an.get("recommendationKey", "none").replace("_", " ").title()
            num_analysts = info_an.get("numberOfAnalystOpinions", 0)
            
            c_an1, c_an2 = st.columns([1, 2])
            
            with c_an1:
                st.metric("Consensus", rec_key)
                st.metric("Analysts count", num_analysts)
                
            with c_an2:
                # Ratings Breakdown
                try:
                    rec_sum = ticker_obj.recommendations_summary
                    if rec_sum is not None and not rec_sum.empty:
                        # usually columns: period, strongBuy, buy, hold, sell, strongSell
                        # taking the latest period (row 0)
                        latest_rec = rec_sum.iloc[0]
                        
                        rec_data = {
                            "Strong Buy": latest_rec.get("strongBuy", 0),
                            "Buy": latest_rec.get("buy", 0),
                            "Hold": latest_rec.get("hold", 0),
                            "Sell": latest_rec.get("sell", 0),
                            "Strong Sell": latest_rec.get("strongSell", 0)
                        }
                        
                        df_rec = pd.DataFrame([rec_data]).T
                        df_rec.columns = ["Count"]
                        
                        # Plot Horizontal Bar
                        st.bar_chart(df_rec, horizontal=True, color="#aa00ff")
                    else:
                        st.info("Detailed ratings breakdown not available.")
                except Exception as e:
                    st.info(f"Could not load ratings breakdown: {e}")

            st.divider()
            
            # --- Price Targets ---
            st.subheader("🔮 12-Month Price Forecast")
            
            tgt_low = info_an.get("targetLowPrice")
            tgt_mean = info_an.get("targetMeanPrice")
            tgt_high = info_an.get("targetHighPrice")
            cur_price = info_an.get("currentPrice", current_price)
            
            if tgt_mean:
                c_t1, c_t2, c_t3 = st.columns(3)
                
                def fmt_upside(target, current):
                    if not target or not current: return "0%"
                    up = ((target - current) / current) * 100
                    # Return simple signed string; Streamlit metric handles the coloring (Red for negative)
                    return f"{up:+.2f}%"

                c_t1.metric("Low Target", f"{tgt_low:.2f} $" if tgt_low else "N/A", fmt_upside(tgt_low, cur_price))
                c_t2.metric("Average Target", f"{tgt_mean:.2f} $" if tgt_mean else "N/A", fmt_upside(tgt_mean, cur_price))
                c_t3.metric("High Target", f"{tgt_high:.2f} $" if tgt_high else "N/A", fmt_upside(tgt_high, cur_price))
                
                # --- Chart (Native Streamlit) ---
                # Combine Historical Data with Future Projections
                
                # 1. Get History (1y)
                df_use = tech_df.copy() if not tech_df.empty else price_df.copy()
                
                if "Date" in df_use.columns:
                    df_use["Date"] = pd.to_datetime(df_use["Date"])
                    df_use = df_use.set_index("Date")
                
                hist_series = df_use["Close"]
                
                if not hist_series.empty:
                    last_date = hist_series.index[-1]
                    if isinstance(last_date, (pd.Timestamp, datetime)):
                        last_price = hist_series.iloc[-1]
                        future_date = last_date + timedelta(days=365)
                        
                        joined_index = hist_series.index.union(pd.Index([future_date]))
                        df_final = pd.DataFrame(index=joined_index)
                        
                        df_final.loc[hist_series.index, "History"] = hist_series
                        
                        df_final.loc[last_date, "High Target"] = last_price
                        df_final.loc[future_date, "High Target"] = tgt_high
                        
                        df_final.loc[last_date, "Mean Target"] = last_price
                        df_final.loc[future_date, "Mean Target"] = tgt_mean
                        
                        df_final.loc[last_date, "Low Target"] = last_price
                        df_final.loc[future_date, "Low Target"] = tgt_low
                        
                        st.caption("Price History (Blue) vs Analyst Targets (Colored Lines)")
                        st.line_chart(df_final, color=["#00aaff", "#00ff00", "#ffff00", "#ff0000"])
                    else:
                        st.warning("Could not parse dates for historical chart.")
                
            else:
                st.warning("No price target data available.")
                
            # --- Analyst Actions List ---
            st.divider()
            st.subheader("📋 Détails des Analystes (Upgrades / Downgrades)")
            
            try:
                upgrades = ticker_obj.upgrades_downgrades
                if upgrades is not None and not upgrades.empty:
                    # Clean and Format
                    # 1. Sort Descending
                    upgrades = upgrades.sort_index(ascending=False).head(20)
                    
                    # 2. Reset Index to get Date column
                    upgrades = upgrades.reset_index() # GradeDate becomes a column
                    
                    # 3. Rename Columns
                    # typical cols: GradeDate, Firm, ToGrade, FromGrade, Action
                    # Map to something user-friendly
                    col_map = {
                        "GradeDate": "Date",
                        "Firm": "Firme",
                        "ToGrade": "Nouvelle Note",
                        "FromGrade": "Ancienne Note",
                        "Action": "Action"
                    }
                    upgrades = upgrades.rename(columns=col_map)
                    
                    # 4. Filter columns if needed (keep key ones)
                    cols_to_show = ["Date", "Firme", "Action", "Nouvelle Note", "Ancienne Note"]
                    # Ensure they exist (sometimes Yahoo varies)
                    cols_to_show = [c for c in cols_to_show if c in upgrades.columns]
                    upgrades = upgrades[cols_to_show]
                    
                    # 5. Format Date (YYYY-MM-DD)
                    if "Date" in upgrades.columns:
                        upgrades["Date"] = pd.to_datetime(upgrades["Date"]).dt.strftime('%Y-%m-%d')

                    # 6. Styling Function
                    def style_grades(val):
                        if not isinstance(val, str): return ''
                        v = val.lower()
                        # Green
                        if any(x in v for x in ['buy', 'outperform', 'overweight', 'positive', 'strong buy']):
                            return 'color: #2ecc71; font-weight: bold;'
                        # Red
                        if any(x in v for x in ['sell', 'underperform', 'underweight', 'negative', 'reduce']):
                            return 'color: #e74c3c; font-weight: bold;'
                        # Orange/Yellow
                        if any(x in v for x in ['hold', 'neutral', 'equal', 'market', 'sector']):
                            return 'color: #f39c12;'
                        return ''

                    # Apply Style
                    # Check if 'Nouvelle Note' exists
                    target_col = "Nouvelle Note"
                    if target_col in upgrades.columns:
                        styled_df = upgrades.style.map(style_grades, subset=[target_col])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(upgrades, use_container_width=True, hide_index=True)
                        
                else:
                    st.info("No specific analyst upgrades/downgrades data found.")
            except Exception as e:
                st.info(f"Could not load analyst list: {e}")

        except Exception as e:
            st.error(f"Error loading analyst data: {e}")

    with t_ins:
        st.subheader("👥 Insider Trading")
        if not data['insiders'].empty:
            st.dataframe(data['insiders'].head(10))
        else:
            st.info("No insider data found.")

    with t_tech:
        st.subheader("📉 Technical Analysis")
        c1, c2 = st.columns(2)
        c1.metric("Bull Flag Score", f"{tech['score']}/10")
        c2.metric("Pattern", "Bull Flag" if tech['is_bull_flag'] else "None")
        
        # UI Controls for Chart
        col_overlays, col_indicators = st.columns(2)
        with col_overlays:
            selected_overlays = st.multiselect(
                "Price Overlays",
                ["SMA20", "SMA50", "SMA100", "SMA200", "Bollinger Bands"],
                default=["SMA50", "SMA200"]
            )
        with col_indicators:
            selected_indicators = st.multiselect(
                "Subplot Indicators",
                ["Volume", "RSI", "MACD", "ATR", "Stochastic", "OBV"],
                default=["Volume", "RSI"]
            )
            
        plot_technical_chart(tech_df, ticker_final, selected_overlays, selected_indicators)

        # --- Short Interest Section ---
        st.divider()
        st.subheader("📊 Short Interest")
        
        # 1. Current Short Data from yfinance info
        try:
            import yfinance as yf
            ticker_obj_short = yf.Ticker(ticker_final)
            info_short = ticker_obj_short.info or {}
            
            short_pct = info_short.get("shortPercentOfFloat")
            short_ratio = info_short.get("shortRatio")  # Days to Cover
            short_count = info_short.get("sharesShort")
            short_prev = info_short.get("sharesShortPriorMonth")
            
            if short_pct is not None or short_ratio is not None:
                c_s1, c_s2, c_s3, c_s4 = st.columns(4)
                
                with c_s1:
                    if short_pct is not None:
                        st.metric("Short % of Float", f"{short_pct*100:.2f}%")
                    else:
                        st.metric("Short % of Float", "N/A")
                        
                with c_s2:
                    if short_ratio is not None:
                        st.metric("Days to Cover", f"{short_ratio:.2f}")
                    else:
                        st.metric("Days to Cover", "N/A")
                        
                with c_s3:
                    if short_count is not None:
                        st.metric("Shares Short", f"{short_count:,.0f}")
                    else:
                        st.metric("Shares Short", "N/A")
                        
                with c_s4:
                    if short_count is not None and short_prev is not None and short_prev > 0:
                        change_pct = ((short_count - short_prev) / short_prev) * 100
                        st.metric("vs Prior Month", f"{short_count:,.0f}", delta=f"{change_pct:+.1f}%", delta_color="inverse")
                    else:
                        st.metric("vs Prior Month", "N/A")
            else:
                st.info("Short interest data not available for this ticker.")
        except Exception as e:
            st.info(f"Could not load current short data: {e}")
        
        # 2. Historical Short Interest from Nasdaq API
        try:
            from fetchers.short_interest import get_historical_short_interest
            
            df_short = get_historical_short_interest(ticker_final)
            
            if not df_short.empty:
                st.markdown("##### 📈 Historical Short Interest (Nasdaq)")
                
                # Clean numeric columns
                import numpy as np
                for col in ["Short Interest", "Avg Daily Volume", "Days to Cover"]:
                    if col in df_short.columns:
                        df_short[col] = pd.to_numeric(
                            df_short[col].astype(str).str.replace(",", ""), errors="coerce"
                        )
                
                # Compute Short % of Float using yfinance float shares
                float_shares = info_short.get("floatShares") if 'info_short' in dir() else None
                if float_shares is None:
                    try:
                        float_shares = yf.Ticker(ticker_final).info.get("floatShares")
                    except:
                        pass
                
                if float_shares and float_shares > 0 and "Short Interest" in df_short.columns:
                    df_short["Short % of Float"] = (df_short["Short Interest"] / float_shares) * 100
                
                # Chart: Short % of Float over time (PRIMARY CHART)
                if "Short % of Float" in df_short.columns and "Date" in df_short.columns:
                    spf_df = df_short.set_index("Date")[["Short % of Float"]].dropna()
                    if not spf_df.empty:
                        st.markdown("**Short % of Float — Évolution**")
                        st.line_chart(spf_df, color="#ff3366", height=280)
                        
                        # Min / Max / Current labels
                        c_min, c_max, c_cur = st.columns(3)
                        c_min.metric("Min (période)", f"{spf_df['Short % of Float'].min():.2f}%")
                        c_max.metric("Max (période)", f"{spf_df['Short % of Float'].max():.2f}%")
                        c_cur.metric("Dernier", f"{spf_df['Short % of Float'].iloc[-1]:.2f}%")
                
                # Chart: Short Interest (absolute count) over time
                if "Short Interest" in df_short.columns and "Date" in df_short.columns:
                    chart_df = df_short.set_index("Date")[["Short Interest"]].dropna()
                    if not chart_df.empty:
                        st.markdown("**Short Interest (nombre d'actions)**")
                        st.line_chart(chart_df, color="#ff6600", height=250)
                
                # Chart: Days to Cover over time
                if "Days to Cover" in df_short.columns and "Date" in df_short.columns:
                    dtc_df = df_short.set_index("Date")[["Days to Cover"]].dropna()
                    if not dtc_df.empty:
                        st.markdown("**Days to Cover**")
                        st.area_chart(dtc_df, color="#aa00ff", height=200)
                
                # Raw data table
                with st.expander("📋 Raw Short Interest Data"):
                    display_cols = [c for c in ["Date", "Short Interest", "Short % of Float", "Avg Daily Volume", "Days to Cover"] if c in df_short.columns]
                    st.dataframe(df_short[display_cols].sort_values("Date", ascending=False).head(24), use_container_width=True, hide_index=True)
            else:
                st.caption("Historical short interest data not available from Nasdaq for this ticker.")
        except Exception as e:
            st.caption(f"Could not load historical short interest: {e}")

    # Fund (handled above)
    # with tabs[6]: ...
    
    # Fund (handled above)
    
    with t_score:
        st.subheader("📊 Scorecard Pro")
        c1, c2, c3 = st.columns(3)
        c1.metric("Health", f"{scores['health']}/10")
        c2.metric("Growth", f"{scores['growth']}/10")
        c3.metric("Value", f"{scores['valuation']}/10")
        fig = plot_radar(scores, tech['score'])
        if fig:
            st.pyplot(fig)
        st.markdown(f"**Piotroski F-Score:** {piotroski if piotroski else 'N/A'}/9")
        st.markdown(f"**Altman Z-Score:** {altman_z:.2f}")

    with t_ai:
        st.subheader("🤖 AI Analyst (Groq)")
        if st.button("✨ Generate Full Report"):
            with st.spinner("Analyzing..."):
                rep, err = ai_analyst_report(metrics, bench_data, scores, tech, api_key)
                if rep:
                    st.markdown(rep)
                else:
                    st.error(err)
