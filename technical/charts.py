"""
Technical Charts - Price chart visualization
"""
import pandas as pd
import streamlit as st

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


def plot_technical_chart(df: pd.DataFrame, ticker: str, overlays: list = None, indicators: list = None):
    """
    Plot a technical price chart with dynamic overlays and subplots.
    
    Args:
        df: DataFrame with price data and indicators
        ticker: Stock ticker symbol
        overlays: List of overlays to plot on main chart ["SMA20", "BB", etc.]
        indicators: List of indicators to plot in subplots ["RSI", "MACD", "Volume", etc.]
    """
    if not MATPLOTLIB_OK:
        st.warning("⚠️ 'matplotlib' n'est pas installé.")
        return
    
    if df is None or df.empty:
        st.warning("No price history to chart.")
        return

    overlays = overlays or []
    indicators = indicators or []
    
    # Calculate total subplots: 1 main price chart + 1 for each selected indicator
    n_subplots = 1 + len(indicators)
    
    # Adjust figure size based on number of subplots
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 5 + 2*len(indicators)), sharex=True)
    
    # Handle single axis case (no subplots)
    if n_subplots == 1:
        axes = [axes]
    
    # --- MAIN CHART (Price + Overlays) ---
    ax_price = axes[0]
    ax_price.plot(df["Date"], df["Close"], linewidth=2, label="Close", color="#1f77b4")
    
    # Overlays
    if "SMA20" in overlays and "SMA20" in df.columns:
        ax_price.plot(df["Date"], df["SMA20"], linewidth=1, label="SMA20", color="orange")
    if "SMA50" in overlays and "SMA50" in df.columns:
        ax_price.plot(df["Date"], df["SMA50"], linewidth=1, label="SMA50", color="green")
    if "SMA100" in overlays and "SMA100" in df.columns:
        ax_price.plot(df["Date"], df["SMA100"], linewidth=1, label="SMA100", color="purple")
    if "SMA200" in overlays and "SMA200" in df.columns:
        ax_price.plot(df["Date"], df["SMA200"], linewidth=1, label="SMA200", color="red")
        
    if "Bollinger Bands" in overlays and "BB_Upper" in df.columns:
        ax_price.plot(df["Date"], df["BB_Upper"], linewidth=0.8, color="gray", alpha=0.5, label="BB")
        ax_price.plot(df["Date"], df["BB_Lower"], linewidth=0.8, color="gray", alpha=0.5)
        ax_price.fill_between(df["Date"], df["BB_Upper"], df["BB_Lower"], color="gray", alpha=0.1)

    ax_price.set_title(f"{ticker} — Price History")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.3)

    # --- SUBPLOTS (Indicators) ---
    for i, ind in enumerate(indicators):
        ax = axes[i + 1]
        
        if ind == "RSI":
            ax.plot(df["Date"], df["RSI14"], color="purple", label="RSI (14)")
            ax.axhline(70, linestyle="--", color="red", linewidth=0.8)
            ax.axhline(30, linestyle="--", color="green", linewidth=0.8)
            ax.fill_between(df["Date"], 70, 30, color="purple", alpha=0.05)
            ax.set_ylabel("RSI")
            
        elif ind == "MACD":
            ax.plot(df["Date"], df["MACD"], label="MACD", color="blue")
            ax.plot(df["Date"], df["MACDSignal"], label="Signal", color="orange")
            ax.bar(df["Date"], df["MACD"]-df["MACDSignal"], label="Hist", color="gray", alpha=0.3)
            ax.set_ylabel("MACD")
            
        elif ind == "Volume":
            colors = ["green" if o >= c else "red" for o, c in zip(df["Open"], df["Close"])] if "Open" in df.columns else "blue"
            ax.bar(df["Date"], df["Volume"], color=colors, alpha=0.6, label="Volume")
            ax.set_ylabel("Volume")
            
        elif ind == "ATR":
            ax.plot(df["Date"], df["ATR"], color="brown", label="ATR (14)")
            ax.set_ylabel("ATR")
            
        elif ind == "Stochastic":
            ax.plot(df["Date"], df["Stoch_K"], label="%K", color="blue")
            ax.plot(df["Date"], df["Stoch_D"], label="%D", color="orange")
            ax.axhline(80, linestyle="--", color="red", linewidth=0.8)
            ax.axhline(20, linestyle="--", color="green", linewidth=0.8)
            ax.set_ylabel("Stoch")
            
        elif ind == "OBV":
            ax.plot(df["Date"], df["OBV"], color="teal", label="OBV")
            ax.set_ylabel("OBV")
            
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def plot_fundamental_overlay(price_df: pd.DataFrame, fund_df: pd.DataFrame, metric: str, ticker: str):
    """
    Plot stock price overlaying a fundamental metric (bar chart).
    
    Args:
        price_df: DataFrame with "Date" and "Close"
        fund_df: DataFrame with dates as columns (or index) and metrics as rows
        metric: Name of the metric to plot (e.g., "Total Revenue")
        ticker: Stock symbol
    """
    if not MATPLOTLIB_OK:
        st.warning("⚠️ 'matplotlib' n'est pas installé.")
        return
        
    if price_df is None or price_df.empty:
        st.warning("No price data.")
        return
        
    if fund_df is None or fund_df.empty:
        st.warning(f"No fundamental data for {metric}.")
        return

    # Prepare Fundamental Data
    # Yahoo Finance returns dates as columns. StockAnalysis too. We need to Transpose.
    df_fund = fund_df.copy()
    
    # Check if we need to transpose: if columns look like dates/years/TTM
    first_col = str(df_fund.columns[0])
    if isinstance(df_fund.columns[0], pd.Timestamp) or "20" in first_col or "TTM" in first_col or "FY" in first_col:
         df_fund = df_fund.T # Transpose to have Dates as Index
    
    # Debug: Check columns after potential transpose
    # st.write(df_fund.head())
    
    # Check if metric exists
    # Fuzzy match for metric name
    found_col = None
    metric_clean = metric.lower().replace(" ", "")
    for col in df_fund.columns:
        if metric_clean in str(col).lower().replace(" ", ""):
            found_col = col
            break
    
    if not found_col:
        st.warning(f"Metric '{metric}' not found. Available: {list(df_fund.columns)[:10]}...")
        return
        
    # Align dates
    # Handle StockAnalysis formats: "TTM", "FY 2024", "2024"
    if "TTM" in df_fund.index:
        # Replace TTM with today
        df_fund = df_fund.rename(index={"TTM": pd.Timestamp.now().normalize()})
    
    # Clean "FY " prefix
    df_fund.index = df_fund.index.astype(str).str.replace("FY ", "", regex=False)
    # Removing any remaining string artifacts if necessary
    
    # Convert to datetime with coerce to handle any garbage
    df_fund.index = pd.to_datetime(df_fund.index, errors='coerce')
    df_fund = df_fund.dropna().sort_index()
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Bar Chart for Fundamentals (Axis 1)
    bars = ax1.bar(df_fund.index, df_fund[found_col], color="skyblue", alpha=0.5, label=found_col, width=20)
    # Adjust width dynamically? 20 days is a guess. If annual, maybe 300.
    if len(df_fund) > 1:
        delta = (df_fund.index[1] - df_fund.index[0]).days
        if delta > 300: # Annual
             bars[0].set_width(300)
             for b in bars: b.set_width(300)
        else: # Quarterly
             bars[0].set_width(80)
             for b in bars: b.set_width(80)
             
    ax1.set_ylabel(metric, color="skyblue", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="skyblue")
    
    # Line Chart for Price (Axis 2)
    ax2 = ax1.twinx()
    ax2.plot(price_df["Date"], price_df["Close"], color="#1f77b4", linewidth=2, label="Stock Price")
    ax2.set_ylabel("Stock Price ($)", color="#1f77b4", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="#1f77b4")
    
    plt.title(f"{ticker}: {metric} vs Price Evolution")
    
    # Combined Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    
    ax1.grid(True, alpha=0.15)
    st.pyplot(fig)
