import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Curated list of popular stocks to track for earnings
# (Fetching *all* market earnings is slow/impossible without premium API)
POPULAR_TICKERS = [
    # US Tech & Mega Cap
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "AMD", "INTC",
    "CRM", "ADBE", "ORCL", "IBM", "CSCO", "UBER", "ABNB", "PLTR", "SNOW", "SHOP", "SQ", "PYPL",
    # US Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP",
    # US Consumer & Pharma
    "JNJ", "PFE", "LLY", "MRK", "KO", "PEP", "MCD", "SBUX", "WMT", "TGT", "COST", "DIS", "CMCSA",
    # US Industrial/Energy
    "XOM", "CVX", "BA", "CAT", "DE", "GE", "F", "GM",
    # TSX (Canada)
    "RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO", # Banks
    "SHOP.TO", "CSU.TO", "GIB-A.TO", # Tech
    "CNQ.TO", "SU.TO", "CVE.TO", "ENB.TO", "TRP.TO", # Energy
    "CNR.TO", "CP.TO", # Rail
    "BCE.TO", "T.TO", "RCI-B.TO", # Telecom
    "ATD.TO", "DOL.TO", "L.TO", "WN.TO", "SAP.TO", # Consumer
    "BAM.TO", "BN.TO", "POW.TO" # Finance/Asset
]

@st.cache_data(ttl=3600*4) # Cache for 4 hours
def get_upcoming_earnings(tickers):
    """
    Fetch next earnings date for a list of tickers.
    Returns DataFrame: [Ticker, Date, Estimate, Revenue]
    """
    data = []
    
    # Progress bar
    progress_text = "Scanning market for upcoming earnings..."
    my_bar = st.progress(0, text=progress_text)
    
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        try:
            # Update progress
            if i % 10 == 0:
                my_bar.progress(i/total, text=f"Checking {ticker}...")
                
            t = yf.Ticker(ticker)
            cal = t.calendar
            
            earnings_date = None
            
            # Check structure (Dict vs DataFrame)
            if cal is not None:
                # If dataframe
                if isinstance(cal, pd.DataFrame):
                    pass # Not common in new yfinance
                # If dict
                if isinstance(cal, dict):
                    dates = cal.get("Earnings Date", [])
                    if dates:
                        earnings_date = dates[0]
            
            if earnings_date:
                # Check if it is in the future (or very recent past)
                if isinstance(earnings_date, (datetime, pd.Timestamp)):
                    earnings_date = earnings_date.date()
                
                today = datetime.now().date()
                if earnings_date >= today:
                    row = {
                        "Ticker": ticker,
                        "Date": earnings_date,
                        "Day": earnings_date.strftime("%A"), # Weekday
                        "Formatted": earnings_date.strftime("%b %d"),
                    }
                    data.append(row)
                    
        except Exception:
            pass
            
    my_bar.empty()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    return df.sort_values("Date")

def render_earnings_calendar():
    st.title("📅 Earnings Calendar")
    st.caption("Upcoming earnings for popular market cap companies (US & TSX).")
    
    # Fetch Data
    df = get_upcoming_earnings(POPULAR_TICKERS)
    
    if df.empty:
        st.info("No upcoming earnings found for the tracked watchlist in the near future.")
        return

    # Navigation: Dropdown for Dates
    dates = sorted(df["Date"].unique())
    
    # Format options for dropdown
    date_options = {d: f"{d.strftime('%b %d')} ({d.strftime('%A')})" for d in dates}
    
    selected_date = st.selectbox(
        "📅 Select a Date to view earnings:",
        options=dates,
        format_func=lambda x: date_options[x]
    )
    
    # Filter Data
    day_df = df[df["Date"] == selected_date]
    
    st.divider()
    st.subheader(f"Earnings for {date_options[selected_date]}")
    
    # Horizontal Layout (Grid)
    cols = st.columns(4)
    
    for idx, row in day_df.reset_index().iterrows():
        with cols[idx % 4]:
            ticker = row['Ticker']
            
            clean_ticker = ticker.lower()
            logo_url = f"https://logos.stockanalysis.com/{clean_ticker}.svg"
            
            # Card Styling: White BG, Black Text, Shadow
            st.markdown(f"""
            <div style="
                background-color: white;
                color: black;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                transition: transform 0.2s;
            ">
                <img src="{logo_url}" style="
                    width: 64px; 
                    height: 64px; 
                    border-radius: 50%; 
                    margin-bottom: 12px;
                    object-fit: contain;
                    background-color: #f8f9fa; /* Light grey bg for logo fallback */
                " onerror="this.style.display='none'">
                <br>
                <div style="font-weight: 800; font-size: 1.4em; color: #1a1a1a;">{ticker}</div>
                <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                    {row['Day']}
                </div>
            </div>
            """, unsafe_allow_html=True)
