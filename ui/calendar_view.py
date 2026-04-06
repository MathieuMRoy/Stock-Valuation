import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime


POPULAR_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "AMD", "INTC",
    "CRM", "ADBE", "ORCL", "IBM", "CSCO", "UBER", "ABNB", "PLTR", "SNOW", "SHOP", "SQ", "PYPL",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP",
    "JNJ", "PFE", "LLY", "MRK", "KO", "PEP", "MCD", "SBUX", "WMT", "TGT", "COST", "DIS", "CMCSA",
    "XOM", "CVX", "BA", "CAT", "DE", "GE", "F", "GM",
    "RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO",
    "SHOP.TO", "CSU.TO", "GIB-A.TO",
    "CNQ.TO", "SU.TO", "CVE.TO", "ENB.TO", "TRP.TO",
    "CNR.TO", "CP.TO",
    "BCE.TO", "T.TO", "RCI-B.TO",
    "ATD.TO", "DOL.TO", "L.TO", "WN.TO", "SAP.TO",
    "BAM.TO", "BN.TO", "POW.TO",
]


@st.cache_data(ttl=3600 * 4)
def get_upcoming_earnings(tickers):
    """
    Fetch next earnings dates for a curated list of tickers.
    Returns a dataframe with [Ticker, Date, Day, Formatted].
    """
    data = []
    progress_text = "Scanning market for upcoming earnings..."
    progress_bar = st.progress(0, text=progress_text)

    total = len(tickers)
    for idx, ticker in enumerate(tickers):
        try:
            if idx % 10 == 0:
                progress_bar.progress(idx / total, text=f"Checking {ticker}...")

            ticker_obj = yf.Ticker(ticker)
            calendar = ticker_obj.calendar
            earnings_date = None

            if isinstance(calendar, dict):
                dates = calendar.get("Earnings Date", [])
                if dates:
                    earnings_date = dates[0]

            if earnings_date:
                if isinstance(earnings_date, (datetime, pd.Timestamp)):
                    earnings_date = earnings_date.date()

                today = datetime.now().date()
                if earnings_date >= today:
                    data.append(
                        {
                            "Ticker": ticker,
                            "Date": earnings_date,
                            "Day": earnings_date.strftime("%A"),
                            "Formatted": earnings_date.strftime("%b %d"),
                        }
                    )
        except Exception:
            pass

    progress_bar.empty()

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data).sort_values("Date")


def render_earnings_calendar():
    """Render the earnings calendar view."""
    st.subheader("Earnings Calendar")
    st.caption("Upcoming earnings for popular market cap companies in the US and Canada.")

    df = get_upcoming_earnings(POPULAR_TICKERS)
    if df.empty:
        st.info("No upcoming earnings found for the tracked watchlist in the near future.")
        return

    dates = sorted(df["Date"].unique())
    date_options = {date: f"{date.strftime('%b %d')} ({date.strftime('%A')})" for date in dates}

    selected_date = st.selectbox(
        "Select a date to view earnings:",
        options=dates,
        format_func=lambda value: date_options[value],
    )

    day_df = df[df["Date"] == selected_date]

    st.divider()
    st.subheader(f"Earnings for {date_options[selected_date]}")

    columns = st.columns(4)
    for idx, row in day_df.reset_index(drop=True).iterrows():
        with columns[idx % 4]:
            ticker = row["Ticker"]
            logo_url = f"https://logos.stockanalysis.com/{ticker.lower()}.svg"
            st.markdown(
                f"""
                <div class="vmp-earnings-card">
                    <img src="{logo_url}" class="vmp-earnings-logo" onerror="this.style.display='none'">
                    <div class="vmp-earnings-ticker">{ticker}</div>
                    <div class="vmp-earnings-day">{row['Day']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
