"""Earnings calendar UI - cleaner planning view."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from data import TICKER_DB


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


def _company_name_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in TICKER_DB:
        if item.startswith("---") or item.lower().startswith("autre"):
            continue
        if " - " in item:
            ticker, name = item.split(" - ", 1)
            mapping[ticker.strip().upper()] = name.strip()
    return mapping


def _country_for_ticker(ticker: str) -> str:
    return "Canada" if ".TO" in ticker or ".V" in ticker else "USA"


def _open_ticker_in_analyzer(ticker: str):
    st.session_state["app_mode"] = "Stock Analyzer"
    st.session_state["selected_ticker_override"] = ticker
    st.session_state["stock_picker_force_sync"] = True
    st.rerun()


@st.cache_data(ttl=3600 * 4)
def get_upcoming_earnings(tickers: list[str]) -> pd.DataFrame:
    """Fetch next earnings dates for a curated list of tickers."""
    company_map = _company_name_map()
    data: list[dict[str, object]] = []

    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            calendar = ticker_obj.calendar
            earnings_date = None

            if isinstance(calendar, dict):
                dates = calendar.get("Earnings Date", [])
                if dates:
                    earnings_date = dates[0]

            if not earnings_date:
                continue

            if isinstance(earnings_date, (datetime, pd.Timestamp)):
                earnings_date = earnings_date.date()

            today = datetime.now().date()
            if earnings_date < today:
                continue

            data.append(
                {
                    "Ticker": ticker,
                    "Company": company_map.get(ticker.upper(), ticker),
                    "Country": _country_for_ticker(ticker),
                    "Date": earnings_date,
                    "Day": earnings_date.strftime("%A"),
                    "Formatted": earnings_date.strftime("%b %d"),
                    "DaysUntil": (earnings_date - today).days,
                }
            )
        except Exception:
            continue

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data).sort_values(["Date", "Ticker"])


def render_earnings_calendar():
    """Render the earnings calendar view."""
    st.subheader("Calendrier des resultats")
    st.caption("Repere rapidement les prochaines publications et ouvre ensuite le titre dans l'analyseur principal.")

    with st.spinner("Chargement du calendrier des resultats..."):
        df = get_upcoming_earnings(POPULAR_TICKERS)

    if df.empty:
        st.info("Aucune publication proche n'a ete detectee sur la watchlist suivie.")
        return

    today = datetime.now().date()
    week_limit = today + timedelta(days=7)
    month_limit = today + timedelta(days=30)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Evenements suivis", f"{len(df)}")
    metric_cols[1].metric("Cette semaine", f"{len(df[df['Date'] <= week_limit])}")
    metric_cols[2].metric("USA", f"{len(df[df['Country'] == 'USA'])}")
    metric_cols[3].metric("Canada", f"{len(df[df['Country'] == 'Canada'])}")

    filter_cols = st.columns([1.3, 1.1, 1.2, 1.2])
    view_mode = filter_cols[0].selectbox(
        "Vue",
        ["7 prochains jours", "30 prochains jours", "Tout", "Date precise"],
    )
    country_filter = filter_cols[1].selectbox("Pays", ["Tous", "USA", "Canada"])
    search_text = filter_cols[2].text_input("Recherche ticker")
    selected_date = None
    if view_mode == "Date precise":
        selected_date = filter_cols[3].selectbox(
            "Date",
            options=sorted(df["Date"].unique()),
            format_func=lambda value: value.strftime("%b %d (%A)"),
        )
    else:
        filter_cols[3].write("")
        filter_cols[3].write("")
        filter_cols[3].caption("Bascule en vue precise pour filtrer sur une date unique.")

    filtered = df.copy()
    if view_mode == "7 prochains jours":
        filtered = filtered[filtered["Date"] <= week_limit]
    elif view_mode == "30 prochains jours":
        filtered = filtered[filtered["Date"] <= month_limit]
    elif view_mode == "Date precise" and selected_date:
        filtered = filtered[filtered["Date"] == selected_date]

    if country_filter != "Tous":
        filtered = filtered[filtered["Country"] == country_filter]
    if search_text.strip():
        query = search_text.strip().upper()
        filtered = filtered[
            filtered["Ticker"].str.upper().str.contains(query) | filtered["Company"].str.upper().str.contains(query)
        ]

    if filtered.empty:
        st.warning("Aucun evenement ne correspond aux filtres actuels.")
        return

    for event_date, group in filtered.groupby("Date", sort=True):
        day_offset = (event_date - today).days
        if day_offset == 0:
            offset_label = "Aujourd'hui"
        elif day_offset == 1:
            offset_label = "Demain"
        else:
            offset_label = f"Dans {day_offset} jours"

        st.markdown(f"#### {event_date.strftime('%b %d, %Y')} - {offset_label}")
        columns = st.columns(3)
        for idx, (_, row) in enumerate(group.reset_index(drop=True).iterrows()):
            with columns[idx % 3]:
                with st.container(border=True):
                    st.caption(f"{row['Country'].upper()} | {row['Day']}")
                    st.markdown(f"### {row['Ticker']}")
                    st.write(row["Company"])
                    st.caption(f"Publication attendue le {row['Formatted']}")
                    if st.button("Ouvrir dans l'analyseur", key=f"calendar_open_{row['Ticker']}_{event_date}"):
                        _open_ticker_in_analyzer(str(row["Ticker"]))
