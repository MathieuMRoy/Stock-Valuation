"""Earnings calendar UI - cleaner planning view."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from data import TICKER_DB
from services import fetch_nasdaq_earnings_calendar


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


def _coerce_calendar_date(value) -> date | None:
    """Normalize Yahoo calendar date payloads."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _fetch_watchlist_earnings(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Fallback for Canadian/watchlist names not covered well by Nasdaq."""
    company_map = _company_name_map()
    data: list[dict[str, object]] = []

    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            calendar = ticker_obj.calendar
            earnings_date = _coerce_calendar_date(calendar.get("Earnings Date")) if isinstance(calendar, dict) else None
            if not earnings_date or earnings_date < start_date or earnings_date > end_date:
                continue

            data.append(
                {
                    "Ticker": ticker,
                    "Company": company_map.get(ticker.upper(), ticker),
                    "Country": _country_for_ticker(ticker),
                    "Date": earnings_date,
                    "Day": earnings_date.strftime("%A"),
                    "Formatted": earnings_date.strftime("%b %d"),
                    "DaysUntil": (earnings_date - date.today()).days,
                    "Time": "Not supplied",
                    "EPS Forecast": "N/A",
                    "Estimates": "N/A",
                    "Fiscal Quarter": "N/A",
                    "Market Cap": "N/A",
                    "Last Year EPS": "N/A",
                    "Source": "Yahoo fallback",
                }
            )
        except Exception:
            continue

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data).sort_values(["Date", "Ticker"])


@st.cache_data(ttl=3600 * 4)
def get_upcoming_earnings(start_iso: str, end_iso: str, include_watchlist_fallback: bool) -> pd.DataFrame:
    """Fetch upcoming earnings, primarily from Nasdaq's earnings calendar."""
    start_date = pd.to_datetime(start_iso).date()
    end_date = pd.to_datetime(end_iso).date()

    nasdaq_df = fetch_nasdaq_earnings_calendar(start_date, end_date)
    frames = [nasdaq_df] if not nasdaq_df.empty else []

    if include_watchlist_fallback or not frames:
        fallback_df = _fetch_watchlist_earnings(POPULAR_TICKERS, start_date, end_date)
        if not fallback_df.empty:
            frames.append(fallback_df)

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["Ticker", "Date"], keep="first")
        .sort_values(["Date", "Ticker"])
    )


def render_earnings_calendar():
    """Render the earnings calendar view."""
    st.subheader("Calendrier des resultats")
    st.caption("Source principale: calendrier Nasdaq. Tu peux ensuite ouvrir un titre directement dans l'analyseur.")

    today = datetime.now().date()
    filter_cols = st.columns([1.2, 1.1, 1.2, 1.2])
    view_mode = filter_cols[0].selectbox(
        "Vue",
        ["7 prochains jours", "30 prochains jours", "60 prochains jours", "Date precise"],
    )
    country_filter = filter_cols[1].selectbox("Marche", ["Tous", "USA / Nasdaq", "Canada"])
    search_text = filter_cols[2].text_input("Recherche ticker ou compagnie")
    include_watchlist_fallback = filter_cols[3].checkbox(
        "Ajouter watchlist Canada",
        value=False,
        help="Ajoute un fallback Yahoo pour les tickers .TO/.V de la watchlist.",
    )

    selected_date = None
    if view_mode == "Date precise":
        selected_date = st.date_input("Date precise", value=today)
        start_date = selected_date
        end_date = selected_date
    elif view_mode == "7 prochains jours":
        start_date = today
        end_date = today + timedelta(days=7)
    elif view_mode == "30 prochains jours":
        start_date = today
        end_date = today + timedelta(days=30)
    else:
        start_date = today
        end_date = today + timedelta(days=60)

    with st.spinner("Chargement du calendrier Nasdaq..."):
        df = get_upcoming_earnings(start_date.isoformat(), end_date.isoformat(), include_watchlist_fallback)

    if df.empty:
        st.info("Aucune publication n'a ete detectee pour cette periode. Essaie une autre date ou active le fallback watchlist Canada.")
        return

    week_limit = today + timedelta(days=7)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Evenements suivis", f"{len(df)}")
    metric_cols[1].metric("Cette semaine", f"{len(df[df['Date'] <= week_limit])}")
    metric_cols[2].metric("Nasdaq", f"{len(df[df['Source'] == 'Nasdaq'])}")
    metric_cols[3].metric("Canada", f"{len(df[df['Country'] == 'Canada'])}")

    filtered = df.copy()
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
                    st.caption(f"{row['Country'].upper()} | {row['Day']} | {row.get('Time', 'Not supplied')}")
                    st.markdown(f"### {row['Ticker']}")
                    st.write(row["Company"])
                    st.caption(f"Publication attendue le {row['Formatted']} | Source: {row.get('Source', 'N/A')}")
                    st.caption(
                        f"EPS consensus: {row.get('EPS Forecast', 'N/A')} | "
                        f"Est.: {row.get('Estimates', 'N/A')} | Quarter: {row.get('Fiscal Quarter', 'N/A')}"
                    )
                    if st.button("Ouvrir dans l'analyseur", key=f"calendar_open_{row['Ticker']}_{event_date}"):
                        _open_ticker_in_analyzer(str(row["Ticker"]))

    with st.expander("Tableau complet", expanded=False):
        st.dataframe(
            filtered[
                [
                    "Date",
                    "Time",
                    "Ticker",
                    "Company",
                    "EPS Forecast",
                    "Estimates",
                    "Fiscal Quarter",
                    "Market Cap",
                    "Source",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
