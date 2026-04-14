"""Screener UI - productized screening workspace."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from services import run_screener


def _open_ticker_in_analyzer(ticker: str):
    st.session_state["app_mode"] = "Stock Analyzer"
    st.session_state["selected_ticker_override"] = ticker
    st.session_state["stock_picker_force_sync"] = True
    st.rerun()


def _render_candidate_card(row: pd.Series, key_prefix: str):
    with st.container(border=True):
        st.caption(row["bucket"].upper())
        st.markdown(f"### {row['ticker']}")
        st.write(
            f"Prix actuel **{row['price']:.2f} $** | valeur estimee **{row['intrinsic']:.2f} $**"
        )
        st.metric("Upside estime", f"{row['upside']:.1f}%")
        if st.button("Ouvrir dans l'analyseur", key=f"{key_prefix}_{row['ticker']}"):
            _open_ticker_in_analyzer(str(row["ticker"]))


def render_screener():
    """Render the screener mode UI."""
    st.subheader("Screener d'idees")
    st.caption(
        "Scanne plusieurs secteurs pour sortir une shortlist de titres a fort upside, puis ouvre les meilleurs dossiers directement dans l'analyseur."
    )

    with st.container(border=True):
        top_left, top_right = st.columns([3, 2])
        with top_left:
            st.markdown("#### Parametres de scan")
            min_market_cap = st.number_input("Capitalisation minimale (USD)", value=1_000_000_000, step=250_000_000)
            max_tickers_per_sector = st.slider("Max tickers par secteur", 10, 80, 20, step=5)
        with top_right:
            st.markdown("#### Hypotheses rapides")
            fallback_fcf_growth_pct = st.number_input("Croissance FCF fallback (%)", value=15.0, step=0.5)
            wacc_pct = st.number_input("WACC (%)", value=10.0, step=0.5)
            run_scan = st.button("Lancer le screener", use_container_width=True)

    if run_scan:
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def _on_progress(step: int, total: int, sector_name: str, geography_name: str):
            progress_bar.progress(min(step / total, 1.0))
            progress_text.write(f"Scan en cours: {sector_name} ({geography_name})...")

        results = run_screener(
            minimum_market_cap=min_market_cap,
            max_tickers_per_sector=max_tickers_per_sector,
            fallback_fcf_growth_pct=fallback_fcf_growth_pct,
            wacc_pct=wacc_pct,
            progress_callback=_on_progress,
        )

        progress_bar.progress(1.0)
        progress_text.write("Scan termine.")
        st.session_state["screener_results"] = [row.as_row() for row in results]
        st.session_state["screener_run_meta"] = {
            "min_market_cap": min_market_cap,
            "max_tickers_per_sector": max_tickers_per_sector,
            "fallback_fcf_growth_pct": fallback_fcf_growth_pct,
            "wacc_pct": wacc_pct,
        }

    stored_results = st.session_state.get("screener_results", [])
    if not stored_results:
        st.info("Lance un scan pour afficher une shortlist exploitable.")
        return

    dataframe = pd.DataFrame(stored_results)
    if dataframe.empty:
        st.warning("Le scan n'a retourne aucun candidat avec les filtres actuels.")
        return

    dataframe["country"] = dataframe["bucket"].apply(lambda value: "Canada" if "Canada" in str(value) else "USA")
    dataframe["sector"] = dataframe["bucket"].apply(lambda value: str(value).split(" (")[0])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Candidats", f"{len(dataframe)}")
    metric_cols[1].metric("Upside median", f"{dataframe['upside'].median():.1f}%")
    metric_cols[2].metric("Upside moyen", f"{dataframe['upside'].mean():.1f}%")
    best_row = dataframe.sort_values("upside", ascending=False).iloc[0]
    metric_cols[3].metric("Meilleur dossier", best_row["ticker"], delta=f"{best_row['upside']:.1f}%")

    filter_cols = st.columns([1.2, 1.2, 1.2, 1.2, 1.5])
    country_filter = filter_cols[0].selectbox("Pays", ["Tous", "USA", "Canada"])
    min_upside = filter_cols[1].slider("Upside min (%)", -20, 300, 0, step=5)
    sector_options = sorted(dataframe["sector"].dropna().unique().tolist())
    selected_sectors = filter_cols[2].multiselect("Secteurs", sector_options)
    sort_label = filter_cols[3].selectbox(
        "Tri",
        ["Upside decroissant", "Upside croissant", "Ticker A-Z", "Secteur A-Z"],
    )
    search_text = filter_cols[4].text_input("Recherche ticker")

    filtered = dataframe.copy()
    if country_filter != "Tous":
        filtered = filtered[filtered["country"] == country_filter]
    if selected_sectors:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]
    filtered = filtered[filtered["upside"] >= min_upside]
    if search_text.strip():
        query = search_text.strip().upper()
        filtered = filtered[filtered["ticker"].str.upper().str.contains(query)]

    if sort_label == "Upside decroissant":
        filtered = filtered.sort_values("upside", ascending=False)
    elif sort_label == "Upside croissant":
        filtered = filtered.sort_values("upside", ascending=True)
    elif sort_label == "Ticker A-Z":
        filtered = filtered.sort_values("ticker")
    else:
        filtered = filtered.sort_values(["sector", "ticker"])

    action_cols = st.columns([1.4, 1])
    with action_cols[0]:
        st.caption(
            f"{len(filtered)} resultat(s) apres filtres. Parametres actifs: cap min {st.session_state.get('screener_run_meta', {}).get('min_market_cap', min_market_cap):,.0f} $, "
            f"WACC {st.session_state.get('screener_run_meta', {}).get('wacc_pct', wacc_pct):.1f}%."
        )
    with action_cols[1]:
        csv_buffer = StringIO()
        filtered[["ticker", "sector", "country", "price", "intrinsic", "upside"]].to_csv(csv_buffer, index=False)
        st.download_button(
            "Exporter CSV",
            data=csv_buffer.getvalue(),
            file_name="screener_shortlist.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if filtered.empty:
        st.warning("Aucun candidat ne correspond aux filtres actuels.")
        return

    st.markdown("#### Shortlist prioritaire")
    spotlight = filtered.head(6).reset_index(drop=True)
    spotlight_cols = st.columns(3)
    for idx, row in spotlight.iterrows():
        with spotlight_cols[idx % 3]:
            _render_candidate_card(row, key_prefix="screener_open")

    st.markdown("#### Tableau complet")
    display_df = filtered[["ticker", "sector", "country", "price", "intrinsic", "upside"]].rename(
        columns={
            "ticker": "Ticker",
            "sector": "Secteur",
            "country": "Pays",
            "price": "Prix",
            "intrinsic": "Valeur estimee",
            "upside": "Upside (%)",
        }
    )
    st.dataframe(
        display_df.style.format({"Prix": "{:.2f}", "Valeur estimee": "{:.2f}", "Upside (%)": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )
