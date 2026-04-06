"""
Sidebar UI component for configuration and mode-aware controls.
"""
import os

import streamlit as st

from fetchers.yahoo_finance import FINANCIAL_DATA_CACHE_VERSION


MODE_OPTIONS = [
    "Stock Analyzer",
    "Earnings Calendar",
    "AI Screener (Top Upside)",
]


def render_sidebar() -> tuple[str, str, dict]:
    """
    Render the sidebar and return the current app state.

    Returns:
        Tuple of (api_key, selected_mode, sidebar_state)
    """
    try:
        stored_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        stored_api_key = ""

    if not stored_api_key:
        stored_api_key = os.getenv("GOOGLE_API_KEY", "")

    sidebar_state: dict = {}

    with st.sidebar:
        st.header("Configuration")

        mode = st.radio("Workspace", MODE_OPTIONS, index=0)
        st.caption("Passe d'une vue d'analyse a une autre sans perdre le theme ni le contexte.")

        st.divider()

        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Optionnel si GOOGLE_API_KEY est deja defini dans Streamlit secrets ou dans l'environnement.",
        )

        if stored_api_key:
            st.success("Google API key loaded from secrets/environment.")
        else:
            st.caption("Pour Streamlit Cloud, ajoute GOOGLE_API_KEY dans Secrets.")

        st.divider()

        if mode == "Stock Analyzer":
            st.subheader("Analyzer Controls")
            manual_shares = st.number_input("Manual Shares (Millions)", value=0.0, step=1.0)
            sidebar_state["manual_shares"] = manual_shares
            st.caption("Utile quand Yahoo Finance ne remonte pas bien le nombre d'actions.")
            st.info(
                "Cette vue charge d'abord les donnees coeur, puis les graphiques techniques "
                "et l'IA seulement quand tu ouvres les sections correspondantes."
            )
        elif mode == "Earnings Calendar":
            st.subheader("Calendar Focus")
            st.caption("Vue rapide des catalyseurs a venir sur une watchlist large.")
            st.info("Le calendrier est mis en cache quelques heures pour rester fluide.")
        else:
            st.subheader("Screener Notes")
            st.caption("Utilise le moteur de screening pour sortir une short-list avant une analyse detaillee.")
            st.info("Le scan complet peut prendre un moment selon le nombre de tickers par secteur.")

        st.divider()

        if st.button("Reset Cache"):
            st.cache_data.clear()
            st.rerun()
        st.caption("A utiliser si les donnees semblent incorrectes.")

        st.caption(f"Data cache version: {FINANCIAL_DATA_CACHE_VERSION}")

    return api_key or stored_api_key, mode, sidebar_state
