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
        desired_mode = st.session_state.get("app_mode", MODE_OPTIONS[0])
        if desired_mode not in MODE_OPTIONS:
            desired_mode = MODE_OPTIONS[0]

        st.header("Configuration")

        mode = st.radio("Espace de travail", MODE_OPTIONS, index=MODE_OPTIONS.index(desired_mode))
        st.session_state["app_mode"] = mode
        st.caption("Passe d'une vue a l'autre sans perdre le theme ni le contexte.")

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
        with st.container(border=True):
            st.caption("STATUT DES DONNEES")
            st.markdown("**Marche + fondamentaux**")
            st.write("Quotes live, etats financiers caches et news contextualisees.")
            st.caption(f"Cache actif: {FINANCIAL_DATA_CACHE_VERSION}")

        if mode == "Stock Analyzer":
            st.subheader("Controles de l'analyseur")
            manual_shares = st.number_input("Actions manuelles (millions)", value=0.0, step=1.0)
            sidebar_state["manual_shares"] = manual_shares
            st.caption("Utile quand Yahoo Finance ne remonte pas bien le nombre d'actions.")
            st.info(
                "Cette vue charge d'abord les donnees coeur, puis les graphiques techniques "
                "et l'IA seulement quand tu ouvres les sections correspondantes."
            )
        elif mode == "Earnings Calendar":
            st.subheader("Focus calendrier")
            st.caption("Vue rapide des catalyseurs a venir sur une watchlist suivie.")
            st.info("Ideal pour reperer les publications les plus proches et basculer vers l'analyse detaillee.")
        else:
            st.subheader("Focus screener")
            st.caption("Sors une shortlist exploitable avant de lancer une analyse titre par titre.")
            st.info("Le scan complet peut prendre un moment selon le nombre de tickers par secteur.")

        st.divider()

        if st.button("Reset Cache"):
            st.cache_data.clear()
            st.rerun()
        st.caption("A utiliser si les donnees semblent incorrectes.")

    return api_key or stored_api_key, mode, sidebar_state
