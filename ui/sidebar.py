"""
Sidebar UI Component - Configuration sidebar for the app
"""
import os

import streamlit as st
from fetchers.yahoo_finance import FINANCIAL_DATA_CACHE_VERSION


def render_sidebar() -> str:
    """
    Render the sidebar with configuration options.
    
    Returns:
        Google API key entered by user or loaded from secrets/environment
    """
    try:
        stored_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        stored_api_key = ""

    if not stored_api_key:
        stored_api_key = os.getenv("GOOGLE_API_KEY", "")

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "🔑 Google Gemini API Key",
            type="password",
            help="Optionnel si GOOGLE_API_KEY est deja defini dans Streamlit secrets ou dans l'environnement."
        )

        if stored_api_key:
            st.success("Google API key loaded from secrets/environment.")
        else:
            st.caption("Pour Streamlit Cloud, ajoute GOOGLE_API_KEY dans Secrets.")

        st.divider()

        if st.button("🗑️ Reset Cache"):
            st.cache_data.clear()
            st.rerun()
        st.caption("À utiliser si les données semblent incorrectes.")

        st.caption(f"Data cache version: {FINANCIAL_DATA_CACHE_VERSION}")

    return api_key or stored_api_key
