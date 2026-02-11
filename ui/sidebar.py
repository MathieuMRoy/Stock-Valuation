"""
Sidebar UI Component - Configuration sidebar for the app
"""
import streamlit as st


def render_sidebar() -> str:
    """
    Render the sidebar with configuration options.
    
    Returns:
        API key entered by user
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Gratuit sur console.groq.com. Commence par gsk_"
        )
        
        st.divider()
        
        if st.button("🗑️ Reset Cache"):
            st.cache_data.clear()
            st.rerun()
        st.caption("À utiliser si les données semblent incorrectes.")
    
    return api_key
