"""
Configuration settings for Valuation Master Pro
"""
import streamlit as st

# Page configuration
PAGE_TITLE = "Valuation Master Pro"
PAGE_ICON = "📱"
PAGE_LAYOUT = "centered"

def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT
    )

def display_title():
    """Display main app title"""
    st.title("📱 Valuation Master Pro")
    st.caption("Cash • Sales • Earnings • Health • Insiders • AI + Screener")
