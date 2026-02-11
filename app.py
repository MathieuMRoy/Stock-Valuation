"""
Valuation Master Pro - Main Application Entry Point

A comprehensive stock valuation tool with multiple analysis methods:
- DCF (Discounted Cash Flow)
- P/S (Price to Sales) multiples
- P/E (Price to Earnings) multiples
- Asset-based valuation
- Technical analysis
- AI-powered analyst reports

Run with: streamlit run app.py
"""
import streamlit as st
import os
import sys
import shutil
import tempfile

# --- SSL CERTIFICATE FIX ---
# Fix for "error setting certificate verify locations" on Windows with custom venv/paths
# The path often fails due to spaces or unicode characters in OneDrive/User folders.
# We copy the cert file to a safe temporary location.
try:
    base_path = os.getcwd()
    # Attempt to find the cert in the local .venv or standard location
    potential_paths = [
        os.path.join(base_path, ".venv", "Lib", "site-packages", "certifi", "cacert.pem"),
        os.path.join(base_path, ".venv", "lib", "site-packages", "certifi", "cacert.pem"),
    ]
    
    cert_source = None
    for p in potential_paths:
        if os.path.exists(p):
            cert_source = p
            break
            
    if cert_source:
        # Copy to temp dir (safe path)
        temp_dir = tempfile.gettempdir()
        target_cert = os.path.join(temp_dir, "cacert_fix.pem")
        
        # Only copy if it doesn't exist or is different size, to avoid constant IO
        should_copy = True
        if os.path.exists(target_cert):
            if os.path.getsize(target_cert) == os.path.getsize(cert_source):
                should_copy = False
                
        if should_copy:
            shutil.copy2(cert_source, target_cert)
            
        # Set Environment Variables
        os.environ["REQUESTS_CA_BUNDLE"] = target_cert
        os.environ["SSL_CERT_FILE"] = target_cert
except Exception as e:
    pass
# ---------------------------

from config.settings import configure_page, display_title
from ui import render_sidebar, render_stock_analyzer, render_screener, render_earnings_calendar


def main():
    """Main application entry point."""
    # Configure page (must be first Streamlit command)
    configure_page()
    
    # Render sidebar and get API key
    api_key = render_sidebar()
    
    # Display main title
    display_title()
    
    # Mode selection
    mode = st.sidebar.radio("Mode", ["Earnings Calendar", "Stock Analyzer", "AI Screener (Top Upside)"], index=1)
    
    # Render appropriate mode
    if mode == "Stock Analyzer":
        render_stock_analyzer(api_key)
    elif mode == "Earnings Calendar":
        render_earnings_calendar()
    else:
        render_screener()


if __name__ == "__main__":
    main()
