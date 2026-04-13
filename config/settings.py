"""
Configuration settings and shared visual theme for Valuation Master Pro.
"""

import streamlit as st


PAGE_TITLE = "Valuation Master Pro"
PAGE_ICON = "📈"
PAGE_LAYOUT = "wide"


def inject_global_styles():
    """Inject the global visual theme used across the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        :root {
            --vmp-bg: #07111a;
            --vmp-surface: rgba(15, 24, 36, 0.82);
            --vmp-surface-strong: rgba(18, 31, 47, 0.96);
            --vmp-border: rgba(133, 164, 196, 0.20);
            --vmp-text: #f5f7fb;
            --vmp-muted: #9db0c4;
            --vmp-accent: #f3b24f;
            --vmp-accent-2: #3db8b2;
            --vmp-shadow: 0 24px 60px rgba(0, 0, 0, 0.30);
        }

        html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], p, span, label, li, div {
            font-family: "IBM Plex Sans", sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: "Sora", sans-serif !important;
            letter-spacing: -0.03em;
            color: var(--vmp-text);
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(61, 184, 178, 0.12), transparent 28%),
                radial-gradient(circle at 88% 14%, rgba(243, 178, 79, 0.14), transparent 24%),
                linear-gradient(180deg, #08111b 0%, #0a1320 44%, #071018 100%);
            color: var(--vmp-text);
        }

        [data-testid="stAppViewContainer"] {
            background: transparent;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.4rem;
            padding-bottom: 3.2rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(11, 20, 31, 0.98) 0%, rgba(16, 29, 44, 0.98) 100%);
            border-right: 1px solid var(--vmp-border);
        }

        .vmp-hero {
            position: relative;
            overflow: hidden;
            margin: 0.25rem 0 1.75rem 0;
            padding: 1.6rem 1.7rem 1.5rem 1.7rem;
            border-radius: 28px;
            border: 1px solid var(--vmp-border);
            background:
                radial-gradient(circle at top right, rgba(243, 178, 79, 0.18), transparent 30%),
                radial-gradient(circle at left center, rgba(61, 184, 178, 0.16), transparent 28%),
                linear-gradient(145deg, rgba(14, 24, 38, 0.96), rgba(10, 18, 28, 0.90));
            box-shadow: var(--vmp-shadow);
        }

        .vmp-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--vmp-muted);
            font-size: 0.83rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .vmp-title {
            margin-top: 0.9rem;
            font-family: "Sora", sans-serif;
            font-size: clamp(2.4rem, 4vw, 4rem);
            line-height: 0.96;
            font-weight: 800;
            color: var(--vmp-text);
        }

        .vmp-title-accent {
            background: linear-gradient(135deg, #f7c56a 0%, #f2a24d 48%, #53c7c0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .vmp-subtitle {
            max-width: 820px;
            margin-top: 0.85rem;
            color: var(--vmp-muted);
            font-size: 1rem;
            line-height: 1.65;
        }

        .vmp-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.1rem;
        }

        .vmp-badge {
            padding: 0.48rem 0.78rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--vmp-text);
            font-size: 0.88rem;
            font-weight: 500;
        }

        .vmp-mode-banner {
            margin: 0.1rem 0 1.25rem 0;
            padding: 1.15rem 1.2rem;
            border-radius: 22px;
            border: 1px solid var(--vmp-border);
            background: linear-gradient(180deg, rgba(17, 28, 43, 0.92), rgba(12, 20, 32, 0.90));
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
        }

        .vmp-mode-label {
            color: var(--vmp-accent);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .vmp-mode-title {
            margin-top: 0.3rem;
            font-family: "Sora", sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--vmp-text);
        }

        .vmp-mode-copy {
            margin-top: 0.25rem;
            color: var(--vmp-muted);
            line-height: 1.55;
        }

        .vmp-overview {
            position: relative;
            overflow: hidden;
            margin: 1rem 0 1rem 0;
            padding: 1.35rem 1.35rem 1.15rem 1.35rem;
            border-radius: 24px;
            border: 1px solid var(--vmp-border);
            background:
                radial-gradient(circle at top right, rgba(243, 178, 79, 0.12), transparent 26%),
                radial-gradient(circle at left center, rgba(61, 184, 178, 0.12), transparent 28%),
                linear-gradient(145deg, rgba(14, 24, 38, 0.96), rgba(10, 18, 28, 0.92));
            box-shadow: var(--vmp-shadow);
        }

        .vmp-overview-kicker {
            color: var(--vmp-accent);
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .vmp-overview-title {
            margin-top: 0.4rem;
            font-family: "Sora", sans-serif;
            font-size: clamp(1.45rem, 3vw, 2.35rem);
            font-weight: 700;
            color: var(--vmp-text);
            letter-spacing: -0.03em;
        }

        .vmp-overview-copy {
            max-width: 860px;
            margin-top: 0.5rem;
            color: var(--vmp-muted);
            line-height: 1.65;
        }

        .vmp-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }

        .vmp-chip {
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.05);
            color: var(--vmp-text);
            font-size: 0.86rem;
            font-weight: 600;
        }

        .vmp-insight-card {
            height: 100%;
            min-height: 168px;
            margin: 0.5rem 0 1rem 0;
            padding: 1rem 1rem 0.95rem 1rem;
            border-radius: 20px;
            border: 1px solid var(--vmp-border);
            background: linear-gradient(180deg, rgba(16, 27, 41, 0.90), rgba(12, 20, 32, 0.86));
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.15);
        }

        .vmp-insight-label {
            color: var(--vmp-muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .vmp-insight-value {
            margin-top: 0.45rem;
            font-family: "Sora", sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--vmp-text);
            letter-spacing: -0.03em;
        }

        .vmp-insight-copy {
            margin-top: 0.5rem;
            color: var(--vmp-muted);
            line-height: 1.6;
            font-size: 0.92rem;
        }

        .vmp-provenance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.85rem;
            margin: 0.25rem 0 1rem 0;
        }

        .vmp-provenance-card {
            min-height: 142px;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--vmp-border);
            background: linear-gradient(180deg, rgba(16, 27, 41, 0.88), rgba(10, 18, 28, 0.84));
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
        }

        .vmp-provenance-label {
            color: var(--vmp-accent);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .vmp-provenance-source {
            margin-top: 0.45rem;
            color: var(--vmp-text);
            font-size: 0.98rem;
            font-weight: 700;
        }

        .vmp-provenance-meta {
            margin-top: 0.3rem;
            color: var(--vmp-muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .vmp-provenance-copy {
            margin-top: 0.45rem;
            color: var(--vmp-muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .vmp-context-banner {
            margin: 0.3rem 0 1rem 0;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--vmp-border);
            background: linear-gradient(180deg, rgba(16, 27, 41, 0.88), rgba(10, 18, 28, 0.82));
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
        }

        .vmp-context-title {
            color: var(--vmp-text);
            font-size: 0.96rem;
            font-weight: 700;
        }

        .vmp-context-copy {
            margin-top: 0.3rem;
            color: var(--vmp-muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .vmp-agent-trace {
            margin: 0.1rem 0 0.9rem 0;
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
            border: 1px solid rgba(82, 198, 186, 0.18);
            background: linear-gradient(180deg, rgba(12, 24, 38, 0.9), rgba(9, 18, 29, 0.86));
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }

        .vmp-agent-trace.is-live {
            border-color: rgba(242, 162, 77, 0.28);
            background: linear-gradient(180deg, rgba(27, 31, 22, 0.88), rgba(17, 22, 17, 0.84));
        }

        .vmp-agent-trace-label {
            color: #f2a24d;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .vmp-agent-trace-title {
            margin-top: 0.2rem;
            color: var(--vmp-text);
            font-size: 1rem;
            font-weight: 700;
        }

        .vmp-agent-trace-copy {
            margin-top: 0.28rem;
            color: var(--vmp-muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .vmp-agent-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }

        .vmp-agent-chip {
            padding: 0.4rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.09);
            background: rgba(255, 255, 255, 0.05);
            color: var(--vmp-text);
            font-size: 0.82rem;
            font-weight: 600;
        }

        [data-testid="stMetric"] {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--vmp-border);
            background: linear-gradient(180deg, rgba(16, 27, 41, 0.90), rgba(12, 20, 32, 0.86));
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stMetricLabel"] {
            color: var(--vmp-muted);
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            font-family: "Sora", sans-serif;
            color: var(--vmp-text);
            letter-spacing: -0.03em;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border-radius: 20px;
            border: 1px solid var(--vmp-border);
            background: linear-gradient(180deg, rgba(16, 27, 41, 0.88), rgba(12, 20, 32, 0.84));
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.15);
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
            border-radius: 16px !important;
            border: 1px solid var(--vmp-border) !important;
            background: rgba(17, 28, 43, 0.82) !important;
            color: var(--vmp-text) !important;
            box-shadow: none !important;
        }

        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stRadio label,
        .stSlider label {
            color: var(--vmp-muted) !important;
            font-weight: 600;
        }

        div.stButton > button {
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: linear-gradient(135deg, #14867f 0%, #1d6fa5 100%);
            color: white;
            font-weight: 700;
            padding: 0.62rem 1rem;
            box-shadow: 0 12px 24px rgba(20, 134, 127, 0.28);
            transition: transform 0.16s ease, box-shadow 0.16s ease, filter 0.16s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.04);
            box-shadow: 0 16px 28px rgba(29, 111, 165, 0.30);
        }

        [data-testid="stChatMessage"] {
            border-radius: 20px;
            border: 1px solid var(--vmp-border);
            background: linear-gradient(180deg, rgba(14, 24, 38, 0.92), rgba(12, 20, 32, 0.86));
            padding: 0.35rem 0.4rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.14);
        }

        [data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid var(--vmp-border);
        }

        [data-testid="stDataFrame"],
        .stTable {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--vmp-border);
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #f2a24d 0%, #4bc5bb 100%);
        }

        .vmp-earnings-card {
            border-radius: 20px;
            padding: 18px 18px 16px 18px;
            text-align: center;
            border: 1px solid var(--vmp-border);
            background: linear-gradient(180deg, rgba(17, 28, 43, 0.92), rgba(12, 20, 32, 0.84));
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
            margin-bottom: 18px;
        }

        .vmp-earnings-logo {
            width: 64px;
            height: 64px;
            border-radius: 18px;
            margin-bottom: 14px;
            object-fit: contain;
            background: rgba(255, 255, 255, 0.06);
            padding: 10px;
        }

        .vmp-earnings-ticker {
            font-family: "Sora", sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            color: var(--vmp-text);
        }

        .vmp-earnings-day {
            margin-top: 0.28rem;
            color: var(--vmp-muted);
            font-size: 0.92rem;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .vmp-hero {
                padding: 1.2rem 1.05rem 1.15rem 1.05rem;
            }

            .vmp-title {
                font-size: clamp(2rem, 8vw, 2.8rem);
            }

            .vmp-subtitle {
                font-size: 0.95rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT,
        initial_sidebar_state="expanded",
    )
    inject_global_styles()


def display_title():
    """Display the main app hero block."""
    st.markdown(
        """
        <div class="vmp-hero">
            <div class="vmp-kicker">Finance analytics studio</div>
            <div class="vmp-title">
                Valuation <span class="vmp-title-accent">Master Pro</span>
            </div>
            <div class="vmp-subtitle">
                A cleaner, presentation-ready workspace for valuation, catalysts, technical signals,
                market sentiment and AI-assisted stock research.
            </div>
            <div class="vmp-badges">
                <span class="vmp-badge">DCF + Multiples</span>
                <span class="vmp-badge">Technical Signals</span>
                <span class="vmp-badge">Insiders + Short Interest</span>
                <span class="vmp-badge">AI Multi-Agent Chat</span>
                <span class="vmp-badge">Screening + Earnings</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_mode_banner(mode: str):
    """Display a small presentation card for the currently selected mode."""
    mode_content = {
        "Stock Analyzer": (
            "Core Workspace",
            "Analyze one stock with valuation, quality, technicals, catalysts and AI commentary in one place.",
        ),
        "Earnings Calendar": (
            "Event Watch",
            "Track upcoming earnings dates for a curated watchlist and navigate catalysts more quickly.",
        ),
        "AI Screener (Top Upside)": (
            "Discovery Engine",
            "Scan sectors for upside candidates using a fast DCF-style pass and prioritize what deserves deeper work.",
        ),
    }
    label, copy = mode_content.get(mode, ("Workspace", "Explore the application."))
    st.markdown(
        f"""
        <div class="vmp-mode-banner">
            <div class="vmp-mode-label">{label}</div>
            <div class="vmp-mode-title">{mode}</div>
            <div class="vmp-mode-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
