"""
Stock Analyzer UI - Main stock analysis interface
"""
from datetime import date
from html import escape
from uuid import uuid4

import pandas as pd
import streamlit as st

from data import TICKER_DB
from fetchers.yahoo_finance import FINANCIAL_DATA_CACHE_VERSION
from fetchers.stock_analysis import get_extended_history
from valuation import calculate_valuation, solve_reverse_dcf, display_relative_analysis, compute_asset_based_value
from technical import (
    fetch_price_history,
    add_indicators,
    bull_flag_score,
    summarize_technical_setup,
    plot_technical_chart,
    plot_fundamental_overlay,
)
from scoring import plot_radar
from ai import build_ai_chat_signature, chat_with_ai_analyst, create_ai_chat_session
from services import (
    INVESTOR_OBJECTIVES,
    blend_reasonable_intrinsic_values,
    build_investor_objective_snapshot,
    prepare_analyzer_snapshot,
    profile_label,
    quality_label,
    risk_label,
    sector_profile_summary,
    upside_pct,
    valuation_label,
)


SECTION_OPTIONS = [
    ("Fondamentaux", "Historique financier, marges, liquidite et qualite operationnelle."),
    ("DCF", "Valorisation basee sur les flux de tresorerie avec scenarios et sensibilites."),
    ("Ventes", "Lecture du multiple de ventes face aux pairs et au benchmark."),
    ("Benefices", "Lecture du P/E et du positionnement relatif au marche."),
    ("Actifs", "Valeur plancher basee sur le bilan."),
    ("Analystes", "Consensus, objectifs de prix et changements de recommandations."),
    ("Insiders", "Transactions d'initiés recuperees depuis Yahoo Finance."),
    ("Technique", "Tendance, momentum, overlays et short interest."),
    ("Scorecard", "Synthese sante, croissance et valorisation en un coup d'oeil."),
    ("Chat IA", "Conversation multi-agents sur la valorisation, les news et les comparaisons."),
]


def _format_currency(value: float | None) -> str:
    """Format a currency number for overview cards."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f} $"
    except Exception:
        return "N/A"


def _format_market_cap(value: float | None) -> str:
    """Format large market-cap style numbers in compact form."""
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f} T$"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} B$"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f} M$"
    return f"{value:,.0f} $"


def _format_pct(value: float | None, decimals: int = 1, signed: bool = False) -> str:
    """Format percentage values that are already expressed in percentage points."""
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return "N/A"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{decimals}f}%"


def _format_ratio(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return "N/A"
    return f"{number:.{decimals}f}x" if number > 0 else "N/A"


def _render_context_banner(title: str, copy: str):
    """Render a compact context banner above the chat or data source blocks."""
    st.markdown(
        f"""
        <div class="vmp-context-banner">
            <div class="vmp-context-title">{escape(title)}</div>
            <div class="vmp-context-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_quality_badges(quality_status: str, sector_label: str, technical_snapshot: dict | None = None):
    """Render compact reliability/context badges near the top of the report."""
    tech = technical_snapshot or {}
    technical_label = tech.get("trend_label") or "Technique lazy-load"
    st.markdown(
        f"""
        <div class="vmp-chip-row">
            <span class="vmp-chip">Donnees: {escape(quality_status)}</span>
            <span class="vmp-chip">Secteur: {escape(sector_label)}</span>
            <span class="vmp-chip">Technique: {escape(str(technical_label))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_export_report(
    *,
    ticker: str,
    snapshot,
    blended_intrinsic: float | None,
    valuation_gap_raw: float | None,
    analyst_target: float,
    quality_status: str,
    sector_profile: dict,
    technical_snapshot: dict | None = None,
) -> str:
    """Build a clean markdown export of the current stock analysis."""
    tech = technical_snapshot or {}
    sector_metrics = ", ".join(sector_profile.get("primary_metrics") or [])
    caveats = [
        sector_profile.get("valuation_caveat"),
        sector_profile.get("risk_caveat"),
        *snapshot.data_quality_reasons[:4],
        *snapshot.valuation_warnings[:4],
    ]
    caveat_lines = "\n".join(f"- {item}" for item in dict.fromkeys(item for item in caveats if item))

    technical_lines = "Technical data has not been loaded yet."
    if tech.get("data_quality") and tech.get("data_quality") != "insufficient":
        technical_lines = "\n".join(
            [
                f"- Trend: {tech.get('trend_label', 'N/A')}",
                f"- Score: {_format_ratio(tech.get('technical_score_out_of_10'), 1).replace('x', '/10')}",
                f"- RSI: {_format_pct(tech.get('rsi14'), 1).replace('%', '')}",
                f"- 3M momentum: {_format_pct(tech.get('momentum_3m_pct'), signed=True)}",
                f"- Drawdown from 52W high: {_format_pct(tech.get('drawdown_from_52w_high_pct'), signed=True)}",
            ]
        )

    return "\n".join(
        [
            f"# {ticker} stock analysis",
            "",
            f"Generated: {date.today().isoformat()}",
            f"Data quality: {quality_status}",
            f"Sector profile: {sector_profile.get('label', 'N/A')}",
            "",
            "## Snapshot",
            f"- Current price: {_format_currency(snapshot.current_price)}",
            f"- Blended fair value: {_format_currency(blended_intrinsic)}",
            f"- Blended upside: {_format_pct(valuation_gap_raw, signed=True)}",
            f"- Analyst target: {_format_currency(analyst_target if analyst_target > 0 else None)}",
            f"- Market cap: {_format_market_cap(snapshot.market_cap)}",
            "",
            "## Sector lens",
            f"- Business model: {sector_profile.get('business_model', 'N/A')}",
            f"- Primary metrics: {sector_metrics or 'N/A'}",
            f"- Benchmark: {snapshot.bench_data.get('name', 'N/A')}",
            "",
            "## Core metrics",
            f"- Sales growth: {snapshot.sales_growth * 100:.1f}%",
            f"- EPS growth: {snapshot.eps_growth * 100:.1f}%",
            f"- P/S: {_format_ratio(snapshot.ps)}",
            f"- P/E: {_format_ratio(snapshot.pe)}",
            f"- Forward P/E: {_format_ratio(snapshot.forward_pe)}",
            f"- Piotroski: {snapshot.piotroski if snapshot.piotroski is not None else 'N/A'}",
            f"- Altman Z: {snapshot.altman_z if snapshot.altman_z is not None else 'N/A'}",
            "",
            "## Technicals",
            technical_lines,
            "",
            "## Caveats",
            caveat_lines or "- No major caveat flagged by the app.",
            "",
            "Educational use only. This is not financial advice.",
        ]
    )


def _valuation_candidates_for_sector(base_res: tuple, sector_profile: dict) -> tuple[dict[str, float], list[str]]:
    """Select valuation methods that make sense for the company's sector."""
    key = str(sector_profile.get("key") or "default")
    all_candidates = {"DCF": base_res[0], "P/S": base_res[1], "P/E": base_res[2]}

    if key == "financial":
        return {"P/E": base_res[2]}, ["Sector guardrail: bank/financial blend excludes DCF and P/S shortcuts."]
    if key == "energy":
        return {"DCF": base_res[0], "P/E": base_res[2]}, ["Sector guardrail: energy blend excludes P/S and favours cash-flow plus cycle-aware earnings."]
    if key in {"industrial", "healthcare"}:
        return {"DCF": base_res[0], "P/E": base_res[2]}, [f"Sector guardrail: {key} blend favours earnings and cash-flow over pure sales multiples."]
    if key == "software":
        return all_candidates, ["Sector guardrail: software/platform blend keeps P/S because earnings may still be scaling."]

    return all_candidates, []


def _render_provenance_panel(entries: list[dict]):
    """Render provenance cards with fully native Streamlit blocks."""
    if not entries:
        return

    for start in range(0, len(entries), 2):
        row_entries = entries[start : start + 2]
        cols = st.columns(len(row_entries))
        for col, entry in zip(cols, row_entries):
            with col:
                with st.container(border=True):
                    label = str(entry.get("label", "")).strip()
                    source = str(entry.get("source", "")).strip()
                    meta = str(entry.get("meta", "")).strip()
                    copy = str(entry.get("copy", "")).strip()

                    if label:
                        st.caption(label.upper())
                    if source:
                        st.markdown(f"**{source}**")
                    if meta:
                        st.caption(meta)
                    if copy:
                        st.write(copy)


def _render_agent_trace(trace: dict | None, placeholder=None, live: bool = False):
    """Render the active AI-agent trace for a single assistant response."""
    if not trace:
        if placeholder:
            placeholder.empty()
        return

    lead_label = trace.get("lead_agent_label") or trace.get("final_author_label") or "Superviseur"
    final_label = trace.get("final_author_label")
    specialist_labels = trace.get("specialist_labels") or []
    chip_labels = specialist_labels or trace.get("labels") or [lead_label]
    if trace.get("used_supervisor") and "Superviseur" not in chip_labels:
        chip_labels = [*chip_labels, "Superviseur"]

    copy_parts = []
    if live:
        copy_parts.append(f"En train de mobiliser {lead_label.lower()}.")
    else:
        copy_parts.append(f"Reponse pilotee par {lead_label.lower()}.")
    if final_label and final_label != lead_label:
        copy_parts.append(f"Synthese finale par {final_label.lower()}.")

    chip_html = "".join(f"""<span class="vmp-agent-chip">{escape(label)}</span>""" for label in chip_labels)
    html = f"""
        <div class="vmp-agent-trace{' is-live' if live else ''}">
            <div class="vmp-agent-trace-label">{'Agents actifs' if live else 'Agents mobilises'}</div>
            <div class="vmp-agent-trace-title">{escape(lead_label)}</div>
            <div class="vmp-agent-trace-copy">{escape(' '.join(copy_parts))}</div>
            <div class="vmp-agent-chip-row">{chip_html}</div>
        </div>
    """

    target = placeholder if placeholder is not None else st
    target.markdown(html, unsafe_allow_html=True)


def _chat_message_exists(messages: list[dict], request_id: str, role: str) -> bool:
    """Check whether a chat message for a given request was already recorded."""
    return any(
        message.get("request_id") == request_id and message.get("role") == role
        for message in messages
    )


def _render_overview_card(
    ticker: str,
    long_name: str,
    sector_name: str,
    benchmark_name: str,
    valuation_label: str,
    risk_label: str,
    profile_label: str,
):
    """Render the high-level presentation card for the selected stock."""
    st.markdown(
        f"""
        <div class="vmp-overview">
            <div class="vmp-overview-kicker">{ticker} | {sector_name}</div>
            <div class="vmp-overview-title">{long_name}</div>
            <div class="vmp-overview-copy">
                Lecture rapide: {valuation_label.lower()}, {risk_label.lower()} et un profil {profile_label.lower()}
                face au benchmark {benchmark_name}.
            </div>
            <div class="vmp-chip-row">
                <span class="vmp-chip">{valuation_label}</span>
                <span class="vmp-chip">{risk_label}</span>
                <span class="vmp-chip">{profile_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_insight_card(label: str, value: str, copy: str):
    """Render a compact insight card."""
    st.markdown(
        f"""
        <div class="vmp-insight-card">
            <div class="vmp-insight-label">{label}</div>
            <div class="vmp-insight-value">{value}</div>
            <div class="vmp-insight-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_picker_options() -> list[tuple[str, str]]:
    """Transform the ticker database into cleaner grouped labels for the selector."""
    options: list[tuple[str, str]] = []
    current_group = "Selection rapide"
    for item in TICKER_DB:
        if item.startswith("---"):
            current_group = item.strip("- ").strip()
            continue
        if item.lower().startswith("autre"):
            options.append(("Autre • Saisie manuelle", item))
        else:
            options.append((f"{current_group} • {item}", item))
    return options


def _picker_label_for_ticker(ticker: str, picker_options: list[tuple[str, str]]) -> str | None:
    """Find the current picker label matching a target ticker."""
    target = (ticker or "").upper().strip()
    if not target:
        return None
    for display, raw in picker_options:
        if raw.upper().startswith(f"{target} "):
            return display
    return None


def _build_picker_options() -> list[tuple[str, str]]:
    """Override the picker labels with clean ASCII group labels."""
    options: list[tuple[str, str]] = []
    current_group = "Selection rapide"
    for item in TICKER_DB:
        if item.startswith("---"):
            current_group = item.strip("- ").strip()
            continue
        if item.lower().startswith("autre"):
            options.append(("Autre | Saisie manuelle", item))
        else:
            options.append((f"{current_group} | {item}", item))
    return options


def _picker_label_for_ticker(ticker: str, picker_options: list[tuple[str, str]]) -> str | None:
    """Override ticker label resolution with the cleaned picker labels."""
    target = (ticker or "").upper().strip()
    if not target:
        return None
    for display, raw in picker_options:
        if raw.upper().startswith(f"{target} "):
            return display
    return None


def render_stock_analyzer(api_key: str, sidebar_state: dict | None = None):
    """
    Render the Stock Analyzer mode UI.
    
    Args:
        api_key: Google API key for AI analysis
    """
    sidebar_state = sidebar_state or {}

    st.subheader("Rechercher une entreprise")
    picker_options = _build_picker_options()
    picker_labels = [label for label, _raw in picker_options]
    selected_override = st.session_state.get("selected_ticker_override") or "MSFT"
    default_picker = _picker_label_for_ticker(str(selected_override), picker_options) or picker_labels[min(2, len(picker_labels) - 1)]
    if st.session_state.get("stock_picker_force_sync") or st.session_state.get("stock_picker_display") not in picker_labels:
        st.session_state["stock_picker_display"] = default_picker
        st.session_state["stock_picker_force_sync"] = False

    choice_display = st.selectbox("Choisis une action suivie", picker_labels, key="stock_picker_display")
    choice = dict(picker_options)[choice_display]

    ticker_final = "MSFT"
    if choice.lower().startswith("autre"):
        ticker_input = st.text_input("Ticker", "").upper()
        if ticker_input:
            ticker_final = ticker_input
    elif "-" in choice:
        ticker_final = choice.split("-")[0].strip()

    st.session_state["selected_ticker_override"] = ticker_final
    st.caption(f"Analyse en cours: **{ticker_final}**")
    manual_shares = float(sidebar_state.get("manual_shares", 0.0) or 0.0)
    snapshot = prepare_analyzer_snapshot(
        ticker_final,
        manual_shares_millions=manual_shares,
        cache_version=FINANCIAL_DATA_CACHE_VERSION,
    )
    data = snapshot.data
    
    if data.get("error"):
        st.warning(f"Data fetch warning: {data['error']}")
    
    current_price = snapshot.current_price

    if current_price <= 0:
        st.error("Prix introuvable. Vérifiez le ticker.")
        st.stop()

    bs = snapshot.bs
    inc = snapshot.inc
    cf = snapshot.cf
    piotroski = snapshot.piotroski
    
    # Shares override
    shares = snapshot.shares
    st.sidebar.markdown("### 🔧 Data Override")
    if snapshot.manual_shares_applied:
        st.sidebar.success(f"Using manual shares: {shares:,.0f}")
    if snapshot.share_count_unavailable:
        st.warning("Share count unavailable. Enter it manually in the sidebar if needed.")

    market_cap = snapshot.market_cap
    altman_z = snapshot.altman_z

    revenue_ttm = snapshot.revenue_ttm
    cfo_ttm = snapshot.cfo_ttm
    capex_ttm = snapshot.capex_ttm
    fcf_ttm = snapshot.fcf_ttm
    cash = snapshot.cash
    debt = snapshot.debt
    eps_ttm = snapshot.eps_ttm
    pe = snapshot.pe
    forward_pe = snapshot.forward_pe
    quote_currency = snapshot.quote_currency
    financial_currency = snapshot.financial_currency
    ps = snapshot.ps
    cur_sales_gr = snapshot.sales_growth
    cur_eps_gr = snapshot.eps_growth
    bench_data = snapshot.bench_data
    is_financial = snapshot.is_financial
    sector_profile = snapshot.sector_profile
    metrics = snapshot.metrics
    scores = snapshot.scores

    tech_bundle: dict | None = None

    def load_technical_bundle() -> dict:
        nonlocal tech_bundle
        if tech_bundle is None:
            with st.spinner("Loading technical dataset..."):
                price_df = fetch_price_history(ticker_final, "1y")
                tech_df = add_indicators(price_df)
                pattern = bull_flag_score(tech_df)
                tech = summarize_technical_setup(tech_df, pattern)
            tech_bundle = {"price_df": price_df, "tech_df": tech_df, "tech": tech}
        return tech_bundle

    # Help section
    with st.expander(f"Market context: {bench_data['name']} vs {ticker_final}", expanded=False):
        st.write(f"**Peers:** {bench_data.get('peers', 'N/A')}")
        st.markdown("### 🏢 Sector / Peer Averages")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peer Sales Gr.", f"{bench_data['gr_sales']:.0f}%")
        c2.metric("Peer EPS Gr.", f"{bench_data.get('gr_eps', 0):.0f}%")
        c3.metric("Peer Target P/S", f"{bench_data['ps']}x")
        c4.metric("Peer Target P/E", f"{bench_data.get('pe', 20)}x")

        st.divider()

        st.markdown(f"### 📍 {ticker_final} Current Metrics (Actual)")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Actual Sales Gr.", f"{cur_sales_gr*100:.1f}%", delta_color="off")
        c6.metric("Actual EPS Gr.", f"{cur_eps_gr*100:.1f}%", delta_color="off")
        c7.metric("Actual P/S", f"{ps:.1f}x", delta_color="off")
        c8.metric("Actual P/E", f"{pe:.1f}x", delta_color="off")

    # Assumptions
    with st.expander("⚙️ Edit Assumptions", expanded=False):
        c1, c2, c3 = st.columns(3)
        gr_sales = c1.number_input("Sales Growth %", value=float(bench_data['gr_sales']))
        gr_fcf = c2.number_input("FCF Growth %", value=float(bench_data['gr_fcf']))
        wacc = c3.number_input("WACC %", value=float(bench_data['wacc']))
        c4, c5 = st.columns(2)
        target_pe = c4.number_input("Target P/E", value=float(bench_data.get('pe', 20)))
        target_ps = c5.number_input("Target P/S", value=float(bench_data['ps']))

    # Calculation scenarios
    def run_calc(g_fac, m_fac, w_adj):
        return calculate_valuation(
            gr_sales/100*g_fac, gr_fcf/100*g_fac, 0.10, wacc/100 + w_adj, 
            target_ps*m_fac, target_pe*m_fac, 
            revenue_ttm, fcf_ttm, eps_ttm, cash, debt, shares
        )

    bear_res = run_calc(0.8, 0.8, 0.01)
    base_res = run_calc(1.0, 1.0, 0.0)
    bull_res = run_calc(1.2, 1.2, -0.01)

    sector_candidates, sector_guardrail_notes = _valuation_candidates_for_sector(base_res, sector_profile)
    blended_intrinsic, guardrail_warnings = blend_reasonable_intrinsic_values(sector_candidates, current_price)
    valuation_gap_raw = upside_pct(blended_intrinsic, current_price)
    valuation_gap = valuation_gap_raw if valuation_gap_raw is not None else 0.0
    analyst_target = float(data.get("target_price") or 0)
    analyst_gap = ((analyst_target / current_price) - 1) * 100 if analyst_target and current_price else 0.0
    next_earnings = snapshot.next_earnings
    valuation_verdict = valuation_label(valuation_gap)
    risk_verdict = risk_label(altman_z, piotroski, is_financial=is_financial)
    profile_verdict = profile_label(cur_sales_gr, cur_eps_gr, ps, float(bench_data.get("ps", 0) or 0), is_financial=is_financial)

    _render_overview_card(
        ticker=ticker_final,
        long_name=str(data.get("long_name") or ticker_final),
        sector_name=str(data.get("sector") or "Default"),
        benchmark_name=bench_data["name"],
        valuation_label=valuation_verdict,
        risk_label=risk_verdict,
        profile_label=profile_verdict,
    )

    top_metrics = st.columns(5)
    top_metrics[0].metric("Prix actuel", f"{current_price:.2f} $")
    top_metrics[1].metric(
        "Juste valeur mixte",
        _format_currency(blended_intrinsic),
        delta=f"{valuation_gap:+.1f}%" if valuation_gap_raw is not None else None,
    )
    top_metrics[2].metric(
        "Objectif analystes",
        _format_currency(analyst_target if analyst_target > 0 else None),
        delta=f"{analyst_gap:+.1f}%" if analyst_target > 0 else None,
    )
    if is_financial:
        top_metrics[3].metric("Bilan", "Modele bancaire")
    else:
        top_metrics[3].metric("Cash net / dette", _format_market_cap(cash - debt))
    top_metrics[4].metric("Prochaine publication", next_earnings)

    revenue_basis = snapshot.revenue_basis
    eps_basis = snapshot.eps_basis
    pe_basis = "Yahoo trailing P/E" if float(data.get("pe_ratio", 0) or 0) else "Computed from price / EPS TTM"
    as_of_date = date.today().isoformat()
    recent_news_count = snapshot.recent_news_count
    press_release_count = snapshot.press_release_count
    all_quality_notes = list(
        dict.fromkeys(
            [
                *snapshot.data_quality_reasons,
                *snapshot.valuation_warnings,
                *sector_guardrail_notes,
                *guardrail_warnings,
            ]
        )
    )
    quality_status = quality_label(snapshot.data_quality)
    cached_technical_snapshot = st.session_state.get(f"ai_chat_tech_{ticker_final}", {})
    _render_quality_badges(quality_status, str(sector_profile.get("label") or "General"), cached_technical_snapshot)

    _render_context_banner(
        "Etat des donnees",
        f"Statut {quality_status}. Mis a jour au {as_of_date}. Cache {FINANCIAL_DATA_CACHE_VERSION}. Les prix viennent du flux live, les etats sont caches, et le chat consomme ce meme contexte date.",
    )
    _render_context_banner(
        "Lecture sectorielle",
        (
            f"{sector_profile_summary(sector_profile)} "
            f"{sector_profile.get('valuation_caveat', '')} "
            f"{sector_profile.get('risk_caveat', '')}"
        ),
    )
    if all_quality_notes:
        with st.expander("Notes de qualite des donnees et garde-fous", expanded=snapshot.data_quality == "critical"):
            for note in all_quality_notes[:8]:
                st.caption(f"- {note}")
    with st.expander("Rapport exportable et limites de lecture", expanded=False):
        st.caption("Le rapport reprend les chiffres visibles, les garde-fous et les limites sectorielles.")
        report_markdown = _build_export_report(
            ticker=ticker_final,
            snapshot=snapshot,
            blended_intrinsic=blended_intrinsic,
            valuation_gap_raw=valuation_gap_raw,
            analyst_target=analyst_target,
            quality_status=quality_status,
            sector_profile=sector_profile,
            technical_snapshot=cached_technical_snapshot,
        )
        st.download_button(
            "Exporter le rapport Markdown",
            data=report_markdown,
            file_name=f"{ticker_final}_valuation_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.caption(
            "Limite importante: les modeles de valorisation ne remplacent pas une verification manuelle des etats financiers, "
            "surtout pour les banques, les ressources et les dossiers cycliques."
        )
    _render_provenance_panel(
        [
            {
                "label": "Marche",
                "source": "Yahoo Finance quote + recommendations",
                "meta": f"Au {as_of_date} | Devise de cotation {quote_currency} | Qualite {quality_status}",
                "copy": f"Le prix actuel et l'objectif analystes viennent du flux live de {ticker_final}.",
            },
            {
                "label": "Contexte P/E",
                "source": f"{pe_basis} + Yahoo forward P/E",
                "meta": f"Au {as_of_date} | Devise financiere {financial_currency}",
                "copy": f"Trailing P/E {pe:.1f}x, forward P/E {forward_pe:.1f}x et EPS TTM {eps_ttm:.2f}. Base EPS: {eps_basis}.",
            },
            {
                "label": "Ventes et bilan",
                "source": "Yahoo quarterly / annual statements",
                "meta": f"Au {as_of_date} | Base revenu: {revenue_basis}",
                "copy": (
                    f"Revenu TTM {_format_market_cap(revenue_ttm)}, cash {_format_market_cap(cash)} et dette {_format_market_cap(debt)} alimentent le modele de valorisation."
                    if not is_financial
                    else f"Revenu TTM {_format_market_cap(revenue_ttm)} et dernieres lignes d'etats utilisees, sans raccourci de dette nette pour les banques."
                ),
            },
            {
                "label": "Garde-fous valuation",
                "source": "Sanity checks internes",
                "meta": f"{len(guardrail_warnings)} sorties filtrees | {len(sector_guardrail_notes)} regle(s) sectorielle(s)",
                "copy": "La juste valeur mixte ignore les sorties negatives, manquantes ou trop extremes afin d'eviter les faux upside spectaculaires.",
            },
            {
                "label": "Lecture sectorielle",
                "source": sector_profile.get("label", "General operating company"),
                "meta": "Modele d'affaires + metriques prioritaires",
                "copy": sector_profile.get("valuation_caveat", "La lecture depend du secteur et du cycle."),
            },
            {
                "label": "Contexte IA news",
                "source": "yfinance news + Google News RSS + Yahoo calendar",
                "meta": f"Au {as_of_date} | {recent_news_count} news de marche, {press_release_count} communiques",
                "copy": f"Le chat utilise des titres dates et le calendrier des resultats, avec liens lorsque disponibles.",
            },
        ]
    )

    insight_cols = st.columns(3)
    with insight_cols[0]:
        _render_insight_card(
            "Contexte marche",
            bench_data["name"],
            f"Pairs: {bench_data.get('peers', 'N/A')}. P/S cible {bench_data['ps']}x et P/E cible {bench_data.get('pe', 20)}x.",
        )
    with insight_cols[1]:
        if is_financial:
            _render_insight_card(
                "Profil bancaire",
                f"EPS {cur_eps_gr * 100:.1f}% | P/E {pe:.1f}x",
                "Pour les banques, les depots, le mix de financement et les ratios de capital comptent plus que le FCF yield ou la Rule of 40.",
            )
        else:
            _render_insight_card(
                "Profil operationnel",
                f"Sales {cur_sales_gr * 100:.1f}% | EPS {cur_eps_gr * 100:.1f}%",
                f"FCF yield {(fcf_ttm / market_cap) * 100:.1f}% and Rule of 40 {metrics['rule_40'] * 100:.1f}%.",
            )
    with insight_cols[2]:
        if is_financial:
            _render_insight_card(
                "Qualite du dossier",
                f"Piotroski {piotroski if piotroski else 'N/A'} | Altman N/A",
                f"Capitalisation {_format_market_cap(market_cap)} avec {shares / 1_000_000:.1f}M d'actions. Altman Z et la dette nette ne sont pas des raccourcis fiables pour les banques.",
            )
        else:
            _render_insight_card(
                "Qualite du dossier",
                f"Piotroski {piotroski if piotroski else 'N/A'} | Altman {altman_z:.2f}",
                f"Capitalisation {_format_market_cap(market_cap)} avec {shares / 1_000_000:.1f}M d'actions dans le modele.",
            )
    
    st.markdown("### Analyse detaillee")

    # Define Section Menu
    section = st.radio(
        "Section",
        [
            "📊 Fundamentals",
            "💵 DCF (Cash)",
            "📈 Sales (P/S)",
            "💰 Earnings (P/E)",
            "🧱 Assets",
            "🎯 Analystes",
            "👥 Insiders",
            "📉 Tech",
            "📊 Scorecard",
            "🤖 AI Agent",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if "Fundamentals" in section:
        section_key = "Fundamentals"
    elif "DCF" in section:
        section_key = "DCF"
    elif "Sales" in section:
        section_key = "Sales"
    elif "Earnings" in section:
        section_key = "Earnings"
    elif "Assets" in section:
        section_key = "Assets"
    elif "Analyst" in section:
        section_key = "Analysts"
    elif "Insiders" in section:
        section_key = "Insiders"
    elif "Tech" in section:
        section_key = "Tech"
    elif "Scorecard" in section:
        section_key = "Scorecard"
    else:
        section_key = "AI Chat"

    st.caption(
        {
            "Fundamentals": "Etats historiques, marges, liquidite et qualite operationnelle.",
            "DCF": "Valorisation par les flux avec scenarios et matrice de sensibilite.",
            "Sales": "Lecture du multiple de ventes face au groupe de pairs.",
            "Earnings": "Lecture du multiple de benefices et du positionnement relatif.",
            "Assets": "Valeur plancher portee par le bilan.",
            "Analysts": "Consensus, objectifs de prix et mouvements de recommandations.",
            "Insiders": "Transactions recentes des dirigeants depuis Yahoo Finance.",
            "Tech": "Tendance, overlays, oscillateurs et short interest.",
            "Scorecard": "Sante, croissance et valorisation en une seule synthese.",
            "AI Chat": "Conversation multi-agents sur la valo, les news, les catalyseurs et les comparaisons.",
        }[section_key]
    )

    if section_key in {"Analysts", "Tech", "Scorecard"}:
        technical_data = load_technical_bundle()
        price_df = technical_data["price_df"]
        tech_df = technical_data["tech_df"]
        tech = technical_data["tech"]
        st.session_state[f"ai_chat_tech_{ticker_final}"] = dict(tech)
    else:
        price_df = pd.DataFrame()
        tech_df = pd.DataFrame()
        tech = st.session_state.get(
            f"ai_chat_tech_{ticker_final}",
            {"score": 0, "is_bull_flag": False},
        )

    # --- SECTION: FUNDAMENTALS ---
    if section == "📊 Fundamentals":
        st.subheader("📊 Extended Financial Dashboard")
        st.caption("Source: StockAnalysis.com")
        
        # Display Frequency - FORCE ANNUAL
        st.caption("Frequency: Annual")
        freq_dashboard = "Annual"
        
        # Fetch Extended Data ("version=3" to ensure cache bust if needed, or stick to 2)
        with st.spinner("Fetching extended historical data..."):
            sa_data = get_extended_history(ticker_final, version=2)
            
        if sa_data.get("error"):
            st.error(f"Could not load extended data: {sa_data['error']}")
        else:
            # 1. Prepare Helper Function for Data Processing
            def process_financial_metrics(inc_df, bs_df, cf_df):
                if inc_df.empty and bs_df.empty and cf_df.empty:
                    return pd.DataFrame()
                    
                # Transpose
                def clean_transpose(df):
                    if df.empty: return pd.DataFrame()
                    df_t = df.T
                    if "TTM" in df_t.index: df_t = df_t.rename(index={"TTM": "Last 12M"})
                    return df_t

                df_inc = clean_transpose(inc_df)
                df_bs = clean_transpose(bs_df)
                df_cf = clean_transpose(cf_df)
                
                # Merge
                df_all = df_inc.join(df_bs, lsuffix="_inc", rsuffix="_bs", how="outer")
                df_all = df_all.join(df_cf, rsuffix="_cf", how="outer")
                
                # Sort
                try:
                    df_all.index = df_all.index.astype(str)
                    
                    # Extract 4-digit year for proper sorting
                    df_all["_sort_val"] = pd.to_numeric(df_all.index.str.extract(r'(\d{4})')[0], errors='coerce')
                    
                    # Sort Ascending (Oldest -> Newest)
                    df_all = df_all.sort_values("_sort_val", ascending=True)
                    df_all = df_all.drop(columns=["_sort_val"])
                except Exception as e:
                    pass

                # Helper col finder
                def get_col(df, candidates):
                    for c in df.columns:
                        for cand in candidates:
                            if cand.lower().replace(" ", "") == str(c).lower().replace(" ", ""):
                                return c
                    return None
                    
                # --- Calculations ---
                
                # FCF
                col_ocf = get_col(df_all, ["Operating Cash Flow", "Cash From Operations"])
                col_capex = get_col(df_all, ["Capital Expenditures", "CapEx"])
                if col_ocf and col_capex:
                     df_all["Free Cash Flow"] = df_all[col_ocf] - df_all[col_capex].abs()

                # Margins
                col_rev = get_col(df_all, ["Total Revenue", "Revenue"])
                col_gp = get_col(df_all, ["Gross Profit"])
                col_op = get_col(df_all, ["Operating Income"])
                col_ni = get_col(df_all, ["Net Income", "Net Income Common Stockholders"])
                
                if col_rev:
                    if col_gp: df_all["Gross Margin %"] = (df_all[col_gp] / df_all[col_rev]) * 100
                    if col_op: df_all["Operating Margin %"] = (df_all[col_op] / df_all[col_rev]) * 100
                    if col_ni: df_all["Net Margin %"] = (df_all[col_ni] / df_all[col_rev]) * 100

                # ROIC Proxy
                col_equity = get_col(df_all, ["Total Stockholders' Equity", "Total Equity", "Shareholders' Equity"])
                col_debt = get_col(df_all, ["Total Debt"])
                col_cash = get_col(df_all, ["Cash & Equivalents", "Cash and Equivalents", "Total Cash & Equivalents"])
                
                if col_op and col_equity and col_debt and col_cash:
                    nopat = df_all[col_op] * 0.79
                    invested_capital = df_all[col_equity] + df_all[col_debt] - df_all[col_cash]
                    df_all["ROIC %"] = (nopat / invested_capital) * 100
                
                # Shares
                # (We assign this to specific column name to find it later easily)
                col_shares = get_col(df_all, ["Shares Outstanding (Basic)", "Shares Outstanding", "Weighted Average Shares"])
                if col_shares:
                    df_all["Shares Outstanding"] = df_all[col_shares]

                # Liquidity Ratios
                col_ca = get_col(df_all, ["Total Current Assets", "Current Assets"])
                col_cl = get_col(df_all, ["Total Current Liabilities", "Current Liabilities"])
                col_inv = get_col(df_all, ["Inventory", "Inventories"])
                col_ta = get_col(df_all, ["Total Assets"])
                col_opex = get_col(df_all, ["Total Operating Expenses", "Operating Expenses"])
                
                if not col_opex:
                     c_cogs = get_col(df_all, ["Cost of Revenue", "COGS"])
                     c_sgna = get_col(df_all, ["Selling, General and Administrative", "SG&A", "Operating Expenses"])
                     if c_cogs and c_sgna:
                         df_all["Total_OpEx_Calc"] = df_all[c_cogs] + df_all[c_sgna]
                         col_opex = "Total_OpEx_Calc"
                
                if col_ca and col_cl:
                    df_all["Current Ratio (Fonds de roulement)"] = df_all[col_ca] / df_all[col_cl]
                    df_all["Net Working Capital (FRN)"] = df_all[col_ca] - df_all[col_cl]
                    if col_ta:
                        df_all["FRN / Total Assets"] = (df_all["Net Working Capital (FRN)"] / df_all[col_ta]) * 100
                    if col_inv:
                        df_all["Quick Ratio (Trésorerie)"] = (df_all[col_ca] - df_all[col_inv]) / df_all[col_cl]
                    if col_cash:
                        df_all["Cash Ratio (Liquidité immédiate)"] = df_all[col_cash] / df_all[col_cl]
                    if col_opex:
                        daily_opex = df_all[col_opex] / 365
                        df_all["Defensive Interval (Days)"] = df_all[col_ca] / daily_opex
                        
                # Clean up Infs and NaNs
                import numpy as np
                df_all = df_all.replace([np.inf, -np.inf], np.nan)
                
                return df_all

            # 2. Process Datasets (Annual Only)
            df_annual = process_financial_metrics(
                sa_data.get("inc_a", pd.DataFrame()),
                sa_data.get("bs_a", pd.DataFrame()),
                sa_data.get("cf_a", pd.DataFrame())
            )
            
            df_main = df_annual
            
            if df_main.empty:
                st.warning("No data found.")
            else:
                pass

                # Helper Col Finder
                def get_col(df, candidates):
                    for c in df.columns:
                         for cand in candidates:
                             if cand.lower().replace(" ", "") == str(c).lower().replace(" ", ""):
                                 return c
                    return None

                # Plot Grid
                metrics_grid = [
                    {"label": "Revenue", "match": ["Revenue", "Total Revenue"], "color": "#ffaa00"}, 
                    {"label": "Net Income", "match": ["Net Income", "Profit"], "color": "#00aaff"}, 
                    {"label": "EBITDA", "match": ["EBITDA"], "color": "#aa00ff"}, 
                    {"label": "EPS", "match": ["EPS (Basic)", "Earnings Per Share"], "color": "#ffff00"}, 
                    {"label": "Free Cash Flow", "match": ["Free Cash Flow"], "color": "#00ff00"},
                    {"label": "Dividends Paid", "match": ["Dividends Paid"], "color": "#ff00aa"},
                ]
                
                cols = st.columns(3)
                for i, m in enumerate(metrics_grid):
                    col = cols[i % 3]
                    found_col = get_col(df_main, m["match"])
                    with col:
                        if found_col:
                            st.markdown(f"**{m['label']}**")
                            st.bar_chart(df_main[found_col], color=m["color"], height=200)
                        else:
                            st.info(f"{m['label']} data not available")

                # Restore Balance Sheet Health
                st.markdown("---")
                st.markdown("### 💰 Balance Sheet Health")
                c_bal1, c_bal2 = st.columns(2)
                
                col_cash = get_col(df_main, ["Cash & Equivalents", "Cash and Equivalents", "Total Cash & Equivalents"])
                col_debt = get_col(df_main, ["Total Debt"])
                
                with c_bal1:
                     st.markdown("**Cash vs Total Debt**")
                     if col_cash and col_debt:
                         st.bar_chart(df_main[[col_cash, col_debt]], color=["#00ff00", "#ff0000"], height=250)
                     else:
                         st.info("Balance sheet data missing for Cash/Debt comparison")

                # Restore Advanced Metrics
                st.markdown("---")
                c_adv_title, c_adv_freq = st.columns([3, 1])
                with c_adv_title:
                    st.markdown("### 📈 Advanced Metrics (Margins, Efficiency, Liquidity)")
                with c_adv_freq:
                    st.caption("Annual")
                
                df_adv = df_annual
                
                if df_adv.empty:
                    st.warning("No data for Advanced Metrics.")
                else:
                    dynamic_options = [
                        "Gross Margin %", "Operating Margin %", "Net Margin %", 
                        "ROIC %", "Shares Outstanding",
                        "Current Ratio (Fonds de roulement)", "Quick Ratio (Trésorerie)", "Cash Ratio (Liquidité immédiate)",
                        "Net Working Capital (FRN)", "FRN / Total Assets", "Defensive Interval (Days)"
                    ]
                    
                    metrics_descriptions = {
                        "Current Ratio (Fonds de roulement)": "Actif CT / Passif CT. Capacité à payer les dettes courantes. (>1 visé)",
                        "Quick Ratio (Trésorerie)": "(Actif CT - Stocks) / Passif CT. Capacité à payer sans vendre les stocks (plus prudent).",
                        "Cash Ratio (Liquidité immédiate)": "Cash / Passif CT. Cash disponible immédiatement.",
                        "Net Working Capital (FRN)": "Actif CT - Passif CT. Marge de sécurité en dollars.",
                        "FRN / Total Assets": "Part de l'actif total qui représente la liquidité nette.",
                        "Defensive Interval (Days)": "Actif CT / (Dépenses/Jour). Jours de survie sans aucun revenu.",
                        "Gross Margin %": "Marge Brute (Pricing Power).",
                        "Operating Margin %": "Marge Opérationnelle (Efficacité).",
                        "Net Margin %": "Marge Nette (Profitabilité finale).",
                        "ROIC %": "Retour sur Capital Investi (Qualité du business).",
                        "Shares Outstanding": "Nombre d'actions (Baisse = Rachats = Positif)."
                    }
                    
                    available_dynamic = [opt for opt in dynamic_options if opt in df_adv.columns]
                    
                    selected_dynamic = st.multiselect("Select additional metrics to visualize:", available_dynamic, default=["Operating Margin %", "Shares Outstanding"] if "Shares Outstanding" in available_dynamic else [])
                    
                    if selected_dynamic:
                        dyn_cols = st.columns(2)
                        for i, metric_name in enumerate(selected_dynamic):
                            col = dyn_cols[i % 2]
                            with col:
                                st.markdown(f"**{metric_name}**")
                                color = "#00aaff"
                                if "Margin" in metric_name: color = "#ffaa00"
                                if "ROIC" in metric_name: color = "#aa00ff"
                                if "Shares" in metric_name: color = "#ff00aa" 
                                if "Ratio" in metric_name: color = "#00ffaa"
                                if "FRN" in metric_name: color = "#00ffaa"
                                if "Defensive" in metric_name: color = "#55aaff"
                                
                                st.bar_chart(df_adv[metric_name], color=color, height=250)
                                desc = metrics_descriptions.get(metric_name)
                                if desc: st.caption(desc)

    elif section == "💵 DCF (Cash)":
        st.subheader("💵 Buy Price (DCF)")
        st.caption(sector_profile.get("valuation_caveat", "Use DCF with sanity checks."))
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[0]:.2f} $", delta=f"{base_res[0]-current_price:.2f}")
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[0]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[0]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[0]:.2f} $")
        
        st.markdown("##### Reverse DCF")
        implied_g = solve_reverse_dcf(current_price, fcf_ttm, wacc/100, shares, cash, debt)
        st.metric("Market Implied Growth", f"{implied_g*100:.1f}%")
        
        # Sensitivity matrix
        st.markdown("##### 🌡️ Sensitivity Matrix (Price vs Growth & WACC)")
        sens_wacc = [wacc-1, wacc-0.5, wacc, wacc+0.5, wacc+1]
        sens_growth = [gr_fcf-2, gr_fcf-1, gr_fcf, gr_fcf+1, gr_fcf+2]
        res_matrix = []
        for w in sens_wacc:
            row_vals = []
            for g in sens_growth:
                val, _, _ = calculate_valuation(0, g/100, 0, w/100, 0, 0, revenue_ttm, fcf_ttm, 0, cash, debt, shares)
                row_vals.append(val)
            res_matrix.append(row_vals)
        df_sens = pd.DataFrame(res_matrix, index=[f"WACC {w:.1f}%" for w in sens_wacc], columns=[f"Gr {g:.1f}%" for g in sens_growth])
        st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f} $"))

    elif section == "📈 Sales (P/S)":
        st.subheader("📈 Buy Price (Sales)")
        st.caption(sector_profile.get("valuation_caveat", "Compare P/S against similar business models."))
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[1]:.2f} $", delta=f"{base_res[1]-current_price:.2f}")
        s1, s2, s3 = st.columns(3)
        s1.metric("Current P/S", f"{ps:.1f}x")
        s2.metric("Revenue TTM", _format_market_cap(revenue_ttm))
        s3.metric("Peer Target P/S", f"{float(bench_data.get('ps', 0) or 0):.1f}x")
        st.caption(f"Source: market cap from Yahoo quote, revenue from {revenue_basis}. As of {as_of_date}.")
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[1]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[1]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[1]:.2f} $")
        st.write("")
        display_relative_analysis(ps, float(bench_data.get('ps', 3)), "P/S", bench_data['name'])

    elif section == "💰 Earnings (P/E)":
        st.subheader("💰 Buy Price (P/E)")
        st.caption(sector_profile.get("valuation_caveat", "Compare P/E against peers with similar cyclicality and margin profile."))
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f} $")
        c2.metric("Intrinsic (Neutral)", f"{base_res[2]:.2f} $", delta=f"{base_res[2]-current_price:.2f}")
        p1, p2, p3 = st.columns(3)
        p1.metric("Trailing P/E", f"{pe:.1f}x")
        p2.metric("Forward P/E", f"{forward_pe:.1f}x" if forward_pe > 0 else "N/A")
        p3.metric("EPS TTM", f"{eps_ttm:.2f}")
        st.caption(
            f"Trailing basis: {pe_basis}. Forward P/E comes from Yahoo consensus. Quote ccy {quote_currency}; financial ccy {financial_currency}. As of {as_of_date}."
        )
        st.divider()
        c_bear, c_base, c_bull = st.columns(3)
        c_bear.metric("🐻 Bear", f"{bear_res[2]:.2f} $")
        c_base.metric("🎯 Neutral", f"{base_res[2]:.2f} $")
        c_bull.metric("🐂 Bull", f"{bull_res[2]:.2f} $")
        st.write("")
        display_relative_analysis(pe, float(bench_data.get('pe', 20)), "P/E", bench_data['name'])

    elif section == "🧱 Assets":
        st.subheader("🧱 Asset Based Value")
        ab = compute_asset_based_value(bs, shares)
        c1, c2 = st.columns(2)
        c1.metric("NAV / Share", f"{ab['nav_ps']:.2f} $")
        c2.metric("Tangible NAV", f"{ab['tnav_ps']:.2f} $")
        st.caption(ab["notes"])

    elif section == "🎯 Analystes":
        st.subheader("🎯 Analyst Ratings & Price Targets")
        
        # We need fresh info for analysts
        try:
            import yfinance as yf
            from datetime import timedelta, datetime
            
            ticker_obj = yf.Ticker(ticker_final)
            info_an = ticker_obj.info
            
            # --- Consensul & Ratings ---
            rec_key = info_an.get("recommendationKey", "none").replace("_", " ").title()
            num_analysts = info_an.get("numberOfAnalystOpinions", 0)
            
            c_an1, c_an2 = st.columns([1, 2])
            
            with c_an1:
                st.metric("Consensus", rec_key)
                st.metric("Analysts count", num_analysts)
                
            with c_an2:
                # Ratings Breakdown
                try:
                    rec_sum = ticker_obj.recommendations_summary
                    if rec_sum is not None and not rec_sum.empty:
                        # usually columns: period, strongBuy, buy, hold, sell, strongSell
                        # taking the latest period (row 0)
                        latest_rec = rec_sum.iloc[0]
                        
                        rec_data = {
                            "Strong Buy": latest_rec.get("strongBuy", 0),
                            "Buy": latest_rec.get("buy", 0),
                            "Hold": latest_rec.get("hold", 0),
                            "Sell": latest_rec.get("sell", 0),
                            "Strong Sell": latest_rec.get("strongSell", 0)
                        }
                        
                        df_rec = pd.DataFrame([rec_data]).T
                        df_rec.columns = ["Count"]
                        
                        # Plot Horizontal Bar
                        st.bar_chart(df_rec, horizontal=True, color="#aa00ff")
                    else:
                        st.info("Detailed ratings breakdown not available.")
                except Exception as e:
                    st.info(f"Could not load ratings breakdown: {e}")

            st.divider()
            
            # --- Price Targets ---
            st.subheader("🔮 12-Month Price Forecast")
            
            tgt_low = info_an.get("targetLowPrice")
            tgt_mean = info_an.get("targetMeanPrice")
            tgt_high = info_an.get("targetHighPrice")
            cur_price = info_an.get("currentPrice", current_price)
            
            if tgt_mean:
                c_t1, c_t2, c_t3 = st.columns(3)
                
                def fmt_upside(target, current):
                    if not target or not current: return "0%"
                    up = ((target - current) / current) * 100
                    # Return simple signed string; Streamlit metric handles the coloring (Red for negative)
                    return f"{up:+.2f}%"

                c_t1.metric("Low Target", f"{tgt_low:.2f} $" if tgt_low else "N/A", fmt_upside(tgt_low, cur_price))
                c_t2.metric("Average Target", f"{tgt_mean:.2f} $" if tgt_mean else "N/A", fmt_upside(tgt_mean, cur_price))
                c_t3.metric("High Target", f"{tgt_high:.2f} $" if tgt_high else "N/A", fmt_upside(tgt_high, cur_price))
                
                # --- Chart (Native Streamlit) ---
                # Combine Historical Data with Future Projections
                
                # 1. Get History (1y)
                df_use = tech_df.copy() if not tech_df.empty else price_df.copy()
                
                if "Date" in df_use.columns:
                    df_use["Date"] = pd.to_datetime(df_use["Date"])
                    df_use = df_use.set_index("Date")
                
                hist_series = df_use["Close"]
                
                if not hist_series.empty:
                    last_date = hist_series.index[-1]
                    if isinstance(last_date, (pd.Timestamp, datetime)):
                        last_price = hist_series.iloc[-1]
                        future_date = last_date + timedelta(days=365)
                        
                        joined_index = hist_series.index.union(pd.Index([future_date]))
                        df_final = pd.DataFrame(index=joined_index)
                        
                        df_final.loc[hist_series.index, "History"] = hist_series
                        
                        df_final.loc[last_date, "High Target"] = last_price
                        df_final.loc[future_date, "High Target"] = tgt_high
                        
                        df_final.loc[last_date, "Mean Target"] = last_price
                        df_final.loc[future_date, "Mean Target"] = tgt_mean
                        
                        df_final.loc[last_date, "Low Target"] = last_price
                        df_final.loc[future_date, "Low Target"] = tgt_low
                        
                        st.caption("Price History (Blue) vs Analyst Targets (Colored Lines)")
                        st.line_chart(df_final, color=["#00aaff", "#00ff00", "#ffff00", "#ff0000"])
                    else:
                        st.warning("Could not parse dates for historical chart.")
                
            else:
                st.warning("No price target data available.")
                
            # --- Analyst Actions List ---
            st.divider()
            st.subheader("📋 Détails des Analystes (Upgrades / Downgrades)")
            
            try:
                upgrades = ticker_obj.upgrades_downgrades
                if upgrades is not None and not upgrades.empty:
                    # Clean and Format
                    # 1. Sort Descending
                    upgrades = upgrades.sort_index(ascending=False).head(20)
                    
                    # 2. Reset Index to get Date column
                    upgrades = upgrades.reset_index() # GradeDate becomes a column
                    
                    # 3. Rename Columns
                    # typical cols: GradeDate, Firm, ToGrade, FromGrade, Action
                    # Map to something user-friendly
                    col_map = {
                        "GradeDate": "Date",
                        "Firm": "Firme",
                        "ToGrade": "Nouvelle Note",
                        "FromGrade": "Ancienne Note",
                        "Action": "Action"
                    }
                    upgrades = upgrades.rename(columns=col_map)
                    
                    # 4. Filter columns if needed (keep key ones)
                    cols_to_show = ["Date", "Firme", "Action", "Nouvelle Note", "Ancienne Note"]
                    # Ensure they exist (sometimes Yahoo varies)
                    cols_to_show = [c for c in cols_to_show if c in upgrades.columns]
                    upgrades = upgrades[cols_to_show]
                    
                    # 5. Format Date (YYYY-MM-DD)
                    if "Date" in upgrades.columns:
                        upgrades["Date"] = pd.to_datetime(upgrades["Date"]).dt.strftime('%Y-%m-%d')

                    # 6. Styling Function
                    def style_grades(val):
                        if not isinstance(val, str): return ''
                        v = val.lower()
                        # Green
                        if any(x in v for x in ['buy', 'outperform', 'overweight', 'positive', 'strong buy']):
                            return 'color: #2ecc71; font-weight: bold;'
                        # Red
                        if any(x in v for x in ['sell', 'underperform', 'underweight', 'negative', 'reduce']):
                            return 'color: #e74c3c; font-weight: bold;'
                        # Orange/Yellow
                        if any(x in v for x in ['hold', 'neutral', 'equal', 'market', 'sector']):
                            return 'color: #f39c12;'
                        return ''

                    # Apply Style
                    # Check if 'Nouvelle Note' exists
                    target_col = "Nouvelle Note"
                    if target_col in upgrades.columns:
                        styled_df = upgrades.style.map(style_grades, subset=[target_col])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(upgrades, use_container_width=True, hide_index=True)
                        
                else:
                    st.info("No specific analyst upgrades/downgrades data found.")
            except Exception as e:
                st.info(f"Could not load analyst list: {e}")

        except Exception as e:
            st.error(f"Error loading analyst data: {e}")

    elif section == "👥 Insiders":
        st.subheader("👥 Insider Trading")
        if not data['insiders'].empty:
            st.dataframe(data['insiders'].head(10))
        else:
            st.info("No insider data found.")

    elif section == "📉 Tech":
        st.subheader("📉 Technical Analysis")
        st.caption(
            "Lecture technique basee sur tendance, RSI, moyennes mobiles, momentum, volatilite, drawdown et zones de support/resistance."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Technical score", f"{tech.get('score', 0):.1f}/10")
        c2.metric("Trend", tech.get("trend_label", "N/A"))
        c3.metric("RSI 14", f"{float(tech.get('rsi14') or 0):.1f}" if tech.get("rsi14") is not None else "N/A")
        c4.metric("3M momentum", _format_pct(tech.get("momentum_3m_pct"), signed=True))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("vs SMA50", _format_pct(tech.get("distance_to_sma50_pct"), signed=True))
        c6.metric("vs SMA200", _format_pct(tech.get("distance_to_sma200_pct"), signed=True))
        c7.metric("Drawdown 52W", _format_pct(tech.get("drawdown_from_52w_high_pct"), signed=True))
        c8.metric("Volatilite 20j", _format_pct(tech.get("volatility_20d_pct")))

        support_cols = st.columns(3)
        support_cols[0].metric("Support 60j", _format_currency(tech.get("support_60d")))
        support_cols[1].metric("Resistance 60j", _format_currency(tech.get("resistance_60d")))
        support_cols[2].metric("Timing risk", tech.get("timing_risk_label", "N/A"))

        with st.expander("Comment lire ces signaux", expanded=False):
            st.write(
                "Le score technique sert surtout au timing. Un bon dossier fondamental peut rester faible techniquement, "
                "et un bon setup technique ne suffit pas a valider une these d'investissement."
            )
            st.caption(
                f"Derniere date de prix: {tech.get('last_price_date', 'N/A')}. "
                f"Donnees utilisees: {tech.get('data_points', 'N/A')} sessions."
            )
        
        # UI Controls for Chart
        col_overlays, col_indicators = st.columns(2)
        with col_overlays:
            selected_overlays = st.multiselect(
                "Price Overlays",
                ["SMA20", "SMA50", "SMA100", "SMA200", "Bollinger Bands"],
                default=["SMA50", "SMA200"]
            )
        with col_indicators:
            selected_indicators = st.multiselect(
                "Subplot Indicators",
                ["Volume", "RSI", "MACD", "ATR", "Stochastic", "OBV"],
                default=["Volume", "RSI"]
            )
            
        plot_technical_chart(tech_df, ticker_final, selected_overlays, selected_indicators)

        # --- Short Interest Section ---
        st.divider()
        st.subheader("📊 Short Interest")
        
        # 1. Current Short Data from yfinance info
        try:
            import yfinance as yf
            ticker_obj_short = yf.Ticker(ticker_final)
            info_short = ticker_obj_short.info or {}
            
            short_pct = info_short.get("shortPercentOfFloat")
            short_ratio = info_short.get("shortRatio")  # Days to Cover
            short_count = info_short.get("sharesShort")
            short_prev = info_short.get("sharesShortPriorMonth")
            
            if short_pct is not None or short_ratio is not None:
                c_s1, c_s2, c_s3, c_s4 = st.columns(4)
                
                with c_s1:
                    if short_pct is not None:
                        st.metric("Short % of Float", f"{short_pct*100:.2f}%")
                    else:
                        st.metric("Short % of Float", "N/A")
                        
                with c_s2:
                    if short_ratio is not None:
                        st.metric("Days to Cover", f"{short_ratio:.2f}")
                    else:
                        st.metric("Days to Cover", "N/A")
                        
                with c_s3:
                    if short_count is not None:
                        st.metric("Shares Short", f"{short_count:,.0f}")
                    else:
                        st.metric("Shares Short", "N/A")
                        
                with c_s4:
                    if short_count is not None and short_prev is not None and short_prev > 0:
                        change_pct = ((short_count - short_prev) / short_prev) * 100
                        st.metric("vs Prior Month", f"{short_count:,.0f}", delta=f"{change_pct:+.1f}%", delta_color="inverse")
                    else:
                        st.metric("vs Prior Month", "N/A")
            else:
                st.info("Short interest data not available for this ticker.")
        except Exception as e:
            st.info(f"Could not load current short data: {e}")
        
        # 2. Historical Short Interest from Nasdaq API
        try:
            from fetchers.short_interest import get_historical_short_interest
            
            df_short = get_historical_short_interest(ticker_final)
            
            if not df_short.empty:
                st.markdown("##### 📈 Historical Short Interest (Nasdaq)")
                
                # Clean numeric columns
                import numpy as np
                for col in ["Short Interest", "Avg Daily Volume", "Days to Cover"]:
                    if col in df_short.columns:
                        df_short[col] = pd.to_numeric(
                            df_short[col].astype(str).str.replace(",", ""), errors="coerce"
                        )
                
                # Compute Short % of Float using yfinance float shares
                float_shares = info_short.get("floatShares") if 'info_short' in dir() else None
                if float_shares is None:
                    try:
                        float_shares = yf.Ticker(ticker_final).info.get("floatShares")
                    except:
                        pass
                
                if float_shares and float_shares > 0 and "Short Interest" in df_short.columns:
                    df_short["Short % of Float"] = (df_short["Short Interest"] / float_shares) * 100
                
                # Chart: Short % of Float over time (PRIMARY CHART)
                if "Short % of Float" in df_short.columns and "Date" in df_short.columns:
                    spf_df = df_short.set_index("Date")[["Short % of Float"]].dropna()
                    if not spf_df.empty:
                        st.markdown("**Short % of Float — Évolution**")
                        st.line_chart(spf_df, color="#ff3366", height=280)
                        
                        # Min / Max / Current labels
                        c_min, c_max, c_cur = st.columns(3)
                        c_min.metric("Min (période)", f"{spf_df['Short % of Float'].min():.2f}%")
                        c_max.metric("Max (période)", f"{spf_df['Short % of Float'].max():.2f}%")
                        c_cur.metric("Dernier", f"{spf_df['Short % of Float'].iloc[-1]:.2f}%")
                
                # Chart: Short Interest (absolute count) over time
                if "Short Interest" in df_short.columns and "Date" in df_short.columns:
                    chart_df = df_short.set_index("Date")[["Short Interest"]].dropna()
                    if not chart_df.empty:
                        st.markdown("**Short Interest (nombre d'actions)**")
                        st.line_chart(chart_df, color="#ff6600", height=250)
                
                # Chart: Days to Cover over time
                if "Days to Cover" in df_short.columns and "Date" in df_short.columns:
                    dtc_df = df_short.set_index("Date")[["Days to Cover"]].dropna()
                    if not dtc_df.empty:
                        st.markdown("**Days to Cover**")
                        st.area_chart(dtc_df, color="#aa00ff", height=200)
                
                # Raw data table
                with st.expander("📋 Raw Short Interest Data"):
                    display_cols = [c for c in ["Date", "Short Interest", "Short % of Float", "Avg Daily Volume", "Days to Cover"] if c in df_short.columns]
                    st.dataframe(df_short[display_cols].sort_values("Date", ascending=False).head(24), use_container_width=True, hide_index=True)
            else:
                st.caption("Historical short interest data not available from Nasdaq for this ticker.")
        except Exception as e:
            st.caption(f"Could not load historical short interest: {e}")

    elif section == "📊 Scorecard":
        st.subheader("📊 Scorecard Pro")
        c1, c2, c3 = st.columns(3)
        c1.metric("Health", f"{scores['health']}/10")
        c2.metric("Growth", f"{scores['growth']}/10")
        c3.metric("Value", f"{scores['valuation']}/10")
        fig = plot_radar(scores, tech['score'])
        if fig:
            st.pyplot(fig)
        st.markdown(f"**Piotroski F-Score:** {piotroski if piotroski else 'N/A'}/9")
        if is_financial:
            st.markdown("**Altman Z-Score:** N/A for banks and other financial institutions")
        else:
            st.markdown(f"**Altman Z-Score:** {altman_z:.2f}")

    elif section == "🤖 AI Agent":
        st.subheader("🤖 AI Analyst Chat (Multi-Agent ADK)")
        st.caption("Tu peux maintenant discuter avec l'agent. Il utilise plusieurs sous-agents specialises: fondamentaux, technique, comparaison, actualites, signaux de marche, filings SEC et risque.")
        st.caption("Chaque reponse affiche aussi quels sous-agents ont ete mobilises pendant l'analyse.")

        objective_labels = [preset["label"] for preset in INVESTOR_OBJECTIVES.values()]
        selected_objective_label = st.radio(
            "Objectif de comparaison",
            objective_labels,
            horizontal=True,
            key=f"ai_objective_{ticker_final}",
        )
        objective_key = next(
            (key for key, preset in INVESTOR_OBJECTIVES.items() if preset["label"] == selected_objective_label),
            "balanced",
        )
        objective_snapshot = build_investor_objective_snapshot(objective_key)

        _render_context_banner(
            "Comparison lens",
            f"{objective_snapshot['label']}: {objective_snapshot['description']}. The AI uses this as the default comparison frame unless your prompt asks for another lens.",
        )

        chat_signature = build_ai_chat_signature(metrics, bench_data, scores, tech, objective_snapshot)
        if st.session_state.get("ai_agent_signature") != chat_signature:
            st.session_state["ai_agent_signature"] = chat_signature
            st.session_state["ai_agent_context"] = None
            st.session_state["ai_agent_pending_request"] = None
            st.session_state["ai_agent_processing_request_id"] = None
            st.session_state["ai_agent_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Bonjour. Je suis ton analyste multi-agents pour {ticker_final}. "
                        f"Objectif actif: {objective_snapshot['label']}. "
                        "Tu peux me demander la valorisation, les risques, l'actualite recente, les analystes, les insiders, le short interest, les filings SEC ou une comparaison avec une autre action."
                    ),
                }
            ]

        col_intro, col_reset = st.columns([4, 1])
        with col_intro:
            st.caption("Exemples: « Quels sont les risques principaux ? », « Cette action semble-t-elle chere ? », « Quel profil d'investisseur lui correspond ? »")
            st.caption("Tu peux aussi demander: actualite recente, analystes, achats insiders, short interest, prochaine publication, chiffres SEC officiels ou une comparaison selon un objectif croissance / value / defensif / revenu / court terme.")
        with col_reset:
            if st.button("Reset chat", key=f"reset_ai_chat_{ticker_final}"):
                st.session_state["ai_agent_context"] = None
                st.session_state["ai_agent_pending_request"] = None
                st.session_state["ai_agent_processing_request_id"] = None
                st.session_state["ai_agent_messages"] = [
                    {
                        "role": "assistant",
                        "content": (
                            f"Chat reinitialise pour {ticker_final}. "
                            f"Objectif actif: {objective_snapshot['label']}. "
                            "Tu peux maintenant me poser une question sur l'actualite, les analystes, les insiders, le short interest, les filings SEC, la valorisation ou une comparaison."
                        ),
                    }
                ]
                st.rerun()

        for message in st.session_state.get("ai_agent_messages", []):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    _render_agent_trace(message.get("agent_trace"))
                st.markdown(message["content"])

        pending_request = st.session_state.get("ai_agent_pending_request")
        processing_request_id = st.session_state.get("ai_agent_processing_request_id")
        chat_is_busy = bool(pending_request or processing_request_id)

        user_prompt = st.chat_input(
            f"Pose une question sur {ticker_final}, son actualite, ses analystes ou compare-le a une autre action avec un angle {objective_snapshot['label'].lower()}...",
            disabled=chat_is_busy,
        )
        if user_prompt:
            request_id = str(uuid4())
            normalized_prompt = user_prompt.strip()
            if normalized_prompt:
                if not _chat_message_exists(st.session_state["ai_agent_messages"], request_id, "user"):
                    st.session_state["ai_agent_messages"].append(
                        {"role": "user", "content": normalized_prompt, "request_id": request_id}
                    )
                st.session_state["ai_agent_pending_request"] = {
                    "id": request_id,
                    "prompt": normalized_prompt,
                }
                st.rerun()

        pending_request = st.session_state.get("ai_agent_pending_request")
        if pending_request and not st.session_state.get("ai_agent_processing_request_id"):
            request_id = str(pending_request.get("id") or uuid4())
            prompt_text = str(pending_request.get("prompt") or "").strip()
            if prompt_text:
                st.session_state["ai_agent_processing_request_id"] = request_id

                if not st.session_state.get("ai_agent_context"):
                    chat_context, init_error = create_ai_chat_session(metrics, bench_data, scores, tech, api_key, objective_snapshot)
                    if init_error:
                        if not _chat_message_exists(st.session_state["ai_agent_messages"], request_id, "assistant"):
                            st.session_state["ai_agent_messages"].append(
                                {"role": "assistant", "content": init_error, "request_id": request_id}
                            )
                        st.session_state["ai_agent_pending_request"] = None
                        st.session_state["ai_agent_processing_request_id"] = None
                        st.rerun()
                    st.session_state["ai_agent_context"] = chat_context

                with st.chat_message("assistant"):
                    trace_placeholder = st.empty()
                    with st.spinner("Analyse en cours..."):
                        reply, err, agent_trace = chat_with_ai_analyst(
                            st.session_state["ai_agent_context"],
                            prompt_text,
                            on_trace_update=lambda trace: _render_agent_trace(trace, placeholder=trace_placeholder, live=True),
                        )
                        if reply:
                            _render_agent_trace(agent_trace, placeholder=trace_placeholder)
                            st.markdown(reply)
                            if not _chat_message_exists(st.session_state["ai_agent_messages"], request_id, "assistant"):
                                st.session_state["ai_agent_messages"].append(
                                    {
                                        "role": "assistant",
                                        "content": reply,
                                        "agent_trace": agent_trace,
                                        "request_id": request_id,
                                    }
                                )
                        else:
                            trace_placeholder.empty()
                            st.error(err)
                            if not _chat_message_exists(st.session_state["ai_agent_messages"], request_id, "assistant"):
                                st.session_state["ai_agent_messages"].append(
                                    {
                                        "role": "assistant",
                                        "content": err,
                                        "request_id": request_id,
                                    }
                                )

                st.session_state["ai_agent_pending_request"] = None
                st.session_state["ai_agent_processing_request_id"] = None
                st.rerun()
