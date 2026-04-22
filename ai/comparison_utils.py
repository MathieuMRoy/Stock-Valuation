"""Comparison-oriented wording helpers shared by the AI chat runtime."""

from __future__ import annotations

from typing import Any

from .primitives import format_compact_number, format_pct, format_ratio, to_float, to_percent
from .router import normalize_intent_text


def clean_business_model_hint(
    hint: str | None,
    sector_name: str | None = None,
    benchmark_name: str | None = None,
) -> str | None:
    raw_hint = (hint or "").strip()
    if raw_hint and raw_hint.lower() not in {"none", "n/a"}:
        cleaned = raw_hint.replace("_", " ").strip()
        return cleaned[:1].lower() + cleaned[1:] if cleaned else None

    combined = " ".join([str(sector_name or ""), str(benchmark_name or "")]).lower()
    if "energy" in combined or "oil" in combined or "gas" in combined:
        return "un producteur d'energie sensible au cycle des matieres premieres"
    if "bank" in combined or "financial" in combined:
        return "un acteur financier sensible a la qualite du capital et du credit"
    if "consumer app" in combined or "platform" in combined or "streaming" in combined:
        return "une plateforme numerique orientee croissance"
    if "saas" in combined or "cloud" in combined or "software" in combined or "technology" in combined:
        return "une entreprise logicielle ou plateforme sensible au rerating de croissance"
    if "pharma" in combined or "biotech" in combined:
        return "une societe sante influencee par son pipeline et la regulation"
    return None


def business_model_sentence(company: dict[str, Any]) -> str:
    ticker = company.get("ticker") or company.get("company_name") or "Ce dossier"
    hint = clean_business_model_hint(
        company.get("business_model_hint"),
        company.get("sector_name"),
        company.get("benchmark_name"),
    )
    if hint:
        return f"{ticker} se lit surtout comme {hint}."
    sector_name = company.get("sector_name") or "son secteur"
    return f"{ticker} reste principalement expose a la dynamique de {sector_name}."


def format_balance_sheet_profile(company: dict[str, Any]) -> str:
    net_cash = to_float(company.get("net_cash"))
    if net_cash is not None:
        if net_cash > 0:
            balance_label = f"net cash {format_compact_number(net_cash, ' $')}"
        elif net_cash < 0:
            balance_label = f"net debt {format_compact_number(abs(net_cash), ' $')}"
        else:
            balance_label = "bilan neutre"
    else:
        balance_label = "bilan non documente"
    piotroski = company.get("piotroski_score", "N/A")
    return f"{balance_label}, Piotroski {piotroski}"


def company_display_name(company: dict[str, Any]) -> str:
    return str(company.get("ticker") or company.get("company_name") or "L'action").strip()


def company_archetype(company: dict[str, Any]) -> str:
    benchmark_name = str(company.get("benchmark_name") or "").lower()
    sector_name = str(company.get("sector_name") or "").lower()
    hint = clean_business_model_hint(
        company.get("business_model_hint"),
        company.get("sector_name"),
        company.get("benchmark_name"),
    )

    if hint:
        return hint
    if "gafam" in benchmark_name or "big tech" in benchmark_name:
        return "un grand acteur technologique deja etabli"
    if "consumer apps" in benchmark_name or "platform" in benchmark_name:
        return "une plateforme numerique orientee croissance"
    if "banks" in benchmark_name or "financial" in sector_name:
        return "un acteur financier regule"
    if "energy" in sector_name:
        return "un dossier plus cyclique lie a l'energie"
    if "technology" in sector_name:
        return "une valeur technologique"
    if "consumer" in sector_name:
        return "une valeur de consommation"
    return f"une entreprise du secteur {company.get('sector_name') or 'analyse'}"


def describe_matchup(current_company: dict[str, Any], other_company: dict[str, Any], is_cross_sector: bool) -> str:
    current_name = company_display_name(current_company)
    other_name = company_display_name(other_company)
    current_archetype = company_archetype(current_company)
    other_archetype = company_archetype(other_company)

    if is_cross_sector:
        return (
            f"{current_name} se lit plutot comme {current_archetype}, alors que {other_name} ressemble davantage a {other_archetype}. "
            "Comme le match-up est intersectoriel, il faut surtout separer la these de croissance de la these defensive."
        )

    return (
        f"{current_name} et {other_name} evoluent dans un univers comparable, "
        f"mais avec deux profils differents: {current_name} reste plutot {current_archetype}, "
        f"tandis que {other_name} se lit davantage comme {other_archetype}."
    )


def winner_name(current_company: dict[str, Any], other_company: dict[str, Any], current_value: Any, other_value: Any) -> str | None:
    current_number = to_float(current_value)
    other_number = to_float(other_value)
    if current_number is None and other_number is None:
        return None
    if current_number is None:
        return company_display_name(other_company)
    if other_number is None:
        return company_display_name(current_company)
    if abs(current_number - other_number) < 0.35:
        return None
    return company_display_name(current_company) if current_number > other_number else company_display_name(other_company)


def describe_growth_tradeoff(current_company: dict[str, Any], other_company: dict[str, Any]) -> str:
    current_name = company_display_name(current_company)
    other_name = company_display_name(other_company)
    current_sales = to_percent(current_company.get("sales_growth_pct"))
    other_sales = to_percent(other_company.get("sales_growth_pct"))
    current_eps = to_percent(current_company.get("eps_growth_pct"))
    other_eps = to_percent(other_company.get("eps_growth_pct"))

    metric_line = (
        f"{current_name}: ventes {format_pct(current_company.get('sales_growth_pct'))}, EPS {format_pct(current_company.get('eps_growth_pct'))}. "
        f"{other_name}: ventes {format_pct(other_company.get('sales_growth_pct'))}, EPS {format_pct(other_company.get('eps_growth_pct'))}."
    )

    sales_leader = None
    eps_leader = None
    if current_sales is not None and other_sales is not None and abs(current_sales - other_sales) >= 5:
        sales_leader = current_name if current_sales > other_sales else other_name
    if current_eps is not None and other_eps is not None and abs(current_eps - other_eps) >= 5:
        eps_leader = current_name if current_eps > other_eps else other_name

    if sales_leader and eps_leader and sales_leader == eps_leader:
        insight = f"Le momentum de croissance penche plutot vers {sales_leader}."
    elif sales_leader and eps_leader and sales_leader != eps_leader:
        insight = (
            f"Le signal est partage: {sales_leader} accelere davantage sur les ventes, "
            f"tandis que {eps_leader} transforme mieux cette croissance en EPS."
        )
    elif sales_leader:
        insight = f"L'avantage principal vient surtout de la croissance du chiffre d'affaires pour {sales_leader}."
    elif eps_leader:
        insight = f"L'avantage principal vient surtout de la croissance de l'EPS pour {eps_leader}."
    else:
        insight = "La dynamique de croissance reste assez proche, sans qu'un nom prenne vraiment le large."

    return f"{metric_line} {insight}"


def describe_valuation_tradeoff(current_company: dict[str, Any], other_company: dict[str, Any]) -> str:
    current_name = company_display_name(current_company)
    other_name = company_display_name(other_company)
    current_trailing_pe = to_float(current_company.get("pe_ratio"))
    other_trailing_pe = to_float(other_company.get("pe_ratio"))
    current_ps = to_float(current_company.get("ps_ratio"))
    other_ps = to_float(other_company.get("ps_ratio"))
    current_forward_pe = to_float(current_company.get("forward_pe_ratio"))
    other_forward_pe = to_float(other_company.get("forward_pe_ratio"))

    metrics_line = [f"Trailing P/E {format_ratio(current_company.get('pe_ratio'))} vs {format_ratio(other_company.get('pe_ratio'))}"]
    if current_forward_pe is not None or other_forward_pe is not None:
        metrics_line.append(f"forward P/E {format_ratio(current_company.get('forward_pe_ratio'))} vs {format_ratio(other_company.get('forward_pe_ratio'))}")
    if current_ps is not None or other_ps is not None:
        metrics_line.append(f"P/S {format_ratio(current_company.get('ps_ratio'))} vs {format_ratio(other_company.get('ps_ratio'))}")

    cheaper_on_pe = None
    cheaper_on_ps = None
    if current_trailing_pe is not None and other_trailing_pe is not None and abs(current_trailing_pe - other_trailing_pe) >= 2:
        cheaper_on_pe = current_name if current_trailing_pe < other_trailing_pe else other_name
    if current_ps is not None and other_ps is not None and abs(current_ps - other_ps) >= 0.6:
        cheaper_on_ps = current_name if current_ps < other_ps else other_name

    if cheaper_on_pe and cheaper_on_ps and cheaper_on_pe == cheaper_on_ps:
        insight = f"Sur les multiples principaux, {cheaper_on_pe} ressort comme l'option la moins chere."
    elif cheaper_on_pe and cheaper_on_ps and cheaper_on_pe != cheaper_on_ps:
        insight = f"La lecture est partagee: {cheaper_on_pe} semble moins cher sur le P/E, tandis que {cheaper_on_ps} parait plus raisonnable sur le P/S."
    elif cheaper_on_pe:
        insight = f"L'avantage de valorisation se voit surtout sur le P/E en faveur de {cheaper_on_pe}."
    elif cheaper_on_ps:
        insight = f"L'avantage de valorisation se voit surtout sur le P/S en faveur de {cheaper_on_ps}."
    else:
        insight = "La valorisation relative ne departage pas clairement les deux dossiers pour l'instant."

    return f"{'; '.join(metrics_line)}. {insight}"


def describe_balance_tradeoff(current_company: dict[str, Any], other_company: dict[str, Any]) -> str:
    current_name = company_display_name(current_company)
    other_name = company_display_name(other_company)
    current_health = to_float(current_company.get("health_score")) or 0.0
    other_health = to_float(other_company.get("health_score")) or 0.0
    current_net_cash = to_float(current_company.get("net_cash"))
    other_net_cash = to_float(other_company.get("net_cash"))

    metrics_line = (
        f"{current_name}: {format_balance_sheet_profile(current_company)}. "
        f"{other_name}: {format_balance_sheet_profile(other_company)}."
    )

    if current_net_cash is not None and other_net_cash is not None:
        if current_net_cash > 0 and other_net_cash < 0:
            insight = f"{current_name} dispose du meilleur coussin de bilan grace a sa position nette de cash."
        elif other_net_cash > 0 and current_net_cash < 0:
            insight = f"{other_name} dispose du meilleur coussin de bilan grace a sa position nette de cash."
        elif abs(current_health - other_health) >= 1.0:
            leader = current_name if current_health > other_health else other_name
            insight = f"La robustesse operationnelle penche plutot vers {leader}."
        else:
            insight = "Les deux bilans paraissent globalement solides, avec davantage de nuances que de rupture nette."
    else:
        if abs(current_health - other_health) >= 1.0:
            leader = current_name if current_health > other_health else other_name
            insight = f"La lecture de bilan reste legerement plus solide pour {leader}."
        else:
            insight = "La lecture de bilan ne montre pas d'ecart decisif."

    return f"{metrics_line} {insight}"


def objective_conclusion(
    objective_label: str,
    current_company: dict[str, Any],
    other_company: dict[str, Any],
    current_technical: dict[str, Any],
    other_technical: dict[str, Any],
) -> str:
    objective_key = (objective_label or "Equilibre").lower()
    current_name = current_company.get("ticker") or current_company.get("company_name") or "L'action actuelle"
    other_name = other_company.get("ticker") or other_company.get("company_name") or "L'autre action"

    if "croissance" in objective_key:
        current_value = current_company.get("growth_score")
        other_value = other_company.get("growth_score")
        leader = current_name if (current_value or 0) >= (other_value or 0) else other_name
        return f"Sous un angle croissance, {leader} ressort mieux."
    if "value" in objective_key:
        current_value = current_company.get("valuation_score")
        other_value = other_company.get("valuation_score")
        leader = current_name if (current_value or 0) >= (other_value or 0) else other_name
        return f"Sous un angle value, {leader} semble offrir le meilleur point d'entree relatif."
    if "defens" in objective_key:
        current_value = current_company.get("health_score")
        other_value = other_company.get("health_score")
        leader = current_name if (current_value or 0) >= (other_value or 0) else other_name
        return f"Sous un angle defensif, {leader} parait le plus robuste."
    if "revenu" in objective_key:
        current_income = (current_company.get("dividend_yield_pct") or 0) + max((current_company.get("fcf_yield_pct") or 0), 0)
        other_income = (other_company.get("dividend_yield_pct") or 0) + max((other_company.get("fcf_yield_pct") or 0), 0)
        leader = current_name if current_income >= other_income else other_name
        return f"Sous un angle revenu, {leader} parait le plus favorable."
    if "court terme" in objective_key:
        leader = current_name if (current_technical.get("technical_score_out_of_10") or 0) >= (other_technical.get("technical_score_out_of_10") or 0) else other_name
        return f"Sous un angle court terme, {leader} parait avoir le meilleur setup."

    growth_leader = winner_name(current_company, other_company, current_company.get("growth_score"), other_company.get("growth_score"))
    value_leader = winner_name(current_company, other_company, current_company.get("valuation_score"), other_company.get("valuation_score"))
    health_leader = winner_name(current_company, other_company, current_company.get("health_score"), other_company.get("health_score"))

    current_score = sum(
        [
            to_float(current_company.get("growth_score")) or 0,
            to_float(current_company.get("valuation_score")) or 0,
            to_float(current_company.get("health_score")) or 0,
        ]
    )
    other_score = sum(
        [
            to_float(other_company.get("growth_score")) or 0,
            to_float(other_company.get("valuation_score")) or 0,
            to_float(other_company.get("health_score")) or 0,
        ]
    )
    if growth_leader and health_leader and growth_leader != health_leader:
        return (
            f"Sous un angle equilibre, {growth_leader} garde l'avantage sur le potentiel de croissance, "
            f"alors que {health_leader} reste le choix le plus defensif."
        )
    if value_leader and health_leader and value_leader != health_leader and not growth_leader:
        return (
            f"Sous un angle equilibre, {value_leader} semble offrir la meilleure lecture de valorisation, "
            f"tandis que {health_leader} garde le bilan le plus rassurant."
        )
    if abs(current_score - other_score) < 1.0:
        return "Sous un angle equilibre, il n'y a pas d'ecart ecrasant; le choix depend surtout de ce que tu privilegies entre potentiel, valorisation et stabilite."
    leader = current_name if current_score > other_score else other_name
    return f"Sous un angle equilibre, {leader} a legerement l'avantage."


def comparison_focus_from_message(user_message: str | None) -> str | None:
    message = normalize_intent_text(user_message or "")
    focus_markers = {
        "croissance": ("croissance", "growth", "upside", "momentum"),
        "valorisation": ("valorisation", "valuation", "pe", "p/e", "ps", "p/s", "cher", "cheap"),
        "solidite": ("risque", "risk", "defensif", "bilan", "solidite", "robuste", "resilience"),
        "technique": ("technique", "technical", "rsi", "trend", "setup", "chart"),
    }
    for focus, markers in focus_markers.items():
        if any(marker in message for marker in markers):
            return focus
    return None


def comparison_needs_technical(user_message: str | None, objective_label: str | None) -> bool:
    objective_key = normalize_intent_text(objective_label or "")
    if "court terme" in objective_key or "short term" in objective_key:
        return True
    return comparison_focus_from_message(user_message) == "technique"
