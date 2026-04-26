"""Technical-analysis helpers for the AI specialist chat."""

from __future__ import annotations

from typing import Any

from .primitives import to_float
from .response_utils import compose_professional_response


TECHNICAL_DETAIL_KEYS = (
    "latest_close",
    "sma20",
    "sma50",
    "sma100",
    "sma200",
    "distance_to_sma20_pct",
    "distance_to_sma50_pct",
    "distance_to_sma200_pct",
    "rsi14",
    "macd",
    "macd_signal",
    "atr_pct",
    "momentum_1m_pct",
    "momentum_3m_pct",
    "momentum_6m_pct",
    "drawdown_from_52w_high_pct",
    "volatility_20d_pct",
    "volatility_60d_pct",
    "volume_vs_20d_pct",
    "support_60d",
    "resistance_60d",
)


def technical_snapshot_is_sparse(snapshot: dict[str, Any] | None) -> bool:
    """Return True when the chat only has a pattern score and no real price context."""
    payload = snapshot or {}
    return not any(to_float(payload.get(key)) is not None for key in TECHNICAL_DETAIL_KEYS)


def technical_score_is_reliable(snapshot: dict[str, Any] | None) -> bool:
    payload = snapshot or {}
    score = to_float(payload.get("technical_score_out_of_10"))
    data_points = to_float(payload.get("data_points")) or 0
    return score is not None and (score > 0 or data_points >= 80)


def _fmt_number(value: Any, decimals: int = 1) -> str:
    number = to_float(value)
    return f"{number:.{decimals}f}" if number is not None else "N/A"


def _fmt_price(value: Any) -> str:
    number = to_float(value)
    return f"{number:.2f}" if number is not None else "N/A"


def _fmt_pct(value: Any, decimals: int = 1, signed: bool = False) -> str:
    number = to_float(value)
    if number is None:
        return "N/A"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{decimals}f}%"


def _above_below(distance_pct: Any) -> str:
    distance = to_float(distance_pct)
    if distance is None:
        return "position inconnue"
    side = "au-dessus" if distance >= 0 else "sous"
    return f"{side} ({_fmt_pct(distance, signed=True)})"


def _rsi_reading(rsi: Any) -> str:
    value = to_float(rsi)
    if value is None:
        return "RSI non disponible"
    if value >= 70:
        label = "zone de surachat"
    elif value <= 30:
        label = "zone de faiblesse/survente"
    elif value >= 55:
        label = "momentum positif"
    elif value <= 45:
        label = "momentum faible"
    else:
        label = "zone neutre"
    return f"RSI {_fmt_number(value, 1)} ({label})"


def _trend_stance(snapshot: dict[str, Any]) -> str:
    score = to_float(snapshot.get("technical_score_out_of_10"))
    dist50 = to_float(snapshot.get("distance_to_sma50_pct"))
    dist200 = to_float(snapshot.get("distance_to_sma200_pct"))
    momentum_1m = to_float(snapshot.get("momentum_1m_pct"))
    rsi = to_float(snapshot.get("rsi14"))

    positives = 0
    negatives = 0
    for distance in (dist50, dist200):
        if distance is None:
            continue
        positives += int(distance > 0)
        negatives += int(distance < 0)
    if momentum_1m is not None:
        positives += int(momentum_1m > 3)
        negatives += int(momentum_1m < -3)
    if rsi is not None:
        positives += int(50 <= rsi <= 70)
        negatives += int(rsi < 40 or rsi > 75)
    if score is not None and score > 0:
        positives += int(score >= 7)
        negatives += int(score < 5)

    if positives >= negatives + 2:
        return "constructif"
    if negatives >= positives + 2:
        return "fragile"
    return "mitige"


def _trend_body(snapshot: dict[str, Any]) -> str:
    close = _fmt_price(snapshot.get("latest_close"))
    sma20 = _fmt_price(snapshot.get("sma20"))
    sma50 = _fmt_price(snapshot.get("sma50"))
    sma200 = _fmt_price(snapshot.get("sma200"))
    return (
        f"Dernier prix analyse {close}. Par rapport aux moyennes: SMA20 {sma20} "
        f"{_above_below(snapshot.get('distance_to_sma20_pct'))}, SMA50 {sma50} "
        f"{_above_below(snapshot.get('distance_to_sma50_pct'))}, SMA200 {sma200} "
        f"{_above_below(snapshot.get('distance_to_sma200_pct'))}."
    )


def _momentum_body(snapshot: dict[str, Any]) -> str:
    pieces = [
        f"1 mois {_fmt_pct(snapshot.get('momentum_1m_pct'), signed=True)}",
        f"3 mois {_fmt_pct(snapshot.get('momentum_3m_pct'), signed=True)}",
        f"6 mois {_fmt_pct(snapshot.get('momentum_6m_pct'), signed=True)}",
        _rsi_reading(snapshot.get("rsi14")),
    ]
    macd_status = snapshot.get("macd_status")
    if macd_status:
        pieces.append(f"MACD {macd_status}")
    if technical_score_is_reliable(snapshot):
        pieces.append(f"score maison {_fmt_number(snapshot.get('technical_score_out_of_10'), 1)}/10")
    else:
        pieces.append("score maison non exploitable pour l'instant")
    return "; ".join(pieces) + "."


def _timing_risk_body(snapshot: dict[str, Any]) -> str:
    bull_flag = "oui" if snapshot.get("bull_flag_detected") else "non"
    drawdown = _fmt_pct(snapshot.get("drawdown_from_52w_high_pct"), signed=True)
    atr = _fmt_pct(snapshot.get("atr_pct"))
    volume = _fmt_pct(snapshot.get("volume_vs_20d_pct"), signed=True)
    volatility = _fmt_pct(snapshot.get("volatility_20d_pct"))
    support = _fmt_price(snapshot.get("support_60d"))
    resistance = _fmt_price(snapshot.get("resistance_60d"))
    return (
        f"Bull flag detecte: {bull_flag}. Distance du plus haut 52 semaines: {drawdown}; "
        f"ATR/price: {atr}; volatilite annualisee 20j: {volatility}; volume vs moyenne 20 jours: {volume}. "
        f"Zone 60j: support {support}, resistance {resistance}."
    )


def _practical_read(snapshot: dict[str, Any]) -> str:
    stance = _trend_stance(snapshot)
    dist50 = to_float(snapshot.get("distance_to_sma50_pct"))
    dist200 = to_float(snapshot.get("distance_to_sma200_pct"))
    if stance == "constructif":
        return (
            "Lecture pratique: le setup peut soutenir une entree progressive, mais le signal doit rester confirme "
            "par le maintien au-dessus des moyennes et un RSI qui ne devient pas trop etire."
        )
    if stance == "fragile":
        watch = "la SMA50" if dist50 is not None and dist50 < 0 else "la SMA200"
        if dist200 is not None and dist200 < 0:
            watch = "la SMA200"
        return (
            f"Lecture pratique: ce n'est pas encore un signal fort; j'attendrais surtout une reprise claire de {watch} "
            "ou un meilleur momentum avant de conclure positivement."
        )
    return (
        "Lecture pratique: le signal est partage. Pour un investisseur, ca sert surtout au timing; "
        "la these principale doit encore venir des fondamentaux, de la valorisation et des catalyseurs."
    )


def compose_technical_response(
    ticker: str,
    snapshot: dict[str, Any] | None,
    *,
    source_refs: dict[str, Any] | None = None,
) -> str:
    """Build a useful technical-agent answer without relying on generic LLM phrasing."""
    payload = snapshot or {}
    clean_ticker = (ticker or "le titre").upper()

    if technical_snapshot_is_sparse(payload) or payload.get("data_quality") == "insufficient":
        return compose_professional_response(
            (
                f"Je n'ai pas assez de donnees techniques fiables pour juger {clean_ticker}; "
                "je prefere ne pas transformer un score vide en signal baissier."
            ),
            [
                (
                    "Ce qui manque",
                    "Il manque un historique de prix exploitable avec RSI, moyennes mobiles, momentum et volume recent.",
                ),
                (
                    "Lecture pratique",
                    "Sans ces donnees, la bonne conclusion est prudente: utiliser la valorisation et les fondamentaux, "
                    "puis revenir au signal technique quand l'historique de prix est disponible.",
                ),
            ],
            source_refs=source_refs,
        )

    stance = _trend_stance(payload)
    opening = f"Techniquement, {clean_ticker} a un setup {stance} selon les donnees de prix disponibles."
    as_of = payload.get("last_price_date")
    if as_of:
        opening += f" Donnees de prix jusqu'au {as_of}."

    return compose_professional_response(
        opening,
        [
            ("Tendance", _trend_body(payload)),
            ("Momentum", _momentum_body(payload)),
            ("Risque de timing", _timing_risk_body(payload)),
            ("Lecture investisseur", _practical_read(payload)),
        ],
        source_refs=source_refs,
    )
