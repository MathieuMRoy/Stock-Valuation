"""Guardrails for valuation outputs shown in the Streamlit UI."""

from __future__ import annotations

from typing import Any

from .data_quality import safe_float


MAX_REASONABLE_UPSIDE_PCT = 300.0
MAX_REASONABLE_PRICE_MULTIPLE = 5.0


def upside_pct(intrinsic_value: Any, current_price: Any) -> float | None:
    intrinsic = safe_float(intrinsic_value)
    price = safe_float(current_price)
    if intrinsic <= 0 or price <= 0:
        return None
    return (intrinsic / price - 1) * 100


def is_reasonable_intrinsic_value(
    intrinsic_value: Any,
    current_price: Any,
    *,
    max_upside_pct: float = MAX_REASONABLE_UPSIDE_PCT,
) -> bool:
    intrinsic = safe_float(intrinsic_value)
    price = safe_float(current_price)
    if intrinsic <= 0 or price <= 0:
        return False
    if intrinsic > price * MAX_REASONABLE_PRICE_MULTIPLE:
        return False
    gap = upside_pct(intrinsic, price)
    return gap is not None and -95.0 <= gap <= max_upside_pct


def blend_reasonable_intrinsic_values(
    values: dict[str, Any],
    current_price: Any,
    *,
    max_upside_pct: float = MAX_REASONABLE_UPSIDE_PCT,
) -> tuple[float | None, list[str]]:
    """Blend only valuation outputs that pass basic sanity checks."""
    accepted: list[float] = []
    warnings: list[str] = []
    price = safe_float(current_price)

    for label, raw_value in values.items():
        value = safe_float(raw_value)
        if value <= 0:
            warnings.append(f"{label}: unavailable or non-positive.")
            continue
        if not is_reasonable_intrinsic_value(value, price, max_upside_pct=max_upside_pct):
            gap = upside_pct(value, price)
            gap_text = f"{gap:+.1f}%" if gap is not None else "N/A"
            warnings.append(f"{label}: filtered as unrealistic ({value:.2f}, upside {gap_text}).")
            continue
        accepted.append(value)

    if not accepted:
        return None, warnings
    return sum(accepted) / len(accepted), warnings
