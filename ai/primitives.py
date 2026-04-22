"""Low-level parsing and formatting helpers for the AI chat layer."""

from __future__ import annotations

from typing import Any


def to_percent(value: Any) -> float | None:
    number = to_float(value)
    if number is None:
        return None
    return number * 100 if abs(number) <= 1 else number


def to_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return to_float(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def format_compact_number(value: Any, suffix: str = "") -> str:
    number = to_float(value)
    if number is None:
        return "N/A"
    abs_value = abs(number)
    if abs_value >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B{suffix}"
    if abs_value >= 1_000_000:
        return f"{number / 1_000_000:.2f} M{suffix}"
    return f"{number:.2f}{suffix}"


def format_pct(value: Any) -> str:
    number = to_percent(value)
    return f"{number:.1f}%" if number is not None else "N/A"


def format_ratio(value: Any) -> str:
    number = to_float(value)
    if number is None or number <= 0:
        return "N/A"
    return f"{number:.2f}x"
