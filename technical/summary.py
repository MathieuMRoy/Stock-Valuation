"""Reusable technical-analysis summary helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _date_label(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return pd.Timestamp(parsed).date().isoformat()
    text = str(value or "").strip()
    return text[:10] if text else None


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _latest(row: pd.Series, column: str) -> float | None:
    if column not in row:
        return None
    return _to_float(row.get(column))


def _pct_change_over_sessions(close_series: pd.Series, sessions: int) -> float | None:
    if close_series is None or close_series.empty or len(close_series) <= sessions:
        return None
    latest = _to_float(close_series.iloc[-1])
    previous = _to_float(close_series.iloc[-sessions - 1])
    if latest is None or previous is None or previous <= 0:
        return None
    return ((latest / previous) - 1) * 100


def _distance_pct(latest_close: float | None, reference: float | None) -> float | None:
    if latest_close is None or reference is None or reference <= 0:
        return None
    return ((latest_close / reference) - 1) * 100


def _volatility_pct(close_series: pd.Series, sessions: int) -> float | None:
    if close_series is None or len(close_series) <= sessions:
        return None
    returns = close_series.pct_change().dropna().iloc[-sessions:]
    if returns.empty:
        return None
    return float(returns.std() * (252 ** 0.5) * 100)


def _trend_label(snapshot: dict[str, Any]) -> str:
    score = snapshot.get("technical_score_out_of_10")
    dist50 = snapshot.get("distance_to_sma50_pct")
    dist200 = snapshot.get("distance_to_sma200_pct")
    momentum_3m = snapshot.get("momentum_3m_pct")

    if score is not None and score >= 7 and (dist50 or 0) > 0 and (dist200 or 0) > 0:
        return "Trend positive"
    if score is not None and score <= 4.5 and ((dist50 or 0) < 0 or (dist200 or 0) < 0):
        return "Trend fragile"
    if momentum_3m is not None and momentum_3m > 8:
        return "Momentum improving"
    return "Mixed setup"


def _timing_risk_label(snapshot: dict[str, Any]) -> str:
    rsi = snapshot.get("rsi14")
    drawdown = snapshot.get("drawdown_from_52w_high_pct")
    volatility = snapshot.get("volatility_20d_pct")

    if rsi is not None and rsi >= 75:
        return "Overbought risk"
    if drawdown is not None and drawdown <= -35:
        return "High drawdown"
    if volatility is not None and volatility >= 60:
        return "High volatility"
    if rsi is not None and 45 <= rsi <= 65:
        return "Balanced timing"
    return "Watch timing"


def _technical_score(snapshot: dict[str, Any], pattern_score: float | None) -> float:
    score = 4.0

    for key in ("distance_to_sma20_pct", "distance_to_sma50_pct", "distance_to_sma200_pct"):
        distance = snapshot.get(key)
        if distance is None:
            continue
        score += 0.8 if distance > 0 else -0.4

    sma50 = snapshot.get("sma50")
    sma200 = snapshot.get("sma200")
    if sma50 is not None and sma200 is not None:
        score += 0.8 if sma50 > sma200 else -0.5

    for key in ("momentum_1m_pct", "momentum_3m_pct", "momentum_6m_pct"):
        momentum = snapshot.get(key)
        if momentum is None:
            continue
        score += 0.5 if momentum > 0 else -0.25

    rsi = snapshot.get("rsi14")
    if rsi is not None:
        if 45 <= rsi <= 65:
            score += 1.0
        elif 35 <= rsi < 45 or 65 < rsi <= 75:
            score += 0.3
        else:
            score -= 0.7

    macd = snapshot.get("macd")
    macd_signal = snapshot.get("macd_signal")
    if macd is not None and macd_signal is not None:
        score += 0.5 if macd >= macd_signal else -0.3

    drawdown = snapshot.get("drawdown_from_52w_high_pct")
    if drawdown is not None:
        if drawdown > -10:
            score += 0.4
        elif drawdown < -30:
            score -= 0.5

    if pattern_score is not None:
        score = (score * 0.75) + (pattern_score * 0.25)

    return round(max(0.0, min(10.0, score)), 1)


def summarize_technical_setup(df: pd.DataFrame, pattern: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize trend, momentum, volatility and support/resistance from an indicator frame."""
    if df is None or df.empty or "Close" not in df.columns:
        return {
            "score": 0.0,
            "technical_score_out_of_10": None,
            "is_bull_flag": False,
            "bull_flag_detected": False,
            "data_quality": "insufficient",
            "notes": "No usable price history.",
        }

    clean_df = df.dropna(subset=["Close"]).copy()
    close_series = _numeric_series(clean_df, "Close")
    if clean_df.empty or close_series.empty:
        return {
            "score": 0.0,
            "technical_score_out_of_10": None,
            "is_bull_flag": False,
            "bull_flag_detected": False,
            "data_quality": "insufficient",
            "notes": "No usable close prices.",
        }

    latest_row = clean_df.iloc[-1]
    latest_close = _to_float(close_series.iloc[-1])
    pattern_payload = pattern or {}
    pattern_score = _to_float(pattern_payload.get("score"))

    snapshot: dict[str, Any] = {
        "data_quality": "rich" if len(close_series) >= 180 else "limited" if len(close_series) >= 60 else "insufficient",
        "data_points": int(len(close_series)),
        "latest_close": latest_close,
        "last_price_date": _date_label(latest_row.get("Date")) if "Date" in latest_row else None,
        "sma20": _latest(latest_row, "SMA20"),
        "sma50": _latest(latest_row, "SMA50"),
        "sma100": _latest(latest_row, "SMA100"),
        "sma200": _latest(latest_row, "SMA200"),
        "rsi14": _latest(latest_row, "RSI14"),
        "macd": _latest(latest_row, "MACD"),
        "macd_signal": _latest(latest_row, "MACDSignal"),
        "atr": _latest(latest_row, "ATR"),
        "momentum_1m_pct": _pct_change_over_sessions(close_series, 21),
        "momentum_3m_pct": _pct_change_over_sessions(close_series, 63),
        "momentum_6m_pct": _pct_change_over_sessions(close_series, 126),
        "high_52w": _to_float(close_series.max()),
        "low_52w": _to_float(close_series.min()),
        "volatility_20d_pct": _volatility_pct(close_series, 20),
        "volatility_60d_pct": _volatility_pct(close_series, 60),
        "support_20d": _to_float(close_series.iloc[-20:].min()) if len(close_series) >= 20 else None,
        "resistance_20d": _to_float(close_series.iloc[-20:].max()) if len(close_series) >= 20 else None,
        "support_60d": _to_float(close_series.iloc[-60:].min()) if len(close_series) >= 60 else None,
        "resistance_60d": _to_float(close_series.iloc[-60:].max()) if len(close_series) >= 60 else None,
        "is_bull_flag": bool(pattern_payload.get("is_bull_flag", False)),
        "bull_flag_detected": bool(pattern_payload.get("is_bull_flag", False)),
        "notes": pattern_payload.get("notes") or "Trend, momentum and volatility summary.",
    }

    for source_key, target_key in (
        ("sma20", "distance_to_sma20_pct"),
        ("sma50", "distance_to_sma50_pct"),
        ("sma200", "distance_to_sma200_pct"),
    ):
        snapshot[target_key] = _distance_pct(latest_close, snapshot.get(source_key))

    high_52w = snapshot.get("high_52w")
    if latest_close is not None and high_52w:
        snapshot["drawdown_from_52w_high_pct"] = _distance_pct(latest_close, high_52w)

    atr = snapshot.get("atr")
    if latest_close is not None and atr is not None and latest_close > 0:
        snapshot["atr_pct"] = (atr / latest_close) * 100

    if snapshot.get("macd") is not None and snapshot.get("macd_signal") is not None:
        snapshot["macd_status"] = "positif" if snapshot["macd"] >= snapshot["macd_signal"] else "negatif"

    volume_series = _numeric_series(clean_df, "Volume")
    if len(volume_series) >= 20:
        avg_20d = volume_series.iloc[-20:].mean()
        latest_volume = _to_float(volume_series.iloc[-1])
        if latest_volume is not None and avg_20d:
            snapshot["volume_vs_20d_pct"] = ((latest_volume / avg_20d) - 1) * 100

    score = _technical_score(snapshot, pattern_score)
    snapshot["score"] = score
    snapshot["technical_score_out_of_10"] = score
    snapshot["trend_label"] = _trend_label(snapshot)
    snapshot["timing_risk_label"] = _timing_risk_label(snapshot)
    snapshot["setup_label"] = (
        "Bullish continuation" if snapshot.get("is_bull_flag") else f"{snapshot['trend_label']} / {snapshot['timing_risk_label']}"
    )
    return snapshot
