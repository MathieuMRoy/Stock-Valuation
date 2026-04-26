"""
Overall Score - Composite scoring for stock evaluation
"""


def score_out_of_10(metrics: dict, bench: dict) -> dict:
    """
    Calculate composite scores (0-10) for various dimensions.
    
    Args:
        metrics: Dictionary with current stock metrics (pe, sales_gr, net_cash, etc.)
        bench: Benchmark data for comparison
    
    Returns:
        Dictionary with overall, health, growth, valuation, and sector scores
    """
    sector_key = str(metrics.get("sector_profile_key") or "default").lower()
    peer_pe = float(bench.get("pe", 20) or 20)
    peer_ps = float(bench.get("ps", 3) or 3)
    pe = float(metrics.get("pe", 0) or 0)
    ps = float(metrics.get("ps", 0) or 0)
    sales_growth = float(metrics.get("sales_gr", 0) or 0)
    eps_growth = float(metrics.get("eps_gr", 0) or 0)
    fcf_yield = float(metrics.get("fcf_yield", 0) or 0)
    rule_40 = float(metrics.get("rule_40", 0) or 0)
    net_cash = float(metrics.get("net_cash", 0) or 0)
    piotroski = metrics.get("piotroski") or 0

    overall = 5.0
    health = 6.0
    growth = 5.5
    valuation = 5.5

    if sector_key == "financial":
        if pe > 0 and pe <= peer_pe:
            valuation += 1.5
            overall += 0.8
        if eps_growth > 0.05:
            growth += 1.0
            overall += 0.5
        if piotroski >= 6:
            health += 1.2
            overall += 0.7
    elif sector_key == "energy":
        if pe > 0 and pe <= peer_pe:
            valuation += 1.0
            overall += 0.6
        if fcf_yield >= 0.08:
            valuation += 1.0
            health += 0.6
            overall += 0.8
        if net_cash > 0:
            health += 0.8
            overall += 0.5
    elif sector_key == "software":
        if sales_growth >= 0.18:
            growth += 1.4
            overall += 0.8
        if rule_40 >= 0.35:
            growth += 0.8
            health += 0.6
            overall += 0.6
        if ps > 0 and ps <= peer_ps * 1.15:
            valuation += 0.9
            overall += 0.4
    else:
        if pe > 0 and pe < peer_pe:
            valuation += 1.0
            overall += 1.0
        if sales_growth > 0.10:
            growth += 1.0
            overall += 1.0
        if net_cash > 0:
            health += 1.0
            overall += 1.0

    return {
        "overall": min(9.5, overall),
        "health": min(9.5, health),
        "growth": min(9.5, growth),
        "valuation": min(9.5, valuation),
        "sector": 7.0 if sector_key != "default" else 5.0
    }
