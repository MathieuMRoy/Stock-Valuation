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
    overall = 5.0
    
    # Valuation check
    if metrics['pe'] > 0 and metrics['pe'] < float(bench.get('pe', 20)):
        overall += 1
    
    # Growth check
    if metrics['sales_gr'] > 0.10:
        overall += 1
    
    # Financial health
    if metrics['net_cash'] > 0:
        overall += 1
    
    return {
        "overall": min(9.5, overall),
        "health": 7.0,
        "growth": 6.0,
        "valuation": 6.0,
        "sector": 5.0
    }
