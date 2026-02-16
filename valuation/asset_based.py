"""
Asset-Based Valuation - NAV and Tangible NAV calculations
"""
import pandas as pd
from fetchers.yahoo_finance import get_item_safe


def compute_asset_based_value(bs: pd.DataFrame, shares: float) -> dict:
    """
    Compute asset-based valuation (NAV and Tangible NAV per share).
    
    Args:
        bs: Balance sheet DataFrame
        shares: Shares outstanding
    
    Returns:
        Dictionary with nav_ps, tnav_ps, and notes
    """
    if bs is None or bs.empty or shares <= 0:
        return {"nav_ps": 0.0, "tnav_ps": 0.0, "notes": "Balance sheet unavailable."}
    
    total_assets = get_item_safe(bs, ["TotalAssets", "Total Assets"])
    total_liab = get_item_safe(bs, ["TotalLiab", "Total Liabilities"])
    goodwill = get_item_safe(bs, ["Goodwill"])
    intangibles = get_item_safe(bs, ["IntangibleAssets"])
    
    equity = total_assets - total_liab
    t_equity = (total_assets - goodwill - intangibles) - total_liab
    
    nav_ps = equity / shares if shares > 0 else 0.0
    tnav_ps = t_equity / shares if shares > 0 else 0.0
    
    notes = f"Assets={total_assets/1e9:.2f}B, Liab={total_liab/1e9:.2f}B"
    
    return {"nav_ps": float(nav_ps), "tnav_ps": float(tnav_ps), "notes": notes}
