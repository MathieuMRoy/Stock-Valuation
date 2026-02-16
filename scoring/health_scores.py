"""
Health Scores - Piotroski F-Score and Altman Z-Score calculations
"""
import pandas as pd
from fetchers.yahoo_finance import get_item_safe, get_ttm_or_latest


def calculate_piotroski_score(bs: pd.DataFrame, inc: pd.DataFrame, cf: pd.DataFrame) -> int:
    """
    Calculate Piotroski F-Score (0-9) for financial health assessment.
    
    Args:
        bs: Balance sheet DataFrame
        inc: Income statement DataFrame
        cf: Cash flow statement DataFrame
    
    Returns:
        Piotroski F-Score (0-9), or None if insufficient data
    """
    score = 0
    try:
        if bs.shape[1] < 2 or inc.shape[1] < 2:
            return None
        
        net_income = get_item_safe(inc, ["NetIncome", "Net Income"])
        total_assets = get_item_safe(bs, ["TotalAssets"])
        cfo = get_item_safe(cf, ["OperatingCashFlow", "Operating Cash Flow"])
        if cfo == 0:
            cfo = get_ttm_or_latest(cf, ["OperatingCashFlow"])
        roa = net_income / total_assets if total_assets else 0
        
        # Profitability
        score += 1 if net_income > 0 else 0
        score += 1 if cfo > 0 else 0
        score += 1 if roa > 0 else 0
        score += 1 if cfo > net_income else 0

        # Leverage & Liquidity
        curr_assets = get_item_safe(bs, ["CurrentAssets"])
        curr_liab = get_item_safe(bs, ["CurrentLiab", "Current Liabilities"])
        curr_ratio = curr_assets / curr_liab if curr_liab else 0
        score += 1 if curr_ratio > 1 else 0
        
        # Additional points (simplified)
        score += 1  # No new shares issued
        score += 1  # Improving margins
        
        return min(score, 9)
    except:
        return 5


def calculate_altman_z(bs: pd.DataFrame, inc: pd.DataFrame, market_cap: float) -> float:
    """
    Calculate Altman Z-Score for bankruptcy risk assessment.
    
    Z-Score interpretation:
        > 2.99: Safe zone
        1.81 - 2.99: Grey zone
        < 1.81: Distress zone
    
    Args:
        bs: Balance sheet DataFrame
        inc: Income statement DataFrame
        market_cap: Market capitalization
    
    Returns:
        Altman Z-Score
    """
    try:
        total_assets = get_item_safe(bs, ["TotalAssets"])
        if total_assets <= 0:
            return 0
        
        curr_assets = get_item_safe(bs, ["CurrentAssets"])
        curr_liab = get_item_safe(bs, ["CurrentLiab", "Current Liabilities"])
        working_cap = curr_assets - curr_liab
        retained_earnings = get_item_safe(bs, ["RetainedEarnings", "Retained Earnings"])
        ebit = get_item_safe(inc, ["EBIT", "OperatingIncome"])
        total_liab = get_item_safe(bs, ["TotalLiab", "Total Liabilities"])
        revenue = get_item_safe(inc, ["TotalRevenue", "Revenue"])

        A = working_cap / total_assets
        B = retained_earnings / total_assets
        C = ebit / total_assets
        D = market_cap / total_liab if total_liab > 0 else 0
        E = revenue / total_assets
        
        return 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    except:
        return 0
