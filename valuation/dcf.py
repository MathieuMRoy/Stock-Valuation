"""
DCF Valuation - Discounted Cash Flow and reverse DCF calculations
"""


def calculate_valuation(
    gr_sales: float,
    gr_fcf: float,
    gr_eps: float,
    wacc_val: float,
    ps_target: float,
    pe_target: float,
    revenue: float,
    fcf: float,
    eps: float,
    cash: float,
    debt: float,
    shares: float
) -> tuple:
    """
    Calculate intrinsic value using DCF, P/S, and P/E methods.
    
    Args:
        gr_sales: Sales growth rate (decimal)
        gr_fcf: FCF growth rate (decimal)
        gr_eps: EPS growth rate (decimal)
        wacc_val: Weighted average cost of capital (decimal)
        ps_target: Target P/S multiple
        pe_target: Target P/E multiple
        revenue: Trailing twelve months revenue
        fcf: Trailing twelve months free cash flow
        eps: Trailing twelve months EPS
        cash: Cash and equivalents
        debt: Total debt
        shares: Shares outstanding
    
    Returns:
        Tuple of (price_dcf, price_sales, price_earnings)
    """
    current_fcf = float(fcf or 0)
    safe_shares = max(shares, 1.0)
    
    # DCF Calculation
    if current_fcf <= 0 or wacc_val <= 0:
        price_dcf = 0.0
    else:
        fcf_projections = [current_fcf * (1 + gr_fcf) ** (i + 1) for i in range(5)]
        terminal_val = (fcf_projections[-1] * 1.03) / max((wacc_val - 0.03), 1e-6)
        pv_fcf = sum([val / ((1 + wacc_val) ** (i + 1)) for i, val in enumerate(fcf_projections)])
        price_dcf = ((pv_fcf + (terminal_val / ((1 + wacc_val) ** 5))) + cash - debt) / safe_shares

    # P/S Calculation (using WACC for discounting)
    if revenue <= 0:
        price_sales = 0.0
    else:
        future_market_cap = (revenue * ((1 + gr_sales) ** 5)) * ps_target
        discounted_mc = future_market_cap / ((1 + wacc_val) ** 5)
        price_sales = discounted_mc / safe_shares

    # P/E Calculation
    if eps <= 0:
        price_earnings = 0.0
    else:
        eps_future = eps * ((1 + gr_eps) ** 5)
        price_earnings = (eps_future * pe_target) / ((1 + wacc_val) ** 5)

    return float(price_dcf), float(price_sales), float(price_earnings)


def solve_reverse_dcf(
    current_price: float,
    fcf: float,
    wacc: float,
    shares: float,
    cash: float,
    debt: float
) -> float:
    """
    Solve for implied growth rate given current price (reverse DCF).
    
    Args:
        current_price: Current stock price
        fcf: Trailing twelve months free cash flow
        wacc: Weighted average cost of capital (decimal)
        shares: Shares outstanding
        cash: Cash and equivalents
        debt: Total debt
    
    Returns:
        Implied growth rate (decimal)
    """
    if fcf <= 0 or current_price <= 0:
        return 0.0
    
    low, high = -0.50, 1.00
    for _ in range(30):
        mid = (low + high) / 2
        val, _, _ = calculate_valuation(0, mid, 0, wacc, 0, 0, 0, fcf, 0, cash, debt, shares)
        if val > current_price:
            high = mid
        else:
            low = mid
    
    return (low + high) / 2
