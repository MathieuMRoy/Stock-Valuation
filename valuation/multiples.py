"""
Multiples Analysis - Relative valuation using P/S and P/E
"""
import streamlit as st


def display_relative_analysis(current: float, benchmark: float, metric_name: str, group_name: str):
    """
    Display relative valuation analysis comparing current vs benchmark multiples.
    
    Args:
        current: Current multiple (P/S or P/E)
        benchmark: Benchmark multiple from peer group
        metric_name: Name of the metric (e.g., "P/S", "P/E")
        group_name: Name of the peer group
    """
    if current <= 0 or benchmark <= 0:
        st.caption("Relative analysis unavailable.")
        return
    
    diff = ((current - benchmark) / benchmark) * 100
    
    if diff < -10:
        st.success(f"**Undervalued 🟢** (discount of {abs(diff):.0f}% vs {group_name})")
    elif diff > 10:
        st.error(f"**Overvalued 🔴** (premium of {diff:.0f}% vs {group_name})")
    else:
        st.warning(f"**Fair Value 🟡** (aligned vs {group_name})")
    
    st.write(f"Current {metric_name} **{current:.1f}x** vs Peer **{benchmark:.1f}x**")
