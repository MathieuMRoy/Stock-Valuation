"""
Radar Chart - Visual scorecard representation
"""
import numpy as np
import streamlit as st

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


def plot_radar(scores: dict, tech_score: float = 5.0, dividend_score: float = 5.0):
    """
    Create a radar chart showing scores across different dimensions.
    
    Args:
        scores: Dictionary with valuation, growth, health scores
        tech_score: Technical analysis score
        dividend_score: Dividend score
    
    Returns:
        Matplotlib figure or None if matplotlib is not available
    """
    if not MATPLOTLIB_OK:
        return None
    
    labels = ["VALUE", "GROWTH", "HEALTH", "TECH", "DIV"]
    values = [
        scores.get("valuation", 5),
        scores.get("growth", 5),
        scores.get("health", 5),
        tech_score,
        dividend_score
    ]
    
    # Close the polygon
    values += values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.30)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    return fig
