"""Font configuration for different display contexts.

This module provides centralized font size configurations for plots
that need to render differently in interactive chat vs PDF export.

Usage:
    from src.visualization.font_config import get_plotly_fonts, DisplayContext

    def plot_something(..., context: DisplayContext = "chat"):
        fonts = get_plotly_fonts(context)
        fig.update_layout(font=dict(size=fonts["global"]))
"""

from typing import Literal, Dict, Any

# Type alias for context parameter
DisplayContext = Literal["chat", "pdf"]


# =============================================================================
# Font Size Configurations
# =============================================================================

PLOTLY_CHARTS_FONTS: Dict[DisplayContext, Dict[str, int]] = {
    "chat": {
        "global": 13,
        "title": 13,
        "axis_label": 13,
        "tick_label": 13,
        "colorbar_title": 13,
        "colorbar_tick": 13,
        "heatmap_text": 13,
    },
    "pdf": {
        "global": 18,
        "title": 20,
        "axis_label": 18,
        "tick_label": 16,
        "colorbar_title": 18,
        "colorbar_tick": 16,
        "heatmap_text": 16,
    },
}

ELATE_FONTS: Dict[DisplayContext, Dict[str, int]] = {
    "chat": {
        "title": 16,
        "axis_label": 14,
        "tick_label": 14,
        "direction_label": 11,
        "colorbar_title": 12,
        "colorbar_tick": 11,
        "marker_text": 11,
    },
    "pdf": {
        "title": 24,
        "axis_label": 24,
        "tick_label": 24,
        "direction_label": 24,
        "colorbar_title": 24,
        "colorbar_tick": 24,
        "marker_text": 18,
    },
}

QHA_FONTS: Dict[DisplayContext, Dict[str, int]] = {
    "chat": {
        "global": 12,
        "title": 16,
        "axis_label": 12,
        "tick_label": 12,
        "colorbar_title": 12,
        "colorbar_tick": 11,
        "legend": 10,
        # 3D surface specific
        "surface_title": 14,
        "surface_axis": 13,
        "surface_tick": 12,
    },
    "pdf": {
        "global": 16,
        "title": 20,
        "axis_label": 16,
        "tick_label": 16,
        "colorbar_title": 20,
        "colorbar_tick": 20,
        "legend": 14,
        # 3D surface specific
        "surface_title": 24,
        "surface_axis": 24,
        "surface_tick": 18,
    },
}

MATPLOTLIB_FONTS: Dict[DisplayContext, Dict[str, Any]] = {
    "chat": {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend_fontsize": 8,
    },
    "pdf": {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend_fontsize": 10,
    },
}


# =============================================================================
# Accessor Functions
# =============================================================================

def get_plotly_fonts(context: DisplayContext = "chat") -> Dict[str, int]:
    """Get font sizes for plotly_charts module.

    Args:
        context: Display context - "chat" for interactive, "pdf" for export

    Returns:
        Dictionary of font sizes for different elements
    """
    return PLOTLY_CHARTS_FONTS[context]


def get_elate_fonts(context: DisplayContext = "chat") -> Dict[str, int]:
    """Get font sizes for elate_plots module.

    Args:
        context: Display context - "chat" for interactive, "pdf" for export

    Returns:
        Dictionary of font sizes for different elements
    """
    return ELATE_FONTS[context]


def get_qha_fonts(context: DisplayContext = "chat") -> Dict[str, int]:
    """Get font sizes for qha_plots module.

    Args:
        context: Display context - "chat" for interactive, "pdf" for export

    Returns:
        Dictionary of font sizes for different elements
    """
    return QHA_FONTS[context]


def get_matplotlib_fonts(context: DisplayContext = "chat") -> Dict[str, Any]:
    """Get font sizes for matplotlib plots in report_generator.

    Args:
        context: Display context - "chat" for interactive, "pdf" for export

    Returns:
        Dictionary of matplotlib rcParams-style font settings
    """
    return MATPLOTLIB_FONTS[context]
