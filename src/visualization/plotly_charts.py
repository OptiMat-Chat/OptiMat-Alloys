"""
Plotly-based chart generation for analysis results.

This module creates interactive charts for structural analysis
and radial distribution functions.
"""

from typing import Dict
from ase import Atoms
from plotly.graph_objs import Figure
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from .font_config import get_plotly_fonts, DisplayContext


def plot_structural_analysis(
    atoms: Atoms,
    context: DisplayContext = "chat"
) -> Figure:
    """
    Create bar chart of structural analysis results.

    Uses PTM analysis to show fraction of each crystal structure type.
    Colors match OVITO's PTM color scheme.

    Args:
        atoms: ASE Atoms object to analyze
        context: Display context - "chat" for interactive, "pdf" for export

    Returns:
        Plotly Figure object

    Examples:
        >>> fig = plot_structural_analysis(atoms)
        >>> fig.show()
    """
    from ..core.analysis import structural_analysis

    fonts = get_plotly_fonts(context)
    fractions = structural_analysis(atoms)

    # OVITO PTM color map (approximation)
    ptm_colors = {
        "fcc":     "#00ff00",  # green
        "hcp":     "#ff0000",  # red
        "bcc":     "#0000ff",  # blue
        "ico":     "#ffff00",  # yellow
        "sc":      "#8a2be2",  # purple (simple cubic)
        "diamond": "#00ffff",  # cyan (cubic diamond)
        "other":   "#bfbfbf",  # gray
    }

    order = ["fcc", "hcp", "bcc", "ico", "sc", "diamond", "other"]
    labels = ["FCC", "HCP", "BCC", "ICO", "Simple cubic", "Cubic diamond", "Other"]

    y = [fractions.get(k, 0.0) for k in order]
    colors = [ptm_colors[k] for k in order]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=y,
            marker_color=colors,
            marker_line_color="black",
            marker_line_width=1,
            text=[f"{v:.1f}%" for v in y],
            textposition="outside"
        )
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_title="Fraction (%)",
        xaxis_title="Structure",
        margin=dict(l=60, r=40, t=80, b=60),
        yaxis=dict(range=[0, max(y)*1.15]),  # 15% extra headroom
        font=dict(size=fonts["global"]),
        xaxis=dict(tickfont=dict(size=fonts["tick_label"])),
    )

    return fig


def plot_coordination_rdf(
    atoms: Atoms,
    cutoff: float = 10.0,
    n_bins: int = 200,
    context: DisplayContext = "chat"
) -> Figure:
    """
    Create line plot of radial distribution function.

    Shows total g(r) and partial pair correlations on the same plot.
    The total RDF is displayed with a bold black solid line,
    while partial RDFs use colored thinner solid lines.

    Args:
        atoms: ASE Atoms object to analyze
        cutoff: Maximum distance for RDF (Angstroms)
        n_bins: Number of histogram bins
        context: Display context - "chat" for interactive, "pdf" for export

    Returns:
        Plotly Figure object

    Examples:
        >>> fig = plot_coordination_rdf(atoms, cutoff=10.0)
        >>> fig.show()
    """
    from ..core.analysis import compute_coordination_rdf

    fonts = get_plotly_fonts(context)
    rdf_data = compute_coordination_rdf(atoms, cutoff=cutoff, n_bins=n_bins)

    r = rdf_data["r"]
    g_total = rdf_data["g_total"]
    partial = rdf_data["partial"]

    # Create figure
    fig = go.Figure()

    # Add partial RDFs first (colored solid lines, will appear behind total)
    if partial:
        # Sort partial RDFs alphabetically for consistent ordering
        for pair_name in sorted(partial.keys()):
            fig.add_trace(go.Scatter(
                x=r,
                y=partial[pair_name],
                name=pair_name,
                mode="lines",
                line=dict(width=1.5),
                showlegend=True
            ))

    # Add total RDF last (drawn on top, highlighted style)
    fig.add_trace(go.Scatter(
        x=r,
        y=g_total,
        name="Total",
        mode="lines",
        line=dict(color="black", width=3),
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="r (Å)",
        yaxis_title="g(r)",
        legend_title_text="Pair Type",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="left",
            x=0,
            font=dict(size=fonts["global"])
        ),
        margin=dict(b=160),
        template="plotly_white",
        font=dict(size=fonts["global"]),
        xaxis=dict(
            title=dict(font=dict(size=fonts["axis_label"])),
            tickfont=dict(size=fonts["tick_label"])
        ),
        yaxis=dict(
            title=dict(font=dict(size=fonts["axis_label"])),
            tickfont=dict(size=fonts["tick_label"])
        ),
    )

    return fig


def plot_stiffness_tensor_heatmap(
    C_voigt: np.ndarray,
    context: DisplayContext = "chat"
) -> Figure:
    """
    Create heatmap visualization of elastic stiffness tensor.

    Displays the 6x6 stiffness tensor in Voigt notation with color-coded
    values and annotations. Uses a diverging color scale to highlight
    large vs small components.

    Args:
        C_voigt: Elastic stiffness tensor in Voigt form (6x6 matrix, units: GPa)
        context: Display context - "chat" for interactive, "pdf" for export

    Returns:
        Plotly Figure object

    Examples:
        >>> C = np.random.rand(6, 6) * 200  # Example tensor
        >>> fig = plot_stiffness_tensor_heatmap(C)
        >>> fig.show()
    """
    fonts = get_plotly_fonts(context)

    # Voigt notation labels with subscripts
    labels = ["C₁₁", "C₂₂", "C₃₃", "C₂₃", "C₁₃", "C₁₂"]

    # Create text annotations (formatted values)
    text = [[f"{C_voigt[i, j]:.1f}" for j in range(6)] for i in range(6)]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=C_voigt,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": fonts["heatmap_text"]},
            colorscale="RdBu_r",
            colorbar=dict(
                title=dict(text="GPa", font={"size": fonts["colorbar_title"]}),
                tickfont={"size": fonts["colorbar_tick"]}
            ),
            hovertemplate="<b>%{y} - %{x}</b><br>Stiffness: %{z:.2f} GPa<extra></extra>"
        )
    )

    fig.update_layout(
        title={
            "text": "Elastic Stiffness Tensor (Voigt Form)",
            "font": {"size": fonts["title"]}
        },
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
        width=2260,
        height=700,
        xaxis=dict(
            side="top",
            showgrid=False,
            showline=False,
            ticks="",
            tickfont={"size": fonts["tick_label"]}
        ),
        yaxis=dict(
            autorange="reversed",  # Standard matrix convention (row 0 at top)
            showgrid=False,
            showline=False,
            ticks="",
            tickfont={"size": fonts["tick_label"]}
        ),
        font={"size": fonts["global"]}
    )

    # Rectangular cells: twice as wide as tall
    fig.update_yaxes(scaleanchor="x", scaleratio=0.5)

    return fig
