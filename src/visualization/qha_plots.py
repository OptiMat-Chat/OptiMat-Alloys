#!/usr/bin/env python3
"""
QHA (Quasi-Harmonic Approximation) and thermal conductivity plotting functions.

This module provides Plotly-based visualizations for:
- QHA properties: B(T), V(T), α(T), Cp(T)
- Thermal conductivity: κ(T)

Author: OptiMat Alloys Development Team
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from .font_config import get_qha_fonts, DisplayContext


def plot_qha_properties_individual(
    qha_data: Dict[str, np.ndarray],
    composition: str = "Unknown",
    structure_type: str = "Unknown",
    context: DisplayContext = "chat"
) -> List[go.Figure]:
    """
    Create 4 individual plots for finite temperature properties.

    Creates separate interactive Plotly figures for:
    1. Gibbs free energy G(T)
    2. Bulk modulus B(T)
    3. Linear thermal expansion coefficient α(T)
    4. Isobaric heat capacity Cp(T)

    Args:
        qha_data: Dictionary from compute_qha_properties() containing:
            - temperatures: Temperature array (K)
            - gibbs_free_energy: Gibbs free energy (kJ/mol)
            - bulk_modulus: Bulk modulus (GPa)
            - thermal_expansion: Linear thermal expansion coefficient (1/K)
            - heat_capacity_p: Cp (J/K/mol)
        composition: Chemical composition (e.g., "Cu", "CuAg")
        structure_type: Structure type (e.g., "fcc", "bcc")
        context: Display context - "chat" for interactive (default), "pdf" for export

    Returns:
        List of 4 Plotly Figure objects [G_fig, B_fig, alpha_fig, Cp_fig]
    """
    fonts = get_qha_fonts(context)

    # Extract data
    temperatures = qha_data['temperatures']
    gibbs_free_energy = qha_data['gibbs_free_energy']
    bulk_modulus = qha_data['bulk_modulus']
    thermal_expansion = qha_data['thermal_expansion']
    heat_capacity_p = qha_data['heat_capacity_p']

    # Convert thermal expansion to 10⁻⁶ K⁻¹ for better readability
    thermal_expansion_scaled = thermal_expansion * 1e6

    figures = []

    # Common layout settings
    common_layout = dict(
        plot_bgcolor='white',
        width=1600,
        height=1000,
        font=dict(size=fonts["global"], color='black'),
        margin=dict(l=80, r=40, t=80, b=60)
    )

    # Plot 1: Gibbs Free Energy G(T)
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(
        x=temperatures,
        y=gibbs_free_energy,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='G(T)',
        hovertemplate='T = %{x:.1f} K<br>G = %{y:.3f} kJ/mol<extra></extra>'
    ))
    fig_g.update_layout(
        title=dict(
            text=f"Gibbs Free Energy: {composition} ({structure_type})",
            font=dict(size=fonts["title"], color='black')
        ),
        xaxis=dict(title="Temperature (K)", showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(title="G (kJ/mol)", showgrid=True, gridwidth=1, gridcolor='lightgray'),
        showlegend=False,
        **common_layout
    )
    figures.append(fig_g)

    # Plot 2: Bulk Modulus B(T)
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(
        x=temperatures,
        y=bulk_modulus,
        mode='lines',
        line=dict(color='#ff7f0e', width=2),
        name='B(T)',
        hovertemplate='T = %{x:.1f} K<br>B = %{y:.2f} GPa<extra></extra>'
    ))
    fig_b.update_layout(
        title=dict(
            text=f"Bulk Modulus: {composition} ({structure_type})",
            font=dict(size=fonts["title"], color='black')
        ),
        xaxis=dict(title="Temperature (K)", showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(title="B (GPa)", showgrid=True, gridwidth=1, gridcolor='lightgray'),
        showlegend=False,
        **common_layout
    )
    figures.append(fig_b)

    # Plot 3: Thermal Expansion α(T)
    fig_alpha = go.Figure()
    fig_alpha.add_trace(go.Scatter(
        x=temperatures,
        y=thermal_expansion_scaled,
        mode='lines',
        line=dict(color='#2ca02c', width=2),
        name='α(T)',
        hovertemplate='T = %{x:.1f} K<br>α = %{y:.2f} × 10⁻⁶ K⁻¹<extra></extra>'
    ))
    fig_alpha.update_layout(
        title=dict(
            text=f"Linear Thermal Expansion: {composition} ({structure_type})",
            font=dict(size=fonts["title"], color='black')
        ),
        xaxis=dict(title="Temperature (K)", showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(title="α (10⁻⁶ K⁻¹)", showgrid=True, gridwidth=1, gridcolor='lightgray'),
        showlegend=False,
        **common_layout
    )
    figures.append(fig_alpha)

    # Plot 4: Heat Capacity Cp(T)
    fig_cp = go.Figure()
    fig_cp.add_trace(go.Scatter(
        x=temperatures,
        y=heat_capacity_p,
        mode='lines',
        line=dict(color='#d62728', width=2),
        name='Cp(T)',
        hovertemplate='T = %{x:.1f} K<br>Cp = %{y:.2f} J/(K mol)<extra></extra>'
    ))
    fig_cp.update_layout(
        title=dict(
            text=f"Isobaric Heat Capacity: {composition} ({structure_type})",
            font=dict(size=fonts["title"], color='black')
        ),
        xaxis=dict(title="Temperature (K)", showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(title="Cp (J/(K mol))", showgrid=True, gridwidth=1, gridcolor='lightgray'),
        showlegend=False,
        **common_layout
    )
    figures.append(fig_cp)

    return figures


def plot_qha_properties(
    qha_data: Dict[str, np.ndarray],
    composition: str = "Unknown",
    structure_type: str = "Unknown",
    context: DisplayContext = "chat"
) -> go.Figure:
    """
    Create 4-panel plot for QHA properties vs temperature (legacy, use plot_qha_properties_individual instead).

    Creates interactive Plotly figure with subplots:
    1. Gibbs free energy G(T)
    2. Bulk modulus B(T)
    3. Linear thermal expansion coefficient α(T)
    4. Isobaric heat capacity Cp(T)

    Args:
        qha_data: Dictionary from compute_qha_properties() containing:
            - temperatures: Temperature array (K)
            - gibbs_free_energy: Gibbs free energy (kJ/mol)
            - bulk_modulus: Bulk modulus (GPa)
            - thermal_expansion: Linear thermal expansion coefficient (1/K)
            - heat_capacity_p: Cp (J/K/mol)
        composition: Chemical composition (e.g., "Cu", "CuAg")
        structure_type: Structure type (e.g., "fcc", "bcc")
        context: Display context - "chat" for interactive (default), "pdf" for export

    Returns:
        Plotly Figure object with 4 subplots
    """
    fonts = get_qha_fonts(context)

    # Extract data
    temperatures = qha_data['temperatures']
    gibbs_free_energy = qha_data['gibbs_free_energy']
    bulk_modulus = qha_data['bulk_modulus']
    thermal_expansion = qha_data['thermal_expansion']
    heat_capacity_p = qha_data['heat_capacity_p']

    # Create 4-panel subplot (2×2 grid)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Gibbs Free Energy G(T)",
            "Bulk Modulus B(T)",
            "Linear Thermal Expansion α(T)",
            "Isobaric Heat Capacity C<sub>p</sub>(T)"
        ],
        vertical_spacing=0.20,
        horizontal_spacing=0.15
    )

    # Panel 1: Gibbs free energy G(T)
    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=gibbs_free_energy,
            mode='lines',
            line=dict(color='#1f77b4', width=2),
            name='G(T)',
            hovertemplate='T = %{x:.1f} K<br>G = %{y:.3f} kJ/mol<extra></extra>'
        ),
        row=1, col=1
    )

    # Panel 2: Bulk modulus B(T)
    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=bulk_modulus,
            mode='lines',
            line=dict(color='#ff7f0e', width=2),
            name='B(T)',
            hovertemplate='T = %{x:.1f} K<br>B = %{y:.2f} GPa<extra></extra>'
        ),
        row=1, col=2
    )

    # Panel 3: Linear thermal expansion α(T)
    # Convert to 10⁻⁶ K⁻¹ for better readability
    thermal_expansion_scaled = thermal_expansion * 1e6

    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=thermal_expansion_scaled,
            mode='lines',
            line=dict(color='#2ca02c', width=2),
            name='α(T)',
            hovertemplate='T = %{x:.1f} K<br>α = %{y:.2f} × 10⁻⁶ K⁻¹<extra></extra>'
        ),
        row=2, col=1
    )

    # Panel 4: Heat capacity Cp(T)
    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=heat_capacity_p,
            mode='lines',
            line=dict(color='#d62728', width=2),
            name='Cp(T)',
            hovertemplate='T = %{x:.1f} K<br>Cp = %{y:.2f} J/(K mol)<extra></extra>'
        ),
        row=2, col=2
    )

    # Update axes labels and styling
    # Row 1, Col 1: Gibbs free energy
    fig.update_xaxes(
        title_text="Temperature (K)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="G (kJ/mol)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )

    # Row 1, Col 2: Bulk modulus
    fig.update_xaxes(
        title_text="Temperature (K)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="B (GPa)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=2
    )

    # Row 2, Col 1: Thermal expansion
    fig.update_xaxes(
        title_text="Temperature (K)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="α (10⁻⁶ K⁻¹)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )

    # Row 2, Col 2: Heat capacity
    fig.update_xaxes(
        title_text="Temperature (K)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=2
    )
    fig.update_yaxes(
        title_text="Cp (J/(K mol))",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=2
    )

    # Overall layout
    fig.update_layout(
        title=dict(
            text=f"QHA Properties: {composition} ({structure_type})",
            font=dict(size=fonts["title"], color='black')
        ),
        showlegend=False,
        plot_bgcolor='white',
        width=2800,
        height=2000,
        font=dict(size=fonts["global"], color='black'),
        margin=dict(l=80, r=40, t=100, b=80)
    )

    return fig


def plot_thermal_conductivity(
    kappa_data: Dict[str, np.ndarray],
    composition: str = "Unknown",
    structure_type: str = "Unknown",
    context: DisplayContext = "chat"
) -> go.Figure:
    """
    Create plot for thermal conductivity vs temperature.

    Creates interactive Plotly figure showing:
    - κ_xx, κ_yy, κ_zz (directional components)
    - κ_iso (isotropic average)

    Args:
        kappa_data: Dictionary from compute_thermal_conductivity() containing:
            - temperatures: Temperature array (K)
            - kappa_xx: Thermal conductivity in xx (W/(m K))
            - kappa_yy: Thermal conductivity in yy (W/(m K))
            - kappa_zz: Thermal conductivity in zz (W/(m K))
            - kappa_iso: Isotropic average (W/(m K))
        composition: Chemical composition (e.g., "Cu", "CuAg")
        structure_type: Structure type (e.g., "fcc", "bcc")
        context: Display context - "chat" for interactive (default), "pdf" for export

    Returns:
        Plotly Figure object
    """
    fonts = get_qha_fonts(context)

    # Extract data
    temperatures = kappa_data['temperatures']
    kappa_xx = kappa_data['kappa_xx']
    kappa_yy = kappa_data['kappa_yy']
    kappa_zz = kappa_data['kappa_zz']
    kappa_iso = kappa_data['kappa_iso']

    # Create figure
    fig = go.Figure()

    # Add traces for each direction
    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=kappa_xx,
            mode='lines',
            line=dict(color='#1f77b4', width=2, dash='solid'),
            name='κ<sub>xx</sub>',
            hovertemplate='T = %{x:.1f} K<br>κ<sub>xx</sub> = %{y:.2f} W/(m K)<extra></extra>'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=kappa_yy,
            mode='lines',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            name='κ<sub>yy</sub>',
            hovertemplate='T = %{x:.1f} K<br>κ<sub>yy</sub> = %{y:.2f} W/(m K)<extra></extra>'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=kappa_zz,
            mode='lines',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            name='κ<sub>zz</sub>',
            hovertemplate='T = %{x:.1f} K<br>κ<sub>zz</sub> = %{y:.2f} W/(m K)<extra></extra>'
        )
    )

    # Add isotropic average (thicker line)
    fig.add_trace(
        go.Scatter(
            x=temperatures,
            y=kappa_iso,
            mode='lines',
            line=dict(color='black', width=3, dash='solid'),
            name='κ<sub>iso</sub> (avg)',
            hovertemplate='T = %{x:.1f} K<br>κ<sub>iso</sub> = %{y:.2f} W/(m K)<extra></extra>'
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Thermal Conductivity: {composition} ({structure_type})",
            font=dict(size=fonts["title"], color='black')
        ),
        xaxis=dict(
            title="Temperature (K)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Thermal Conductivity κ (W/(m K))",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        width=2400,
        height=1400,
        font=dict(size=fonts["global"], color='black'),
        legend=dict(
            x=1.0,
            y=1.0,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=fonts["legend"])
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def plot_qha_with_thermal_conductivity(
    qha_data: Dict[str, np.ndarray],
    kappa_data: Dict[str, np.ndarray],
    composition: str = "Unknown",
    structure_type: str = "Unknown",
    context: DisplayContext = "chat"
) -> go.Figure:
    """
    Create combined 5-panel plot for QHA + thermal conductivity.

    Creates interactive Plotly figure with subplots:
    1. Bulk modulus B(T)
    2. Volume V(T)
    3. Linear thermal expansion coefficient α(T)
    4. Isobaric heat capacity Cp(T)
    5. Thermal conductivity κ(T)

    Args:
        qha_data: Dictionary from compute_qha_properties()
        kappa_data: Dictionary from compute_thermal_conductivity()
        composition: Chemical composition
        structure_type: Structure type
        context: Display context - "chat" for interactive (default), "pdf" for export

    Returns:
        Plotly Figure object with 5 subplots
    """
    fonts = get_qha_fonts(context)

    # Extract QHA data
    temperatures = qha_data['temperatures']
    bulk_modulus = qha_data['bulk_modulus']
    volume = qha_data['volume']
    thermal_expansion = qha_data['thermal_expansion'] * 1e6  # Scale to 10⁻⁶ K⁻¹
    heat_capacity_p = qha_data['heat_capacity_p']

    # Extract thermal conductivity
    kappa_iso = kappa_data['kappa_iso']
    temps_kappa = kappa_data['temperatures']

    # Create 5-panel subplot (3 rows, 2 cols, last row spans both columns)
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter", "colspan": 2}, None]
        ],
        subplot_titles=[
            "Bulk Modulus B(T)",
            "Volume V(T)",
            "Linear Thermal Expansion α(T)",
            "Isobaric Heat Capacity C<sub>p</sub>(T)",
            "Thermal Conductivity κ(T)"
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.10
    )

    # Panel 1: Bulk modulus
    fig.add_trace(
        go.Scatter(
            x=temperatures, y=bulk_modulus,
            mode='lines', line=dict(color='#1f77b4', width=2),
            name='B(T)',
            hovertemplate='T = %{x:.1f} K<br>B = %{y:.2f} GPa<extra></extra>'
        ),
        row=1, col=1
    )

    # Panel 2: Volume
    fig.add_trace(
        go.Scatter(
            x=temperatures, y=volume,
            mode='lines', line=dict(color='#ff7f0e', width=2),
            name='V(T)',
            hovertemplate='T = %{x:.1f} K<br>V = %{y:.3f} Å³<extra></extra>'
        ),
        row=1, col=2
    )

    # Panel 3: Thermal expansion
    fig.add_trace(
        go.Scatter(
            x=temperatures, y=thermal_expansion,
            mode='lines', line=dict(color='#2ca02c', width=2),
            name='α(T)',
            hovertemplate='T = %{x:.1f} K<br>α = %{y:.2f} × 10⁻⁶ K⁻¹<extra></extra>'
        ),
        row=2, col=1
    )

    # Panel 4: Heat capacity
    fig.add_trace(
        go.Scatter(
            x=temperatures, y=heat_capacity_p,
            mode='lines', line=dict(color='#d62728', width=2),
            name='Cp(T)',
            hovertemplate='T = %{x:.1f} K<br>Cp = %{y:.2f} J/(K mol)<extra></extra>'
        ),
        row=2, col=2
    )

    # Panel 5: Thermal conductivity
    fig.add_trace(
        go.Scatter(
            x=temps_kappa, y=kappa_iso,
            mode='lines', line=dict(color='#9467bd', width=3),
            name='κ(T)',
            hovertemplate='T = %{x:.1f} K<br>κ = %{y:.2f} W/(m K)<extra></extra>'
        ),
        row=3, col=1
    )

    # Update axes labels
    axes_config = [
        (1, 1, "Temperature (K)", "B (GPa)"),
        (1, 2, "Temperature (K)", "V (Å³)"),
        (2, 1, "Temperature (K)", "α (10⁻⁶ K⁻¹)"),
        (2, 2, "Temperature (K)", "Cp (J/(K mol))"),
        (3, 1, "Temperature (K)", "κ (W/(m K))")
    ]

    for row, col, xlabel, ylabel in axes_config:
        fig.update_xaxes(
            title_text=xlabel,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=row, col=col
        )
        fig.update_yaxes(
            title_text=ylabel,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=row, col=col
        )

    # Overall layout
    fig.update_layout(
        title=dict(
            text=f"QHA + Thermal Conductivity: {composition} ({structure_type})",
            font=dict(size=fonts["title"], color='black')
        ),
        showlegend=False,
        plot_bgcolor='white',
        width=2800,
        height=2800,
        font=dict(size=fonts["global"], color='black'),
        margin=dict(l=60, r=40, t=100, b=60)
    )

    return fig


def plot_qha_volume_surfaces(
    qha_data: Dict[str, np.ndarray],
    composition: str = "Unknown",
    structure_type: str = "Unknown",
    context: DisplayContext = "chat"
) -> List[go.Figure]:
    """
    Create 3D surface plots for F(T,V), S(T,V), Cv(T,V) with equilibrium paths.

    These plots show how phonon thermodynamic properties vary with both
    temperature and volume across the QHA volume grid. Additionally, the
    F(V,T) and Cv(V,T) surfaces include red curves showing the equilibrium
    paths G(T) and Cp(T), respectively, demonstrating how QHA extracts
    temperature-dependent properties by minimizing over volume.

    Args:
        qha_data: Dictionary from compute_qha_properties() containing:
            - temperatures: 1D array (K)
            - volumes_used: 1D array (Å³)
            - volume: 1D array (Å³) - V(T), equilibrium volume at each T
            - gibbs_free_energy: 1D array (kJ/mol) - G(T) at equilibrium
            - heat_capacity_p: 1D array (J/(K mol)) - Cp(T) at equilibrium
            - helmholtz_volume: 2D array (n_temps, n_vols) kJ/mol
            - entropy_volume: 2D array (n_temps, n_vols) J/(K·mol)
            - cv_volume: 2D array (n_temps, n_vols) J/(K·mol)
        composition: Chemical composition (e.g., "Cu", "CuAg")
        structure_type: Structure type (e.g., "fcc", "bcc")
        context: Display context - "chat" for interactive (default), "pdf" for export

    Returns:
        List of 3 Plotly Figure objects [F_fig, S_fig, Cv_fig]
        - F_fig: Helmholtz free energy surface with G(T) equilibrium path (red)
        - S_fig: Entropy surface (no equilibrium path)
        - Cv_fig: Heat capacity surface with Cp(T) equilibrium path (red)
    """
    fonts = get_qha_fonts(context)

    temps = qha_data.get('temperatures_2d', qha_data['temperatures'])
    vols = qha_data['volumes_used']

    # Create meshgrid for surface plot
    # indexing='ij' ensures T varies along rows, V along columns
    T_mesh, V_mesh = np.meshgrid(temps, vols, indexing='ij')

    # Extract equilibrium path data for overlay on surfaces
    temps_eq = qha_data.get('temperatures')
    vols_eq = qha_data.get('volume')  # V(T) - equilibrium volume at each T
    gibbs_eq = qha_data.get('gibbs_free_energy')  # G(T) - Gibbs free energy at equilibrium
    cp_eq = qha_data.get('heat_capacity_p')  # Cp(T) - isobaric heat capacity

    figures = []

    # Property definitions: (key, title, z_label, colorscale)
    properties = [
        ('helmholtz_volume', 'Helmholtz Free Energy F(T,V)', 'F (kJ/mol)', 'Viridis'),
        ('entropy_volume', 'Entropy S(T,V)', 'S (J/K/mol)', 'Plasma'),
        ('cv_volume', 'Heat Capacity Cv(T,V)', 'Cv (J/K/mol)', 'Inferno'),
    ]

    for key, title, z_label, colorscale in properties:
        if key not in qha_data:
            continue

        Z = qha_data[key]  # Shape: (n_temps, n_vols)

        fig = go.Figure(data=[
            go.Surface(
                x=T_mesh,
                y=V_mesh,
                z=Z,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(text=z_label, side='right', font=dict(size=fonts["colorbar_title"])),
                    tickfont=dict(size=fonts["colorbar_tick"]),
                    thickness=20,
                    len=0.7
                ),
                hovertemplate=(
                    'T = %{x:.0f} K<br>'
                    'V = %{y:.2f} Å³<br>'
                    f'{z_label.split()[0]} = %{{z:.2f}}<extra></extra>'
                ),
                name='Surface'
            )
        ])

        # Add equilibrium path overlay based on property type
        if key == 'helmholtz_volume' and gibbs_eq is not None and vols_eq is not None:
            # Add G(T) equilibrium path on F(V,T) surface
            fig.add_trace(go.Scatter3d(
                x=temps_eq,
                y=vols_eq,
                z=gibbs_eq,
                mode='lines+markers',
                name='Equilibrium G(T)',
                line=dict(color='red', width=6),
                marker=dict(size=4, color='red', symbol='circle'),
                hovertemplate=(
                    'T = %{x:.0f} K<br>'
                    'V = %{y:.2f} Å³<br>'
                    'G = %{z:.2f} kJ/mol<extra>Equilibrium Path</extra>'
                ),
                showlegend=True
            ))

        elif key == 'cv_volume' and cp_eq is not None and vols_eq is not None:
            # Add Cp(T) equilibrium path on Cv(V,T) surface
            fig.add_trace(go.Scatter3d(
                x=temps_eq,
                y=vols_eq,
                z=cp_eq,
                mode='lines+markers',
                name='Equilibrium Cp(T)',
                line=dict(color='red', width=6),
                marker=dict(size=4, color='red', symbol='circle'),
                hovertemplate=(
                    'T = %{x:.0f} K<br>'
                    'V = %{y:.2f} Å³<br>'
                    'Cp = %{z:.2f} J/(K mol)<extra>Isobaric Path</extra>'
                ),
                showlegend=True
            ))

        # Add "with Equilibrium Path" to title if equilibrium path was added
        title_suffix = ""
        if (key == 'helmholtz_volume' and gibbs_eq is not None) or \
           (key == 'cv_volume' and cp_eq is not None):
            title_suffix = " with Equilibrium Path"

        fig.update_layout(
            title=dict(
                text=f"{title}{title_suffix}: {composition} ({structure_type})",
                font=dict(size=fonts["surface_title"], color='black')
            ),
            scene=dict(
                xaxis_title='Temperature (K)',
                yaxis_title='Volume (Å³)',
                zaxis_title=z_label,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                xaxis=dict(showgrid=True, gridcolor='lightgray', title=dict(font=dict(size=fonts["surface_axis"])), tickfont=dict(size=fonts["surface_tick"])),
                yaxis=dict(showgrid=True, gridcolor='lightgray', title=dict(font=dict(size=fonts["surface_axis"])), tickfont=dict(size=fonts["surface_tick"])),
                zaxis=dict(showgrid=True, gridcolor='lightgray', title=dict(font=dict(size=fonts["surface_axis"])), tickfont=dict(size=fonts["surface_tick"])),
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=fonts["legend"])
            ),
            width=1600,
            height=1400,
            margin=dict(l=50, r=50, t=80, b=50),
            font=dict(size=fonts["global"], color='black')
        )

        figures.append(fig)

    return figures
