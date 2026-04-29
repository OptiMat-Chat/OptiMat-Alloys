"""
ELATE-based directional property visualizations using Plotly.

Creates interactive 2D plots showing how elastic properties
vary with crystal orientation (directional dependence).

Key visualizations:
- 1×3 horizontal layout with Cartesian projections (XY, XZ, YZ planes)
- Property symbols (E, G, ν, K, β, v_s, v_l) for concise labeling
- Black curves with colored extrema (red max, blue min)
- Crystallographic direction labels (plane-specific, e.g., [100], [110])
- Isotropic reference circles showing mean value
- Standardized axis scaling for cross-plane comparison
- Simplified hover tooltips (symbol + angle only)
- Anisotropy summary tables

Phase 2 enhancements (2025-10-12):
- 3D surface plot commented out (preserved for future restoration)
- Professional publication-ready styling
- Enhanced scientific clarity and usability
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mechelastic.core import ELATE
from typing import Dict, List, Literal

from .font_config import get_elate_fonts, DisplayContext


def create_anisotropy_summary_table(properties: Dict) -> str:
    """
    Create markdown table summarizing anisotropy metrics and property comparisons.

    Parameters
    ----------
    properties : dict
        ELATE properties dictionary from compute_elate_properties()

    Returns
    -------
    markdown : str
        Formatted markdown tables with:
        - Voigt/Reuss/Hill comparison
        - Anisotropy measures
        - Directional property ranges
        - Ductility indicators

    Examples
    --------
    >>> from src.core.elate_analysis import compute_elate_properties
    >>> props = compute_elate_properties(C_voigt, density_g_cm3=5.0)
    >>> table_md = create_anisotropy_summary_table(props)
    >>> print(table_md)
    """
    # Helper function for safe division
    def safe_divide(a, b):
        return a / b if b != 0 else 0

    # Calculate anisotropy ratios
    E_ratio = safe_divide(
        properties['max_youngs_modulus_GPa'],
        properties['min_youngs_modulus_GPa']
    )
    G_ratio = safe_divide(
        properties['max_shear_modulus_GPa'],
        properties['min_shear_modulus_GPa']
    )

    # Determine isotropy classification
    AU = properties['universal_anisotropy_index']
    if np.isinf(AU) or np.isnan(AU):
        isotropy_class = "Highly Anisotropic (∞)"
    elif AU < 0.1:
        isotropy_class = "Nearly Isotropic"
    elif AU < 1.0:
        isotropy_class = "Weakly Anisotropic"
    elif AU < 5.0:
        isotropy_class = "Moderately Anisotropic"
    else:
        isotropy_class = "Highly Anisotropic"

    # Determine ductility from Pugh ratio
    pugh = properties['pugh_ratio_hill']
    if pugh > 1.75:
        ductility = "Ductile"
    else:
        ductility = "Brittle"

    # Format infinity values for display
    def format_value(val, decimals=1, unit=""):
        if np.isinf(val):
            return "∞" if val > 0 else "-∞"
        elif np.isnan(val):
            return "N/A"
        else:
            return f"{val:.{decimals}f}{unit}"

    # Build markdown table
    table = f"""
## Elastic Property Comparison: Voigt, Reuss, Hill Averaging

| Property | Voigt (Upper Bound) | Reuss (Lower Bound) | Hill (Average) | Spread |
|----------|--------------------:|--------------------:|---------------:|-------:|
| **Bulk Modulus (K)** | {properties['voigt_bulk_modulus_GPa']:.1f} GPa | {properties['reuss_bulk_modulus_GPa']:.1f} GPa | {properties['hill_bulk_modulus_GPa']:.1f} GPa | {abs(properties['voigt_bulk_modulus_GPa'] - properties['reuss_bulk_modulus_GPa']):.1f} GPa |
| **Shear Modulus (G)** | {properties['voigt_shear_modulus_GPa']:.1f} GPa | {properties['reuss_shear_modulus_GPa']:.1f} GPa | {properties['hill_shear_modulus_GPa']:.1f} GPa | {abs(properties['voigt_shear_modulus_GPa'] - properties['reuss_shear_modulus_GPa']):.1f} GPa |
| **Young's Modulus (E)** | {properties['voigt_youngs_modulus_GPa']:.1f} GPa | {properties['reuss_youngs_modulus_GPa']:.1f} GPa | {properties['hill_youngs_modulus_GPa']:.1f} GPa | {abs(properties['voigt_youngs_modulus_GPa'] - properties['reuss_youngs_modulus_GPa']):.1f} GPa |
| **Poisson's Ratio (ν)** | {properties['voigt_poisson_ratio']:.3f} | {properties['reuss_poisson_ratio']:.3f} | {properties['hill_poisson_ratio']:.3f} | {abs(properties['voigt_poisson_ratio'] - properties['reuss_poisson_ratio']):.3f} |

## Anisotropy Measures

| Metric | Value | Interpretation |
|--------|------:|----------------|
| **Universal Anisotropy (AU)** | {format_value(properties['universal_anisotropy_index'], 3)} | {isotropy_class} (0=isotropic) |
| **Shear Anisotropy** | {format_value(properties['shear_anisotropy'], 3)} | Directional G variation |
| **Young's Anisotropy** | {format_value(properties['youngs_anisotropy'], 3)} | Directional E variation |
| **Poisson Anisotropy** | {format_value(properties['poisson_anisotropy'], 3)} | Directional ν variation |

## Directional Property Ranges

| Property | Minimum | Maximum | Anisotropy Ratio | Notes |
|----------|--------:|--------:|-----------------:|-------|
| **Young's Modulus (E)** | {properties['min_youngs_modulus_GPa']:.1f} GPa | {properties['max_youngs_modulus_GPa']:.1f} GPa | {format_value(E_ratio, 2)}× | Stiffness variation |
| **Poisson's Ratio (ν)** | {properties['min_poisson_ratio']:.3f} | {properties['max_poisson_ratio']:.3f} | - | {'⚠️ **Auxetic behavior!**' if properties['has_auxetic_behavior'] else 'Normal behavior'} |
| **Shear Modulus (G)** | {properties['min_shear_modulus_GPa']:.1f} GPa | {properties['max_shear_modulus_GPa']:.1f} GPa | {format_value(G_ratio, 2)}× | Shear resistance variation |

## Mechanical Behavior Indicators

| Property | Value | Classification |
|----------|------:|----------------|
| **Pugh Ratio (K/G)** | {pugh:.2f} | {ductility} (>1.75 = ductile) |

## Wave Speed Ranges (Acoustic Properties)

| Wave Type | Minimum Speed | Maximum Speed |
|-----------|-------------:|-------------:|
| **Shear Wave** | {properties['min_shear_wave_speed_m_s']:.0f} m/s | {properties['max_shear_wave_speed_m_s']:.0f} m/s |
| **Compression Wave** | {properties['min_compression_wave_speed_m_s']:.0f} m/s | {properties['max_compression_wave_speed_m_s']:.0f} m/s |
"""

    return table


def create_anisotropy_tables_separated(properties: Dict) -> Dict[str, str]:
    """
    Create 5 separate markdown tables for anisotropy analysis.

    This function splits the comprehensive anisotropy summary into 5 focused
    tables for clearer presentation in the UI. Each table covers a specific
    aspect of elastic behavior.

    Parameters
    ----------
    properties : dict
        ELATE properties dictionary from compute_elate_properties()

    Returns
    -------
    tables : dict
        Dictionary with 5 keys:
        - 'voigt_reuss_hill': Elastic property comparison (K, G, E, ν)
        - 'anisotropy_measures': Anisotropy indices (AU, shear, Young's, Poisson)
        - 'directional_ranges': Min/max values with anisotropy ratios
        - 'mechanical_indicators': Ductility (Pugh ratio), behavior classification
        - 'wave_speeds': Acoustic wave speed ranges

    Examples
    --------
    >>> from src.core.elate_analysis import compute_elate_properties
    >>> props = compute_elate_properties(C_voigt, density_g_cm3=5.0)
    >>> tables = create_anisotropy_tables_separated(props)
    >>> print(tables['voigt_reuss_hill'])
    >>> print(tables['anisotropy_measures'])
    """
    import numpy as np

    # Helper function for safe division
    def safe_divide(a, b):
        return a / b if b != 0 else 0

    # Calculate anisotropy ratios
    E_ratio = safe_divide(
        properties['max_youngs_modulus_GPa'],
        properties['min_youngs_modulus_GPa']
    )
    G_ratio = safe_divide(
        properties['max_shear_modulus_GPa'],
        properties['min_shear_modulus_GPa']
    )

    # Determine isotropy classification
    AU = properties['universal_anisotropy_index']
    if np.isinf(AU) or np.isnan(AU):
        isotropy_class = "Highly Anisotropic (∞)"
    elif AU < 0.1:
        isotropy_class = "Nearly Isotropic"
    elif AU < 1.0:
        isotropy_class = "Weakly Anisotropic"
    elif AU < 5.0:
        isotropy_class = "Moderately Anisotropic"
    else:
        isotropy_class = "Highly Anisotropic"

    # Determine ductility from Pugh ratio
    pugh = properties['pugh_ratio_hill']
    if pugh > 1.75:
        ductility = "Ductile"
    else:
        ductility = "Brittle"

    # Format infinity values for display
    def format_value(val, decimals=1, unit=""):
        if np.isinf(val):
            return "∞" if val > 0 else "-∞"
        elif np.isnan(val):
            return "N/A"
        else:
            return f"{val:.{decimals}f}{unit}"

    # Table 1: Voigt/Reuss/Hill Comparison
    voigt_reuss_hill = f"""| Property | Voigt (Upper Bound) | Reuss (Lower Bound) | Hill (Average) | Spread |
|----------|--------------------:|--------------------:|---------------:|-------:|
| **Bulk Modulus (K)** | {properties['voigt_bulk_modulus_GPa']:.1f} GPa | {properties['reuss_bulk_modulus_GPa']:.1f} GPa | {properties['hill_bulk_modulus_GPa']:.1f} GPa | {abs(properties['voigt_bulk_modulus_GPa'] - properties['reuss_bulk_modulus_GPa']):.1f} GPa |
| **Shear Modulus (G)** | {properties['voigt_shear_modulus_GPa']:.1f} GPa | {properties['reuss_shear_modulus_GPa']:.1f} GPa | {properties['hill_shear_modulus_GPa']:.1f} GPa | {abs(properties['voigt_shear_modulus_GPa'] - properties['reuss_shear_modulus_GPa']):.1f} GPa |
| **Young's Modulus (E)** | {properties['voigt_youngs_modulus_GPa']:.1f} GPa | {properties['reuss_youngs_modulus_GPa']:.1f} GPa | {properties['hill_youngs_modulus_GPa']:.1f} GPa | {abs(properties['voigt_youngs_modulus_GPa'] - properties['reuss_youngs_modulus_GPa']):.1f} GPa |
| **Poisson's Ratio (ν)** | {properties['voigt_poisson_ratio']:.3f} | {properties['reuss_poisson_ratio']:.3f} | {properties['hill_poisson_ratio']:.3f} | {abs(properties['voigt_poisson_ratio'] - properties['reuss_poisson_ratio']):.3f} |"""

    # Table 2: Anisotropy Measures
    anisotropy_measures = f"""| Metric | Value | Interpretation |
|--------|------:|----------------|
| **Universal Anisotropy (AU)** | {format_value(properties['universal_anisotropy_index'], 3)} | {isotropy_class} (0=isotropic) |
| **Shear Anisotropy** | {format_value(properties['shear_anisotropy'], 3)} | Directional G variation |
| **Young's Anisotropy** | {format_value(properties['youngs_anisotropy'], 3)} | Directional E variation |
| **Poisson Anisotropy** | {format_value(properties['poisson_anisotropy'], 3)} | Directional ν variation |"""

    # Table 3: Directional Property Ranges
    directional_ranges = f"""| Property | Minimum | Maximum | Anisotropy Ratio | Notes |
|----------|--------:|--------:|-----------------:|-------|
| **Young's Modulus (E)** | {properties['min_youngs_modulus_GPa']:.1f} GPa | {properties['max_youngs_modulus_GPa']:.1f} GPa | {format_value(E_ratio, 2)}× | Stiffness variation |
| **Poisson's Ratio (ν)** | {properties['min_poisson_ratio']:.3f} | {properties['max_poisson_ratio']:.3f} | - | {'⚠️ **Auxetic behavior!**' if properties['has_auxetic_behavior'] else 'Normal behavior'} |
| **Shear Modulus (G)** | {properties['min_shear_modulus_GPa']:.1f} GPa | {properties['max_shear_modulus_GPa']:.1f} GPa | {format_value(G_ratio, 2)}× | Shear resistance variation |"""

    # Table 4: Mechanical Behavior Indicators
    mechanical_indicators = f"""| Property | Value | Classification |
|----------|------:|----------------|
| **Pugh Ratio (K/G)** | {pugh:.2f} | {ductility} (>1.75 = ductile) |"""

    # Table 5: Wave Speed Ranges
    wave_speeds = f"""| Wave Type | Minimum Speed | Maximum Speed |
|-----------|-------------:|-------------:|
| **Shear Wave** | {properties['min_shear_wave_speed_m_s']:.0f} m/s | {properties['max_shear_wave_speed_m_s']:.0f} m/s |
| **Compression Wave** | {properties['min_compression_wave_speed_m_s']:.0f} m/s | {properties['max_compression_wave_speed_m_s']:.0f} m/s |"""

    return {
        'voigt_reuss_hill': voigt_reuss_hill,
        'anisotropy_measures': anisotropy_measures,
        'directional_ranges': directional_ranges,
        'mechanical_indicators': mechanical_indicators,
        'wave_speeds': wave_speeds
    }


def _create_single_projection_figure(
    plane_name: str,
    miller_index: str,
    theta_func,
    phi_func,
    x_label: str,
    y_label: str,
    angle_label: str,
    calc_func,
    angles: np.ndarray,
    angles_deg: np.ndarray,
    symbol: str,
    prop_name: str,
    unit_str: str,
    iso_value: float,
    global_max_val: float,
    context: DisplayContext = "chat"
) -> go.Figure:
    """
    Create a single projection plane figure with all Phase 2 enhancements.

    Parameters
    ----------
    plane_name : str
        Plane name (XY, XZ, or YZ)
    miller_index : str
        Miller index annotation (e.g., "(001)")
    theta_func : callable
        Function mapping angle to theta
    phi_func : callable
        Function mapping angle to phi
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    angle_label : str
        Angle label for hover (θ or φ)
    calc_func : callable
        ELATE property calculation function
    angles : np.ndarray
        Angle array in radians
    angles_deg : np.ndarray
        Angle array in degrees
    symbol : str
        Property symbol (E, G, ν, etc.)
    unit_str : str
        Unit string with space prefix (e.g., " GPa")
    iso_value : float
        Isotropic reference value
    global_max_val : float
        Global maximum for axis scaling

    Returns
    -------
    fig : go.Figure
        Individual projection figure
    """
    fonts = get_elate_fonts(context)

    # Compute property values at each angle
    r_values = []
    for angle in angles:
        theta = theta_func(angle)
        phi = phi_func(angle)
        try:
            value = calc_func(theta, phi)
            r_values.append(value)
        except Exception:
            r_values.append(np.nan)

    r_values = np.array(r_values)

    # Convert polar to Cartesian coordinates
    if plane_name == "XY":
        # XY plane: x = r*cos(phi), y = r*sin(phi)
        x_coords = r_values * np.cos(angles)
        y_coords = r_values * np.sin(angles)
    else:  # XZ and YZ
        # XZ and YZ planes: x = r*sin(theta), z = r*cos(theta)
        x_coords = r_values * np.sin(angles)
        y_coords = r_values * np.cos(angles)

    # Create figure
    fig = go.Figure()

    # Add black curve with fill
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            fill='toself',
            line=dict(color='black', width=2),
            fillcolor='rgba(200, 200, 200, 0.15)',
            customdata=np.column_stack([angles_deg, r_values]),
            hovertemplate=(
                f'{symbol}: %{{customdata[1]:.2f}}{unit_str}<br>'
                f'{angle_label}: %{{customdata[0]:.1f}}°<br>'
                '<extra></extra>'
            ),
            name=f"{plane_name} plane",
            showlegend=False
        )
    )

    # ==========================================
    # EXTREMA MARKERS (red max, blue min)
    # ==========================================
    valid_mask = ~np.isnan(r_values)
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        valid_r = r_values[valid_mask]

        # Find max and min
        max_idx_local = valid_indices[np.argmax(valid_r)]
        min_idx_local = valid_indices[np.argmin(valid_r)]

        # Maximum marker (red)
        fig.add_trace(
            go.Scatter(
                x=[x_coords[max_idx_local]],
                y=[y_coords[max_idx_local]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='circle', line=dict(width=2, color='darkred')),
                name='Maximum',
                showlegend=True,
                hovertemplate=f'Max: {r_values[max_idx_local]:.2f}{unit_str}<extra></extra>'
            )
        )

        # Maximum annotation - REMOVED (hover tooltip already shows value)
        # fig.add_annotation(
        #     x=x_coords[max_idx_local],
        #     y=y_coords[max_idx_local],
        #     text=f"Max: {r_values[max_idx_local]:.1f}{unit_str}",
        #     showarrow=True,
        #     arrowhead=2,
        #     arrowsize=1,
        #     arrowwidth=2,
        #     arrowcolor='red',
        #     ax=30,
        #     ay=-30,
        #     font=dict(size=10, color='red', family='Arial Black')
        # )

        # Minimum marker (blue)
        fig.add_trace(
            go.Scatter(
                x=[x_coords[min_idx_local]],
                y=[y_coords[min_idx_local]],
                mode='markers',
                marker=dict(size=12, color='blue', symbol='circle', line=dict(width=2, color='darkblue')),
                name='Minimum',
                showlegend=True,
                hovertemplate=f'Min: {r_values[min_idx_local]:.2f}{unit_str}<extra></extra>'
            )
        )

        # Minimum annotation - REMOVED (hover tooltip already shows value)
        # fig.add_annotation(
        #     x=x_coords[min_idx_local],
        #     y=y_coords[min_idx_local],
        #     text=f"Min: {r_values[min_idx_local]:.1f}{unit_str}",
        #     showarrow=True,
        #     arrowhead=2,
        #     arrowsize=1,
        #     arrowwidth=2,
        #     arrowcolor='blue',
        #     ax=-30,
        #     ay=30,
        #     font=dict(size=10, color='blue', family='Arial Black')
        # )

    # ==========================================
    # CRYSTALLOGRAPHIC DIRECTION LABELS
    # ==========================================
    if plane_name == "XY":
        directions = [
            (0, "[100]"), (45, "[110]"), (90, "[010]"), (135, "[1̄10]"),
            (180, "[1̄00]"), (225, "[1̄1̄0]"), (270, "[01̄0]"), (315, "[11̄0]")
        ]
    elif plane_name == "XZ":
        directions = [
            (0, "[001]"), (45, "[101]"), (90, "[100]"), (135, "[10̄1]"), (180, "[00̄1]")
        ]
    else:  # YZ
        directions = [
            (0, "[001]"), (45, "[011]"), (90, "[010]"), (135, "[01̄1]"), (180, "[00̄1]")
        ]

    # Add direction labels
    for angle_deg, label in directions:
        angle_rad = np.radians(angle_deg)
        angle_idx = np.argmin(np.abs(angles_deg - angle_deg))
        r_at_angle = r_values[angle_idx]

        if not np.isnan(r_at_angle):
            r_label = r_at_angle * 1.18  # 18% beyond data

            if plane_name == "XY":
                x_pos = r_label * np.cos(angle_rad)
                y_pos = r_label * np.sin(angle_rad)
            else:
                x_pos = r_label * np.sin(angle_rad)
                y_pos = r_label * np.cos(angle_rad)

            fig.add_annotation(
                x=x_pos, y=y_pos,
                text=label,
                showarrow=False,
                font=dict(size=fonts["direction_label"], color='#505050', family='Arial')
            )

    # ==========================================
    # ISOTROPIC REFERENCE CIRCLE
    # ==========================================
    angles_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = iso_value * np.cos(angles_circle)
    y_circle = iso_value * np.sin(angles_circle)

    fig.add_trace(
        go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            line=dict(color='gray', width=1.5, dash='dash'),
            opacity=0.5,
            name='Isotropic (Mean)',
            showlegend=True,
            hovertemplate=f'Isotropic: {iso_value:.2f}{unit_str}<extra></extra>'
        )
    )

    # ==========================================
    # AXES AND LAYOUT
    # ==========================================
    axis_common = dict(
        scaleanchor="x",
        scaleratio=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray',
        range=[-global_max_val, global_max_val],
        showticklabels=True
    )

    fig.update_xaxes(**axis_common, title=dict(text=x_label, font=dict(size=fonts["axis_label"])), tickfont=dict(size=fonts["tick_label"]))
    fig.update_yaxes(**axis_common, title=dict(text=y_label, font=dict(size=fonts["axis_label"])), tickfont=dict(size=fonts["tick_label"]))

    fig.update_layout(
        title=dict(
            text=f"{prop_name} - {plane_name} Plane {miller_index}",
            font=dict(size=fonts["title"]),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=800,
        width=800,
        template="plotly_white",
        margin=dict(t=100, b=80, l=80, r=80),
        hovermode='closest'
    )

    return fig


def plot_directional_property_2d_projections(
    elate: ELATE,
    property_type: Literal["YOUNG", "SHEAR", "POISSON", "BULK", "LC", "SHEAR_SPEED", "COMPRESSION_SPEED"],
    context: DisplayContext = "chat"
) -> List[go.Figure]:
    """
    Create 3 separate projection plane figures for XY, XZ, YZ planes.

    Each plane is rendered as an independent 800×800px figure with all Phase 2 enhancements:
    black curves, extrema markers, crystallographic direction labels, and isotropic reference circles.

    Parameters
    ----------
    elate : ELATE
        Initialized ELATE object with elastic tensor
    property_type : str
        Property to visualize:
        - "YOUNG": Young's modulus (GPa)
        - "SHEAR": Shear modulus (GPa)
        - "POISSON": Poisson's ratio
        - "BULK": Bulk modulus (GPa)
        - "LC": Linear compressibility (1/TPa)
        - "SHEAR_SPEED": Shear wave speed (m/s)
        - "COMPRESSION_SPEED": Compression wave speed (m/s)
    context : DisplayContext, optional
        Display context - "chat" for interactive (default), "pdf" for export

    Returns
    -------
    figures : List[go.Figure]
        List of 3 figures [XY_plane, XZ_plane, YZ_plane]

    Examples
    --------
    >>> from mechelastic.core import ELATE
    >>> import numpy as np
    >>> C = np.eye(6) * 100
    >>> elate = ELATE(s=C.tolist(), density=5000)
    >>> figures = plot_directional_property_2d_projections(elate, "YOUNG")
    >>> for fig in figures:
    ...     fig.show()
    """
    # Property symbols, labels, and units
    property_info = {
        "YOUNG": ("E", "Young's Modulus", "GPa"),
        "SHEAR": ("G", "Shear Modulus", "GPa"),
        "POISSON": ("ν", "Poisson's Ratio", ""),
        "BULK": ("K", "Bulk Modulus", "GPa"),
        "LC": ("β", "Linear Compressibility", "1/TPa"),
        "SHEAR_SPEED": ("v_s", "Shear Wave Speed", "m/s"),
        "COMPRESSION_SPEED": ("v_l", "Compression Wave Speed", "m/s")
    }

    symbol, prop_name, units = property_info.get(property_type, (property_type, property_type, ""))
    unit_str = f" {units}" if units else ""

    # Generate angular data (360 points for smooth curve)
    angles = np.linspace(0, 2*np.pi, 360)
    angles_deg = np.degrees(angles)

    # Property calculation function mapping
    property_funcs = {
        "YOUNG": lambda theta, phi: elate.elas.Young([theta, phi]),
        "SHEAR": lambda theta, phi: elate.elas.shear([theta, phi, 0]),
        "POISSON": lambda theta, phi: elate.elas.Poisson([theta, phi, 0]),
        "BULK": lambda theta, phi: elate.elas.Bulk([theta, phi, 0]),
        "LC": lambda theta, phi: elate.elas.LC([theta, phi]),
        "SHEAR_SPEED": lambda theta, phi: elate.elas.Shear_Speed([theta, phi, 0]),
        "COMPRESSION_SPEED": lambda theta, phi: elate.elas.Compression_Speed([theta, phi, 0])
    }

    calc_func = property_funcs.get(property_type)
    if calc_func is None:
        raise ValueError(f"Unknown property type: {property_type}")

    # Define plane configurations: (name, miller_index, theta_func, phi_func, x_label, y_label, angle_label)
    planes = [
        ("XY", "(001)", lambda a: np.pi/2, lambda a: a, "X", "Y", "φ"),      # XY plane: theta=π/2, phi varies
        ("XZ", "(010)", lambda a: a, lambda a: 0, "X", "Z", "θ"),            # XZ plane: theta varies, phi=0
        ("YZ", "(100)", lambda a: a, lambda a: np.pi/2, "Y", "Z", "θ")      # YZ plane: theta varies, phi=π/2
    ]

    # ==========================================
    # GLOBAL AXIS SCALING (calculate first)
    # ==========================================
    all_r_values = []

    for plane_name, miller_index, theta_func, phi_func, x_label, y_label, angle_label in planes:
        r_values = []
        for angle in angles:
            theta = theta_func(angle)
            phi = phi_func(angle)
            try:
                value = calc_func(theta, phi)
                r_values.append(value)
            except Exception:
                r_values.append(np.nan)
        all_r_values.extend(r_values)

    # Calculate global values
    all_r_values = np.array(all_r_values)
    valid_values = all_r_values[~np.isnan(all_r_values)]
    global_max_val = np.max(np.abs(valid_values)) * 1.15
    iso_value = np.mean(valid_values)

    # ==========================================
    # CREATE 3 SEPARATE FIGURES
    # ==========================================
    figures = []

    for plane_name, miller_index, theta_func, phi_func, x_label, y_label, angle_label in planes:
        fig = _create_single_projection_figure(
            plane_name=plane_name,
            miller_index=miller_index,
            theta_func=theta_func,
            phi_func=phi_func,
            x_label=x_label,
            y_label=y_label,
            angle_label=angle_label,
            calc_func=calc_func,
            angles=angles,
            angles_deg=angles_deg,
            symbol=symbol,
            prop_name=prop_name,
            unit_str=unit_str,
            iso_value=iso_value,
            global_max_val=global_max_val,
            context=context
        )
        figures.append(fig)

    return figures


def plot_directional_property_3d(
    elate: ELATE,
    property_type: Literal["YOUNG", "SHEAR", "POISSON", "BULK", "LC", "SHEAR_SPEED", "COMPRESSION_SPEED"],
    context: DisplayContext = "chat"
) -> go.Figure:
    """
    Create 3D spherical surface plot of directional property.

    Shows how elastic property varies with all spatial orientations.
    Surface radius represents property magnitude in each direction.

    Parameters
    ----------
    elate : ELATE
        Initialized ELATE object
    property_type : str
        Property to visualize:
        - "YOUNG": Young's modulus (GPa)
        - "SHEAR": Shear modulus (GPa)
        - "POISSON": Poisson's ratio
        - "BULK": Bulk modulus (GPa)
        - "LC": Linear compressibility (1/TPa)
        - "SHEAR_SPEED": Shear wave speed (m/s)
        - "COMPRESSION_SPEED": Compression wave speed (m/s)
    context : DisplayContext, optional
        Display context - "chat" for interactive (default), "pdf" for export

    Returns
    -------
    fig : plotly.graph_objects.Figure
        3D surface plot

    Notes
    -----
    For highly anisotropic materials, surface will be distorted.
    For isotropic materials, surface will be nearly spherical.

    Examples
    --------
    >>> fig = plot_directional_property_3d(elate, "POISSON")
    >>> fig.show()
    """
    # Property labels and units
    property_info = {
        "YOUNG": ("Young's Modulus", "GPa"),
        "SHEAR": ("Shear Modulus", "GPa"),
        "POISSON": ("Poisson's Ratio", ""),
        "BULK": ("Bulk Modulus", "GPa"),
        "LC": ("Linear Compressibility", "1/TPa"),
        "SHEAR_SPEED": ("Shear Wave Speed", "m/s"),
        "COMPRESSION_SPEED": ("Compression Wave Speed", "m/s")
    }

    prop_name, units = property_info.get(property_type, (property_type, ""))
    title = f"{prop_name} ({units})" if units else prop_name

    # Property calculation function mapping
    # Note: Some properties require 3 parameters [theta, phi, chi]
    # For 2D projections, chi=0 is used as the reference orientation
    property_funcs = {
        "YOUNG": lambda theta, phi: elate.elas.Young([theta, phi]),
        "SHEAR": lambda theta, phi: elate.elas.shear([theta, phi, 0]),
        "POISSON": lambda theta, phi: elate.elas.Poisson([theta, phi, 0]),
        "BULK": lambda theta, phi: elate.elas.Bulk([theta, phi, 0]),
        "LC": lambda theta, phi: elate.elas.LC([theta, phi]),
        "SHEAR_SPEED": lambda theta, phi: elate.elas.Shear_Speed([theta, phi, 0]),
        "COMPRESSION_SPEED": lambda theta, phi: elate.elas.Compression_Speed([theta, phi, 0])
    }

    calc_func = property_funcs.get(property_type)
    if calc_func is None:
        raise ValueError(f"Unknown property type: {property_type}")

    # Get font sizes for this context
    fonts = get_elate_fonts(context)

    # Generate spherical mesh
    n_theta = 60  # polar angle resolution
    n_phi = 120   # azimuthal angle resolution

    theta = np.linspace(0, np.pi, n_theta)      # 0 to π
    phi = np.linspace(0, 2*np.pi, n_phi)        # 0 to 2π

    THETA, PHI = np.meshgrid(theta, phi)

    # Calculate property values using real ELATE calculations
    R = np.zeros_like(THETA)
    for i in range(THETA.shape[0]):
        for j in range(THETA.shape[1]):
            try:
                R[i, j] = calc_func(THETA[i, j], PHI[i, j])
            except Exception:
                # Handle edge cases where calculation might fail
                R[i, j] = np.nan

    # Convert to Cartesian coordinates
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # Helper function to convert Miller indices to spherical coordinates
    def miller_to_spherical(h, k, l):
        """Convert Miller indices [hkl] to spherical coordinates (theta, phi)."""
        norm = np.sqrt(h**2 + k**2 + l**2)
        if norm == 0:
            return 0, 0
        h_norm, k_norm, l_norm = h/norm, k/norm, l/norm

        # Spherical coordinates: x=sin(θ)cos(φ), y=sin(θ)sin(φ), z=cos(θ)
        theta = np.arccos(np.clip(l_norm, -1, 1))  # Polar angle
        phi = np.arctan2(k_norm, h_norm)            # Azimuthal angle
        return theta, phi

    # Define key crystallographic directions to label
    directions = [
        # Primary crystal axes
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (-1, 0, 0), (0, -1, 0), (0, 0, -1),
        # Face diagonals
        (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
        # Body diagonal
        (1, 1, 1), (-1, -1, -1)
    ]

    # Calculate property values at crystallographic directions
    dir_x, dir_y, dir_z, dir_labels = [], [], [], []
    for h, k, l in directions:
        theta, phi = miller_to_spherical(h, k, l)
        try:
            r_value = calc_func(theta, phi)
            if not np.isnan(r_value) and not np.isinf(r_value):
                # Calculate 3D position on surface
                x_dir = r_value * np.sin(theta) * np.cos(phi)
                y_dir = r_value * np.sin(theta) * np.sin(phi)
                z_dir = r_value * np.cos(theta)

                dir_x.append(x_dir)
                dir_y.append(y_dir)
                dir_z.append(z_dir)

                # Format label: [hkl] with bar notation for negatives
                def format_index(val):
                    if val < 0:
                        return f"<span style='text-decoration:overline'>{abs(val)}</span>"
                    return str(val)

                # Create label with direction and value
                label = f"[{h}{k}{l}]<br>{r_value:.1f}"
                if units:
                    label += f" {units}"
                dir_labels.append(label)
        except Exception:
            continue

    # Create surface plot with semi-transparency (no hover)
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=R,
            colorscale='Turbo',
            opacity=0.7,  # Semi-transparent for better depth perception
            colorbar=dict(
                title=dict(text=title, font=dict(size=fonts["colorbar_title"])),
                tickfont=dict(size=fonts["colorbar_tick"])
            ),
            hoverinfo='skip',  # Disable hover (Plotly bug workaround)
            lighting=dict(
                ambient=0.7,  # Increased for better visibility with transparency
                diffuse=0.9,  # Increased for better surface definition
                specular=0.3,  # Slightly increased for highlights
                roughness=0.4  # Slightly decreased for smoother appearance
            ),
            showscale=True
        )
    ])

    # Add labeled markers at crystallographic directions
    if len(dir_x) > 0:
        fig.add_trace(go.Scatter3d(
            x=dir_x,
            y=dir_y,
            z=dir_z,
            mode='markers+text',
            marker=dict(
                size=6,
                color='#333333',
                symbol='circle',
                line=dict(color='black', width=2)
            ),
            text=dir_labels,
            textposition='top center',
            textfont=dict(
                size=fonts["marker_text"],
                color='black',
                family='Arial Black'
            ),
            hoverinfo='text',
            hovertext=dir_labels,
            name='Crystallographic Directions',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text=f"{title} - 3D Directional Representation",
            font=dict(size=fonts["title"])
        ),
        scene=dict(
            xaxis=dict(
                title="",
                showbackground=False,
                showticklabels=False,
                showgrid=False
            ),
            yaxis=dict(
                title="",
                showbackground=False,
                showticklabels=False,
                showgrid=False
            ),
            zaxis=dict(
                title="",
                showbackground=False,
                showticklabels=False,
                showgrid=False
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        height=700,
        template="plotly_white",
        margin=dict(t=60, b=40, l=40, r=40)
    )

    return fig
