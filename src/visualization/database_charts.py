"""
Database statistics visualization module.

Generates interactive Plotly charts for database growth, composition coverage,
element usage, and calculator distribution.
"""

from typing import List, Dict, Tuple, Any
from datetime import datetime
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure


# All 118 elements of the periodic table (117 ORB-supported, Og unsupported)
# Database schema now tracks all 118 elements for comprehensive coverage
# Note: Heatmap visualizations may be less readable with 118 elements vs original 48
ELEMENTS = [
    "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be", "Bh", "Bi", "Bk", "Br",
    "C", "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Cn", "Co", "Cr", "Cs", "Cu", "Db", "Ds", "Dy",
    "Er", "Es", "Eu", "F", "Fe", "Fl", "Fm", "Fr", "Ga", "Gd", "Ge", "H", "He", "Hf", "Hg",
    "Ho", "Hs", "I", "In", "Ir", "K", "Kr", "La", "Li", "Lr", "Lu", "Lv", "Mc", "Md", "Mg",
    "Mn", "Mo", "Mt", "N", "Na", "Nb", "Nd", "Ne", "Nh", "Ni", "No", "Np", "O", "Og", "Os",
    "P", "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rf", "Rg", "Rh",
    "Rn", "Ru", "S", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te",
    "Th", "Ti", "Tl", "Tm", "Ts", "U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr"
]

# Ember & Stone Palette - 10 clearly distinguishable colors
DARK_RED = '#8B0000'          # Deep burgundy (primary accent)
DARK_SLATE_GRAY = '#2F4F4F'   # Darkest gray (greenish tint)
SADDLE_BROWN = '#8B4513'      # Deep chocolate brown
ORANGE_RED = '#FF4500'        # Vibrant orange
SIENNA = '#A0522D'            # Mid-range brown
SLATE_GRAY = '#708090'        # Cool mid-gray
TOMATO = '#FF6347'            # Bright red-orange
PERU = '#CD853F'              # Golden tan
LIGHT_SALMON = '#FFA07A'      # Light peachy orange
BURLYWOOD = '#DEBB87'         # Beige (lightest)
LIGHT_SLATE_GRAY = '#C0C0C0'  # Light gray (for low values)


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """
    Interpolate between two hex colors.

    Args:
        color1: Starting color in hex format (e.g., '#FF0000')
        color2: Ending color in hex format
        t: Interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated color in hex format
    """
    # Convert hex to RGB
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

    # Interpolate
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)

    # Convert back to hex
    return f'#{r:02X}{g:02X}{b:02X}'


def format_composition_subscript(composition: str) -> str:
    """
    Convert composition string to HTML with subscripted numbers.

    Transforms chemical formulas for better readability in hover text.

    Args:
        composition: Composition string like "Cu50.0Ag50.0"

    Returns:
        HTML-formatted string like "Cu<sub>50.0</sub>Ag<sub>50.0</sub>"

    Example:
        >>> format_composition_subscript("Cu50.0Ag50.0")
        "Cu<sub>50.0</sub>Ag<sub>50.0</sub>"
    """
    import re
    # Pattern: Element symbol (uppercase + optional lowercase) followed by numbers
    pattern = r'([A-Z][a-z]?)(\d+\.?\d*)'
    return re.sub(pattern, r'\1<sub>\2</sub>', composition)


def create_growth_curve(rows: List[Any]) -> Figure:
    """
    Create dual-layer growth visualization with daily bars and cumulative curve.

    Shows database expansion over time with:
    - Daily contribution bars at bottom (hover shows formulas)
    - Cumulative curve overlapping on top (hover shows total count)

    Args:
        rows: List of database rows from ASE database.select()

    Returns:
        Plotly figure with bars and cumulative line on dual y-axes
    """
    from collections import defaultdict

    # Extract timestamps and composition strings together
    daily_structures = defaultdict(list)  # {date: [(timestamp, composition)]}

    for row in rows:
        timestamp_str = row.data.get('timestamp')
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                # Fallback to ctime if timestamp parsing fails
                dt = datetime.fromtimestamp(row.ctime * 86400)  # ctime is in days
        else:
            # Fallback to ctime
            dt = datetime.fromtimestamp(row.ctime * 86400)

        # Get composition string
        composition = row.key_value_pairs.get('composition_string', 'Unknown')

        # Group by date (day level)
        date_key = dt.date()
        daily_structures[date_key].append((dt, composition))

    # Sort dates
    sorted_dates = sorted(daily_structures.keys())

    # Calculate daily counts and formulas
    daily_counts = []
    daily_formulas = []
    for date in sorted_dates:
        structures = daily_structures[date]
        daily_counts.append(len(structures))
        # Sort formulas for consistent display
        formulas = sorted([comp for _, comp in structures])
        daily_formulas.append(formulas)

    # Calculate cumulative counts
    cumulative_counts = [sum(daily_counts[:i+1]) for i in range(len(daily_counts))]

    # Create hover text for bars (shows formula list)
    hover_texts_bars = []
    for date, formulas in zip(sorted_dates, daily_formulas):
        hover_text = f"<b>{date.strftime('%Y-%m-%d')}</b><br>"
        hover_text += f"<b>{len(formulas)} structure{'s' if len(formulas) != 1 else ''} added:</b><br>"

        # Limit to 20 formulas to avoid huge hover boxes
        if len(formulas) <= 20:
            hover_text += "<br>".join([f"• {format_composition_subscript(f)}" for f in formulas])
        else:
            hover_text += "<br>".join([f"• {format_composition_subscript(f)}" for f in formulas[:20]])
            hover_text += f"<br>• ... and {len(formulas) - 20} more"

        hover_texts_bars.append(hover_text)

    # Create hover text for cumulative curve (shows count only)
    hover_texts_cumulative = [
        f"<b>{date.strftime('%Y-%m-%d')}</b><br>"
        f"<b>Total Structures: {count}</b>"
        for date, count in zip(sorted_dates, cumulative_counts)
    ]

    # Define color scheme: Ember & Stone palette (Cool gray for daily, Red for cumulative)
    DAILY_COLOR = SLATE_GRAY  # Cool mid-gray (muted, supporting data)
    DAILY_LIGHT = 'rgba(112, 128, 144, 0.7)'  # Slate gray with transparency
    CUMULATIVE_COLOR = DARK_RED  # Deep burgundy (bold accent, primary metric)
    CUMULATIVE_LIGHT = 'rgba(139, 0, 0, 0.15)'  # Light red for grid

    # Create figure
    fig = go.Figure()

    # Add daily contribution bars (bottom layer) - Slate Gray
    fig.add_trace(go.Bar(
        x=sorted_dates,
        y=daily_counts,
        name='Daily Contributions',
        marker=dict(
            color=DAILY_LIGHT,  # Slate gray with transparency
            line=dict(color=DAILY_COLOR, width=1)  # Solid slate gray border
        ),
        hovertext=hover_texts_bars,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor=DAILY_COLOR,  # Slate gray background
            font=dict(color='white', size=14)  # White text
        ),
        yaxis='y1'
    ))

    # Add cumulative curve (top layer) - Dark Red
    fig.add_trace(go.Scatter(
        x=sorted_dates,
        y=cumulative_counts,
        mode='lines+markers',
        name='Cumulative Total',
        line=dict(color=CUMULATIVE_COLOR, width=3),  # Dark red line
        marker=dict(
            size=8,
            color=CUMULATIVE_COLOR,  # Dark red markers
            line=dict(color='white', width=1)
        ),
        hovertext=hover_texts_cumulative,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor=CUMULATIVE_COLOR,  # Dark red background
            font=dict(color='white', size=14)  # White text
        ),
        yaxis='y2'  # Secondary y-axis
    ))

    # Configure layout with dual y-axes and color-coordinated styling
    fig.update_layout(
        title={
            'text': "Living Database Growth Over Time",
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)'  # Neutral gray grid
        ),
        yaxis=dict(
            title=dict(
                text="Daily Contributions",
                font=dict(color=DAILY_COLOR)  # Slate gray title
            ),
            side='left',
            showgrid=False,
            rangemode='tozero',
            tickfont=dict(color=DAILY_COLOR)  # Slate gray tick labels
        ),
        yaxis2=dict(
            title=dict(
                text="Cumulative Structure Count",
                font=dict(color=CUMULATIVE_COLOR)  # Burgundy title
            ),
            side='right',
            overlaying='y',
            showgrid=True,
            gridcolor=CUMULATIVE_LIGHT,  # Light red grid
            rangemode='tozero',
            tickfont=dict(color=CUMULATIVE_COLOR)  # Burgundy tick labels
        ),
        template="plotly_white",
        font=dict(size=14),
        hovermode='closest',  # Separate hover for bars and curve
        margin=dict(l=80, r=80, t=100, b=80),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_element_usage_chart(rows: List[Any]) -> Figure:
    """
    Create bar chart showing element usage frequency across all structures.

    Uses 5-color gradient from Ember & Stone palette to visually encode frequency:
    - Highest usage: DARK_RED (most prominent)
    - Lowest usage: BURLYWOOD (muted)

    Args:
        rows: List of database rows from ASE database.select()

    Returns:
        Plotly bar chart with gradient coloring by frequency
    """
    # Count element usage (presence threshold: 5%)
    element_counts = defaultdict(int)

    for row in rows:
        for element in ELEMENTS:
            fraction = row.key_value_pairs.get(f'{element}_fraction', 0)
            if fraction > 0.05:  # Only count if >5% present
                element_counts[element] += 1

    if not element_counts:
        # No elements found
        fig = go.Figure()
        fig.add_annotation(
            text="No element data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Sort by frequency (descending)
    sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
    elements = [e for e, _ in sorted_elements]
    counts = [c for _, c in sorted_elements]

    # Create 5-color gradient based on frequency
    # Higher frequency -> darker/more prominent colors
    max_count = max(counts) if counts else 1
    min_count = min(counts) if counts else 0

    # Define 5-segment gradient (Ember & Stone palette)
    gradient_colors = [DARK_RED, SADDLE_BROWN, SIENNA, PERU, BURLYWOOD]

    # Assign colors based on normalized frequency
    bar_colors = []
    for count in counts:
        # Normalize to 0-1 range
        if max_count == min_count:
            normalized = 0.5
        else:
            normalized = (count - min_count) / (max_count - min_count)

        # Map to 5 segments (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        segment_idx = min(int(normalized * 5), 4)  # 0-4 range

        # Interpolate within segment
        segment_start = segment_idx / 5
        segment_end = (segment_idx + 1) / 5
        if segment_end - segment_start > 0:
            t = (normalized - segment_start) / (segment_end - segment_start)
        else:
            t = 0

        # Interpolate between segment colors
        if segment_idx == 4:
            color = gradient_colors[4]
        else:
            color = interpolate_color(gradient_colors[segment_idx], gradient_colors[segment_idx + 1], t)

        bar_colors.append(color)

    # Create bar chart
    fig = go.Figure(go.Bar(
        x=elements,
        y=counts,
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>' +
                      'Used in %{y} structures<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': "Element Usage Frequency",
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        xaxis=dict(
            title="Element",
            showgrid=False,
            tickangle=-45
        ),
        yaxis=dict(
            title="Structure Count",
            showgrid=True
        ),
        template="plotly_white",
        font=dict(size=14),
        height=600,
        margin=dict(l=80, r=40, t=100, b=80),
        showlegend=False
    )

    return fig


def create_calculator_distribution(rows: List[Any]) -> Figure:
    """
    Create pie chart showing calculator usage distribution.

    Args:
        rows: List of database rows from ASE database.select()

    Returns:
        Plotly pie chart showing calculator distribution
    """
    # Count calculator usage
    calculator_counts = defaultdict(int)

    for row in rows:
        calculator = row.key_value_pairs.get('calculator_name', 'Unknown')
        calculator_counts[calculator] += 1

    # Prepare data
    calculators = list(calculator_counts.keys())
    counts = list(calculator_counts.values())

    # Create figure (Midnight Ruby palette)
    fig = go.Figure(data=[go.Pie(
        labels=calculators,
        values=counts,
        hole=0.3,
        marker=dict(
            colors=[DARK_RED, SADDLE_BROWN, ORANGE_RED],  # Ember & Stone: burgundy, brown, orange
            line=dict(color='white', width=2)
        ),
        textinfo='none',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>' +
                      'Structures: %{value}<br>' +
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>'
    )])

    fig.update_layout(
        title={
            'text': "Calculator Usage Distribution",
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        template="plotly_white",
        font=dict(size=14),
        height=600,
        margin=dict(l=40, r=40, t=100, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_structure_type_distribution(rows: List[Any]) -> Figure:
    """
    Create bar chart showing structure type distribution.

    Args:
        rows: List of database rows from ASE database.select()

    Returns:
        Plotly bar chart showing structure type distribution
    """
    # Count structure types
    structure_counts = defaultdict(int)

    for row in rows:
        structure_type = row.key_value_pairs.get('target_structure', 'Unknown')
        structure_counts[structure_type] += 1

    # Prepare data (sorted by count)
    sorted_structures = sorted(structure_counts.items(), key=lambda x: x[1], reverse=True)
    structures = [s for s, _ in sorted_structures]
    counts = [c for _, c in sorted_structures]

    # Assign colors from palette
    colors = [DARK_RED, SADDLE_BROWN, SIENNA, PERU, BURLYWOOD, SLATE_GRAY]
    bar_colors = [colors[i % len(colors)] for i in range(len(structures))]

    # Create bar chart
    fig = go.Figure(go.Bar(
        x=structures,
        y=counts,
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>' +
                      'Structures: %{y}<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': "Structure Type Distribution",
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        xaxis=dict(
            title="Structure Type",
            showgrid=False
        ),
        yaxis=dict(
            title="Structure Count",
            showgrid=True
        ),
        template="plotly_white",
        font=dict(size=14),
        height=600,
        margin=dict(l=80, r=40, t=100, b=80),
        showlegend=False
    )

    return fig


def create_composition_complexity_distribution(rows: List[Any]) -> Figure:
    """
    Create bar chart showing composition complexity distribution.

    Args:
        rows: List of database rows from ASE database.select()

    Returns:
        Plotly bar chart showing unary/binary/ternary/etc distribution
    """
    # Count by number of elements (skip invalid entries, aggregate 10+)
    complexity_counts = defaultdict(int)

    for row in rows:
        num_elements = row.key_value_pairs.get('num_elements', 0)
        if num_elements > 0:  # Skip structures with 0 or missing num_elements
            # Aggregate structures with ≥10 elements into single category
            key = min(num_elements, 10)  # Cap at 10
            complexity_counts[key] += 1

    # Prepare data (sorted by number of elements)
    sorted_complexity = sorted(complexity_counts.items())
    labels = []
    for n, _ in sorted_complexity:
        if n == 1:
            labels.append('Unary')
        elif n == 2:
            labels.append('Binary')
        elif n == 3:
            labels.append('Ternary')
        elif n == 4:
            labels.append('Quaternary')
        elif n == 5:
            labels.append('Quinary')
        elif n == 10:
            labels.append('10+-element')
        else:
            labels.append(f'{n}-element')

    counts = [c for _, c in sorted_complexity]

    # Assign colors from palette
    colors = [DARK_RED, SADDLE_BROWN, SIENNA, PERU, BURLYWOOD, SLATE_GRAY]
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

    # Create bar chart
    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>' +
                      'Structures: %{y}<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': "Composition Complexity Distribution",
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        },
        xaxis=dict(
            title="Complexity",
            showgrid=False
        ),
        yaxis=dict(
            title="Structure Count",
            showgrid=True,
            range=[0, max(counts) * 1.15] if counts else [0, 1]
        ),
        template="plotly_white",
        font=dict(size=14),
        height=600,
        margin=dict(l=80, r=40, t=100, b=80)
    )

    return fig


def create_supercell_size_distribution(rows: List[Any]) -> Figure:
    """
    Create histogram showing supercell size (atom count) distribution.

    Groups structures into bins by number of atoms to show size trends.

    Args:
        rows: List of database rows from ASE database.select()

    Returns:
        Plotly bar chart (histogram style) showing size distribution
    """
    # Extract atom counts from optimized_num_atoms field
    atom_counts = []
    for row in rows:
        count = row.key_value_pairs.get('optimized_num_atoms', 0)
        if count > 0:  # Skip invalid entries
            atom_counts.append(count)

    if not atom_counts:
        # No data
        fig = go.Figure()
        fig.add_annotation(
            text="No supercell size data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Define bins
    bin_edges = [0, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['1-100', '101-200', '201-500', '500-1000', '1000+']

    # Count structures in each bin
    bin_counts = [0] * len(bin_labels)
    for count in atom_counts:
        for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if low < count <= high:
                bin_counts[i] += 1
                break

    # Calculate percentages
    total = sum(bin_counts)
    percentages = [f"{(c/total*100):.1f}%" if total > 0 else "0%" for c in bin_counts]

    # Assign colors from palette (5 bins)
    colors = [DARK_RED, SADDLE_BROWN, SIENNA, PERU, BURLYWOOD, SLATE_GRAY]
    bar_colors = [colors[i % len(colors)] for i in range(len(bin_labels))]

    # Create bar chart (histogram style)
    fig = go.Figure(go.Bar(
        x=bin_labels,
        y=bin_counts,
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{x} atoms</b><br>' +
                      'Structures: %{y}<br>' +
                      'Percentage: %{customdata}<br>' +
                      '<extra></extra>',
        customdata=percentages
    ))

    fig.update_layout(
        title={'text': "Supercell Size Distribution", 'font': {'size': 20}},
        xaxis_title="Number of Atoms",
        yaxis_title="Structure Count",
        template="plotly_white",
        font=dict(size=14),
        height=600,
        margin=dict(l=80, r=40, t=100, b=80),
        showlegend=False
    )

    return fig


def create_property_availability_heatmap(rows: List[Any]) -> Figure:
    """
    Create heatmap showing property availability by structure type.

    Displays which property categories have been computed for each structure type,
    helping identify coverage gaps.

    Property Categories:
    - Thermodynamic (0 K): Formation energy, mixing energy, density (always present)
    - Structural: Local atomic structure distribution, coordination/RDF (from generate_alloy_supercell)
    - Elastic: Elastic tensor, moduli, anisotropy (from calculate_elastic_properties)

    Args:
        rows: List of database rows from ASE database.select()

    Returns:
        Plotly heatmap showing property coverage percentages
    """
    # Define structure types and property categories
    structure_types = ['fcc', 'bcc', 'hcp', 'diamond']
    property_categories = ['Thermodynamic (0 K)', 'Structural', 'Elastic']

    # Property details for hover
    property_details = {
        'Thermodynamic (0 K)': ['Formation energy', 'Mixing energy', 'Potential energy', 'Density'],
        'Structural': ['Local atomic structure distribution', 'Coordination/RDF'],
        'Elastic': ['Elastic tensor', 'Bulk modulus', 'Shear modulus', "Young's modulus", 'Poisson ratio', 'Anisotropy']
    }

    # Initialize coverage matrix (4 structure types × 3 property categories)
    coverage_matrix = [[0.0 for _ in property_categories] for _ in structure_types]
    structure_counts = {st: 0 for st in structure_types}

    # Count structures by type
    for row in rows:
        target = row.key_value_pairs.get('target_structure', '').lower()
        if target in structure_types:
            structure_counts[target] += 1

    # Calculate property availability
    for i, struct_type in enumerate(structure_types):
        if structure_counts[struct_type] == 0:
            continue  # Skip if no structures of this type

        # Filter rows by structure type
        type_rows = [r for r in rows if r.key_value_pairs.get('target_structure', '').lower() == struct_type]

        # Thermodynamic (always present)
        coverage_matrix[i][0] = 100.0

        # Structural (check PTM and stress)
        structural_count = sum(1 for r in type_rows if 'PTM_structural_analysis_in_percent' in r.data)
        coverage_matrix[i][1] = (structural_count / len(type_rows)) * 100

        # Elastic (check elastic tensor)
        elastic_count = sum(1 for r in type_rows if 'elastic_stiffness_tensor_voigt_GPa' in r.data)
        coverage_matrix[i][2] = (elastic_count / len(type_rows)) * 100

    # Create hover text with property details
    hover_text = []
    for i, struct_type in enumerate(structure_types):
        row_hover = []
        for j, prop_cat in enumerate(property_categories):
            details = '<br>'.join(['• ' + p for p in property_details[prop_cat]])
            hover = f"<b>{struct_type.upper()} - {prop_cat}</b><br>" + \
                    f"Coverage: {coverage_matrix[i][j]:.1f}%<br>" + \
                    f"<br><b>Properties:</b><br>{details}"
            row_hover.append(hover)
        hover_text.append(row_hover)

    # Create heatmap
    fig = go.Figure(go.Heatmap(
        z=coverage_matrix,
        x=property_categories,
        y=[s.upper() for s in structure_types],
        colorscale=[
            [0.0, DARK_RED],      # 0%: Dark red
            [0.5, SIENNA],        # 50%: Brown
            [1.0, BURLYWOOD]      # 100%: Light tan
        ],
        text=[[f"{val:.0f}%" for val in row] for row in coverage_matrix],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text,
        colorbar=dict(
            title=dict(
                text="Coverage (%)",
                side="right"
            ),
            tickmode="linear",
            tick0=0,
            dtick=25
        )
    ))

    fig.update_layout(
        title={'text': "Property Availability by Structure Type", 'font': {'size': 20}},
        xaxis_title="Property Category",
        yaxis_title="Structure Type",
        template="plotly_white",
        font=dict(size=14),
        height=600,
        margin=dict(l=80, r=120, t=100, b=80)
    )

    return fig
