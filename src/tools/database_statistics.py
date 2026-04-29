"""
Database Statistics Tool

Visualizes database growth and distribution statistics.
"""

from typing import Annotated, Dict, List, Optional, Any
from collections import Counter
import chainlit as cl

# Visualization modules
from src.visualization.database_charts import (
    create_growth_curve,
    create_element_usage_chart,
    create_calculator_distribution,
    create_structure_type_distribution,
    create_composition_complexity_distribution,
    create_supercell_size_distribution,
    create_property_availability_heatmap
)

# Storage modules
from src.storage.database import create_structure_database


async def visualize_database_statistics_internal(
    chart_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Internal function to generate database statistics visualizations.

    Called by both the action button callback and the agent tool.

    Args:
        chart_types: List of chart types to generate. Options: 'growth', 'composition',
                    'elements', 'calculators', 'structures', 'complexity', 'all'

    Returns:
        Dictionary with structure count, charts generated, and insights
    """
    # Default to core charts
    if chart_types is None or 'all' in chart_types:
        chart_types = ['growth', 'elements', 'calculators', 'structures', 'complexity', 'supercell_size', 'property_availability']

    # Query database
    db = create_structure_database()
    rows = list(db._get_db().select())

    if len(rows) == 0:
        # No message needed - database visualization already shows "0 structures"
        return {"structure_count": 0, "charts_generated": 0}

    # Generate charts based on requested types
    figures = []

    if 'growth' in chart_types:
        fig = create_growth_curve(rows)
        figures.append(('Growth Over Time', fig))

    # Composition heatmap removed per user request

    if 'elements' in chart_types:
        fig = create_element_usage_chart(rows)
        figures.append(('Element Usage', fig))

    if 'calculators' in chart_types:
        fig = create_calculator_distribution(rows)
        figures.append(('Calculator Distribution', fig))

    if 'structures' in chart_types:
        fig = create_structure_type_distribution(rows)
        figures.append(('Structure Types', fig))

    if 'complexity' in chart_types:
        fig = create_composition_complexity_distribution(rows)
        figures.append(('Composition Complexity', fig))

    if 'supercell_size' in chart_types:
        fig = create_supercell_size_distribution(rows)
        figures.append(('Supercell Size Distribution', fig))

    if 'property_availability' in chart_types:
        fig = create_property_availability_heatmap(rows)
        figures.append(('Property Availability', fig))

    # Create Plotly elements for Chainlit
    elements_list = [
        cl.Plotly(name=name, figure=fig, display="page", size="large")
        for name, fig in figures
    ]

    # Generate summary statistics
    num_elements_dist = {}
    for row in rows:
        n = row.key_value_pairs.get('num_elements', 0)
        num_elements_dist[n] = num_elements_dist.get(n, 0) + 1

    # Most common element (all 118 elements)
    element_counts = Counter()
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
    for row in rows:
        for elem in ELEMENTS:
            if row.key_value_pairs.get(f'{elem}_fraction', 0) > 0.05:
                element_counts[elem] += 1

    top_elements = element_counts.most_common(5)

    # Return elements for caller to display (no message sent from here)
    return {
        "structure_count": len(rows),
        "charts_generated": len(figures),
        "chart_types": [name for name, _ in figures],
        "top_elements": [elem for elem, _ in top_elements],
        "composition_distribution": num_elements_dist,
        "elements": elements_list  # Return Plotly chart elements
    }


@cl.step(type="tool")  # type: ignore
async def visualize_database_statistics(
    chart_types: Annotated[Optional[List[str]], "Chart types (default: all)"] = None
) -> Annotated[Dict[str, Any], "Statistics summary with counts, charts, and distributions."]:
    """
    Generate interactive charts showing database growth, element usage, and structure distributions.
    
    This tool provides comprehensive statistics about the materials database,
    including temporal growth, element usage frequency, calculator preferences,
    and structure type distribution.
    
    Charts Generated:
    - Growth Over Time: Cumulative structure count showing database expansion
    - Element Usage: Frequency of each element across all structures
    - Calculator Distribution: Usage of Direct vs Conservative ORB calculators
    - Structure Types: Distribution of target structures (fcc/bcc/hcp/sc/diamond)
    - Composition Complexity: Unary/Binary/Ternary/Quaternary/Quinary distribution
    
    All charts are interactive Plotly figures (zoom, pan, hover) displayed in the chat.

    Returns:
        Summary dictionary containing:
        - structure_count: Total number of structures in database
        - charts_generated: Number of charts created
        - chart_types: List of chart names generated
        - top_elements: Most frequently used elements
        - composition_distribution: Count of unary/binary/ternary/etc structures
    """
    # Get statistics from internal function
    result = await visualize_database_statistics_internal(chart_types=chart_types)

    # Send message with charts when agent calls this tool
    if result.get('elements'):
        chart_names = " ".join(result['chart_types'])
        await cl.Message(
            content=f"**📊 Database Statistics**\n{chart_names}",
            elements=result['elements']
        ).send()

    # Return metadata for agent (without elements)
    return {k: v for k, v in result.items() if k != 'elements'}


