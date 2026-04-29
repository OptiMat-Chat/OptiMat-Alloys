"""PDF report generator for OptiMat Alloys.

Generates publication-ready PDF reports including:
- Structure summary and visualization
- Computational methods documentation
- Results tables and figures
- References in publication style
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import tempfile

import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, ListFlowable, ListItem, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .references import (
    COMPUTATIONAL_METHODS, PUBLICATION_REFERENCES,
    get_methods_for_calculation, get_all_references_for_calculation,
    get_publication_references, get_software_versions, format_inline_citations,
    ReferenceTracker
)
from ..visualization.font_config import get_matplotlib_fonts, DisplayContext


# ===== UNICODE FONT REGISTRATION =====
# Register DejaVu Sans for Unicode superscript support in PDF tables
def _register_unicode_fonts():
    """Register Unicode-compliant fonts for PDF generation."""
    # DejaVu Sans paths on Linux (commonly pre-installed)
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    ]

    dejavu_path = None
    dejavu_bold_path = None

    for path in font_paths:
        if os.path.exists(path):
            if 'Bold' in path:
                dejavu_bold_path = path
            else:
                dejavu_path = path

    if dejavu_path:
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', dejavu_path))
            if dejavu_bold_path:
                pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', dejavu_bold_path))
            else:
                pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', dejavu_path))
            return True
        except Exception:
            pass
    return False


# Register fonts at module load
_UNICODE_FONTS_AVAILABLE = _register_unicode_fonts()

# Font names to use (with fallback to Helvetica if DejaVu not available)
_TABLE_FONT = 'DejaVuSans' if _UNICODE_FONTS_AVAILABLE else 'Helvetica'
_TABLE_FONT_BOLD = 'DejaVuSans-Bold' if _UNICODE_FONTS_AVAILABLE else 'Helvetica-Bold'


def generate_qha_plots(
    qha_data: Dict[str, Any],
    output_dir: Path,
    context: DisplayContext = "pdf"
) -> Dict[str, Path]:
    """Generate QHA property plots as PNG files for PDF embedding.

    Creates 4 plots: Gibbs energy, bulk modulus, thermal expansion, heat capacity.

    Args:
        qha_data: Dictionary containing temperature_dependent QHA data.
                  Accepts both nested format {'temperature_dependent': {...}}
                  and flat format {'temperatures': [...], 'gibbs_energies': [...], ...}
        output_dir: Directory to save plot images
        context: Display context - "pdf" for export (default), "chat" for interactive

    Returns:
        Dictionary mapping plot names to file paths
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for QHA plots")
        return {}

    # Handle both nested and flat data structures
    # Nested: {'temperature_dependent': {'temperatures': [...], ...}}
    # Flat: {'temperatures': [...], 'gibbs_energies': [...], ...}
    if 'temperature_dependent' in qha_data:
        temp_data = qha_data['temperature_dependent']
    elif 'temperatures' in qha_data:
        # Data is already in flat format
        temp_data = qha_data
    else:
        logger.warning(f"QHA data structure not recognized. Keys: {list(qha_data.keys())}")
        return {}

    temperatures = np.array(temp_data.get('temperatures', []))

    if len(temperatures) == 0:
        logger.warning(f"No temperature data found in QHA data. temp_data keys: {list(temp_data.keys())}")
        return {}

    # Get all property arrays and find minimum length
    # (Phonopy QHA produces 1 fewer derived property point than temperature points due to central differences)
    gibbs = np.array(temp_data.get('gibbs_energies', []))
    bulk_mod = np.array(temp_data.get('bulk_moduli', []))
    alpha = np.array(temp_data.get('thermal_expansion', []))
    heat_cap = np.array(temp_data.get('heat_capacities', []))

    # Find minimum length to handle off-by-one from Phonopy
    array_lengths = [len(temperatures)]
    for arr in [gibbs, bulk_mod, alpha, heat_cap]:
        if len(arr) > 0:
            array_lengths.append(len(arr))
    min_len = min(array_lengths)

    # Truncate all arrays to minimum length
    temperatures = temperatures[:min_len]
    gibbs = gibbs[:min_len] if len(gibbs) > 0 else gibbs
    bulk_mod = bulk_mod[:min_len] if len(bulk_mod) > 0 else bulk_mod
    alpha = alpha[:min_len] if len(alpha) > 0 else alpha
    heat_cap = heat_cap[:min_len] if len(heat_cap) > 0 else heat_cap

    logger.info(f"Generating QHA plots with {len(temperatures)} temperature points (truncated to match property arrays)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    # Common plot settings - Arial font with context-aware sizes
    from matplotlib.ticker import MaxNLocator
    fonts = get_matplotlib_fonts(context)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': fonts['font.size'],
        'axes.labelsize': fonts['axes.labelsize'],
        'axes.titlesize': fonts['axes.titlesize'],
        'xtick.labelsize': fonts['xtick.labelsize'],
        'ytick.labelsize': fonts['ytick.labelsize'],
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
    })

    # 1. Gibbs Free Energy vs Temperature
    if len(gibbs) > 0:
        fig, ax = plt.subplots()
        ax.plot(temperatures, gibbs, 'b-', linewidth=1.5)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Gibbs Free Energy (kJ/mol)')
        ax.set_title('Gibbs Free Energy vs Temperature')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        fig.tight_layout()
        path = output_dir / 'qha_gibbs.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_paths['gibbs'] = path

    # 2. Bulk Modulus vs Temperature
    if len(bulk_mod) > 0:
        fig, ax = plt.subplots()
        ax.plot(temperatures, bulk_mod, 'r-', linewidth=1.5)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Bulk Modulus (GPa)')
        ax.set_title('Bulk Modulus vs Temperature')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        fig.tight_layout()
        path = output_dir / 'qha_bulk_modulus.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_paths['bulk_modulus'] = path

    # 3. Thermal Expansion vs Temperature
    if len(alpha) > 0:
        fig, ax = plt.subplots()
        # Convert to 10^-5 K^-1 for readability
        ax.plot(temperatures, alpha * 1e5, 'g-', linewidth=1.5)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(r'Thermal Expansion ($\times 10^{-5}$ K$^{-1}$)')
        ax.set_title('Thermal Expansion Coefficient vs Temperature')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        fig.tight_layout()
        path = output_dir / 'qha_thermal_expansion.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_paths['thermal_expansion'] = path

    # 4. Heat Capacity vs Temperature
    if len(heat_cap) > 0:
        fig, ax = plt.subplots()
        ax.plot(temperatures, heat_cap, 'm-', linewidth=1.5)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Heat Capacity Cp (J/mol/K)')
        ax.set_title('Isobaric Heat Capacity vs Temperature')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        fig.tight_layout()
        path = output_dir / 'qha_heat_capacity.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_paths['heat_capacity'] = path

    return plot_paths


def generate_rdf_plot(
    rdf_data: Dict[str, Any],
    output_dir: Path,
    context: DisplayContext = "pdf"
) -> Optional[Path]:
    """Generate RDF plot as PNG file for PDF embedding.

    Args:
        rdf_data: Dictionary containing 'r_values', 'g_r_values' (total),
                 and optionally 'partial' dict with pair-specific RDFs
        context: Display context - "pdf" for export (default), "chat" for interactive

    Returns:
        Path to the generated plot, or None if generation fails
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for RDF plot")
        return None

    r = np.array(rdf_data.get('r_values', []))
    g_r = np.array(rdf_data.get('g_r_values', []))
    partial = rdf_data.get('partial', {})

    if len(r) == 0 or len(g_r) == 0:
        logger.warning(f"RDF data empty. Keys: {list(rdf_data.keys())}")
        return None

    if len(r) != len(g_r):
        logger.warning(f"RDF data length mismatch: r={len(r)}, g(r)={len(g_r)}")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RDF plot settings - Arial font with context-aware sizes
    from matplotlib.ticker import MaxNLocator
    fonts = get_matplotlib_fonts(context)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': fonts['font.size'],
        'axes.labelsize': fonts['axes.labelsize'],
        'axes.titlesize': fonts['axes.titlesize'],
        'xtick.labelsize': fonts['xtick.labelsize'],
        'ytick.labelsize': fonts['ytick.labelsize'],
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots()

    # Plot partial RDFs first (thinner, colored lines)
    if partial:
        colors = plt.cm.tab10.colors
        for i, (pair_name, g_partial) in enumerate(sorted(partial.items())):
            g_partial_arr = np.array(g_partial)
            if len(g_partial_arr) == len(r):
                ax.plot(r, g_partial_arr, label=pair_name,
                       linewidth=1.0, color=colors[i % len(colors)], alpha=0.7)

    # Plot total RDF last (thick black line on top)
    ax.plot(r, g_r, 'k-', linewidth=2.0, label='Total')

    ax.set_xlabel(r'Distance r ($\AA$)')
    ax.set_ylabel('g(r)')
    ax.set_title('Radial Distribution Function')
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(r))
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

    # Always show legend (at minimum "Total" is labeled)
    ax.legend(loc='upper right', fontsize=fonts['legend_fontsize'], framealpha=0.9)

    fig.tight_layout()

    path = output_dir / 'rdf_plot.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Generated RDF plot: {path} (partials: {len(partial)})")
    return path


def create_styles() -> Dict[str, ParagraphStyle]:
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    custom_styles = {
        'Title': ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
        ),
        'Subtitle': ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=20,
        ),
        'Heading1': ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2c3e50'),
        ),
        'Heading2': ParagraphStyle(
            'Heading2',
            parent=styles['Heading2'],
            fontSize=13,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e'),
        ),
        'Heading3': ParagraphStyle(
            'Heading3',
            parent=styles['Heading3'],
            fontSize=11,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor('#455a64'),
        ),
        'Body': ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14,
        ),
        'Caption': ParagraphStyle(
            'Caption',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=15,
        ),
        'Reference': ParagraphStyle(
            'Reference',
            parent=styles['Normal'],
            fontSize=9,
            leftIndent=20,
            firstLineIndent=-20,
            spaceAfter=5,
            leading=12,
        ),
        'TableHeader': ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        'TableCell': ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
        ),
        'ParamCell': ParagraphStyle(
            'ParamCell',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_LEFT,
            leading=10,
            wordWrap='CJK',
        ),
        'Footer': ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
    }

    return custom_styles


def create_header_footer(canvas, doc):
    """Add header and footer to each page."""
    canvas.saveState()

    # Footer with page number
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.grey)
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch, f"Page {page_num}")

    # Footer with OptiMat Alloys attribution
    canvas.drawRightString(
        letter[0] - 0.75 * inch, 0.5 * inch,
        "Generated by OptiMat Alloys"
    )

    canvas.restoreState()


def format_composition(composition: Dict[str, float]) -> str:
    """Format composition dictionary as chemical formula string."""
    parts = []
    for element, fraction in sorted(composition.items()):
        if fraction > 0:
            if fraction == 1.0:
                parts.append(element)
            else:
                # Convert to subscript-style percentage
                parts.append(f"{element}<sub>{fraction*100:.1f}</sub>")
    return "".join(parts)

# ===== SECTION BUILDER FUNCTIONS =====
# Helper functions to build individual report sections for cleaner code organization


def _build_title_page(
    structure_id: str,
    composition: Dict[str, float],
    structure_data: Dict[str, Any],
    styles: Dict,
    composition_string: Optional[str] = None,
) -> List:
    """Build title page elements.

    Returns:
        List of reportlab elements for the title page
    """
    elements = []

    elements.append(Spacer(1, 1 * inch))
    elements.append(Paragraph("Structure and Properties Report", styles['Title']))

    # Composition as subtitle
    comp_str = composition_string if composition_string else format_composition(composition)
    elements.append(Paragraph(f"Composition: {comp_str}", styles['Subtitle']))
    elements.append(Spacer(1, 0.5 * inch))

    # Structure ID and date
    elements.append(Paragraph(
        f"Structure ID: <font face='Courier'>{structure_id}</font>",
        styles['Body']
    ))
    elements.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Body']
    ))

    # Calculator info
    calculator = structure_data.get('calculator_name', 'Unknown')
    device = structure_data.get('device_type', 'Unknown')
    elements.append(Paragraph(
        f"Calculator: {calculator} ({device})",
        styles['Body']
    ))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(PageBreak())

    return elements


def _build_methods_section(
    calculation_types: List[str],
    structure_data: Dict[str, Any],
    styles: Dict,
    ref_tracker: 'ReferenceTracker',
) -> List:
    """Build computational methods section.

    Returns:
        List of reportlab elements
    """
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("4. Computational Methods", styles['Heading1']))

    calculator_name = structure_data.get('calculator_name', None)
    all_methods = []
    for calc_type in calculation_types:
        methods = get_methods_for_calculation(calc_type, calculator_name=calculator_name)
        for method in methods:
            if method.name not in [m.name for m in all_methods]:
                all_methods.append(method)

    for method in all_methods:
        elements.append(Paragraph(method.name, styles['Heading2']))
        inline_cite = ref_tracker.cite(method.references) if method.references else ""
        description_with_cite = f"{method.description} {inline_cite}" if inline_cite else method.description
        elements.append(Paragraph(description_with_cite, styles['Body']))

        if method.parameters:
            param_data = [['Parameter', 'Value']]
            is_calculator_entry = method.name.endswith(('Potential', 'Potentials'))
            if is_calculator_entry and calculator_name:
                param_data.append([
                    Paragraph('<b>Model used</b>', styles['ParamCell']),
                    Paragraph(f'<b>{calculator_name}</b>', styles['ParamCell']),
                ])
            for param, value in method.parameters.items():
                param_data.append([
                    Paragraph(param.replace('_', ' ').title(), styles['ParamCell']),
                    Paragraph(str(value), styles['ParamCell']),
                ])

            param_table = Table(param_data, colWidths=[1.9 * inch, 5.1 * inch])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(param_table)
            elements.append(Spacer(1, 0.2 * inch))

    # Software versions
    elements.append(Paragraph("Software Versions", styles['Heading2']))
    versions = get_software_versions()
    version_text = ", ".join([f"{pkg}: {ver}" for pkg, ver in versions.items()])
    elements.append(Paragraph(version_text, styles['Body']))

    return elements


def _build_references_section(
    styles: Dict,
    ref_tracker: 'ReferenceTracker',
) -> List:
    """Build references section.

    Returns:
        List of reportlab elements
    """
    elements = []
    elements.append(PageBreak())
    elements.append(Paragraph("5. References", styles['Heading1']))

    pub_refs = ref_tracker.get_reference_list()
    for ref in pub_refs:
        elements.append(Paragraph(ref, styles['Reference']))

    return elements



def generate_structure_report(
    structure_id: str,
    composition: Dict[str, float],
    structure_data: Dict[str, Any],
    calculation_types: List[str],
    output_path: Path,
    image_paths: Optional[Dict[str, Path]] = None,
    composition_string: Optional[str] = None,
) -> Path:
    """Generate a comprehensive PDF report for a structure.

    Args:
        structure_id: UUID of the structure
        composition: Element composition dictionary
        structure_data: Dictionary containing all computed properties
        calculation_types: List of calculation types performed
            ('alloy_generation', 'elastic', 'qha', 'thermal_conductivity')
        output_path: Path to save the PDF
        image_paths: Optional dictionary of image paths to embed
            {'structure': Path, 'elastic': Path, 'qha': Path, ...}
        composition_string: Optional pre-formatted composition string
            (preserves original element order from user input)

    Returns:
        Path to the generated PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = create_styles()

    # Create document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    elements = []

    # Figure counter for sequential numbering throughout the report
    fig_num = 1

    # Reference tracker for numbered citations in order of appearance
    ref_tracker = ReferenceTracker()

    # ===== TITLE PAGE =====
    elements.append(Spacer(1, 1 * inch))
    elements.append(Paragraph("Structure and Properties Report", styles['Title']))

    # Composition as subtitle (use provided string to preserve original element order)
    comp_str = composition_string if composition_string else format_composition(composition)
    elements.append(Paragraph(f"Composition: {comp_str}", styles['Subtitle']))

    elements.append(Spacer(1, 0.5 * inch))

    # Structure ID and date
    elements.append(Paragraph(
        f"Structure ID: <font face='Courier'>{structure_id}</font>",
        styles['Body']
    ))
    elements.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Body']
    ))

    # Calculator info
    calculator = structure_data.get('calculator_name', 'Unknown')
    device = structure_data.get('device_type', 'Unknown')
    elements.append(Paragraph(
        f"Calculator: {calculator} ({device})",
        styles['Body']
    ))

    elements.append(Spacer(1, 0.3 * inch))

    # Note: Cover page structure image removed per user request

    elements.append(PageBreak())

    # ===== STRUCTURE SUMMARY =====
    elements.append(Paragraph("1. Structure Summary", styles['Heading1']))

    # Basic properties table
    summary_data = [
        ['Property', 'Value', 'Unit'],
    ]

    if structure_data.get('num_atoms') is not None:
        summary_data.append(['Number of atoms', str(structure_data['num_atoms']), ''])
    if structure_data.get('target_structure'):
        summary_data.append(['Target structure', structure_data['target_structure'].upper(), ''])
    if structure_data.get('density_g_per_cm3') is not None:
        summary_data.append(['Density', f"{structure_data['density_g_per_cm3']:.4f}", 'g/cm³'])
    if structure_data.get('volume_per_atom_A3') is not None:
        summary_data.append(['Volume per atom', f"{structure_data['volume_per_atom_A3']:.4f}", 'Å³'])

    # Cell parameters
    if 'cell_parameters' in structure_data and structure_data['cell_parameters']:
        cell = structure_data['cell_parameters']
        if cell.get('a') is not None:
            summary_data.append(['Lattice parameter a', f"{cell['a']:.4f}", 'Å'])
        if cell.get('b') is not None:
            summary_data.append(['Lattice parameter b', f"{cell['b']:.4f}", 'Å'])
        if cell.get('c') is not None:
            summary_data.append(['Lattice parameter c', f"{cell['c']:.4f}", 'Å'])
        if cell.get('alpha') is not None:
            summary_data.append(['Angle α', f"{cell['alpha']:.2f}", '°'])
        if cell.get('beta') is not None:
            summary_data.append(['Angle β', f"{cell['beta']:.2f}", '°'])
        if cell.get('gamma') is not None:
            summary_data.append(['Angle γ', f"{cell['gamma']:.2f}", '°'])

    # Energy
    if structure_data.get('energy_per_atom_eV') is not None:
        summary_data.append(['Energy per atom', f"{structure_data['energy_per_atom_eV']:.6f}", 'eV'])
    if structure_data.get('formation_energy_ground_state_reference_eV_per_atom') is not None:
        summary_data.append([
            'Formation energy',
            f"{structure_data['formation_energy_ground_state_reference_eV_per_atom']:.6f}",
            'eV/atom'
        ])

    if len(summary_data) > 1:
        table = Table(summary_data, colWidths=[2.5 * inch, 2 * inch, 1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), _TABLE_FONT),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        # Formation energy explanation (if formation energy was computed)
        if structure_data.get('formation_energy_ground_state_reference_eV_per_atom') is not None:
            elements.append(Paragraph(
                "<b>Formation Energy:</b> E<sub>form</sub> = E<sub>alloy</sub> - Σ(x<sub>i</sub> × E<sub>ref,i</sub>), "
                "where E<sub>alloy</sub> is the energy per atom of the relaxed alloy structure, "
                "x<sub>i</sub> is the atomic fraction of element i, "
                "and E<sub>ref,i</sub> is the ground-state reference energy of pure element i. "
                "Reference energies are obtained by relaxing each element in five crystal structures "
                "(sc, bcc, fcc, hcp, diamond) and selecting the minimum energy per atom. "
                "Negative values indicate thermodynamic stability relative to pure elements.",
                styles['Body']
            ))
            elements.append(Spacer(1, 0.2 * inch))

    # OVITO Structure Visualizations
    elements.append(Paragraph("Relaxed Structure Visualization", styles['Heading2']))

    # Element-colored structure image
    if image_paths and 'structure_elements' in image_paths:
        img_path = image_paths['structure_elements']
        if Path(img_path).exists():
            elements.append(Image(str(img_path), width=3.5 * inch, height=2.6 * inch))
            elements.append(Paragraph(
                f"Figure {fig_num}: Relaxed structure colored by element",
                styles['Caption']
            ))
            fig_num += 1
            elements.append(Spacer(1, 0.15 * inch))

    # PTM-colored structure image
    if image_paths and 'structure_analysis' in image_paths:
        img_path = image_paths['structure_analysis']
        if Path(img_path).exists():
            elements.append(Image(str(img_path), width=3.5 * inch, height=2.6 * inch))
            elements.append(Paragraph(
                f"Figure {fig_num}: Relaxed structure colored by PTM structural analysis",
                styles['Caption']
            ))
            fig_num += 1
            elements.append(Spacer(1, 0.15 * inch))

    # PTM Analysis table (compact summary alongside images)
    if 'PTM_structural_analysis_in_percent' in structure_data:
        elements.append(Paragraph("PTM Analysis Summary", styles['Heading2']))
        elements.append(Paragraph(
            "<b>Polyhedral Template Matching (PTM)</b> identifies local crystal structure by matching "
            "atomic neighborhoods to ideal structure templates. The table shows the fraction of atoms "
            "identified as each structure type (FCC, BCC, HCP, etc.). "
            f"{ref_tracker.cite('ptm_larsen2016')}",
            styles['Body']
        ))
        elements.append(Spacer(1, 0.1 * inch))
        ptm = structure_data['PTM_structural_analysis_in_percent']
        ptm_data = [['Structure Type', 'Fraction (%)']]
        for struct_type, fraction in sorted(ptm.items(), key=lambda x: -x[1]):
            if fraction > 0.1:  # Only show significant fractions
                ptm_data.append([struct_type.upper(), f"{fraction:.2f}"])

        if len(ptm_data) > 1:
            ptm_table = Table(ptm_data, colWidths=[2 * inch, 1.5 * inch])
            ptm_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))
            elements.append(ptm_table)
            elements.append(Spacer(1, 0.3 * inch))

    # RDF Plot (Radial Distribution Function)
    if 'rdf_data' in structure_data:
        elements.append(Paragraph("Radial Distribution Function", styles['Heading2']))
        elements.append(Paragraph(
            "<b>g(r)</b> represents the probability of finding an atom at distance r relative to a uniform distribution. "
            "g(r) = 1 for an ideal gas; peaks indicate preferred neighbor distances (coordination shells). "
            "For multi-element systems, partial RDFs are concentration-weighted: "
            "g<sub>total</sub>(r) = Σ c<sub>i</sub>c<sub>j</sub>g<sub>ij</sub>(r), "
            "where c<sub>i</sub> and c<sub>j</sub> are atomic concentrations of elements i and j, "
            "and g<sub>ij</sub>(r) is the partial RDF for the i-j pair. "
            "Computed with cutoff radius = 10 Å and 200 histogram bins.",
            styles['Body']
        ))
        elements.append(Spacer(1, 0.1 * inch))
        rdf_plot_dir = output_path.parent / "rdf_plots"
        rdf_plot_path = generate_rdf_plot(structure_data['rdf_data'], rdf_plot_dir)
        if rdf_plot_path and rdf_plot_path.exists():
            elements.append(Image(str(rdf_plot_path), width=4 * inch, height=2.8 * inch))
            elements.append(Paragraph(
                f"Figure {fig_num}: Radial distribution function g(r)",
                styles['Caption']
            ))
            fig_num += 1
            elements.append(Spacer(1, 0.3 * inch))

    # ===== ELASTIC PROPERTIES =====
    if 'elastic' in calculation_types and 'elastic_properties' in structure_data:
        elements.append(PageBreak())
        elements.append(Paragraph("2. Elastic Properties", styles['Heading1']))

        elastic = structure_data['elastic_properties']

        # 1. Stiffness tensor (first)
        if 'stiffness_tensor' in elastic:
            elements.append(Paragraph("Stiffness Tensor (GPa)", styles['Heading2']))
            tensor = elastic['stiffness_tensor']
            tensor_data = [[''] + [f'C{j+1}' for j in range(6)]]
            for i in range(6):
                row = [f'C{i+1}']
                for j in range(6):
                    val = tensor[i][j] if isinstance(tensor[i][j], (int, float)) else 0
                    row.append(f'{val:.1f}')
                tensor_data.append(row)

            tensor_table = Table(tensor_data, colWidths=[0.6 * inch] * 7)
            tensor_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
                ('FONTNAME', (0, 0), (0, -1), _TABLE_FONT_BOLD),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(tensor_table)
            elements.append(Spacer(1, 0.3 * inch))

        # Elastic Stability (Born Criterion)
        if 'elastic_stability' in structure_data:
            elements.append(Paragraph("Elastic Stability (Born Criterion)", styles['Heading2']))
            stability = structure_data['elastic_stability']
            is_stable = stability.get('born_criterion_satisfied', stability.get('is_elastically_stable', False))
            eigenvalues = stability.get('eigenvalues', [])

            status_text = "STABLE" if is_stable else "UNSTABLE"
            status_color = colors.HexColor('#27ae60') if is_stable else colors.HexColor('#e74c3c')

            stability_data = [
                ['Status', 'Stiffness Tensor Eigenvalues (GPa)'],
                [status_text, ', '.join(f'{ev:.2f}' for ev in eigenvalues) if eigenvalues else 'N/A']
            ]

            stability_table = Table(stability_data, colWidths=[2 * inch, 4.5 * inch])
            stability_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TEXTCOLOR', (0, 1), (0, 1), status_color),
                ('FONTNAME', (0, 1), (0, 1), _TABLE_FONT_BOLD),
            ]))
            elements.append(stability_table)
            # Explanatory note for stability
            elements.append(Paragraph(
                "<i>Note: STABLE = all stiffness tensor eigenvalues are positive (Born criterion satisfied).</i>",
                styles['Body']
            ))
            elements.append(Spacer(1, 0.3 * inch))

        # ELATE Analysis - Single Crystal Elastic Properties
        if 'elate_properties' in structure_data:
            elate = structure_data['elate_properties']

            elements.append(Paragraph("Single Crystal Elastic Properties", styles['Heading2']))

            # Add explanation for single crystal properties
            elements.append(Paragraph(
                "<b>Directional Properties:</b> Single crystal elastic properties vary with crystallographic direction "
                "due to anisotropy. All directional properties are computed from the compliance tensor S = C<sup>-1</sup>:",
                styles['Body']
            ))
            elements.append(Paragraph(
                "<b>Young's modulus:</b> 1/E(n) = S<sub>ijkl</sub> n<sub>i</sub> n<sub>j</sub> n<sub>k</sub> n<sub>l</sub>. "
                "<b>Linear compressibility:</b> β(n) = S<sub>ijkl</sub> δ<sub>kl</sub> n<sub>i</sub> n<sub>j</sub>. "
                "<b>Shear modulus:</b> 1/G(n,m) = 4 S<sub>ijkl</sub> n<sub>i</sub> m<sub>j</sub> n<sub>k</sub> m<sub>l</sub>. "
                "<b>Poisson's ratio:</b> ν(n,m) = -S<sub>ijkl</sub> n<sub>i</sub> n<sub>j</sub> m<sub>k</sub> m<sub>l</sub> / S<sub>ijkl</sub> n<sub>i</sub> n<sub>j</sub> n<sub>k</sub> n<sub>l</sub>, "
                "where S<sub>ijkl</sub> is the compliance tensor, δ<sub>kl</sub> is the Kronecker delta, "
                "n is the loading direction, and m is the perpendicular direction for lateral response.",
                styles['Body']
            ))
            elements.append(Paragraph(
                "<b>Auxetic Behavior:</b> Negative Poisson's ratio in certain directions indicates the material "
                "expands laterally when stretched (unusual mechanical behavior).",
                styles['Body']
            ))
            elements.append(Paragraph(
                "<b>Wave velocities:</b> v<sub>s</sub>(n) = √(G(n)/ρ) for shear waves, "
                "v<sub>l</sub>(n) = √((K(n) + 4G(n)/3)/ρ) for longitudinal/compression waves, "
                "where v<sub>s</sub> is the shear wave velocity, v<sub>l</sub> is the longitudinal wave velocity, "
                "G(n) is the directional shear modulus, K(n) is the directional bulk modulus, "
                "n is the propagation direction, and ρ is the material density.",
                styles['Body']
            ))
            elements.append(Spacer(1, 0.15 * inch))

            # Build combined single crystal table
            has_auxetic = elate.get('has_auxetic_behavior', False)
            min_shear_wave = elate.get('min_shear_wave_speed_m_s', 0)

            single_crystal_data = [
                ['Property', 'Min', 'Max', 'Unit'],
                ["Young's Modulus (E)",
                 f"{elate.get('min_youngs_modulus_GPa', 0):.2f}",
                 f"{elate.get('max_youngs_modulus_GPa', 0):.2f}", 'GPa'],
                ['Shear Modulus (G)',
                 f"{elate.get('min_shear_modulus_GPa', 0):.2f}",
                 f"{elate.get('max_shear_modulus_GPa', 0):.2f}", 'GPa'],
                ["Poisson's Ratio (ν)",
                 f"{elate.get('min_poisson_ratio', 0):.4f}",
                 f"{elate.get('max_poisson_ratio', 0):.4f}", ''],
                ['Linear Compressibility (β)',
                 f"{elate.get('min_linear_compressibility_TPa_inv', 0):.3f}",
                 f"{elate.get('max_linear_compressibility_TPa_inv', 0):.3f}", '1/TPa'],
            ]

            # Add wave velocities if computed
            if min_shear_wave > 0:
                single_crystal_data.append([
                    'Shear Wave Velocity (Vs)',
                    f"{elate.get('min_shear_wave_speed_m_s', 0):.1f}",
                    f"{elate.get('max_shear_wave_speed_m_s', 0):.1f}", 'm/s'])
                single_crystal_data.append([
                    'Compression Wave (Vp)',
                    f"{elate.get('min_compression_wave_speed_m_s', 0):.1f}",
                    f"{elate.get('max_compression_wave_speed_m_s', 0):.1f}", 'm/s'])

            # Auxetic behavior indicator (A^U moved to Polycrystalline table)
            auxetic_text = 'ν < 0 in some dirs' if has_auxetic else ''
            single_crystal_data.append([
                'Auxetic Behavior',
                'Yes' if has_auxetic else 'No',
                auxetic_text, ''])

            single_crystal_table = Table(single_crystal_data, colWidths=[2.2 * inch, 1.2 * inch, 1.6 * inch, 0.6 * inch])
            single_crystal_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8e44ad')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))
            elements.append(single_crystal_table)
            elements.append(Spacer(1, 0.3 * inch))

        # Polycrystalline (Averaged) Elastic Properties - at end of section
        if 'elate_properties' in structure_data:
            elate = structure_data['elate_properties']

            elements.append(Paragraph("Polycrystalline (Averaged) Elastic Properties", styles['Heading2']))

            # Voigt averaging explanation with equations
            elements.append(Paragraph(
                f"<b>Voigt average</b> (uniform strain, upper bound) {ref_tracker.cite('voigt1928')}: "
                "K<sub>V</sub> = [(C<sub>11</sub>+C<sub>22</sub>+C<sub>33</sub>)+2(C<sub>12</sub>+C<sub>13</sub>+C<sub>23</sub>)]/9, "
                "G<sub>V</sub> = [(C<sub>11</sub>+C<sub>22</sub>+C<sub>33</sub>)-(C<sub>12</sub>+C<sub>13</sub>+C<sub>23</sub>)+3(C<sub>44</sub>+C<sub>55</sub>+C<sub>66</sub>)]/15.",
                styles['Body']
            ))

            # Reuss averaging explanation with explicit equations
            elements.append(Paragraph(
                f"<b>Reuss average</b> (uniform stress, lower bound) {ref_tracker.cite('reuss1929')}: "
                "K<sub>R</sub> = 1/[(S<sub>11</sub>+S<sub>22</sub>+S<sub>33</sub>)+2(S<sub>12</sub>+S<sub>13</sub>+S<sub>23</sub>)], "
                "G<sub>R</sub> = 15/[4(S<sub>11</sub>+S<sub>22</sub>+S<sub>33</sub>)-4(S<sub>12</sub>+S<sub>13</sub>+S<sub>23</sub>)+3(S<sub>44</sub>+S<sub>55</sub>+S<sub>66</sub>)], "
                "where S = C<sup>-1</sup> is the compliance tensor.",
                styles['Body']
            ))

            # Hill averaging explanation
            elements.append(Paragraph(
                f"<b>Hill average</b> (arithmetic mean) {ref_tracker.cite('hill1952')}: "
                "K<sub>H</sub> = (K<sub>V</sub>+K<sub>R</sub>)/2, G<sub>H</sub> = (G<sub>V</sub>+G<sub>R</sub>)/2. "
                "Hill averages are typically closest to experimental values for polycrystalline materials.",
                styles['Body']
            ))
            elements.append(Paragraph(
                "<b>Pugh Ratio (K/G):</b> Empirical ductility criterion. K/G > 1.75 suggests ductile behavior; "
                "K/G < 1.75 suggests brittle behavior.",
                styles['Body']
            ))
            AU = elate.get('universal_anisotropy_index', 0)
            aniso_interp = 'isotropic' if AU < 0.1 else 'anisotropic'
            elements.append(Paragraph(
                "<b>Universal Anisotropy Index:</b> A<sup>U</sup> = 5(G<sub>V</sub>/G<sub>R</sub>) + (K<sub>V</sub>/K<sub>R</sub>) - 6 = "
                f"<b>{AU:.4f}</b> ({aniso_interp}). "
                "A<sup>U</sup> = 0 for perfect isotropy; larger values indicate stronger anisotropy. "
                f"{ref_tracker.cite('anisotropy_ranganathan2008')}",
                styles['Body']
            ))
            elements.append(Spacer(1, 0.15 * inch))

            # Polycrystalline properties table
            poly_data = [['Property', 'Voigt', 'Reuss', 'Hill', 'Unit']]
            poly_data.append([
                'Bulk Modulus (K)',
                f"{elate.get('voigt_bulk_modulus_GPa', 0):.2f}",
                f"{elate.get('reuss_bulk_modulus_GPa', 0):.2f}",
                f"{elate.get('hill_bulk_modulus_GPa', 0):.2f}",
                'GPa'
            ])
            poly_data.append([
                'Shear Modulus (G)',
                f"{elate.get('voigt_shear_modulus_GPa', 0):.2f}",
                f"{elate.get('reuss_shear_modulus_GPa', 0):.2f}",
                f"{elate.get('hill_shear_modulus_GPa', 0):.2f}",
                'GPa'
            ])
            # Pugh Ratio (K/G) - ductile if > 1.75, brittle if < 1.75
            K_V = elate.get('voigt_bulk_modulus_GPa', 0)
            K_R = elate.get('reuss_bulk_modulus_GPa', 0)
            K_H = elate.get('hill_bulk_modulus_GPa', 0)
            G_V = elate.get('voigt_shear_modulus_GPa', 0)
            G_R = elate.get('reuss_shear_modulus_GPa', 0)
            G_H = elate.get('hill_shear_modulus_GPa', 0)
            pugh_V = K_V / G_V if G_V > 0 else 0
            pugh_R = K_R / G_R if G_R > 0 else 0
            pugh_H = K_H / G_H if G_H > 0 else 0
            poly_data.append([
                'Pugh Ratio (K/G)',
                f"{pugh_V:.2f}",
                f"{pugh_R:.2f}",
                f"{pugh_H:.2f}",
                ''
            ])

            poly_table = Table(poly_data, colWidths=[1.8 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 0.6 * inch])
            poly_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))
            elements.append(poly_table)
            elements.append(Spacer(1, 0.3 * inch))

        # ELATE directional property figures (if available)
        elate_images = image_paths.get('elate', {}) if image_paths else {}
        if elate_images:
            elements.append(PageBreak())
            elements.append(Paragraph("Directional Elastic Property Visualizations", styles['Heading2']))
            elements.append(Paragraph(
                "The following plots show the directional dependence of elastic properties. "
                "2D projections show property variation in crystallographic planes (XY, XZ, YZ), "
                "while 3D surfaces show the full angular dependence.",
                styles['Body']
            ))
            elements.append(Spacer(1, 0.2 * inch))

            # Property types with display names
            property_types = [
                ('young', "Young's Modulus E (GPa)"),
                ('shear', "Shear Modulus G (GPa)"),
                ('poisson', "Poisson's Ratio ν"),
                ('lc', "Linear Compressibility β (1/TPa)"),
                ('shear_speed', "Shear Wave Speed v<sub>s</sub> (m/s)"),
                ('compression_speed', "Compression Wave Speed v<sub>l</sub> (m/s)"),
            ]

            for prop_key, prop_name in property_types:
                # Check if this property has images
                xy_path = elate_images.get(f'{prop_key}_xy')
                xz_path = elate_images.get(f'{prop_key}_xz')
                yz_path = elate_images.get(f'{prop_key}_yz')
                surf_path = elate_images.get(f'{prop_key}_3d')

                has_2d = any(p and Path(p).exists() for p in [xy_path, xz_path, yz_path])
                has_3d = surf_path and Path(surf_path).exists()

                if has_2d or has_3d:
                    elements.append(Paragraph(prop_name, styles['Heading3']))

                    # 2D projections in a row (3 images side by side)
                    if has_2d:
                        row_images = []
                        plane_labels = []
                        for path, label in [(xy_path, 'XY (001)'), (xz_path, 'XZ (010)'), (yz_path, 'YZ (100)')]:
                            if path and Path(path).exists():
                                row_images.append(Image(str(path), width=2.1*inch, height=2.1*inch))
                                plane_labels.append(label)
                            else:
                                row_images.append('')
                                plane_labels.append('')

                        if any(row_images):
                            img_table = Table([row_images], colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
                            img_table.setStyle(TableStyle([
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ]))
                            elements.append(img_table)

                            # Caption with plane labels
                            elements.append(Paragraph(
                                f"Figure {fig_num}: {prop_name} - 2D projections ({', '.join([l for l in plane_labels if l])})",
                                styles['Caption']
                            ))
                            fig_num += 1
                            elements.append(Spacer(1, 0.15 * inch))

                    # 3D surface (centered)
                    if has_3d:
                        elements.append(Image(str(surf_path), width=4*inch, height=3.5*inch))
                        elements.append(Paragraph(
                            f"Figure {fig_num}: {prop_name} - 3D directional surface",
                            styles['Caption']
                        ))
                        fig_num += 1
                        elements.append(Spacer(1, 0.2 * inch))

        # Legacy single elastic image (fallback for old structures)
        elif image_paths and 'elastic' in image_paths:
            img_path = image_paths['elastic']
            if Path(img_path).exists():
                elements.append(Image(str(img_path), width=5 * inch, height=4 * inch))
                elements.append(Paragraph(
                    f"Figure {fig_num}: Directional elastic properties (ELATE analysis)",
                    styles['Caption']
                ))
                fig_num += 1

    # ===== QHA PROPERTIES =====
    if 'qha' in calculation_types and 'qha_properties' in structure_data:
        elements.append(PageBreak())
        elements.append(Paragraph("3. Finite Temperature Properties (QHA)", styles['Heading1']))

        qha = structure_data['qha_properties']

        # QHA explanation with equations
        elements.append(Paragraph(
            "<b>Quasi-Harmonic Approximation (QHA)</b> computes temperature-dependent properties by allowing "
            "phonon frequencies to depend on volume while treating phonons as harmonic at each volume.",
            styles['Body']
        ))
        elements.append(Paragraph(
            "<b>Helmholtz Free Energy:</b> F(V,T) = E<sub>static</sub>(V) + F<sub>phonon</sub>(V,T), "
            "where E<sub>static</sub> is the 0K DFT/ML-potential energy and F<sub>phonon</sub> includes vibrational entropy.",
            styles['Body']
        ))
        elements.append(Paragraph(
            "<b>Gibbs Free Energy:</b> G(T,P) = min<sub>V</sub>[F(V,T) + PV]. Equilibrium volume V(T) is found by minimizing G.",
            styles['Body']
        ))
        elements.append(Paragraph(
            "<b>Derived Properties:</b> "
            "B(T) = V(d<sup>2</sup>G/dV<sup>2</sup>) (bulk modulus), "
            "α(T) = (1/V)(dV/dT)<sub>P</sub> (thermal expansion), "
            "C<sub>p</sub>(T) = -T(d<sup>2</sup>G/dT<sup>2</sup>)<sub>P</sub> (isobaric heat capacity). "
            f"{ref_tracker.cite('phonopy2015')}",
            styles['Body']
        ))
        elements.append(Spacer(1, 0.15 * inch))

        # Dynamical Stability (from phonon frequencies)
        qha_stable = structure_data.get('qha_dynamically_stable')
        qha_has_imaginary = structure_data.get('qha_has_imaginary_modes', False)
        qha_min_freq = structure_data.get('qha_min_frequency_THz')

        if qha_stable is not None:
            elements.append(Paragraph("Dynamical Stability", styles['Heading2']))

            if qha_stable:
                stability_text = (
                    "<b>Status: STABLE</b> — No imaginary phonon frequencies detected across all volumes. "
                    "The structure is dynamically stable at the harmonic level."
                )
                status_color = colors.HexColor('#27ae60')  # Green
            else:
                min_freq_str = f"{qha_min_freq:.3f} THz" if qha_min_freq is not None else "N/A"
                stability_text = (
                    f"<b>Status: UNSTABLE</b> — Imaginary phonon frequencies detected (min: {min_freq_str}). "
                    "Negative frequencies indicate dynamical instability; the structure may spontaneously "
                    "transform to a lower-energy configuration. Computed thermodynamic properties should "
                    "be interpreted with caution."
                )
                status_color = colors.HexColor('#e74c3c')  # Red

            elements.append(Paragraph(stability_text, styles['Body']))
            elements.append(Paragraph(
                "<i>Note: Imaginary frequencies appear as negative values in phonon calculations. "
                "A small threshold (0.1 THz) is used to distinguish real instabilities from numerical noise.</i>",
                styles['Body']
            ))
            elements.append(Spacer(1, 0.15 * inch))

        # Key temperatures table
        if 'temperature_dependent' in qha:
            temp_data = qha['temperature_dependent']
            key_temps = [0, 300, 500, 800, 1000]

            elements.append(Paragraph("Properties at Key Temperatures", styles['Heading2']))
            qha_table_data = [['T (K)', 'B (GPa)', 'V (Å³)', 'α (×10⁻⁵ K⁻¹)', 'Cp (J/mol/K)']]

            temperatures = temp_data.get('temperatures', [])
            for T in key_temps:
                if T in temperatures:
                    idx = temperatures.index(T)
                    row = [str(T)]

                    # Use plural keys to match database storage
                    B = temp_data.get('bulk_moduli', [])
                    row.append(f"{B[idx]:.2f}" if idx < len(B) else 'N/A')

                    V = temp_data.get('volumes', [])
                    row.append(f"{V[idx]:.4f}" if idx < len(V) else 'N/A')

                    alpha = temp_data.get('thermal_expansion', [])
                    if idx < len(alpha):
                        row.append(f"{alpha[idx] * 1e5:.2f}")
                    else:
                        row.append('N/A')

                    Cp = temp_data.get('heat_capacities', [])
                    row.append(f"{Cp[idx]:.2f}" if idx < len(Cp) else 'N/A')

                    qha_table_data.append(row)

            if len(qha_table_data) > 1:
                qha_table = Table(qha_table_data, colWidths=[1 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.4 * inch])
                qha_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                ]))
                elements.append(qha_table)
                elements.append(Spacer(1, 0.3 * inch))

        # Generate QHA plots
        qha_plot_dir = output_path.parent / "qha_plots"
        qha_plots = generate_qha_plots(qha, qha_plot_dir)

        if qha_plots:
            elements.append(Paragraph("Temperature-Dependent Property Plots", styles['Heading2']))

            # QHA plots - continues sequential numbering from previous sections
            plot_configs = [
                ('gibbs', 'Gibbs Free Energy G(T)'),
                ('bulk_modulus', 'Bulk Modulus B(T)'),
                ('thermal_expansion', 'Thermal Expansion α(T)'),
                ('heat_capacity', 'Heat Capacity Cp(T)'),
            ]

            for plot_key, plot_title in plot_configs:
                if plot_key in qha_plots:
                    plot_path = qha_plots[plot_key]
                    if plot_path.exists():
                        elements.append(Image(str(plot_path), width=4 * inch, height=2.8 * inch))
                        elements.append(Paragraph(
                            f"Figure {fig_num}: {plot_title}",
                            styles['Caption']
                        ))
                        fig_num += 1

            elements.append(Spacer(1, 0.2 * inch))

        # QHA surface plots (F(T,V), S(T,V), Cv(T,V))
        qha_surfaces = image_paths.get('qha_surfaces', {}) if image_paths else {}
        if qha_surfaces:
            elements.append(Paragraph("Volume-Dependent Property Surfaces", styles['Heading2']))
            elements.append(Paragraph(
                "3D surfaces showing how thermodynamic properties vary with both temperature and volume. "
                "These visualizations reveal the full thermodynamic landscape used in QHA calculations.",
                styles['Body']
            ))
            elements.append(Spacer(1, 0.15 * inch))

            surface_configs = [
                ('qha_ftv_surface', 'Helmholtz Free Energy F(T,V)'),
                ('qha_stv_surface', 'Entropy S(T,V)'),
                ('qha_cvtv_surface', 'Heat Capacity Cv(T,V)'),
            ]

            for surf_key, surf_title in surface_configs:
                surf_path = qha_surfaces.get(surf_key)
                if surf_path and Path(surf_path).exists():
                    elements.append(Image(str(surf_path), width=4.5 * inch, height=3.5 * inch))
                    elements.append(Paragraph(
                        f"Figure {fig_num}: {surf_title} surface",
                        styles['Caption']
                    ))
                    fig_num += 1
                    elements.append(Spacer(1, 0.15 * inch))

        # Legacy QHA image if available (combined plot)
        elif image_paths and 'qha' in image_paths:
            img_path = image_paths['qha']
            if Path(img_path).exists():
                elements.append(Image(str(img_path), width=5 * inch, height=4 * inch))
                elements.append(Paragraph(
                    f"Figure {fig_num}: Temperature-dependent thermodynamic properties",
                    styles['Caption']
                ))
                fig_num += 1

    # ===== COMPUTATIONAL METHODS =====
    elements.append(PageBreak())
    elements.append(Paragraph("4. Computational Methods", styles['Heading1']))

    # Gather all methods used (pass calculator_name for correct method descriptions)
    calculator_name = structure_data.get('calculator_name', None)
    all_methods = []
    for calc_type in calculation_types:
        methods = get_methods_for_calculation(calc_type, calculator_name=calculator_name)
        for method in methods:
            if method.name not in [m.name for m in all_methods]:
                all_methods.append(method)

    for method in all_methods:
        elements.append(Paragraph(method.name, styles['Heading2']))
        # Add inline citations using the reference tracker (maintains order of appearance)
        inline_cite = ref_tracker.cite(method.references) if method.references else ""
        description_with_cite = f"{method.description} {inline_cite}" if inline_cite else method.description
        elements.append(Paragraph(description_with_cite, styles['Body']))

        # Parameters table
        if method.parameters:
            param_data = [['Parameter', 'Value']]
            is_calculator_entry = method.name.endswith(('Potential', 'Potentials'))
            if is_calculator_entry and calculator_name:
                param_data.append([
                    Paragraph('<b>Model used</b>', styles['ParamCell']),
                    Paragraph(f'<b>{calculator_name}</b>', styles['ParamCell']),
                ])
            for param, value in method.parameters.items():
                param_data.append([
                    Paragraph(param.replace('_', ' ').title(), styles['ParamCell']),
                    Paragraph(str(value), styles['ParamCell']),
                ])

            param_table = Table(param_data, colWidths=[1.9 * inch, 5.1 * inch])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(param_table)
            elements.append(Spacer(1, 0.2 * inch))

    # Software versions
    elements.append(Paragraph("Software Versions", styles['Heading2']))
    versions = get_software_versions()
    version_text = ", ".join([f"{pkg}: {ver}" for pkg, ver in versions.items()])
    elements.append(Paragraph(version_text, styles['Body']))

    # ===== REFERENCES =====
    elements.append(PageBreak())
    elements.append(Paragraph("5. References", styles['Heading1']))

    # Get references in order of first appearance (from ref_tracker)
    pub_refs = ref_tracker.get_reference_list()
    for ref in pub_refs:
        elements.append(Paragraph(ref, styles['Reference']))

    # Build PDF
    doc.build(elements, onFirstPage=create_header_footer, onLaterPages=create_header_footer)

    return output_path



def generate_minimal_report(
    structure_id: str,
    composition: Dict[str, float],
    basic_data: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Generate a minimal single-page report for structures without detailed calculations.

    Args:
        structure_id: UUID of the structure
        composition: Element composition dictionary
        basic_data: Basic structure data (energy, density, etc.)
        output_path: Path to save the PDF

    Returns:
        Path to the generated PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = create_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    elements = []

    # Title
    elements.append(Paragraph("Structure Summary", styles['Title']))
    comp_str = format_composition(composition)
    elements.append(Paragraph(f"Composition: {comp_str}", styles['Subtitle']))

    elements.append(Spacer(1, 0.3 * inch))

    # Basic info
    elements.append(Paragraph(
        f"Structure ID: <font face='Courier'>{structure_id}</font>",
        styles['Body']
    ))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Body']
    ))

    elements.append(Spacer(1, 0.3 * inch))

    # Properties table
    data = [['Property', 'Value', 'Unit']]
    for key, value in basic_data.items():
        if isinstance(value, float):
            data.append([key.replace('_', ' ').title(), f"{value:.4f}", ''])
        elif isinstance(value, (int, str)):
            data.append([key.replace('_', ' ').title(), str(value), ''])

    if len(data) > 1:
        table = Table(data, colWidths=[2.5 * inch, 2 * inch, 1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), _TABLE_FONT_BOLD),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(table)

    doc.build(elements, onFirstPage=create_header_footer, onLaterPages=create_header_footer)

    return output_path
