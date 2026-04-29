"""
Composition Parser Utility

Parses composition strings like "Ag75Cu25" into elements and fractions.
Handles various formats that quantized LLMs might produce.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Valid element symbols (from ASE)
VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
}


@dataclass
class ParsedComposition:
    """Result of parsing a composition string."""
    elements: List[str]
    fractions: List[float]
    raw_values: List[float]  # Original numeric values before normalization
    format_detected: str  # Description of detected format

    def is_valid(self) -> bool:
        """Check if parsed composition is valid."""
        if not self.elements or not self.fractions:
            return False
        if len(self.elements) != len(self.fractions):
            return False
        # Check fractions sum to ~1
        total = sum(self.fractions)
        if not (0.99 <= total <= 1.01):
            return False
        # Check all elements are valid
        for elem in self.elements:
            if elem not in VALID_ELEMENTS:
                return False
        return True


def validate_elements(elements: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a list of element symbols.

    Args:
        elements: List of element symbols to validate

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_elements(['Cu', 'Ag'])
        (True, None)
        >>> validate_elements(['Cu', 'Xx'])
        (False, "Invalid element symbol: 'Xx'")
    """
    for elem in elements:
        if elem not in VALID_ELEMENTS:
            return False, f"Invalid element symbol: '{elem}'"
    return True, None


def parse_composition_string(comp: str) -> Optional[ParsedComposition]:
    """
    Parse a composition string into elements and fractions.

    Handles multiple formats:
    - "Ag75Cu25" or "Ag75-Cu25" -> 75% Ag, 25% Cu
    - "Cu-Ag 75-25" or "Cu Ag 75 25" -> 75% Cu, 25% Ag
    - "AgCu" or "Ag-Cu" -> 50% each (equal parts)
    - "Ag3Cu1" -> 75% Ag, 25% Cu (ratio notation)
    - "Ag0.75Cu0.25" -> 75% Ag, 25% Cu (decimal fractions)

    Args:
        comp: Composition string to parse

    Returns:
        ParsedComposition if successful, None if parsing fails

    Examples:
        >>> result = parse_composition_string("Ag75Cu25")
        >>> result.elements
        ['Ag', 'Cu']
        >>> result.fractions
        [0.75, 0.25]
    """
    if not comp or not isinstance(comp, str):
        return None

    # Clean input
    comp = comp.strip()
    if not comp:
        return None

    # Pattern 1: Element+Number format (e.g., "Ag75Cu25", "Ag75-Cu25")
    # Matches: Element symbol followed by number (integer or decimal)
    pattern1 = r'([A-Z][a-z]?)(\d+(?:\.\d+)?)'
    matches = re.findall(pattern1, comp)

    if len(matches) >= 2:
        elements = [m[0] for m in matches]
        values = [float(m[1]) for m in matches]

        # Validate elements
        is_valid, _ = validate_elements(elements)
        if not is_valid:
            return None

        # Determine if values are percentages, ratios, or fractions
        total = sum(values)

        if all(0 < v <= 1 for v in values) and 0.99 <= total <= 1.01:
            # Already fractions (e.g., "Ag0.75Cu0.25")
            fractions = values
            format_detected = "decimal fractions"
        elif 95 <= total <= 105:
            # Percentages (e.g., "Ag75Cu25")
            fractions = [v / total for v in values]
            format_detected = "percentages"
        else:
            # Ratios (e.g., "Ag3Cu1" -> 3:1 ratio)
            fractions = [v / total for v in values]
            format_detected = f"ratios (sum={total})"

        return ParsedComposition(
            elements=elements,
            fractions=fractions,
            raw_values=values,
            format_detected=format_detected
        )

    # Pattern 2: Separated format (e.g., "Cu Ag 75 25", "Cu-Ag 75-25")
    # Try to find elements and numbers separately
    # NOTE: Must check this BEFORE elements-only pattern to handle "Cu Ag 75 25" correctly
    elem_pattern = r'([A-Z][a-z]?)'
    num_pattern = r'(\d+(?:\.\d+)?)'

    elem_matches = [m for m in re.findall(elem_pattern, comp) if m in VALID_ELEMENTS]
    num_matches = [float(m) for m in re.findall(num_pattern, comp)]

    if len(elem_matches) >= 2 and len(num_matches) >= 2:
        # Take only as many numbers as elements
        n = min(len(elem_matches), len(num_matches))
        elements = elem_matches[:n]
        values = num_matches[:n]

        total = sum(values)
        if total > 0:
            fractions = [v / total for v in values]
            return ParsedComposition(
                elements=elements,
                fractions=fractions,
                raw_values=values,
                format_detected="separated format"
            )

    # Pattern 3: Elements only (e.g., "AgCu", "Ag-Cu", "Cu Ag")
    # Split by common separators and extract element symbols (no numbers present)
    parts = re.findall(elem_pattern, comp)

    # Filter to valid elements only
    elements = [p for p in parts if p in VALID_ELEMENTS]

    if len(elements) >= 2:
        # Equal parts for all elements
        n = len(elements)
        fractions = [1.0 / n] * n
        return ParsedComposition(
            elements=elements,
            fractions=fractions,
            raw_values=[1.0] * n,
            format_detected="elements only (equal parts)"
        )

    return None


def format_composition_from_parsed(parsed: ParsedComposition) -> str:
    """
    Format a parsed composition back to a standard string.

    Args:
        parsed: ParsedComposition object

    Returns:
        Formatted string like "Ag75Cu25"

    Examples:
        >>> parsed = ParsedComposition(['Ag', 'Cu'], [0.75, 0.25], [75, 25], 'percentages')
        >>> format_composition_from_parsed(parsed)
        'Ag75Cu25'
    """
    parts = []
    for elem, frac in zip(parsed.elements, parsed.fractions):
        percentage = frac * 100
        if percentage == int(percentage):
            parts.append(f"{elem}{int(percentage)}")
        else:
            parts.append(f"{elem}{percentage:.1f}")
    return "".join(parts)
