"""
Element support testing for ORB calculators.

This module provides tools for testing which elements are supported
by each ORB calculator variant. It systematically tests all elements
in the periodic table and records success/failure for each.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from ase import Atoms
from ase.build import bulk
import math


def test_element_support(
    element: str,
    calculator: str,
    structure: Literal['sc', 'bcc', 'fcc', 'hcp', 'diamond'] = "fcc",
    device: str = "cuda",
    timeout: float = 60.0
) -> Dict[str, any]:
    """
    Test if an element is supported by a calculator.

    Creates a simple unit cell with radii-based lattice constant,
    then attempts a single-point energy calculation to verify
    the calculator can handle this element.

    Args:
        element: Element symbol (e.g., 'Cu', 'Fe', 'Si')
        calculator: Calculator name (e.g., 'orb-v3-direct-20-omat')
        structure: Crystal structure to test (default: fcc)
        device: Device to use ('cuda' or 'cpu')
        timeout: Maximum time in seconds for test (default: 60)

    Returns:
        Dictionary with test results:
        {
            'element': str,
            'calculator': str,
            'structure': str,
            'supported': bool,
            'energy': float or None,  # eV/atom
            'lattice_constant': float or None,  # Angstrom
            'num_atoms': int or None,
            'error': str or None,
            'device': str
        }

    Examples:
        >>> result = test_element_support('Cu', 'orb-v3-direct-20-omat')
        >>> result['supported']
        True
        >>> result['energy']  # doctest: +SKIP
        -3.721
    """
    from .calculators import load_calculator

    result = {
        'element': element,
        'calculator': calculator,
        'structure': structure,
        'supported': False,
        'energy': None,
        'lattice_constant': None,
        'num_atoms': None,
        'error': None,
        'device': device
    }

    try:
        # Get radii-based initial lattice constant
        from .structure_builder import estimate_alloy_lattice_constant_radii

        try:
            a0 = estimate_alloy_lattice_constant_radii(structure, [element], [1.0])
        except (ValueError, KeyError) as e:
            # Element not in radii database - use ASE covalent radius as fallback
            from ase.data import covalent_radii, atomic_numbers
            try:
                z = atomic_numbers[element]
                r_cov = float(covalent_radii[z])

                # Convert covalent to effective metallic radius (empirical factor)
                r_metal = r_cov * 1.2

                # Compute lattice constant from radius
                if structure == "sc":
                    a0 = 2 * r_metal
                elif structure == "bcc":
                    a0 = (4 * r_metal) / math.sqrt(3)
                elif structure == "fcc":
                    a0 = 2 * math.sqrt(2) * r_metal
                elif structure == "diamond":
                    a0 = (8 * r_metal) / math.sqrt(3)
                elif structure == "hcp":
                    a0 = 2 * r_metal
                else:
                    raise ValueError(f"Unknown structure: {structure}")

            except Exception as radius_error:
                result['error'] = f"Cannot determine atomic radius: {str(radius_error)}"
                return result

        # Build unit cell
        cubic = structure in ["sc", "fcc", "bcc", "diamond"]

        if structure == "hcp":
            c = math.sqrt(8.0 / 3.0) * a0
            atoms = bulk(element, "hcp", a=a0, c=c)
        else:
            atoms = bulk(element, structure, a=a0, cubic=cubic)

        result['lattice_constant'] = float(a0)
        result['num_atoms'] = len(atoms)

        # Load calculator
        calc = load_calculator(model=calculator, device=device)
        atoms.calc = calc

        # Attempt energy calculation (single-point, no optimization)
        energy = atoms.get_potential_energy()
        energy_per_atom = energy / len(atoms)

        result['energy'] = float(energy_per_atom)
        result['supported'] = True

    except Exception as e:
        result['error'] = str(e)
        result['supported'] = False

    return result


def test_all_elements(
    calculator: str,
    structures: Optional[List[str]] = None,
    device: str = "cuda",
    output_file: Optional[str] = None,
    max_workers: int = 1
) -> Dict[str, Dict[str, any]]:
    """
    Test all elements in the periodic table for a calculator.

    Args:
        calculator: Calculator name (e.g., 'orb-v3-direct-20-omat')
        structures: List of structures to test (default: ['fcc', 'bcc', 'sc'])
        device: Device to use ('cuda' or 'cpu')
        output_file: Path to save results JSON (default: auto-generated)
        max_workers: Number of parallel workers (GPU: use 1, CPU: can use >1)

    Returns:
        Dictionary mapping element symbols to test results

    Notes:
        - Tests all 119 elements from ASE
        - For GPU, use max_workers=1 to avoid memory issues
        - Results are saved incrementally to avoid data loss
    """
    from ase.data import chemical_symbols

    if structures is None:
        structures = ['fcc', 'bcc', 'sc']

    # Get all elements (skip index 0 which is empty string)
    all_elements = [s for s in chemical_symbols[1:] if s]

    print(f"Testing {len(all_elements)} elements for calculator: {calculator}")
    print(f"Structures to test: {structures}")
    print(f"Device: {device}")

    # Auto-generate output filename if not provided
    if output_file is None:
        safe_calc_name = calculator.replace("-", "_")
        output_file = f"data/element_support/element_support_{safe_calc_name}.json"

    results = {}

    # Test each element
    for idx, element in enumerate(all_elements):
        print(f"\n[{idx+1}/{len(all_elements)}] Testing {element}...")

        element_results = {}

        # Test multiple structures for robustness
        for struct in structures:
            print(f"  Structure: {struct}...", end=" ")

            try:
                test_result = test_element_support(
                    element=element,
                    calculator=calculator,
                    structure=struct,
                    device=device
                )

                element_results[struct] = test_result

                if test_result['supported']:
                    print(f"✓ E = {test_result['energy']:.4f} eV/atom")
                else:
                    print(f"✗ {test_result.get('error', 'Failed')}")

            except Exception as e:
                print(f"✗ Exception: {str(e)}")
                element_results[struct] = {
                    'element': element,
                    'calculator': calculator,
                    'structure': struct,
                    'supported': False,
                    'error': str(e),
                    'device': device
                }

        # Determine overall support (supported in at least one structure)
        overall_supported = any(
            result.get('supported', False)
            for result in element_results.values()
        )

        results[element] = {
            'supported': overall_supported,
            'structures': element_results
        }

        # Save incrementally
        save_results(results, output_file, calculator, in_progress=True)

    # Save final results
    save_results(results, output_file, calculator, in_progress=False)

    # Print summary
    supported_count = sum(1 for r in results.values() if r['supported'])
    print(f"\n{'='*60}")
    print(f"Testing complete!")
    print(f"Supported elements: {supported_count}/{len(all_elements)}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    return results


def save_results(
    results: Dict[str, Dict],
    output_file: str,
    calculator: str,
    in_progress: bool = False
) -> None:
    """
    Save test results to JSON file with metadata.

    Args:
        results: Test results dictionary
        output_file: Output file path
        calculator: Calculator name
        in_progress: Whether testing is still in progress
    """
    # Count supported elements
    supported_count = sum(1 for r in results.values() if r.get('supported', False))

    # Create metadata
    metadata = {
        "_metadata": {
            "calculator": calculator,
            "tested_at": datetime.now().isoformat(),
            "total_elements": len(results),
            "supported_count": supported_count,
            "in_progress": in_progress,
            "version": "1.0"
        }
    }

    # Merge metadata with results
    output_data = {**metadata, **results}

    # Write to file
    output_path = Path(output_file)
    with output_path.open('w') as f:
        json.dump(output_data, f, indent=2, sort_keys=True)


def load_element_support(
    calculator: str,
    base_dir: str = "data/element_support"
) -> Dict[str, bool]:
    """
    Load element support data for a calculator.

    Args:
        calculator: Calculator name (e.g., 'orb-v3-direct-20-omat')
        base_dir: Directory containing element support files (default: data/element_support)

    Returns:
        Dictionary mapping element symbols to support status (bool)

    Raises:
        FileNotFoundError: If element support file not found

    Examples:
        >>> support = load_element_support('orb-v3-direct-20-omat')  # doctest: +SKIP
        >>> support['Cu']
        True
        >>> support['He']
        False
    """
    safe_calc_name = calculator.replace("-", "_")
    file_path = Path(base_dir) / f"element_support_{safe_calc_name}.json"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Element support file not found: {file_path}\n"
            f"Run element testing first: python scripts/test_element_support.py"
        )

    with file_path.open('r') as f:
        data = json.load(f)

    # Extract support status for each element
    support = {}
    for key, value in data.items():
        if key != "_metadata":
            support[key] = value.get('supported', False)

    return support


def get_supported_elements(
    calculator: str,
    base_dir: str = "data/element_support"
) -> List[str]:
    """
    Get list of supported elements for a calculator.

    Args:
        calculator: Calculator name
        base_dir: Directory containing element support files (default: data/element_support)

    Returns:
        List of supported element symbols

    Examples:
        >>> elements = get_supported_elements('orb-v3-direct-20-omat')  # doctest: +SKIP
        >>> 'Cu' in elements
        True
        >>> len(elements) > 50
        True
    """
    support = load_element_support(calculator, base_dir)
    return [elem for elem, supported in support.items() if supported]


def print_support_summary(
    calculator: str,
    base_dir: str = "data/element_support"
) -> None:
    """
    Print summary of element support for a calculator.

    Args:
        calculator: Calculator name
        base_dir: Directory containing element support files (default: data/element_support)
    """
    safe_calc_name = calculator.replace("-", "_")
    file_path = Path(base_dir) / f"element_support_{safe_calc_name}.json"

    with file_path.open('r') as f:
        data = json.load(f)

    metadata = data.get("_metadata", {})

    print(f"\n{'='*60}")
    print(f"Element Support Summary: {calculator}")
    print(f"{'='*60}")
    print(f"Total elements tested: {metadata.get('total_elements', 0)}")
    print(f"Supported elements: {metadata.get('supported_count', 0)}")
    print(f"Tested at: {metadata.get('tested_at', 'Unknown')}")

    if metadata.get('in_progress', False):
        print(f"\n⚠️  Warning: Testing still in progress")

    # Group elements by support status
    supported = []
    unsupported = []

    for elem, info in data.items():
        if elem == "_metadata":
            continue
        if info.get('supported', False):
            supported.append(elem)
        else:
            unsupported.append(elem)

    print(f"\nSupported ({len(supported)}):")
    print(", ".join(supported))

    print(f"\nUnsupported ({len(unsupported)}):")
    print(", ".join(unsupported))
    print(f"{'='*60}\n")
