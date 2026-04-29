"""
Reference data management for element properties.

This module handles:
- Building initial cells for elements
- Computing and storing lattice constants
- Computing and storing reference energies
- Loading reference data for formation energy calculations
"""

import json
from typing import Dict, List, Literal, Optional
from ase import Atoms


ReferenceMode = Literal["ground_state", "same_structure"]


def initial_cell(structure: str, element: str) -> Atoms:
    """
    Construct an ASE bulk cell with an estimated lattice constant.

    Uses radii-based estimate for initial lattice constant guess.

    Args:
        structure: Lattice structure (sc, bcc, fcc, hcp, diamond)
        element: Element symbol

    Returns:
        ASE Atoms object with initial cell

    Examples:
        >>> atoms = initial_cell("fcc", "Cu")
        >>> len(atoms)
        4
    """
    from ase.build import bulk
    import math
    from .structure_builder import estimate_alloy_lattice_constant_radii

    a0 = estimate_alloy_lattice_constant_radii(structure, [element], [1.0])

    # For many structures, a cubic unit cell is preferred
    cubic = structure in ["sc", "fcc", "bcc", "diamond"]

    if structure == "hcp":
        # Ideal c/a for hcp
        c = math.sqrt(8.0 / 3.0) * a0
        return bulk(element, "hcp", a=a0, c=c)
    else:
        return bulk(element, structure, a=a0, cubic=cubic)


def extract_lattice_constant_a(atoms: Atoms, structure: str) -> float:
    """
    Extract 'a' lattice parameter from relaxed cell.

    Args:
        atoms: ASE Atoms object (typically relaxed)
        structure: Lattice structure type

    Returns:
        Lattice constant 'a' in Angstroms

    Examples:
        >>> a = extract_lattice_constant_a(atoms, "fcc")
        >>> 3.0 < a < 5.0  # Typical range for metals
        True
    """
    a, b, c = atoms.cell.lengths()
    if structure in ["sc", "bcc", "fcc", "diamond"]:
        # Should be cubic after hydrostatic scaling
        return float(a)
    elif structure == "hcp":
        # Report a only
        return float(a)
    else:
        return float(a)


def get_supported_elements(calculator: str, use_fallback: bool = True) -> List[str]:
    """
    Get list of supported elements for a calculator.

    Args:
        calculator: Calculator name (e.g., 'orb-v3-direct-20-omat')
        use_fallback: If True and element testing not run, use default 48 elements

    Returns:
        List of supported element symbols

    Raises:
        FileNotFoundError: If element testing results not found and use_fallback=False

    Examples:
        >>> elements = get_supported_elements("orb-v3-direct-20-omat")  # doctest: +SKIP
        >>> len(elements) > 40
        True
    """
    try:
        from .element_testing import get_supported_elements as load_supported
        return load_supported(calculator)
    except FileNotFoundError:
        if use_fallback:
            # Fall back to original 48 core elements (for backwards compatibility)
            print(f"⚠️  Element testing results not found for {calculator}")
            print(f"   Using fallback list of 48 core elements (118 tracked in database, 117 ORB-supported)")
            print(f"   Run: python scripts/test_element_support.py --calculator {calculator}")
            return [
                "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Sc", "Ti", "V", "Cr", "Mn",
                "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf",
                "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Si", "Ge"
            ]
        else:
            raise


def precompute_and_save(
    hydrostatic_cell_relaxation: bool = True,
    optimizer: str = "FIRE",
    fmax: float = 0.01,
    calculator: str = "orb-v3-direct-20-omat",
    cache: Optional['ReferenceDataCache'] = None,
    elements: Optional[List[str]] = None
) -> None:
    """
    Precompute reference lattice constants and energies for elements.

    Relaxes single-element structures for all supported elements and
    saves results to JSON files with metadata.

    Args:
        hydrostatic_cell_relaxation: Allow cell shape changes
        optimizer: Optimization algorithm (FIRE or LBFGS)
        fmax: Force convergence threshold (eV/Å)
        calculator: Calculator name (e.g., 'orb-v3-direct-20-omat')
        cache: ReferenceDataCache instance (if None, creates one for this calculator)
        elements: List of elements to precompute (if None, uses supported elements)

    Examples:
        >>> from src.storage.cache import get_reference_cache
        >>> cache = get_reference_cache(calculator="orb-v3-direct-20-omat")
        >>> precompute_and_save(fmax=0.005, calculator="orb-v3-direct-20-omat", cache=cache)
    """
    from .optimization import relax_atoms

    # Create cache if not provided
    if cache is None:
        from ..storage.cache import ReferenceDataCache
        cache = ReferenceDataCache(calculator=calculator)

    structures = ["sc", "bcc", "fcc", "hcp", "diamond"]

    # Get elements to precompute
    if elements is None:
        elements = get_supported_elements(calculator, use_fallback=True)

    print(f"\nPrecomputing reference data for {len(elements)} elements...")
    print(f"Calculator: {calculator}")
    print(f"Structures: {', '.join(structures)}")
    print(f"Optimizer: {optimizer}, fmax: {fmax}\n")

    latcs: Dict[str, Dict[str, Optional[float]]] = {}
    enes: Dict[str, Dict[str, Optional[float]]] = {}

    for elem in elements:
        latcs[elem] = {}
        enes[elem] = {}
        for struct in structures:
            try:
                atoms = initial_cell(struct, elem)
                atoms = relax_atoms(
                    atoms,
                    hydrostatic_cell_relaxation=hydrostatic_cell_relaxation,
                    optimizer=optimizer,
                    fmax=fmax,
                    calculator=calculator,
                )

                # Evaluate metrics
                e_per_atom = atoms.get_potential_energy() / len(atoms)
                a_relaxed = extract_lattice_constant_a(atoms, struct)

                latcs[elem][struct] = float(a_relaxed)
                enes[elem][struct] = float(e_per_atom)

                print(f"[OK] {elem:>2s} {struct:>7s}  a = {a_relaxed:.4f} Å  E/atom = {e_per_atom:.6f} eV")
            except Exception as exc:
                latcs[elem][struct] = None
                enes[elem][struct] = None
                print(f"[FAIL] {elem} {struct}: {exc}")

    # Save using cache with metadata
    cache.save_with_metadata(
        lattice_data=latcs,
        energy_data=enes,
        fmax=fmax,
        optimizer=optimizer,
        hydrostatic=hydrostatic_cell_relaxation
    )


def load_reference_energies(
    json_path: str,
    reference_mode: ReferenceMode = "ground_state",
    structure: Optional[str] = None,
) -> Dict[str, float]:
    """
    Load per-element reference energies from JSON file.

    Args:
        json_path: Path to energies_per_atom.json
        reference_mode: Either "ground_state" (min across structures) or
                       "same_structure" (specific structure)
        structure: Required if reference_mode is "same_structure"

    Returns:
        Dictionary mapping element symbols to reference energies (eV/atom)

    Raises:
        ValueError: If no valid energies found or structure missing

    Examples:
        >>> refs = load_reference_energies("energies_per_atom.json", "ground_state")
        >>> refs["Cu"]  # doctest: +SKIP
        -3.721

        >>> refs = load_reference_energies("energies_per_atom.json", "same_structure", "fcc")
        >>> refs["Cu"]  # doctest: +SKIP
        -3.721
    """
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Filter out metadata
    data = {k: v for k, v in raw_data.items() if k != "_metadata"}

    refs: Dict[str, float] = {}
    for elem, struct_map in data.items():
        # Filter out Nones
        clean = {s: e for s, e in (struct_map or {}).items() if e is not None}
        if not clean:
            raise ValueError(f"No valid reference energies for element {elem} in {json_path}.")

        if reference_mode == "ground_state":
            # Min over available structures
            refs[elem] = float(min(clean.values()))
        elif reference_mode == "same_structure":
            if structure is None:
                raise ValueError("`structure` must be provided when reference_mode='same_structure'.")
            if structure not in clean:
                raise ValueError(f"Missing {structure} energy for element {elem}. Available: {list(clean)}")
            refs[elem] = float(clean[structure])
        else:
            raise ValueError(f"Unknown reference_mode: {reference_mode}")

    return refs
