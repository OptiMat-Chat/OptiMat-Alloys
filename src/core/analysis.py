"""
Structural analysis functions using OVITO.

This module provides tools for analyzing atomic structures using
polyhedral template matching (PTM) and radial distribution functions (RDF).
"""

from typing import Dict, List
from ase import Atoms
import numpy as np
import threading

# Global lock for OVITO operations (OVITO is not thread-safe)
_ovito_lock = threading.Lock()


def structural_analysis(atoms: Atoms) -> Dict[str, float]:
    """
    Analyze crystal structure using Polyhedral Template Matching.

    Uses OVITO's PTM modifier to identify local crystal structures
    for each atom and returns the fraction of each structure type.

    Args:
        atoms: ASE Atoms object to analyze

    Returns:
        Dictionary with structure fractions (as percentages):
        {
            "sc": float,      # Simple cubic
            "fcc": float,     # Face-centered cubic
            "bcc": float,     # Body-centered cubic
            "hcp": float,     # Hexagonal close-packed
            "diamond": float, # Diamond cubic
            "ico": float,     # Icosahedral
            "other": float    # Unidentified/disordered
        }

    Examples:
        >>> analysis = structural_analysis(atoms)
        >>> print(f"FCC fraction: {analysis['fcc']:.1f}%")
    """
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import StaticSource, Pipeline
    from ovito.modifiers import PolyhedralTemplateMatchingModifier

    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))

    ptm = PolyhedralTemplateMatchingModifier()
    ptm.rmsd_cutoff = 0.0
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True
    pipeline.modifiers.append(ptm)

    data = pipeline.compute()

    st = data.particles["Structure Type"].array  # per-atom integer labels
    n = st.size

    T = PolyhedralTemplateMatchingModifier.Type  # alias

    def pct(t):
        return float(np.count_nonzero(st == t)) / n * 100.0

    return {
        "sc":      pct(T.SC),
        "fcc":     pct(T.FCC),
        "bcc":     pct(T.BCC),
        "hcp":     pct(T.HCP),
        "diamond": pct(T.CUBIC_DIAMOND),
        "ico":     pct(T.ICO),
        "other":   pct(T.OTHER),
    }


def compute_density(atoms: Atoms) -> float:
    """
    Calculate density in g/cm³.

    Args:
        atoms: ASE Atoms object

    Returns:
        Density in g/cm³

    Examples:
        >>> density = compute_density(atoms)
        >>> print(f"Density: {density:.3f} g/cm³")
    """
    # Atomic mass unit to grams: 1 u = 1.66053906660e-24 g
    # Å³ to cm³: 1 Å³ = 1e-24 cm³
    mass_g = atoms.get_masses().sum() * 1.66053906660e-24
    volume_cm3 = atoms.get_volume() * 1e-24
    return mass_g / volume_cm3


def compute_coordination_rdf(
    atoms: Atoms,
    cutoff: float = 10.0,
    n_bins: int = 200
) -> Dict[str, np.ndarray]:
    """
    Compute radial distribution function (RDF).

    Uses OVITO's coordination analysis modifier to compute
    pair distribution functions. For multi-element systems,
    computes both partial RDFs (one per element pair) and
    the total RDF as a weighted sum: g_total(r) = Σ c_i × c_j × g_ij(r).

    Args:
        atoms: ASE Atoms object
        cutoff: Maximum distance for RDF (Angstroms)
        n_bins: Number of histogram bins

    Returns:
        Dictionary with RDF data:
        {
            "r": np.ndarray,           # Radial distances
            "g_total": np.ndarray,     # Total RDF (always computed)
            "partial": Dict[str, np.ndarray]  # Partial RDFs by pair type (empty for single element)
        }

    Examples:
        >>> rdf = compute_coordination_rdf(atoms, cutoff=10.0)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(rdf["r"], rdf["g_total"], label="Total")
        >>> for pair, g in rdf["partial"].items():
        ...     plt.plot(rdf["r"], g, label=pair)
    """
    import ovito
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import StaticSource, Pipeline
    from ovito.modifiers import CoordinationAnalysisModifier

    # Copy atoms and remove calculator to avoid OVITO conversion issues
    atoms_copy = atoms.copy()
    atoms_copy.calc = None

    # Use lock for OVITO thread safety (OVITO is not thread-safe)
    with _ovito_lock:
        try:
            # Clear OVITO scene before computation
            ovito.scene.pipelines.clear()

            # Convert ASE atoms to OVITO format
            data = ase_to_ovito(atoms_copy)
            pipeline = Pipeline(source=StaticSource(data=data))

            # Add coordination analysis modifier
            coord_mod = CoordinationAnalysisModifier(
                cutoff=cutoff,
                number_of_bins=n_bins,
                partial=True
            )
            pipeline.modifiers.append(coord_mod)

            data = pipeline.compute()

            # Extract RDF data
            rdf_table = data.tables["coordination-rdf"]
            arr = rdf_table.xy()  # shape (N, 1 + C)
            r = arr[:, 0]
            Y = arr[:, 1:]

            result = {"r": r}

            if Y.shape[1] == 1:
                # Single total RDF
                result["g_total"] = Y[:, 0]
                result["partial"] = {}
            else:
                # Partial RDFs
                pair_names = list(rdf_table.y.component_names)
                result["partial"] = {name: Y[:, i] for i, name in enumerate(pair_names)}

                # Compute total g(r) as weighted sum of partials
                # Formula: g_total(r) = Σ c_i × c_j × g_ij(r)
                # where c_i, c_j are elemental concentrations
                from collections import Counter
                counts = Counter(atoms_copy.get_chemical_symbols())
                total_atoms = len(atoms_copy)
                concentrations = {el: count / total_atoms for el, count in counts.items()}

                g_total = np.zeros_like(r)
                for pair_name, g_ij in result["partial"].items():
                    el_i, el_j = pair_name.split('-')
                    c_i = concentrations.get(el_i, 0.0)
                    c_j = concentrations.get(el_j, 0.0)

                    if el_i == el_j:
                        # Homo-atomic pair (Cu-Cu, Ni-Ni): count once
                        g_total += c_i * c_j * g_ij
                    else:
                        # Hetero-atomic pair (Cu-Ni): count twice (Cu→Ni and Ni→Cu)
                        g_total += 2.0 * c_i * c_j * g_ij

                result["g_total"] = g_total

        finally:
            # Always clean up OVITO scene
            ovito.scene.pipelines.clear()

    return result


def reorder_partial_rdf(
    partial: Dict[str, np.ndarray],
    element_order: List[str]
) -> Dict[str, np.ndarray]:
    """Reorder partial RDF dictionary to match user's element order.

    This function only changes the ORDER of keys in the dictionary,
    not the data. Each key (e.g., "Ag-Ag") still maps to its original
    RDF array - we're just sorting which key comes first when iterating.

    Args:
        partial: Dict of partial RDFs with keys like 'Cu-Ag', 'Ag-Ag'
        element_order: User's element order, e.g., ['Cu', 'Ag']

    Returns:
        Reordered dict with pairs sorted by element_order

    Example:
        For element_order=['Cu', 'Ag']:
        Input:  {'Ag-Ag': [...], 'Ag-Cu': [...], 'Cu-Cu': [...]}
        Output: {'Cu-Cu': [...], 'Cu-Ag': [...], 'Ag-Ag': [...]}

        Note: 'Ag-Cu' becomes 'Cu-Ag' in the sorted output because
        Cu comes first in element_order. The data array is unchanged.
    """
    if not partial or not element_order:
        return partial

    def pair_sort_key(pair_name: str) -> tuple:
        el_i, el_j = pair_name.split('-')
        # Get index in element_order (use high value if not found)
        idx_i = element_order.index(el_i) if el_i in element_order else 999
        idx_j = element_order.index(el_j) if el_j in element_order else 999
        return (idx_i, idx_j)

    sorted_keys = sorted(partial.keys(), key=pair_sort_key)
    return {k: partial[k] for k in sorted_keys}
