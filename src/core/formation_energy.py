"""
Formation energy calculations for alloys.

This module computes formation energies using reference energies
from single-element calculations.
"""

from typing import Dict, List, Optional
import numpy as np


def formation_energy_per_atom(
    elements: List[str],
    fractions: List[float],
    energies_ref: Dict[str, float],
    *,
    E_total: Optional[float] = None,
    natoms: Optional[int] = None,
    E_per_atom: Optional[float] = None,
    tol: float = 1e-6,
) -> float:
    """
    Compute alloy formation energy per atom (eV/atom).

    Formation energy is defined as:
        E_form = E_alloy_per_atom - sum_i x_i * E_ref_i

    where x_i are atomic fractions and E_ref_i are reference energies
    for pure elements.

    Args:
        elements: List of element symbols
        fractions: Atomic fractions (must sum to 1)
        energies_ref: Dictionary of reference energies per element
        E_total: Total energy of the alloy (eV) - alternative to E_per_atom
        natoms: Number of atoms - required if using E_total
        E_per_atom: Energy per atom of the alloy (eV/atom) - alternative to E_total
        tol: Tolerance for fraction sum validation

    Returns:
        Formation energy per atom (eV/atom)

    Raises:
        ValueError: If inputs are invalid or inconsistent

    Examples:
        >>> refs = {"Cu": -3.72, "Ag": -2.95}
        >>> # CuAg alloy at -3.35 eV/atom
        >>> E_form = formation_energy_per_atom(
        ...     ["Cu", "Ag"], [0.5, 0.5], refs,
        ...     E_per_atom=-3.35
        ... )
        >>> E_form  # doctest: +SKIP
        -0.015

        >>> # Using total energy
        >>> E_form = formation_energy_per_atom(
        ...     ["Cu", "Ag"], [0.5, 0.5], refs,
        ...     E_total=-335.0, natoms=100
        ... )
        >>> E_form  # doctest: +SKIP
        -0.015
    """
    if E_per_atom is None:
        if E_total is None or natoms is None:
            raise ValueError("Provide either E_per_atom, or (E_total and natoms).")
        E_per_atom = float(E_total) / int(natoms)

    if len(elements) != len(fractions):
        raise ValueError("`elements` and `fractions` must have equal length.")

    s = float(sum(fractions))
    if not np.isclose(s, 1.0, rtol=0, atol=tol):
        raise ValueError(f"Fractions must sum to 1 (got {s}).")

    # Verify refs
    for el in elements:
        if el not in energies_ref:
            raise ValueError(f"Missing reference energy for element '{el}'.")

    mix_ref = sum(x * energies_ref[el] for el, x in zip(elements, fractions))
    return float(E_per_atom - mix_ref)
