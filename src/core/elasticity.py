"""
Elastic constant calculations for atomistic structures.

This module provides tools for computing elastic stiffness tensors
using finite difference strain-energy methods.
"""

from itertools import combinations_with_replacement, product
from typing import Optional, Dict, Any
import threading
import numpy as np
from ase import Atoms
from ase.units import GPa
from .optimization import StructureOptimizer
from .cancellation import check_cancellation, ProgressCallback, format_progress_message


def compute_elastic_stiffness_tensor(
    atoms: Atoms,
    epsilon: float = 1e-2,
    relax_kwargs: Optional[Dict[str, Any]] = None,
    cancellation_event: Optional[threading.Event] = None,
    progress_callback: ProgressCallback = None
) -> np.ndarray:
    """
    Calculate elastic stiffness tensor (relaxed/Born tensor) in Voigt form.

    Uses finite difference strain-energy method. The structure is subjected to
    small strain deformations (including asymmetric ones) and atomic positions
    are relaxed at constant cell for each strained state. The symmetric stiffness
    tensor is extracted from the quadratic energy-strain relationship via
    overdetermined least-squares fitting.

    This computes the **relaxed** (Born) stiffness tensor, which represents
    real material elastic behavior and is measured experimentally.

    **Cancellation Support**: This function supports cooperative cancellation.
    Pass a threading.Event to enable graceful stopping between deformations.
    If cancelled, a ComputationCancelledException is raised with progress info.

    Parameters
    ----------
    atoms
        Input structure; should be fully relaxed with attached calculator.
        The calculator is used for all energy evaluations.
    epsilon
        Magnitude of applied strain (default: 1e-2 = 1%).
        Must be >= 0.01 to avoid numerical noise with fmax=0.005 relaxation.
    relax_kwargs
        Keyword arguments for StructureOptimizer.relax().
        Default: {'fmax': 0.005, 'optimizer': 'FIRE', 'max_steps': 100}
    cancellation_event
        Threading event for cooperative cancellation (optional).
        Set this event to request graceful stop between deformations.
    progress_callback
        Callback function for progress updates (optional).
        Called as: callback(current_step, total_steps, status_message)

    Returns
    -------
    C_voigt : np.ndarray
        Elastic stiffness tensor in Voigt notation (6x6 matrix) with units of GPa.
        Indices: 0=11, 1=22, 2=33, 3=23, 4=13, 5=12 (Voigt convention).

    Raises
    ------
    ComputationCancelledException
        If cancellation_event is set during computation.

    Notes
    -----
    - Number of deformations: 180 (includes asymmetric strains)
    - Each deformation requires atomic relaxation (computationally expensive)
    - Typical runtime: 20-40 minutes for 500-atom system on GPU
    - Uses overdetermined least-squares to recover symmetric C tensor
    - The calculator must be attached to the input atoms object
    - Cancellation checks occur between deformations (safe checkpoints)

    Examples
    --------
    >>> from src.core.calculators import load_calculator
    >>> calc = load_calculator('orb-v3-direct-20-omat')
    >>> atoms.calc = calc
    >>> C = compute_elastic_stiffness_tensor(atoms, epsilon=1e-2)
    >>> C.shape
    (6, 6)
    >>> print(f"Bulk modulus (Voigt): {(C[0,0]+C[1,1]+C[2,2]+2*(C[0,1]+C[0,2]+C[1,2]))/9:.1f} GPa")

    >>> # With cancellation support:
    >>> import threading
    >>> event = threading.Event()
    >>> C = compute_elastic_stiffness_tensor(atoms, epsilon=1e-2, cancellation_event=event)
    >>> # User can call event.set() to cancel
    """
    if atoms.calc is None:
        raise ValueError("Atoms object must have a calculator attached")

    # Validate epsilon
    if epsilon < 0.01:
        raise ValueError(
            f"Strain magnitude epsilon={epsilon:.0e} is too small. "
            f"Must be >= 0.01 (1%) to avoid numerical noise with fmax=0.005 relaxation."
        )

    # Default relaxation parameters
    if relax_kwargs is None:
        relax_kwargs = {
            'fmax': 0.005,
            'optimizer': 'FIRE',
            'max_steps': 100,  # Limited steps (structures already near equilibrium)
            'hydrostatic_strain': False  # Keep cell fixed during relaxation
        }

    # Set up strain deformations
    # Generate all combinations of strain tensor components (reference: calorine)
    # Uses asymmetric strains; least-squares fitting recovers symmetric C tensor
    deformations = []
    for i, j in combinations_with_replacement(range(9), r=2):
        for s1, s2 in product([-1, 1], repeat=2):
            S = np.zeros((3, 3))
            S.flat[i] = s1
            S.flat[j] = s2
            deformations.append(S)

    deformations = np.array(deformations) * epsilon

    # Compute reference energy of undeformed structure
    reference_energy = atoms.get_potential_energy()

    # Create optimizer using the calculator already attached to atoms
    optimizer = StructureOptimizer(atoms.calc)

    # Compute strain energies
    energies = []
    for idx, S in enumerate(deformations):
        # Check for cancellation request (cooperative cancellation)
        check_cancellation(cancellation_event, idx, len(deformations), "elastic tensor calculation")

        # Report progress to UI (if callback provided)
        if progress_callback:
            status_msg = format_progress_message(
                idx + 1,  # Display as 1-indexed for user
                len(deformations),
                "Computing deformation",
                show_percentage=True
            )
            progress_callback(idx, len(deformations), status_msg)

        # Create deformed structure
        deformed_structure = atoms.copy()
        deformed_structure.calc = atoms.calc

        # Apply strain to cell (ASE uses row vectors: right multiply)
        cell = deformed_structure.get_cell()
        cell = cell @ (np.eye(3) + S.T)
        deformed_structure.set_cell(cell, scale_atoms=True)

        # Relax atomic positions at constant cell
        try:
            relaxed = optimizer.relax(deformed_structure, **relax_kwargs)
            energy = relaxed.get_potential_energy()
            energies.append(energy - reference_energy)
        except Exception as e:
            raise RuntimeError(
                f"Relaxation failed for deformation {idx}/{len(deformations)}: {e}"
            )

    energies = np.array(energies)

    # Extract stiffness tensor (full rank 3x3x3x3)
    # Energy = 0.5 * V * C_ijkl * ε_ij * ε_kl
    # where V is volume and ε is strain tensor
    SS = np.einsum('nij,nkl->nijkl', deformations, deformations)
    M = SS.reshape(len(SS), -1)
    M *= 0.5

    # Solve linear system: M @ C = energies
    C, *_ = np.linalg.lstsq(M, energies, rcond=None)
    C = C.reshape(3, 3, 3, 3)

    # Normalize by volume to get stress units, then convert to GPa
    C /= (atoms.cell.volume * GPa)

    # Convert from full tensor to Voigt form (6x6)
    # Voigt indices: 11→0, 22→1, 33→2, 23→3, 13→4, 12→5 (reference: calorine)
    voigt_indices = np.array([1, 1, 2, 2, 3, 3, 2, 3, 3, 1, 1, 2]).reshape(-1, 2) - 1

    C_voigt = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            v1 = voigt_indices[i]
            v2 = voigt_indices[j]
            C_voigt[i, j] = C[v1[0], v1[1], v2[0], v2[1]]

    return C_voigt


def compute_elastic_moduli(C_voigt: np.ndarray) -> Dict[str, float]:
    """
    Compute elastic moduli from stiffness tensor using Voigt averaging.

    Voigt averaging assumes uniform strain throughout the material and
    provides an upper bound on polycrystalline elastic constants.

    Parameters
    ----------
    C_voigt
        Elastic stiffness tensor in Voigt form (6x6 matrix, units: GPa)

    Returns
    -------
    moduli : dict
        Dictionary containing:
        - 'bulk_modulus_GPa': Bulk modulus K (resistance to volume change)
        - 'shear_modulus_GPa': Shear modulus G (resistance to shear)
        - 'youngs_modulus_GPa': Young's modulus E (stiffness in tension)
        - 'poisson_ratio': Poisson's ratio ν (lateral strain ratio)

    Notes
    -----
    Voigt averaging formulas:
    - K = (C11 + C22 + C33 + 2(C12 + C13 + C23)) / 9
    - G = ((C11 + C22 + C33) - (C12 + C13 + C23) + 3(C44 + C55 + C66)) / 15
    - E = 9KG / (3K + G)
    - ν = (3K - 2G) / (6K + 2G)

    Examples
    --------
    >>> moduli = compute_elastic_moduli(C_voigt)
    >>> print(f"Bulk modulus: {moduli['bulk_modulus_GPa']:.1f} GPa")
    """
    # Extract diagonal and off-diagonal components
    C11, C22, C33 = C_voigt[0, 0], C_voigt[1, 1], C_voigt[2, 2]
    C12, C13, C23 = C_voigt[0, 1], C_voigt[0, 2], C_voigt[1, 2]
    C44, C55, C66 = C_voigt[3, 3], C_voigt[4, 4], C_voigt[5, 5]

    # Voigt bulk modulus
    K = (C11 + C22 + C33 + 2 * (C12 + C13 + C23)) / 9.0

    # Voigt shear modulus
    G = ((C11 + C22 + C33) - (C12 + C13 + C23) + 3 * (C44 + C55 + C66)) / 15.0

    # Young's modulus
    E = 9 * K * G / (3 * K + G)

    # Poisson's ratio
    nu = (3 * K - 2 * G) / (6 * K + 2 * G)

    return {
        'bulk_modulus_GPa': float(K),
        'shear_modulus_GPa': float(G),
        'youngs_modulus_GPa': float(E),
        'poisson_ratio': float(nu)
    }
