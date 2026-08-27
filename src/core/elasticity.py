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

# Valid range for the finite-difference strain magnitude, as a FRACTION (0.01 = 1%).
# Single source of truth — src/tools/elastic_properties.py imports these so the tool
# guard, the parameter annotation and this core check cannot drift apart.
#   Lower bound: below 1% the strain energies are comparable to the noise floor of an
#   fmax=0.005 eV/A relaxation, so the least-squares fit degrades.
#   Upper bound: the energy-strain fit truncates at 4th order, so beyond ~2% anharmonic
#   contributions start contaminating the harmonic constants.
EPSILON_MIN = 0.01
EPSILON_MAX = 0.02


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
        Must lie in [EPSILON_MIN, EPSILON_MAX] = [0.01, 0.02]; see those constants.
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

    # Validate epsilon. This is the backstop for direct callers (batch scripts);
    # calculate_elastic_properties applies the same bounds earlier with a friendlier
    # message so bad LLM calls fail before a calculator is loaded.
    if not (EPSILON_MIN <= epsilon <= EPSILON_MAX):
        raise ValueError(
            f"Strain magnitude epsilon={epsilon:g} is outside the valid range "
            f"[{EPSILON_MIN}, {EPSILON_MAX}]. epsilon is a FRACTION, not a percentage "
            f"({EPSILON_MIN} = {EPSILON_MIN * 100:g}%). Below {EPSILON_MIN} the strain "
            f"energies approach the noise floor of an fmax=0.005 eV/A relaxation; above "
            f"{EPSILON_MAX} anharmonic terms contaminate the harmonic fit."
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


def _moduli_from_K_G(K: float, G: float) -> Dict[str, float]:
    """Derive Young's modulus and Poisson's ratio from bulk and shear moduli."""
    denom_E = 3.0 * K + G
    denom_nu = 6.0 * K + 2.0 * G
    E = 9.0 * K * G / denom_E if denom_E != 0 else float('nan')
    nu = (3.0 * K - 2.0 * G) / denom_nu if denom_nu != 0 else float('nan')
    return {'youngs': float(E), 'poisson': float(nu)}


def compute_elastic_moduli(C_voigt: np.ndarray) -> Dict[str, Any]:
    """
    Compute polycrystalline elastic moduli from the stiffness tensor.

    Returns all three standard averaging conventions:

    - **Voigt** — assumes uniform strain; an upper bound.
    - **Reuss** — assumes uniform stress; a lower bound. Requires inverting C.
    - **Hill** — the arithmetic mean of Voigt and Reuss. This is the standard
      estimate for an untextured polycrystal, which is what a random supercell
      represents and what experimental K and G are measured on. Prefer it for
      reporting, and for any Pugh-ratio comparison (Pugh's 1.75 threshold was
      calibrated on experimental polycrystal moduli).

    Parameters
    ----------
    C_voigt
        Elastic stiffness tensor in Voigt form (6x6 matrix, units: GPa)

    Returns
    -------
    moduli : dict
        Voigt values under both the unqualified legacy keys
        (``bulk_modulus_GPa`` etc.) and explicit ``*_voigt_GPa`` keys, plus
        ``*_reuss_GPa`` and ``*_hill_GPa``. The unqualified keys are Voigt for
        backward compatibility — rows written before this function grew the
        Reuss/Hill keys carry Voigt values under those same names, so their
        meaning must not change.

        Reuss and Hill entries are ``None`` when C is singular (an invalid
        tensor), so callers must handle that rather than assume floats.
        ``reuss_available`` is a bool flag for convenient branching.

    Notes
    -----
    Voigt averaging:
    - K_V = (C11 + C22 + C33 + 2(C12 + C13 + C23)) / 9
    - G_V = ((C11 + C22 + C33) - (C12 + C13 + C23) + 3(C44 + C55 + C66)) / 15

    Reuss averaging, from the compliance S = C^-1:
    - K_R = 1 / (S11 + S22 + S33 + 2(S12 + S13 + S23))
    - G_R = 15 / (4(S11 + S22 + S33) - 4(S12 + S13 + S23) + 3(S44 + S55 + S66))

    Then E = 9KG / (3K + G) and nu = (3K - 2G) / (6K + 2G) for each convention.

    Examples
    --------
    >>> moduli = compute_elastic_moduli(C_voigt)
    >>> print(f"Bulk modulus (Hill): {moduli['bulk_modulus_hill_GPa']:.1f} GPa")
    """
    # --- Voigt (uniform strain, upper bound) ---
    C11, C22, C33 = C_voigt[0, 0], C_voigt[1, 1], C_voigt[2, 2]
    C12, C13, C23 = C_voigt[0, 1], C_voigt[0, 2], C_voigt[1, 2]
    C44, C55, C66 = C_voigt[3, 3], C_voigt[4, 4], C_voigt[5, 5]

    K_V = (C11 + C22 + C33 + 2 * (C12 + C13 + C23)) / 9.0
    G_V = ((C11 + C22 + C33) - (C12 + C13 + C23) + 3 * (C44 + C55 + C66)) / 15.0
    voigt = _moduli_from_K_G(K_V, G_V)

    moduli: Dict[str, Any] = {
        # Legacy unqualified keys — Voigt, unchanged meaning for backward compatibility
        'bulk_modulus_GPa': float(K_V),
        'shear_modulus_GPa': float(G_V),
        'youngs_modulus_GPa': voigt['youngs'],
        'poisson_ratio': voigt['poisson'],
        # Explicit Voigt
        'bulk_modulus_voigt_GPa': float(K_V),
        'shear_modulus_voigt_GPa': float(G_V),
        'youngs_modulus_voigt_GPa': voigt['youngs'],
        'poisson_ratio_voigt': voigt['poisson'],
    }

    # --- Reuss (uniform stress, lower bound) — needs the compliance tensor ---
    # Reuss is only physically meaningful for a positive-definite C. A tensor that
    # fails the Born stability criterion is still invertible, but S = C^-1 then
    # produces wild or negative K_R/G_R, which would poison the Hill average and
    # the headline moduli. Refuse Reuss/Hill in that case and let callers fall
    # back to Voigt, which stays a well-defined arithmetic combination.
    C_arr = np.asarray(C_voigt, dtype=float)
    C_sym = 0.5 * (C_arr + C_arr.T)
    try:
        positive_definite = bool(np.min(np.linalg.eigvalsh(C_sym)) > 0)
    except np.linalg.LinAlgError:
        positive_definite = False

    S = None
    if positive_definite:
        try:
            S = np.linalg.inv(C_arr)
        except np.linalg.LinAlgError:
            S = None

    if S is None or not np.all(np.isfinite(S)):
        moduli.update({
            'reuss_available': False,
            'bulk_modulus_reuss_GPa': None, 'shear_modulus_reuss_GPa': None,
            'youngs_modulus_reuss_GPa': None, 'poisson_ratio_reuss': None,
            'bulk_modulus_hill_GPa': None, 'shear_modulus_hill_GPa': None,
            'youngs_modulus_hill_GPa': None, 'poisson_ratio_hill': None,
        })
        return moduli

    S_diag_normal = S[0, 0] + S[1, 1] + S[2, 2]
    S_off_normal = S[0, 1] + S[0, 2] + S[1, 2]
    S_shear = S[3, 3] + S[4, 4] + S[5, 5]

    denom_K_R = S_diag_normal + 2 * S_off_normal
    denom_G_R = 4 * S_diag_normal - 4 * S_off_normal + 3 * S_shear

    if denom_K_R == 0 or denom_G_R == 0:
        moduli.update({
            'reuss_available': False,
            'bulk_modulus_reuss_GPa': None, 'shear_modulus_reuss_GPa': None,
            'youngs_modulus_reuss_GPa': None, 'poisson_ratio_reuss': None,
            'bulk_modulus_hill_GPa': None, 'shear_modulus_hill_GPa': None,
            'youngs_modulus_hill_GPa': None, 'poisson_ratio_hill': None,
        })
        return moduli

    K_R = 1.0 / denom_K_R
    G_R = 15.0 / denom_G_R

    # Defensive: a positive-definite C should always give positive bounds. If it
    # somehow does not, treat Reuss as unavailable rather than emitting nonsense.
    if not (np.isfinite(K_R) and np.isfinite(G_R) and K_R > 0 and G_R > 0):
        moduli.update({
            'reuss_available': False,
            'bulk_modulus_reuss_GPa': None, 'shear_modulus_reuss_GPa': None,
            'youngs_modulus_reuss_GPa': None, 'poisson_ratio_reuss': None,
            'bulk_modulus_hill_GPa': None, 'shear_modulus_hill_GPa': None,
            'youngs_modulus_hill_GPa': None, 'poisson_ratio_hill': None,
        })
        return moduli

    reuss = _moduli_from_K_G(K_R, G_R)

    # --- Hill (arithmetic mean of the two bounds) ---
    K_H = 0.5 * (K_V + K_R)
    G_H = 0.5 * (G_V + G_R)
    hill = _moduli_from_K_G(K_H, G_H)

    moduli.update({
        'reuss_available': True,
        'bulk_modulus_reuss_GPa': float(K_R),
        'shear_modulus_reuss_GPa': float(G_R),
        'youngs_modulus_reuss_GPa': reuss['youngs'],
        'poisson_ratio_reuss': reuss['poisson'],
        'bulk_modulus_hill_GPa': float(K_H),
        'shear_modulus_hill_GPa': float(G_H),
        'youngs_modulus_hill_GPa': hill['youngs'],
        'poisson_ratio_hill': hill['poisson'],
    })
    return moduli
