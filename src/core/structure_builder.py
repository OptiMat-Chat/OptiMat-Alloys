"""
Structure building and alloy composition optimization.

This module provides tools for constructing alloy supercells with
optimal replication factors to match target compositions and atom counts.
"""

from typing import List, Literal, Tuple
from ase import Atoms
from ase.build import bulk
import numpy as np
import math
from pathlib import Path


def estimate_alloy_lattice_constant_radii(
    structure: Literal['sc', 'bcc', 'fcc', 'hcp', 'diamond'],
    elements: List[str],
    fractions: List[float]
) -> float:
    """
    Estimate alloy lattice constant using hard-sphere atomic radii.

    Uses atomic radii and crystal structure geometry to estimate
    the lattice parameter of an alloy via weighted average (rule of mixtures).

    NOTE: This is a radii-based estimate. For better accuracy, use
    estimate_alloy_lattice_constant_vegard() which uses relaxed
    lattice constants of pure elements.

    Args:
        structure: Crystal structure type
        elements: List of element symbols (e.g., ['Al', 'Cu'])
        fractions: Atomic fractions (must sum to 1)

    Returns:
        Estimated lattice constant in Angstroms

    Raises:
        ValueError: If inputs are invalid or element not in database

    Examples:
        >>> estimate_alloy_lattice_constant_radii('fcc', ['Al', 'Cu'], [0.7, 0.3])
        3.95  # Approximate value in Angstroms
    """
    valid_structures = {"sc", "bcc", "fcc", "hcp", "diamond"}
    if structure not in valid_structures:
        raise ValueError("Structure must be one of: " + ", ".join(valid_structures))

    if len(elements) != len(fractions):
        raise ValueError("The length of elements and fractions must be equal.")

    total_fraction = sum(fractions)
    if not np.isclose(total_fraction, 1.0, rtol=1e-3):
        raise ValueError(f"The sum of the atomic fractions must equal 1. Got sum = {total_fraction}")

    # Load atomic radii from database (supports all 119 elements)
    from .radii_database import RadiiDatabase
    db_path = Path(__file__).parent.parent.parent / "data" / "radii" / "atomic_radii.json"
    db = RadiiDatabase(database_file=str(db_path))

    for elem, frac in zip(elements, fractions):
        if frac < 0 or frac > 1:
            raise ValueError(f"Atomic fraction for {elem} must be between 0 and 1. Got {frac}")

    # Get atomic radii for all elements (auto mode: prefer metallic, fall back to covalent)
    atomic_radii = {}
    for elem in elements:
        try:
            atomic_radii[elem] = db.get_radius(elem, radius_type="auto")
        except ValueError as e:
            raise ValueError(f"Atomic radius for element {elem} not found in database: {e}")

    # Effective atomic radius via rule of mixtures (weighted average)
    effective_r = sum(frac * atomic_radii[elem] for elem, frac in zip(elements, fractions))

    # Estimate lattice constant using hard-sphere relations
    if structure == "sc":
        a = 2 * effective_r
    elif structure == "bcc":
        a = (4 * effective_r) / np.sqrt(3)
    elif structure == "fcc":
        a = 2 * np.sqrt(2) * effective_r
    elif structure == "diamond":
        a = (8 * effective_r) / np.sqrt(3)
    elif structure == "hcp":
        a = 2 * effective_r
    else:
        raise ValueError("Unsupported structure type.")

    return float(a)


def estimate_alloy_lattice_constant_vegard(
    structure: Literal['sc', 'bcc', 'fcc', 'hcp', 'diamond'],
    elements: List[str],
    fractions: List[float],
    calculator: str = "orb-v3-direct-20-omat",
    bowing_parameter: float = 0.0
) -> float:
    """
    Estimate alloy lattice constant using Vegard's law with relaxed data.

    Uses relaxed lattice constants of pure elements from reference data
    and applies Vegard's law (linear mixing rule) to estimate the alloy
    lattice constant. This is more accurate than radii-based estimates.

    Args:
        structure: Crystal structure type
        elements: List of element symbols (e.g., ['Cu', 'Ag'])
        fractions: Atomic fractions (must sum to 1)
        calculator: Calculator name for loading reference data
        bowing_parameter: Non-linearity correction (default 0 = linear Vegard)

    Returns:
        Estimated lattice constant in Angstroms

    Raises:
        ValueError: If inputs invalid or reference data missing
        FileNotFoundError: If reference data files don't exist

    Notes:
        Formula:
          Linear Vegard's law: a_alloy = Σ(x_i × a_i)
          With bowing: a_alloy = Σ(x_i × a_i) - b × Π(x_i)

        The bowing parameter accounts for non-ideal mixing. For most
        metallic alloys, b ≈ 0 (ideal mixing). For semiconductors,
        b can be 0.1-0.5 Å.

    Examples:
        >>> estimate_alloy_lattice_constant_vegard('fcc', ['Cu', 'Ag'], [0.5, 0.5])
        3.88  # Weighted average of Cu (3.615) and Ag (4.144)

        >>> # With bowing parameter for non-ideal solution
        >>> estimate_alloy_lattice_constant_vegard(
        ...     'fcc', ['Cu', 'Ag'], [0.5, 0.5], bowing_parameter=0.1
        ... )
        3.86  # Slightly smaller due to negative bowing
    """
    # Validate inputs
    valid_structures = {"sc", "bcc", "fcc", "hcp", "diamond"}
    if structure not in valid_structures:
        raise ValueError("Structure must be one of: " + ", ".join(valid_structures))

    if len(elements) != len(fractions):
        raise ValueError("The length of elements and fractions must be equal.")

    total_fraction = sum(fractions)
    if not np.isclose(total_fraction, 1.0, rtol=1e-3):
        raise ValueError(f"The sum of the atomic fractions must equal 1. Got sum = {total_fraction}")

    for elem, frac in zip(elements, fractions):
        if frac < 0 or frac > 1:
            raise ValueError(f"Atomic fraction for {elem} must be between 0 and 1. Got {frac}")

    # Load reference lattice constants
    from ..storage.cache import get_reference_cache

    cache = get_reference_cache(calculator)

    try:
        lattice_data = cache.load_lattice_constants()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference lattice constants not found for calculator: {calculator}\n"
            f"Run precompute_and_save() first to generate reference data."
        )

    # Look up pure element lattice constants
    a_values = []
    for elem in elements:
        if elem not in lattice_data:
            raise ValueError(
                f"No lattice constant data for element {elem}.\n"
                f"Element may not be supported by calculator: {calculator}"
            )

        if structure not in lattice_data[elem]:
            raise ValueError(
                f"No lattice constant for {elem} in structure {structure}.\n"
                f"Available structures: {list(lattice_data[elem].keys())}"
            )

        a_i = lattice_data[elem][structure]
        if a_i is None:
            raise ValueError(
                f"Element {elem} failed to relax in structure {structure}.\n"
                f"Cannot use Vegard's law. Try radii-based estimate instead."
            )

        a_values.append(float(a_i))

    # Apply Vegard's law (linear mixing)
    a_alloy = sum(x_i * a_i for x_i, a_i in zip(fractions, a_values))

    # Optional: Add bowing correction for non-ideal solutions
    if bowing_parameter != 0.0:
        # Bowing term: product of all fractions
        product_term = 1.0
        for x_i in fractions:
            product_term *= x_i

        a_alloy -= bowing_parameter * product_term

    return float(a_alloy)


# Backward compatibility: keep old function name
def estimate_alloy_lattice_constant(
    structure: Literal['sc', 'bcc', 'fcc', 'hcp', 'diamond'],
    elements: List[str],
    fractions: List[float]
) -> float:
    """
    Estimate alloy lattice constant (backward compatibility wrapper).

    This function maintains backward compatibility with existing code.
    It calls estimate_alloy_lattice_constant_radii() under the hood.

    For new code, prefer:
    - estimate_alloy_lattice_constant_vegard() for accuracy (uses relaxed data)
    - estimate_alloy_lattice_constant_radii() for initial guesses

    Args:
        structure: Crystal structure type
        elements: List of element symbols
        fractions: Atomic fractions (must sum to 1)

    Returns:
        Estimated lattice constant in Angstroms
    """
    return estimate_alloy_lattice_constant_radii(structure, elements, fractions)


def lattice_constant_from_atomic_volume(
    structure: Literal['sc', 'bcc', 'fcc', 'hcp', 'diamond'],
    volume: float
) -> float:
    """
    Compute cubic lattice constant from per-atom volume.

    Args:
        structure: Crystal structure type
        volume: Atomic volume in Angstrom^3

    Returns:
        Computed lattice constant in Angstroms

    Raises:
        ValueError: If volume is non-positive

    Notes:
        Assumptions/relations:
          - sc:      V_atom = a^3
          - bcc:     V_atom = a^3 / 2
          - fcc:     V_atom = a^3 / 4
          - diamond: V_atom = a^3 / 8
          - hcp:     V_atom = (√3/4) a^2 c, with ideal c/a = √(8/3)
                     => V_atom = (√2/2) a^3
    """
    if volume <= 0:
        raise ValueError("volume must be positive (Å^3).")

    multipliers = {
        'sc': 1.0,
        'bcc': 2.0,
        'fcc': 4.0,
        'diamond': 8.0,
        'hcp': np.sqrt(2.0),  # assumes ideal c/a
    }

    m = multipliers[structure]
    return float((m * volume) ** (1.0 / 3.0))


class AlloyBuilder:
    """
    Builds alloy supercells with optimized composition and geometry.

    This class handles the creation of alloy supercells by:
    1. Computing optimal replication factors
    2. Distributing atoms to match target composition
    3. Building the base supercell structure

    Examples:
        >>> builder = AlloyBuilder("fcc", ["Al", "Cu"], [0.75, 0.25])
        >>> rfs, counts = builder.compute_replication_factors(512)
        >>> supercell = builder.build_supercell(4.0, rfs)
    """

    def __init__(
        self,
        structure: Literal['sc', 'bcc', 'fcc', 'hcp', 'diamond'],
        elements: List[str],
        fractions: List[float]
    ):
        """
        Initialize alloy builder.

        Args:
            structure: Crystal structure type
            elements: List of element symbols
            fractions: Atomic fractions (must sum to 1)

        Raises:
            ValueError: If inputs are invalid
        """
        self.structure = structure
        self.elements = elements
        self.fractions = np.array(fractions, dtype=float)
        self._validate()

    def _validate(self):
        """Validate builder inputs."""
        valid = {"sc", "bcc", "fcc", "hcp", "diamond"}
        if self.structure not in valid:
            raise ValueError(f"Invalid structure: {self.structure}. Must be one of {valid}")

        if len(self.elements) != len(self.fractions):
            raise ValueError("Elements and fractions length mismatch")

        if not np.isclose(sum(self.fractions), 1.0, rtol=1e-3):
            raise ValueError(f"Fractions must sum to 1, got {sum(self.fractions)}")

        for elem, frac in zip(self.elements, self.fractions):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction for {elem} must be in [0, 1], got {frac}")

    def compute_replication_factors(
        self,
        target_num_atoms: int
    ) -> Tuple[Tuple[int, int, int], List[int]]:
        """
        Compute optimal replication factors and atom counts.

        This method finds replication factors (nx, ny, nz) that:
        1. Match the target composition as closely as possible
        2. Produce approximately the target number of atoms
        3. Create a roughly cubic supercell (minimize aspect ratio variation)

        Args:
            target_num_atoms: Desired total number of atoms

        Returns:
            Tuple of:
                - (nx, ny, nz): Replication factors
                - [count1, count2, ...]: Atom count for each element

        Notes:
            Uses a brute-force optimization over reasonable replication
            factor ranges to minimize a composite loss function.
        """
        unit_cell_counts = {'sc': 1, 'bcc': 2, 'fcc': 4, 'hcp': 2, 'diamond': 8}
        unit_cell_count = unit_cell_counts[self.structure]

        # Calculate maximum replication factor to search
        rf_max = int(np.cbrt(target_num_atoms / unit_cell_count)) + 1

        best_rfs = (1, 1, 1)
        best_counts = np.ones(len(self.fractions), dtype=int)
        best_loss = float('inf')

        for rf_a in range(1, rf_max + 1):
            for rf_b in range(1, rf_max + 1):
                for rf_c in range(1, rf_max + 1):
                    num_atoms = unit_cell_count * rf_a * rf_b * rf_c
                    counts = self._distribute_atoms(num_atoms)
                    loss = self._compute_loss(counts, num_atoms, target_num_atoms,
                                             rf_a, rf_b, rf_c)

                    if loss < best_loss:
                        best_loss = loss
                        best_rfs = (rf_a, rf_b, rf_c)
                        best_counts = counts

        return best_rfs, best_counts.tolist()

    def _distribute_atoms(self, num_atoms: int) -> np.ndarray:
        """
        Distribute atoms to match target fractions.

        Uses integer rounding with remainder distribution to ensure
        exact atom count and best match to target fractions.

        Args:
            num_atoms: Total number of atoms to distribute

        Returns:
            Array of atom counts per element
        """
        counts = np.array([int(f * num_atoms) for f in self.fractions])
        remainders = [f * num_atoms - int(f * num_atoms) for f in self.fractions]
        diff = num_atoms - counts.sum()

        # Distribute leftover atoms to species with highest fractional remainder
        for _ in range(diff):
            i = np.argmax(remainders)
            counts[i] += 1
            remainders[i] = -1  # Mark as processed

        return counts

    def _compute_loss(
        self,
        counts: np.ndarray,
        num_atoms: int,
        target_num_atoms: int,
        rf_a: int,
        rf_b: int,
        rf_c: int
    ) -> float:
        """
        Compute optimization loss for given configuration.

        Loss has three components:
        1. Composition loss: deviation from target fractions
        2. Aspect ratio loss: deviation from cubic shape
        3. Size loss: deviation from target atom count

        Args:
            counts: Atom counts per element
            num_atoms: Total atoms in this configuration
            target_num_atoms: Desired atom count
            rf_a, rf_b, rf_c: Replication factors

        Returns:
            Total loss value (lower is better)
        """
        # Composition loss
        actual_fractions = counts / num_atoms
        composition_loss = np.mean((self.fractions - actual_fractions)**2)

        # Aspect ratio loss (prefer cubic supercells)
        rf_mean = np.mean([rf_a, rf_b, rf_c])
        rf_loss = np.mean([
            (rf_a / rf_mean - 1)**2,
            (rf_b / rf_mean - 1)**2,
            (rf_c / rf_mean - 1)**2
        ])

        # Size loss
        num_atoms_loss = (num_atoms / target_num_atoms - 1)**2

        return composition_loss + rf_loss + num_atoms_loss

    def build_supercell(
        self,
        lattice_constant: float,
        replication_factors: Tuple[int, int, int]
    ) -> Atoms:
        """
        Build supercell with given lattice constant and replication.

        Args:
            lattice_constant: Lattice parameter in Angstroms
            replication_factors: (nx, ny, nz) replication tuple

        Returns:
            ASE Atoms object with all atoms set to first element
            (chemical symbols should be modified afterward for alloys)

        Notes:
            - For cubic structures (sc, fcc, bcc, diamond), creates cubic unit cell
            - For HCP, uses ideal c/a ratio
        """
        # For many structures, a cubic unit cell is preferred
        cubic = self.structure in ["sc", "fcc", "bcc", "diamond"]

        if self.structure == "hcp":
            # Ideal c/a for hcp
            c = math.sqrt(8.0/3.0) * lattice_constant
            base = bulk(self.elements[0], "hcp", a=lattice_constant, c=c)
        else:
            base = bulk(self.elements[0], self.structure, a=lattice_constant, cubic=cubic)

        # Replicate base structure to make supercell
        supercell = base * replication_factors

        return supercell


# Legacy function name for backward compatibility
def compute_replication_factors(
    structure: str,
    target_fractions: List[float],
    target_num_atoms: int
) -> Tuple[Tuple[int, int, int], List[int]]:
    """
    Compute replication factors (legacy function).

    This function exists for backward compatibility. New code should
    use the AlloyBuilder class directly.

    Args:
        structure: Crystal structure type
        target_fractions: Atomic fractions
        target_num_atoms: Target atom count

    Returns:
        Tuple of (replication_factors, atom_counts)
    """
    # Create dummy element list (actual elements don't matter for this calculation)
    elements = [f"El{i}" for i in range(len(target_fractions))]
    builder = AlloyBuilder(structure, elements, target_fractions)
    return builder.compute_replication_factors(target_num_atoms)
