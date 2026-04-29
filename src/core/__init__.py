"""Core business logic modules (framework-independent)"""

from .calculators import CalculatorManager, load_calculator
from .structure_builder import (
    AlloyBuilder,
    estimate_alloy_lattice_constant,
    lattice_constant_from_atomic_volume,
    compute_replication_factors
)
from .optimization import StructureOptimizer, relax_atoms
from .sqs import SQSGenerator
from .analysis import structural_analysis, compute_density, compute_coordination_rdf
from .reference_data import (
    initial_cell,
    extract_lattice_constant_a,
    precompute_and_save,
    load_reference_energies,
    ReferenceMode
)
from .formation_energy import formation_energy_per_atom

__all__ = [
    "CalculatorManager",
    "load_calculator",
    "AlloyBuilder",
    "estimate_alloy_lattice_constant",
    "lattice_constant_from_atomic_volume",
    "compute_replication_factors",
    "StructureOptimizer",
    "relax_atoms",
    "SQSGenerator",
    "structural_analysis",
    "compute_density",
    "compute_coordination_rdf",
    "initial_cell",
    "extract_lattice_constant_a",
    "precompute_and_save",
    "load_reference_energies",
    "ReferenceMode",
    "formation_energy_per_atom",
]
