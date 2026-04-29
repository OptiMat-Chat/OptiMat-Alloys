"""
Special Quasirandom Structure (SQS) generation.

This module provides tools for generating SQS structures that mimic
random alloys by minimizing short-range order.
"""

from typing import Dict, Optional
from ase import Atoms
import numpy as np


class SQSGenerator:
    """
    Generates special quasirandom structures for alloys.

    SQS structures are designed to approximate a random alloy by
    minimizing correlations in neighbor shells (short-range order).

    Examples:
        >>> gen = SQSGenerator(iterations=1000000)
        >>> composition = {"Al": 24, "Cu": 8}
        >>> optimized, objective = gen.generate(supercell, composition)
    """

    def __init__(
        self,
        iterations: int = 10000000,
        shell_weights: Optional[Dict[int, float]] = None
    ):
        """
        Initialize SQS generator.

        Args:
            iterations: Number of Monte Carlo swap attempts (default: 10 million)
            shell_weights: Weight for each neighbor shell in optimization.
                          Default: {1: 1.0, 2: 0.5}
                          Shell 1 = nearest neighbors (most important)
                          Shell 2 = next-nearest neighbors
        """
        self.iterations = iterations
        self.shell_weights = shell_weights or {1: 1.0, 2: 0.5}

    def generate(
        self,
        supercell: Atoms,
        composition: Dict[str, int]
    ) -> tuple[Atoms, float]:
        """
        Generate SQS structure via optimization.

        Args:
            supercell: Base supercell structure (all atoms typically one element)
            composition: Target composition as {element: count} dict
                        e.g., {"Al": 24, "Cu": 8}

        Returns:
            Tuple of:
                - Optimized ASE Atoms object with SQS arrangement
                - Objective function value (lower is better)

        Notes:
            - Requires sqsgenerator package
            - For single-element systems, returns input unchanged
            - Lower objective values indicate better randomness
            - Falls back to random generation if sqsgenerator not available
        """
        if len(composition) == 1:
            # Single element, no SQS needed
            return supercell, 0.0

        try:
            from sqsgenerator import optimize, from_ase, to_ase
            from sqsgenerator.core import LogLevel
        except ImportError:
            print("WARNING: sqsgenerator not installed, falling back to random arrangement")
            print("Install with: pip install sqsgenerator")
            return self.generate_random(supercell, composition), None

        # Validate composition
        total_atoms = sum(composition.values())
        if total_atoms != len(supercell):
            raise ValueError(
                f"Composition total ({total_atoms}) doesn't match "
                f"supercell size ({len(supercell)})"
            )

        # sqsgenerator v0.5.3 API
        try:
            # Convert ASE atoms to sqsgenerator Structure
            sqs_structure = from_ase(supercell)

            # Build configuration dictionary
            # NOTE: shell_weights must have INTEGER keys, not strings!
            config = {
                "structure": {
                    "lattice": sqs_structure.lattice.tolist(),
                    "coords": sqs_structure.frac_coords.tolist(),
                    "species": sqs_structure.species,
                    "supercell": [1, 1, 1]  # Already a supercell, no further replication needed
                },
                "composition": composition,
                "shell_weights": self.shell_weights,  # Use integer keys directly
                "iterations": self.iterations
            }

            # Print SQS configuration
            print(f"\n{'='*60}")
            print(f"Starting SQS optimization")
            print(f"{'='*60}")
            print(f"Iterations: {self.iterations:,}")
            print(f"Composition: {composition}")
            print(f"Shell weights: {self.shell_weights}")
            print(f"Supercell atoms: {len(supercell)}")
            print(f"{'='*60}\n")

            # Run SQS optimization with info-level logging
            result_pack = optimize(config, level=LogLevel.info)

            # Extract best result
            best_result = result_pack.best()
            objective = float(best_result.objective)

            # Get structure (it's a method, not a property!)
            structure = best_result.structure()

            # Convert back to ASE Atoms
            optimized = to_ase(structure)

            # Print completion message
            print(f"\n{'='*60}")
            print(f"SQS optimization complete!")
            print(f"{'='*60}")
            print(f"Final objective value: {objective:.6f}")
            print(f"{'='*60}\n")

            return optimized, objective
        except Exception as e:
            print(f"WARNING: SQS optimization failed: {e}")
            print("Falling back to random arrangement")
            import traceback
            traceback.print_exc()
            return self.generate_random(supercell, composition), None

    def generate_random(
        self,
        supercell: Atoms,
        composition: Dict[str, int],
        seed: Optional[int] = None
    ) -> Atoms:
        """
        Generate random alloy arrangement (fast fallback).

        This is a simple random arrangement without SQS optimization.
        Useful for testing or when SQS is not available.

        Args:
            supercell: Base supercell structure
            composition: Target composition as {element: count} dict
            seed: Random seed for reproducibility (optional)

        Returns:
            ASE Atoms object with random arrangement

        Raises:
            ValueError: If composition doesn't match supercell size
        """
        total_atoms = sum(composition.values())
        if total_atoms != len(supercell):
            raise ValueError(
                f"Composition total ({total_atoms}) doesn't match "
                f"supercell size ({len(supercell)})"
            )

        if seed is not None:
            np.random.seed(seed)

        # Create list of symbols
        symbols = []
        for element, count in composition.items():
            symbols.extend([element] * count)

        # Shuffle randomly
        np.random.shuffle(symbols)

        # Set chemical symbols
        supercell.set_chemical_symbols(symbols)

        return supercell

    def generate_with_fallback(
        self,
        supercell: Atoms,
        composition: Dict[str, int],
        use_sqs: bool = True,
        random_seed: Optional[int] = None
    ) -> tuple[Atoms, Optional[float]]:
        """
        Generate SQS with automatic fallback to random on failure.

        Args:
            supercell: Base supercell structure
            composition: Target composition
            use_sqs: If True, attempt SQS optimization. If False, use random.
            random_seed: Seed for random generation (fallback only)

        Returns:
            Tuple of:
                - ASE Atoms object
                - Objective value (None if random fallback used)
        """
        if not use_sqs or len(composition) == 1:
            return self.generate_random(supercell, composition, random_seed), None

        try:
            return self.generate(supercell, composition)
        except Exception as e:
            print(f"WARNING: SQS optimization failed: {e}")
            print("Falling back to random arrangement")
            return self.generate_random(supercell, composition, random_seed), None
