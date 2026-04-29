"""
Caching system for reference data.

This module provides caching for computationally expensive reference
calculations (lattice constants, reference energies) to avoid
redundant recomputation.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple


class ReferenceDataCache:
    """
    Cache manager for reference lattice constants and energies.

    Handles loading, saving, and validating cached reference data
    to avoid recomputing expensive relaxations.
    """

    def __init__(
        self,
        calculator: str = "orb-v3-direct-20-omat",
        base_dir: str = "data/reference"
    ):
        """
        Initialize cache with calculator-specific file paths.

        Args:
            calculator: Calculator name (e.g., 'orb-v3-direct-20-omat')
            base_dir: Base directory for reference data files (default: data/reference)

        Examples:
            >>> cache = ReferenceDataCache(calculator="orb-v3-direct-20-omat")
            >>> cache.is_available()  # doctest: +SKIP
            True
        """
        self.calculator = calculator
        self.base_dir = Path(base_dir)

        # Generate versioned filenames
        safe_calc_name = calculator.replace("-", "_")
        self.lattice_json_path = self.base_dir / f"lattice_constants_{safe_calc_name}.json"
        self.energy_json_path = self.base_dir / f"energies_per_atom_{safe_calc_name}.json"

        self._lattice_data: Optional[Dict] = None
        self._energy_data: Optional[Dict] = None

    def is_available(self) -> bool:
        """
        Check if cached reference data exists and validates.

        Returns:
            True if both files exist AND metadata matches expected calculator

        Examples:
            >>> cache = ReferenceDataCache(calculator="orb-v3-direct-20-omat")
            >>> cache.is_available()  # doctest: +SKIP
            True
        """
        if not (self.lattice_json_path.exists() and self.energy_json_path.exists()):
            return False

        # Validate metadata matches calculator
        return self._validate_metadata()

    def _validate_metadata(self) -> bool:
        """
        Validate that JSON metadata matches expected calculator.

        Returns:
            True if metadata is valid and matches calculator

        Notes:
            Checks energy file for metadata header. If missing or mismatched,
            returns False to trigger recomputation.
        """
        try:
            with self.energy_json_path.open("r") as f:
                data = json.load(f)

            metadata = data.get("_metadata", {})
            stored_calc = metadata.get("calculator", "")

            if stored_calc != self.calculator:
                print(f"Warning: Calculator mismatch in {self.energy_json_path}")
                print(f"  Expected: {self.calculator}")
                print(f"  Found: {stored_calc}")
                return False

            return True
        except Exception as e:
            print(f"Metadata validation failed: {e}")
            return False

    def load_lattice_constants(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Load lattice constants from cache.

        Returns:
            Dictionary mapping element -> structure -> lattice constant

        Raises:
            FileNotFoundError: If cache file doesn't exist

        Examples:
            >>> cache = ReferenceDataCache(calculator="orb-v3-direct-20-omat")
            >>> data = cache.load_lattice_constants()  # doctest: +SKIP
            >>> data["Cu"]["fcc"]  # doctest: +SKIP
            3.615
        """
        if self._lattice_data is None:
            with self.lattice_json_path.open("r") as f:
                raw_data = json.load(f)

            # Filter out metadata
            self._lattice_data = {k: v for k, v in raw_data.items() if k != "_metadata"}
        return self._lattice_data

    def load_energies(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Load reference energies from cache.

        Returns:
            Dictionary mapping element -> structure -> energy per atom

        Raises:
            FileNotFoundError: If cache file doesn't exist

        Examples:
            >>> cache = ReferenceDataCache(calculator="orb-v3-direct-20-omat")
            >>> data = cache.load_energies()  # doctest: +SKIP
            >>> data["Cu"]["fcc"]  # doctest: +SKIP
            -3.721
        """
        if self._energy_data is None:
            with self.energy_json_path.open("r") as f:
                raw_data = json.load(f)

            # Filter out metadata
            self._energy_data = {k: v for k, v in raw_data.items() if k != "_metadata"}
        return self._energy_data

    def load_both(self) -> Tuple[Dict, Dict]:
        """
        Load both lattice constants and energies.

        Returns:
            Tuple of (lattice_data, energy_data)

        Examples:
            >>> cache = ReferenceDataCache()
            >>> lattices, energies = cache.load_both()  # doctest: +SKIP
        """
        return self.load_lattice_constants(), self.load_energies()

    def save_lattice_constants(
        self,
        data: Dict[str, Dict[str, Optional[float]]]
    ) -> None:
        """
        Save lattice constants to cache.

        Args:
            data: Dictionary mapping element -> structure -> lattice constant

        Examples:
            >>> cache = ReferenceDataCache()
            >>> data = {"Cu": {"fcc": 3.615}}
            >>> cache.save_lattice_constants(data)  # doctest: +SKIP
        """
        with self.lattice_json_path.open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        self._lattice_data = data

    def save_energies(
        self,
        data: Dict[str, Dict[str, Optional[float]]]
    ) -> None:
        """
        Save reference energies to cache.

        Args:
            data: Dictionary mapping element -> structure -> energy

        Examples:
            >>> cache = ReferenceDataCache()
            >>> data = {"Cu": {"fcc": -3.721}}
            >>> cache.save_energies(data)  # doctest: +SKIP
        """
        with self.energy_json_path.open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        self._energy_data = data

    def save_both(
        self,
        lattice_data: Dict[str, Dict[str, Optional[float]]],
        energy_data: Dict[str, Dict[str, Optional[float]]]
    ) -> None:
        """
        Save both lattice constants and energies.

        Args:
            lattice_data: Lattice constant data
            energy_data: Energy data

        Examples:
            >>> cache = ReferenceDataCache()
            >>> lattices = {"Cu": {"fcc": 3.615}}
            >>> energies = {"Cu": {"fcc": -3.721}}
            >>> cache.save_both(lattices, energies)  # doctest: +SKIP
        """
        self.save_lattice_constants(lattice_data)
        self.save_energies(energy_data)

    def save_with_metadata(
        self,
        lattice_data: Dict[str, Dict[str, Optional[float]]],
        energy_data: Dict[str, Dict[str, Optional[float]]],
        fmax: float,
        optimizer: str,
        hydrostatic: bool
    ) -> None:
        """
        Save reference data with metadata header.

        Args:
            lattice_data: Lattice constant data
            energy_data: Energy data
            fmax: Force convergence criterion used
            optimizer: Optimizer used (FIRE/LBFGS)
            hydrostatic: Whether hydrostatic cell relaxation was used

        Examples:
            >>> cache = ReferenceDataCache(calculator="orb-v3-direct-20-omat")
            >>> cache.save_with_metadata(lattices, energies, 0.005, "FIRE", True)  # doctest: +SKIP
        """
        from datetime import datetime

        metadata = {
            "_metadata": {
                "calculator": self.calculator,
                "fmax": fmax,
                "optimizer": optimizer,
                "hydrostatic_cell_relaxation": hydrostatic,
                "generated_at": datetime.now().isoformat(),
                "num_elements": len(lattice_data),
                "structures": list(next(iter(lattice_data.values())).keys()) if lattice_data else []
            }
        }

        # Merge metadata with data (metadata first for visibility)
        lattice_output = {**metadata, **lattice_data}
        energy_output = {**metadata, **energy_data}

        with self.lattice_json_path.open("w") as f:
            json.dump(lattice_output, f, indent=2, sort_keys=True)

        with self.energy_json_path.open("w") as f:
            json.dump(energy_output, f, indent=2, sort_keys=True)

        self._lattice_data = lattice_data
        self._energy_data = energy_data

        print(f"✅ Reference data saved to:")
        print(f"   {self.lattice_json_path}")
        print(f"   {self.energy_json_path}")

    def clear_memory(self) -> None:
        """
        Clear in-memory cache (but not files).

        Examples:
            >>> cache = ReferenceDataCache()
            >>> cache.clear_memory()
        """
        self._lattice_data = None
        self._energy_data = None

    def get_paths(self) -> Tuple[str, str]:
        """
        Get absolute paths to cache files.

        Returns:
            Tuple of (lattice_path, energy_path) as strings for compatibility

        Examples:
            >>> cache = ReferenceDataCache()
            >>> lat_path, eng_path = cache.get_paths()
            >>> lat_path.endswith("lattice_constants.json")
            True
        """
        return (
            str(self.lattice_json_path.resolve()),
            str(self.energy_json_path.resolve())
        )


# Global cache instances (one per calculator)
_reference_caches: Dict[str, ReferenceDataCache] = {}


def get_reference_cache(
    calculator: str = "orb-v3-direct-20-omat",
    base_dir: str = "data/reference"
) -> ReferenceDataCache:
    """
    Get or create reference data cache for a specific calculator.

    Args:
        calculator: Calculator name (e.g., 'orb-v3-direct-20-omat')
        base_dir: Base directory for reference data files (default: data/reference)

    Returns:
        ReferenceDataCache instance for the specified calculator

    Examples:
        >>> cache = get_reference_cache(calculator="orb-v3-direct-20-omat")
        >>> isinstance(cache, ReferenceDataCache)
        True
    """
    global _reference_caches
    if calculator not in _reference_caches:
        _reference_caches[calculator] = ReferenceDataCache(calculator, base_dir)
    return _reference_caches[calculator]


def clear_reference_cache() -> None:
    """
    Clear all global reference caches.

    Examples:
        >>> clear_reference_cache()
    """
    global _reference_caches
    for cache in _reference_caches.values():
        cache.clear_memory()
