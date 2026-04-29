"""
Atomic radii database management.

This module provides a unified interface for atomic radii from multiple sources,
preferring metallic radii for metals and covalent radii for other elements.
"""

import json
from pathlib import Path
from typing import Dict, Literal, Optional
from ase.data import covalent_radii, atomic_numbers, chemical_symbols, atomic_masses


class RadiiDatabase:
    """
    Manage atomic radii from multiple sources.

    Provides unified access to atomic radii, preferring metallic radii
    for metals (when available) and falling back to covalent radii
    for other elements.

    Examples:
        >>> db = RadiiDatabase()
        >>> db.get_radius('Cu')  # doctest: +SKIP
        1.32
        >>> db.get_radius('Si', radius_type='covalent')  # doctest: +SKIP
        1.11
    """

    # Metallic radii (Angstroms) for 48 core metals
    # Source: Hardcoded values optimized for metallic bonding
    # Note: Covalent radii used as fallback for other 69 validated elements
    METALLIC_RADII = {
        "Li": 1.28, "Be": 0.96, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "K": 2.03,
        "Ca": 1.76, "Rb": 2.16, "Sr": 1.91, "Cs": 2.35, "Ba": 2.16, "Sc": 1.44,
        "Ti": 1.36, "V": 1.34, "Cr": 1.27, "Mn": 1.39, "Fe": 1.32, "Co": 1.26,
        "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Y": 1.90, "Zr": 1.46, "Nb": 1.34,
        "Mo": 1.30, "Tc": 1.27, "Ru": 1.25, "Rh": 1.25, "Pd": 1.39, "Ag": 1.45,
        "Cd": 1.48, "Hf": 1.44, "Ta": 1.43, "W": 1.30, "Re": 1.37, "Os": 1.28,
        "Ir": 1.35, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Ga": 1.26, "In": 1.44,
        "Sn": 1.41, "Tl": 1.48, "Pb": 1.46, "Bi": 1.48, "Si": 1.11, "Ge": 1.22
    }

    def __init__(self, database_file: Optional[str] = None):
        """
        Initialize radii database.

        Args:
            database_file: Path to atomic radii JSON file (if None, uses built-in data)
                          Default location: data/radii/atomic_radii.json
        """
        self.database_file = database_file
        self._data: Optional[Dict] = None

        # Load from file if provided
        if database_file and Path(database_file).exists():
            self._load_from_file()

    def _load_from_file(self) -> None:
        """Load radii database from JSON file."""
        with open(self.database_file, 'r') as f:
            raw_data = json.load(f)
        # Filter out metadata
        self._data = {k: v for k, v in raw_data.items() if k != "_metadata"}

    def get_radius(
        self,
        element: str,
        radius_type: Literal["auto", "metallic", "covalent"] = "auto"
    ) -> float:
        """
        Get atomic radius for an element.

        Args:
            element: Element symbol (e.g., 'Cu', 'Si')
            radius_type: Type of radius to return
                - "auto": Prefer metallic, fall back to covalent
                - "metallic": Only metallic radius
                - "covalent": Only covalent radius

        Returns:
            Atomic radius in Angstroms

        Raises:
            ValueError: If element not found or radius type unavailable

        Examples:
            >>> db = RadiiDatabase()
            >>> db.get_radius('Cu')  # Returns metallic radius
            1.32
            >>> db.get_radius('H', radius_type='covalent')  # Returns covalent radius
            0.31
        """
        # If database file loaded, use it
        if self._data and element in self._data:
            elem_data = self._data[element]

            if radius_type == "metallic":
                if elem_data.get("metallic_radius") is None:
                    raise ValueError(f"No metallic radius for element {element}")
                return float(elem_data["metallic_radius"])

            elif radius_type == "covalent":
                if elem_data.get("covalent_radius") is None:
                    raise ValueError(f"No covalent radius for element {element}")
                return float(elem_data["covalent_radius"])

            elif radius_type == "auto":
                # Prefer metallic, fall back to covalent
                return float(elem_data.get("preferred_radius"))

        # Otherwise use built-in data
        if radius_type == "metallic":
            if element not in self.METALLIC_RADII:
                raise ValueError(f"No metallic radius for element {element}")
            return self.METALLIC_RADII[element]

        elif radius_type == "covalent":
            z = atomic_numbers.get(element)
            if z is None:
                raise ValueError(f"Unknown element: {element}")
            return float(covalent_radii[z])

        elif radius_type == "auto":
            # Prefer metallic if available
            if element in self.METALLIC_RADII:
                return self.METALLIC_RADII[element]
            else:
                # Fall back to covalent
                z = atomic_numbers.get(element)
                if z is None:
                    raise ValueError(f"Unknown element: {element}")
                return float(covalent_radii[z])

        else:
            raise ValueError(f"Unknown radius_type: {radius_type}")

    def has_metallic_radius(self, element: str) -> bool:
        """
        Check if metallic radius is available for element.

        Args:
            element: Element symbol

        Returns:
            True if metallic radius available

        Examples:
            >>> db = RadiiDatabase()
            >>> db.has_metallic_radius('Cu')
            True
            >>> db.has_metallic_radius('H')
            False
        """
        if self._data and element in self._data:
            return self._data[element].get("metallic_radius") is not None
        return element in self.METALLIC_RADII

    @staticmethod
    def build_database() -> Dict[str, Dict]:
        """
        Build comprehensive radii database from multiple sources.

        Combines:
        - Metallic radii (48 core metals, hardcoded - optimized for metallic bonding)
        - ASE covalent radii (119 elements - fallback for other 69 validated elements)
        - Atomic numbers and masses

        Returns:
            Dictionary mapping element symbols to radii data

        Examples:
            >>> data = RadiiDatabase.build_database()
            >>> data['Cu']['metallic_radius']
            1.32
            >>> data['Cu']['covalent_radius']
            1.32
        """
        from datetime import datetime

        database = {}

        # Process all elements in ASE
        for symbol in chemical_symbols[1:]:  # Skip index 0 (empty)
            if not symbol:
                continue

            z = atomic_numbers[symbol]

            # Get covalent radius from ASE
            r_cov = float(covalent_radii[z])

            # Get metallic radius if available
            r_metal = RadiiDatabase.METALLIC_RADII.get(symbol, None)

            # Determine preferred radius (prefer metallic)
            if r_metal is not None:
                r_preferred = r_metal
                radius_type = "metallic"
            else:
                r_preferred = r_cov
                radius_type = "covalent"

            database[symbol] = {
                "atomic_number": z,
                "atomic_mass": float(atomic_masses[z]),
                "covalent_radius": r_cov,
                "metallic_radius": r_metal,
                "preferred_radius": r_preferred,
                "radius_type": radius_type
            }

        return database

    @staticmethod
    def save_database(
        output_file: str = "data/radii/atomic_radii.json",
        pretty: bool = True
    ) -> None:
        """
        Build and save radii database to JSON file.

        Args:
            output_file: Output file path (default: data/radii/atomic_radii.json)
            pretty: If True, use indented JSON formatting

        Examples:
            >>> RadiiDatabase.save_database("data/radii/atomic_radii.json")  # doctest: +SKIP
        """
        from datetime import datetime

        database = RadiiDatabase.build_database()

        # Add metadata
        metadata = {
            "_metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "num_elements": len(database),
                "sources": [
                    "Metallic radii: Hardcoded (48 core metals)",
                    "Covalent radii: ASE library (119 elements, fallback for 69 validated)",
                    "Atomic masses: ASE library"
                ],
                "description": (
                    "Comprehensive atomic radii database. "
                    "Metallic radii preferred for metals, covalent for others."
                )
            }
        }

        # Merge metadata with database
        output = {**metadata, **database}

        # Save to file
        with open(output_file, 'w') as f:
            if pretty:
                json.dump(output, f, indent=2, sort_keys=True)
            else:
                json.dump(output, f, sort_keys=True)

        print(f"✅ Atomic radii database saved to: {output_file}")
        print(f"   Total elements: {len(database)}")
        print(f"   Elements with metallic radii: {sum(1 for d in database.values() if d['metallic_radius'] is not None)}")
        print(f"   Elements with covalent radii: {len(database)}")

    def get_all_elements(self) -> list:
        """
        Get list of all available elements.

        Returns:
            List of element symbols

        Examples:
            >>> db = RadiiDatabase()
            >>> elements = db.get_all_elements()
            >>> 'Cu' in elements
            True
            >>> len(elements) > 100
            True
        """
        if self._data:
            return list(self._data.keys())
        else:
            # Return all elements from ASE
            return [s for s in chemical_symbols[1:] if s]

    def get_element_info(self, element: str) -> Dict:
        """
        Get complete information for an element.

        Args:
            element: Element symbol

        Returns:
            Dictionary with element data

        Examples:
            >>> db = RadiiDatabase()
            >>> info = db.get_element_info('Cu')  # doctest: +SKIP
            >>> info['atomic_number']
            29
        """
        if self._data and element in self._data:
            return self._data[element]
        else:
            # Build on the fly
            if element not in atomic_numbers:
                raise ValueError(f"Unknown element: {element}")

            z = atomic_numbers[element]
            r_cov = float(covalent_radii[z])
            r_metal = self.METALLIC_RADII.get(element, None)

            if r_metal is not None:
                r_preferred = r_metal
                radius_type = "metallic"
            else:
                r_preferred = r_cov
                radius_type = "covalent"

            return {
                "atomic_number": z,
                "atomic_mass": float(atomic_masses[z]),
                "covalent_radius": r_cov,
                "metallic_radius": r_metal,
                "preferred_radius": r_preferred,
                "radius_type": radius_type
            }


# Global instance for convenience
_default_database: Optional[RadiiDatabase] = None


def get_radii_database(database_file: Optional[str] = None) -> RadiiDatabase:
    """
    Get the global radii database instance.

    Args:
        database_file: Path to atomic radii JSON file (if None, uses built-in data)
                      Default location: data/radii/atomic_radii.json

    Returns:
        RadiiDatabase instance

    Examples:
        >>> db = get_radii_database()
        >>> db.get_radius('Cu')
        1.32
    """
    global _default_database

    # If database_file specified, always create new instance
    if database_file:
        return RadiiDatabase(database_file)

    # Otherwise use global singleton
    if _default_database is None:
        _default_database = RadiiDatabase()

    return _default_database
