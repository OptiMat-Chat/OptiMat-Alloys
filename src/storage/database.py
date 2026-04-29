"""
ASE database management for storing generated structures.

This module provides the interface for the global structure database.
Legacy session-based storage has been replaced with a centralized database.
"""

# Re-export from global_database for backward compatibility
from src.storage.global_database import (
    GlobalStructureDatabase,
    create_global_database
)

# Alias for compatibility
StructureDatabase = GlobalStructureDatabase


def create_structure_database(base_dir: str = None) -> GlobalStructureDatabase:
    """
    Factory function to create a global structure database.

    Args:
        base_dir: Optional base directory for structures (default: "structures")

    Returns:
        GlobalStructureDatabase instance

    Examples:
        >>> db = create_structure_database()
        >>> isinstance(db, GlobalStructureDatabase)
        True
    """
    return create_global_database(base_dir)
