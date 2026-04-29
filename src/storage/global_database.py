"""
Global structure database for OptiMat Alloys.

This module provides a centralized database for all generated structures
across all conversations. Each structure gets a unique global ID and
its own directory for storing associated files.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from ase import Atoms
from ase.db import connect
from ase.db.core import Database


class GlobalStructureDatabase:
    """
    Global database for atomic structures with per-structure file storage.

    All structures are stored in a single database with globally unique UUIDs.
    Each structure gets its own directory for images, trajectories, etc.

    Directory structure:
        structures/
        ├── database.db                        # Global ASE database (SQLite/PostgreSQL)
        └── a1b2c3d4e5f6.../                  # Structure UUID folders (32-char hex)
            ├── structure_elements.png
            ├── structure_analysis.png
            └── relaxation.traj

    Note:
        Uses ASE's built-in unique_id (32-character hex UUID) for global uniqueness
        across all database instances (local and cloud).
    """

    BASE_DIR = "structures"
    DB_NAME = "database.db"

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize global structure database.

        Args:
            base_dir: Base directory for structures (default: "structures")

        Examples:
            >>> db = GlobalStructureDatabase()
            >>> db.write(atoms, metadata)  # doctest: +SKIP
        """
        self.base_dir = Path(base_dir or self.BASE_DIR)
        self.db_path = self.base_dir / self.DB_NAME
        self._db: Optional[Database] = None

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_db(self) -> Database:
        """
        Get or create database connection.

        Returns:
            ASE Database object
        """
        if self._db is None:
            self._db = connect(str(self.db_path))
        return self._db

    def _get_row_by_uuid(self, entry_uuid: str):
        """
        Get database row by UUID.

        Args:
            entry_uuid: ASE unique_id (32-character hex string)

        Returns:
            ASE database row

        Raises:
            KeyError: If UUID not found
        """
        db = self._get_db()
        return db.get(unique_id=entry_uuid)

    def resolve_to_uuid(self, entry_reference: Union[int, str]) -> str:
        """
        Resolve integer ID or UUID string to UUID.

        Use this at the tool layer before calling database methods.
        Integer IDs are local to this database instance only.
        UUIDs are globally unique across all instances.

        Handles common LLM-generated prefixes like "#111", "ID 111",
        "structure 111", "Structure #111", etc.

        Args:
            entry_reference: Either integer ID (e.g., 5) or UUID (32-character hex string)

        Returns:
            UUID string (32-character hex)

        Raises:
            KeyError: If entry not found

        Examples:
            >>> uuid = db.resolve_to_uuid(5)  # Resolve int ID to UUID
            >>> uuid = db.resolve_to_uuid("5")  # String integer from LLM
            >>> uuid = db.resolve_to_uuid("#111")  # LLM prefix: hash
            >>> uuid = db.resolve_to_uuid("ID 111")  # LLM prefix: ID
            >>> uuid = db.resolve_to_uuid("structure 111")  # LLM prefix: structure
            >>> uuid = db.resolve_to_uuid("a1b2c3d4...")  # Pass-through UUID
        """
        import re

        # Handle integer (direct)
        if isinstance(entry_reference, int):
            row = self._get_db().get(id=entry_reference)
            return row.unique_id

        # Handle string inputs
        if isinstance(entry_reference, str):
            cleaned = entry_reference.strip()

            # Direct digit string (e.g., "161" from LLM)
            if cleaned.isdigit():
                row = self._get_db().get(id=int(cleaned))
                return row.unique_id

            # Strip common LLM-generated prefixes: "#111", "ID 111", "id:111",
            # "Structure 111", "structure #111", "Structure ID 111", etc.
            # Pattern: optional prefix words + optional punctuation + digits
            match = re.match(
                r'^(?:structure\s+)?(?:id\s*)?[#:\s]*(\d+)$',
                cleaned,
                re.IGNORECASE
            )
            if match:
                row = self._get_db().get(id=int(match.group(1)))
                return row.unique_id

        # Already a UUID string
        return entry_reference

    def write(
        self,
        atoms: Atoms,
        key_value_pairs: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Write structure to database with metadata.

        Automatically creates a directory for the structure and adds
        timestamp for provenance tracking.

        Args:
            atoms: ASE Atoms object to store
            key_value_pairs: Searchable metadata (str, int, float, bool only)
            data: Complex metadata (dicts, lists, nested structures)
            metadata: Legacy parameter - will be stored as 'data' (deprecated)

        Returns:
            Dict with 'id' (int) and 'uuid' (str, 32-character hex)
            Example: {"id": 5, "uuid": "a1b2c3d4e5f6..."}

        Examples:
            >>> kvp = {"structure": "fcc", "Cu_fraction": 0.5}
            >>> data = {"analysis": {"fcc": 95.2}}
            >>> result = db.write(atoms, key_value_pairs=kvp, data=data)  # doctest: +SKIP
            >>> result["id"]  # doctest: +SKIP
            5
            >>> len(result["uuid"])  # doctest: +SKIP
            32
        """
        db = self._get_db()

        # Handle legacy metadata parameter (backward compatibility)
        if metadata is not None:
            if data is None:
                data = {}
            data.update(metadata)

        # Add timestamp for provenance
        if data is None:
            data = {}
        data["timestamp"] = datetime.now().isoformat()

        # Ensure key_value_pairs is a dict (ASE requires dict, not None)
        if key_value_pairs is None:
            key_value_pairs = {}

        # Write to database with both key_value_pairs and data
        entry_id = db.write(atoms, key_value_pairs=key_value_pairs, data=data)

        # Get the UUID from the written entry
        row = db.get(id=entry_id)
        entry_uuid = row.unique_id

        # Create structure directory using UUID
        structure_dir = self.get_structure_directory(entry_uuid)
        structure_dir.mkdir(parents=True, exist_ok=True)

        return {"id": entry_id, "uuid": entry_uuid}

    def read(self, entry_uuid: str) -> Atoms:
        """
        Read structure from database by UUID.

        Args:
            entry_uuid: UUID (32-character hex string). Use resolve_to_uuid() first
                       if you have an integer ID.

        Returns:
            ASE Atoms object

        Raises:
            KeyError: If entry not found

        Examples:
            >>> atoms = db.read("a1b2c3d4...")  # By UUID  # doctest: +SKIP
        """
        row = self._get_row_by_uuid(entry_uuid)
        return row.toatoms()

    def get_metadata(self, entry_uuid: str) -> Dict[str, Any]:
        """
        Get metadata for a database entry.

        Returns both searchable metadata (key_value_pairs) and complex data (data)
        merged into a single dictionary.

        Args:
            entry_uuid: UUID (32-character hex string). Use resolve_to_uuid() first
                       if you have an integer ID.

        Returns:
            Dictionary of metadata (both key_value_pairs and data merged)

        Raises:
            KeyError: If entry not found

        Examples:
            >>> metadata = db.get_metadata("a1b2c3d4...")  # By UUID  # doctest: +SKIP
            >>> metadata["structure"]  # doctest: +SKIP
            'fcc'
        """
        row = self._get_row_by_uuid(entry_uuid)
        # Merge both key_value_pairs (searchable) and data (complex) fields
        # If there are duplicate keys, data takes precedence
        return {**dict(row.key_value_pairs), **dict(row.data)}

    def update_data(self, entry_uuid: str, data: Dict[str, Any]) -> None:
        """
        Update metadata for an existing database entry.

        Merges new data with existing metadata. The 'data' field is updated
        while preserving existing entries not specified in the update.

        Args:
            entry_uuid: UUID (32-character hex string). Use resolve_to_uuid() first
                       if you have an integer ID.
            data: Dictionary of new/updated metadata fields

        Raises:
            KeyError: If entry not found

        Examples:
            >>> db.update_data("a1b2c3d4...", {"elastic_tensor": [[...]]})  # By UUID  # doctest: +SKIP
        """
        db = self._get_db()

        # Read existing entry by UUID
        row = self._get_row_by_uuid(entry_uuid)

        # Merge existing data with new data
        updated_data = dict(row.data) if row.data else {}
        updated_data.update(data)

        # Add update timestamp
        updated_data["last_updated"] = datetime.now().isoformat()

        # Update database entry (ASE update requires integer ID)
        db.update(id=row.id, data=updated_data)

    def update_key_value_pairs(self, entry_uuid: str, key_value_pairs: Dict[str, Any]) -> None:
        """
        Update searchable key_value_pairs for an existing database entry.

        Unlike `update_data()` which stores complex nested data in the 'data' field,
        this method updates the searchable key_value_pairs that can be queried
        using ASE database selection syntax.

        Only scalar types (str, int, float, bool) are allowed in key_value_pairs.

        Args:
            entry_uuid: UUID (32-character hex string). Use resolve_to_uuid() first
                       if you have an integer ID.
            key_value_pairs: Dictionary of searchable fields to add/update

        Raises:
            KeyError: If entry not found
            TypeError: If values are not scalar types

        Examples:
            >>> db.update_key_value_pairs("a1b2c3d4...", {"has_qha_data": True})  # By UUID  # doctest: +SKIP
        """
        db = self._get_db()

        # Read existing entry by UUID
        row = self._get_row_by_uuid(entry_uuid)

        # Validate all values are scalar types
        for key, value in key_value_pairs.items():
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(
                    f"key_value_pairs values must be scalar types (str, int, float, bool, None). "
                    f"Got {type(value).__name__} for key '{key}'"
                )

        # Update using ASE's database update with keyword arguments
        db.update(id=row.id, **key_value_pairs)

    def count(self) -> int:
        """
        Get total number of entries in database.

        Returns:
            Number of structures stored

        Examples:
            >>> count = db.count()  # doctest: +SKIP
            >>> count >= 0
            True
        """
        db = self._get_db()
        return db.count()

    def select(self, **kwargs) -> List[str]:
        """
        Select entries matching criteria.

        Args:
            **kwargs: Selection criteria (e.g., structure="fcc")

        Returns:
            List of entry UUIDs matching criteria (32-character hex strings)

        Examples:
            >>> uuids = db.select(structure="fcc")  # doctest: +SKIP
            >>> all(isinstance(u, str) and len(u) == 32 for u in uuids)  # doctest: +SKIP
            True
        """
        db = self._get_db()
        return [row.unique_id for row in db.select(**kwargs)]

    def get_structure_directory(self, entry_uuid: str) -> Path:
        """
        Get the directory path for a specific structure.

        Args:
            entry_uuid: UUID (32-character hex string). Use resolve_to_uuid() first
                       if you have an integer ID.

        Returns:
            Path to structure directory

        Examples:
            >>> path = db.get_structure_directory("a1b2c3d4...")  # By UUID
            >>> "a1b2c3d4" in str(path)
            True
        """
        return self.base_dir / entry_uuid

    def get_structure_path(self, entry_uuid: str, filename: str) -> str:
        """
        Get the full path for a file within a structure's directory.

        Args:
            entry_uuid: UUID (32-character hex string). Use resolve_to_uuid() first
                       if you have an integer ID.
            filename: Name of file (e.g., "structure_elements.png")

        Returns:
            Full path to file as string

        Examples:
            >>> path = db.get_structure_path("a1b2c3d4...", "image.png")  # By UUID
            >>> "a1b2c3d4" in path and "image.png" in path
            True
        """
        return str(self.get_structure_directory(entry_uuid) / filename)

    def get_database_path(self) -> str:
        """
        Get the full path to the database file.

        Returns:
            Absolute path to database file (as string for compatibility)

        Examples:
            >>> path = db.get_database_path()
            >>> path.endswith("database.db")
            True
        """
        return str(self.db_path.resolve())

    def close(self) -> None:
        """
        Close database connection.

        Examples:
            >>> db.close()  # doctest: +SKIP
        """
        if self._db is not None:
            # ASE database doesn't require explicit close, but we reset it
            self._db = None


def create_global_database(base_dir: Optional[str] = None) -> GlobalStructureDatabase:
    """
    Factory function to create a GlobalStructureDatabase.

    Args:
        base_dir: Optional base directory for structures

    Returns:
        GlobalStructureDatabase instance

    Examples:
        >>> db = create_global_database()
        >>> isinstance(db, GlobalStructureDatabase)
        True
    """
    return GlobalStructureDatabase(base_dir)
