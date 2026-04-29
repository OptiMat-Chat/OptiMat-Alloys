"""
Environment file management utilities.

This module provides safe, cross-platform utilities for reading and writing
.env files, preserving existing variables and comments.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnvFileError(Exception):
    """Exception raised for .env file operations"""
    pass


def get_env_file_path() -> Path:
    """
    Get the path to the .env file in the project root.

    Returns:
        Path: Path to .env file
    """
    # Assumes this file is in src/utils/ and .env is in project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / ".env"


def read_env_file() -> list[str]:
    """
    Read all lines from .env file, creating it if it doesn't exist.

    Returns:
        list[str]: Lines from .env file (with newlines preserved)

    Raises:
        EnvFileError: If .env file cannot be read
    """
    env_path = get_env_file_path()

    try:
        if not env_path.exists():
            logger.info(f"Creating new .env file at {env_path}")

            # Try to copy from .env.example to preserve default configuration
            env_example_path = env_path.parent / ".env.example"
            if env_example_path.exists():
                logger.info(f"Initializing .env from .env.example template")
                import shutil
                shutil.copy(env_example_path, env_path)
                os.chmod(env_path, 0o600)  # Set restricted permissions

                # Read the newly created file
                with open(env_path, 'r', encoding='utf-8') as f:
                    return f.readlines()
            else:
                # Fallback: create empty file if .env.example doesn't exist
                logger.warning(".env.example not found, creating empty .env")
                env_path.touch(mode=0o600)
                return []

        with open(env_path, 'r', encoding='utf-8') as f:
            return f.readlines()

    except Exception as e:
        raise EnvFileError(f"Failed to read .env file: {e}")


def write_env_file(lines: list[str]) -> None:
    """
    Write lines to .env file atomically.

    Uses atomic write with temporary file to prevent corruption.

    Args:
        lines: Lines to write (should include newlines)

    Raises:
        EnvFileError: If .env file cannot be written
    """
    env_path = get_env_file_path()
    tmp_path = env_path.with_suffix('.tmp')

    try:
        # Write to temporary file
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # Set permissions before moving (Unix-like systems)
        if os.name != 'nt':  # Not Windows
            os.chmod(tmp_path, 0o600)

        # Atomic replace
        os.replace(tmp_path, env_path)
        logger.info(f"Successfully updated .env file at {env_path}")

    except Exception as e:
        # Clean up temporary file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise EnvFileError(f"Failed to write .env file: {e}")


def update_env_variable(key: str, value: str) -> Tuple[bool, str]:
    """
    Update or add a variable in the .env file.

    This function:
    - Preserves all existing variables and comments
    - Updates the value if key exists
    - Appends new key-value pair if key doesn't exist
    - Uses atomic writes to prevent corruption
    - Works cross-platform (Linux/Windows)

    Args:
        key: Environment variable name (e.g., 'OPENAI_API_KEY')
        value: Environment variable value (e.g., 'sk-...')

    Returns:
        Tuple[bool, str]: (success, message)

    Example:
        success, message = update_env_variable('OPENAI_API_KEY', 'sk-...')
        if success:
            print(f"Success: {message}")
        else:
            print(f"Error: {message}")
    """
    if not key or not isinstance(key, str):
        return False, "Invalid key: must be non-empty string"

    if not isinstance(value, str):
        return False, "Invalid value: must be string"

    try:
        # Read existing .env file
        lines = read_env_file()

        # Find and update existing key, or mark for append
        key_found = False
        new_lines = []

        for line in lines:
            stripped = line.strip()

            # Check if this line contains our key
            if stripped and not stripped.startswith('#') and '=' in stripped:
                line_key = stripped.split('=', 1)[0].strip()
                if line_key == key:
                    new_lines.append(f"{key}={value}\n")
                    key_found = True
                    continue

            # Preserve other lines as-is
            new_lines.append(line)

        # Append if key wasn't found
        if not key_found:
            # Add blank line before new entry if file isn't empty
            if new_lines and not new_lines[-1].strip() == '':
                new_lines.append('\n')
            new_lines.append(f"{key}={value}\n")

        # Write atomically
        write_env_file(new_lines)

        # Reload environment variables from .env
        from dotenv import load_dotenv
        load_dotenv(override=True)

        action = "updated" if key_found else "added"
        return True, f"Successfully {action} {key} in .env file"

    except EnvFileError as e:
        logger.error(f"EnvFileError: {e}")
        return False, str(e)

    except Exception as e:
        logger.error(f"Unexpected error updating .env: {e}")
        return False, f"Unexpected error: {e}"


def get_env_variable(key: str) -> Optional[str]:
    """
    Get environment variable value, checking both runtime and .env file.

    Args:
        key: Environment variable name

    Returns:
        Optional[str]: Variable value or None if not found
    """
    # First check runtime environment
    value = os.getenv(key)
    if value:
        return value

    # Then check .env file directly
    try:
        lines = read_env_file()
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '=' in stripped:
                line_key, line_value = stripped.split('=', 1)
                if line_key.strip() == key:
                    return line_value.strip()
    except EnvFileError:
        pass

    return None


def is_read_only_filesystem() -> bool:
    """
    Check if the filesystem is read-only (common in cloud deployments).

    Returns:
        bool: True if filesystem is read-only, False otherwise
    """
    env_path = get_env_file_path()
    test_file = env_path.parent / '.write_test'

    try:
        # Try to create a test file
        test_file.touch()
        test_file.unlink()
        return False
    except (OSError, PermissionError):
        return True


def can_write_env_file() -> Tuple[bool, str]:
    """
    Check if we can safely write to the .env file.

    Returns:
        Tuple[bool, str]: (can_write, reason)
    """
    # Check if filesystem is read-only
    if is_read_only_filesystem():
        return False, "Filesystem is read-only (common in cloud deployments)"

    env_path = get_env_file_path()

    # Check if .env exists and is writable
    if env_path.exists() and not os.access(env_path, os.W_OK):
        return False, f".env file exists but is not writable: {env_path}"

    # Check if parent directory is writable (for creating new .env)
    if not env_path.exists() and not os.access(env_path.parent, os.W_OK):
        return False, f"Cannot create .env file in: {env_path.parent}"

    return True, "Can write to .env file"
