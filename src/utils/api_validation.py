"""
OpenAI API key validation utilities.

This module provides utilities for validating OpenAI API keys by making
minimal test calls to the OpenAI API.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


async def validate_openai_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate an OpenAI API key by making a minimal test call.

    Makes a lightweight API call to OpenAI's models.list() endpoint to verify:
    - The API key is properly formatted
    - The API key is valid and authenticated
    - The account has available quota
    - The network connection works

    This call costs approximately $0.0001 and completes in ~1-2 seconds.

    Args:
        api_key: OpenAI API key to validate (should start with 'sk-')

    Returns:
        Tuple[bool, str]: (is_valid, message)
            - is_valid: True if key is valid and working
            - message: Human-readable success or error message

    Examples:
        >>> valid, msg = await validate_openai_api_key('sk-...')
        >>> if valid:
        ...     print(f"Success: {msg}")
        ... else:
        ...     print(f"Error: {msg}")
    """
    from openai import AsyncOpenAI
    import asyncio

    # Basic format check
    if not api_key or not isinstance(api_key, str):
        return False, "API key is empty or invalid type"

    api_key = api_key.strip()

    if not api_key.startswith('sk-'):
        return False, "API key must start with 'sk-' (found invalid format)"

    if len(api_key) < 20:
        return False, "API key is too short (expected 40+ characters)"

    # Try to make a test API call
    try:
        client = AsyncOpenAI(api_key=api_key, timeout=10.0)

        # Make minimal API call (just lists available models)
        # This is one of the cheapest operations possible
        await asyncio.wait_for(client.models.list(), timeout=10.0)

        logger.info("OpenAI API key validated successfully")
        return True, "API key validated successfully ✓"

    except asyncio.TimeoutError:
        logger.error("API key validation timed out")
        return False, "Network timeout - check your internet connection"

    except Exception as e:
        error_str = str(e).lower()
        logger.error(f"API key validation failed: {e}")

        # Parse specific error types
        if "invalid" in error_str or "incorrect" in error_str:
            return False, "Invalid API key - authentication failed"

        elif "quota" in error_str or "exceeded" in error_str:
            # Key is valid but quota is exhausted
            return False, "API key valid but quota exceeded - add billing"

        elif "permission" in error_str:
            return False, "API key valid but lacks required permissions"

        elif "rate" in error_str or "limit" in error_str:
            return False, "Rate limit exceeded - try again in a few seconds"

        elif "network" in error_str or "connection" in error_str:
            return False, "Network error - check your internet connection"

        elif "timeout" in error_str:
            return False, "Request timed out - check your internet connection"

        else:
            # Generic error
            return False, f"Validation failed: {str(e)[:100]}"


def format_api_key_for_display(api_key: str) -> str:
    """
    Format an API key for safe display by masking most characters.

    Args:
        api_key: Full API key

    Returns:
        str: Masked API key (e.g., 'sk-...xyz123')

    Examples:
        >>> format_api_key_for_display('sk-abcd1234efgh5678')
        'sk-...5678'
    """
    if not api_key or len(api_key) < 8:
        return "***"

    # Show first 3 chars and last 4 chars
    return f"{api_key[:3]}...{api_key[-4:]}"


def is_api_key_format_valid(api_key: str) -> Tuple[bool, str]:
    """
    Check if an API key has valid format without making API calls.

    Performs quick format validation:
    - Checks if it starts with 'sk-'
    - Checks length
    - Checks for obvious issues

    This is faster than validate_openai_api_key() but doesn't guarantee
    the key actually works.

    Args:
        api_key: API key to check

    Returns:
        Tuple[bool, str]: (is_valid_format, message)
    """
    if not api_key or not isinstance(api_key, str):
        return False, "API key is empty or invalid type"

    api_key = api_key.strip()

    if not api_key:
        return False, "API key is empty"

    if not api_key.startswith('sk-'):
        return False, "API key must start with 'sk-'"

    if len(api_key) < 20:
        return False, f"API key too short (found {len(api_key)} chars, expected 40+)"

    if ' ' in api_key:
        return False, "API key contains spaces (invalid format)"

    return True, "API key format looks valid"
