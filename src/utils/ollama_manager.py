"""
Ollama service management utilities.

Handles:
- Checking Ollama availability
- Auto-starting Ollama server
- Querying available models
- Filtering models by capability (tool calling, VRAM)

This module provides the bridge between the Ollama backend
(src/agents/local_models/) and the Chainlit UI for model selection.
"""

import asyncio
import json
import shutil
from collections.abc import AsyncGenerator
import subprocess
from typing import Optional

from src.agents.local_models.ollama_factory import (
    check_ollama_available,
    list_available_models,
)
from src.agents.local_models.ollama_config import OLLAMA_MODELS, DEFAULT_OLLAMA_HOST


def is_ollama_installed() -> bool:
    """
    Check if Ollama binary is available in PATH.

    Returns:
        True if 'ollama' command is found, False otherwise
    """
    return shutil.which("ollama") is not None


async def ensure_ollama_running(timeout: int = 30) -> tuple[bool, str]:
    """
    Ensure Ollama server is running, starting it if necessary.

    This function checks if Ollama is available, and if not:
    1. Verifies Ollama is installed
    2. Starts 'ollama serve' as a background process
    3. Polls for availability until ready or timeout

    The Ollama process is detached (start_new_session=True) so it
    continues running after OptiMat Alloys exits.

    Args:
        timeout: Maximum seconds to wait for Ollama to start

    Returns:
        Tuple of (success: bool, message: str)
        - success: True if Ollama is running
        - message: Status description for UI display
    """
    # Check if already running
    if check_ollama_available():
        return True, "Ollama server is running"

    # Check if installed
    if not is_ollama_installed():
        return False, (
            "Ollama is not installed. "
            "Please install from https://ollama.com and try again."
        )

    # Try to start Ollama
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process
        )
    except Exception as e:
        return False, f"Failed to start Ollama: {str(e)}"

    # Wait for Ollama to become available
    for i in range(timeout):
        await asyncio.sleep(1)
        if check_ollama_available():
            return True, f"Ollama started successfully (took {i + 1}s)"

    return False, f"Ollama failed to start within {timeout} seconds"


def get_available_ollama_models() -> list[dict]:
    """
    Get list of Ollama models that are installed locally.

    This function:
    1. Queries Ollama for installed models
    2. Cross-references with OLLAMA_MODELS config for metadata
    3. Returns formatted list suitable for UI display

    Returns:
        List of dicts with:
        - name: Model identifier for Ollama (e.g., 'qwen2.5:14b')
        - display_name: Human-readable name
        - description: Brief description of capabilities
        - vram_gb: Approximate VRAM requirement
        - has_tool_calling: Whether model supports function calling

        Returns empty list if Ollama is not running or no models installed.
    """
    try:
        installed_models = list_available_models()
    except ConnectionError:
        return []

    if not installed_models:
        return []

    available = []
    for model_id in installed_models:
        # Check if model is in our configuration
        config = OLLAMA_MODELS.get(model_id)

        if config:
            # Known model with full metadata
            available.append({
                "name": model_id,
                "display_name": config.get("name", model_id),
                "description": config.get("description", ""),
                "vram_gb": config.get("vram_gb", 0),
                "has_tool_calling": config.get("function_calling", False),
            })
        else:
            # Model installed but not in our config
            # Try to match base name (e.g., 'qwen2.5' matches 'qwen2.5:14b')
            base_name = model_id.split(":")[0] if ":" in model_id else model_id
            matched = False

            for config_name, config_data in OLLAMA_MODELS.items():
                config_base = config_name.split(":")[0]
                if base_name == config_base:
                    # Found a variant - use its metadata with actual model name
                    available.append({
                        "name": model_id,
                        "display_name": f"{config_data.get('name', model_id)} (variant)",
                        "description": config_data.get("description", ""),
                        "vram_gb": config_data.get("vram_gb", 0),
                        "has_tool_calling": config_data.get("function_calling", False),
                    })
                    matched = True
                    break

            if not matched:
                # Unknown model - include with warning
                available.append({
                    "name": model_id,
                    "display_name": model_id,
                    "description": "Custom model (tool calling support unknown)",
                    "vram_gb": 0,
                    "has_tool_calling": None,  # Unknown
                })

    # Sort: known tool-calling models first, then by name
    available.sort(key=lambda x: (
        x["has_tool_calling"] is not True,  # True first, then False/None
        x["name"]
    ))

    return available


def get_recommended_models() -> list[str]:
    """
    Get list of recommended model names to pull if none are installed.

    Returns:
        List of model identifiers suitable for 'ollama pull <model>'
    """
    return [
        "qwen2.5:14b",      # Good balance, 8GB VRAM
        "qwen3:8b",         # Fast, 5GB VRAM
        "mistral-small:24b", # Best tool calling, 14GB VRAM
        "llama3.1:8b",      # Official tool support, 5GB VRAM
    ]


def is_model_installed(model_name: str) -> bool:
    """
    Check if a specific Ollama model is installed locally.

    Args:
        model_name: Model identifier (e.g., 'gpt-oss:20b', 'qwen2.5:14b')

    Returns:
        True if the model is installed, False otherwise
    """
    try:
        installed_models = list_available_models()
        return model_name in installed_models
    except ConnectionError:
        return False


def get_model_vram_gb(
    model_name: str, host: str = DEFAULT_OLLAMA_HOST
) -> Optional[float]:
    """
    Estimate VRAM (GB) needed to run an Ollama model.

    Strategy:
    1. Curated value from OLLAMA_MODELS (validated against tool calling).
    2. Otherwise, query /api/tags for the model's on-disk size — GGUF
       weights load roughly 1:1 into VRAM. KV cache adds extra at long
       contexts, which the caller's recommended-headroom should cover.
    3. Return None if not curated and not installed (or server unreachable).
    """
    curated = OLLAMA_MODELS.get(model_name, {}).get("vram_gb")
    if curated is not None and curated > 0:
        return float(curated)

    import json
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as response:
            data = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    for model in data.get("models", []):
        if model.get("name") == model_name:
            size_bytes = model.get("size", 0)
            if size_bytes > 0:
                return round(size_bytes / 1e9, 1)
            break
    return None


async def pull_model(model_name: str, timeout: int = 1800) -> tuple[bool, str]:
    """
    Pull (download) an Ollama model.

    This function runs 'ollama pull <model_name>' and waits for completion.
    Large models (20B+) may take several minutes to download.

    Args:
        model_name: Model identifier to pull (e.g., 'gpt-oss:20b')
        timeout: Maximum seconds to wait for download (default: 5 minutes)

    Returns:
        Tuple of (success: bool, message: str)
        - success: True if model was pulled successfully
        - message: Status description or error message
    """
    if not is_ollama_installed():
        return False, "Ollama is not installed"

    # Ensure Ollama server is running (required for pull)
    if not check_ollama_available():
        start_success, start_msg = await ensure_ollama_running(timeout=30)
        if not start_success:
            return False, f"Cannot start Ollama server: {start_msg}"

    try:
        # Run ollama pull with timeout
        process = await asyncio.create_subprocess_exec(
            "ollama", "pull", model_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return False, f"Download timed out after {timeout} seconds"

        if process.returncode == 0:
            return True, f"Successfully installed {model_name}"
        else:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            return False, f"Failed to pull {model_name}: {error_msg}"

    except Exception as e:
        return False, f"Error pulling model: {str(e)}"


async def pull_model_with_progress(
    model_name: str,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: int = 1800
) -> AsyncGenerator[tuple[str, int], None]:
    """
    Pull an Ollama model with streaming progress updates.

    Uses Ollama's HTTP API (POST /api/pull) which streams JSON lines
    with download progress. Yields (status_message, percent) tuples
    in real time as the download progresses.

    Args:
        model_name: Model identifier to pull (e.g., 'gpt-oss:20b')
        host: Ollama server URL
        timeout: Maximum seconds to wait (default: 30 minutes for large models)

    Yields:
        Tuples of (status_message: str, percent_complete: int)
        - percent is 0-100 during download, -1 for non-download phases

    Example:
        async for status, percent in pull_model_with_progress("gpt-oss:20b"):
            print(f"{status} ({percent}%)")
    """
    import urllib.request
    import urllib.error

    queue: asyncio.Queue[tuple[str, int] | None] = asyncio.Queue()

    url = f"{host}/api/pull"
    data = json.dumps({"name": model_name}).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    def _stream_pull():
        """Blocking HTTP request that pushes progress to queue."""
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                for line in response:
                    line = line.decode().strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        status = obj.get("status", "")
                        completed = obj.get("completed", 0)
                        total = obj.get("total", 0)

                        if total > 0:
                            percent = int(completed / total * 100)
                        else:
                            percent = -1

                        queue.put_nowait((status, percent))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            queue.put_nowait((f"Error: {e}", -1))
        finally:
            queue.put_nowait(None)  # Signal completion

    # Start blocking pull in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _stream_pull)

    # Yield progress as it arrives
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
