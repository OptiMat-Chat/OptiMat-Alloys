"""
Factory for creating Ollama model clients.

This module provides a centralized way to create Ollama model clients,
following the same pattern as model_factory.py for OpenAI models.

Requires:
    - Ollama installed and running: https://ollama.com
    - autogen-ext[ollama] package: pip install autogen-ext[ollama]
"""

from typing import Any, Optional

from .ollama_config import (
    OLLAMA_MODELS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_HOST,
)


def create_ollama_client(
    model_name: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.0,
    options: Optional[dict] = None,
) -> Any:
    """
    Create Ollama model client.

    This function creates an OllamaChatCompletionClient configured for
    the specified model. Requires Ollama to be running locally.

    Args:
        model_name: Ollama model identifier. Supported models:
            Priority 1 (GPT-4.1-mini competitive):
            - mistral-small:24b: Excellent tool calling (~14GB VRAM)
            - qwen2.5:32b: Best reasoning (~16GB VRAM)
            - qwen3:32b: Latest with streaming tools (~16GB VRAM)

            Priority 2 (Fast fallbacks):
            - llama3.1:8b: Official tool support (~5GB VRAM)
            - qwen2.5:14b: Good balance (~8GB VRAM)

        host: Ollama server URL. Default: http://localhost:11434
        temperature: Sampling temperature for response generation.
            - 0.0: Deterministic (default, recommended for scientific tasks)
            - 0.0-2.0: More creative/random responses
        options: Additional Ollama options (num_ctx, num_gpu, etc.)

    Returns:
        OllamaChatCompletionClient configured for the specified model

    Raises:
        ImportError: If autogen-ext[ollama] is not installed
        ConnectionError: If Ollama server is not running

    Examples:
        >>> # Create default client (qwen2.5:14b)
        >>> client = create_ollama_client()

        >>> # Create client for specific model
        >>> client = create_ollama_client("mistral-small:24b")

        >>> # Create client with custom options
        >>> client = create_ollama_client(
        ...     "qwen2.5:32b",
        ...     options={"num_ctx": 8192}
        ... )

    Notes:
        - Ollama must be running: `ollama serve`
        - Model must be pulled: `ollama pull <model_name>`
        - No API key required (local inference)
    """
    try:
        from autogen_ext.models.ollama import OllamaChatCompletionClient
    except ImportError as e:
        raise ImportError(
            "Ollama support requires autogen-ext[ollama]. "
            "Install with: pip install 'autogen-ext[ollama]'"
        ) from e

    # Merge default options with provided options
    client_options = options or {}

    # Always provide model_info — same approach as OpenRouter and Xiaomi clients.
    # AutoGen's built-in registry only covers OpenAI/Anthropic/Google/Meta models,
    # so all Ollama models need explicit model_info.
    # Vision support is auto-detected from model name patterns (e.g., "-vl", "vision").
    name_lower = model_name.lower()
    is_vision = any(tag in name_lower for tag in ["-vl", "vision", "-visual"])
    family = model_name.split(":")[0] if ":" in model_name else model_name

    client = OllamaChatCompletionClient(
        model=model_name,
        host=host,
        options=client_options,
        model_info={
            "vision": is_vision,
            "function_calling": True,
            "json_output": True,
            "family": family,
            "structured_output": True,
        },
    )

    return client


def get_ollama_model_info(model_name: str) -> str:
    """
    Get human-readable description of an Ollama model.

    Args:
        model_name: Model identifier

    Returns:
        Descriptive string about the model

    Examples:
        >>> get_ollama_model_info("qwen2.5:14b")
        'Good balance of speed and capability'
    """
    return OLLAMA_MODELS.get(model_name, {}).get(
        "description", "Ollama model"
    )


def check_ollama_available(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """
    Check if Ollama server is running and accessible.

    Args:
        host: Ollama server URL

    Returns:
        True if Ollama is accessible, False otherwise
    """
    import urllib.request
    import urllib.error

    try:
        # Ollama API endpoint for version/health check
        url = f"{host}/api/version"
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def list_available_models(host: str = DEFAULT_OLLAMA_HOST) -> list[str]:
    """
    List models currently available in Ollama.

    Args:
        host: Ollama server URL

    Returns:
        List of model names available locally

    Raises:
        ConnectionError: If Ollama is not accessible
    """
    import json
    import urllib.request
    import urllib.error

    try:
        url = f"{host}/api/tags"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            return [model["name"] for model in data.get("models", [])]
    except (urllib.error.URLError, TimeoutError) as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {host}. "
            "Make sure Ollama is running: `ollama serve`"
        ) from e


# Model metadata for UI display (mirrors model_factory.py pattern)
MODEL_INFO = {
    model_name: {
        "name": info["name"],
        "description": info["description"],
        "context_window": f"{info['context_window']} tokens",
        "vram_required": f"~{info['vram_gb']}GB",
        "use_case": info.get("use_case", "General purpose"),
    }
    for model_name, info in OLLAMA_MODELS.items()
}
