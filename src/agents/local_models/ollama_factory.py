"""
Factory for creating Ollama model clients.

This module provides a centralized way to create Ollama model clients,
following the same pattern as model_factory.py for OpenAI models.

Requires:
    - Ollama installed and running: https://ollama.com

The client talks to the Ollama daemon's OpenAI-compatible endpoint
({host}/v1). The daemon itself proxies cloud models (e.g.
gpt-oss:120b-cloud) and handles their authentication via OLLAMA_API_KEY
or the saved device key, so one client class covers local AND cloud models.
"""

import os
from typing import Any

from agent_framework.openai import OpenAIChatCompletionClient

from .ollama_config import (
    OLLAMA_MODELS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_HOST,
)


def create_ollama_client(
    model_name: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
) -> Any:
    """
    Create Ollama model client.

    Creates an Agent Framework OpenAI-compatible chat client pointed at the
    Ollama daemon. Requires Ollama to be running locally.

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

    Returns:
        OpenAIChatCompletionClient configured for the specified model

    Examples:
        >>> # Create default client
        >>> client = create_ollama_client()

        >>> # Create client for specific model
        >>> client = create_ollama_client("mistral-small:24b")

    Notes:
        - Local models: Ollama must be running (`ollama serve`) and the
          model pulled (`ollama pull <model_name>`). No API key is needed;
          the OpenAI SDK requires a non-empty value, so "ollama" is used as
          the documented placeholder.
        - Cloud models WITH an OLLAMA_API_KEY: requests go directly to
          ollama.com's OpenAI-compatible endpoint. (The local daemon
          authenticates cloud proxying with a device key from
          `ollama signin`, not the env API key — an API-key-only setup hit
          a runtime 401 through the daemon, in the AutoGen era too.)
        - Cloud models WITHOUT a key: the daemon proxy path is preserved
          for device-authorized (`ollama signin`) setups.
        - Context length is configured daemon-side (OLLAMA_CONTEXT_LENGTH).
    """
    api_key = os.getenv("OLLAMA_API_KEY")
    is_cloud_model = OLLAMA_MODELS.get(model_name, {}).get("cloud", False)
    if is_cloud_model and api_key:
        return OpenAIChatCompletionClient(
            model=model_name,
            base_url="https://ollama.com/v1",
            api_key=api_key,
        )
    base = host.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return OpenAIChatCompletionClient(
        model=model_name,
        base_url=base,
        api_key=api_key or "ollama",
    )


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
