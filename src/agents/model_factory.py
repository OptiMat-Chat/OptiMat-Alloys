"""
Factory for creating OpenRouter and Ollama model clients dynamically.

This module provides a centralized way to create model clients with
different configurations, enabling runtime model switching without
requiring application restart.

Supports:
- OpenRouter free models (cloud API, via "free:" prefix) - GPT-OSS, Qwen, etc.
- Ollama models (local/cloud inference, requires Ollama server) - GPT-OSS, Qwen, Mistral, etc.
"""

import os
from autogen_core.models import ChatCompletionClient, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import Any, Literal, Optional


# Free models available via OpenRouter (prefix with "free:" in UI)
FREE_MODEL_PREFIX = "free:"


def is_free_model(model_name: str) -> bool:
    """
    Check if a model name indicates a free OpenRouter model.

    Free models are identified by:
    - "free:" prefix (e.g., "free:openai/gpt-oss-120b")
    - ":free" suffix (e.g., "openai/gpt-oss-120b:free")

    Args:
        model_name: Model identifier to check

    Returns:
        True if this is a free model request
    """
    return (
        model_name.startswith(FREE_MODEL_PREFIX) or
        model_name.endswith(":free") or
        model_name == "free"  # Generic "free" selection
    )


class RateLimitedOpenAIChatCompletionClient(OpenAIChatCompletionClient):
    """OpenAI-compatible client with per-request rate limiting for OpenRouter.

    Adds a 3-second delay before every create() and create_stream() call
    to prevent hitting OpenRouter's 20 req/min rate limit during multi-turn
    agent reasoning (tool calls, follow-ups within a single user message).
    """

    async def create(self, messages, *, tools=[], tool_choice="auto",
                     json_output=None, extra_create_args={},
                     cancellation_token=None):
        from src.core.openrouter_client import wait_before_openrouter_request
        await wait_before_openrouter_request()
        return await super().create(
            messages, tools=tools, tool_choice=tool_choice,
            json_output=json_output, extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )

    async def create_stream(self, messages, *, tools=[], tool_choice="auto",
                           json_output=None, extra_create_args={},
                           cancellation_token=None,
                           max_consecutive_empty_chunk_tolerance=0,
                           include_usage=None):
        from src.core.openrouter_client import wait_before_openrouter_request
        await wait_before_openrouter_request()
        async for chunk in super().create_stream(
            messages, tools=tools, tool_choice=tool_choice,
            json_output=json_output, extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
            max_consecutive_empty_chunk_tolerance=max_consecutive_empty_chunk_tolerance,
            include_usage=include_usage,
        ):
            yield chunk


def _create_openrouter_client(
    model_id: str,
    api_key: str,
    temperature: float = 0.0
) -> RateLimitedOpenAIChatCompletionClient:
    """
    Create a rate-limited OpenAI-compatible client pointing to OpenRouter.

    Uses RateLimitedOpenAIChatCompletionClient to enforce a 3-second delay
    between ALL API calls (not just per user message), preventing 429 errors
    during multi-turn agent reasoning.

    Args:
        model_id: OpenRouter model ID (e.g., "openai/gpt-oss-120b:free")
        api_key: OpenRouter API key
        temperature: Sampling temperature

    Returns:
        RateLimitedOpenAIChatCompletionClient configured for OpenRouter
    """
    from src.core.openrouter_client import OPENROUTER_BASE_URL

    return RateLimitedOpenAIChatCompletionClient(
        model=model_id,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="unknown",
        ),
        extra_headers={
            "HTTP-Referer": "https://github.com/vladturlo/OptiMat-Chat",
            "X-Title": "OptiMat Alloys",
        }
    )


async def create_free_model_client(
    temperature: float = 0.0,
    model_id: str = "z-ai/glm-4.5-air:free"
) -> tuple[Any, str]:
    """
    Create a model client for a specific free OpenRouter model.

    Args:
        temperature: Sampling temperature
        model_id: Specific OpenRouter model ID to use

    Returns:
        Tuple of (model_client, model_id)

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set
    """
    from src.core.openrouter_client import get_model_display_name

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Get your free API key from: https://openrouter.ai/keys"
        )

    print(f"✓ Using free model: {get_model_display_name(model_id)}", flush=True)

    # Create client for the specified model
    client = _create_openrouter_client(model_id, api_key, temperature)

    return client, model_id


# Model metadata for UI display and documentation
MODEL_INFO = {
    # OpenRouter free models (individual entries)
    "openai/gpt-oss-120b:free": {
        "name": "GPT-OSS 120B (Free)",
        "description": "Free 117B model via OpenRouter (best free option)",
        "context_window": "128K tokens",
        "use_case": "General purpose, excellent tool calling",
        "provider": "openrouter"
    },
    "openai/gpt-oss-20b:free": {
        "name": "GPT-OSS 20B (Free)",
        "description": "Free 21B model via OpenRouter (faster, smaller)",
        "context_window": "128K tokens",
        "use_case": "Quick queries, lighter tasks",
        "provider": "openrouter"
    },
    "qwen/qwen3-coder:free": {
        "name": "Qwen3 Coder 480B (Free)",
        "description": "Free 480B MoE coder model (optimized for tool calling)",
        "context_window": "256K tokens",
        "use_case": "Agentic coding, function calling, tool use",
        "provider": "openrouter"
    },
}

# List of free model IDs for easy reference (ordered by recommendation)
# Used as fallback when dynamic discovery fails
FREE_MODELS = [
    "z-ai/glm-4.5-air:free",       # Excellent (10/10) - 106B MoE, great tool calling
    "openai/gpt-oss-120b:free",    # Recommended (9/10)
    "openai/gpt-oss-20b:free",     # Good (9/10) - fastest, generous limit
    "qwen/qwen3-coder:free",       # Good (9/10) - 50 req/day limit
]


def get_model_info(model_name: str) -> str:
    """
    Get human-readable description of a model.

    Args:
        model_name: Model identifier

    Returns:
        Descriptive string about the model

    Examples:
        >>> get_model_info("gpt-4.1-mini")
        'Balanced model (recommended, 2x faster, 83% cheaper)'
    """
    return MODEL_INFO.get(model_name, {}).get("description", "GPT-4.1 family model")


def get_available_models(include_free: bool = True) -> list[str]:
    """
    Get list of available model identifiers for UI dropdown.

    Args:
        include_free: Whether to include free OpenRouter models

    Returns:
        List of model identifiers
    """
    if include_free:
        return list(FREE_MODELS)
    return []


def get_model_display_options() -> dict[str, str]:
    """
    Get model options formatted for Chainlit Select widget.

    Returns:
        Dictionary mapping model ID to display name

    Examples:
        >>> get_model_display_options()
        {'gpt-4.1': 'GPT-4.1', 'gpt-4.1-mini': 'GPT-4.1 Mini', ...}
    """
    return {
        model_id: info["name"]
        for model_id, info in MODEL_INFO.items()
    }


def has_openrouter_key() -> bool:
    """
    Check if OpenRouter API key is configured.

    Returns:
        True if OPENROUTER_API_KEY environment variable is set
    """
    return bool(os.environ.get("OPENROUTER_API_KEY"))


async def validate_free_models(api_key: Optional[str] = None) -> list[str]:
    """
    Validate FREE_MODELS against OpenRouter API.

    Checks which models from the static FREE_MODELS list are currently
    available on OpenRouter. Returns available models in preferred order.
    Falls back to static list if validation fails.

    Args:
        api_key: Optional OpenRouter API key. If not provided,
            uses OPENROUTER_API_KEY environment variable.

    Returns:
        List of available model IDs in preferred order.
        Falls back to FREE_MODELS if API call fails.

    Examples:
        >>> models = await validate_free_models()
        >>> print(models[0])  # First available model (default)
        'z-ai/glm-4.5-air:free'
    """
    from src.core.openrouter_client import discover_free_models

    try:
        # Get all available free models from OpenRouter
        discovered = await discover_free_models(api_key)
        discovered_set = set(discovered)

        # Filter static list to only include available models
        available = [m for m in FREE_MODELS if m in discovered_set]

        if available:
            # Show which specific models are available
            available_names = [m.split('/')[1].replace(':free', '') for m in available]
            print(f"✓ Validated {len(available)}/{len(FREE_MODELS)} free models: {', '.join(available_names)}", flush=True)
            return available
        else:
            print("⚠ No tested free models available, using static list", flush=True)
    except Exception as e:
        print(f"⚠ Free model validation failed ({e}), using static list", flush=True)

    return FREE_MODELS


def create_unified_model_client(
    provider: Literal["openrouter", "ollama"],
    model_name: str,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Any:
    """
    Create model client for OpenRouter or Ollama provider.

    Args:
        provider: Model provider:
            - "openrouter" for free cloud models
            - "ollama" for local/cloud inference
        model_name: Model identifier appropriate for the provider:
            - OpenRouter: "openai/gpt-oss-120b:free", etc.
            - Ollama: "qwen2.5:14b", "gpt-oss:120b-cloud", etc.
        temperature: Sampling temperature (0.0 = deterministic)
        **kwargs: Additional provider-specific options (e.g., Ollama host)

    Returns:
        ChatCompletionClient configured for the specified provider/model

    Raises:
        ValueError: If provider is not recognized
    """
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your free API key from: https://openrouter.ai/keys"
            )
        return _create_openrouter_client(model_name, api_key, temperature)

    elif provider == "ollama":
        from src.agents.local_models.ollama_factory import create_ollama_client
        return create_ollama_client(
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Supported providers: 'openrouter', 'ollama'"
        )


def get_unified_model_info(provider: str, model_name: str) -> str:
    """
    Get description for any provider's model.

    Args:
        provider: Model provider ("openrouter" or "ollama")
        model_name: Model identifier

    Returns:
        Human-readable description of the model
    """
    if provider == "openrouter":
        return get_model_info(model_name)
    elif provider == "ollama":
        from src.agents.local_models.ollama_factory import get_ollama_model_info
        return get_ollama_model_info(model_name)
    return f"Unknown model: {model_name}"


async def get_all_available_models() -> list[dict]:
    """
    Get all available models from OpenAI, OpenRouter, and Ollama.

    This function builds a unified list of models for UI dropdown display.
    It always includes:
    1. Ollama cloud models (most reliable, no rate limits) - shown first
    2. Installed local Ollama models
    3. The preferred Ollama model (even if not installed, marked accordingly)
    4. Free OpenRouter models (fallback, may hit provider rate limits)

    Returns:
        List of dicts with:
        - id: Unique identifier for dropdown value (e.g., "gpt-4.1 (OpenAI)")
        - provider: "openai", "openrouter", or "ollama"
        - model_name: Raw model name for API calls
        - display_name: Human-readable name for dropdown
        - description: Brief description
        - is_installed: True if model is ready to use
    """
    from src.utils.ollama_manager import (
        is_ollama_installed,
        get_available_ollama_models,
    )
    from src.agents.local_models.ollama_config import (
        PREFERRED_OLLAMA_MODEL,
        OLLAMA_MODELS,
    )
    from src.core.openrouter_client import discover_free_models

    models = []

    # 1. Add Ollama models first (most reliable — no provider rate limits)
    if is_ollama_installed():
        available_ollama = get_available_ollama_models()
        installed_names = {m["name"] for m in available_ollama}

        # Collect cloud model names to avoid duplicating them in the installed list
        cloud_model_names = {mid for mid, info in OLLAMA_MODELS.items() if info.get("cloud")}

        # 1a. Cloud models always first (regardless of install status)
        for model_id, info in OLLAMA_MODELS.items():
            if info.get("cloud"):
                models.append({
                    "id": f"{model_id} (Ollama)",
                    "provider": "ollama",
                    "model_name": model_id,
                    "display_name": f"{info.get('name', model_id)} (Ollama Cloud)",
                    "description": info.get("description", "Cloud-hosted model"),
                    "is_installed": True,
                })

        # 1b. Installed local models (skip cloud models already listed above)
        for model in available_ollama:
            if model["name"] in cloud_model_names:
                continue
            models.append({
                "id": f"{model['name']} (Ollama)",
                "provider": "ollama",
                "model_name": model["name"],
                "display_name": f"{model['display_name']} (Ollama)",
                "description": model["description"],
                "is_installed": True,
            })

        # 1c. Always show preferred model even if not installed
        if PREFERRED_OLLAMA_MODEL not in installed_names:
            pref_info = OLLAMA_MODELS.get(PREFERRED_OLLAMA_MODEL, {})
            models.append({
                "id": f"{PREFERRED_OLLAMA_MODEL} (Ollama)",
                "provider": "ollama",
                "model_name": PREFERRED_OLLAMA_MODEL,
                "display_name": f"{pref_info.get('name', PREFERRED_OLLAMA_MODEL)} (Ollama) [Click to Install]",
                "description": pref_info.get("description", "Preferred model - will auto-install"),
                "is_installed": False,
            })

    # 2. Add free OpenRouter models (fallback — may hit provider rate limits)
    if has_openrouter_key():
        discovered_ids = set(await discover_free_models(
            api_key=os.environ.get("OPENROUTER_API_KEY")
        ))

        for model_id in FREE_MODELS:
            if model_id not in discovered_ids:
                continue  # Skip models no longer on OpenRouter
            info = MODEL_INFO.get(model_id, {})
            display_name = info.get("name", model_id)
            description = info.get("description", "Free model via OpenRouter")

            models.append({
                "id": f"{model_id} (OpenRouter Free)",
                "provider": "openrouter",
                "model_name": model_id,
                "display_name": f"{display_name} (OpenRouter Free)",
                "description": description,
                "is_installed": True,
            })

    return models


def parse_model_selection(selection: str) -> tuple[str, str]:
    """
    Parse unified dropdown selection into provider and model name.

    Args:
        selection: Dropdown value like "gpt-4.1 (OpenAI)", "gpt-oss:20b (Ollama)",
                   or "openai/gpt-oss-120b:free (OpenRouter Free)"

    Returns:
        Tuple of (provider, model_name)
        - provider: "openai", "openrouter", or "ollama"
        - model_name: Raw model name for API calls

    Examples:
        >>> parse_model_selection("gpt-4.1 (OpenAI)")
        ('openai', 'gpt-4.1')
        >>> parse_model_selection("gpt-oss:20b (Ollama)")
        ('ollama', 'gpt-oss:20b')
        >>> parse_model_selection("openai/gpt-oss-120b:free (OpenRouter Free)")
        ('openrouter', 'openai/gpt-oss-120b:free')
        >>> parse_model_selection("GPT-OSS 20B (Ollama) [Not Installed]")
        ('ollama', 'GPT-OSS 20B')
    """
    if " (OpenRouter Free)" in selection:
        # Handle "model (OpenRouter Free)" format
        model_part = selection.split(" (OpenRouter Free)")[0]
        return "openrouter", model_part
    elif " (Ollama)" in selection:
        # Handle both "model (Ollama)" and "model (Ollama) [Not Installed]"
        model_part = selection.split(" (Ollama)")[0]
        return "ollama", model_part
    else:
        # Fallback: assume OpenRouter
        return "openrouter", selection
