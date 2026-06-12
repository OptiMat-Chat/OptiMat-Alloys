"""
OpenRouter API client for free model support.

This module handles:
- Dynamic model discovery from OpenRouter API
- Rate limiting with auto-wait (20 req/min, 50-1000 req/day)
- Model selection and fallback logic

The /api/v1/models endpoint is FREE (no token cost) - it's just a model listing API.

Rate Limits:
- 20 requests per minute (all users)
- 50 requests per day (without $10+ credits)
- 1,000 requests per day (with $10+ credits purchased)

Usage:
    from src.core.openrouter_client import discover_free_models, create_openrouter_client

    # Discover available free models (called on session start)
    models = await discover_free_models()
    # Returns: ["openai/gpt-oss-120b:free", "openai/gpt-oss-20b:free", ...]

    # Create client for specific model
    client = create_openrouter_client(models[0], api_key)
"""

import os
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Optional
from openai import AsyncOpenAI


# OpenRouter API configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default free models (fallback if discovery fails) - ordered by recommendation
DEFAULT_FREE_MODELS = [
    "z-ai/glm-4.5-air:free",       # Excellent (10/10) - 106B MoE, great tool calling
    "openai/gpt-oss-120b:free",    # Recommended (9/10)
    "openai/gpt-oss-20b:free",     # Good (9/10) - fastest, generous limit
    "qwen/qwen3-coder:free",       # Good (9/10) - 50 req/day limit
]

# Selection criteria for free models
FREE_MODEL_CRITERIA = {
    "max_price": 0,
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "supported_parameters": ["tools"],
}

# Rate limit configuration
MAX_REQUESTS_PER_MINUTE = 20
DAILY_LIMIT_WITH_CREDITS = 1000
DAILY_LIMIT_WITHOUT_CREDITS = 50

# Delay between requests to avoid hitting 20 req/min limit
# 3 seconds = max 20 req/min, provides safety margin
OPENROUTER_REQUEST_DELAY_SECONDS = 3.0

# Cache for discovered models
_discovered_models: list[dict] = []
_discovery_time: Optional[datetime] = None
CACHE_DURATION = timedelta(hours=1)

# Rate limit tracking
_daily_request_count = 0
_last_reset_date: Optional[datetime] = None
_minute_requests: list[datetime] = []
_last_request_time: Optional[datetime] = None  # For delay enforcement


class FreeModelsUnavailableError(Exception):
    """Raised when no free models are available."""
    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit retries are exhausted."""
    pass


# =============================================================================
# Model Discovery
# =============================================================================

async def discover_free_models(
    api_key: Optional[str] = None,
    force_refresh: bool = False
) -> list[str]:
    """
    Query OpenRouter API to discover available free models.

    Called once per session start. Results are cached for 1 hour.

    Cost: FREE (model listing endpoint, not LLM inference)
    Time: ~200-500ms

    Args:
        api_key: Optional OpenRouter API key (not required for model listing)
        force_refresh: Force re-discovery even if cache is fresh

    Returns:
        List of model IDs sorted by preference, e.g.:
        ["openai/gpt-oss-120b:free", "openai/gpt-oss-20b:free", "meta-llama/llama-3.3-70b-instruct:free"]
    """
    global _discovered_models, _discovery_time

    # Return cached results if fresh
    if not force_refresh and _discovery_time:
        if datetime.now() - _discovery_time < CACHE_DURATION:
            return [m["id"] for m in _discovered_models]

    print("[OpenRouter] Discovering available free models...", flush=True)

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        # Use requests (synchronous) - fast enough for model listing (~300ms)
        # Wrapped in to_thread for async compatibility
        def _fetch_models():
            resp = requests.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers=headers,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()

        data = await asyncio.to_thread(_fetch_models)

    except requests.Timeout:
        print("[OpenRouter] Discovery timed out, using defaults", flush=True)
        return DEFAULT_FREE_MODELS
    except Exception as e:
        print(f"[OpenRouter] Discovery error: {e}, using defaults", flush=True)
        return DEFAULT_FREE_MODELS

    # Filter models matching our criteria
    free_models = []
    for model in data.get("data", []):
        if _matches_criteria(model):
            free_models.append(model)

    if not free_models:
        print("[OpenRouter] No models match criteria, using defaults", flush=True)
        return DEFAULT_FREE_MODELS

    # Rank by preference
    _discovered_models = _rank_models(free_models)
    _discovery_time = datetime.now()

    model_ids = [m["id"] for m in _discovered_models]
    print(f"[OpenRouter] Found {len(model_ids)} free models with tool support", flush=True)

    return model_ids


def get_discovered_models_data() -> dict[str, dict]:
    """
    Return cached model metadata from the last discover_free_models() call.

    Keys are model IDs, values contain name, description, and context_length
    as returned by the OpenRouter API. Returns empty dict if discovery hasn't
    run yet or returned no results.
    """
    return {
        m["id"]: {
            "name": m.get("name", m["id"]),
            "description": m.get("description", ""),
            "context_length": m.get("context_length", 0),
        }
        for m in _discovered_models
    }


def _matches_criteria(model: dict) -> bool:
    """Check if model matches all our selection criteria."""
    # Criterion 1 & 2: Free pricing
    pricing = model.get("pricing", {})
    prompt_price = pricing.get("prompt", "1")
    completion_price = pricing.get("completion", "1")

    # Handle both string "0" and numeric 0
    if str(prompt_price) != "0" or str(completion_price) != "0":
        return False

    # Criterion 1: Text input/output
    arch = model.get("architecture", {})
    input_mod = arch.get("input_modalities", [])
    output_mod = arch.get("output_modalities", [])

    if "text" not in input_mod or "text" not in output_mod:
        return False

    # Criterion 3: Tool/function calling support
    supported_params = model.get("supported_parameters", [])
    if "tools" not in supported_params:
        return False

    return True


def _rank_models(models: list[dict]) -> list[dict]:
    """
    Rank models by preference. Higher score = better.

    Ranking criteria: OpenAI models first, then by model size and context length.
    """
    def score(model: dict) -> float:
        model_id = model["id"].lower()
        s = 0.0

        # Criterion 5: Prefer OpenAI models (best SDK compatibility)
        if model_id.startswith("openai/"):
            s += 1000

        # Prefer well-known providers
        if "meta-llama" in model_id or "llama" in model_id:
            s += 500
        if "mistral" in model_id:
            s += 400
        if "qwen" in model_id:
            s += 300

        # Criterion 4: Prefer larger models (estimate from name)
        if "120b" in model_id:
            s += 120
        elif "70b" in model_id:
            s += 70
        elif "30b" in model_id or "27b" in model_id:
            s += 30
        elif "20b" in model_id:
            s += 20
        elif "7b" in model_id or "8b" in model_id:
            s += 8

        # Prefer larger context
        context = model.get("context_length", 0)
        s += context / 10000  # e.g., 131K context adds 13.1 points

        return s

    return sorted(models, key=score, reverse=True)


# =============================================================================
# Rate Limiting
# =============================================================================

def get_exact_wait_time() -> int:
    """
    Calculate exact seconds to wait until per-minute rate limit allows next request.

    Returns:
        Number of seconds to wait (0 if no wait needed)
    """
    global _minute_requests

    now = datetime.now()

    # Clean up requests older than 1 minute
    _minute_requests = [t for t in _minute_requests if (now - t).total_seconds() < 60]

    if len(_minute_requests) >= MAX_REQUESTS_PER_MINUTE - 1:
        # Calculate wait time until oldest request expires
        oldest = _minute_requests[0]
        seconds_since_oldest = (now - oldest).total_seconds()
        wait_time = max(0, 61 - seconds_since_oldest)
        return int(wait_time)

    return 0


def _track_request():
    """Track request for both daily and per-minute limits."""
    global _daily_request_count, _last_reset_date, _minute_requests

    now = datetime.now()
    today = now.date()

    # Daily tracking
    if _last_reset_date != today:
        _daily_request_count = 0
        _last_reset_date = today
    _daily_request_count += 1

    # Per-minute tracking
    _minute_requests.append(now)


def get_daily_usage() -> tuple[int, int]:
    """
    Get current daily usage stats.

    Returns:
        Tuple of (requests_made, limit)
    """
    return _daily_request_count, DAILY_LIMIT_WITH_CREDITS


async def _wait_for_rate_limit():
    """Wait if we're approaching per-minute limit."""
    wait_time = get_exact_wait_time()
    if wait_time > 0:
        print(f"[Rate limit] Waiting {wait_time}s before request...", flush=True)
        await asyncio.sleep(wait_time)


async def wait_before_openrouter_request():
    """
    Wait if needed to enforce delay between OpenRouter requests.

    This prevents hitting the 20 requests/minute rate limit by ensuring
    at least OPENROUTER_REQUEST_DELAY_SECONDS between consecutive calls.

    Should be called before each OpenRouter API request (chat completions).
    """
    global _last_request_time

    if _last_request_time is not None:
        elapsed = (datetime.now() - _last_request_time).total_seconds()
        wait_time = OPENROUTER_REQUEST_DELAY_SECONDS - elapsed

        if wait_time > 0:
            print(f"[Rate limit] Waiting {wait_time:.1f}s before OpenRouter request...", flush=True)
            await asyncio.sleep(wait_time)

    _last_request_time = datetime.now()


# =============================================================================
# OpenRouter Client Creation
# =============================================================================

def create_openrouter_client(model_id: str, api_key: str) -> AsyncOpenAI:
    """
    Create an OpenAI-compatible async client pointing to OpenRouter.

    Args:
        model_id: The OpenRouter model ID (e.g., "openai/gpt-oss-120b:free")
        api_key: OpenRouter API key

    Returns:
        AsyncOpenAI client configured for OpenRouter
    """
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/OptiMat-Chat/OptiMat-Alloys",  # Optional: for rankings
            "X-Title": "OptiMat Alloys",  # Optional: for rankings
        }
    )


async def make_request_with_auto_wait(request_coro):
    """
    Execute an async request with automatic rate limit handling.

    This function:
    1. Checks if we need to wait (per-minute limit)
    2. Makes the request
    3. Handles 429 errors with automatic retry

    Args:
        request_coro: The coroutine to execute (e.g., client.chat.completions.create(...))

    Returns:
        The response from the request

    Example:
        response = await make_request_with_auto_wait(
            client.chat.completions.create(model=model, messages=messages)
        )
    """
    # Check if we need to wait (per-minute limit)
    await _wait_for_rate_limit()

    # Track this request
    _track_request()

    try:
        response = await request_coro
        return response

    except Exception as e:
        error_str = str(e).lower()

        # Handle rate limit errors
        if "429" in error_str or "rate limit" in error_str:
            # Try to extract retry-after from error
            retry_after = 5  # Default wait

            print(f"[Rate limit] Server returned 429, waiting {retry_after}s...", flush=True)
            await asyncio.sleep(retry_after)

            # Retry the request
            return await make_request_with_auto_wait(request_coro)

        # Re-raise other errors
        raise


# =============================================================================
# Model Selection
# =============================================================================

async def select_best_free_model(
    api_key: Optional[str] = None,
    preferred_models: Optional[list[str]] = None
) -> Optional[str]:
    """
    Select the best available free model.

    Args:
        api_key: OpenRouter API key for discovery
        preferred_models: Optional list of preferred model IDs to check first

    Returns:
        Model ID of best available model, or None if none available
    """
    # Discover available models
    available = await discover_free_models(api_key)

    if not available:
        return None

    # If preferred models specified, try those first
    if preferred_models:
        for model in preferred_models:
            if model in available:
                return model

    # Return best available
    return available[0] if available else None


def get_model_display_name(model_id: str) -> str:
    """
    Get a user-friendly display name for a model.

    Args:
        model_id: The OpenRouter model ID

    Returns:
        Human-readable name
    """
    # Remove :free suffix and format nicely
    name = model_id.replace(":free", "")

    # Common transformations
    name_map = {
        "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B",
        "openai/gpt-oss-20b": "OpenAI GPT-OSS 20B",
        "meta-llama/llama-3.3-70b-instruct": "Meta Llama 3.3 70B",
    }

    return name_map.get(name, name)
