"""
Ollama model configuration for OptiMat Alloys.

This module defines supported Ollama models with their capabilities,
VRAM requirements, and recommended use cases.

Models are selected based on:
1. Native tool/function calling support (required for agent tools)
2. VRAM requirements (optimized for 16GB GPUs)
3. Reasoning and coding capabilities (comparable to GPT-4.1-mini)
"""

# Ollama model definitions
# Models tested and validated for tool calling with AutoGen
OLLAMA_MODELS = {
    # ==========================================================================
    # Cloud-hosted models (no local VRAM needed)
    # ==========================================================================
    "gpt-oss:120b-cloud": {
        "name": "GPT-OSS 120B Cloud (OpenAI Open Source)",
        "function_calling": True,
        "vram_gb": 0,  # Cloud-hosted, no local VRAM needed
        "context_window": 128000,
        "description": "120B via Ollama Cloud (MXFP4 quantized). Multi-turn tool-calling verified end-to-end.",
        "use_case": "Multi-turn agentic tasks, function calling, PDF report generation. No local VRAM required.",
        "cloud": True,
        "quantization": "MXFP4",
    },
    # ==========================================================================
    # Priority 1: Best candidates for GPT-4.1-mini competitive performance
    # ==========================================================================
    "gpt-oss:20b": {
        "name": "GPT-OSS 20B (OpenAI Open Source)",
        "function_calling": True,
        "vram_gb": 12,  # Q4 quantization
        "context_window": 128000,
        "description": "OpenAI's open-source reasoning model with native tool calling",
        "use_case": "Agentic tasks, function calling, structured outputs — if multi-turn reliability suffers at this quantized local size, switch to gpt-oss:120b-cloud.",
        "quantization": "Q4_K_M",
    },
    "mistral-small:24b": {
        "name": "Mistral Small 24B",
        "function_calling": True,
        "vram_gb": 14,  # Q4 quantization
        "context_window": 32000,
        "description": "Excellent tool calling (community recommended)",
        "use_case": "Best for complex tool orchestration",
        "quantization": "Q4_K_M",
    },
    "qwen2.5:32b": {
        "name": "Qwen 2.5 32B",
        "function_calling": True,
        "vram_gb": 16,  # Q4 quantization
        "context_window": 128000,
        "description": "Best reasoning in class",
        "use_case": "Complex reasoning and scientific analysis",
        "quantization": "Q4_K_M",
    },
    "qwen3:32b": {
        "name": "Qwen 3 32B",
        "function_calling": True,
        "vram_gb": 16,  # Q4 quantization
        "context_window": 128000,
        "description": "Latest, streaming tool support",
        "use_case": "Cutting-edge capabilities with streaming",
        "quantization": "Q4_K_M",
    },
    "qwen3:4b": {
        "name": "Qwen 3 4B",
        "function_calling": True,
        "vram_gb": 2.5,  # Q4 quantization
        "context_window": 128000,
        "description": "Tiny model rivaling Qwen2.5-72B performance",
        "use_case": "CPU-friendly agent tasks, memory-constrained setups",
        "quantization": "Q4_K_M",
    },
    # ==========================================================================
    # Priority 2: Reliable fallbacks (faster, less VRAM)
    # ==========================================================================
    "qwen3:8b": {
        "name": "Qwen 3 8B",
        "function_calling": True,
        "vram_gb": 5,  # Q4 quantization
        "context_window": 128000,
        "description": "Balanced Qwen3 with native tool calling",
        "use_case": "Good reasoning with moderate VRAM usage",
        "quantization": "Q4_K_M",
    },
    "llama3.1:8b": {
        "name": "Llama 3.1 8B",
        "function_calling": True,
        "vram_gb": 5,
        "context_window": 128000,
        "description": "Official Ollama tool support",
        "use_case": "Fast inference, development testing",
        "quantization": "Q4_K_M",
    },
    "qwen2.5:14b": {
        "name": "Qwen 2.5 14B",
        "function_calling": True,
        "vram_gb": 8,
        "context_window": 128000,
        "description": "Good balance of speed and capability",
        "use_case": "Balanced performance for daily use",
        "quantization": "Q4_K_M",
    },
    # ==========================================================================
    # Vision-Language Models (Multimodal)
    # ==========================================================================
    "qwen3-vl:2b": {
        "name": "Qwen 3 VL 2B",
        "function_calling": True,
        "vram_gb": 2,
        "context_window": 256000,
        "description": "Vision-language model for text and image analysis",
        "use_case": "Analyze structure visualizations, interpret plots",
        "quantization": "Q4_K_M",
        "multimodal": True,
    },
    "qwen3-vl:4b": {
        "name": "Qwen 3 VL 4B",
        "function_calling": True,
        "vram_gb": 3,
        "context_window": 256000,
        "description": "Vision-language model with improved reasoning",
        "use_case": "Crystal structure analysis, complex plot interpretation",
        "quantization": "Q4_K_M",
        "multimodal": True,
    },
    "qwen3-vl:8b": {
        "name": "Qwen 3 VL 8B",
        "function_calling": True,
        "vram_gb": 6,
        "context_window": 256000,
        "description": "Strongest VL model for advanced image understanding",
        "use_case": "Detailed structural analysis, publication figure interpretation",
        "quantization": "Q4_K_M",
        "multimodal": True,
    },
}

# Default model for testing
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"

# Preferred model for auto-pull when user selects Ollama
# This model will be shown in the dropdown even if not installed,
# and will be automatically downloaded when selected
PREFERRED_OLLAMA_MODEL = "gpt-oss:20b"

# Ollama server configuration
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def get_models_by_vram(max_vram_gb: float) -> dict:
    """
    Filter models that fit within specified VRAM limit.

    Args:
        max_vram_gb: Maximum available VRAM in GB

    Returns:
        Dictionary of models that fit within VRAM limit
    """
    return {
        name: info
        for name, info in OLLAMA_MODELS.items()
        if info["vram_gb"] <= max_vram_gb
    }


def get_models_with_tool_calling() -> dict:
    """
    Get all models that support native tool/function calling.

    Returns:
        Dictionary of models with tool calling support
    """
    return {
        name: info
        for name, info in OLLAMA_MODELS.items()
        if info["function_calling"]
    }


def get_model_names() -> list[str]:
    """
    Get list of all available model names.

    Returns:
        List of model identifiers for Ollama
    """
    return list(OLLAMA_MODELS.keys())
