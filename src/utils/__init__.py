"""
Utility modules for OptiMat Alloys.

This package provides platform detection, path handling, and other utilities.
"""

from .platform import (
    detect_platform,
    is_wsl2,
    get_platform_info,
    check_cuda_availability,
)

from .ollama_manager import (
    is_ollama_installed,
    ensure_ollama_running,
    get_available_ollama_models,
    get_recommended_models,
    get_model_vram_gb,
)

__all__ = [
    "detect_platform",
    "is_wsl2",
    "get_platform_info",
    "check_cuda_availability",
    "is_ollama_installed",
    "ensure_ollama_running",
    "get_available_ollama_models",
    "get_recommended_models",
    "get_model_vram_gb",
]
