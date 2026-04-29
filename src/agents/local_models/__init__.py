"""
Local model integration for OptiMat Alloys.

This module provides support for local LLM inference using Ollama,
enabling offline operation and cost-free model experimentation.

Usage:
    from src.agents.local_models import create_ollama_client, OLLAMA_MODELS

    client = create_ollama_client("qwen2.5:14b")
"""

from .ollama_factory import create_ollama_client, get_ollama_model_info
from .ollama_config import OLLAMA_MODELS

__all__ = ["create_ollama_client", "get_ollama_model_info", "OLLAMA_MODELS"]
