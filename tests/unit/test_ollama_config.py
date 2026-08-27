"""The Ollama model registry must stay internally consistent: every entry
carries the fields the UI and factory read, the default model is one of them,
and the VRAM and tool-calling filters actually filter.

These helpers have no runtime callers — get_models_by_vram,
get_models_with_tool_calling and get_model_names are used by the settings UI and
by operators sizing a model to a GPU, not by the agent loop. Nothing else would
catch a regression in them, which is why they are covered here.

Ported from tests/test_ollama_integration.py, deleted in d11d023. The rest of
that file tested AutoGen's OllamaChatCompletionClient against a live daemon and
did not survive the Agent Framework migration; these five checks did, because
they only read the config module. To read what was dropped:

    git show d11d023^:tests/test_ollama_integration.py

Note on test_every_model_has_the_required_fields: four of those five fields are
already enforced the hard way. ollama_factory builds MODEL_INFO as a module-level
dict comprehension that subscripts name, description, context_window and vram_gb
directly, so a model missing any of them raises KeyError on import and the app
does not start. This test still states the invariant in one place, and it is the
only check on function_calling, which MODEL_INFO does not read.
"""

from src.agents.local_models.ollama_config import (
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_MODELS,
    get_models_by_vram,
    get_models_with_tool_calling,
)

REQUIRED_FIELDS = (
    "name",
    "function_calling",
    "vram_gb",
    "context_window",
    "description",
)


class TestRegistryShape:
    def test_models_defined(self):
        assert len(OLLAMA_MODELS) > 0

    def test_default_model_is_in_the_registry(self):
        assert DEFAULT_OLLAMA_MODEL in OLLAMA_MODELS

    def test_every_model_has_the_required_fields(self):
        for model_name, model_info in OLLAMA_MODELS.items():
            for field in REQUIRED_FIELDS:
                assert field in model_info, f"{model_name} missing {field}"


class TestFilters:
    def test_vram_filter_excludes_models_that_do_not_fit(self):
        models_8gb = get_models_by_vram(8)
        assert len(models_8gb) > 0
        for model_info in models_8gb.values():
            assert model_info["vram_gb"] <= 8

        # A larger budget can only admit more models, never fewer.
        models_16gb = get_models_by_vram(16)
        assert len(models_16gb) >= len(models_8gb)

    def test_tool_calling_filter_returns_only_tool_capable_models(self):
        tool_models = get_models_with_tool_calling()
        assert len(tool_models) > 0
        for model_info in tool_models.values():
            assert model_info["function_calling"] is True
