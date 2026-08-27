"""Model factories must produce Agent Framework chat clients with the exact
AutoGen-era wiring: OpenRouter base URL + attribution headers + rate-limit
middleware; Ollama via the daemon's OpenAI-compatible endpoint (which proxies
both local and cloud models, so one client class covers both providers).
"""

import pytest

from agent_framework.openai import OpenAIChatCompletionClient

from src.agents.middleware import openrouter_rate_limit
from src.agents.model_factory import create_unified_model_client
from src.agents.local_models.ollama_factory import create_ollama_client
from src.core.openrouter_client import OPENROUTER_BASE_URL

OR_MODEL = "z-ai/glm-4.5-air:free"


@pytest.fixture
def openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")


class TestOpenRouterClient:
    def test_returns_af_chat_completion_client(self, openrouter_key):
        client = create_unified_model_client("openrouter", OR_MODEL)
        assert isinstance(client, OpenAIChatCompletionClient)
        assert client.model == OR_MODEL
        assert client.base_url == OPENROUTER_BASE_URL

    def test_attribution_headers_unchanged(self, openrouter_key):
        client = create_unified_model_client("openrouter", OR_MODEL)
        assert client.default_headers == {
            "HTTP-Referer": "https://github.com/OptiMat-Chat/OptiMat-Alloys",
            "X-Title": "OptiMat Alloys",
        }

    def test_rate_limit_middleware_attached(self, openrouter_key):
        client = create_unified_model_client("openrouter", OR_MODEL)
        assert openrouter_rate_limit in client.chat_middleware

    def test_missing_key_error_message_unchanged(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError) as exc:
            create_unified_model_client("openrouter", OR_MODEL)
        assert str(exc.value) == (
            "OPENROUTER_API_KEY environment variable not set. "
            "Get your free API key from: https://openrouter.ai/keys"
        )


class TestOllamaClient:
    # NOTE: these assert on `client.auth_headers`, not `default_headers`.
    # openai <= 2.6.x put Authorization in default_headers; 2.54.0 moved it to
    # the separate auth_headers property, leaving default_headers with only
    # Accept / Content-Type / User-Agent / X-Stainless-*. The key still reaches
    # the request either way — only the attribute changed.
    def test_points_at_daemon_openai_endpoint(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        client = create_ollama_client("qwen3:4b")
        assert isinstance(client, OpenAIChatCompletionClient)
        assert client.model == "qwen3:4b"
        assert client.base_url == "http://localhost:11434/v1"

    def test_custom_host_respected(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        client = create_ollama_client("qwen3:4b", host="http://gpu-box:11434")
        assert client.base_url == "http://gpu-box:11434/v1"

    @pytest.mark.parametrize(
        "host",
        [
            "http://gpu-box:11434/",
            "http://gpu-box:11434/v1",
            "http://gpu-box:11434/v1/",
        ],
    )
    def test_host_normalization(self, monkeypatch, host):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        client = create_ollama_client("qwen3:4b", host=host)
        assert client.base_url == "http://gpu-box:11434/v1"

    def test_empty_env_key_falls_back_to_placeholder(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "")
        client = create_ollama_client("qwen3:4b")
        assert client.client.auth_headers["Authorization"] == "Bearer ollama"

    def test_api_key_from_env_reaches_auth_header(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "olm-secret")
        client = create_ollama_client("gpt-oss:120b-cloud")
        assert client.client.auth_headers["Authorization"] == "Bearer olm-secret"

    def test_placeholder_key_when_env_unset(self, monkeypatch):
        # Ollama needs no key for local models, but the OpenAI SDK requires
        # a non-empty value; "ollama" is the documented placeholder.
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        client = create_ollama_client("qwen3:4b")
        assert client.client.auth_headers["Authorization"] == "Bearer ollama"

    def test_no_openrouter_pacing_on_ollama(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        client = create_ollama_client("qwen3:4b")
        assert openrouter_rate_limit not in (client.chat_middleware or [])

    def test_cloud_model_with_api_key_goes_direct_to_ollama_cloud(self, monkeypatch):
        # The local daemon authenticates cloud proxying with a device key
        # (ollama signin), NOT the env API key — an API-key-only setup got a
        # runtime 401 in the AutoGen era too. With a key present, cloud
        # models talk to ollama.com's OpenAI-compatible endpoint directly.
        monkeypatch.setenv("OLLAMA_API_KEY", "olm-secret")
        client = create_ollama_client("gpt-oss:120b-cloud")
        assert client.base_url == "https://ollama.com/v1"
        assert client.client.auth_headers["Authorization"] == "Bearer olm-secret"

    def test_cloud_model_without_key_uses_daemon_device_auth(self, monkeypatch):
        # No API key: preserve the daemon proxy path (device key from
        # `ollama signin`), same as the AutoGen-era behavior.
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        client = create_ollama_client("gpt-oss:120b-cloud")
        assert client.base_url == "http://localhost:11434/v1"

    def test_local_model_with_key_still_uses_daemon(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "olm-secret")
        client = create_ollama_client("qwen3:4b")
        assert client.base_url == "http://localhost:11434/v1"


class TestUnifiedDispatch:
    def test_ollama_provider_dispatches(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        client = create_unified_model_client("ollama", "qwen3:4b")
        assert isinstance(client, OpenAIChatCompletionClient)
        assert client.base_url == "http://localhost:11434/v1"

    def test_unknown_provider_error_message_unchanged(self):
        with pytest.raises(ValueError) as exc:
            create_unified_model_client("anthropic", "claude")
        assert str(exc.value) == (
            "Unknown provider: anthropic. "
            "Supported providers: 'openrouter', 'ollama'"
        )

    def test_stale_kwargs_fail_loudly_on_openrouter(self, openrouter_key):
        # The temperature kwarg was removed (now Agent default_options);
        # stale call sites must crash, not silently drop the parameter.
        with pytest.raises(TypeError, match="temperature"):
            create_unified_model_client("openrouter", OR_MODEL, temperature=0.0)

    def test_stale_kwargs_fail_loudly_on_ollama(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        with pytest.raises(TypeError):
            create_unified_model_client("ollama", "qwen3:4b", temperature=0.0)
