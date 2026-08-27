# Local Models Integration

This module provides support for local LLM inference using [Ollama](https://ollama.com), enabling:

- **Offline operation** - No internet required after model download
- **Cost-free inference** - No API costs for local models
- **Data privacy** - All data stays on your machine
- **Experimentation** - Test different models without API limits

## Prerequisites

### 1. Install Ollama

```bash
# Linux/WSL
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# Download from https://ollama.com/download
```

### 2. Start Ollama Server

```bash
ollama serve
```

### 3. Pull Models

```bash
# Priority 1: Best for GPT-4.1-mini competitive performance
ollama pull mistral-small:24b    # ~14GB VRAM, excellent tool calling
ollama pull qwen2.5:32b          # ~16GB VRAM, best reasoning
ollama pull qwen3:32b            # ~16GB VRAM, latest with streaming

# Priority 2: Fast fallbacks
ollama pull llama3.1:8b          # ~5GB VRAM, official tool support
ollama pull qwen2.5:14b          # ~8GB VRAM, good balance
```

### 4. Install Python Dependencies

```bash
pip install 'agent-framework-openai>=1.8.1'
```

The client talks to the Ollama daemon's OpenAI-compatible endpoint
(`{host}/v1`); no Ollama-specific SDK is needed.

## Usage

### Basic Usage

```python
from src.agents.local_models import create_ollama_client

# Create client with default model (qwen2.5:14b)
client = create_ollama_client()

# Create client for specific model
client = create_ollama_client("mistral-small:24b")
```

### With Agent Factory

```python
from src.agents.factory import AgentFactory
from src.agents.local_models import create_ollama_client

# Create Ollama client
model_client = create_ollama_client("qwen2.5:14b")

# Use with existing agent factory
agent = AgentFactory.create_scientist(
    model_client=model_client,
    tools=[...],
)
```

### Check Ollama Status

```python
from src.agents.local_models.ollama_factory import (
    check_ollama_available,
    list_available_models,
)

# Check if Ollama is running
if check_ollama_available():
    print("Ollama is ready!")

    # List available models
    models = list_available_models()
    print(f"Available models: {models}")
```

## Supported Models

| Model | VRAM | Tool Calling | Best For |
|-------|------|--------------|----------|
| `mistral-small:24b` | ~14GB | Native | Tool orchestration |
| `qwen2.5:32b` | ~16GB | Native | Complex reasoning |
| `qwen3:32b` | ~16GB | Native | Latest capabilities |
| `llama3.1:8b` | ~5GB | Native | Fast development |
| `qwen2.5:14b` | ~8GB | Native | Balanced performance |

## VRAM Requirements

Models use Q4_K_M quantization by default. Actual VRAM usage may vary.

| GPU VRAM | Recommended Models |
|----------|-------------------|
| 8GB | `llama3.1:8b` |
| 12GB | `qwen2.5:14b`, `llama3.1:8b` |
| 16GB | All models (Q4 quantization) |
| 24GB+ | All models (higher precision) |

## Troubleshooting

### "Cannot connect to Ollama"

1. Check if Ollama is running: `ollama serve`
2. Verify the port: default is `http://localhost:11434`
3. Check firewall settings

### "Model not found"

1. Pull the model first: `ollama pull <model_name>`
2. Verify with: `ollama list`

### Out of Memory

1. Use a smaller model
2. Close other GPU applications
3. Try a lower context window: set `OLLAMA_CONTEXT_LENGTH` on the daemon

### Slow Inference

1. Ensure GPU is being used: `nvidia-smi`
2. Check Ollama GPU support: `ollama run <model> --verbose`
3. Consider smaller model for development

## Configuration Options

```python
client = create_ollama_client(
    model_name="qwen2.5:14b",
    host="http://localhost:11434",  # Custom host
)
```

Sampling temperature is configured on the agent (default 0.0 via
`AgentFactory`), not the client. Runtime options such as context window,
GPU count, and threads are configured daemon-side (e.g.
`OLLAMA_CONTEXT_LENGTH`, `OLLAMA_NUM_PARALLEL` environment variables).

## Testing

```bash
pytest tests/unit/test_model_factory.py -k Ollama -v
```

14 offline unit tests — no daemon, no API key, no network. They cover client
construction and routing: which endpoint a model resolves to, host
normalization, how the API key reaches the auth header, that OpenRouter's
rate-limit pacing is not applied here, and cloud-vs-local dispatch.

There is no integration suite. `tests/test_ollama_integration.py` — which
checked the `ollama_config` helpers and probed a live daemon — was deleted in
`d11d023`. To read it:

```bash
git show d11d023^:tests/test_ollama_integration.py
```

## References

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Tool Calling](https://ollama.com/blog/tool-support)
- [Agent Framework Ollama Provider](https://learn.microsoft.com/en-us/agent-framework/agents/providers/ollama)
