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
pip install 'autogen-ext[ollama]'
```

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
3. Try lower context window: `options={"num_ctx": 4096}`

### Slow Inference

1. Ensure GPU is being used: `nvidia-smi`
2. Check Ollama GPU support: `ollama run <model> --verbose`
3. Consider smaller model for development

## Configuration Options

```python
client = create_ollama_client(
    model_name="qwen2.5:14b",
    host="http://localhost:11434",  # Custom host
    temperature=0.0,                 # Deterministic output
    options={
        "num_ctx": 8192,            # Context window size
        "num_gpu": 1,               # Number of GPUs
        "num_thread": 8,            # CPU threads
    }
)
```

## Testing

Run the integration tests:

```bash
pytest tests/test_ollama_integration.py -v
```

## References

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Tool Calling](https://ollama.com/blog/tool-support)
- [AutoGen Ollama Integration](https://microsoft.github.io/autogen/docs/tutorial/models)
