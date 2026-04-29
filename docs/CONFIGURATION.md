# Configuration Guide

This document provides detailed configuration information for OptiMat Alloys, including calculator selection, model configuration, and advanced settings.

## Calculator Selection

OptiMat Alloys supports eight ML interatomic potentials across four families, each with different accuracy/speed tradeoffs and elemental coverage. Selection is exposed in the Chainlit settings panel; the default is `orb-v3-conservative-inf-omat` (most accurate).

### Available Calculators

**ORB (Orbital Materials)** — runs in the main conda env:
- **`orb-v3-conservative-inf-omat`** (default): Most accurate, uses backpropagation for forces
- **`orb-v3-direct-20-omat`**: 2–3× faster, uses direct force calculation

**NequIP foundation models** — run in the separate `optimat-nequip` conda env (e3nn version conflict with the main env):
- **`nequip-oam-xl`**: F1 = 0.906, highest accuracy of the NequIP set
- **`nequip-oam-l`**: F1 = 0.893, trained on OMat24 + MPtrj + sAlex
- **`nequip-mp-l`**: F1 = 0.761, trained on MPtrj only

**MACE-MPA-0** (Matbench SOTA, 89 elements) — main env:
- **`mace-mpa-0-medium`**: Recommended for general materials

**MACE-OMAT-0** (best for phonons, 89 elements) — main env:
- **`mace-omat-0-medium`**
- **`mace-omat-0-small`**

The widget order, default index, and tooltip live in `run_chat.py` (the Calculator `Select` widget). Calculator dispatch and the `SupportedModel` type live in `src/core/calculators.py`; NequIP models are routed through `calculator_service.get_calculator()` to the separate environment.

### How to Select Calculator

#### During Chat Start

Calculator selection widget appears when starting a new chat:
- Located in Chainlit settings panel
- Selection stored in user session (`cl.user_session.set("default_calculator", ...)`)
- Reference data automatically precomputed if not available

#### During Active Session

Change calculator via Chainlit settings menu:
- New calculator applies to future structures only (immutable structures approach)
- User notified that existing structures remain unchanged
- Reference data checked for availability

### Immutable Structures Approach

Each structure remembers which calculator it was created with:
- `calculator_name` field stored in database metadata
- `calculate_elastic_properties` tool uses original calculator for consistency
- Prevents mixing results from different calculators
- Enables reproducibility and provenance tracking

**Benefits**:
- Reproducible results (always know which calculator was used)
- No confusion from mixing calculator results
- Safe to change default calculator without affecting existing structures

### Reference Data Versioning

Reference energies and lattice constants are calculator-specific to ensure accuracy.

#### File Naming Convention

```
energies_per_atom_{calculator}.json
lattice_constants_{calculator}.json
```

**Examples** (one pair per supported calculator):
```
energies_per_atom_orb_v3_conservative_inf_omat.json
energies_per_atom_orb_v3_direct_20_omat.json
energies_per_atom_nequip_oam_xl.json
energies_per_atom_mace_mpa_0_medium.json
energies_per_atom_mace_omat_0_medium.json
lattice_constants_orb_v3_conservative_inf_omat.json
lattice_constants_nequip_oam_xl.json
...
```

`src/storage/cache.py` builds these names by replacing `-` with `_` in the calculator key.

#### Metadata Validation

Each reference data file includes a JSON header with:
- Calculator name (e.g., "orb-v3-direct-20-omat")
- fmax convergence criterion (e.g., 0.005)
- Optimizer used (e.g., "FIRE")
- Timestamp of generation

**Example header**:
```json
{
  "metadata": {
    "calculator": "orb-v3-direct-20-omat",
    "fmax": 0.005,
    "optimizer": "FIRE",
    "timestamp": "2025-01-15T10:30:00Z"
  },
  "data": {
    "Cu": -3.721,
    "Ag": -2.855,
    ...
  }
}
```

#### Cache Management

The `ReferenceDataCache` in `src/storage/cache.py` handles versioning:
- Automatically selects correct file based on calculator name
- Validates metadata to ensure compatibility
- Raises error if reference data is invalid or missing

#### Lazy Evaluation

Reference data is computed on-demand:
- First time a calculator is used, reference data is generated
- Generation can take hours (117 elements × 5 structures, ~8-16 hours per calculator)
- Progress displayed to user during generation
- Subsequent uses load cached data instantly

### Code References

- **Settings UI / session storage / settings update**: `run_chat.py` — calculator Select widget, stored in `cl.user_session`, handled mid-session via the settings update callback.
- **Tool integration**: `generate_alloy_supercell` reads the calculator from session in `run_chat.py`.
- **Cache system**: `src/storage/cache.py` — `ReferenceDataCache` (versioned file handling and metadata validation).
- **Reference data**: `src/core/reference_data.py` — `precompute_and_save` (precomputation with cache support).

### Search by Calculator

In chat, mention the calculator and the agent will pass `calculator_name` to `search_database`. Shorthands `'mace'`, `'orb'`, `'nequip'` and full names are both accepted (resolution lives in `CALCULATOR_SHORTHANDS`, applied in `src/tools/database_search.py`).

Examples:
- *"elastic properties of CoCrFeNi with MACE"* → `search_database(composition_string='CoCrFeNi', calculator_name='mace')`
- *"Cu-Ag structures using orb-v3-conservative-inf-omat"* → `search_database(composition_string='Cu-Ag', calculator_name='orb-v3-conservative-inf-omat')`

Under the hood, `calculator_name` is stored in each row's ASE `key_value_pairs` (set when structures are created in `src/tools/alloy_generation.py` and `src/tools/recompute_structure.py`), so it can also be queried directly via the ASE database API if needed for scripting.

## Model Configuration

Model selection happens at runtime via the Chainlit settings dropdown rather than a static config file. The available models are declared in `OLLAMA_MODELS` (defined in `src/agents/local_models/ollama_config.py`, imported by `run_chat.py`), and the active selection is parsed by `parse_model_selection()` in `run_chat.py`. The default fallback is `gpt-oss:120b-cloud`. OpenRouter free-tier models (`z-ai/glm-4.5-air:free`, `openai/gpt-oss-120b:free`, `openai/gpt-oss-20b:free`, `qwen/qwen3-coder:free`) are listed in `src/core/openrouter_client.py`. Users can switch models mid-session.

API keys are taken from environment variables (`OLLAMA_API_KEY`, `OPENROUTER_API_KEY`) or entered through the in-app prompts and saved to `.env` (which is in `.gitignore`). See "Environment Variables" below for which key is needed when.

## Chainlit Settings

**File**: `.chainlit/config.toml`

### Session Timeouts

```toml
[project]
session_timeout = 86400          # 24 hours — keeps long QHA/elastic calculations alive across reloads
user_session_timeout = 1296000   # 15 days — preserves API keys and chat profile across browser sessions
```

### File Uploads

```toml
[features.spontaneous_file_upload]
enabled = true
accept = ["*/*"]                 # Any MIME type (CIF, POSCAR, XYZ, etc.)
max_files = 20
max_size_mb = 500
```

### Chain of Thought Display

```toml
[UI]
cot = "full"                # Shows all tool calls and reasoning
```

Options:
- `"full"`: Show all tool calls, reasoning, and intermediate steps
- `"tool_call"`: Show only tool calls
- `"hidden"`: Hide all internal reasoning (show only final output)

## Agent System Message

The default scientist prompt lives in `AgentFactory.DEFAULT_MESSAGES["scientist"]` in `src/agents/factory.py` (the dict starts around line 22; the prompt itself spans roughly lines 23–87). It's wired up by `AgentFactory.create_scientist()` at `src/agents/factory.py:96`.

### Current Configuration

The system message has grown well beyond plain scientific interpretation. It now covers:
- **Tool execution discipline**: call tools immediately, never ask for confirmation, never describe what you'll do.
- **Cached-data-first rule**: any property/comparison/ranking question must hit `search_database` before any compute tool.
- **Tool selection cheatsheet**: maps phrasings ("generate", "show", "compare", "rank") to specific tools.
- **Composition input format**: `composition_string` examples (`Cu-Ag`, `Ag75Cu25`, `Ag3Cu1`).
- **Calculator filtering**: pass `calculator_name` to `search_database` when the user names a calculator; shorthands `mace`/`orb`/`nequip` accepted.
- **Stability assessment thresholds**: PTM `structural_match_percent < 90%` → instability warning; `born_criterion_satisfied=False` → mechanical instability warning.
- **Calculator comparison routing**: regenerate via settings change vs. benchmark via `recompute_structure`.
- **Termination protocol**: always end answer responses with a follow-up question ending in `?` (required for the RoundRobinGroupChat termination check).

To inspect or tweak the prompt, read or edit `src/agents/factory.py` directly. `AgentFactory.get_default_scientist_message()` returns the same string programmatically.

## Tool Result Interpretation

### Settings

```python
# run_chat.py:633 (and line 905 for the secondary agent)
reflect_on_tool_use=False  # Disabled to avoid AutoGen bug #6328
```

The rationale comment is at `run_chat.py:610`. `parallel_tool_calls` is **not** set on the model client — sequential tool execution is achieved by the agent's system message ("Execute tools sequentially") and the RoundRobinGroupChat loop, not by an explicit OpenAI flag.

### Why This Approach

Instead of using AutoGen's built-in `reflect_on_tool_use=True` parameter, we achieve tool result interpretation through enhanced system message prompt engineering.

**Reason**: Avoids AutoGen issue #6328, where AutoGen's built-in `reflect_on_tool_use=True` reflection path fails against current OpenAI-compatible servers.

### How It Works

1. **Tool Execution**: Agent calls a tool; the system message instructs sequential, one-at-a-time execution.
2. **Result Return**: Tool returns `ToolCallSummaryMessage` to the agent.
3. **RoundRobinGroupChat**: Continues because no "?" termination marker is detected.
4. **Natural Interpretation**: Agent makes next inference with tool results in conversation history.
5. **Guided Analysis**: The system message instructs the agent to interpret results scientifically.
6. **Termination**: Agent ends its final response with "?" to signal task completion.

### Benefits

- ✅ Avoids AutoGen framework bugs (no buggy reflection code path)
- ✅ More natural, conversational interpretation
- ✅ Full control over interpretation depth and style
- ✅ Customizable per tool type via system message
- ✅ Sequential tool execution maintained (no parallel calls)
- ✅ Future-proof against AutoGen version changes

### System Message Example

From `src/agents/factory.py:23-44`:

```python
"When you receive tool execution results, you MUST:
1. Carefully analyze the scientific data returned by the tool
2. Explain what the results mean in materials science context
3. Interpret key metrics (formation energy, structure fractions, density)
4. Highlight significant findings or anomalies
5. Connect results back to the user's research question
6. Never simply echo raw tool output - provide expert interpretation"
```

### Reference

See AutoGen GitHub issue #6328 for details on the upstream bug.

## Environment Variables

Which keys are needed depends on which model provider you select in the Chainlit settings. The app prompts for missing keys at runtime and writes accepted values back to `.env` (which is gitignored).

### Provider keys (one of these is required at minimum)

```bash
# Ollama Cloud (used by default; *-cloud models in OLLAMA_MODELS)
OLLAMA_API_KEY=...

# OpenRouter (used when an OpenRouter free-tier model is selected)
OPENROUTER_API_KEY=...
```

- `OLLAMA_API_KEY` is read at startup (`run_chat.py:201`) and a prompt fires if it's absent and no cached `ollama login` device key is present.
- `OPENROUTER_API_KEY` is requested on-demand the first time an OpenRouter model is selected (`run_chat.py:747`).
- `OPENAI_API_KEY` is **not** validated at startup anymore — the OpenAI provider is no longer used in the standard build.

### Optional

```bash
# Disable PyTorch compilation (CUDA 12.4 compatibility)
TORCH_COMPILE_DISABLE=1

# Allow duplicate OpenMP libraries (silences the KMP warning)
KMP_DUPLICATE_LIB_OK=TRUE
```

### Loading from .env

OptiMat Alloys automatically loads environment variables from `.env` via `python-dotenv` at startup. Keys entered through the in-app prompts are persisted via `update_env_variable()` so they survive restarts.

**Example `.env`** (Ollama Cloud + a NequIP/MACE workflow on a CUDA 12.4 box):
```bash
OLLAMA_API_KEY=...
OPENROUTER_API_KEY=...
TORCH_COMPILE_DISABLE=1
KMP_DUPLICATE_LIB_OK=TRUE
```

## See Also

- [Maintenance Guide](MAINTENANCE.md) - Troubleshooting configuration issues
