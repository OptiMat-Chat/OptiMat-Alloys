# NequIP Worker Architecture

This document describes the multi-environment calculator architecture implemented to support NequIP calculators alongside ORB and MACE.

## Background: The e3nn Version Conflict

OptiMat Alloys supports multiple neural network potentials:
- **ORB Models**: Universal potential trained on OMat24 dataset
- **MACE**: Message-passing neural network potential (requires `e3nn==0.4.4`)
- **NequIP**: Equivariant neural network potential (requires `e3nn>=0.5.6`)

**The Problem**: MACE and NequIP have incompatible e3nn version requirements that cannot coexist in the same Python environment.

**The Solution**: Run NequIP in a separate conda environment (`optimat-nequip`) with a subprocess worker that communicates via stdin/stdout IPC.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Main Process (optimat-alloys env, e3nn 0.4.4)                 │
│                                                             │
│  ORB calculations  → load_calculator() directly (in-process)│
│  MACE calculations → load_calculator() directly (in-process)│
│                                                             │
│  NequIP calculations → spawn worker subprocess:             │
│     ┌─────────────────────────────────────────────────┐    │
│     │ Worker Process (optimat-nequip env, e3nn 0.5.6) │    │
│     │ - Starts on first NequIP calculation            │    │
│     │ - Killed after 10 min idle                      │    │
│     │ - Auto-restarts when needed                     │    │
│     └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **On-demand startup**: Worker only starts when NequIP is first requested
- **Automatic idle timeout**: Worker killed after 10 minutes of inactivity (saves GPU memory)
- **Auto-restarts**: Worker automatically restarts if needed
- **Auto-environment creation**: `optimat-nequip` conda environment created on first use
- **Calculator immutability**: Structures remember which calculator created them

### IPC Protocol

Communication uses JSON messages over stdin/stdout:

```json
// Request: Load calculator
{"action": "load", "model": "nequip-oam-l", "device": "cuda"}

// Request: Calculate
{"action": "calculate", "atoms": {"symbols": [...], "positions": [...], "cell": [...], "pbc": [...]}}

// Response: Calculation result
{"energy": -123.45, "forces": [[...]], "stress": [...]}

// Request: Shutdown
{"action": "shutdown"}
```

## Key Files

| File | Purpose |
|------|---------|
| `src/core/calculator_service.py` | Worker management, IPC, auto-env creation |
| `src/core/nequip_worker.py` | Standalone worker process |
| `environment-nequip.yml` | Conda environment specification |

## Configuration

The NequIP worker can be configured via environment variables to support different system configurations.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEQUIP_ENV_PATH` | `~/miniconda3/envs/optimat-nequip/bin` | Path to the NequIP conda environment bin directory |
| `NEQUIP_CACHE_DIR` | `~/.cache/nequip` | Directory for compiled model cache |

### Custom Conda Installation

If conda is installed in a non-standard location, set `NEQUIP_ENV_PATH` to point to the correct bin directory:

```bash
# Example for /opt/conda installation
export NEQUIP_ENV_PATH=/opt/conda/envs/optimat-nequip/bin

# Example for anaconda3
export NEQUIP_ENV_PATH=~/anaconda3/envs/optimat-nequip/bin

# Example for miniforge3
export NEQUIP_ENV_PATH=~/miniforge3/envs/optimat-nequip/bin

# Example for Docker/system-level conda
export NEQUIP_ENV_PATH=/opt/miniconda3/envs/optimat-nequip/bin
```

### Custom Cache Location

To use a different location for compiled NequIP models:

```bash
# Example: Store models on a separate drive
export NEQUIP_CACHE_DIR=/data/nequip-models

# Example: Use a temporary directory
export NEQUIP_CACHE_DIR=/tmp/nequip-cache
```

### Configuration in .env File

Add these variables to your `.env` file for persistent configuration:

```bash
# .env
NEQUIP_ENV_PATH=/path/to/conda/envs/optimat-nequip/bin
NEQUIP_CACHE_DIR=/path/to/nequip/cache
```

## Problems Encountered & Solutions

During implementation, we encountered 10 significant issues. This table documents each problem, its symptoms, root cause, and solution.

| # | Problem | Symptom | Root Cause | Solution |
|---|---------|---------|------------|----------|
| 1 | IPC pollution | JSON parsing errors | C extensions write directly to stdout fd | `os.dup2()` fd-level redirect |
| 2 | No loading feedback | Silent waiting during model load | stderr not relayed to parent | Add stderr reader task |
| 3 | Output buffering | Print statements delayed/missing | Thread pool stdout buffering | `flush=True`, `PYTHONUNBUFFERED=1` |
| 4 | ASE optimizer output | Relaxation steps not visible | ASE doesn't flush stdout | Use `logfile='-'` |
| 5 | Worker hangs | Indefinite hang after READY | `conda run` doesn't forward stdin | Use direct Python path |
| 6 | Wrong nequip-compile | e3nn import errors | PATH inheritance from parent | Use full path to binary |
| 7 | FlushingLogFile crash | "expected os.PathLike" error | ASE validates logfile as path | Removed custom class |
| 8 | Manual env creation | RuntimeError on first use | Environment doesn't exist | Auto-create on first use |
| 9 | QHA bypasses service | "requires optimat-nequip" error | `qha_wrapper.py` calls `load_calculator()` directly | Pass calculator from caller |
| 10 | ASE caching broken | Analysis phase timeout after relaxation | `NequIPSyncCalculator` doesn't call `super().calculate()` | Add `Calculator.calculate()` call |

### Detailed Problem Analysis

#### Problem 1: IPC Pollution

**Symptom**: `json.JSONDecodeError` when parsing worker responses.

**Root Cause**: NequIP and its dependencies (PyTorch, e3nn) contain C extensions that write directly to file descriptor 1 (stdout), bypassing Python's `sys.stdout`. These messages pollute the IPC channel.

**Solution**: Redirect stdout at the file descriptor level using `os.dup2()`:
```python
# At the very top of nequip_worker.py
_original_stdout_fd = os.dup(1)  # Save original stdout fd
os.dup2(2, 1)  # Redirect fd 1 (stdout) to fd 2 (stderr)
_original_stdout = os.fdopen(_original_stdout_fd, 'w')  # Create file for IPC
sys.stdout = sys.stderr  # Also redirect Python-level stdout
```

This ensures ALL output (including C extensions) goes to stderr, while only explicit IPC messages go to the saved stdout fd.

#### Problem 2: No Loading Feedback

**Symptom**: Terminal shows nothing while NequIP model is being downloaded/compiled (5+ minutes).

**Root Cause**: Worker prints status to stderr, but parent process wasn't reading stderr.

**Solution**: Add background task in `calculator_service.py` to relay stderr:
```python
async def _read_worker_stderr(self):
    """Background task to read and display stderr from worker."""
    while self.worker_process and self.worker_process.returncode is None:
        line = await self.worker_process.stderr.readline()
        if line:
            print(f"[NequIP] {line.decode().strip()}", flush=True)
```

#### Problem 3: Output Buffering

**Symptom**: Progress messages appear all at once instead of incrementally.

**Root Cause**: When running via Chainlit's `cl.make_async()`, stdout is buffered because it's no longer connected to a TTY.

**Solution**:
- Add `flush=True` to all `print()` calls in calculator_service.py, elastic_properties.py, qha_wrapper.py
- Set `PYTHONUNBUFFERED=1` environment variable for worker
- Use `-u` flag when spawning Python subprocess

#### Problem 4: ASE Optimizer Output

**Symptom**: Relaxation steps (energy, fmax per iteration) not visible in terminal.

**Root Cause**: ASE's optimizer `logfile` parameter defaults to stdout but doesn't flush.

**Solution**: Explicitly set `logfile='-'` (ASE's stdout mode):
```python
opt = FIRE(filtered, logfile='-')
```

*Note*: We initially tried a custom `FlushingLogFile` class, but ASE's internal validation rejected it (Problem 7).

#### Problem 5: Worker Hangs (CRITICAL)

**Symptom**: Worker starts, sends READY, then hangs indefinitely on first calculation.

**Root Cause**: `conda run` does NOT forward stdin to subprocesses. The IPC channel was broken:
- Parent sends request via stdin pipe
- `conda run` intercepts stdin but doesn't forward it
- Worker never receives the request

**Solution**: Bypass `conda run` entirely and use the direct Python interpreter path:
```python
# Before (broken):
"conda", "run", "-n", "optimat-nequip", "python", str(worker_script)

# After (working) — current code, parameterized via NEQUIP_ENV_PATH:
nequip_python = Path(NEQUIP_ENV_PATH) / "python"
str(nequip_python), "-u", str(worker_script)
```

The env var defaults to `~/miniconda3/envs/optimat-nequip/bin`, matching the original hardcoded path. See **Configuration → Environment Variables** above for how to override it on non-standard conda installs.

#### Problem 6: Wrong nequip-compile

**Symptom**: `WeightsUnpickler error: Unsupported global: GLOBAL slice` when compiling model.

**Root Cause**: After bypassing `conda run`, the worker inherits the parent's PATH. When calling `subprocess.run(["nequip-compile", ...])`, it finds the wrong binary at `/envs/optimat-alloys/bin/nequip-compile` instead of `/envs/optimat-nequip/bin/nequip-compile`.

**Solution**: Use the full path to `nequip-compile` resolved through `NEQUIP_ENV_PATH`:
```python
nequip_bin = Path(NEQUIP_ENV_PATH)
cmd = [str(nequip_bin / "nequip-compile"), ...]
```

#### Problem 7: FlushingLogFile Crash

**Symptom**: `expected str, bytes or os.PathLike object, not FlushingLogFile`

**Root Cause**: ASE's optimizer code passes the logfile to internal functions that validate it as a path-like object. Our custom `FlushingLogFile` class didn't implement `__fspath__()`.

**Solution**: Removed the custom class and reverted to `logfile='-'`. Output flushing is handled by `PYTHONUNBUFFERED=1` instead.

#### Problem 8: Manual Environment Creation

**Symptom**: `RuntimeError: NequIP environment not found` on first use.

**Root Cause**: The `optimat-nequip` conda environment didn't exist.

**Solution**: Auto-create environment on first use with live progress:
```python
async def _create_nequip_environment(self):
    print("NequIP environment not found. Creating it now...", flush=True)
    print("This is a one-time setup that may take 5-10 minutes.", flush=True)

    process = await asyncio.create_subprocess_exec(
        "conda", "env", "create", "-f", str(env_file),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    while True:
        line = await process.stdout.readline()
        if not line:
            break
        print(f"[conda] {line.decode().strip()}", flush=True)
```

#### Problem 9: QHA Wrapper Bypasses Calculator Service

**Symptom**: Error when using NequIP for anharmonic properties:
```
NequIP calculator 'nequip-oam-xl' requires the optimat-nequip environment.
Use 'from src.core.calculator_service import get_calculator_service...' instead.
```

**Root Cause**: The `qha_wrapper.py` module called `load_calculator()` directly to get a CPU calculator for relaxation, bypassing the `calculator_service` that handles NequIP's subprocess worker:
```python
# In qha_wrapper.py (broken)
cpu_calc = load_calculator(model=model_name, device='cpu')
```

**Solution**: Pass both GPU and CPU calculators from the caller (`anharmonic_properties.py`) instead of loading inside `qha_wrapper.py`:
```python
# In anharmonic_properties.py
service = get_calculator_service()
gpu_calc = await service.get_calculator_sync(model=model_name, device='cuda')
cpu_calc = await service.get_calculator_sync(model=model_name, device='cpu')

qha_data = await cl.make_async(compute_qha_properties)(
    atoms=atoms,
    calculator=gpu_calc,
    cpu_calculator=cpu_calc,  # NEW parameter
    ...
)

# In qha_wrapper.py
def compute_qha_properties(
    atoms: Atoms,
    calculator: Calculator,
    cpu_calculator: Calculator,  # NEW parameter
    ...
):
    optimizer = StructureOptimizer(cpu_calculator)  # Use passed-in calculator
```

#### Problem 10: ASE Caching Broken (Analysis Phase Timeout)

**Symptom**: After both GPU and CPU relaxation complete successfully, the analysis phase ("analyzing and saving results") times out. The calculation appears stuck despite relaxation finishing.

**Root Cause**: `NequIPSyncCalculator.calculate()` didn't call `super().calculate()`, which is required for ASE's result caching mechanism.

ASE's `Calculator.calculate()` does critical housekeeping:
```python
def calculate(self, atoms=None, properties=None, system_changes=all_changes):
    if atoms is not None:
        self.atoms = atoms.copy()  # Critical for caching!
```

Without this, `self.atoms` is never set. When tools call `get_potential_energy()`, `get_forces()`, and `get_stress()` during analysis, ASE's `get_property()` thinks the atoms have changed and triggers a recalculation for each property:

```python
# In recompute_structure.py analysis phase:
energy_per_atom = atoms.get_potential_energy()  # IPC call #1 (30+ sec on CPU)
forces = atoms.get_forces()                      # IPC call #2 (30+ sec on CPU)
stress = atoms.get_stress()                      # IPC call #3 (30+ sec on CPU)
# Total: 90+ seconds of unnecessary recalculation!
```

**Solution**: Add `Calculator.calculate()` call at the start of `NequIPSyncCalculator.calculate()`:
```python
def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes):
    # Call parent to set self.atoms for ASE caching (Problem 10 fix)
    Calculator.calculate(self, atoms, properties, system_changes)

    if atoms is None:
        atoms = self.atoms

    # Schedule async calculation on the event loop
    future = asyncio.run_coroutine_threadsafe(...)
```

This enables ASE's caching: the first property request triggers a calculation, and subsequent calls use cached results.

## Usage

### For Tools (Synchronous ASE Interface)

```python
from src.core.calculator_service import get_calculator_service

service = get_calculator_service()
calc = await service.get_calculator_sync("nequip-oam-l", "cuda")

# Works like any ASE calculator
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### For Async Contexts (More Efficient)

```python
service = get_calculator_service()
proxy = await service.get_calculator("nequip-oam-l", "cuda")

if hasattr(proxy, 'is_nequip_proxy'):
    # NequIP: Use async calculate
    result = await proxy.calculate_async(atoms)
    energy = result['energy']
    forces = np.array(result['forces'])
else:
    # ORB/MACE: Use standard Calculator
    atoms.calc = proxy
    energy = atoms.get_potential_energy()
```

## Lessons Learned

1. **`conda run` has a stdin forwarding bug** - Always use direct Python interpreter paths for subprocess IPC

2. **File descriptor redirect catches ALL output** - Python-level `sys.stdout` redirect only catches `print()`, not C extension output. Use `os.dup2()` for complete capture.

3. **ASE optimizers have internal path validation** - Can't use custom file-like objects for logfile parameter. Use the built-in `logfile='-'` option.

4. **Always use `flush=True` in thread pools** - When code runs via `cl.make_async()` or similar thread pool wrappers, stdout is no longer line-buffered. Explicit flushing is required for real-time output.

5. **PATH inheritance in subprocesses** - When bypassing conda's environment activation, subprocesses inherit parent's PATH. Use absolute paths for environment-specific binaries.

6. **Never call `load_calculator()` directly for NequIP** - All calculator loading must go through `calculator_service.get_calculator_sync()` to ensure NequIP uses the subprocess worker. Pass calculators as parameters to functions instead of loading them internally.

7. **ASE Calculator subclasses must call `super().calculate()`** - ASE's result caching depends on `self.atoms` being set by the parent's `calculate()` method. Without this, every call to `get_potential_energy()`, `get_forces()`, etc. triggers a new calculation instead of using cached results.

## Supported NequIP Models

| Model Name | NequIP.net Model | Description |
|------------|------------------|-------------|
| `nequip-oam-l` | `mir-group/NequIP-OAM-L:0.1` | Large OAM model |
| `nequip-oam-xl` | `mir-group/NequIP-OAM-XL:0.1` | Extra-large OAM model |
| `nequip-mp-l` | `mir-group/NequIP-MP-L:0.1` | Large Materials Project model |

Models are automatically downloaded and compiled on first use. Compiled models are cached at `~/.cache/nequip/`.

## Troubleshooting

### Worker Fails to Start

Check that `environment-nequip.yml` exists in the project root. The environment will be created automatically on first use.

### Model Compilation Fails

Ensure CUDA is available and the correct version is installed. Check `~/.cache/nequip/` for partially downloaded models that may need to be deleted.

### IPC Errors

If you see JSON parsing errors, check that no other code is writing to stdout in the worker process. All output should go through stderr or the dedicated IPC channel.

### Memory Issues

NequIP models can be memory-intensive. The worker is automatically killed after 10 minutes of idle time to free GPU memory. You can also manually shutdown using:
```python
service = get_calculator_service()
await service.shutdown()
```
