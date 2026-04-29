"""
Calculator Service for Multi-Environment Architecture.

This module manages calculator loading across different conda environments:
- ORB and MACE calculators run in-process (main optimat-alloys env)
- NequIP calculators run in a worker subprocess (optimat-nequip env)

The NequIP worker is started on-demand and killed after idle timeout.

For tools that need synchronous calculators:
    service = get_calculator_service()
    calc = await service.get_calculator_sync("nequip-oam-l", "cuda")
    atoms.calc = calc  # Works like a normal ASE calculator
    energy = atoms.get_potential_energy()

For async contexts (more efficient):
    proxy = await service.get_calculator("nequip-oam-l", "cuda")
    if hasattr(proxy, 'is_nequip_proxy'):
        result = await proxy.calculate_async(atoms)
        energy = result['energy']
"""

import asyncio
import concurrent.futures
import json
import os
import pickle
import base64
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import numpy as np


# Idle timeout for worker processes (seconds)
IDLE_TIMEOUT = 600  # 10 minutes

# Check interval for idle workers (seconds)
CHECK_INTERVAL = 60  # 1 minute

# Configurable NequIP environment path (for portability across different systems)
# Default value matches the original hardcoded path for backwards compatibility
NEQUIP_ENV_PATH = os.environ.get(
    "NEQUIP_ENV_PATH",
    str(Path.home() / "miniconda3" / "envs" / "optimat-nequip" / "bin")
)


class NequIPCalculatorProxy:
    """
    Proxy for NequIP calculator that runs in a separate process.

    Implements the same interface as ASE Calculator but forwards
    all calculations to the worker subprocess.

    For use with Chainlit's async tools:
        proxy = await service.get_calculator("nequip-oam-l", "cuda")
        result = await proxy.calculate_async(atoms)
        energy = result['energy']
        forces = np.array(result['forces'])
    """

    def __init__(self, service: 'CalculatorService', model: str, device: str):
        self.service = service
        self.model = model
        self.device = device
        self._loaded = False

    async def ensure_loaded(self):
        """Ensure the calculator is loaded in the worker."""
        if not self._loaded:
            await self.service._send_to_worker({
                "action": "load",
                "model": self.model,
                "device": self.device
            })
            self._loaded = True

    async def calculate_async(self, atoms: Atoms) -> Dict[str, Any]:
        """
        Calculate energy, forces, and stress for atoms.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dict with 'energy', 'forces', 'stress' keys
        """
        await self.ensure_loaded()

        # Serialize atoms for IPC
        atoms_dict = atoms_to_dict(atoms)

        response = await self.service._send_to_worker({
            "action": "calculate",
            "atoms": atoms_dict
        })

        return response

    @property
    def is_nequip_proxy(self) -> bool:
        """Marker property to identify this as a NequIP proxy."""
        return True


class NequIPSyncCalculator(Calculator):
    """
    Synchronous ASE Calculator wrapper for NequIP proxy.

    This class bridges the async NequIPCalculatorProxy to the synchronous
    ASE Calculator interface, allowing NequIP to be used with existing tools
    that expect standard ASE calculators.

    Usage:
        service = get_calculator_service()
        calc = await service.get_calculator_sync("nequip-oam-l", "cuda")
        atoms.calc = calc
        energy = atoms.get_potential_energy()  # Works synchronously

    Note:
        This wrapper runs async code in the main event loop using
        asyncio.run_coroutine_threadsafe(). It assumes the calling code
        is running in a thread pool (e.g., via cl.make_async).
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, proxy: NequIPCalculatorProxy, loop: asyncio.AbstractEventLoop):
        """
        Initialize the sync wrapper.

        Args:
            proxy: The NequIPCalculatorProxy to wrap
            loop: The asyncio event loop to use for async calls
        """
        super().__init__()
        self.proxy = proxy
        self.loop = loop
        self._model = proxy.model
        self._device = proxy.device

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: list = ['energy', 'forces', 'stress'],
        system_changes: list = all_changes
    ):
        """
        Calculate properties for atoms using the NequIP worker.

        This method is called by ASE when atoms.get_potential_energy() etc.
        are invoked. It runs the async calculation synchronously by
        scheduling it on the event loop.
        """
        # Call parent to set self.atoms for ASE caching (Problem 10 fix)
        # This enables ASE's result caching mechanism so subsequent calls
        # to get_potential_energy/get_forces/get_stress use cached results
        Calculator.calculate(self, atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        # Schedule async calculation on the event loop
        future = asyncio.run_coroutine_threadsafe(
            self.proxy.calculate_async(atoms),
            self.loop
        )

        # Wait for result (blocking)
        try:
            result = future.result(timeout=600)  # 10 minute timeout
        except concurrent.futures.TimeoutError:
            raise RuntimeError("NequIP calculation timed out after 600 seconds")
        except Exception as e:
            raise RuntimeError(f"NequIP calculation failed: {e}")

        # Store results in ASE format
        self.results = {
            'energy': result['energy'],
            'forces': np.array(result['forces']),
            'stress': np.array(result['stress']),
        }

    @property
    def is_nequip_sync(self) -> bool:
        """Marker property to identify this as a NequIP sync wrapper."""
        return True


class CalculatorService:
    """
    Manages calculator loading across different conda environments.

    - ORB and MACE: Loaded in-process (main environment)
    - NequIP: Loaded in worker subprocess (separate environment)

    Usage:
        service = get_calculator_service()

        # For ORB/MACE (returns Calculator directly)
        calc = await service.get_calculator("mace-mpa-0-medium", "cuda")
        atoms.calc = calc

        # For NequIP (returns proxy, use async methods)
        proxy = await service.get_calculator("nequip-oam-l", "cuda")
        result = await proxy.calculate_async(atoms)
    """

    def __init__(self):
        self.worker_process: Optional[asyncio.subprocess.Process] = None
        self.last_used: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._response_queue: asyncio.Queue = asyncio.Queue()

    async def get_calculator(
        self,
        model: str,
        device: str = "cuda"
    ) -> Union[Calculator, NequIPCalculatorProxy]:
        """
        Get calculator, spawning worker if needed for NequIP.

        Args:
            model: Calculator model name (e.g., 'orb-v3-direct-20-omat', 'nequip-oam-l')
            device: Device to use ('cuda' or 'cpu')

        Returns:
            Calculator for ORB/MACE, or NequIPCalculatorProxy for NequIP
        """
        if model.startswith("nequip-"):
            return await self._get_nequip_calculator(model, device)
        else:
            # ORB and MACE run in-process
            from .calculators import load_calculator
            return load_calculator(model, device)

    async def get_calculator_sync(
        self,
        model: str,
        device: str = "cuda"
    ) -> Calculator:
        """
        Get a synchronous ASE Calculator for any model type.

        This method returns a Calculator that can be used with standard
        ASE patterns (atoms.calc = calc; atoms.get_potential_energy()).

        For NequIP models, this wraps the async proxy in a sync wrapper
        that bridges async calls to the sync ASE interface.

        Args:
            model: Calculator model name (e.g., 'orb-v3-direct-20-omat', 'nequip-oam-l')
            device: Device to use ('cuda' or 'cpu')

        Returns:
            ASE-compatible Calculator for any model type
        """
        if model.startswith("nequip-"):
            proxy = await self._get_nequip_calculator(model, device)
            loop = asyncio.get_event_loop()
            return NequIPSyncCalculator(proxy, loop)
        else:
            # ORB and MACE run in-process
            from .calculators import load_calculator
            return load_calculator(model, device)

    async def _get_nequip_calculator(
        self,
        model: str,
        device: str
    ) -> NequIPCalculatorProxy:
        """Spawn NequIP worker if not running, return proxy."""
        async with self._lock:
            if self.worker_process is None or self.worker_process.returncode is not None:
                await self._start_worker()
            self.last_used = datetime.now()

        return NequIPCalculatorProxy(self, model, device)

    async def _start_worker(self):
        """Start NequIP worker in separate conda environment."""
        print("Starting NequIP worker (first use, this may take a moment)...", flush=True)

        worker_script = Path(__file__).parent / "nequip_worker.py"

        # Get the Python interpreter from the nequip environment
        # Note: We use direct path instead of 'conda run' because conda run
        # does NOT forward stdin properly, breaking IPC communication
        # Path is configurable via NEQUIP_ENV_PATH for portability
        nequip_python = Path(NEQUIP_ENV_PATH) / "python"

        if not nequip_python.exists():
            await self._create_nequip_environment()

        # Create environment with PYTHONUNBUFFERED for immediate output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self.worker_process = await asyncio.create_subprocess_exec(
            str(nequip_python), "-u", str(worker_script),  # -u for unbuffered stdout/stderr
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Wait for worker to signal ready
        ready_line = await self.worker_process.stdout.readline()
        if b"READY" not in ready_line:
            stderr = await self.worker_process.stderr.read()
            raise RuntimeError(f"NequIP worker failed to start: {stderr.decode()}")

        print("✓ NequIP worker started", flush=True)

        # Start background tasks to read stdout (JSON responses) and stderr (status messages)
        asyncio.create_task(self._read_worker_responses())
        asyncio.create_task(self._read_worker_stderr())

    async def _create_nequip_environment(self):
        """Create the optimat-nequip conda environment on first use."""
        print("=" * 60, flush=True)
        print("NequIP environment not found. Creating it now...", flush=True)
        print("This is a one-time setup that may take 5-10 minutes.", flush=True)
        print("=" * 60, flush=True)

        # Find environment-nequip.yml (relative to project root)
        env_file = Path(__file__).parent.parent.parent / "environment-nequip.yml"
        if not env_file.exists():
            raise RuntimeError(
                f"Cannot find environment-nequip.yml at {env_file}. "
                "Please ensure the file exists in the project root."
            )

        print(f"Using environment file: {env_file}", flush=True)

        # Run conda env create with live output
        process = await asyncio.create_subprocess_exec(
            "conda", "env", "create", "-f", str(env_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
        )

        # Stream output to console
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            print(f"[conda] {line.decode().strip()}", flush=True)

        await process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                "Failed to create NequIP environment. "
                "Please check the output above for errors."
            )

        print("=" * 60, flush=True)
        print("✓ NequIP environment created successfully!", flush=True)
        print("=" * 60, flush=True)

    async def _read_worker_responses(self):
        """Background task to read responses from worker."""
        while self.worker_process and self.worker_process.returncode is None:
            try:
                line = await self.worker_process.stdout.readline()
                if not line:
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue  # Skip empty lines

                # Skip non-JSON lines (conda output, library messages)
                if not line_str.startswith('{'):
                    print(f"[NequIP worker] {line_str}", file=sys.stderr, flush=True)
                    continue

                response = json.loads(line_str)
                await self._response_queue.put(response)
            except json.JSONDecodeError as e:
                # Log but don't break - could be stray library output
                print(f"Warning: Non-JSON from worker: {line_str[:100]}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Error reading worker response: {e}")
                break

    async def _read_worker_stderr(self):
        """Background task to read and display stderr from worker (loading status)."""
        while self.worker_process and self.worker_process.returncode is None:
            try:
                line = await self.worker_process.stderr.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if line_str:
                    # Relay important status messages to console
                    print(f"[NequIP] {line_str}", flush=True)
            except Exception:
                break

    async def _send_to_worker(self, request: Dict) -> Dict:
        """Send request to worker and wait for response."""
        if self.worker_process is None or self.worker_process.returncode is not None:
            raise RuntimeError("NequIP worker not running")

        # Send request
        request_line = json.dumps(request) + "\n"
        self.worker_process.stdin.write(request_line.encode())
        await self.worker_process.stdin.drain()

        # Update last used time
        self.last_used = datetime.now()

        # Wait for response
        response = await self._response_queue.get()

        if "error" in response:
            raise RuntimeError(f"NequIP worker error: {response['error']}")

        return response

    async def check_idle(self) -> bool:
        """
        Check if worker is idle and kill if timeout exceeded.

        Returns:
            True if worker was killed, False otherwise
        """
        if self.worker_process is None or self.worker_process.returncode is not None:
            return False

        if self.last_used is None:
            return False

        elapsed = (datetime.now() - self.last_used).total_seconds()
        if elapsed > IDLE_TIMEOUT:
            await self._kill_worker()
            return True

        return False

    async def _kill_worker(self):
        """Kill the NequIP worker process."""
        if self.worker_process is None:
            return

        print(f"Killing idle NequIP worker (idle for {IDLE_TIMEOUT}s)")

        try:
            # Send shutdown command
            shutdown_request = json.dumps({"action": "shutdown"}) + "\n"
            self.worker_process.stdin.write(shutdown_request.encode())
            await self.worker_process.stdin.drain()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(self.worker_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill if not responding
                self.worker_process.kill()
                await self.worker_process.wait()
        except Exception as e:
            print(f"Error killing worker: {e}")
            if self.worker_process:
                self.worker_process.kill()

        self.worker_process = None
        self.last_used = None
        print("✓ NequIP worker stopped")

    async def shutdown(self):
        """Shutdown the service and kill any workers."""
        await self._kill_worker()


# Singleton instance
_calculator_service: Optional[CalculatorService] = None


def get_calculator_service() -> CalculatorService:
    """Get the singleton CalculatorService instance."""
    global _calculator_service
    if _calculator_service is None:
        _calculator_service = CalculatorService()
    return _calculator_service


async def check_idle_workers():
    """Check and kill idle workers. Call this periodically."""
    service = get_calculator_service()
    await service.check_idle()


# Serialization helpers for IPC

def atoms_to_dict(atoms: Atoms) -> Dict:
    """Serialize ASE Atoms to dictionary for JSON transport."""
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.get_cell().tolist(),
        "pbc": atoms.get_pbc().tolist(),
    }


def dict_to_atoms(d: Dict) -> Atoms:
    """Deserialize dictionary to ASE Atoms."""
    return Atoms(
        symbols=d["symbols"],
        positions=np.array(d["positions"]),
        cell=np.array(d["cell"]),
        pbc=d["pbc"],
    )
