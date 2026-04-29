#!/usr/bin/env python
"""
NequIP Calculator Worker Process.

This script runs in the optimat-nequip conda environment and handles
NequIP calculator operations. It communicates with the main process
via stdin/stdout using JSON messages.

Protocol:
- Receives JSON requests on stdin (one per line)
- Sends JSON responses on stdout (one per line)
- Prints "READY" on stdout when initialized

Request actions:
- {"action": "load", "model": "nequip-oam-l", "device": "cuda"}
- {"action": "calculate", "atoms": {...}}
- {"action": "shutdown"}

This worker is started on-demand by CalculatorService when a NequIP
calculator is first requested, and killed after idle timeout.

Usage (internal, called by calculator_service.py):
    conda run -n optimat-nequip python src/core/nequip_worker.py
"""

import sys
import os

# CRITICAL: Redirect stdout to stderr at FILE DESCRIPTOR level.
# This catches ALL output including C extensions, conda run messages,
# and library output (PyTorch, NequIP, e3nn).
# Python-level sys.stdout redirect only catches print() calls, not C-level writes.
_original_stdout_fd = os.dup(1)  # Save original stdout file descriptor
os.dup2(2, 1)  # Redirect fd 1 (stdout) to fd 2 (stderr)
_original_stdout = os.fdopen(_original_stdout_fd, 'w')  # Create file object for IPC

# Also redirect Python-level stdout (for print() calls in Python code)
sys.stdout = sys.stderr

import json

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from ase import Atoms


# Suppress warnings during import
import warnings
warnings.filterwarnings("ignore")


# Configurable paths via environment variables (for portability across different systems)
# Default values match the original hardcoded paths for backwards compatibility
NEQUIP_ENV_PATH = os.environ.get(
    "NEQUIP_ENV_PATH",
    str(Path.home() / "miniconda3" / "envs" / "optimat-nequip" / "bin")
)
NEQUIP_CACHE_DIR = os.environ.get(
    "NEQUIP_CACHE_DIR",
    str(Path.home() / ".cache" / "nequip")
)


def dict_to_atoms(d: Dict) -> Atoms:
    """Deserialize dictionary to ASE Atoms."""
    return Atoms(
        symbols=d["symbols"],
        positions=np.array(d["positions"]),
        cell=np.array(d["cell"]),
        pbc=d["pbc"],
    )


def send_response(response: Dict):
    """Send JSON response to the original stdout (IPC channel)."""
    _original_stdout.write(json.dumps(response) + '\n')
    _original_stdout.flush()


def send_error(message: str):
    """Send error response."""
    send_response({"error": message})


class NequIPWorker:
    """Worker that handles NequIP calculator operations."""

    # Model mapping: our name -> nequip.net name
    MODEL_MAP = {
        "nequip-oam-l": "mir-group/NequIP-OAM-L:0.1",
        "nequip-oam-xl": "mir-group/NequIP-OAM-XL:0.1",
        "nequip-mp-l": "mir-group/NequIP-MP-L:0.1",
    }

    def __init__(self):
        self.calculator = None
        self.current_model: Optional[str] = None
        self.current_device: Optional[str] = None

    def load_calculator(self, model: str, device: str):
        """Load NequIP calculator."""
        if model == self.current_model and device == self.current_device:
            return  # Already loaded

        from nequip.ase import NequIPCalculator

        nequip_model = self.MODEL_MAP.get(model)
        if not nequip_model:
            raise ValueError(f"Unknown NequIP model: {model}. Choose from: {list(self.MODEL_MAP.keys())}")

        # Cache directory for compiled models (configurable via NEQUIP_CACHE_DIR)
        cache_dir = Path(NEQUIP_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Compiled model path (device-specific)
        compiled_path = cache_dir / f"{model}_{device}.nequip.pth"

        # Compile if not cached
        if not compiled_path.exists():
            print(f"Downloading and compiling {model} for {device}...", file=sys.stderr)
            print("This only needs to be done once per model/device combination.", file=sys.stderr)

            # Use full path to nequip-compile from the nequip environment
            # This is necessary because we bypass conda run (which doesn't forward stdin),
            # so the worker inherits the parent's PATH which may have wrong environment first
            # Path is configurable via NEQUIP_ENV_PATH for portability
            nequip_bin = Path(NEQUIP_ENV_PATH)
            cmd = [
                str(nequip_bin / "nequip-compile"),
                f"nequip.net:{nequip_model}",
                str(compiled_path),
                "--mode", "torchscript",
                "--device", device,
                "--target", "ase"
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"Model compiled and cached at {compiled_path}", file=sys.stderr)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to compile NequIP model {model}: {e.stderr}")

        # Load compiled model
        print(f"Loading NequIP ({model}) on {device.upper()}", file=sys.stderr)
        self.calculator = NequIPCalculator.from_compiled_model(
            str(compiled_path),
            device=device
        )
        self.current_model = model
        self.current_device = device
        print(f"Successfully loaded NequIP ({model})", file=sys.stderr)

    def calculate(self, atoms_dict: Dict) -> Dict[str, Any]:
        """Calculate energy, forces, and stress."""
        if self.calculator is None:
            raise RuntimeError("Calculator not loaded. Call 'load' first.")

        atoms = dict_to_atoms(atoms_dict)
        atoms.calc = self.calculator

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces().tolist()

        # Stress may not always be available
        try:
            stress = atoms.get_stress().tolist()
        except Exception:
            stress = [0.0] * 6

        return {
            "energy": energy,
            "forces": forces,
            "stress": stress,
        }


def main():
    """Main worker loop."""
    worker = NequIPWorker()

    # Signal ready - write to original stdout for IPC
    _original_stdout.write("READY\n")
    _original_stdout.flush()

    while True:
        try:
            # Read request from stdin
            line = sys.stdin.readline()
            if not line:
                break  # EOF, parent process closed pipe

            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
            action = request.get("action")

            if action == "load":
                model = request["model"]
                device = request.get("device", "cuda")
                worker.load_calculator(model, device)
                send_response({"status": "loaded", "model": model, "device": device})

            elif action == "calculate":
                atoms_dict = request["atoms"]
                result = worker.calculate(atoms_dict)
                send_response(result)

            elif action == "shutdown":
                send_response({"status": "shutdown"})
                break

            else:
                send_error(f"Unknown action: {action}")

        except json.JSONDecodeError as e:
            send_error(f"Invalid JSON: {e}")
        except KeyError as e:
            send_error(f"Missing required field: {e}")
        except Exception as e:
            send_error(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
