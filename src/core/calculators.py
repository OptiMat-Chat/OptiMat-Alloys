"""
Calculator management for atomistic simulations.

This module provides a clean interface for loading and managing
ASE-compatible universal potential calculators with automatic CUDA/CPU fallback.

Supported calculator families (in main environment):
- ORB (Orbital Materials): orb-v3-direct-20-omat, orb-v3-conservative-inf-omat
- MACE-MPA (Matbench SOTA): mace-mpa-0-medium
- MACE-OMAT (best for phonons): mace-omat-0-small/medium

For NequIP calculators (nequip-oam-l, nequip-oam-xl, nequip-mp-l):
Use calculator_service.get_calculator() which runs NequIP in a separate
conda environment (optimat-nequip) due to e3nn version conflicts.
"""

from typing import Literal, Optional, Union
from ase.calculators.calculator import Calculator
import os


# Supported model types (main environment)
ORBModel = Literal['orb-v3-direct-20-omat', 'orb-v3-conservative-inf-omat']
MACEMPAModel = Literal['mace-mpa-0-medium']  # Matbench SOTA, only medium exists
MACEOMATModel = Literal['mace-omat-0-small', 'mace-omat-0-medium']  # Best for phonons
SupportedModel = Union[ORBModel, MACEMPAModel, MACEOMATModel]

# NequIP models (handled by calculator_service in separate environment)
NequIPModel = Literal['nequip-oam-l', 'nequip-oam-xl', 'nequip-mp-l']


def _configure_cpu_threads():
    """Auto-configure CPU threads for optimal parallelization.

    Must be called BEFORE importing torch, as PyTorch reads
    OMP_NUM_THREADS at import time.
    """
    omp_threads = os.environ.get('OMP_NUM_THREADS')
    num_cores = os.cpu_count() or 1

    if omp_threads is None:
        # Not set - use all available cores
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
    elif omp_threads == '1':
        # Explicitly set to 1 - likely unintentional for compute workloads
        # Auto-upgrade to all cores with a warning
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
        import warnings
        warnings.warn(
            f"OMP_NUM_THREADS=1 detected. Auto-upgraded to {num_cores} cores "
            f"for better CPU performance. Set OMP_NUM_THREADS>1 to override.",
            UserWarning
        )


# Configure threads before torch import (torch is imported lazily in load())
_configure_cpu_threads()


def _load_mace_calculator(model: str, device: str) -> Calculator:
    """Load a MACE universal potential calculator.

    Args:
        model: MACE model identifier. Supported families:
            - mace-mpa-0-*: MACE-MPA-0 (Matbench SOTA, recommended)
            - mace-omat-0-*: MACE-OMAT-0 (best for phonons)
        device: Target device ('cuda' or 'cpu')

    Returns:
        MACE calculator instance

    Notes:
        All MACE models support 89 elements and are loaded via mace_mp().
        - MACE-MPA-0: Matbench SOTA, default since MACE v0.3.10 (recommended)
        - MACE-OMAT-0: Trained on OMAT dataset, best accuracy for phonons
    """
    from mace.calculators import mace_mp

    # Map our model names to mace_mp model parameter
    model_mapping = {
        # MACE-MPA-0 (Matbench SOTA, only medium exists)
        'mace-mpa-0-medium': 'medium-mpa-0',
        # MACE-OMAT-0 (best for phonons, small and medium)
        'mace-omat-0-small': 'small-omat-0',
        'mace-omat-0-medium': 'medium-omat-0',
    }

    mace_model = model_mapping.get(model)
    if not mace_model:
        raise ValueError(f"Unknown MACE model: {model}")

    print(f"Loading MACE ({mace_model}) on {device.upper()}")
    calc = mace_mp(model=mace_model, device=device, default_dtype="float32")
    print(f"✓ Successfully loaded MACE ({mace_model})")
    return calc


# NOTE: NequIP loading has been moved to src/core/nequip_worker.py
# NequIP requires a separate conda environment (optimat-nequip) due to
# e3nn version conflicts. Use calculator_service.get_calculator() instead.


class CalculatorManager:
    """
    Manages ASE calculator lifecycle with caching and auto-fallback.

    Examples:
        >>> manager = CalculatorManager()
        >>> calc = manager.load('orb-v3-direct-20-omat', device='cuda')
        >>> atoms.calc = calc
    """

    def __init__(self):
        self._cache: dict[str, Calculator] = {}

    def load(
        self,
        model: SupportedModel = 'orb-v3-direct-20-omat',
        device: Literal['cpu', 'cuda'] = 'cuda',
        precision: Optional[str] = None,
        use_cache: bool = True
    ) -> Calculator:
        """
        Load a universal potential calculator with automatic CUDA/CPU fallback.

        Args:
            model: Model identifier. Supported models:
                ORB (Orbital Materials):
                - 'orb-v3-direct-20-omat': Fast model (2-3x faster, forces via direct method)
                - 'orb-v3-conservative-inf-omat': Most accurate (forces via backprop)

                MACE-MPA (89 elements, Matbench SOTA):
                - 'mace-mpa-0-medium': Recommended for general materials

                MACE-OMAT (89 elements, best for phonons):
                - 'mace-omat-0-small/medium': Best for phonon calculations

                NequIP (Foundation models from nequip.net):
                - 'nequip-oam-l': F1=0.893, trained on OMat24+MPtrj+sAlex
                - 'nequip-oam-xl': F1=0.906, highest accuracy
                - 'nequip-mp-l': F1=0.761, trained on MPtrj only

            device: Target device ('cuda' or 'cpu'). Automatically falls back to CPU if CUDA fails.
            precision: Precision mode (ORB only). If None, uses model defaults:
                - 'float32-high' for direct model
                - 'float32-highest' for conservative model
            use_cache: Whether to cache and reuse calculator instances

        Returns:
            Calculator instance ready for use with ASE

        Notes:
            - First call will be slower due to model loading/compilation
            - Set TORCH_COMPILE_DISABLE=1 environment variable to disable compilation
            - CUDA automatically falls back to CPU if unavailable or if errors occur
            - MACE and NequIP models are downloaded automatically on first use
        """
        # Normalize model name
        model = model.lower().replace("_", "-")

        # Determine model family
        is_orb = model.startswith("orb-")
        is_mace = model.startswith("mace-")
        is_nequip = model.startswith("nequip-")

        # Set default precision for ORB models
        if precision is None and is_orb:
            precision = "float32-high" if "direct" in model else "float32-highest"

        # Check cache
        cache_key = f"{model}_{device}_{precision if precision else 'default'}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # NequIP requires separate environment - route to calculator_service
        if is_nequip:
            raise RuntimeError(
                f"NequIP calculator '{model}' requires the optimat-nequip environment. "
                f"Use calculator_service.get_calculator() instead."
            )

        # Supported models list (ORB + MACE only, NequIP uses worker)
        supported_models = [
            # ORB
            "orb-v3-direct-20-omat",
            "orb-v3-conservative-inf-omat",
            # MACE-MPA-0 (Matbench SOTA)
            "mace-mpa-0-medium",
            # MACE-OMAT-0 (best for phonons)
            "mace-omat-0-small", "mace-omat-0-medium",
        ]

        if model not in supported_models:
            raise ValueError(
                f"Model '{model}' not supported. Choose from:\n"
                f"  ORB: orb-v3-direct-20-omat (fast), orb-v3-conservative-inf-omat (accurate)\n"
                f"  MACE-MPA: mace-mpa-0-medium (recommended, Matbench SOTA)\n"
                f"  MACE-OMAT: mace-omat-0-small/medium (best for phonons)\n"
                f"  NequIP: Use calculator_service for nequip-oam-l, nequip-oam-xl, nequip-mp-l"
            )

        # Determine effective device with CUDA fallback
        effective_device = device
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                effective_device = 'cpu'

        # Load calculator based on model family (NequIP handled by calculator_service)
        try:
            if is_mace:
                calc = _load_mace_calculator(model, effective_device)
            elif is_orb:
                calc = self._load_orb_calculator(model, effective_device, precision)
            else:
                raise ValueError(f"Unknown model family for '{model}'")
        except Exception as e:
            if effective_device == 'cuda':
                print(f"Failed to load on CUDA: {e}")
                print("Falling back to CPU")
                effective_device = 'cpu'
                if is_mace:
                    calc = _load_mace_calculator(model, effective_device)
                elif is_orb:
                    calc = self._load_orb_calculator(model, effective_device, precision)
            else:
                raise

        # Cache if requested
        if use_cache:
            self._cache[cache_key] = calc

        return calc

    def _load_orb_calculator(
        self,
        model: str,
        device: str,
        precision: str
    ) -> Calculator:
        """Load an ORB calculator with compilation fallback.

        Args:
            model: ORB model identifier
            device: Target device ('cuda' or 'cpu')
            precision: Precision mode ('float32-high' or 'float32-highest')

        Returns:
            ORBCalculator instance
        """
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # Model configuration
        model_loaders = {
            "orb-v3-direct-20-omat": pretrained.orb_v3_direct_20_omat,
            "orb-v3-conservative-inf-omat": pretrained.orb_v3_conservative_inf_omat,
        }

        loader = model_loaders[model]

        print(f"Loading {model} on {device.upper()} (precision={precision})")
        if device == 'cuda':
            print("Note: First call will be slower due to model compilation")

        try:
            # Try loading with compilation enabled (default for v3)
            calc = ORBCalculator(loader(device=device, precision=precision), device=device)
            print(f"✓ Successfully loaded {model} on {device.upper()}")
        except Exception as compile_error:
            if device == 'cuda':
                # If compilation fails on CUDA, try disabling it
                print(f"Warning: Model compilation failed: {compile_error}")
                print("Trying without compilation (slower but should work)...")

                # Disable torch compilation
                os.environ["TORCH_COMPILE_DISABLE"] = "1"
                calc = ORBCalculator(loader(device=device, precision=precision), device=device)
                print(f"✓ Successfully loaded {model} on CUDA (compilation disabled)")
            else:
                raise

        return calc

    def clear_cache(self):
        """Clear the calculator cache to free memory."""
        self._cache.clear()

    def get_cached_calculators(self) -> list[str]:
        """Return list of cached calculator keys."""
        return list(self._cache.keys())


# Convenience function for backward compatibility
def load_calculator(
    model: SupportedModel = "orb-v3-direct-20-omat",
    device: str = "cuda"
) -> Calculator:
    """
    Load a universal potential ASE calculator.

    This is a convenience function that creates a CalculatorManager
    and loads a calculator. For better performance with multiple
    calculator loads, use CalculatorManager directly.

    Args:
        model: Model identifier. Supported families:
            - ORB: 'orb-v3-direct-20-omat', 'orb-v3-conservative-inf-omat'
            - MACE-MPA: 'mace-mpa-0-medium' (recommended, Matbench SOTA)
            - MACE-OMAT: 'mace-omat-0-small/medium' (best for phonons)
            - NequIP: 'nequip-oam-l', 'nequip-oam-xl', 'nequip-mp-l'
        device: "cpu" or "cuda" (auto-fallback to CPU if CUDA fails)

    Returns:
        Calculator: An ASE-compatible calculator instance

    Notes:
        - First call will be slower due to model loading/compilation
        - ORB direct: 2-3x faster, good for most use cases
        - ORB conservative: Forces via backprop, best accuracy, required for NVE MD
        - MACE-MPA: Matbench SOTA, recommended for general materials
        - MACE-OMAT: Best accuracy for phonon calculations
        - NequIP: Requires optimat-nequip environment, use calculator_service instead
    """
    manager = CalculatorManager()
    return manager.load(model=model, device=device)
