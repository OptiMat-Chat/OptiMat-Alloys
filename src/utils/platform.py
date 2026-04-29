"""
Platform detection and system information utilities for OptiMat Alloys.

This module provides functions to detect the runtime platform (Linux/WSL2),
check CUDA availability, and retrieve system information useful for debugging
and performance optimization.
"""

import platform
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional


def detect_platform() -> str:
    """
    Detect the current platform.

    Returns:
        Platform identifier: "wsl2", "linux", or "unknown"

    Examples:
        >>> platform_name = detect_platform()
        >>> platform_name in ["wsl2", "linux", "unknown"]
        True
    """
    system = platform.system().lower()

    if system == "linux":
        # Check if running in WSL2
        if is_wsl2():
            return "wsl2"
        return "linux"

    return "unknown"


def is_wsl2() -> bool:
    """
    Check if running inside WSL2 (Windows Subsystem for Linux 2).

    Returns:
        True if running in WSL2, False otherwise

    Examples:
        >>> running_in_wsl = is_wsl2()
        >>> isinstance(running_in_wsl, bool)
        True
    """
    try:
        # Check /proc/version for WSL signature
        if Path("/proc/version").exists():
            with open("/proc/version", "r") as f:
                version_string = f.read().lower()
                # WSL2 contains "microsoft" in /proc/version
                return "microsoft" in version_string or "wsl" in version_string
    except (IOError, PermissionError):
        pass

    # Alternative: Check for WSL environment variable
    try:
        import os
        return "WSL_DISTRO_NAME" in os.environ
    except:
        pass

    return False


def check_cuda_availability() -> Tuple[bool, Optional[str]]:
    """
    Check if CUDA is available and return GPU information.

    Returns:
        Tuple of (cuda_available, gpu_name)
        - cuda_available: True if CUDA is available
        - gpu_name: Name of the GPU if available, None otherwise

    Examples:
        >>> cuda_available, gpu_name = check_cuda_availability()
        >>> isinstance(cuda_available, bool)
        True
    """
    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
        else:
            return False, None

    except ImportError:
        return False, None
    except Exception as e:
        # If torch is available but CUDA check fails
        return False, None


def get_platform_info() -> Dict[str, str]:
    """
    Get comprehensive platform information.

    Returns:
        Dictionary containing platform details:
        - platform: Platform identifier (wsl2/linux/unknown)
        - system: OS name
        - release: OS release version
        - version: OS version string
        - machine: Machine architecture
        - python_version: Python version
        - python_executable: Path to Python executable
        - cuda_available: "Yes" or "No"
        - gpu_name: GPU name if available

    Examples:
        >>> info = get_platform_info()
        >>> "platform" in info
        True
        >>> "python_version" in info
        True
    """
    cuda_available, gpu_name = check_cuda_availability()

    info = {
        "platform": detect_platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_executable": sys.executable,
        "cuda_available": "Yes" if cuda_available else "No",
        "gpu_name": gpu_name if gpu_name else "N/A",
    }

    return info


def print_platform_info() -> None:
    """
    Print platform information in a formatted way.

    Useful for debugging and user support.

    Examples:
        >>> print_platform_info()  # doctest: +SKIP
        Platform Information:
        ====================
        Platform: linux
        ...
    """
    info = get_platform_info()

    print("Platform Information:")
    print("=" * 50)
    for key, value in info.items():
        formatted_key = key.replace("_", " ").title()
        print(f"{formatted_key:20} {value}")
    print("=" * 50)


def get_cuda_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        Number of CUDA devices available (0 if none or CUDA not available)

    Examples:
        >>> count = get_cuda_device_count()
        >>> count >= 0
        True
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except ImportError:
        return 0
    except Exception:
        return 0


def get_cuda_memory_info(device: int = 0) -> Tuple[int, int]:
    """
    Get CUDA memory information for a specific device.

    Args:
        device: GPU device index (default: 0)

    Returns:
        Tuple of (total_memory_mb, allocated_memory_mb)
        Returns (0, 0) if CUDA is not available

    Examples:
        >>> total, allocated = get_cuda_memory_info()
        >>> total >= 0 and allocated >= 0
        True
    """
    try:
        import torch
        if torch.cuda.is_available() and device < torch.cuda.device_count():
            total = torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)
            allocated = torch.cuda.memory_allocated(device) // (1024 ** 2)
            return total, allocated
        return 0, 0
    except ImportError:
        return 0, 0
    except Exception:
        return 0, 0


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.

    Returns:
        Dictionary mapping package names to availability (True/False)

    Examples:
        >>> deps = check_dependencies()
        >>> "torch" in deps
        True
    """
    dependencies = [
        "torch",
        "ase",
        "orb_models",
        "ovito",
        "chainlit",
        "autogen_agentchat",
        "numpy",
        "scipy",
        "plotly",
    ]

    results = {}
    for dep in dependencies:
        try:
            __import__(dep)
            results[dep] = True
        except ImportError:
            results[dep] = False

    return results


if __name__ == "__main__":
    # Run diagnostics when module is executed directly
    print_platform_info()
    print()

    print("Dependency Check:")
    print("=" * 50)
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"{dep:20} {status}")
    print("=" * 50)
