"""
Structure optimization and relaxation.

This module provides tools for relaxing atomic structures using
ASE optimizers (FIRE, LBFGS) with optional cell relaxation.
"""

from typing import Literal, Optional
from pathlib import Path
from ase import Atoms
from ase.optimize import FIRE, LBFGS
from ase.filters import FrechetCellFilter
from ase.calculators.calculator import Calculator


class StructureOptimizer:
    """
    Optimizes atomic structures using ASE optimizers.

    Supports both atomic position relaxation and cell relaxation
    (hydrostatic or full strain tensor).

    Examples:
        >>> from src.core.calculators import CalculatorManager
        >>> calc_mgr = CalculatorManager()
        >>> calc = calc_mgr.load('orb-v3-direct-20-omat')
        >>> optimizer = StructureOptimizer(calc)
        >>> relaxed = optimizer.relax(atoms, fmax=0.01)
    """

    def __init__(self, calculator: Calculator):
        """
        Initialize optimizer with a calculator.

        Args:
            calculator: ASE calculator for energy/force evaluation
        """
        self.calculator = calculator

    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.001,
        max_steps: int = 1000,
        optimizer: Literal['FIRE', 'LBFGS'] = 'FIRE',
        hydrostatic_strain: bool = True,
        trajectory_path: Optional[Path] = None
    ) -> Atoms:
        """
        Relax atomic structure.

        Args:
            atoms: ASE Atoms object to relax
            fmax: Force convergence criterion (eV/Angstrom).
                  Optimization stops when max force < fmax.
            max_steps: Maximum number of optimization steps
            optimizer: Optimization algorithm
                - 'FIRE': Fast Inertial Relaxation Engine (robust, slower)
                - 'LBFGS': Limited-memory BFGS (fast, may have convergence issues)
            hydrostatic_strain: If True, allow cell volume to change while
                maintaining shape. If False, keep cell fixed.
            trajectory_path: Optional path to save optimization trajectory

        Returns:
            Relaxed Atoms object (modified in-place)

        Notes:
            - FIRE typically requires more steps but is more robust
            - LBFGS is faster per step but can fail to converge
            - Hydrostatic relaxation is recommended for bulk systems
            - Cell angles are kept fixed in hydrostatic mode
        """
        # Attach calculator
        atoms.calc = self.calculator

        # Setup cell filter
        if hydrostatic_strain:
            # Allow volume change, keep shape
            filtered = FrechetCellFilter(atoms, hydrostatic_strain=True)
        else:
            # Keep cell completely fixed
            filtered = FrechetCellFilter(atoms, mask=[False] * 6)

        # Create optimizer instance (logfile='-' writes to stdout)
        if optimizer == 'LBFGS':
            opt = LBFGS(filtered, logfile='-')
        else:
            opt = FIRE(filtered, logfile='-')

        # Attach trajectory writer if requested
        if trajectory_path:
            from ase.io.trajectory import Trajectory
            trajectory_path = Path(trajectory_path)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            traj = Trajectory(str(trajectory_path), 'w', atoms)
            opt.attach(traj.write, interval=1)

        # Run optimization
        opt.run(fmax=fmax, steps=max_steps)

        # Close trajectory file
        if trajectory_path:
            traj.close()

        return atoms

    def relax_full_cell(
        self,
        atoms: Atoms,
        fmax: float = 0.001,
        max_steps: int = 1000,
        optimizer: Literal['FIRE', 'LBFGS'] = 'FIRE',
        trajectory_path: Optional[Path] = None
    ) -> Atoms:
        """
        Relax structure with full cell degrees of freedom.

        Allows both cell shape and volume to change (all 6 strain components).

        Args:
            atoms: ASE Atoms object to relax
            fmax: Force convergence criterion (eV/Angstrom)
            max_steps: Maximum number of optimization steps
            optimizer: 'FIRE' or 'LBFGS'
            trajectory_path: Optional path to save trajectory

        Returns:
            Relaxed Atoms object

        Notes:
            Use this for systems where cell shape optimization is important
            (e.g., layered materials, orthorhombic crystals).
        """
        atoms.calc = self.calculator

        # Allow all cell degrees of freedom
        filtered = FrechetCellFilter(atoms)

        # Create optimizer instance (logfile='-' writes to stdout)
        if optimizer == 'LBFGS':
            opt = LBFGS(filtered, logfile='-')
        else:
            opt = FIRE(filtered, logfile='-')

        if trajectory_path:
            from ase.io.trajectory import Trajectory
            trajectory_path = Path(trajectory_path)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            traj = Trajectory(str(trajectory_path), 'w', atoms)
            opt.attach(traj.write, interval=1)

        opt.run(fmax=fmax, steps=max_steps)

        if trajectory_path:
            traj.close()

        return atoms


# Legacy function for backward compatibility
def relax_atoms(
    atoms: Atoms,
    hydrostatic_cell_relaxation: bool = True,
    optimizer: Literal['FIRE', 'LBFGS'] = 'FIRE',
    fmax: float = 0.001,
    max_steps: int = 1000,
    calculator: Literal['orb-v3-direct-20-omat', 'orb-v3-conservative-inf-omat'] = 'orb-v3-direct-20-omat',
    device: Literal['cpu', 'cuda'] = 'cuda'
) -> Atoms:
    """
    Relax atoms (legacy function for backward compatibility).

    This function exists for backward compatibility with the original
    run_chat.py code. New code should use StructureOptimizer directly.

    Args:
        atoms: Atoms object to relax
        hydrostatic_cell_relaxation: Allow volume change
        optimizer: 'FIRE' or 'LBFGS'
        fmax: Force convergence criterion
        max_steps: Maximum number of optimization steps
        calculator: Calculator model name
        device: Device for calculator ('cpu' or 'cuda', default: 'cuda')

    Returns:
        Relaxed Atoms object
    """
    from .calculators import load_calculator

    calc = load_calculator(calculator, device=device)
    opt_obj = StructureOptimizer(calc)

    return opt_obj.relax(
        atoms,
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
        hydrostatic_strain=hydrostatic_cell_relaxation
    )
