#!/usr/bin/env python3
"""
QHA (Quasi-Harmonic Approximation) and Phono3py wrapper functions.

This module provides functions for computing temperature-dependent properties:
- QHA: Bulk modulus B(T), Volume V(T), Thermal expansion α(T), Heat capacity Cp(T)
- Phono3py: Thermal conductivity κ(T)

Author: OptiMat Alloys Development Team
"""

import csv
import numpy as np
import threading
from collections import Counter
from math import gcd
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.formula import Formula
from phonopy import Phonopy, PhonopyQHA
from phonopy.structure.atoms import PhonopyAtoms
import warnings

from src.core.cancellation import check_cancellation, ProgressCallback
from src.core.optimization import StructureOptimizer


def ase_to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    """
    Convert ASE Atoms to Phonopy PhonopyAtoms.

    Args:
        atoms: ASE Atoms object

    Returns:
        PhonopyAtoms object for Phonopy
    """
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions()
    )


def get_number_of_formula_units(atoms: Atoms) -> int:
    """
    Determine the number of formula units in the structure.

    The formula unit is found by computing the greatest common divisor (GCD)
    of all element counts. The number of formula units is then the total
    number of atoms divided by atoms per formula unit.

    Examples:
        - Cu32: counts = {Cu: 32}, GCD = 32, formula = Cu, n_formula = 32
        - Li8Co8O16: counts = {Li: 8, Co: 8, O: 16}, GCD = 8, formula = LiCoO2, n_formula = 8
        - Cu16Ag16: counts = {Cu: 16, Ag: 16}, GCD = 16, formula = CuAg, n_formula = 16

    Args:
        atoms: ASE Atoms object

    Returns:
        Number of formula units in the structure
    """
    # Count atoms of each element
    symbols = atoms.get_chemical_symbols()
    element_counts = Counter(symbols)

    # Find GCD of all counts
    counts = list(element_counts.values())
    if len(counts) == 0:
        return 1

    # Compute GCD of all counts
    overall_gcd = reduce(gcd, counts)

    # Number of formula units = total atoms / atoms per formula
    n_formula = len(atoms) // overall_gcd

    return n_formula


def compute_qha_properties(
    atoms: Atoms,
    calculator: Calculator,
    cpu_calculator: Calculator,
    model_name: str,
    num_volumes: int = 11,
    strain_range: float = 0.10,
    mesh: Union[List[int], Tuple[int, int, int]] = (20, 20, 20),
    t_min: float = 0,
    t_max: float = 610,
    t_step: float = 10,
    supercell_matrix: Optional[List[List[int]]] = None,
    primitive_matrix: str = 'auto',
    symprec: float = 5e-3,
    distance: float = 0.01,
    cancellation_event: Optional[threading.Event] = None,
    relaxation_callback: Optional[Callable[[float, float], None]] = None,
    progress_callback: ProgressCallback = None
) -> Dict[str, np.ndarray]:
    """
    Compute QHA (Quasi-Harmonic Approximation) properties.

    This function:
    1. Generates strained volumes around equilibrium
    2. Computes phonons for each volume
    3. Runs QHA analysis to extract temperature-dependent properties

    Args:
        atoms: Relaxed ASE Atoms structure at equilibrium
        calculator: ASE calculator for force calculations (GPU, used for phonons)
        cpu_calculator: ASE calculator for CPU relaxation (passed from caller)
        model_name: Model name string for logging purposes
        num_volumes: Number of volume points (default: 11)
        strain_range: Volume strain range as fraction (default: 0.10 = ±10%)
        mesh: Mesh for phonon calculations (default: 20×20×20)
        t_min: Minimum temperature in K (default: 0)
        t_max: Maximum temperature in K (default: 610)
        t_step: Temperature step in K (default: 10)
        supercell_matrix: Supercell matrix for force constant calculation (default: identity, uses structure as-is)
        primitive_matrix: Primitive cell matrix (default: 'auto')
        symprec: Symmetry precision in Angstrom (default: 5e-3, appropriate for relaxed structures)
        distance: Displacement distance for force constants in Angstrom (default: 0.01)
        cancellation_event: Threading event for cooperative cancellation
        relaxation_callback: Callback called after structure relaxation, receives (V0_initial, V0_relaxed)
        progress_callback: Callback for progress reporting

    Returns:
        Dictionary containing:
            - temperatures: Temperature array (K)
            - gibbs_free_energy: Gibbs free energy G(T) vs T (kJ/mol)
            - bulk_modulus: Bulk modulus B(T) vs T (GPa)
            - volume: Volume V(T) vs T (Å³)
            - thermal_expansion: Linear thermal expansion coefficient α(T) vs T (1/K)
                                 Note: Converted from volumetric β (α = β/3)
            - thermal_expansion_volumetric: Volumetric thermal expansion coefficient β(T) vs T (1/K)
                                            Note: Direct output from PhonopyQHA
            - heat_capacity_p: Isobaric heat capacity Cp(T) vs T (J/K/mol)
            - gruneisen: Grüneisen parameter γ(T) vs T (dimensionless)
            - helmholtz_volume: Helmholtz free energy for each volume and T (kJ/mol)
            - volumes_used: Array of volumes used in QHA (Å³)
            - energies_used: Array of electronic energies at each volume (eV)

    Raises:
        ValueError: If atoms is not relaxed or invalid parameters
        RuntimeError: If QHA calculation fails
        ComputationCancelledException: If cancelled by user
    """
    from phonopy import Phonopy
    from phonopy.qha.core import QHA

    # Validate inputs
    if num_volumes < 5:
        raise ValueError("num_volumes must be at least 5 for QHA")

    if strain_range <= 0 or strain_range > 0.1:
        raise ValueError("strain_range must be between 0 and 0.1")

    # Use structure as-is (no supercell expansion)
    # If supercell_matrix is None, use identity matrix (no expansion)
    if supercell_matrix is None:
        supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Temperature array will be extracted from Phonopy after first volume calculation
    # This ensures perfect alignment between our array and Phonopy's thermal properties arrays
    temperatures = None

    print(f"QHA: Computing phonons for {num_volumes} volumes", flush=True)
    print(f"  Input volume: {atoms.get_volume():.3f} Ų", flush=True)
    print(f"  Supercell: {supercell_matrix}", flush=True)
    print(f"  Mesh: {mesh}", flush=True)
    print(f"  Temperature range: {t_min:.0f} - {t_max:.0f} K (step: {t_step:.0f} K)", flush=True)

    # Relax the input structure first (both cell and positions) to ensure proper equilibrium
    print("\n  Pre-relaxing input structure...", flush=True)
    print("    This ensures we start from true equilibrium before applying volume strains", flush=True)

    # Use CPU calculator for relaxation (GPU calculator used for phonons)
    optimizer = StructureOptimizer(cpu_calculator)

    # Relax using defaults: fmax=0.001, max_steps=1000, hydrostatic_strain=True
    relaxed_atoms = optimizer.relax(atoms.copy())

    V0_initial = atoms.get_volume()
    V0_relaxed = relaxed_atoms.get_volume()
    print(f"    ✓ Relaxation complete", flush=True)
    print(f"    Initial volume: {V0_initial:.3f} Ų", flush=True)
    print(f"    Relaxed volume: {V0_relaxed:.3f} Ų", flush=True)
    print(f"    Volume change: {(V0_relaxed - V0_initial)/V0_initial * 100:+.2f}%", flush=True)

    # Notify caller that relaxation is complete
    if relaxation_callback:
        relaxation_callback(V0_initial, V0_relaxed)

    # Test primitive cell detection after relaxation
    print(f"\n    Testing primitive cell detection after tight relaxation...", flush=True)
    phnpy_test = ase_to_phonopy_atoms(relaxed_atoms)
    phonon_test = Phonopy(phnpy_test, supercell_matrix=np.eye(3), primitive_matrix='auto', symprec=symprec)
    if phonon_test.primitive is not None:
        n_prim_detected = len(phonon_test.primitive)
        print(f"    Detected primitive cell: {n_prim_detected} atoms", flush=True)
        # Check cell parameters (phonon_test.primitive.cell is numpy array)
        cell = phonon_test.primitive.cell
        a_prim = np.linalg.norm(cell[0])
        b_prim = np.linalg.norm(cell[1])
        c_prim = np.linalg.norm(cell[2])
        print(f"    Primitive cell shape: a={a_prim:.4f}, b={b_prim:.4f}, c={c_prim:.4f} Å", flush=True)
        if abs(a_prim - b_prim) < 0.01 and abs(b_prim - c_prim) < 0.01:
            print(f"    ✓ Primitive cell is cubic (symmetric)", flush=True)
        else:
            print(f"    ⚠ Primitive cell is NOT cubic", flush=True)
    else:
        print(f"    ✗ No primitive cell detected (using full structure)", flush=True)

    # Use the relaxed structure as the new equilibrium reference
    atoms = relaxed_atoms
    V0 = atoms.get_volume()

    # Regenerate volume points around the new equilibrium
    volume_strains = np.linspace(-strain_range, strain_range, num_volumes)
    volumes = V0 * (1 + volume_strains)

    print(f"    New equilibrium volume: {V0:.3f} Ų", flush=True)
    print(f"    Volume range for QHA: {volumes[0]:.3f} - {volumes[-1]:.3f} Ų", flush=True)

    # Storage for results
    energies_at_volumes = []  # Electronic energy at each volume (eV)
    fe_phonon_at_volumes = []  # Phonon Helmholtz free energy F_ph(V,T) (kJ/mol)
    entropy_at_volumes = []  # Phonon entropy S_ph(V,T) (J/K/mol)
    cv_at_volumes = []  # Phonon heat capacity Cv_ph(V,T) (J/K/mol)
    n_atoms_primitive = None  # Will be set from first volume's phonon calculation
    n_formula_units_primitive = None  # Formula units in Phonopy primitive cell
    reduced_formula_primitive = None  # Reduced formula of primitive cell

    # Phonon stability tracking (imaginary frequencies = dynamical instability)
    imaginary_at_any_volume = False  # True if any volume has imaginary modes
    overall_min_frequency = float('inf')  # Minimum frequency across all volumes (THz)

    # Process each volume
    for i, (vol, strain) in enumerate(zip(volumes, volume_strains), 1):
        # Check cancellation at the beginning of each volume iteration
        check_cancellation(
            cancellation_event,
            i - 1,  # Completed volumes (0-indexed)
            num_volumes,
            "QHA volume calculation"
        )

        # Report progress
        if progress_callback:
            progress_callback(
                i - 1,
                num_volumes,
                f"Computing QHA for volume {i}/{num_volumes}"
            )

        print(f"\n  [{i}/{num_volumes}] Volume strain: {strain:+.3f} ({vol:.3f} Ų)", flush=True)

        # Create strained structure
        strained_atoms = atoms.copy()
        strained_atoms.set_cell(atoms.get_cell() * (1 + strain)**(1/3), scale_atoms=True)

        # Relax at fixed volume using StructureOptimizer (positions only, cell fixed)
        print(f"    Relaxing (fmax=0.01)...", flush=True)
        strained_atoms = optimizer.relax(strained_atoms, fmax=0.01, max_steps=100, hydrostatic_strain=False)

        # Get energy at this volume
        energy = strained_atoms.get_potential_energy()
        energies_at_volumes.append(energy)

        print(f"    Energy: {energy:.6f} eV", flush=True)

        # Convert to Phonopy atoms
        phnpy_atoms = ase_to_phonopy_atoms(strained_atoms)

        # Create Phonopy object
        phonon = Phonopy(
            phnpy_atoms,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            symprec=symprec
        )

        # Generate displacements
        phonon.generate_displacements(distance=distance)

        # Compute forces for each displacement
        print(f"    Computing forces for {len(phonon.supercells_with_displacements)} displacements...", flush=True)

        forces_list = []
        for supercell in phonon.supercells_with_displacements:
            # Convert back to ASE
            sc_ase = Atoms(
                symbols=supercell.symbols,
                positions=supercell.positions,
                cell=supercell.cell,
                pbc=True
            )
            sc_ase.calc = calculator

            forces = sc_ase.get_forces()
            forces_list.append(forces)

        # Set forces and produce force constants
        phonon.forces = forces_list
        phonon.produce_force_constants()

        print(f"    ✓ Force constants computed", flush=True)

        # Compute thermal properties on mesh
        phonon.run_mesh(mesh)

        # Check for imaginary (negative) frequencies at this volume
        # In Phonopy, imaginary frequencies are represented as negative values (in THz)
        mesh_frequencies = phonon.mesh.frequencies  # Shape: (n_qpoints, n_bands)
        vol_min_freq = float(np.min(mesh_frequencies))
        vol_has_imaginary = vol_min_freq < -0.1  # Threshold for numerical noise (THz)

        if vol_has_imaginary:
            print(f"    ⚠️ Imaginary frequencies detected (min: {vol_min_freq:.3f} THz)")
            imaginary_at_any_volume = True
        if vol_min_freq < overall_min_frequency:
            overall_min_frequency = vol_min_freq

        # Use t_step, t_max, t_min parameters (not temperatures array)
        # This follows the phonopy workflow for thermal properties computation
        phonon.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)

        # Extract thermal properties vs temperature
        tp_dict = phonon.get_thermal_properties_dict()

        # Validate that tp_dict is not None
        if tp_dict is None:
            raise RuntimeError(f"Phonon calculation failed at volume {i}/{num_volumes}: get_thermal_properties_dict() returned None")

        # Extract primitive cell information on first volume
        if i == 1:
            if phonon.primitive is not None:
                n_atoms_primitive = len(phonon.primitive)
                # Compute n_formula_units for the PRIMITIVE CELL (not the input structure!)
                # This is critical for normalizing Cv/Cp correctly
                primitive_symbols = phonon.primitive.symbols
                primitive_formula = Formula.from_list(primitive_symbols)
                reduced_formula_primitive, n_formula_units_primitive = primitive_formula.reduce()
            else:
                n_atoms_primitive = len(strained_atoms)  # Fallback: assume supercell = primitive
                # Fallback: use input structure for formula calculation
                primitive_symbols = atoms.get_chemical_symbols()
                primitive_formula = Formula.from_list(primitive_symbols)
                reduced_formula_primitive, n_formula_units_primitive = primitive_formula.reduce()

        fe_phonon = tp_dict.get('free_energy')  # kJ/mol (phonon contribution)
        entropy = tp_dict.get('entropy')  # J/K/mol (phonon contribution)
        cv = tp_dict.get('heat_capacity')  # J/K/mol (phonon contribution, Cv)

        # Validate thermal properties
        if fe_phonon is None:
            raise RuntimeError(f"Phonon calculation failed at volume {i}/{num_volumes}: free_energy is None")
        if entropy is None:
            raise RuntimeError(f"Phonon calculation failed at volume {i}/{num_volumes}: entropy is None")
        if cv is None:
            raise RuntimeError(f"Phonon calculation failed at volume {i}/{num_volumes}: heat_capacity is None")

        # Extract temperatures from first volume calculation
        if i == 1 and temperatures is None:
            # Get actual temperature array from Phonopy's output
            temperatures = tp_dict.get('temperatures')
            if temperatures is not None:
                print(f"    ✓ Temperature array extracted from Phonopy: {len(temperatures)} points", flush=True)
                print(f"      Range: {temperatures[0]:.0f} - {temperatures[-1]:.0f} K", flush=True)
            else:
                raise RuntimeError("Phonopy thermal properties dict does not contain 'temperatures' key")

        # Store phonon properties (keep in original units for QHA)
        fe_phonon_at_volumes.append(fe_phonon)
        entropy_at_volumes.append(entropy)
        cv_at_volumes.append(cv)

        print(f"    ✓ Thermal properties computed", flush=True)

    # Report completion of all volumes
    if progress_callback:
        progress_callback(
            num_volumes,
            num_volumes,
            f"All {num_volumes} volumes completed"
        )

    # Convert to arrays and transpose to shape (n_temps, n_vols)
    energies_at_volumes = np.array(energies_at_volumes)  # Shape: (n_vols,)
    fe_phonon_at_volumes = np.array(fe_phonon_at_volumes).T  # Shape: (n_temps, n_vols)
    entropy_at_volumes = np.array(entropy_at_volumes).T  # Shape: (n_temps, n_vols)
    cv_at_volumes = np.array(cv_at_volumes).T  # Shape: (n_temps, n_vols)

    # Run QHA
    print(f"\n  Running QHA analysis...", flush=True)

    # Temperatures array was already created at the beginning using linspace
    # No need to extract from tp_dict - we use the same array for consistency

    # CRITICAL: Scale BOTH U_static and volumes to primitive cell size BEFORE passing to PhonopyQHA
    # This ensures:
    # 1. Consistent units with F_phonon (kJ/mol per primitive from Phonopy)
    # 2. Correct E(V) curvature for accurate bulk modulus calculation
    # 3. Both Gibbs free energy AND bulk modulus are correct
    n_atoms_input = len(atoms)
    scaling_factor = n_atoms_primitive / n_atoms_input
    energies_per_primitive = energies_at_volumes * scaling_factor

    # Scale volumes to primitive cell
    # This ensures consistent units with scaled energies for correct bulk modulus
    volumes_primitive = volumes * scaling_factor

    # Create QHA object with correct parameters
    # PhonopyQHA expects:
    # - volumes: in Ų (positional argument) - SCALED to primitive cell!
    # - electronic_energies: in eV (positional argument) - SCALED to primitive cell!
    # - temperatures: in K
    # - free_energy: phonon Helmholtz free energy in kJ/mol (required)
    # - cv: phonon heat capacity in J/K/mol (required)
    # - entropy: phonon entropy in J/K/mol (required)
    # - eos: equation of state
    qha = PhonopyQHA(
        volumes_primitive,  # Positional argument (Ų) - SCALED to primitive cell!
        energies_per_primitive,  # Positional argument (eV) - SCALED to primitive cell!
        temperatures=temperatures,  # K (explicitly provided, so t_max not needed)
        free_energy=fe_phonon_at_volumes,  # kJ/mol per primitive cell, shape: (n_temps, n_vols)
        cv=cv_at_volumes,  # J/K/mol per primitive cell, shape: (n_temps, n_vols)
        entropy=entropy_at_volumes,  # J/K/mol per primitive cell, shape: (n_temps, n_vols)
        eos='vinet'  # Vinet EOS for better behavior at large compressions
    )

    # Extract QHA results (use correct attribute names from Phonopy)
    bulk_modulus = np.asarray(qha.bulk_modulus_temperature)  # GPa vs T
    # PhonopyQHA drops the last temperature point when computing derivative-based properties
    # (e.g., thermal expansion = dV/dT). Slice temperatures to match actual QHA output length.
    # Example: input 62 points (0-610 K) → output 61 points (0-600 K)
    qha_temperatures = temperatures[:len(bulk_modulus)]  # Match actual output length
    gibbs_temperature_eV = np.asarray(qha.gibbs_temperature)  # eV per primitive cell vs T

    # Convert Gibbs free energy from eV per primitive cell to kJ/mol per formula unit
    # PhonopyQHA outputs Gibbs energy for the Phonopy PRIMITIVE CELL in eV
    # We must normalize by n_formula_units_primitive (not input structure n_formula_units!)
    # Standard conversion: 1 eV = 96.485 kJ/mol
    if n_formula_units_primitive is None:
        raise RuntimeError("n_formula_units_primitive was not computed - check QHA volume loop")
    gibbs_temperature = gibbs_temperature_eV * 96.485 / n_formula_units_primitive  # kJ/mol per formula unit vs T

    volume_temperature = np.asarray(qha.volume_temperature)  # Ų vs T
    thermal_expansion_volumetric = np.asarray(qha.thermal_expansion)  # Volumetric β (1/K) vs T
    # Convert volumetric thermal expansion (β) to linear (α) for isotropic materials: α ≈ β/3
    thermal_expansion = thermal_expansion_volumetric / 3.0  # Linear α (1/K) vs T
    cp_temperature_raw = np.asarray(qha.heat_capacity_P_numerical)  # J/(K·mol) per primitive cell vs T
    # Normalize Cp per formula unit
    # PhonopyQHA returns Cp per primitive cell (n_atoms_primitive atoms)
    # We normalize by n_formula_units_primitive (from primitive cell) to get per formula unit
    if n_formula_units_primitive is None:
        raise RuntimeError("n_formula_units_primitive was not computed - check QHA volume loop")
    cp_temperature = cp_temperature_raw / n_formula_units_primitive  # J/(K·mol) per formula unit
    gruneisen_temperature = np.asarray(qha.gruneisen_temperature)  # dimensionless vs T

    # Validate that QHA computed results successfully
    if bulk_modulus is None:
        raise RuntimeError("QHA failed to compute bulk_modulus_temperature. Check if volumes span a sufficient range.")
    if volume_temperature is None:
        raise RuntimeError("QHA failed to compute volume_temperature.")
    if thermal_expansion is None:
        raise RuntimeError("QHA failed to compute thermal_expansion.")
    if cp_temperature is None:
        raise RuntimeError("QHA failed to compute heat_capacity_P_numerical.")

    print(f"  ✓ QHA analysis complete", flush=True)

    # Compute total Helmholtz free energy: F_total(T,V) = U_static(V) + F_phonon(T,V)
    # CRITICAL: Must maintain unit consistency!
    # - U_static: eV per primitive cell (from ORB calculator)
    # - F_phonon: kJ/mol per primitive cell (from Phonopy)
    # Strategy: Convert both to eV per primitive cell, sum, then convert to kJ/mol per formula unit

    # Step 1: Get primitive cell info (already computed during QHA volume loop)
    if n_formula_units_primitive is None:
        raise RuntimeError("n_formula_units_primitive was not computed - check QHA volume loop")
    if n_atoms_primitive is None:
        raise RuntimeError("n_atoms_primitive was not computed - check QHA volume loop")

    print(f"\n  Primitive cell analysis (from Phonopy):", flush=True)
    print(f"    Primitive cell has {n_atoms_primitive} atoms", flush=True)
    print(f"    Reduced formula: {reduced_formula_primitive}", flush=True)
    print(f"    Number of formula units in primitive: {n_formula_units_primitive}", flush=True)
    print(f"    Atoms per formula unit: {n_atoms_primitive / n_formula_units_primitive:.1f}", flush=True)

    # Also compute formula for the INPUT structure (for return dictionary and logging)
    formula_obj = Formula.from_list(atoms.get_chemical_symbols())
    reduced_formula, n_formula_units = formula_obj.reduce()
    atoms_per_formula = len(atoms) / n_formula_units

    # Step 2: Convert F_phonon from kJ/mol → eV (per primitive cell)
    # Phonopy returns kJ/mol per primitive cell (NOT per atom, NOT per formula!)
    fe_phonon_eV = fe_phonon_at_volumes / 96.485  # Shape: (n_temps, n_vols)

    # Step 3: Sum U_static + F_phonon in eV per primitive cell
    # Note: energies_per_primitive was already computed above when scaling for PhonopyQHA
    helmholtz_total_eV = fe_phonon_eV + energies_per_primitive[np.newaxis, :]  # Broadcast U_static
    # Shape: (n_temps, n_vols), Units: eV per primitive cell

    # Step 4: Convert F_total to kJ/mol per formula unit (using PRIMITIVE cell formula units!)
    helmholtz_total = (helmholtz_total_eV * 96.485) / n_formula_units_primitive
    # Shape: (n_temps, n_vols), Units: kJ/mol per formula unit

    # Step 5: Normalize entropy and Cv per formula unit
    # Phonopy returns these per primitive cell (n_atoms_primitive atoms)
    # We normalize by n_formula_units_primitive (from primitive cell) to get per formula unit
    # This matches the normalization used for Gibbs and Helmholtz free energies
    if n_formula_units_primitive is None:
        raise RuntimeError("n_formula_units_primitive was not computed - check QHA volume loop")
    entropy_volume = entropy_at_volumes / n_formula_units_primitive  # J/(K·mol) per formula unit
    cv_volume = cv_at_volumes / n_formula_units_primitive  # J/(K·mol) per formula unit

    # Step 6: Trim 2D arrays to match qha_temperatures length
    # PhonopyQHA drops the last temperature point for derivative-based 1D properties,
    # but 2D arrays (from phonon calculations) retain all points. Trim to match.
    n_qha_temps = len(qha_temperatures)
    helmholtz_total = helmholtz_total[:n_qha_temps]
    entropy_volume = entropy_volume[:n_qha_temps]
    cv_volume = cv_volume[:n_qha_temps]

    # Ensure all arrays are numpy arrays (PhonopyQHA may return lists)
    return {
        'temperatures': np.asarray(qha_temperatures),  # Use QHA's actual temperatures, not input
        'bulk_modulus': np.asarray(bulk_modulus),
        'volume': np.asarray(volume_temperature),
        'thermal_expansion': np.asarray(thermal_expansion),  # Linear α (1/K) - converted from volumetric β
        'thermal_expansion_volumetric': np.asarray(thermal_expansion_volumetric),  # Volumetric β (1/K)
        'heat_capacity_p': np.asarray(cp_temperature),  # J/(K·mol) per formula unit vs T
        'gruneisen': np.asarray(gruneisen_temperature),
        'gibbs_free_energy': np.asarray(gibbs_temperature),  # kJ/mol per formula unit vs T
        'helmholtz_volume': np.asarray(helmholtz_total),  # 2D: (n_temps, n_vols) kJ/mol per formula unit
        'entropy_volume': np.asarray(entropy_volume),  # 2D: (n_temps, n_vols) J/(K·mol) per formula unit
        'cv_volume': np.asarray(cv_volume),  # 2D: (n_temps, n_vols) J/(K·mol) per formula unit
        'volumes_used': np.asarray(volumes_primitive),  # Volumes per primitive cell
        'energies_used': np.asarray(energies_at_volumes),
        'n_formula_units': int(n_formula_units),  # Number of formula units in INPUT structure
        'reduced_formula': str(reduced_formula),  # Reduced chemical formula of INPUT structure (e.g., "CuAg")
        'atoms_per_formula': float(atoms_per_formula),  # Atoms per formula unit (e.g., 2.0 for CuAg)
        'n_atoms_primitive': int(n_atoms_primitive) if n_atoms_primitive is not None else None,  # Phonopy primitive cell size
        'n_formula_units_primitive': int(n_formula_units_primitive) if n_formula_units_primitive is not None else None,  # Formula units in PRIMITIVE cell (used for Cv/Cp normalization)
        'reduced_formula_primitive': str(reduced_formula_primitive) if reduced_formula_primitive is not None else None,  # Reduced formula of PRIMITIVE cell
        # Phonon stability (dynamical stability assessment)
        'has_imaginary_modes': bool(imaginary_at_any_volume),  # True if imaginary frequencies at any volume
        'min_frequency_THz': float(overall_min_frequency) if overall_min_frequency != float('inf') else None,  # Minimum frequency (THz)
        'dynamically_stable': not imaginary_at_any_volume,  # True if no imaginary modes at any volume
    }


def export_qha_csv(
    qha_data: Dict[str, np.ndarray],
    filepath: Union[str, Path]
) -> None:
    """
    Export QHA properties to CSV file.

    Args:
        qha_data: Dictionary from compute_qha_properties()
        filepath: Output CSV file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Extract data
    temps = qha_data['temperatures']
    bulk_modulus = qha_data['bulk_modulus']
    volume = qha_data['volume']
    thermal_expansion = qha_data['thermal_expansion']
    heat_capacity_p = qha_data['heat_capacity_p']
    gruneisen = qha_data['gruneisen']
    gibbs_free_energy = qha_data['gibbs_free_energy']

    # Write CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Temperature (K)',
            'Gibbs Free Energy (kJ/mol)',
            'Bulk Modulus (GPa)',
            'Volume (Ų)',
            'Thermal Expansion (1/K)',
            'Heat Capacity Cp (J/K/mol)',
            'Gruneisen Parameter'
        ])

        # Data rows - use min length to handle case where QHA drops points
        # QHA may drop T=0K or other points when computing certain properties
        num_points = min(len(temps), len(gibbs_free_energy), len(bulk_modulus), len(volume),
                        len(thermal_expansion), len(heat_capacity_p), len(gruneisen))

        for i in range(num_points):
            writer.writerow([
                f"{temps[i]:.1f}",
                f"{gibbs_free_energy[i]:.6f}",
                f"{bulk_modulus[i]:.6f}",
                f"{volume[i]:.6f}",
                f"{thermal_expansion[i]:.9f}",
                f"{heat_capacity_p[i]:.6f}",
                f"{gruneisen[i]:.6f}"
            ])

    print(f"✓ QHA properties exported to: {filepath}", flush=True)


def export_qha_volume_csv(
    qha_data: Dict[str, np.ndarray],
    output_dir: Union[str, Path]
) -> None:
    """
    Export volume-dependent QHA properties (F, S, Cv) to CSV files.

    Creates three separate CSV files, one for each property.
    Each file has temperatures as rows and volumes as columns.

    Args:
        qha_data: Dictionary from compute_qha_properties() containing:
            - temperatures: 1D array (K)
            - volumes_used: 1D array (Å³)
            - helmholtz_volume: 2D array (n_temps, n_vols) kJ/mol
            - entropy_volume: 2D array (n_temps, n_vols) J/(K·mol)
            - cv_volume: 2D array (n_temps, n_vols) J/(K·mol)
        output_dir: Directory to save CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temps = qha_data['temperatures']
    vols = qha_data['volumes_used']

    # Determine number of valid temperature points (QHA may drop points like T=0K)
    # Use the 1D property lengths to ensure consistency between 1D and 2D exports
    num_temps = len(qha_data.get('gibbs_free_energy', temps))

    # Properties to export: (key, filename, units)
    properties = [
        ('helmholtz_volume', 'helmholtz_free_energy_vs_T_V.csv', 'kJ/mol'),
        ('entropy_volume', 'entropy_vs_T_V.csv', 'J/(K·mol)'),
        ('cv_volume', 'heat_capacity_cv_vs_T_V.csv', 'J/(K·mol)'),
    ]

    for key, filename, units in properties:
        if key not in qha_data:
            continue

        data = qha_data[key]  # Shape: (n_temps, n_vols)
        csv_path = output_dir / filename

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header: Temperature (K), V=xxx.xxx Å³, V=xxx.xxx Å³, ...
            header = ['Temperature (K)'] + [f'V={v:.3f} Å³' for v in vols]
            writer.writerow(header)

            # Add units row as comment
            units_row = [f'# Units: {units}'] + [''] * len(vols)
            writer.writerow(units_row)

            # Data rows - only write as many rows as we have in 1D properties
            # This ensures consistency and prevents shape mismatches in plotting
            for i in range(num_temps):
                row = [f"{temps[i]:.1f}"] + [f"{data[i, j]:.6f}" for j in range(len(vols))]
                writer.writerow(row)

        print(f"✓ {key} exported to: {csv_path}", flush=True)


def compute_thermal_conductivity(
    atoms: Atoms,
    calculator: Calculator,
    mesh: Union[List[int], Tuple[int, int, int]] = (20, 20, 20),
    t_min: float = 0,
    t_max: float = 610,
    t_step: float = 10,
    structure_dir: Optional[Path] = None,
    supercell_matrix: Optional[List[List[int]]] = None,
    primitive_matrix: str = 'auto',
    symprec: float = 5e-3,
    distance: float = 0.01,
    cutoff_pair_distance: Optional[float] = None,
    cancellation_event: Optional[threading.Event] = None,
    progress_callback: ProgressCallback = None
) -> Dict[str, np.ndarray]:
    """
    Compute thermal conductivity using phono3py.

    WARNING: This is computationally expensive! It requires:
    - 3rd-order force constants (100s of displacements)
    - Fine mesh for integration
    - Can take hours to days depending on system size

    Args:
        atoms: Relaxed ASE Atoms structure at equilibrium
        calculator: ASE calculator for force calculations
        mesh: Mesh for thermal conductivity calculation (default: 20×20×20)
        t_min: Minimum temperature in K (default: 0)
        t_max: Maximum temperature in K (default: 610)
        t_step: Temperature step in K (default: 10)
        structure_dir: Directory to save HDF5 file (creates thermal_conductivity/ subdirectory)
        supercell_matrix: Supercell matrix for force constant calculation (default: identity, uses structure as-is)
        primitive_matrix: Primitive cell matrix (default: 'auto')
        symprec: Symmetry precision in Angstrom (default: 5e-3, appropriate for relaxed structures)
        distance: Displacement distance for force constants in Angstrom (default: 0.01)
        cutoff_pair_distance: Cutoff distance for 3rd-order force constants (Å)
        cancellation_event: Threading event for cooperative cancellation
        progress_callback: Callback for progress reporting (completed, total, message)

    Returns:
        Dictionary containing:
            - temperatures: Temperature array (K)
            - kappa_xx: Thermal conductivity in xx direction (W/(m K))
            - kappa_yy: Thermal conductivity in yy direction (W/(m K))
            - kappa_zz: Thermal conductivity in zz direction (W/(m K))
            - kappa_iso: Isotropic average thermal conductivity (W/(m K))

    Raises:
        ImportError: If phono3py is not installed
        ValueError: If invalid parameters
        RuntimeError: If calculation fails
        ComputationCancelledException: If cancelled by user
    """
    try:
        from phono3py import Phono3py
    except ImportError:
        raise ImportError(
            "phono3py is not installed. Install with: pip install phono3py"
        )

    # Use structure as-is (no supercell expansion)
    # If supercell_matrix is None, use identity matrix (no expansion)
    if supercell_matrix is None:
        supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    print(f"Phono3py: Computing thermal conductivity", flush=True)
    print(f"  WARNING: This is computationally expensive!", flush=True)
    print(f"  Supercell: {supercell_matrix}", flush=True)
    print(f"  Mesh: {mesh}", flush=True)
    print(f"  Temperature range: {t_min} - {t_max} K", flush=True)

    # Convert to Phonopy atoms
    phnpy_atoms = ase_to_phonopy_atoms(atoms)

    # Create Phono3py object
    phono3py = Phono3py(
        phnpy_atoms,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        symprec=symprec
    )

    # Generate 3rd-order displacements
    print(f"\n  Generating 3rd-order displacements...", flush=True)
    phono3py.generate_displacements(
        distance=distance,
        cutoff_pair_distance=cutoff_pair_distance
    )

    n_disp = len(phono3py.supercells_with_displacements)
    print(f"  Number of displacements: {n_disp}", flush=True)
    print(f"  This will take a long time...", flush=True)

    # Compute forces for each displacement
    print(f"\n  Computing forces (this may take hours)...", flush=True)

    forces_list = []
    for i, supercell in enumerate(phono3py.supercells_with_displacements, 1):
        # Check cancellation at the beginning of each displacement
        check_cancellation(
            cancellation_event,
            i - 1,  # Completed displacements (0-indexed)
            n_disp,
            "thermal conductivity force calculation"
        )

        # Report progress
        if progress_callback:
            progress_callback(
                i - 1,
                n_disp,
                f"Computing forces for displacement {i}/{n_disp}"
            )

        if i % 10 == 0:
            print(f"    Progress: {i}/{n_disp} ({i/n_disp*100:.1f}%)", flush=True)

        # Convert to ASE
        sc_ase = Atoms(
            symbols=supercell.symbols,
            positions=supercell.positions,
            cell=supercell.cell,
            pbc=True
        )
        sc_ase.calc = calculator

        forces = sc_ase.get_forces()
        forces_list.append(forces)

    print(f"  ✓ All forces computed", flush=True)

    # Set forces and produce force constants
    print(f"\n  Producing 3rd-order force constants...", flush=True)
    phono3py.forces = forces_list
    phono3py.produce_fc3()
    phono3py.produce_fc2()

    print(f"  ✓ Force constants computed", flush=True)

    # Set up thermal conductivity calculation
    print(f"\n  Computing thermal conductivity on {mesh} mesh...", flush=True)

    # Set mesh
    phono3py.mesh_numbers = mesh

    # Run thermal conductivity calculation using RTA (relaxation time approximation)
    # This is faster than LBTE (linearized Boltzmann transport equation)
    temperatures = np.arange(t_min, t_max + t_step, t_step)

    # Filter out T=0 (causes division by zero in phono3py)
    temps_nonzero = temperatures[temperatures > 0]

    # Initialize thermal conductivity calculation
    phono3py.init_phph_interaction()

    # Create thermal_conductivity directory if structure_dir is provided
    if structure_dir is not None:
        kappa_dir = Path(structure_dir) / "thermal_conductivity"
        kappa_dir.mkdir(parents=True, exist_ok=True)

        # Change to this directory so phono3py writes the HDF5 file there
        import os
        original_dir = os.getcwd()
        os.chdir(kappa_dir)
    else:
        kappa_dir = None

    try:
        # Run thermal conductivity for ALL temperatures at once
        # This writes the HDF5 file: kappa-m{mesh[0]}{mesh[1]}{mesh[2]}.hdf5
        phono3py.run_thermal_conductivity(
            temperatures=temps_nonzero.tolist(),
            write_kappa=True  # Write HDF5 file
        )

        # Read thermal conductivity from HDF5 file
        import h5py

        hdf5_filename = f"kappa-m{mesh[0]}{mesh[1]}{mesh[2]}.hdf5"

        with h5py.File(hdf5_filename, 'r') as f:
            print(f"  ✓ Reading thermal conductivity from {hdf5_filename}", flush=True)
            print(f"    HDF5 keys: {list(f.keys())}", flush=True)

            # Read data arrays
            temps_hdf5 = f['temperature'][:]  # Temperature array
            kappa_hdf5 = f['kappa'][:]        # Shape: (n_temps, 6)

            print(f"    kappa array shape: {kappa_hdf5.shape}", flush=True)
            print(f"    Temperature range: {temps_hdf5[0]:.1f} - {temps_hdf5[-1]:.1f} K", flush=True)

        # Extract diagonal components (xx, yy, zz)
        if kappa_hdf5.shape[1] >= 3:
            # Full tensor: [kappa_xx, kappa_yy, kappa_zz, kappa_yz, kappa_xz, kappa_xy]
            kappa_results = kappa_hdf5[:, :3]
        elif kappa_hdf5.shape[1] == 1:
            # Isotropic case: replicate single value
            kappa_results = np.repeat(kappa_hdf5, 3, axis=1)
        else:
            raise RuntimeError(f"Unexpected kappa shape: {kappa_hdf5.shape}")

        # Add T=0 row at the beginning if needed
        if temperatures[0] == 0:
            kappa_results = np.vstack([[0.0, 0.0, 0.0], kappa_results])
            temps_final = temperatures
        else:
            temps_final = temps_hdf5

        print(f"  ✓ Thermal conductivity computed", flush=True)
        idx_300K = np.argmin(np.abs(temps_final - 300))
        print(f"    κ at 300 K: {kappa_results[idx_300K, 0]:.2f} W/(m K) (xx)", flush=True)

    finally:
        # Restore original directory
        if structure_dir is not None:
            os.chdir(original_dir)

    return {
        'temperatures': temps_final,
        'kappa_xx': kappa_results[:, 0],
        'kappa_yy': kappa_results[:, 1],
        'kappa_zz': kappa_results[:, 2],
        'kappa_iso': np.mean(kappa_results, axis=1)  # Isotropic average
    }


def export_thermal_conductivity_csv(
    kappa_data: Dict[str, np.ndarray],
    filepath: Union[str, Path]
) -> None:
    """
    Export thermal conductivity to CSV file.

    Args:
        kappa_data: Dictionary from compute_thermal_conductivity()
        filepath: Output CSV file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Extract data
    temps = kappa_data['temperatures']
    kappa_xx = kappa_data['kappa_xx']
    kappa_yy = kappa_data['kappa_yy']
    kappa_zz = kappa_data['kappa_zz']
    kappa_iso = kappa_data['kappa_iso']

    # Write CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Temperature (K)',
            'κ_xx (W/(m K))',
            'κ_yy (W/(m K))',
            'κ_zz (W/(m K))',
            'κ_iso (W/(m K))'
        ])

        # Data rows
        for i in range(len(temps)):
            writer.writerow([
                f"{temps[i]:.1f}",
                f"{kappa_xx[i]:.6f}",
                f"{kappa_yy[i]:.6f}",
                f"{kappa_zz[i]:.6f}",
                f"{kappa_iso[i]:.6f}"
            ])

    print(f"✓ Thermal conductivity exported to: {filepath}", flush=True)
