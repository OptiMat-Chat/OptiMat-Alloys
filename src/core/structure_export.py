"""Structure export functions for various file formats.

This module provides functions to export ASE Atoms objects to different
file formats commonly used in materials science:
- CIF (Crystallographic Information File)
- VASP POSCAR
- LAMMPS data file
- XYZ format

Also provides CSV export functions for property data.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import csv
import re
import numpy as np
from ase import Atoms
from ase.io import write as ase_write


def parse_composition_string(composition_string: str) -> List[str]:
    """Parse composition string to extract element order.

    Args:
        composition_string: e.g., "Cu50.0Ag50.0"

    Returns:
        List of elements in order, e.g., ['Cu', 'Ag']
    """
    # Match element symbols (1-2 letters, first uppercase)
    elements = re.findall(r'([A-Z][a-z]?)', composition_string)
    return elements


def sort_atoms_by_element_order(atoms: Atoms, element_order: List[str]) -> Atoms:
    """Reorder atoms to group by elements in specified order for file output.

    This does NOT change the physical structure - each atom keeps its original
    (x, y, z) position. Only the order of atoms in the output file changes.

    Args:
        atoms: ASE Atoms object
        element_order: List of element symbols in desired order, e.g., ['Cu', 'Ag']

    Returns:
        New Atoms object with atoms reordered by element (positions preserved)
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Create sorted indices: group atoms by element in user's order
    sorted_indices = []
    for elem in element_order:
        for i, s in enumerate(symbols):
            if s == elem:
                sorted_indices.append(i)

    # Handle any elements not in element_order (append at end)
    for i, s in enumerate(symbols):
        if i not in sorted_indices:
            sorted_indices.append(i)

    # Create new atoms with reordered sequence (positions preserved per atom)
    new_symbols = [symbols[i] for i in sorted_indices]
    new_positions = positions[sorted_indices]  # Each atom keeps its original position
    return Atoms(new_symbols, positions=new_positions, cell=atoms.cell, pbc=atoms.pbc)


def export_to_cif(atoms: Atoms, filepath: Path) -> None:
    """Export atoms to CIF format.

    Args:
        atoms: ASE Atoms object to export
        filepath: Output file path (should end with .cif)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(filepath), atoms, format='cif')


def export_to_vasp(atoms: Atoms, filepath: Path) -> None:
    """Export atoms to VASP POSCAR format.

    Args:
        atoms: ASE Atoms object to export
        filepath: Output file path (typically named POSCAR)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(filepath), atoms, format='vasp', vasp5=True)


def export_to_lammps(atoms: Atoms, filepath: Path, atom_style: str = "atomic") -> None:
    """Export atoms to LAMMPS data file format with atomic masses.

    Args:
        atoms: ASE Atoms object to export
        filepath: Output file path (should end with .lammps)
        atom_style: LAMMPS atom style (default: "atomic" for metals/alloys)
    """
    from ase.data import atomic_masses, chemical_symbols

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # First write with ASE
    ase_write(str(filepath), atoms, format='lammps-data', atom_style=atom_style)

    # Read the file and add Masses section
    with open(filepath, 'r') as f:
        content = f.read()

    # Get unique elements and their type IDs
    symbols = atoms.get_chemical_symbols()
    unique_symbols = []
    for s in symbols:
        if s not in unique_symbols:
            unique_symbols.append(s)

    # Build Masses section
    masses_lines = ["\nMasses\n"]
    for i, symbol in enumerate(unique_symbols, 1):
        atomic_num = chemical_symbols.index(symbol)
        mass = atomic_masses[atomic_num]
        masses_lines.append(f"{i} {mass:.4f}  # {symbol}")
    masses_section = "\n".join(masses_lines) + "\n"

    # Insert Masses section before "Atoms" line
    if "Atoms" in content:
        content = content.replace("\nAtoms", masses_section + "\nAtoms")

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)


def export_to_xyz(atoms: Atoms, filepath: Path) -> None:
    """Export atoms to XYZ format.

    Args:
        atoms: ASE Atoms object to export
        filepath: Output file path (should end with .xyz)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(filepath), atoms, format='xyz')


def export_all_structure_formats(
    atoms: Atoms,
    base_dir: Path,
    structure_id: str,
    composition_string: Optional[str] = None
) -> Dict[str, Path]:
    """Export atoms to all supported structure formats.

    If composition_string is provided, atoms are reordered in the output files
    to match the user's element order (e.g., "Cu50.0Ag50.0" -> Cu first, then Ag).
    This does NOT change atomic positions - only the order in output files.

    Args:
        atoms: ASE Atoms object to export
        base_dir: Base directory for structure files
        structure_id: Structure UUID for naming
        composition_string: Optional composition string to determine element order

    Returns:
        Dictionary mapping format names to file paths
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # If composition_string provided, reorder atoms to match user's element order
    atoms_for_export = atoms
    if composition_string:
        element_order = parse_composition_string(composition_string)
        if element_order:
            atoms_for_export = sort_atoms_by_element_order(atoms, element_order)

    exports = {}

    # CIF
    cif_path = base_dir / "structure.cif"
    export_to_cif(atoms_for_export, cif_path)
    exports["cif"] = cif_path

    # VASP POSCAR
    poscar_path = base_dir / "POSCAR"
    export_to_vasp(atoms_for_export, poscar_path)
    exports["vasp"] = poscar_path

    # LAMMPS
    lammps_path = base_dir / "structure.lammps"
    export_to_lammps(atoms_for_export, lammps_path, atom_style="atomic")
    exports["lammps"] = lammps_path

    # XYZ
    xyz_path = base_dir / "structure.xyz"
    export_to_xyz(atoms_for_export, xyz_path)
    exports["xyz"] = xyz_path

    return exports


def export_elastic_stiffness_csv(stiffness_tensor: np.ndarray, filepath: Path) -> None:
    """Export 6x6 elastic stiffness tensor to CSV.

    Args:
        stiffness_tensor: 6x6 stiffness tensor in Voigt notation (GPa)
        filepath: Output CSV file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Voigt notation labels
    voigt_labels = ['C11', 'C12', 'C13', 'C14', 'C15', 'C16',
                    'C21', 'C22', 'C23', 'C24', 'C25', 'C26',
                    'C31', 'C32', 'C33', 'C34', 'C35', 'C36',
                    'C41', 'C42', 'C43', 'C44', 'C45', 'C46',
                    'C51', 'C52', 'C53', 'C54', 'C55', 'C56',
                    'C61', 'C62', 'C63', 'C64', 'C65', 'C66']

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header with column labels
        writer.writerow([''] + [f'C{i+1}' for i in range(6)])
        # Write rows with row labels
        for i in range(6):
            row = [f'C{i+1}'] + [f'{stiffness_tensor[i, j]:.4f}' for j in range(6)]
            writer.writerow(row)

    # Also write flat format for easier programmatic access
    flat_filepath = filepath.parent / (filepath.stem + '_flat.csv')
    with open(flat_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component', 'value_GPa'])
        for i in range(6):
            for j in range(6):
                writer.writerow([f'C{i+1}{j+1}', f'{stiffness_tensor[i, j]:.4f}'])


def export_qha_properties_csv(
    temperatures: np.ndarray,
    gibbs_energies: Optional[np.ndarray],
    bulk_moduli: Optional[np.ndarray],
    volumes: Optional[np.ndarray],
    thermal_expansion: Optional[np.ndarray],
    heat_capacities: Optional[np.ndarray],
    gruneisen_params: Optional[np.ndarray],
    filepath: Path
) -> None:
    """Export 1D QHA properties vs temperature to CSV.

    Args:
        temperatures: Temperature array (K)
        gibbs_energies: Gibbs free energy (eV)
        bulk_moduli: Bulk modulus (GPa)
        volumes: Volume (Å³)
        thermal_expansion: Thermal expansion coefficient (1/K)
        heat_capacities: Isobaric heat capacity Cp (J/mol/K)
        gruneisen_params: Grüneisen parameter
        filepath: Output CSV file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Build header based on available data
        headers = ['Temperature_K']
        if gibbs_energies is not None:
            headers.append('Gibbs_energy_eV')
        if bulk_moduli is not None:
            headers.append('Bulk_modulus_GPa')
        if volumes is not None:
            headers.append('Volume_A3')
        if thermal_expansion is not None:
            headers.append('Thermal_expansion_1/K')
        if heat_capacities is not None:
            headers.append('Heat_capacity_Cp_J/mol/K')
        if gruneisen_params is not None:
            headers.append('Gruneisen_parameter')

        writer.writerow(headers)

        # Helper function for safe array access (handles different array lengths)
        def safe_get(arr, idx, fmt):
            if arr is not None and idx < len(arr):
                return fmt.format(arr[idx])
            return ''

        # Write data rows
        for i, T in enumerate(temperatures):
            row = [f'{T:.2f}']
            if gibbs_energies is not None:
                row.append(safe_get(gibbs_energies, i, '{:.6f}'))
            if bulk_moduli is not None:
                row.append(safe_get(bulk_moduli, i, '{:.4f}'))
            if volumes is not None:
                row.append(safe_get(volumes, i, '{:.6f}'))
            if thermal_expansion is not None:
                row.append(safe_get(thermal_expansion, i, '{:.6e}'))
            if heat_capacities is not None:
                row.append(safe_get(heat_capacities, i, '{:.6f}'))
            if gruneisen_params is not None:
                row.append(safe_get(gruneisen_params, i, '{:.6f}'))
            writer.writerow(row)


def export_2d_property_csv(
    temperatures: np.ndarray,
    volumes: np.ndarray,
    data: np.ndarray,
    property_name: str,
    filepath: Path
) -> None:
    """Export 2D property data (T x V matrix) to CSV.

    Args:
        temperatures: Temperature array (K)
        volumes: Volume array (Å³)
        data: 2D array of shape (n_temps, n_volumes)
        property_name: Name of the property for header
        filepath: Output CSV file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header: first column is T, rest are volumes
        headers = ['Temperature_K'] + [f'V={v:.4f}_A3' for v in volumes]
        writer.writerow(headers)

        # Write data rows
        for i, T in enumerate(temperatures):
            row = [f'{T:.2f}'] + [f'{data[i, j]:.6f}' for j in range(len(volumes))]
            writer.writerow(row)


def export_thermal_conductivity_csv(
    temperatures: np.ndarray,
    kappa_xx: np.ndarray,
    kappa_yy: np.ndarray,
    kappa_zz: np.ndarray,
    kappa_iso: np.ndarray,
    filepath: Path
) -> None:
    """Export thermal conductivity vs temperature to CSV.

    Args:
        temperatures: Temperature array (K)
        kappa_xx: x-component of thermal conductivity (W/m/K)
        kappa_yy: y-component of thermal conductivity (W/m/K)
        kappa_zz: z-component of thermal conductivity (W/m/K)
        kappa_iso: Isotropic average (W/m/K)
        filepath: Output CSV file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Temperature_K', 'kappa_xx_W/m/K', 'kappa_yy_W/m/K',
                        'kappa_zz_W/m/K', 'kappa_iso_W/m/K'])

        for i, T in enumerate(temperatures):
            writer.writerow([
                f'{T:.2f}',
                f'{kappa_xx[i]:.4f}',
                f'{kappa_yy[i]:.4f}',
                f'{kappa_zz[i]:.4f}',
                f'{kappa_iso[i]:.4f}'
            ])


def export_rdf_csv(
    r_values: np.ndarray,
    g_r_values: np.ndarray,
    filepath: Path,
    partial: Optional[Dict[str, List[float]]] = None
) -> None:
    """Export radial distribution function to CSV.

    Args:
        r_values: Radial distance array (Å)
        g_r_values: g(r) values (total RDF)
        filepath: Output CSV file path
        partial: Optional dict of partial RDFs, e.g. {'Cu-Cu': [...], 'Cu-Ag': [...]}
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Build header: r_A, g_r_total, g_Cu-Cu, g_Cu-Ag, ...
        header = ['r_A', 'g_r_total']
        partial_keys = []
        if partial:
            partial_keys = sorted(partial.keys())
            header.extend([f'g_{key}' for key in partial_keys])
        writer.writerow(header)

        for i, (r, g) in enumerate(zip(r_values, g_r_values)):
            row = [f'{r:.4f}', f'{g:.6f}']
            if partial:
                for key in partial_keys:
                    partial_values = partial[key]
                    if i < len(partial_values):
                        row.append(f'{partial_values[i]:.6f}')
                    else:
                        row.append('')
            writer.writerow(row)


def export_elate_2d_csv(
    elate: Any,
    property_type: str,
    filepath: Path
) -> None:
    """Export ELATE 2D projection data for all 3 planes to CSV.

    Args:
        elate: ELATE object from mechelastic
        property_type: One of YOUNG, SHEAR, POISSON, BULK, LC, SHEAR_SPEED, COMPRESSION_SPEED
        filepath: Output CSV file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Property info for units
    property_info = {
        "YOUNG": ("E", "GPa"),
        "SHEAR": ("G", "GPa"),
        "POISSON": ("nu", ""),
        "BULK": ("K", "GPa"),
        "LC": ("beta", "1/TPa"),
        "SHEAR_SPEED": ("vs", "m/s"),
        "COMPRESSION_SPEED": ("vl", "m/s")
    }

    symbol, units = property_info.get(property_type, (property_type, ""))
    value_col = f"value_{units}" if units else "value"

    # Property calculation functions
    property_funcs = {
        "YOUNG": lambda theta, phi: elate.elas.Young([theta, phi]),
        "SHEAR": lambda theta, phi: elate.elas.shear([theta, phi, 0]),
        "POISSON": lambda theta, phi: elate.elas.Poisson([theta, phi, 0]),
        "BULK": lambda theta, phi: elate.elas.Bulk([theta, phi, 0]),
        "LC": lambda theta, phi: elate.elas.LC([theta, phi]),
        "SHEAR_SPEED": lambda theta, phi: elate.elas.Shear_Speed([theta, phi, 0]),
        "COMPRESSION_SPEED": lambda theta, phi: elate.elas.Compression_Speed([theta, phi, 0])
    }

    calc_func = property_funcs.get(property_type)
    if calc_func is None:
        raise ValueError(f"Unknown property type: {property_type}")

    # Define planes: (name, theta_func, phi_func)
    planes = [
        ("XY", lambda a: np.pi/2, lambda a: a),   # XY plane: theta=π/2, phi varies
        ("XZ", lambda a: a, lambda a: 0),          # XZ plane: theta varies, phi=0
        ("YZ", lambda a: a, lambda a: np.pi/2)    # YZ plane: theta varies, phi=π/2
    ]

    # Generate 360 angles
    angles = np.linspace(0, 2*np.pi, 360)
    angles_deg = np.degrees(angles)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['plane', 'angle_deg', 'x', 'y', value_col])

        for plane_name, theta_func, phi_func in planes:
            for angle, angle_deg_val in zip(angles, angles_deg):
                theta = theta_func(angle)
                phi = phi_func(angle)

                try:
                    value = calc_func(theta, phi)
                except Exception:
                    value = np.nan

                # Convert to Cartesian
                if plane_name == "XY":
                    x = value * np.cos(angle)
                    y = value * np.sin(angle)
                else:  # XZ and YZ
                    x = value * np.sin(angle)
                    y = value * np.cos(angle)

                if not np.isnan(value):
                    writer.writerow([plane_name, f'{angle_deg_val:.2f}', f'{x:.6f}', f'{y:.6f}', f'{value:.6f}'])


def export_elate_3d_csv(
    elate: Any,
    property_type: str,
    filepath: Path
) -> None:
    """Export ELATE 3D surface data to CSV.

    Args:
        elate: ELATE object from mechelastic
        property_type: One of YOUNG, SHEAR, POISSON, BULK, LC, SHEAR_SPEED, COMPRESSION_SPEED
        filepath: Output CSV file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Property info for units
    property_info = {
        "YOUNG": ("E", "GPa"),
        "SHEAR": ("G", "GPa"),
        "POISSON": ("nu", ""),
        "BULK": ("K", "GPa"),
        "LC": ("beta", "1/TPa"),
        "SHEAR_SPEED": ("vs", "m/s"),
        "COMPRESSION_SPEED": ("vl", "m/s")
    }

    symbol, units = property_info.get(property_type, (property_type, ""))
    value_col = f"value_{units}" if units else "value"

    # Property calculation functions
    property_funcs = {
        "YOUNG": lambda theta, phi: elate.elas.Young([theta, phi]),
        "SHEAR": lambda theta, phi: elate.elas.shear([theta, phi, 0]),
        "POISSON": lambda theta, phi: elate.elas.Poisson([theta, phi, 0]),
        "BULK": lambda theta, phi: elate.elas.Bulk([theta, phi, 0]),
        "LC": lambda theta, phi: elate.elas.LC([theta, phi]),
        "SHEAR_SPEED": lambda theta, phi: elate.elas.Shear_Speed([theta, phi, 0]),
        "COMPRESSION_SPEED": lambda theta, phi: elate.elas.Compression_Speed([theta, phi, 0])
    }

    calc_func = property_funcs.get(property_type)
    if calc_func is None:
        raise ValueError(f"Unknown property type: {property_type}")

    # Generate spherical mesh (same resolution as 3D plot)
    n_theta = 60
    n_phi = 120

    theta_arr = np.linspace(0, np.pi, n_theta)
    phi_arr = np.linspace(0, 2*np.pi, n_phi)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['theta_rad', 'phi_rad', 'x', 'y', 'z', value_col])

        for theta in theta_arr:
            for phi in phi_arr:
                try:
                    r = calc_func(theta, phi)
                except Exception:
                    r = np.nan

                if not np.isnan(r) and not np.isinf(r):
                    # Convert to Cartesian
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)

                    writer.writerow([
                        f'{theta:.6f}', f'{phi:.6f}',
                        f'{x:.6f}', f'{y:.6f}', f'{z:.6f}',
                        f'{r:.6f}'
                    ])
