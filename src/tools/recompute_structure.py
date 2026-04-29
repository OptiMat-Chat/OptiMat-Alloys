"""
Recompute Structure Tool

Recomputes an existing structure with a different calculator for comparison studies.
"""

from typing import Annotated, Dict, Optional, Union
import chainlit as cl
import numpy as np
import torch

# Core modules
from src.core.optimization import StructureOptimizer
from src.core.calculator_service import get_calculator_service
from src.core.analysis import structural_analysis, compute_density
from src.core.formation_energy import formation_energy_per_atom
from src.core.reference_data import precompute_and_save, load_reference_energies
from src.core.structure_builder import lattice_constant_from_atomic_volume
from src.storage.cache import get_reference_cache

# Visualization modules
from src.visualization.ovito_renderer import render_atoms, render_structure

# Storage modules
from src.storage.database import create_structure_database

# Session state for memory layer
from src.utils.session_state import SessionState


# All 118 elements (same as alloy_generation.py)
SUPPORTED_ELEMENTS = [
    "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be", "Bh", "Bi", "Bk", "Br",
    "C", "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Cn", "Co", "Cr", "Cs", "Cu", "Db", "Ds", "Dy",
    "Er", "Es", "Eu", "F", "Fe", "Fl", "Fm", "Fr", "Ga", "Gd", "Ge", "H", "He", "Hf", "Hg",
    "Ho", "Hs", "I", "In", "Ir", "K", "Kr", "La", "Li", "Lr", "Lu", "Lv", "Mc", "Md", "Mg",
    "Mn", "Mo", "Mt", "N", "Na", "Nb", "Nd", "Ne", "Nh", "Ni", "No", "Np", "O", "Og", "Os",
    "P", "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rf", "Rg", "Rh",
    "Rn", "Ru", "S", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te",
    "Th", "Ti", "Tl", "Tm", "Ts", "U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr"
]


@cl.step(type="tool")  # type: ignore
async def recompute_structure(
    source_ref: Annotated[Union[int, str], "Structure ID (local) or UUID (global) to recompute"],
    calculator: Annotated[Optional[str], "Calculator name (default: session setting)"] = None,
    fmax: Annotated[float, "Force convergence eV/Å (default 0.001)"] = 0.001,
) -> Annotated[Dict, "New UUID, source UUID, calculator comparison, and analysis."]:
    """
    Recompute an existing structure with a different calculator for rigorous comparison.

    Uses the EXACT SAME atomic configuration (unlike regenerating which creates new SQS).
    Essential for isolating calculator effects from structural randomness.

    Provenance: New entry has 'derived_from' linking to source UUID.
    """

    # Connect to database
    db = create_structure_database()

    # Display reference for user-friendly output (preserve original input format)
    ref_display = str(source_ref) if isinstance(source_ref, int) else source_ref[:8]

    # Resolve int→UUID once at tool layer (UUID is used for all subsequent calls)
    try:
        source_uuid = db.resolve_to_uuid(source_ref)
    except KeyError:
        return {
            "error": f"Structure '{ref_display}' not found in database.",
            "suggestion": "Use search_database tool to find valid structure IDs/UUIDs."
        }

    # Load source structure
    try:
        atoms = db.read(source_uuid)
        source_metadata = db.get_metadata(source_uuid)
    except Exception as e:
        return {
            "error": f"Failed to load structure '{ref_display}': {e}",
            "suggestion": "Check if the structure exists in the database."
        }

    # Extract original metadata
    source_calculator = source_metadata.get("calculator_name", "unknown")
    target_structure = source_metadata.get("target_structure", "unknown")
    elements = source_metadata.get("elements", [])

    # Reconstruct element fractions from metadata
    element_fractions = {}
    for elem in SUPPORTED_ELEMENTS:
        frac = source_metadata.get(f"{elem}_fraction", 0.0)
        if frac > 0:
            element_fractions[elem] = frac

    if not elements and element_fractions:
        elements = list(element_fractions.keys())

    optimized_fractions = [element_fractions.get(e, 0.0) for e in elements] if elements else []

    # Determine calculator to use
    new_calculator = calculator or cl.user_session.get("default_calculator") or "orb-v3-direct-20-omat"  # type: ignore

    # Create TaskList
    task_list = cl.TaskList()
    task_list.status = "Running..."

    task1 = cl.Task(title=f"Loading structure {ref_display}...", status=cl.TaskStatus.DONE)
    await task_list.add_task(task1)
    task2 = cl.Task(title=f"Stage 1: Coarse relaxation (CUDA, fmax=0.01) with {new_calculator}", status=cl.TaskStatus.READY)
    await task_list.add_task(task2)
    task3 = cl.Task(title=f"Stage 2: Fine relaxation (CPU, fmax={fmax})", status=cl.TaskStatus.READY)
    await task_list.add_task(task3)
    task4 = cl.Task(title="Analyzing and saving results", status=cl.TaskStatus.READY)
    await task_list.add_task(task4)

    composition_string = source_metadata.get("composition_string", "unknown")
    message = await cl.Message(
        content=f"Recomputing structure `{ref_display}...` ({composition_string}) with **{new_calculator}** (was: {source_calculator})"
    ).send()

    task1.forId = message.id
    task2.forId = message.id
    task3.forId = message.id
    task4.forId = message.id
    await task_list.send()

    # Clear calculator from atoms before re-relaxation
    atoms.calc = None

    # Stage 1: Coarse relaxation with CUDA
    task2.status = cl.TaskStatus.RUNNING
    await task_list.send()

    # Load calculator (supports ORB/MACE/NequIP via calculator_service)
    service = get_calculator_service()
    calc_cuda = await service.get_calculator_sync(new_calculator, 'cuda')
    optimizer_cuda = StructureOptimizer(calc_cuda)
    atoms = await cl.make_async(optimizer_cuda.relax)(atoms, fmax=0.01, optimizer='FIRE', max_steps=500)

    task2.status = cl.TaskStatus.DONE
    await task_list.send()

    # Stage 2: Fine relaxation with CPU
    task3.status = cl.TaskStatus.RUNNING
    await task_list.send()

    calc_cpu = await service.get_calculator_sync(new_calculator, 'cpu')
    optimizer_cpu = StructureOptimizer(calc_cpu)
    atoms = await cl.make_async(optimizer_cpu.relax)(atoms, fmax=fmax, optimizer='FIRE', max_steps=500)

    task3.status = cl.TaskStatus.DONE
    await task_list.send()

    # === EARLY SAVE: Capture relaxed structure immediately after relaxation ===
    # This ensures the structure is never lost if UI disconnects during analysis
    num_atoms = len(atoms)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    result = db.write(atoms, key_value_pairs={
        # Provenance - CRITICAL: Link to source
        "derived_from": source_uuid,
        "recomputed_from_calculator": source_calculator,
        # Calculator info
        "calculator_name": new_calculator,
        "device_type": device_type,
        # Structure
        "target_structure": target_structure,
        "composition_string": composition_string,
        "optimized_num_atoms": num_atoms,
        # Status marker - will be updated to "complete" after analysis
        "status": "analyzing",
    })
    new_id = result["id"]
    new_uuid = result["uuid"]

    # Analysis phase
    task4.status = cl.TaskStatus.RUNNING
    await task_list.send()

    # Calculate energy per atom (uses cached results from relaxation)
    energy_per_atom = atoms.get_potential_energy() / num_atoms

    # Get cache for new calculator and ensure reference data exists
    cache = get_reference_cache(calculator=new_calculator)
    if not cache.is_available():
        await cl.Message(
            content=(
                f"Reference data not available for {new_calculator}.\n\n"
                f"Starting precomputation now (this will take a few hours)..."
            )
        ).send()
        await cl.make_async(precompute_and_save)(
            hydrostatic_cell_relaxation=True,
            optimizer="FIRE",
            fmax=0.005,
            calculator=new_calculator,
            cache=cache,
        )

    # Get calculator-specific reference data paths
    _, energy_json_path = cache.get_paths()

    # Calculate formation energy and mixing energy
    if elements and optimized_fractions:
        refs = load_reference_energies(energy_json_path, reference_mode="ground_state")
        formation_energy = formation_energy_per_atom(
            elements=elements,
            fractions=optimized_fractions,
            energies_ref=refs,
            E_per_atom=energy_per_atom
        )

        if target_structure != "unknown":
            refs_structure = load_reference_energies(energy_json_path, reference_mode="same_structure", structure=target_structure)
            mixing_energy = formation_energy_per_atom(
                elements=elements,
                fractions=optimized_fractions,
                energies_ref=refs_structure,
                E_per_atom=energy_per_atom
            )
        else:
            mixing_energy = 0.0
    else:
        formation_energy = 0.0
        mixing_energy = 0.0

    # Compute physical properties
    density = compute_density(atoms)
    volume_per_atom = atoms.get_volume() / num_atoms
    lattice_constant = lattice_constant_from_atomic_volume(target_structure, volume_per_atom) if target_structure != "unknown" else 0.0

    # Calculate forces
    forces = atoms.get_forces()
    force_magnitudes = np.linalg.norm(forces, axis=1)
    max_force_magnitude = float(round(float(np.max(force_magnitudes)), 3))

    # Evaluate stress tensor
    stress_dict = {
        k: round(float(v) * 16.0218, 3)
        for k, v in zip(["xx", "yy", "zz", "yz", "xz", "xy"], atoms.get_stress())
    }

    # Perform PTM structural analysis
    ptm_analysis = structural_analysis(atoms)
    dominant_structure = max(ptm_analysis, key=ptm_analysis.get) if ptm_analysis else "unknown"

    # Calculate structural stability
    STRUCTURAL_STABILITY_THRESHOLD = 90.0
    structural_match_percent = ptm_analysis.get(target_structure, 0.0) if ptm_analysis and target_structure != "unknown" else 0.0
    is_structurally_stable = structural_match_percent >= STRUCTURAL_STABILITY_THRESHOLD

    # Build key_value_pairs for UPDATE (analysis results)
    # Note: Basic metadata (provenance, calculator, composition) already saved in early save
    key_value_pairs = {
        # Mark analysis as complete
        "status": "complete",

        # Structure
        "target_structure": target_structure,
        "lattice_constant": float(lattice_constant),
        "optimized_num_atoms": int(num_atoms),

        # Composition
        "num_elements": len(elements) if elements else 0,
        "composition_string": composition_string,

        # Energies
        "potential_energy_eV_per_atom": float(energy_per_atom),
        "formation_energy_ground_state_reference_eV_per_atom": float(formation_energy),
        "mixing_energy_same_structure_reference_eV_per_atom": float(mixing_energy),

        # Physical properties
        "density_g_per_cm3": float(density),
        "max_force_magnitude_eV_per_A": float(max_force_magnitude),

        # Structural stability
        "structural_match_percent": float(structural_match_percent),
        "is_structurally_stable": bool(is_structurally_stable),
    }

    # Initialize all element fractions to 0.0
    for elem in SUPPORTED_ELEMENTS:
        key_value_pairs[f"{elem}_fraction"] = 0.0

    # Override with actual element fractions
    for elem, frac in element_fractions.items():
        key_value_pairs[f"{elem}_fraction"] = float(frac)

    # Build data (complex, non-searchable fields)
    data = {
        "PTM_structural_analysis_in_percent": ptm_analysis,
        "stress_tensor_in_GPa": stress_dict,
        "structural_stability_assessment": {
            "target_structure": target_structure,
            "dominant_structure": dominant_structure,
            "match_percentage": float(structural_match_percent),
            "is_stable": bool(is_structurally_stable),
            "threshold": STRUCTURAL_STABILITY_THRESHOLD,
            "full_ptm_analysis": ptm_analysis
        },
        "elements": elements,
        "target_fractions": optimized_fractions,
    }

    # === UPDATE: Add analysis results to the early-saved structure ===
    db.update_key_value_pairs(new_uuid, key_value_pairs)
    db.update_data(new_uuid, data)

    # Render visualizations
    render_msg = cl.Message(content="Rendering visualizations...")
    await render_msg.send()

    try:
        image_path = render_atoms(atoms, new_uuid, db=db)
        image = cl.Image(path=image_path, name="recomputed structure elements", display="inline")
        await cl.Message(
            content=f"Recomputed structure (element coloring):",
            elements=[image],
        ).send()
    except Exception as e:
        await cl.Message(content=f"Warning: Could not render element visualization: {e}").send()

    try:
        image_path = render_structure(atoms, new_uuid, db=db)
        image = cl.Image(path=image_path, name="recomputed structure analysis", display="inline")
        await cl.Message(
            content=f"Structural analysis (PTM coloring):",
            elements=[image],
        ).send()
    except Exception as e:
        await cl.Message(content=f"Warning: Could not render structure visualization: {e}").send()

    await render_msg.remove()

    # Complete tasks
    task4.status = cl.TaskStatus.DONE
    task_list.status = "Done!"
    await task_list.send()
    await message.remove()

    # Get source energy for comparison
    source_energy = source_metadata.get("potential_energy_eV_per_atom", 0.0)
    source_formation = source_metadata.get("formation_energy_ground_state_reference_eV_per_atom", 0.0)
    source_density = source_metadata.get("density_g_per_cm3", 0.0)
    source_structural_match = source_metadata.get("structural_match_percent", 0.0)

    # Build stability interpretation
    if is_structurally_stable:
        stability_interpretation = f"STABLE: {structural_match_percent:.1f}% {target_structure.upper()} (threshold: {STRUCTURAL_STABILITY_THRESHOLD:.0f}%)"
    else:
        stability_interpretation = f"UNSTABLE: Only {structural_match_percent:.1f}% {target_structure.upper()} (threshold: {STRUCTURAL_STABILITY_THRESHOLD:.0f}%). Dominant: {dominant_structure.upper()}"

    # Record calculation in session state (memory layer)
    session_state = cl.user_session.get("session_state")  # type: ignore
    if session_state:
        session_state.record_calculation(
            calculator=new_calculator,
            supercell_size=num_atoms,
            fmax=fmax
        )

    return {
        "new_structure_id": new_id,  # Integer ID (simple, model-friendly)
        "new_structure_uuid": new_uuid,  # UUID (32-character hex string, globally unique)
        "source_structure_uuid": source_uuid,
        "comparison": {
            "source_calculator": source_calculator,
            "new_calculator": new_calculator,
            "energy_difference_eV_per_atom": float(energy_per_atom - source_energy),
            "formation_energy_difference_eV_per_atom": float(formation_energy - source_formation),
            "density_difference_g_per_cm3": float(density - source_density),
            "structural_match_source_percent": float(source_structural_match),
            "structural_match_new_percent": float(structural_match_percent),
        },
        "new_structure": {
            "composition": composition_string,
            "formation_energy_eV_per_atom": float(formation_energy),
            "mixing_energy_eV_per_atom": float(mixing_energy),
            "max_force_eV_per_A": float(max_force_magnitude),
            "converged": max_force_magnitude <= fmax,
            "density_g_per_cm3": float(density),
            "lattice_constant_A": float(lattice_constant),
            "num_atoms": int(num_atoms)
        },
        "structural_analysis": {
            "dominant_structure": dominant_structure,
            "structure_fractions": ptm_analysis
        },
        "stability": {
            "structural_match_percent": float(structural_match_percent),
            "is_structurally_stable": bool(is_structurally_stable),
            "target_structure": target_structure,
            "interpretation": stability_interpretation
        }
    }
