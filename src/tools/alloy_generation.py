"""
Alloy Generation Tool

Generates and relaxes alloy supercells with SQS optimization.
"""

from typing import Annotated, List, Dict, Literal, Optional
import chainlit as cl
import numpy as np
import time
import torch
import logging
from ase.build import bulk

logger = logging.getLogger(__name__)

# Utilities
from src.utils.composition_parser import parse_composition_string

# Core modules
from src.core.structure_builder import (
    compute_replication_factors,
    estimate_alloy_lattice_constant,
    estimate_alloy_lattice_constant_vegard,
    lattice_constant_from_atomic_volume,
)
from src.core.optimization import StructureOptimizer
from src.core.calculator_service import get_calculator_service
from src.core.sqs import SQSGenerator
from src.core.analysis import structural_analysis, compute_density, compute_coordination_rdf, reorder_partial_rdf
from src.core.formation_energy import formation_energy_per_atom
from src.core.reference_data import precompute_and_save, load_reference_energies
from src.storage.cache import get_reference_cache

# Visualization modules
from src.visualization.ovito_renderer import render_atoms, render_structure
from src.visualization.plotly_charts import plot_coordination_rdf

# Storage modules
from src.storage.database import create_structure_database

# Session state for memory layer
from src.utils.session_state import SessionState


@cl.step(type="tool")  # type: ignore
async def generate_alloy_supercell(
    structure: Annotated[Literal['sc', 'bcc', 'fcc', 'hcp', 'diamond'], "Lattice structure"],
    elements: Annotated[Optional[List[str]], "Element symbols (or use composition_string)"] = None,
    target_fractions: Annotated[Optional[List[float]], "Atomic fractions, sum=1 (or use composition_string)"] = None,
    composition_string: Annotated[str, "e.g., 'Ag75Cu25' or 'Cu-Ag' (takes precedence)"] = "",
    sqs_iterations: Annotated[int, "SQS iterations (0=disabled)"] = 1000000,
    lattice_constant: Annotated[float, "Lattice constant Å (0=auto-estimate)"] = 0.0,
    fmax: Annotated[float, "Force convergence eV/Å (default 0.001)"] = 0.001,
) -> Annotated[Dict, "Structure UUID, composition, quality metrics, structural analysis, and physical properties."]:

    # Get default supercell size from session (set by user in settings panel)
    target_num_atoms = cl.user_session.get("default_supercell_size", 48)  # type: ignore

    # Parse composition_string if provided (takes precedence over elements/target_fractions)
    if composition_string:
        parsed = parse_composition_string(composition_string)
        if parsed and parsed.is_valid():
            elements = parsed.elements
            target_fractions = parsed.fractions
            logger.info(f"Parsed composition_string '{composition_string}' -> elements={elements}, fractions={target_fractions}")
        else:
            raise ValueError(
                f"Could not parse composition_string '{composition_string}'. "
                f"Supported formats: 'Ag75Cu25', 'Ag75-Cu25', 'Ag-Cu' (equal parts), 'Ag3Cu1' (3:1 ratio)"
            )

    # Validate that we have required parameters after potential parsing
    if elements is None or target_fractions is None:
        raise ValueError(
            "Provide either (elements + target_fractions) OR composition_string. "
            "Example: elements=['Ag', 'Cu'], target_fractions=[0.75, 0.25] OR composition_string='Ag75Cu25'"
        )

    # Create the TaskList
    task_list = cl.TaskList()
    task_list.status = "Running..."

     # Create tasks and put them in the ready state
    task1 = cl.Task(title="Optimizing alloy supercell with respect to desired structure, composition, and total number of atoms", status=cl.TaskStatus.READY)
    await task_list.add_task(task1)
    task2 = cl.Task(title="Optimizing site occupancy to achieve quasirandom alloy characteristics", status=cl.TaskStatus.READY)
    await task_list.add_task(task2)
    task3a = cl.Task(title="Stage 1: Coarse relaxation (CUDA, fmax=0.01)", status=cl.TaskStatus.READY)
    await task_list.add_task(task3a)
    task3b = cl.Task(title="Stage 2: Fine relaxation (CPU, fmax=0.001)", status=cl.TaskStatus.READY)
    await task_list.add_task(task3b)
    task4 = cl.Task(title="Analyzing and visualizing the results.", status=cl.TaskStatus.READY)
    await task_list.add_task(task4)


    # Optional: link a message to each task to allow task navigation in the chat history
        # Form composition string to add to the message
    composition = [f"{f*100:.1f}% {e}" for e, f in zip(elements, target_fractions)]
    composition = ", ".join(composition)
    message = await cl.Message(content=f"Calling the tool to generate {structure} alloy supercell composed of {composition}").send()
    task1.forId = message.id
    task2.forId = message.id
    task3a.forId = message.id
    task3b.forId = message.id
    task4.forId = message.id

    # Update the task list in the interface
    await task_list.send()


    # Use the current time in seconds as the seed.
    np.random.seed(int(time.time()))

    # Update the task statuses
    task1.status = cl.TaskStatus.RUNNING
    # Update the task list in the interface
    await task_list.send()


    # Check input for errors
    valid_structures = {"sc", "bcc", "fcc", "hcp", "diamond"}
    if structure not in valid_structures:
        raise ValueError("Structure must be one of: " + ", ".join(valid_structures))

    if len(elements) != len(target_fractions):
        raise ValueError("The length of elements and fractions must be equal.")

    for elem, frac in zip(elements, target_fractions):
        if frac <= 0 or frac > 1:
            raise ValueError(f"Atomic fraction for {elem} must be between 0 and 1. Got {frac}")

    total_fraction = sum(target_fractions)
    if not np.isclose(total_fraction, 1.0, rtol=1e-3):
        raise ValueError(f"The sum of the atomic fractions must equal 1. Got sum = {total_fraction}")


    # Optimize replication factors and atom counts based on provided structure, target fractions and target number of atoms
    replication_factors, counts = compute_replication_factors(structure, target_fractions, target_num_atoms)

    # Calculate optimized fractions
    optimized_fractions = np.array(counts) / np.sum(counts)

    # Get calculator from session for Vegard's law
    calculator_name_for_vegard = cl.user_session.get("default_calculator") or "orb-v3-direct-20-omat"  # type: ignore

    # Estimate lattice constant of alloy if not set.
    if np.isclose(lattice_constant, 0.0, rtol=1e-3):
        # Try Vegard's law first (more accurate, uses relaxed data)
        try:
            lattice_constant = estimate_alloy_lattice_constant_vegard(
                structure,
                elements,
                optimized_fractions,
                calculator=calculator_name_for_vegard
            )
            lattice_method = "Vegard's law"
            print(f"✓ Lattice constant estimated using Vegard's law: {lattice_constant:.4f} Å")
        except (ValueError, FileNotFoundError, KeyError) as e:
            # Fall back to radii-based estimate
            print(f"⚠️  Vegard's law failed ({type(e).__name__}), falling back to radii-based estimate")
            try:
                lattice_constant = estimate_alloy_lattice_constant(structure,elements,optimized_fractions)
                lattice_method = "Hard-sphere radii"
                print(f"✓ Lattice constant estimated using radii: {lattice_constant:.4f} Å")
            except Exception as e2:
                print(f"Error estimating alloy lattice constant: {e2}")
                raise

    # For many structures, a cubic unit cell is preferred.
    cubic = structure in ["sc", "fcc", "bcc", "diamond"]
    # Create the base (primitive) structure using the first element.
    base_structure = bulk(elements[0], structure, a=lattice_constant, cubic=cubic)

    # Replicate base structure to make a supercell
    supercell = base_structure * replication_factors

    # Sanity check
    if len(supercell) != np.sum(counts):
        raise ValueError("Mismatch between supercell atom count and computed counts")

    # Total number of atoms in the supercell.
    num_atoms = len(supercell)


    # Update the task statuses
    task1.status = cl.TaskStatus.DONE
    # Update the task statuses
    task2.status = cl.TaskStatus.RUNNING
    # Update the task list in the interface
    await task_list.send()

    # Generate SQS or random structure using refactored SQSGenerator
    composition_dict = dict(zip(elements, counts))
    sqs_gen = SQSGenerator(iterations=sqs_iterations)

    if sqs_iterations > 0 and len(elements) > 1:
        supercell, objective = sqs_gen.generate(supercell, composition_dict)
        if objective is not None:
            print(f"Optimized objective is {objective}")
    else:
        supercell = sqs_gen.generate_random(supercell, composition_dict)


    # Update the task statuses
    task2.status = cl.TaskStatus.DONE
    # Update the task list in the interface
    await task_list.send()

    # Get calculator from session
    calculator_name_for_relaxation = cl.user_session.get("default_calculator") or "orb-v3-direct-20-omat"  # type: ignore

    # Stage 1: Coarse relaxation with CUDA (fmax=0.01)
    task3a.status = cl.TaskStatus.RUNNING
    await task_list.send()

    # Load calculator (supports ORB/MACE/NequIP via calculator_service)
    service = get_calculator_service()
    calc_cuda = await service.get_calculator_sync(calculator_name_for_relaxation, 'cuda')
    optimizer_cuda = StructureOptimizer(calc_cuda)
    supercell = await cl.make_async(optimizer_cuda.relax)(supercell, fmax=0.01, optimizer='FIRE', max_steps=500)

    task3a.status = cl.TaskStatus.DONE
    await task_list.send()

    # Stage 2: Fine relaxation with CPU (fmax from parameter, default 0.001)
    task3b.status = cl.TaskStatus.RUNNING
    await task_list.send()

    calc_cpu = await service.get_calculator_sync(calculator_name_for_relaxation, 'cpu')
    optimizer_cpu = StructureOptimizer(calc_cpu)
    supercell = await cl.make_async(optimizer_cpu.relax)(supercell, fmax=fmax, optimizer='FIRE', max_steps=500)

    task3b.status = cl.TaskStatus.DONE
    await task_list.send()

    atoms = supercell

    # === EARLY SAVE: Capture relaxed structure immediately after relaxation ===
    # This ensures the structure is never lost if UI disconnects during analysis
    db = create_structure_database()
    num_atoms = len(atoms)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    composition_string = "".join(f"{e}{f*100:.1f}" for e, f in zip(elements, optimized_fractions))

    # Get calculator from session (set during on_chat_start or on_settings_update)
    calculator_name = cl.user_session.get("default_calculator") or "orb-v3-direct-20-omat"  # type: ignore

    result = db.write(atoms, key_value_pairs={
        # Provenance
        "derived_from": "scratch",
        # Calculator info
        "calculator_name": calculator_name,
        "device_type": device_type,
        # Structure
        "target_structure": structure,
        "composition_string": composition_string,
        "optimized_num_atoms": num_atoms,
        # Status marker - will be updated to "complete" after analysis
        "status": "analyzing",
    })
    structure_id = result["id"]
    supercell_uuid = result["uuid"]

    # Update the task statuses
    task4.status = cl.TaskStatus.RUNNING
    # Update the task list in the interface
    await task_list.send()


    # REDUCE NUMBER OF DIGITS AFTER DOT
    # Calculate energy per atom (uses cached results from relaxation).
    energy_per_atom = atoms.get_potential_energy() / num_atoms

    # Get calculator from session to load correct reference data
    calculator_name_for_ref = cl.user_session.get("default_calculator") or "orb-v3-direct-20-omat"  # type: ignore

    # Get cache for correct calculator and ensure reference data exists
    cache = get_reference_cache(calculator=calculator_name_for_ref)
    if not cache.is_available():
        # Trigger precomputation if reference data not available
        await cl.Message(
            content=(
                f"⚠️ **Reference data not available for {calculator_name_for_ref}.**\n\n"
                f"Starting precomputation now (this will take a few hours)..."
            )
        ).send()
        await cl.make_async(precompute_and_save)(
            hydrostatic_cell_relaxation=True,
            optimizer="FIRE",
            fmax=0.005,
            calculator=calculator_name_for_ref,
            cache=cache,
        )

    # Get calculator-specific reference data paths
    _, energy_json_path = cache.get_paths()

    # Calculate ground-state reference formation energy
    refs = load_reference_energies(energy_json_path, reference_mode="ground_state")
    formation_energy = formation_energy_per_atom(elements=elements, fractions=optimized_fractions, energies_ref=refs, E_per_atom=energy_per_atom)
    # Calculate mixing energy for the same structure
    refs_structure = load_reference_energies(energy_json_path, reference_mode="same_structure", structure=structure)
    mixing_energy = formation_energy_per_atom(elements=elements, fractions=optimized_fractions, energies_ref=refs_structure, E_per_atom=energy_per_atom)

     # Compute density
    density = compute_density(atoms)
    #Compute lattice constant from atomic volume assuming the desired structure
    volume_per_atom = atoms.get_volume() / num_atoms
    lattice_constant = lattice_constant_from_atomic_volume(structure,volume_per_atom)
    # Calculate forces and derive mean and maximum force magnitudes.
    forces: List[List[float]] = atoms.get_forces()
    force_magnitudes = np.linalg.norm(forces, axis=1)

    #Compute maximum atomic force magnitude to check convergence
    max_force_magnitude = float(round(float(np.max(force_magnitudes)),3))
    # Evaluate stress tensor; assuming atoms.get_stress() returns a 6-element array.
    stress_dict = {
        k: round(float(v) * 16.0218, 3)
        for k, v in zip(["xx", "yy", "zz", "yz", "xz", "xy"], atoms.get_stress())
    }

    # All 118 elements of the periodic table (117 ORB-supported, Og unsupported)
    # Database schema now tracks all elements for future-proof generalization
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

    # Build key_value_pairs for UPDATE (analysis results)
    # Note: Basic metadata (provenance, calculator, composition) already saved in early save
    key_value_pairs = {
        # Mark analysis as complete
        "status": "complete",

        # Structure
        "lattice_constant": float(lattice_constant),

        # Composition (CRITICAL: per-element fractions for search)
        "num_elements": len(elements),

        # Energies (searchable)
        "potential_energy_eV_per_atom": float(energy_per_atom),
        "formation_energy_ground_state_reference_eV_per_atom": float(formation_energy),
        "mixing_energy_same_structure_reference_eV_per_atom": float(mixing_energy),

        # Physical properties (searchable)
        "density_g_per_cm3": float(density),
        "max_force_magnitude_eV_per_A": float(max_force_magnitude),
    }

    # Initialize all 118 element fractions to 0.0 (all elements tracked in database schema)
    for elem in SUPPORTED_ELEMENTS:
        key_value_pairs[f"{elem}_fraction"] = 0.0

    # Override with actual element fractions
    for elem, frac in zip(elements, optimized_fractions):
        key_value_pairs[f"{elem}_fraction"] = float(frac)

    # Perform PTM structural analysis
    ptm_analysis = structural_analysis(atoms)
    dominant_structure = max(ptm_analysis, key=ptm_analysis.get) if ptm_analysis else "unknown"

    # Calculate structural stability (match between target and analyzed structure)
    STRUCTURAL_STABILITY_THRESHOLD = 90.0
    structural_match_percent = ptm_analysis.get(structure, 0.0) if ptm_analysis else 0.0
    is_structurally_stable = structural_match_percent >= STRUCTURAL_STABILITY_THRESHOLD

    # Add structural stability to searchable metadata
    key_value_pairs["structural_match_percent"] = float(structural_match_percent)
    key_value_pairs["is_structurally_stable"] = bool(is_structurally_stable)

    # Compute RDF (Radial Distribution Function)
    try:
        rdf_result = compute_coordination_rdf(atoms, cutoff=10.0)
        # Reorder partial RDFs to match user's element order (e.g., Cu-Cu, Cu-Ag, Ag-Ag)
        partial_ordered = reorder_partial_rdf(rdf_result.get('partial', {}), elements)
        rdf_data = {
            'r_values': rdf_result['r'].tolist(),
            'g_r_values': rdf_result['g_total'].tolist(),
            'partial': {k: v.tolist() for k, v in partial_ordered.items()}
        }
        logger.info(f"RDF computed successfully: {len(rdf_data['r_values'])} points, "
                   f"{len(rdf_data['partial'])} partial RDFs")
    except Exception as e:
        logger.warning(f"RDF computation failed: {e}")
        rdf_data = None

    # Build data (complex, non-searchable fields)
    data = {
        # Complex analysis results
        "PTM_structural_analysis_in_percent": ptm_analysis,
        "stress_tensor_in_GPa": stress_dict,
        "rdf_data": rdf_data,  # RDF (Radial Distribution Function)

        # Structural stability assessment (detailed)
        "structural_stability_assessment": {
            "target_structure": structure,
            "dominant_structure": dominant_structure,
            "match_percentage": float(structural_match_percent),
            "is_stable": bool(is_structurally_stable),
            "threshold": STRUCTURAL_STABILITY_THRESHOLD,
            "full_ptm_analysis": ptm_analysis
        },

        # Reference data (for provenance)
        "target_fractions": target_fractions,
        "target_num_atoms": target_num_atoms,
        "elements": elements,
    }

    # === UPDATE: Add analysis results to the early-saved structure ===
    db.update_key_value_pairs(supercell_uuid, key_value_pairs)
    db.update_data(supercell_uuid, data)

    # Form composition string to add to the caption of the figure
    composition = [f"{f*100:.1f}% {e}" for e, f in zip(elements, optimized_fractions)]
    composition = ", ".join(composition)

    # Render element-colored visualization
    # Note: OVITO rendering must be done synchronously (not thread-safe)
    render_msg = cl.Message(content="🎨 Rendering visualizations...")
    await render_msg.send()

    try:
        image_path = render_atoms(atoms, supercell_uuid, db=db)
        image = cl.Image(path=image_path, name="generated alloy composition", display="inline")
        await cl.Message(
            content=f"Here is the final optimized {structure} alloy supercell composed of {composition}.",
            elements=[image],
        ).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Warning: Could not render element visualization: {e}").send()

    # Render structure-colored visualization
    try:
        image_path = render_structure(atoms, supercell_uuid, db=db)
        image = cl.Image(path=image_path, name="generated alloy structure", display="inline")
        await cl.Message(
            content=f"Here is the alloy supercell after structural analysis.",
            elements=[image],
        ).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Warning: Could not render structure visualization: {e}").send()

    await render_msg.remove()

    # Display RDF chart (after OVITO images)
    try:
        rdf_fig = plot_coordination_rdf(atoms, cutoff=10.0)
        rdf_elements = [cl.Plotly(name="RDF Analysis", figure=rdf_fig, display="inline")]
        await cl.Message(content="**Radial Distribution Function:**\nRDF Analysis", elements=rdf_elements).send()
    except Exception as e:
        print(f"Warning: Could not display RDF chart: {e}")

    # Update the task statuses
    task4.status = cl.TaskStatus.DONE
    # Update the task list in the interface
    await task_list.send()

    task_list.status = "Done!"
    await task_list.send()

    await message.remove()

    # Display both ID and UUID prominently for user reference (helps with follow-up queries)
    # Note: Use generic descriptions to avoid triggering tool execution by quantized models
    await cl.Message(
        content=(
            f"📋 **Structure Reference**\n\n"
            f"**ID:** `{structure_id}`\n"
            f"**UUID:** `{supercell_uuid}`\n\n"
            f"Use either ID or UUID for follow-up queries such as:\n"
            f"• Elastic properties calculation\n"
            f"• Structure report generation\n"
            f"• Temperature-dependent property analysis"
        )
    ).send()

    # Build stability interpretation message
    if is_structurally_stable:
        stability_interpretation = f"STABLE: {structural_match_percent:.1f}% {structure.upper()} (threshold: {STRUCTURAL_STABILITY_THRESHOLD:.0f}%)"
    else:
        stability_interpretation = f"UNSTABLE: Only {structural_match_percent:.1f}% {structure.upper()} (threshold: {STRUCTURAL_STABILITY_THRESHOLD:.0f}%). Dominant structure: {dominant_structure.upper()} ({ptm_analysis.get(dominant_structure, 0):.1f}%)"

    # Record calculation in session state (memory layer)
    session_state = cl.user_session.get("session_state")  # type: ignore
    if session_state:
        session_state.record_calculation(
            calculator=calculator_name,
            supercell_size=num_atoms,
            fmax=fmax
        )

    return {
        "structure_id": structure_id,  # Integer ID (simple, model-friendly)
        "supercell_uuid": supercell_uuid,  # UUID (32-character hex string, globally unique)
        "actual_num_atoms": int(num_atoms),  # Actual atom count (may differ from target due to cell constraints)
        "target_num_atoms": int(target_num_atoms),  # Requested target for reference
        "composition": composition_string,
        "structure_quality": {
            "formation_energy_eV_per_atom": float(formation_energy),
            "mixing_energy_eV_per_atom": float(mixing_energy),
            "max_force_eV_per_A": float(max_force_magnitude),
            "converged": max_force_magnitude <= fmax
        },
        "structural_analysis": {
            "dominant_structure": dominant_structure,
            "structure_fractions": ptm_analysis
        },
        "stability": {
            "structural_match_percent": float(structural_match_percent),
            "is_structurally_stable": bool(is_structurally_stable),
            "target_structure": structure,
            "interpretation": stability_interpretation
        },
        "properties": {
            "density_g_per_cm3": float(density),
            "lattice_constant_A": float(lattice_constant)
        }
    }
