#!/usr/bin/env python3
"""
Anharmonic properties tool for computing temperature-dependent properties.
"""

import chainlit as cl
import threading
import json
from typing import Annotated, Dict, List, Optional, Union
from pathlib import Path

from src.storage.database import create_structure_database
from src.core.calculator_service import get_calculator_service
from src.core.cancellation import ComputationCancelledException
from src.core.qha_wrapper import (
    compute_qha_properties,
    compute_thermal_conductivity
)
from src.visualization.qha_plots import (
    plot_qha_properties_individual,
    plot_thermal_conductivity,
    plot_qha_with_thermal_conductivity,
    plot_qha_volume_surfaces
)


def load_experimental_qha_data() -> Dict:
    """
    Load experimental QHA reference data at 300 K.

    Returns:
        Dictionary with experimental data for common elements
    """
    reference_file = Path(__file__).parent.parent.parent / "data" / "reference" / "qha_experimental_300K.json"

    if not reference_file.exists():
        return {}

    with open(reference_file, 'r') as f:
        data = json.load(f)

    return data.get('elements', {})


def compare_with_experimental(
    composition: str,
    structure_type: str,
    bulk_modulus: float,
    thermal_expansion: float,
    heat_capacity: float
) -> Optional[str]:
    """
    Compare computed QHA properties with experimental values.

    Args:
        composition: Chemical formula (e.g., "Cu", "Al")
        structure_type: Crystal structure (e.g., "fcc", "bcc")
        bulk_modulus: Computed bulk modulus at 300 K (GPa)
        thermal_expansion: Computed thermal expansion at 300 K (1/K)
        heat_capacity: Computed heat capacity at 300 K [J/(K mol)]

    Returns:
        Formatted comparison string if experimental data available, None otherwise
    """
    # Load experimental data
    exp_data = load_experimental_qha_data()

    # Check if we have data for this element (only works for pure elements)
    if composition not in exp_data:
        return None

    exp = exp_data[composition]

    # Check if structure matches (optional check, just for info)
    structure_match = exp['structure'].lower() == structure_type.lower()
    structure_note = "" if structure_match else f" (exp: {exp['structure']})"

    # Calculate percentage differences
    def pct_diff(computed, experimental):
        if experimental == 0:
            return 0.0
        return ((computed - experimental) / experimental) * 100

    # Convert thermal expansion to same units (10^-6 K^-1)
    thermal_expansion_scaled = thermal_expansion * 1e6
    exp_alpha = exp['thermal_expansion_1e6_per_K']

    # Calculate differences
    B_diff = pct_diff(bulk_modulus, exp['bulk_modulus_GPa'])
    alpha_diff = pct_diff(thermal_expansion_scaled, exp_alpha)
    Cp_diff = pct_diff(heat_capacity, exp['heat_capacity_J_K_mol'])

    # Format comparison message
    comparison = f"\n\n📊 **Comparison with Experimental Data** (300 K{structure_note}):\n"
    comparison += f"  Property                   | Computed   | Experimental | Difference\n"
    comparison += f"  ---------------------------|------------|--------------|------------\n"
    comparison += f"  Bulk modulus (GPa)         | {bulk_modulus:6.1f}     | {exp['bulk_modulus_GPa']:6.1f}       | {B_diff:+6.1f}%\n"
    comparison += f"  Thermal expansion (10⁻⁶)   | {thermal_expansion_scaled:6.2f}     | {exp_alpha:6.2f}       | {alpha_diff:+6.1f}%\n"
    comparison += f"  Heat capacity (J/(K mol))  | {heat_capacity:6.2f}     | {exp['heat_capacity_J_K_mol']:6.2f}       | {Cp_diff:+6.1f}%\n"

    # Add sources
    if 'sources' in exp:
        comparison += f"\n  📚 Sources: {', '.join(exp['sources'][:2])}"

    return comparison

@cl.step(type="tool")  # type: ignore
async def compute_anharmonic_properties(
    structure_ref: Annotated[Union[int, str], "Structure ID (local) or UUID (global)"],
    num_volumes: Annotated[int, "Volume points for QHA (default 11)"] = 11,
    compute_thermal_conductivity_flag: Annotated[bool, "Enable κ(T) calculation (expensive)"] = False,
    mesh_qha: Annotated[Optional[List[int]], "QHA phonon mesh [nx,ny,nz] (default [20,20,20])"] = None,
    mesh_phono3py: Annotated[Optional[List[int]], "Thermal conductivity mesh [nx,ny,nz] (default [20,20,20])"] = None,
    temperature_range: Annotated[Optional[List[float]], "[Tmin,Tmax,Tstep] in K (default [0,610,10])"] = None
) -> Annotated[Dict, "QHA properties, optional κ(T), and plots."]:
    """
    Compute temperature-dependent properties (B(T), V(T), α(T), Cp(T), γ(T)) via QHA; optionally κ(T) via phono3py.

    Note:
    Volume grid spans ±10% strain around equilibrium volume (90% to 110% of V0).
    This wide range ensures accurate EOS fitting and robust thermal property predictions.

    Returns:
    Dictionary containing:
    - qha_properties: QHA results (always computed)
    - thermal_conductivity: κ(T) results (only if requested)
    - plots: Interactive Plotly visualizations (displayed in UI)
    - metadata: Composition, structure, calculator info
    """
    # Initialize database
    db = create_structure_database()

    # Display reference for user-friendly output (preserve original input format)
    ref_display = str(structure_ref) if isinstance(structure_ref, int) else structure_ref[:8]

    # Resolve int→UUID once at tool layer (UUID is used for all subsequent calls)
    try:
        structure_uuid = db.resolve_to_uuid(structure_ref)
    except KeyError:
        await cl.Message(content=f"❌ Structure {ref_display}... not found in database.").send()
        return {"error": "Structure not found"}

    # Default parameters
    if mesh_qha is None:
        mesh_qha = [20, 20, 20]
    if mesh_phono3py is None:
        mesh_phono3py = [20, 20, 20]
    if temperature_range is None:
        temperature_range = [0, 610, 10]

    t_min, t_max, t_step = temperature_range

    # Create cancellation event for cooperative cancellation
    cancellation_event = threading.Event()

    # Store cancellation event for cleanup
    cl.user_session.set("current_cancellation_event", cancellation_event)

    # Create task list
    task_list = cl.TaskList()
    task_list.status = "Computing Finite Temperature Properties"

    # Task 1: Load structure
    load_task = cl.Task(
        title="Load structure from database",
        status=cl.TaskStatus.READY
    )
    await task_list.add_task(load_task)

    # Task 2: Structure relaxation
    relax_task = cl.Task(
        title="Relax structure (conservative model, fmax=0.001)",
        status=cl.TaskStatus.READY
    )
    await task_list.add_task(relax_task)

    # Task 3: Finite temperature calculation
    qha_task = cl.Task(
        title=f"Compute finite temperature properties ({num_volumes} volumes)",
        status=cl.TaskStatus.READY
    )
    await task_list.add_task(qha_task)

    # Task 3: Thermal conductivity (optional)
    kappa_task = None
    if compute_thermal_conductivity_flag:
        kappa_task = cl.Task(
            title="Compute thermal conductivity (phono3py) - This will take hours!",
            status=cl.TaskStatus.READY
        )
        await task_list.add_task(kappa_task)

    # Task 4: Visualization
    viz_task = cl.Task(
        title="Create plots",
        status=cl.TaskStatus.READY
    )
    await task_list.add_task(viz_task)

    # Task 5: Update database
    db_task = cl.Task(
        title="Update database",
        status=cl.TaskStatus.READY
    )
    await task_list.add_task(db_task)

    await task_list.send()

    try:
        # ============================================================
        # Phase 1: Load structure
        # ============================================================
        load_task.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # Load structure from database
        atoms = db.read(structure_uuid)
        metadata = db.get_metadata(structure_uuid)

        composition = metadata.get("composition_string", "Unknown")
        structure_type = metadata.get("target_structure", "unknown")
        calculator_name = metadata.get("calculator_name", "orb-v3-direct-20-omat")

        await cl.Message(
            content=f"✓ Loaded structure: {composition} ({structure_type})\n"
                    f"  - ID/UUID: {structure_uuid}\n"
                    f"  - Atoms: {len(atoms)}\n"
                    f"  - Calculator: {calculator_name}"
        ).send()

        # Warn about memory constraints for medium/large cells
        num_atoms = len(atoms)
        if num_atoms > 100:
            await cl.Message(
                content=f"⚠️ **Memory Warning**: QHA calculations are recommended for small cells (48 atoms). "
                        f"This structure has **{num_atoms} atoms** and may trigger memory constraints or take significantly longer. "
                        f"Consider using a smaller supercell for QHA analysis."
            ).send()

        load_task.status = cl.TaskStatus.DONE
        await task_list.send()

        # ============================================================
        # Phase 2: Structure relaxation + QHA computation
        # ============================================================
        # Relaxation happens inside compute_qha_properties, but we track it separately
        relax_task.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # Get calculator from Chainlit UI settings
        model_name = cl.user_session.get("default_calculator")

        # Display actual temperature range (t_max is set to 1010 to include 1000 K with step=10)
        t_max_display = t_max - t_step if t_step > 0 else t_max

        await cl.Message(
            content=f"Computing finite temperature properties...\n"
                    f"  - Number of volumes: {num_volumes} (±10% strain)\n"
                    f"  - Mesh: {mesh_qha}\n"
                    f"  - Temperature: {t_min} - {t_max_display} K (step: {t_step} K)\n\n"
                    f"⏰ This will take some time..."
        ).send()

        # Load calculators (supports ORB/MACE/NequIP via calculator_service)
        # GPU calculator for phonon force evaluations, CPU calculator for relaxation
        service = get_calculator_service()
        gpu_calc = await service.get_calculator_sync(model_name, 'cuda')
        cpu_calc = await service.get_calculator_sync(model_name, 'cpu')

        # Define callbacks to update task status
        # We need to bridge sync callbacks to async task list updates
        import asyncio
        loop = asyncio.get_event_loop()

        def relaxation_callback(v0_initial: float, v0_relaxed: float):
            """Update task list when relaxation completes."""
            relax_task.status = cl.TaskStatus.DONE
            relax_task.title = f"Relax structure (V: {v0_initial:.1f} → {v0_relaxed:.1f} Å³)"
            qha_task.status = cl.TaskStatus.RUNNING
            # Schedule task list update in the event loop
            asyncio.run_coroutine_threadsafe(task_list.send(), loop)

        def qha_progress_callback(completed: int, total: int, message: str):
            """Update QHA task with current progress."""
            qha_task.title = f"Compute finite temperature properties ({completed}/{total} volumes completed)"
            # Schedule task list update in the event loop
            # This bridges the sync callback (running in thread pool) to async UI updates
            asyncio.run_coroutine_threadsafe(task_list.send(), loop)

        # Compute QHA
        try:
            qha_data = await cl.make_async(compute_qha_properties)(
                atoms=atoms,
                calculator=gpu_calc,
                cpu_calculator=cpu_calc,
                model_name=model_name,
                num_volumes=num_volumes,
                strain_range=0.10,  # ±10%
                mesh=tuple(mesh_qha),
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                cancellation_event=cancellation_event,
                relaxation_callback=relaxation_callback,
                progress_callback=qha_progress_callback
            )
        except MemoryError as e:
            await cl.Message(
                content=(
                    f"❌ **Out of Memory** — QHA calculation for {len(atoms)} atoms requires more RAM "
                    f"than available on this device.\n\n"
                    f"**This structure is too large for QHA on this hardware.** "
                    f"Please use a smaller supercell (48 atoms recommended for QHA).\n\n"
                    f"Do NOT retry with coarser settings — the results would be unreliable."
                )
            ).send()
            return {"error": "Out of memory — structure too large for QHA on this device"}
        except Exception as e:
            error_str = str(e)
            # Catch numpy/memory allocation errors
            if "Unable to allocate" in error_str or "MemoryError" in error_str:
                await cl.Message(
                    content=(
                        f"❌ **Out of Memory** — QHA calculation for {len(atoms)} atoms requires more RAM "
                        f"than available ({error_str.split('Unable to allocate')[1].split('for')[0].strip() if 'Unable to allocate' in error_str else 'too much memory'}).\n\n"
                        f"**This structure is too large for QHA on this hardware.** "
                        f"Please use a smaller supercell (48 atoms recommended for QHA).\n\n"
                        f"Do NOT retry with coarser settings — the results would be unreliable."
                    )
                ).send()
                return {"error": "Out of memory — structure too large for QHA on this device"}
            # Log full traceback to file for debugging
            import traceback
            with open('/tmp/qha_error.log', 'w') as f:
                f.write(f"QHA Error: {str(e)}\n\n")
                f.write(traceback.format_exc())
            raise

        # Extract key values at 300 K (index 30 if step=10K)
        idx_300K = int(300 / t_step) if t_step > 0 else 30

        # Validate QHA data before accessing
        if qha_data.get('bulk_modulus') is None:
            raise RuntimeError("QHA computation failed: bulk_modulus is None")
        if qha_data.get('volume') is None:
            raise RuntimeError("QHA computation failed: volume is None")
        if qha_data.get('thermal_expansion') is None:
            raise RuntimeError("QHA computation failed: thermal_expansion is None")
        if qha_data.get('heat_capacity_p') is None:
            raise RuntimeError("QHA computation failed: heat_capacity_p is None")

        # Ensure idx_300K is within bounds (QHA may drop some temperature points)
        max_idx = min(len(qha_data['gibbs_free_energy']), len(qha_data['bulk_modulus']),
                      len(qha_data['volume']), len(qha_data['thermal_expansion']),
                      len(qha_data['heat_capacity_p']))

        if idx_300K >= max_idx:
            # Use the last available index if 300K is out of bounds
            idx_300K = max_idx - 1
            actual_temp = qha_data['temperatures'][idx_300K] if idx_300K < len(qha_data['temperatures']) else (t_min + idx_300K * t_step)
            temp_note = f" (using T={actual_temp:.0f}K, closest available)"
        else:
            temp_note = ""

        # Build success message
        success_msg = f"✓ Finite temperature properties computed successfully!\n\n"

        # Check dynamical stability
        is_stable = qha_data.get('dynamically_stable', True)
        has_imaginary = qha_data.get('has_imaginary_modes', False)
        min_freq = qha_data.get('min_frequency_THz', None)

        if is_stable:
            success_msg += f"**Dynamical Stability:** ✓ Stable (no imaginary phonon modes)\n\n"
        else:
            success_msg += f"**Dynamical Stability:** ⚠️ Unstable (imaginary modes detected, min freq: {min_freq:.3f} THz)\n\n"

        success_msg += f"Results at 300 K{temp_note}:\n"
        success_msg += f"  - Gibbs free energy: {qha_data['gibbs_free_energy'][idx_300K]:.3f} kJ/mol\n"
        success_msg += f"  - Bulk modulus: {qha_data['bulk_modulus'][idx_300K]:.2f} GPa\n"
        success_msg += f"  - Volume: {qha_data['volume'][idx_300K]:.3f} Å³ (primitive cell)\n"
        success_msg += f"  - Thermal expansion: {qha_data['thermal_expansion'][idx_300K]*1e6:.2f} × 10⁻⁶ K⁻¹\n"
        success_msg += f"  - Heat capacity Cp: {qha_data['heat_capacity_p'][idx_300K]:.2f} J/(K mol)"

        # Add experimental comparison if available
        comparison = compare_with_experimental(
            composition=composition,
            structure_type=structure_type,
            bulk_modulus=qha_data['bulk_modulus'][idx_300K],
            thermal_expansion=qha_data['thermal_expansion'][idx_300K],
            heat_capacity=qha_data['heat_capacity_p'][idx_300K]
        )

        if comparison:
            success_msg += comparison

        await cl.Message(content=success_msg).send()

        qha_task.status = cl.TaskStatus.DONE
        await task_list.send()

        # ============================================================
        # Phase 3: Compute thermal conductivity (optional)
        # ============================================================
        kappa_data = None

        if compute_thermal_conductivity_flag:
            kappa_task.status = cl.TaskStatus.RUNNING
            await task_list.send()

            await cl.Message(
                content=f"⚠️ WARNING: Computing thermal conductivity with phono3py\n"
                        f"  - This requires 3rd-order force constants\n"
                        f"  - Expect 100s of force calculations\n"
                        f"  - This may take hours to days!\n"
                        f"  - Mesh: {mesh_phono3py}\n\n"
                        f"⏰ Starting computation..."
            ).send()

            try:
                # Define progress callback for thermal conductivity
                def kappa_progress_callback(completed: int, total: int, message: str):
                    """Update thermal conductivity task with current progress."""
                    kappa_task.title = f"Compute thermal conductivity ({completed}/{total} displacements computed)"
                    # Schedule task list update in the event loop
                    asyncio.run_coroutine_threadsafe(task_list.send(), loop)

                kappa_data = await cl.make_async(compute_thermal_conductivity)(
                    atoms=atoms,
                    calculator=calc,
                    mesh=tuple(mesh_phono3py),
                    t_min=t_min,
                    t_max=t_max,
                    t_step=t_step,
                    structure_dir=db.get_structure_directory(structure_uuid),
                    cancellation_event=cancellation_event,
                    progress_callback=kappa_progress_callback
                )

                # Bounds check for thermal conductivity at 300K
                kappa_max_idx = len(kappa_data['kappa_iso'])
                kappa_idx_300K = min(idx_300K, kappa_max_idx - 1)

                kappa_temp_note = ""
                if kappa_idx_300K != idx_300K:
                    actual_temp = kappa_data['temperatures'][kappa_idx_300K] if kappa_idx_300K < len(kappa_data['temperatures']) else (t_min + kappa_idx_300K * t_step)
                    kappa_temp_note = f" (using T={actual_temp:.0f}K, closest available)"

                await cl.Message(
                    content=f"✓ Thermal conductivity computed!\n\n"
                            f"κ at 300 K{kappa_temp_note}: {kappa_data['kappa_iso'][kappa_idx_300K]:.2f} W/(m K)"
                ).send()

                kappa_task.status = cl.TaskStatus.DONE
                await task_list.send()

            except ImportError as e:
                await cl.Message(
                    content=f"✗ Thermal conductivity calculation failed:\n"
                            f"  phono3py is not installed\n"
                            f"  Install with: pip install phono3py"
                ).send()

                kappa_task.status = cl.TaskStatus.FAILED
                await task_list.send()

            except Exception as e:
                await cl.Message(
                    content=f"✗ Thermal conductivity calculation failed:\n  {str(e)}"
                ).send()

                kappa_task.status = cl.TaskStatus.FAILED
                await task_list.send()

        # ============================================================
        # Phase 4: Create visualizations
        # ============================================================
        viz_task.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # Create output directories for plots
        base_dir = db.get_structure_directory(structure_uuid)
        qha_dir = base_dir / "qha"
        qha_dir.mkdir(parents=True, exist_ok=True)

        elements = []

        # Create 4 individual plots for finite temperature properties
        qha_figs = await cl.make_async(plot_qha_properties_individual)(
            qha_data,
            composition=composition,
            structure_type=structure_type
        )

        # Plot names for the 4 individual plots
        plot_names = [
            "Gibbs Free Energy G(T)",
            "Bulk Modulus B(T)",
            "Thermal Expansion α(T)",
            "Heat Capacity Cp(T)"
        ]

        # Add each plot to elements (no file saving - use export_structure_data for that)
        for fig, name in zip(qha_figs, plot_names):
            elements.append(
                cl.Plotly(
                    name=name,
                    figure=fig,
                    display="page"
                )
            )

        # 3D surface plots for F(T,V), S(T,V), Cv(T,V)
        if 'entropy_volume' in qha_data and 'cv_volume' in qha_data:
            surface_figs = await cl.make_async(plot_qha_volume_surfaces)(
                qha_data,
                composition=composition,
                structure_type=structure_type
            )

            # Add to elements (no file saving - use export_structure_data for that)
            surface_names = ['F(T,V) Surface', 'S(T,V) Surface', 'Cv(T,V) Surface']

            for fig, name in zip(surface_figs, surface_names):
                elements.append(
                    cl.Plotly(
                        name=name,
                        figure=fig,
                        display="page"
                    )
                )

        # Thermal conductivity plot (if available)
        if kappa_data is not None:
            kappa_dir = base_dir / "thermal_conductivity"
            kappa_dir.mkdir(parents=True, exist_ok=True)

            kappa_fig = await cl.make_async(plot_thermal_conductivity)(
                kappa_data,
                composition=composition,
                structure_type=structure_type
            )

            elements.append(
                cl.Plotly(
                    name="Thermal Conductivity",
                    figure=kappa_fig,
                    display="page"
                )
            )

            # Combined plot with QHA and thermal conductivity
            combined_fig = await cl.make_async(plot_qha_with_thermal_conductivity)(
                qha_data,
                kappa_data,
                composition=composition,
                structure_type=structure_type
            )

            elements.append(
                cl.Plotly(
                    name="All Thermal Properties",
                    figure=combined_fig,
                    display="page"
                )
            )

        await cl.Message(
            content=f"✓ Plots created:\n" +
                    "\n".join([elem.name for elem in elements]),
            elements=elements
        ).send()

        viz_task.status = cl.TaskStatus.DONE
        await task_list.send()

        # ============================================================
        # Phase 5: Update database with results
        # ============================================================
        db_task.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # Update metadata with QHA results
        updated_metadata = metadata.copy()

        # Recalculate idx_300K with bounds checking for metadata storage
        idx_300K_safe = int(300 / t_step) if t_step > 0 else 30
        max_idx = min(len(qha_data['gibbs_free_energy']), len(qha_data['bulk_modulus']),
                      len(qha_data['volume']), len(qha_data['thermal_expansion']),
                      len(qha_data['heat_capacity_p']), len(qha_data['gruneisen']))
        idx_300K_safe = min(idx_300K_safe, max_idx - 1)

        # Store QHA values at 300 K (or closest available)
        updated_metadata.update({
            "qha_gibbs_free_energy_300K_kJ_mol": float(qha_data['gibbs_free_energy'][idx_300K_safe]),
            "qha_bulk_modulus_300K_GPa": float(qha_data['bulk_modulus'][idx_300K_safe]),
            "qha_volume_300K_angstrom3": float(qha_data['volume'][idx_300K_safe]),
            "qha_thermal_expansion_300K_per_K": float(qha_data['thermal_expansion'][idx_300K_safe]),
            "qha_heat_capacity_p_300K_J_K_mol": float(qha_data['heat_capacity_p'][idx_300K_safe]),
            "qha_gruneisen_300K": float(qha_data['gruneisen'][idx_300K_safe]),
            "qha_num_volumes": num_volumes,
            "qha_mesh": mesh_qha,
            "qha_temperature_range": temperature_range,
            # Dynamical stability info
            "qha_dynamically_stable": qha_data.get('dynamically_stable', True),
            "qha_has_imaginary_modes": qha_data.get('has_imaginary_modes', False),
            "qha_min_frequency_THz": qha_data.get('min_frequency_THz', None)
        })

        # Store full QHA data arrays in database for later export via export_structure_data
        qha_1d_properties = {
            "temperatures": [float(t) for t in qha_data['temperatures']],
            "gibbs_energies": [float(g) for g in qha_data['gibbs_free_energy']],
            "bulk_moduli": [float(b) for b in qha_data['bulk_modulus']],
            "volumes": [float(v) for v in qha_data['volume']],
            "thermal_expansion": [float(a) for a in qha_data['thermal_expansion']],
            "heat_capacities": [float(c) for c in qha_data['heat_capacity_p']],
            "gruneisen_params": [float(g) for g in qha_data['gruneisen']],
        }
        updated_metadata["qha_1d_properties"] = qha_1d_properties

        # Store 2D QHA data (F, S, Cv vs T and V)
        if 'entropy_volume' in qha_data and 'cv_volume' in qha_data:
            qha_2d_properties = {
                "temperatures": [float(t) for t in qha_data['temperatures']],
                "volumes": [float(v) for v in qha_data['volumes_used']],
                "helmholtz": [[float(f) for f in row] for row in qha_data['helmholtz_volume']],
                "entropy": [[float(s) for s in row] for row in qha_data['entropy_volume']],
                "heat_capacity_cv": [[float(c) for c in row] for row in qha_data['cv_volume']],
            }
            updated_metadata["qha_2d_properties"] = qha_2d_properties

        # Store thermal conductivity if computed
        if kappa_data is not None:
            kappa_max_idx = len(kappa_data['kappa_iso'])
            kappa_idx_300K_safe = min(idx_300K_safe, kappa_max_idx - 1)

            updated_metadata.update({
                "thermal_conductivity_300K_W_m_K": float(kappa_data['kappa_iso'][kappa_idx_300K_safe]),
                "phono3py_mesh": mesh_phono3py
            })

            # Store full thermal conductivity data for later export
            updated_metadata["thermal_conductivity"] = {
                "temperatures": [float(t) for t in kappa_data['temperatures']],
                "kappa_xx": [float(k) for k in kappa_data['kappa_xx']],
                "kappa_yy": [float(k) for k in kappa_data['kappa_yy']],
                "kappa_zz": [float(k) for k in kappa_data['kappa_zz']],
                "kappa_iso": [float(k) for k in kappa_data['kappa_iso']],
            }

        # Update database
        db.update_data(structure_uuid, updated_metadata)

        # Store searchable QHA fields in key_value_pairs
        qha_kvp = {
            "has_qha_data": True,
            "qha_gibbs_free_energy_300K_kJ_mol": float(qha_data['gibbs_free_energy'][idx_300K_safe]),
            "qha_bulk_modulus_300K_GPa": float(qha_data['bulk_modulus'][idx_300K_safe]),
            "qha_thermal_expansion_300K_1e6_per_K": float(qha_data['thermal_expansion'][idx_300K_safe] * 1e6),
            "qha_heat_capacity_p_300K_J_K_mol": float(qha_data['heat_capacity_p'][idx_300K_safe]),
            "qha_gruneisen_300K": float(qha_data['gruneisen'][idx_300K_safe]),
            # Dynamical stability (searchable)
            "qha_dynamically_stable": qha_data.get('dynamically_stable', True),
        }

        if kappa_data is not None:
            qha_kvp["has_thermal_conductivity"] = True
            qha_kvp["thermal_conductivity_300K_W_m_K"] = float(kappa_data['kappa_iso'][kappa_idx_300K_safe])
        else:
            qha_kvp["has_thermal_conductivity"] = False

        db.update_key_value_pairs(structure_uuid, qha_kvp)

        await cl.Message(
            content=f"✓ Database updated with anharmonic properties"
        ).send()

        db_task.status = cl.TaskStatus.DONE
        await task_list.send()

        # ============================================================
        # CONVERT LARGE ARRAYS TO SUMMARIES (avoid token limit)
        # ============================================================

        # Convert QHA data to summary
        qha_summary = None
        if qha_data is not None:
            import numpy as np
            temps = qha_data['temperatures']
            idx_300 = np.argmin(np.abs(temps - 300))

            qha_summary = {
                'temperature_range_K': [float(temps.min()), float(temps.max())],
                'num_points': int(len(temps)),
                'bulk_modulus_at_300K_GPa': float(qha_data['bulk_modulus'][idx_300]),
                'volume_at_300K_angstrom3': float(qha_data['volume'][idx_300]),
                'thermal_expansion_at_300K_per_K': float(qha_data['thermal_expansion'][idx_300]),
                'heat_capacity_p_at_300K_J_K_mol': float(qha_data['heat_capacity_p'][idx_300]),
                'gruneisen_at_300K': float(qha_data['gruneisen'][idx_300]),
                'gibbs_free_energy_at_300K_kJ_mol': float(qha_data['gibbs_free_energy'][idx_300]),
                'num_volumes_computed': int(len(qha_data['volumes_used'])),
                'mesh_used': mesh_qha,
                # Dynamical stability
                'dynamically_stable': qha_data.get('dynamically_stable', True),
                'has_imaginary_modes': qha_data.get('has_imaginary_modes', False),
                'min_frequency_THz': qha_data.get('min_frequency_THz', None)
            }

        # Convert thermal conductivity data to summary
        kappa_summary = None
        if kappa_data is not None:
            import numpy as np
            temps_kappa = kappa_data['temperatures']
            idx_300_kappa = np.argmin(np.abs(temps_kappa - 300))

            kappa_summary = {
                'temperature_range_K': [float(temps_kappa.min()), float(temps_kappa.max())],
                'num_points': int(len(temps_kappa)),
                'kappa_xx_at_300K_W_m_K': float(kappa_data['kappa_xx'][idx_300_kappa]),
                'kappa_yy_at_300K_W_m_K': float(kappa_data['kappa_yy'][idx_300_kappa]),
                'kappa_zz_at_300K_W_m_K': float(kappa_data['kappa_zz'][idx_300_kappa]),
                'kappa_iso_at_300K_W_m_K': float(kappa_data['kappa_iso'][idx_300_kappa]),
                'mesh_used': mesh_phono3py
            }

        # ============================================================
        # Return results
        # ============================================================
        task_list.status = "Finite Temperature Properties Complete!"
        await task_list.send()

        results = {
            "qha_properties": qha_summary,
            "thermal_conductivity": kappa_summary,
            # Note: 'plots' removed from return value since they're already sent to UI via cl.Message
            # This prevents the agent from redundantly describing visualizations
            # Note: CSV export removed - use export_structure_data tool for on-demand export
            "metadata": {
                "composition": composition,
                "structure_type": structure_type,
                "calculator_name": model_name,
                "structure_uuid": structure_uuid
            }
        }

        return results

    except Exception as e:
        # Mark all pending tasks as failed
        for task in [load_task, relax_task, qha_task, viz_task, db_task]:
            if task and task.status == cl.TaskStatus.RUNNING:
                task.status = cl.TaskStatus.FAILED

        if kappa_task and kappa_task.status == cl.TaskStatus.RUNNING:
            kappa_task.status = cl.TaskStatus.FAILED

        task_list.status = "Failed"
        await task_list.send()

        await cl.Message(
            content=f"✗ Anharmonic properties calculation failed:\n{str(e)}"
        ).send()

        raise

    finally:
        # Clean up cancellation event
        cl.user_session.set("current_cancellation_event", None)
