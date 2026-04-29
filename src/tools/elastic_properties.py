"""
Elastic Properties Tool

Calculate comprehensive elastic properties for an existing database structure.

Computes:
- Elastic stiffness tensor (Voigt form, 6x6 matrix in GPa)
- Elastic moduli (bulk, shear, Young's modulus, Poisson's ratio)
- ELATE anisotropy analysis (universal anisotropy index, ductility, directional properties)

The structure is subjected to small strain deformations and atomic positions are relaxed
to minimize energy at each strain state (relaxed/Born tensor). The calculator and device
settings are automatically retrieved from the database entry for consistency.

Results are saved to the database and visualized with heatmaps and charts.
"""

from typing import Annotated, Dict, List, Optional, Any, Union
import chainlit as cl
import numpy as np
from datetime import datetime

# Core modules
from src.core.calculator_service import get_calculator_service
from src.core.elasticity import compute_elastic_stiffness_tensor, compute_elastic_moduli
from src.core.elate_analysis import ElasticAnisotropyAnalyzer
from src.core.elasticity_validation import validate_elastic_tensor, validate_elate_results
from src.core.cancellation import ComputationCancelledException

# Visualization modules
from src.visualization.plotly_charts import plot_stiffness_tensor_heatmap
from src.visualization.elate_plots import (
    create_anisotropy_tables_separated,
    plot_directional_property_2d_projections,
    plot_directional_property_3d
)

# Storage modules
from src.storage.database import create_structure_database

# Session state for memory layer
from src.utils.session_state import SessionState


@cl.step(type="tool")  # type: ignore
async def calculate_elastic_properties(
    structure_ref: Annotated[Union[int, str], "Structure ID (local) or UUID (global)"],
    epsilon: Annotated[float, "Strain magnitude (default 1%, min 1%)"] = 1e-2
) -> Annotated[Dict, "Elastic moduli, stiffness tensor, and anisotropy analysis."]:
    """Calculate elastic stiffness tensor, moduli (K, G, E, ν), and ELATE anisotropy analysis."""
    from datetime import datetime

    # Create task list for progress tracking
    task_list = cl.TaskList()
    task_list.status = "Running..."

    task1 = cl.Task(title="Loading structure and validating relaxation", status=cl.TaskStatus.READY)
    task2 = cl.Task(title="Computing elastic stiffness tensor (180 deformations)", status=cl.TaskStatus.READY)
    task3 = cl.Task(title="Computing elastic moduli", status=cl.TaskStatus.READY)
    task4 = cl.Task(title="Saving results to database", status=cl.TaskStatus.READY)
    task5 = cl.Task(title="Generating heatmap visualization", status=cl.TaskStatus.READY)

    await task_list.add_task(task1)
    await task_list.add_task(task2)
    await task_list.add_task(task3)
    await task_list.add_task(task4)
    await task_list.add_task(task5)

    db = create_structure_database()

    # Display reference for user-friendly output (preserve original input format)
    ref_display = str(structure_ref) if isinstance(structure_ref, int) else structure_ref[:8]

    # Resolve int→UUID once at tool layer (UUID is used for all subsequent calls)
    try:
        structure_uuid = db.resolve_to_uuid(structure_ref)
    except KeyError:
        await cl.Message(content=f"❌ Structure {ref_display}... not found in database.").send()
        task1.status = cl.TaskStatus.FAILED
        await task_list.send()
        return {"error": "Structure not found"}

    message = await cl.Message(content=f"Calculating elastic stiffness tensor for structure {ref_display}...").send()
    task1.forId = message.id
    task2.forId = message.id
    task3.forId = message.id
    task4.forId = message.id
    task5.forId = message.id

    await task_list.send()

    # Task 1: Load and validate structure
    task1.status = cl.TaskStatus.RUNNING
    await task_list.send()

    try:
        atoms = db.read(structure_uuid)
        row = db._get_db().get(unique_id=structure_uuid)
    except Exception as e:
        await cl.Message(content=f"❌ Error loading structure: {e}").send()
        task1.status = cl.TaskStatus.FAILED
        await task_list.send()
        return {"error": str(e)}

    # Validate relaxation quality
    kvp = row.key_value_pairs
    max_force = kvp.get('max_force_magnitude_eV_per_A', 999.0)

    if max_force > 0.005:
        await cl.Message(content=f"⚠️ Warning: Structure has residual forces ({max_force:.4f} eV/Å). Recommend regenerating structure with fmax ≤ 0.005 eV/Å for accurate elastic constants.").send()

    # Read calculator settings from database
    calculator_name = kvp.get('calculator_name', 'orb-v3-direct-20-omat')
    device_type = kvp.get('device_type', 'cpu')
    # Sanitize legacy "cuda+cpu" strings from batch scripts
    if '+' in device_type:
        device_type = device_type.split('+')[0]

    await cl.Message(content=f"Using calculator: **{calculator_name}** (device: {device_type})").send()

    # Load calculator (supports ORB/MACE/NequIP via calculator_service)
    try:
        service = get_calculator_service()
        calc = await service.get_calculator_sync(calculator_name, device_type)
        atoms.calc = calc
    except Exception as e:
        await cl.Message(content=f"❌ Failed to load calculator: {e}").send()
        task1.status = cl.TaskStatus.FAILED
        await task_list.send()
        return {"error": f"Calculator loading failed: {e}"}

    task1.status = cl.TaskStatus.DONE
    await task_list.send()

    # Task 2: Compute stiffness tensor
    task2.status = cl.TaskStatus.RUNNING
    await task_list.send()

    computation_msg = cl.Message(content="⏳ Computing elastic stiffness tensor (this may take several minutes)...")
    await computation_msg.send()

    # Get computation cancellation event from session
    computation_event = cl.user_session.get("computation_cancellation_event")  # type: ignore

    # Create progress callback to update task2 in real-time
    def progress_callback(current: int, total: int, status: str):
        """Update task2 title with current progress (thread-safe)."""
        # Note: Direct task updates from threads may not be safe
        # Progress is primarily for console logging
        # UI updates happen via the exception handler
        print(f"Progress: {status}", flush=True)

    try:
        # Compute tensor using async wrapper to avoid UI freezing
        C_voigt = await cl.make_async(compute_elastic_stiffness_tensor)(
            atoms,
            epsilon=epsilon,
            relax_kwargs={
                'fmax': 0.005,  # Standard convergence for elastic constants
                'optimizer': 'FIRE',
                'max_steps': 100,  # Limited steps for deformed structures (already near equilibrium)
                'hydrostatic_strain': False  # CRITICAL: Keep cell fixed, relax atoms only
            },
            cancellation_event=computation_event,
            progress_callback=progress_callback
        )
    except ComputationCancelledException as e:
        # User cancelled the computation
        await computation_msg.remove()
        await cl.Message(
            content=(
                f"⏹️ **Elastic tensor calculation stopped by user.**\n\n"
                f"**Progress**: {e.completed}/{e.total} deformations completed ({e.progress:.1%})\n\n"
                f"Partial results were not saved. To resume, please restart the calculation."
            )
        ).send()
        task2.status = cl.TaskStatus.FAILED
        task_list.status = "Cancelled"
        await task_list.send()
        return {
            "cancelled": True,
            "completed_deformations": e.completed,
            "total_deformations": e.total,
            "progress_percent": e.progress * 100,
            "message": str(e)
        }
    except Exception as e:
        await cl.Message(content=f"❌ Stiffness tensor calculation failed: {e}").send()
        task2.status = cl.TaskStatus.FAILED
        await task_list.send()
        return {"error": str(e)}

    await computation_msg.remove()

    task2.status = cl.TaskStatus.DONE
    await task_list.send()

    # Task 2.5: Validate elastic tensor
    from src.core.elasticity_validation import (
        validate_elastic_tensor,
        diagnose_invalid_tensor,
        format_validation_message
    )

    validation_result = validate_elastic_tensor(C_voigt)

    if not validation_result.is_valid:
        # Tensor is invalid - diagnose and warn user
        diagnostics = diagnose_invalid_tensor(
            C_voigt,
            atoms=atoms,
            epsilon=epsilon,
            max_force=max_force
        )

        # Format and display validation message
        validation_msg = format_validation_message(validation_result, diagnostics)
        await cl.Message(content=validation_msg).send()

        # Still save tensor to database (with warning flag) but skip ELATE
        task3.status = cl.TaskStatus.RUNNING
        await task_list.send()

        try:
            moduli = compute_elastic_moduli(C_voigt)
            K = moduli['bulk_modulus_GPa']
            G = moduli['shear_modulus_GPa']
            E = moduli['youngs_modulus_GPa']
            nu = moduli['poisson_ratio']
        except Exception as e:
            await cl.Message(content=f"⚠️ Warning: Could not compute derived moduli: {e}").send()
            K = G = E = nu = 0.0

        task3.status = cl.TaskStatus.DONE
        await task_list.send()

        # Skip ELATE analysis
        elate_properties = None

        # Save to database with validation warning
        task4.status = cl.TaskStatus.RUNNING
        await task_list.send()

        # Extract elastic stability from Born criterion
        is_elastically_stable = validation_result.is_positive_definite
        eigenvalues_list = validation_result.eigenvalues.tolist()

        try:
            data_update = {
                "elastic_stiffness_tensor_voigt_GPa": C_voigt.tolist(),
                "elastic_calculation_epsilon": float(epsilon),
                "bulk_modulus_GPa": float(K),
                "shear_modulus_GPa": float(G),
                "youngs_modulus_GPa": float(E),
                "poisson_ratio": float(nu),
                "elastic_calculation_timestamp": datetime.now().isoformat(),
                "elastic_calculator_used": calculator_name,
                "elastic_tensor_validation_failed": True,
                "elastic_tensor_eigenvalues": eigenvalues_list,
                "elastic_tensor_validation_errors": validation_result.errors,
                # Elastic stability assessment
                "elastic_stability_assessment": {
                    "born_criterion_satisfied": is_elastically_stable,
                    "is_stable": is_elastically_stable,
                    "eigenvalues": eigenvalues_list,
                    "condition_number": float(validation_result.condition_number) if validation_result.condition_number is not None else None
                }
            }

            db.update_data(structure_uuid, data_update)

            task4.status = cl.TaskStatus.DONE
            await task_list.send()
        except Exception as e:
            await cl.Message(content=f"❌ Failed to save results: {e}").send()
            task4.status = cl.TaskStatus.FAILED
            await task_list.send()
            return {"error": str(e)}

        # Task 5: Skip visualization for invalid tensor
        task5.status = cl.TaskStatus.DONE
        task_list.status = "Complete (with warnings)"
        await task_list.send()

        # Build elastic stability interpretation
        stability_interpretation = f"UNSTABLE: Born criterion VIOLATED (negative eigenvalues detected)"

        # Record calculation in session state (memory layer)
        session_state = cl.user_session.get("session_state")  # type: ignore
        if session_state:
            session_state.record_calculation(calculator=calculator_name)

        return {
            "structure_uuid": structure_uuid,
            "validation_failed": True,
            "eigenvalues": eigenvalues_list,
            "errors": validation_result.errors,
            "elastic_moduli": {
                "bulk_modulus_GPa": K,
                "shear_modulus_GPa": G,
                "youngs_modulus_GPa": E,
                "poisson_ratio": nu
            },
            "elastic_stability": {
                "born_criterion_satisfied": is_elastically_stable,
                "is_elastically_stable": is_elastically_stable,
                "eigenvalues": eigenvalues_list,
                "interpretation": stability_interpretation
            },
            "stiffness_tensor_GPa": C_voigt.tolist(),
            "calculator_used": calculator_name,
            "calculation_metadata": {
                "epsilon": epsilon,
                "num_deformations": 180,
                "timestamp": datetime.now().isoformat()
            }
        }

    # Tensor is valid - display success message
    validation_msg = format_validation_message(validation_result)
    await cl.Message(content=validation_msg).send()

    # Task 3: Compute derived moduli
    task3.status = cl.TaskStatus.RUNNING
    await task_list.send()

    try:
        moduli = compute_elastic_moduli(C_voigt)
        K = moduli['bulk_modulus_GPa']
        G = moduli['shear_modulus_GPa']
        E = moduli['youngs_modulus_GPa']
        nu = moduli['poisson_ratio']
    except Exception as e:
        await cl.Message(content=f"⚠️ Warning: Could not compute derived moduli: {e}").send()
        K = G = E = nu = 0.0

    task3.status = cl.TaskStatus.DONE
    await task_list.send()

    # Display structure summary with elastic properties (for immediate context)
    summary_md = f"""
**Structure {ref_display}...: {kvp.get('composition_string', 'N/A')}**

**Elastic Properties**

**Bulk modulus**: {K:.1f} GPa
**Shear modulus**: {G:.1f} GPa
**Young's modulus**: {E:.1f} GPa
**Poisson's ratio**: {nu:.3f}

**Strain magnitude (ε)**: {epsilon:.0e}
**Calculator**: {calculator_name}
**Number of deformations**: 180
"""

    await cl.Message(content=summary_md).send()

    # Task 3.5: ELATE anisotropy analysis (optional - requires density)
    elate_properties = None
    density = kvp.get('density_g_per_cm3')

    if density is not None and density > 0:
        import numpy as np
        from src.core.elate_analysis import compute_elate_properties

        task3_5 = cl.Task(title="Computing comprehensive anisotropy analysis", status=cl.TaskStatus.READY)
        await task_list.add_task(task3_5)
        task3_5.forId = message.id
        task3_5.status = cl.TaskStatus.RUNNING
        await task_list.send()

        try:
            # Compute ELATE properties (requires density for wave speeds)
            elate_properties = await cl.make_async(compute_elate_properties)(C_voigt, density)

            # Validate ELATE results (detect silent failures)
            from src.core.elasticity_validation import validate_elate_results
            elate_valid, elate_error = validate_elate_results(elate_properties)

            if not elate_valid:
                # ELATE produced invalid results
                await cl.Message(
                    content=f"⚠️ **Warning**: ELATE analysis produced invalid results.\n\n"
                            f"**Issue**: {elate_error}\n\n"
                            f"Elastic tensor is valid but ELATE calculation failed. "
                            f"Skipping anisotropy analysis."
                ).send()
                elate_properties = None
                task3_5.status = cl.TaskStatus.FAILED
                await task_list.send()
            else:
                # Display quick anisotropy summary
                AU = elate_properties['universal_anisotropy_index']
                if np.isinf(AU):
                    isotropy = "Highly Anisotropic (∞)"
                elif AU < 0.1:
                    isotropy = "Nearly Isotropic"
                elif AU < 1.0:
                    isotropy = "Weakly Anisotropic"
                elif AU < 5.0:
                    isotropy = "Moderately Anisotropic"
                else:
                    isotropy = "Highly Anisotropic"

                pugh = elate_properties['pugh_ratio_hill']
                ductility = "Ductile" if pugh > 1.75 else "Brittle"

                anisotropy_md = f"""
**Anisotropy Analysis**

**Universal Anisotropy Index**: {AU:.3f} ({isotropy})
**Pugh Ratio (K/G)**: {pugh:.2f} ({ductility})
**Auxetic behavior**: {'⚠️ Yes (negative Poisson)' if elate_properties['has_auxetic_behavior'] else 'No'}
"""
                await cl.Message(content=anisotropy_md).send()

                task3_5.status = cl.TaskStatus.DONE
        except Exception as e:
            await cl.Message(content=f"⚠️ Warning: Anisotropy analysis failed: {e}").send()
            task3_5.status = cl.TaskStatus.FAILED

        await task_list.send()
    else:
        await cl.Message(content="ℹ️ Skipping anisotropy analysis (density not available in database).").send()

    # Task 4: Save to database
    task4.status = cl.TaskStatus.RUNNING
    await task_list.send()

    # Extract elastic stability from Born criterion (valid tensor)
    is_elastically_stable_valid = validation_result.is_positive_definite
    eigenvalues_list_valid = validation_result.eigenvalues.tolist()

    try:
        data_update = {
            "elastic_stiffness_tensor_voigt_GPa": C_voigt.tolist(),
            "elastic_calculation_epsilon": float(epsilon),
            "bulk_modulus_GPa": float(K),
            "shear_modulus_GPa": float(G),
            "youngs_modulus_GPa": float(E),
            "poisson_ratio": float(nu),
            "elastic_calculation_timestamp": datetime.now().isoformat(),
            "elastic_calculator_used": calculator_name,
            # Elastic stability assessment
            "elastic_stability_assessment": {
                "born_criterion_satisfied": is_elastically_stable_valid,
                "is_stable": is_elastically_stable_valid,
                "eigenvalues": eigenvalues_list_valid,
                "condition_number": float(validation_result.condition_number) if validation_result.condition_number is not None else None
            }
        }

        # Add ELATE properties if computed
        if elate_properties is not None:
            data_update["elate_properties"] = elate_properties

        db.update_data(structure_uuid, data_update)
    except Exception as e:
        await cl.Message(content=f"⚠️ Warning: Could not save results to database: {e}").send()

    task4.status = cl.TaskStatus.DONE
    await task_list.send()

    # Task 5: Visualize stiffness tensor heatmap
    task5.status = cl.TaskStatus.RUNNING
    await task_list.send()

    # Generate heatmap
    try:
        fig = plot_stiffness_tensor_heatmap(C_voigt)
        elements_list = [cl.Plotly(name="stiffness_tensor", figure=fig, display="inline", size="large")]
        await cl.Message(content="**Elastic Stiffness Tensor:**", elements=elements_list).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Could not generate heatmap: {e}").send()

    task5.status = cl.TaskStatus.DONE
    task_list.status = "Done!"
    await task_list.send()

    await message.remove()

    # Build elastic stability interpretation
    stability_interpretation_valid = f"STABLE: Born criterion satisfied (all eigenvalues positive)"

    # Build comprehensive return value for agent
    result = {
        "structure_uuid": structure_uuid,
        "elastic_moduli": {
            "bulk_modulus_GPa": float(K),
            "shear_modulus_GPa": float(G),
            "youngs_modulus_GPa": float(E),
            "poisson_ratio": float(nu)
        },
        "elastic_stability": {
            "born_criterion_satisfied": is_elastically_stable_valid,
            "is_elastically_stable": is_elastically_stable_valid,
            "eigenvalues": eigenvalues_list_valid,
            "interpretation": stability_interpretation_valid
        },
        "stiffness_tensor_GPa": C_voigt.tolist(),
        "calculator_used": calculator_name,
        "calculation_metadata": {
            "epsilon": float(epsilon),
            "num_deformations": 180,
            "timestamp": data_update["elastic_calculation_timestamp"]
        }
    }

    # Add anisotropy analysis if computed
    if elate_properties is not None:
        import numpy as np
        AU = elate_properties['universal_anisotropy_index']

        # Classify anisotropy
        if np.isinf(AU):
            classification = "Highly Anisotropic (∞)"
        elif AU < 0.1:
            classification = "Nearly Isotropic"
        elif AU < 1.0:
            classification = "Weakly Anisotropic"
        elif AU < 5.0:
            classification = "Moderately Anisotropic"
        else:
            classification = "Highly Anisotropic"

        pugh = elate_properties['pugh_ratio_hill']

        # Extract directional ranges from ELATE
        youngs_min = elate_properties.get('min_youngs_modulus_GPa', 0)
        youngs_max = elate_properties.get('max_youngs_modulus_GPa', 0)
        poisson_min = elate_properties.get('min_poisson_ratio', 0)
        poisson_max = elate_properties.get('max_poisson_ratio', 0)

        result["anisotropy"] = {
            "universal_anisotropy_index": float(AU),
            "classification": classification,
            "pugh_ratio": float(pugh),
            "ductility": "Ductile" if pugh > 1.75 else "Brittle",
            "has_auxetic_behavior": elate_properties['has_auxetic_behavior'],
            "youngs_range_GPa": [float(youngs_min), float(youngs_max)],
            "poisson_range": [float(poisson_min), float(poisson_max)]
        }

    # Record calculation in session state (memory layer)
    session_state = cl.user_session.get("session_state")  # type: ignore
    if session_state:
        session_state.record_calculation(calculator=calculator_name)

    return result
