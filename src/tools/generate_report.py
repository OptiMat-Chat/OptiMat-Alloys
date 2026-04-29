"""
Combined Report Generation Tool

This tool combines the functionality of display_structure_report and export_structure_data:
1. Displays interactive visual report in chat (Plotly charts)
2. Generates PDF report with all figures (including elastic property visualizations)
3. Exports CSV data files
4. Exports structure files (CIF, POSCAR, LAMMPS, XYZ)
5. Provides download links at the end
"""

import csv
import zipfile
from pathlib import Path
from typing import Annotated, Dict, Any, List, Optional, Union
import logging

import chainlit as cl
import numpy as np

# Core modules
from src.core.elate_analysis import ElasticAnisotropyAnalyzer
from src.core.elasticity_validation import validate_elate_results
from src.core.structure_export import (
    export_all_structure_formats,
    export_elastic_stiffness_csv,
    export_qha_properties_csv,
    export_2d_property_csv,
    export_thermal_conductivity_csv,
    export_rdf_csv,
    export_elate_2d_csv,
    export_elate_3d_csv,
)
from src.core.report_generator import generate_structure_report
from src.core.references import generate_bibtex_file, get_all_references_for_calculation

# Visualization modules
from src.visualization.ovito_renderer import render_atoms, render_structure
from src.visualization.plotly_charts import (
    plot_structural_analysis,
    plot_coordination_rdf,
    plot_stiffness_tensor_heatmap
)
from src.visualization.elate_plots import (
    create_anisotropy_tables_separated,
    plot_directional_property_2d_projections,
    plot_directional_property_3d
)
from src.visualization.qha_plots import (
    plot_qha_properties_individual,
    plot_qha_volume_surfaces
)

# Storage modules
from src.storage.database import create_structure_database

logger = logging.getLogger(__name__)


def load_qha_data_from_csv(qha_dir: Path) -> dict:
    """Load QHA data from CSV files for plotting."""
    qha_data = {}

    # Load 1D properties
    csv_path = qha_dir / "qha_properties.csv"
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            data = list(reader)
            qha_data['temperatures'] = np.array([float(row[0]) for row in data])
            qha_data['gibbs_free_energy'] = np.array([float(row[1]) for row in data])
            qha_data['bulk_modulus'] = np.array([float(row[2]) for row in data])
            qha_data['volume'] = np.array([float(row[3]) for row in data])
            qha_data['thermal_expansion'] = np.array([float(row[4]) for row in data])
            qha_data['heat_capacity_p'] = np.array([float(row[5]) for row in data])
            qha_data['gruneisen'] = np.array([float(row[6]) for row in data])

    # Load 2D volume-dependent properties
    volume_files = [
        ('helmholtz_free_energy_vs_T_V.csv', 'helmholtz_volume'),
        ('entropy_vs_T_V.csv', 'entropy_volume'),
        ('heat_capacity_cv_vs_T_V.csv', 'cv_volume'),
    ]

    for filename, key in volume_files:
        csv_path = qha_dir / filename
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                next(reader)  # Skip units comment row

                volumes = []
                for col in header[1:]:
                    vol_str = col.replace('V=', '').replace(' Å³', '').strip()
                    volumes.append(float(vol_str))
                qha_data['volumes_used'] = np.array(volumes)

                data = []
                for row in reader:
                    data.append([float(x) for x in row[1:]])
                qha_data[key] = np.array(data)

    return qha_data


def save_plotly_figure_as_png(fig, output_path: Path, width: int = 800, height: int = 800) -> Optional[Path]:
    """Convert a Plotly figure to PNG using kaleido.

    Args:
        fig: Plotly figure object
        output_path: Path to save the PNG
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Path to saved PNG, or None if failed
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path), width=width, height=height, scale=2)
        return output_path
    except Exception as e:
        logger.warning(f"Failed to save Plotly figure as PNG: {e}")
        return None


def create_data_zip(structure_dir: Path, results: dict, composition_string: str) -> Optional[Path]:
    """Create a ZIP file containing all data files (excluding PDF report).

    Args:
        structure_dir: Directory containing structure files
        results: Dictionary with file paths from report generation
        composition_string: Composition string for naming the ZIP file

    Returns:
        Path to created ZIP file, or None if failed
    """
    zip_path = structure_dir / f"{composition_string}_data.zip"

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add BibTeX file
            if results.get('bibtex_path') and Path(results['bibtex_path']).exists():
                zf.write(results['bibtex_path'], Path(results['bibtex_path']).name)

            # Add structure files
            if results.get('structure_files'):
                for fmt, path in results['structure_files'].items():
                    if Path(path).exists():
                        zf.write(path, f"structures/{Path(path).name}")

            # Add CSV files
            if results.get('csv_files'):
                for csv_path in results['csv_files']:
                    if Path(csv_path).exists():
                        zf.write(csv_path, f"data/{Path(csv_path).name}")

        return zip_path if zip_path.exists() else None
    except Exception as e:
        logger.warning(f"Failed to create data ZIP: {e}")
        return None


@cl.step(type="tool")
async def generate_report(
    structure_ref: Annotated[Union[int, str], "Structure ID (local) or UUID (global)"],
    include_rdf: Annotated[bool, "Include RDF analysis"] = True,
    rdf_cutoff: Annotated[float, "RDF cutoff Å"] = 10.0
) -> Annotated[Dict, "Visual report, PDF, and downloadable files."]:
    """Generate comprehensive report for a structure with visual display and downloadable files."""
    # Create task list
    task_list = cl.TaskList()
    task_list.status = "Running..."

    task1 = cl.Task(title="Loading structure", status=cl.TaskStatus.READY)
    task2 = cl.Task(title="Generating visualizations", status=cl.TaskStatus.READY)
    task3 = cl.Task(title="Displaying analysis", status=cl.TaskStatus.READY)
    task4 = cl.Task(title="Exporting files", status=cl.TaskStatus.READY)
    task5 = cl.Task(title="Generating PDF report", status=cl.TaskStatus.READY)

    await task_list.add_task(task1)
    await task_list.add_task(task2)
    await task_list.add_task(task3)
    await task_list.add_task(task4)
    await task_list.add_task(task5)
    await task_list.send()

    # ===== TASK 1: Load structure =====
    task1.status = cl.TaskStatus.RUNNING
    await task_list.send()

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

    try:
        atoms = db.read(structure_uuid)
        atoms.calc = None  # Remove calculator to avoid OVITO issues
        row = db._get_db().get(unique_id=structure_uuid)
        metadata = db.get_metadata(structure_uuid)
    except Exception as e:
        await cl.Message(content=f"❌ Error loading structure: {e}").send()
        task1.status = cl.TaskStatus.FAILED
        await task_list.send()
        return {"error": str(e)}

    kvp = row.key_value_pairs
    data = row.data
    structure_dir = db.get_structure_directory(structure_uuid)
    composition_string = kvp.get('composition_string', 'Unknown')

    task1.status = cl.TaskStatus.DONE
    await task_list.send()

    # Display metadata summary (ref_display already set at tool start)
    metadata_md = f"""
**Structure {ref_display}...: {composition_string}**

**Composition**: {composition_string}
**Structure**: {kvp.get('target_structure', 'N/A')}
**Atoms**: {kvp.get('optimized_num_atoms', 'N/A')}
**Lattice constant**: {kvp.get('lattice_constant', 0):.4f} Å
**Density**: {kvp.get('density_g_per_cm3', 0):.3f} g/cm³

**Energy (eV/atom)**: {kvp.get('potential_energy_eV_per_atom', 0):.4f}
**Formation energy**: {kvp.get('formation_energy_ground_state_reference_eV_per_atom', 0):.4f}
**Mixing energy**: {kvp.get('mixing_energy_same_structure_reference_eV_per_atom', 0):.4f}

**Max force**: {kvp.get('max_force_magnitude_eV_per_A', 0):.4f} eV/Å
**Calculator**: {kvp.get('calculator_name', 'N/A')}
**Device**: {kvp.get('device_type', 'N/A')}
"""
    await cl.Message(content=metadata_md).send()

    # ===== TASK 2: Generate visualizations =====
    task2.status = cl.TaskStatus.RUNNING
    await task_list.send()

    # Check for cached OVITO visualizations
    elements_img_path = structure_dir / "structure_elements.png"
    analysis_img_path = structure_dir / "structure_analysis.png"

    # Render element-colored if missing
    if not elements_img_path.exists():
        try:
            img_path = render_atoms(atoms, structure_uuid, db=db)
            image = cl.Image(path=img_path, name="element visualization", display="inline")
            await cl.Message(content="**Element-colored visualization:**", elements=[image]).send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Could not render element visualization: {e}").send()
    else:
        image = cl.Image(path=str(elements_img_path), name="element visualization", display="inline")
        await cl.Message(content="**Element-colored visualization (cached):**", elements=[image]).send()

    # Render PTM analysis if missing
    if not analysis_img_path.exists():
        try:
            img_path = render_structure(atoms, structure_uuid, db=db)
            image = cl.Image(path=img_path, name="structure analysis", display="inline")
            await cl.Message(content="**PTM structural analysis:**", elements=[image]).send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Could not render structure visualization: {e}").send()
    else:
        image = cl.Image(path=str(analysis_img_path), name="structure analysis", display="inline")
        await cl.Message(content="**PTM structural analysis (cached):**", elements=[image]).send()

    task2.status = cl.TaskStatus.DONE
    await task_list.send()

    # ===== TASK 3: Display analysis charts and collect figures for PDF =====
    task3.status = cl.TaskStatus.RUNNING
    await task_list.send()

    # Dictionary to collect Plotly figures for PDF export
    collected_figures: Dict[str, Any] = {}

    # Store regeneration data for PDF context rendering
    pdf_regen_data: Dict[str, Any] = {}
    elate_obj_cached = None
    C_voigt_cached = None
    density_cached = None
    qha_plot_data_cached = None
    composition_cached = None
    structure_type_cached = None

    # Structural analysis chart
    try:
        fig = plot_structural_analysis(atoms)
        collected_figures['structural_analysis'] = fig
        elements_list = [cl.Plotly(name="structural_analysis", figure=fig, display="inline", size="large")]
        await cl.Message(content="**Structural analysis summary:**", elements=elements_list).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Could not generate structural analysis chart: {e}").send()

    # RDF chart
    if include_rdf:
        try:
            fig = plot_coordination_rdf(atoms, cutoff=rdf_cutoff)
            collected_figures['rdf'] = fig
            elements_list = [cl.Plotly(name="rdf", figure=fig, display="inline", size="large")]
            await cl.Message(content="**Radial distribution function:**", elements=elements_list).send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Could not generate RDF chart: {e}").send()

    # Elastic properties
    if "elastic_stiffness_tensor_voigt_GPa" in data:
        try:
            C_voigt = np.array(data["elastic_stiffness_tensor_voigt_GPa"])
            K = data.get("bulk_modulus_GPa", 0.0)
            G = data.get("shear_modulus_GPa", 0.0)
            E = data.get("youngs_modulus_GPa", 0.0)
            nu = data.get("poisson_ratio", 0.0)

            elastic_md = f"""
**Elastic Properties**

**Bulk modulus**: {K:.1f} GPa
**Shear modulus**: {G:.1f} GPa
**Young's modulus**: {E:.1f} GPa
**Poisson's ratio**: {nu:.3f}
"""
            await cl.Message(content=elastic_md).send()

            # Stiffness tensor heatmap
            fig = plot_stiffness_tensor_heatmap(C_voigt)
            collected_figures['stiffness_tensor'] = fig
            elements_list = [cl.Plotly(name="stiffness_tensor", figure=fig, display="inline", size="large")]
            await cl.Message(content="**Elastic Stiffness Tensor:**", elements=elements_list).send()

            # ELATE anisotropy visualizations
            if data.get("elastic_tensor_validation_failed", False):
                validation_errors = data.get("elastic_tensor_validation_errors", ["Unknown validation error"])
                await cl.Message(
                    content=f"⚠️ **Elastic tensor validation failed during calculation**\n\n"
                            f"**Issues**: {', '.join(validation_errors)}\n\n"
                            f"Anisotropy analysis was skipped."
                ).send()
            elif "elate_properties" in data:
                try:
                    elate_props = data["elate_properties"]

                    # Validate ELATE results
                    elate_valid, elate_error = validate_elate_results(elate_props)
                    if not elate_valid:
                        await cl.Message(
                            content=f"⚠️ **Warning**: Stored ELATE analysis has invalid results.\n\n"
                                    f"**Issue**: {elate_error}\n\n"
                                    f"Skipping anisotropy visualizations."
                        ).send()
                        raise ValueError("Invalid ELATE results")

                    # Display anisotropy tables
                    tables = create_anisotropy_tables_separated(elate_props)
                    await cl.Message(content="**📊 Elastic Property Comparison**\n\n" + tables['voigt_reuss_hill']).send()
                    await cl.Message(content="**🔍 Anisotropy Measures**\n\n" + tables['anisotropy_measures']).send()
                    await cl.Message(content="**📈 Directional Property Ranges**\n\n" + tables['directional_ranges']).send()
                    await cl.Message(content="**⚙️ Mechanical Behavior Indicators**\n\n" + tables['mechanical_indicators']).send()
                    await cl.Message(content="**🔊 Wave Speed Ranges**\n\n" + tables['wave_speeds']).send()

                    # Get ELATE object for visualizations
                    density_g_cm3 = kvp.get('density_g_per_cm3', 5.0)
                    analyzer = ElasticAnisotropyAnalyzer(C_voigt, density_g_cm3 * 1000)
                    elate_obj = analyzer.get_elate_object()

                    # Cache for PDF regeneration
                    elate_obj_cached = elate_obj
                    C_voigt_cached = C_voigt
                    density_cached = density_g_cm3

                    # Properties to visualize
                    property_types = [
                        ("YOUNG", "Young's Modulus"),
                        ("SHEAR", "Shear Modulus"),
                        ("POISSON", "Poisson's Ratio"),
                        ("LC", "Linear Compressibility"),
                    ]

                    # Add wave speeds if density available
                    if density_g_cm3 and density_g_cm3 > 0:
                        property_types.extend([
                            ("SHEAR_SPEED", "Shear Wave Speed"),
                            ("COMPRESSION_SPEED", "Compression Wave Speed"),
                        ])

                    # Display and collect all elastic property plots
                    for prop_type, prop_name in property_types:
                        figures_2d = plot_directional_property_2d_projections(elate_obj, prop_type)
                        fig_3d = plot_directional_property_3d(elate_obj, prop_type)

                        # Collect for PDF
                        prop_key = prop_type.lower()
                        collected_figures[f'{prop_key}_xy'] = figures_2d[0]
                        collected_figures[f'{prop_key}_xz'] = figures_2d[1]
                        collected_figures[f'{prop_key}_yz'] = figures_2d[2]
                        collected_figures[f'{prop_key}_3d'] = fig_3d

                        # Display in chat
                        elements_list = [
                            cl.Plotly(name="XY Plane (001)", figure=figures_2d[0], display="page", size="large"),
                            cl.Plotly(name="XZ Plane (010)", figure=figures_2d[1], display="page", size="large"),
                            cl.Plotly(name="YZ Plane (100)", figure=figures_2d[2], display="page", size="large"),
                            cl.Plotly(name="3D Surface", figure=fig_3d, display="page", size="large")
                        ]
                        await cl.Message(
                            content=f"**{prop_name} Plots:** XY Plane (001) XZ Plane (010) YZ Plane (100) 3D Surface",
                            elements=elements_list
                        ).send()

                except Exception as e:
                    await cl.Message(content=f"⚠️ Could not generate anisotropy visualizations: {e}").send()

        except Exception as e:
            await cl.Message(content=f"⚠️ Could not generate elastic tensor visualization: {e}").send()

    # QHA properties
    has_qha = kvp.get('has_qha_data', False)
    if has_qha:
        gibbs = kvp.get('qha_gibbs_free_energy_300K_kJ_mol')
        bulk = kvp.get('qha_bulk_modulus_300K_GPa')
        alpha = kvp.get('qha_thermal_expansion_300K_1e6_per_K')
        cp = kvp.get('qha_heat_capacity_p_300K_J_K_mol')
        gruneisen = kvp.get('qha_gruneisen_300K')

        qha_md = f"""
**Finite Temperature Properties (at 300K)**

| Property | Value |
|----------|------:|
| **Gibbs Free Energy (G)** | {f'{gibbs:.2f}' if gibbs else 'N/A'} kJ/mol |
| **Bulk Modulus B(T)** | {f'{bulk:.1f}' if bulk else 'N/A'} GPa |
| **Thermal Expansion (α)** | {f'{alpha:.2f}' if alpha else 'N/A'} ×10⁻⁶/K |
| **Heat Capacity (Cp)** | {f'{cp:.2f}' if cp else 'N/A'} J/(K·mol) |
| **Grüneisen Parameter (γ)** | {f'{gruneisen:.3f}' if gruneisen else 'N/A'} |
"""
        if kvp.get('has_thermal_conductivity', False):
            kappa = kvp.get('thermal_conductivity_300K_W_m_K')
            if kappa:
                qha_md += f"| **Thermal Conductivity (κ)** | {kappa:.1f} W/(m·K) |\n"

        await cl.Message(content=qha_md).send()

        # Display QHA plots - load from database metadata (not CSV, which is created later)
        qha_elements = []
        qha_plot_data = {}

        composition = kvp.get('composition_string', 'Unknown')
        structure_type = kvp.get('target_structure', 'Unknown')

        # Cache for PDF regeneration
        composition_cached = composition
        structure_type_cached = structure_type

        # Load 1D properties from metadata
        if 'qha_1d_properties' in metadata:
            qha_1d = metadata['qha_1d_properties']
            qha_plot_data['temperatures'] = np.array(qha_1d.get('temperatures', []))
            qha_plot_data['gibbs_free_energy'] = np.array(qha_1d.get('gibbs_energies', []))
            qha_plot_data['bulk_modulus'] = np.array(qha_1d.get('bulk_moduli', []))
            qha_plot_data['volume'] = np.array(qha_1d.get('volumes', []))
            qha_plot_data['thermal_expansion'] = np.array(qha_1d.get('thermal_expansion', []))
            qha_plot_data['heat_capacity_p'] = np.array(qha_1d.get('heat_capacities', []))
            qha_plot_data['gruneisen'] = np.array(qha_1d.get('gruneisen_params', []))

        # Load 2D properties from metadata
        if 'qha_2d_properties' in metadata:
            qha_2d = metadata['qha_2d_properties']
            qha_plot_data['helmholtz_volume'] = np.array(qha_2d.get('helmholtz', []))
            qha_plot_data['entropy_volume'] = np.array(qha_2d.get('entropy', []))
            qha_plot_data['cv_volume'] = np.array(qha_2d.get('heat_capacity_cv', []))
            qha_plot_data['volumes_used'] = np.array(qha_2d.get('volumes', []))

            # Fix shape mismatch: 2D arrays may have one extra temperature row
            # (PhonopyQHA drops last T for 1D but 2D retains all points)
            n_temps_1d = len(qha_plot_data.get('temperatures', []))
            if n_temps_1d > 0:
                for key_2d in ['helmholtz_volume', 'entropy_volume', 'cv_volume']:
                    arr = qha_plot_data.get(key_2d)
                    if arr is not None and len(arr) > n_temps_1d:
                        qha_plot_data[key_2d] = arr[:n_temps_1d]

        # Cache for PDF regeneration
        qha_plot_data_cached = qha_plot_data

        try:
            # Display 1D property plots
            if 'temperatures' in qha_plot_data and 'gibbs_free_energy' in qha_plot_data:
                if len(qha_plot_data['temperatures']) > 0:
                    qha_figs = plot_qha_properties_individual(qha_plot_data, composition, structure_type)
                    plot_names = ["Gibbs Free Energy G(T)", "Bulk Modulus B(T)",
                                  "Thermal Expansion α(T)", "Heat Capacity Cp(T)"]
                    for fig, name in zip(qha_figs, plot_names):
                        qha_elements.append(cl.Plotly(name=name, figure=fig, display="page"))
                        collected_figures[f'qha_{name.split()[0].lower()}'] = fig

            # Display 3D surface plots if 2D data is available
            if 'entropy_volume' in qha_plot_data and 'cv_volume' in qha_plot_data:
                if len(qha_plot_data['entropy_volume']) > 0 and len(qha_plot_data['cv_volume']) > 0:
                    surface_figs = plot_qha_volume_surfaces(qha_plot_data, composition, structure_type)
                    surface_names = ['F(T,V) Surface', 'S(T,V) Surface', 'Cv(T,V) Surface']
                    for fig, name in zip(surface_figs, surface_names):
                        qha_elements.append(cl.Plotly(name=name, figure=fig, display="page"))
                        # Collect for PDF export
                        fig_key = name.replace('(', '').replace(')', '').replace(',', '').replace(' ', '_').lower()
                        collected_figures[f'qha_{fig_key}'] = fig

            if qha_elements:
                await cl.Message(
                    content="**Finite Temperature Properties:**\n" + " ".join([e.name for e in qha_elements]),
                    elements=qha_elements
                ).send()

        except Exception as e:
            await cl.Message(content=f"⚠️ Could not load QHA plots: {e}").send()

    task3.status = cl.TaskStatus.DONE
    await task_list.send()

    # ===== TASK 4: Export files =====
    task4.status = cl.TaskStatus.RUNNING
    await task_list.send()

    csv_dir = structure_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "success": True,
        "structure_uuid": structure_uuid,
        "composition_string": composition_string,
        "structure_files": {},
        "csv_files": [],
        "report_path": None,
        "bibtex_path": None,
    }

    # Export structure files
    try:
        exported = await cl.make_async(export_all_structure_formats)(
            atoms, structure_dir, structure_uuid, composition_string
        )
        results['structure_files'] = {k: str(v) for k, v in exported.items()}
    except Exception as e:
        logger.error(f"Failed to export structure files: {e}")

    # Export CSV files
    try:
        # Elastic stiffness tensor
        if 'elastic_stiffness_tensor_voigt_GPa' in metadata:
            tensor = np.array(metadata['elastic_stiffness_tensor_voigt_GPa'])
            csv_path = csv_dir / "elastic_stiffness_tensor.csv"
            await cl.make_async(export_elastic_stiffness_csv)(tensor, csv_path)
            results['csv_files'].append(str(csv_path))

            # ELATE directional properties (if available)
            if 'elate_properties' in metadata:
                try:
                    # Create ELATE object for CSV export
                    density_g_cm3 = kvp.get('density_g_per_cm3', 5.0)
                    analyzer = ElasticAnisotropyAnalyzer(tensor, density_g_cm3 * 1000)
                    elate_obj = analyzer.get_elate_object()

                    # Create ELATE CSV subdirectory
                    elate_csv_dir = csv_dir / "elate"
                    elate_csv_dir.mkdir(parents=True, exist_ok=True)

                    # Properties to export
                    elate_properties = ["YOUNG", "SHEAR", "POISSON", "LC"]
                    if density_g_cm3 and density_g_cm3 > 0:
                        elate_properties.extend(["SHEAR_SPEED", "COMPRESSION_SPEED"])

                    for prop_type in elate_properties:
                        # 2D projections CSV
                        csv_2d = elate_csv_dir / f"{prop_type.lower()}_2d_projections.csv"
                        await cl.make_async(export_elate_2d_csv)(elate_obj, prop_type, csv_2d)
                        results['csv_files'].append(str(csv_2d))

                        # 3D surface CSV
                        csv_3d = elate_csv_dir / f"{prop_type.lower()}_3d_surface.csv"
                        await cl.make_async(export_elate_3d_csv)(elate_obj, prop_type, csv_3d)
                        results['csv_files'].append(str(csv_3d))

                except Exception as e:
                    logger.warning(f"Failed to export ELATE CSV files: {e}")

        # QHA 1D properties
        num_1d_temps = None
        if 'qha_1d_properties' in metadata:
            qha_1d = metadata['qha_1d_properties']
            temps_1d = np.array(qha_1d.get('temperatures', []))

            gibbs = np.array(qha_1d.get('gibbs_energies')) if 'gibbs_energies' in qha_1d else None
            bulk_mod = np.array(qha_1d.get('bulk_moduli')) if 'bulk_moduli' in qha_1d else None
            vols = np.array(qha_1d.get('volumes')) if 'volumes' in qha_1d else None
            therm_exp = np.array(qha_1d.get('thermal_expansion')) if 'thermal_expansion' in qha_1d else None
            heat_cap = np.array(qha_1d.get('heat_capacities')) if 'heat_capacities' in qha_1d else None
            grun = np.array(qha_1d.get('gruneisen_params')) if 'gruneisen_params' in qha_1d else None

            array_lengths = [len(temps_1d)]
            for arr in [gibbs, bulk_mod, vols, therm_exp, heat_cap, grun]:
                if arr is not None and len(arr) > 0:
                    array_lengths.append(len(arr))
            num_1d_temps = min(array_lengths)
            temps_1d = temps_1d[:num_1d_temps]

            csv_path = csv_dir / "finite_temp_properties.csv"
            await cl.make_async(export_qha_properties_csv)(
                temps_1d,
                gibbs[:num_1d_temps] if gibbs is not None else None,
                bulk_mod[:num_1d_temps] if bulk_mod is not None else None,
                vols[:num_1d_temps] if vols is not None else None,
                therm_exp[:num_1d_temps] if therm_exp is not None else None,
                heat_cap[:num_1d_temps] if heat_cap is not None else None,
                grun[:num_1d_temps] if grun is not None else None,
                csv_path
            )
            results['csv_files'].append(str(csv_path))

        # QHA 2D properties
        if 'qha_2d_properties' in metadata:
            qha_2d = metadata['qha_2d_properties']
            temperatures_2d = np.array(qha_2d.get('temperatures', []))
            volumes = np.array(qha_2d.get('volumes', []))

            if num_1d_temps is not None and len(temperatures_2d) > num_1d_temps:
                temperatures_2d = temperatures_2d[:num_1d_temps]

            for prop_name in ['helmholtz', 'entropy', 'heat_capacity_cv']:
                if prop_name in qha_2d:
                    data_2d = np.array(qha_2d[prop_name])
                    if num_1d_temps is not None and data_2d.shape[0] > num_1d_temps:
                        data_2d = data_2d[:num_1d_temps, :]
                    csv_path = csv_dir / f"{prop_name}_vs_T_V.csv"
                    await cl.make_async(export_2d_property_csv)(
                        temperatures_2d, volumes, data_2d, prop_name, csv_path
                    )
                    results['csv_files'].append(str(csv_path))

        # Thermal conductivity
        if 'thermal_conductivity' in metadata:
            tc = metadata['thermal_conductivity']
            csv_path = csv_dir / "thermal_conductivity.csv"
            await cl.make_async(export_thermal_conductivity_csv)(
                np.array(tc.get('temperatures', [])),
                np.array(tc.get('kappa_xx', [])),
                np.array(tc.get('kappa_yy', [])),
                np.array(tc.get('kappa_zz', [])),
                np.array(tc.get('kappa_iso', [])),
                csv_path
            )
            results['csv_files'].append(str(csv_path))

        # RDF data
        rdf_data = metadata.get('rdf_data')
        if rdf_data is not None:
            csv_path = csv_dir / "rdf_data.csv"
            await cl.make_async(export_rdf_csv)(
                np.array(rdf_data.get('r_values', [])),
                np.array(rdf_data.get('g_r_values', [])),
                csv_path,
                partial=rdf_data.get('partial')
            )
            results['csv_files'].append(str(csv_path))

    except Exception as e:
        logger.error(f"Failed to export CSV files: {e}")

    task4.status = cl.TaskStatus.DONE
    await task_list.send()

    # ===== TASK 5: Generate PDF report with elastic figures =====
    task5.status = cl.TaskStatus.RUNNING
    await task_list.send()

    try:
        # Save ELATE figures as PNG for PDF with PDF-optimized font sizes
        elate_png_dir = structure_dir / "elate_plots"
        elate_image_paths = {}

        # Regenerate ELATE figures with context="pdf" for proper font sizing
        if elate_obj_cached is not None:
            property_types_for_pdf = [
                ("YOUNG", "young"),
                ("SHEAR", "shear"),
                ("POISSON", "poisson"),
                ("LC", "lc"),
            ]
            if density_cached and density_cached > 0:
                property_types_for_pdf.extend([
                    ("SHEAR_SPEED", "shear_speed"),
                    ("COMPRESSION_SPEED", "compression_speed"),
                ])

            for prop_type, prop_key in property_types_for_pdf:
                # Regenerate with PDF context
                figures_2d_pdf = plot_directional_property_2d_projections(
                    elate_obj_cached, prop_type, context="pdf"
                )
                fig_3d_pdf = plot_directional_property_3d(
                    elate_obj_cached, prop_type, context="pdf"
                )

                # Save 2D projections
                for plane_idx, plane_name in enumerate(['xy', 'xz', 'yz']):
                    fig_name = f'{prop_key}_{plane_name}'
                    png_path = elate_png_dir / f"{fig_name}.png"
                    saved_path = await cl.make_async(save_plotly_figure_as_png)(
                        figures_2d_pdf[plane_idx], png_path
                    )
                    if saved_path:
                        elate_image_paths[fig_name] = saved_path

                # Save 3D surface
                fig_name = f'{prop_key}_3d'
                png_path = elate_png_dir / f"{fig_name}.png"
                saved_path = await cl.make_async(save_plotly_figure_as_png)(fig_3d_pdf, png_path)
                if saved_path:
                    elate_image_paths[fig_name] = saved_path

        # Save QHA surface plots as PNG for PDF with PDF-optimized font sizes
        qha_png_dir = structure_dir / "qha_plots"
        qha_surface_image_paths = {}

        # Regenerate QHA surface plots with context="pdf"
        if qha_plot_data_cached and 'entropy_volume' in qha_plot_data_cached:
            if len(qha_plot_data_cached.get('entropy_volume', [])) > 0:
                surface_figs_pdf = plot_qha_volume_surfaces(
                    qha_plot_data_cached,
                    composition_cached or 'Unknown',
                    structure_type_cached or 'Unknown',
                    context="pdf"
                )
                surface_keys = ['qha_ftv_surface', 'qha_stv_surface', 'qha_cvtv_surface']
                for fig, fig_name in zip(surface_figs_pdf, surface_keys):
                    png_path = qha_png_dir / f"{fig_name}.png"
                    saved_path = await cl.make_async(save_plotly_figure_as_png)(
                        fig, png_path, width=900, height=700
                    )
                    if saved_path:
                        qha_surface_image_paths[fig_name] = saved_path

        # Build composition dictionary
        composition = {}
        for elem in ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
                    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
                    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']:
            frac_key = f"{elem}_fraction"
            frac = metadata.get(frac_key, 0)
            if frac and frac > 0:
                composition[elem] = frac

        # Build structure data dict
        structure_data = {
            'calculator_name': metadata.get('calculator_name', 'Unknown'),
            'device_type': metadata.get('device_type', 'Unknown'),
            'target_structure': metadata.get('target_structure', 'Unknown'),
            'num_atoms': len(atoms),
            'density_g_per_cm3': metadata.get('density_g_per_cm3'),
            'volume_per_atom_A3': metadata.get('volume_per_atom_A3'),
            'energy_per_atom_eV': metadata.get('energy_per_atom_eV'),
            'formation_energy_ground_state_reference_eV_per_atom': metadata.get(
                'formation_energy_ground_state_reference_eV_per_atom'
            ),
        }

        # Cell parameters
        cell = atoms.get_cell()
        lengths = cell.lengths()
        angles = cell.angles()
        structure_data['cell_parameters'] = {
            'a': lengths[0], 'b': lengths[1], 'c': lengths[2],
            'alpha': angles[0], 'beta': angles[1], 'gamma': angles[2],
        }

        # PTM analysis
        if 'PTM_structural_analysis_in_percent' in metadata:
            structure_data['PTM_structural_analysis_in_percent'] = metadata['PTM_structural_analysis_in_percent']

        # Elastic properties
        if 'elastic_stiffness_tensor_voigt_GPa' in metadata or 'elastic_moduli' in metadata:
            structure_data['elastic_properties'] = {}
            if 'elastic_stiffness_tensor_voigt_GPa' in metadata:
                structure_data['elastic_properties']['stiffness_tensor'] = metadata['elastic_stiffness_tensor_voigt_GPa']
            if 'elastic_moduli' in metadata:
                structure_data['elastic_properties'].update(metadata['elastic_moduli'])

        # Elastic stability
        if 'elastic_stability' in metadata:
            structure_data['elastic_stability'] = metadata['elastic_stability']
        elif 'elastic_stability_assessment' in metadata:
            structure_data['elastic_stability'] = metadata['elastic_stability_assessment']

        # ELATE properties
        if 'elate_properties' in metadata:
            structure_data['elate_properties'] = metadata['elate_properties']

        # QHA properties
        if 'qha_1d_properties' in metadata:
            structure_data['qha_properties'] = {'temperature_dependent': metadata['qha_1d_properties']}

        # RDF data
        if 'rdf_data' in metadata:
            structure_data['rdf_data'] = metadata['rdf_data']

        # Determine calculation types
        calculation_types = ['alloy_generation']
        if 'elastic_stiffness_tensor_voigt_GPa' in metadata:
            calculation_types.append('elastic')
        if 'qha_1d_properties' in metadata:
            calculation_types.append('qha')
        if 'thermal_conductivity' in metadata:
            calculation_types.append('thermal_conductivity')

        # Find image paths
        image_paths = {}

        def process_image_file(img_file):
            if 'structure_elements' in img_file.name or img_file.name == 'element_colored.png':
                image_paths['structure_elements'] = img_file
                if 'structure' not in image_paths:
                    image_paths['structure'] = img_file
            elif 'structure_analysis' in img_file.name or 'ptm' in img_file.name.lower():
                image_paths['structure_analysis'] = img_file
            elif 'structure' in img_file.name and 'elements' not in img_file.name and 'analysis' not in img_file.name:
                if 'structure' not in image_paths:
                    image_paths['structure'] = img_file

        images_dir = structure_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.glob("*.png"):
                process_image_file(img_file)

        for img_file in structure_dir.glob("*.png"):
            process_image_file(img_file)

        # Add ELATE images to image_paths
        image_paths['elate'] = elate_image_paths

        # Add QHA surface images to image_paths
        image_paths['qha_surfaces'] = qha_surface_image_paths

        # Generate PDF report
        report_path = structure_dir / "report.pdf"
        await cl.make_async(generate_structure_report)(
            structure_uuid,
            composition,
            structure_data,
            calculation_types,
            report_path,
            image_paths,
            composition_string,
        )
        results['report_path'] = str(report_path)

        # Generate BibTeX file
        all_refs = []
        for calc_type in calculation_types:
            refs = get_all_references_for_calculation(calc_type)
            for ref in refs:
                if ref not in all_refs:
                    all_refs.append(ref)
        for core_ref in ['ase2017', 'ovito2010']:
            if core_ref not in all_refs:
                all_refs.append(core_ref)

        bibtex_path = structure_dir / "references.bib"
        await cl.make_async(generate_bibtex_file)(all_refs, bibtex_path)
        results['bibtex_path'] = str(bibtex_path)

        task5.status = cl.TaskStatus.DONE

    except Exception as e:
        task5.status = cl.TaskStatus.FAILED
        results['report_error'] = str(e)
        import traceback
        results['report_traceback'] = traceback.format_exc()
        logger.error(f"Failed to generate PDF report: {e}")

    await task_list.send()
    task_list.status = "Done!"
    await task_list.send()

    # ===== Display download buttons =====
    await cl.Message(content=f"**Export complete for structure {ref_display}**").send()

    download_elements = []
    download_names = []

    # Button 1: PDF Report
    if results.get('report_path') and Path(results['report_path']).exists():
        download_names.append("report.pdf")
        download_elements.append(cl.File(name="report.pdf", path=results['report_path'], display="inline"))

    # Button 2: Data ZIP (all other files: structures, CSV, BibTeX)
    zip_path = await cl.make_async(create_data_zip)(structure_dir, results, composition_string)
    if zip_path and zip_path.exists():
        zip_name = zip_path.name
        download_names.append(zip_name)
        download_elements.append(cl.File(name=zip_name, path=str(zip_path), display="inline"))
        results['zip_path'] = str(zip_path)

    if download_elements:
        await cl.Message(
            content="**Download:** " + " ".join(download_names),
            elements=download_elements
        ).send()

    # Return a slim summary to the LLM. The full `results` dict is only used
    # internally (zip building, cl.File elements) — exposing its file paths
    # to the model causes it to confabulate markdown download lists in its
    # reply that look real but point to broken links.
    return {
        "success": results.get("success", True),
        "structure_uuid": results.get("structure_uuid"),
        "composition_string": results.get("composition_string"),
        "note": (
            "Report PDF and data bundle are displayed above as download "
            "buttons. Do not list files or re-render download links in your "
            "reply; the UI already shows them."
        ),
    }
