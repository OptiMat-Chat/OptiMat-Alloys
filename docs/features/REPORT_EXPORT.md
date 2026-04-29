# Report Generation and Data Export Feature

This document describes the on-demand report generation and data export feature for OptiMat Alloys.

## Overview

The `generate_report` tool provides on-demand visual analysis and full data export for a previously computed structure:
- **In-chat output**: structure visualizations (element-colored + PTM analysis), structural-analysis chart, RDF, elastic property tables, QHA plots, ELATE anisotropy tables and directional plots
- **Structure files**: CIF, VASP POSCAR, LAMMPS data, XYZ formats
- **Property CSVs**: elastic tensor, finite-temperature (QHA) properties, 2D F/S/Cv vs. T,V grids, thermal conductivity, RDF, ELATE 2D/3D directional data
- **PDF report**: publication-ready PDF with methods, parameters, and references
- **BibTeX file**: citations for every computational method that was used (calculator + analysis tools)

## Key Design Principle

**On-demand only**: Export only happens when the user explicitly requests it. No automatic file generation during calculations. This:
- Reduces disk usage
- Gives users control over what gets exported
- Keeps the computation pipeline clean

## Usage

### Natural Language Triggers

The agent's `factory.py` system message lists `generate_report` by name. Phrases like the following lead it to call the tool:

| Phrase | Action |
|--------|--------|
| "show / display / view structure 7" | `generate_report(structure_ref=7)` |
| "generate report for Cu-Ag" | `search_database` → resolve UUID → `generate_report(structure_ref=UUID)` |
| "RDF / radial distribution of structure 7" | `generate_report(structure_ref=7, include_rdf=True)` |
| "PDF report" / "export structure" / "download CIF" | `generate_report(structure_ref=…)` — there is no per-format flag; the tool always emits all formats |
| "skip RDF" | `generate_report(..., include_rdf=False)` |

The tool always exports everything it can (structure files in 4 formats + every CSV that has data + PDF + `references.bib`); there is no `export_format` selector.

### Tool Signature

```python
async def generate_report(
    structure_ref: Union[int, str],   # Structure ID (local int) or UUID (global str)
    include_rdf: bool = True,         # Include RDF chart and CSV
    rdf_cutoff: float = 10.0          # RDF cutoff in Å
) -> Dict
```

The tool runs as a 5-task Chainlit task list:

1. **Loading structure** — resolves `structure_ref` to a UUID, reads atoms + metadata.
2. **Generating visualizations** — element-colored + PTM analysis images (cached on disk; only re-rendered if missing).
3. **Displaying analysis** — sends structural-analysis, RDF, elastic tables/heatmap, ELATE tables + 24 directional plots, and QHA/thermal-conductivity plots into the chat.
4. **Exporting files** — writes structure files + CSVs (see [Output Files](#output-files)).
5. **Generating PDF report** — assembles `report.pdf` and writes `references.bib`.

### Example Conversations

```
User: Generate a report for the Cu-Ag alloy I just created
Agent: [search_database to resolve UUID, then generate_report(structure_ref=<uuid>)]

User: Show me structure 7
Agent: [generate_report(structure_ref=7)]

User: Skip the RDF, it's slow on this big cell
Agent: [generate_report(structure_ref=7, include_rdf=False)]
```

## Output Files

Files are saved to `structures/{uuid}/`. CSVs that have no source data are skipped.

```
structures/{uuid}/
├── structure.cif                      # CIF format
├── POSCAR                             # VASP format
├── structure.lammps                   # LAMMPS data (atomic style)
├── structure.xyz                      # XYZ format
├── structure_elements.png             # Element-colored OVITO render (cached)
├── structure_analysis.png             # PTM analysis OVITO render (cached)
├── csv/                               # CSV data files
│   ├── elastic_stiffness_tensor.csv
│   ├── elastic_stiffness_tensor_flat.csv
│   ├── finite_temp_properties.csv     # QHA 1D: B(T), V(T), α(T), Cp(T), γ(T)
│   ├── helmholtz_vs_T_V.csv
│   ├── entropy_vs_T_V.csv
│   ├── heat_capacity_cv_vs_T_V.csv
│   ├── thermal_conductivity.csv
│   ├── rdf_data.csv
│   └── elate/                         # ELATE directional data (one pair per property)
│       ├── young_2d_projections.csv
│       ├── young_3d_surface.csv
│       ├── shear_2d_projections.csv
│       ├── shear_3d_surface.csv
│       └── ...                        # plus poisson, bulk, lc, shear_speed, compression_speed
├── report.pdf                         # Publication-ready report
└── references.bib                     # BibTeX citations
```

## CSV File Formats

### Elastic Stiffness Tensor

`elastic_stiffness_tensor.csv`:
```csv
,C1,C2,C3,C4,C5,C6
C1,168.3,122.6,122.6,0.0,0.0,0.0
C2,122.6,168.3,122.6,0.0,0.0,0.0
...
```

`elastic_stiffness_tensor_flat.csv`:
```csv
component,value_GPa
C11,168.3
C12,122.6
...
```

### Finite-Temperature (QHA) Properties (1D)

`finite_temp_properties.csv`:
```csv
Temperature_K,Gibbs_energy_eV,Bulk_modulus_GPa,Volume_A3,Thermal_expansion_1/K,Heat_capacity_Cp_J/mol/K,Gruneisen_parameter
0.00,-5.234567,142.50,11.8234,0.000000e+00,0.000000,1.234567
10.00,-5.234500,142.48,11.8235,1.234567e-08,0.123456,1.234560
...
```

### QHA Properties (2D - F, S, Cv vs T and V)

`helmholtz_vs_T_V.csv`, `entropy_vs_T_V.csv`, `heat_capacity_cv_vs_T_V.csv`:
```csv
Temperature_K,V=11.5000_A3,V=11.7000_A3,V=11.9000_A3,...
0.00,-5.234567,-5.234500,-5.234433,...
10.00,-5.234560,-5.234493,-5.234426,...
...
```

### Thermal Conductivity

`thermal_conductivity.csv`:
```csv
Temperature_K,kappa_xx_W/m/K,kappa_yy_W/m/K,kappa_zz_W/m/K,kappa_iso_W/m/K
100.00,45.6789,45.6789,45.6789,45.6789
200.00,23.4567,23.4567,23.4567,23.4567
...
```

### RDF Data

`rdf_data.csv` — total RDF plus optional partial RDFs (one column per element pair):
```csv
r_A,g_r_total,g_Cu-Cu,g_Cu-Ag,g_Ag-Ag
0.0500,0.000000,0.000000,0.000000,0.000000
0.1000,0.000000,0.000000,0.000000,0.000000
...
2.5000,1.234567,1.111111,0.222222,1.555555
```

For single-element structures only the `g_r_total` column is present; for binary/ternary alloys the partial columns are added in sorted order.

## PDF Report Structure

The generated PDF report (~6-8 pages) includes:

### 1. Title Page
- Composition and structure ID
- Date and calculator information
- Structure visualization image

### 2. Structure Summary
- Number of atoms, target structure
- Density, volume per atom
- Cell parameters (a, b, c)
- Energy and formation energy
- PTM structural analysis

### 3. Elastic Properties (if computed)
- Elastic moduli table (Voigt/Reuss/Hill)
- 6x6 stiffness tensor
- ELATE anisotropy visualization

### 4. Thermodynamic Properties (if computed)
- QHA properties at key temperatures
- B(T), V(T), α(T), Cp(T) curves
- Thermal conductivity (if computed)

### 5. Computational Methods
- Calculator details — populated dynamically from the structure's stored `calculator_name` (ORB, MACE, NequIP variants are all handled by `references.py::get_methods_for_calculation`)
- Relaxation parameters (optimizer, fmax)
- QHA parameters (volumes, mesh, T range)
- Elastic calculation parameters

### 6. References
- Publication-style citations
- Software versions

## BibTeX References

`references.bib` is assembled by `src/core/references.py::generate_bibtex_file`. Only entries relevant to the methods that actually ran for this structure are included — the calculator citation is selected from `calculator_name`, and analysis citations are added per calculation type (relaxation, elastic, QHA, thermal conductivity, anisotropy, etc.).

Currently defined entry keys (see `src/core/references.py` `BIBTEX_REFERENCES` dict):

- **Calculators / training data**: `orb2024`, `omat24_2024`, `nequip2022`, `allegro2023`, `mace2022`, `mace_mp_2023`, `mace_mpa_2024`, `mace_omat_2024`, `mace_matpes_2024`, `mace_mh_2024`
- **Phonons / QHA / κ**: `phonopy2015`, `qha_togo2010`, `phono3py2015`
- **Elastic + anisotropy**: `elastic_fd_2002`, `elate_gaillac2016`, `mechelastic2021`, `reuss1929`, `hill1952`, `anisotropy_ranganathan2008`
- **Structure analysis / generation**: `ptm_larsen2016`, `sqs_zunger1990`, `sqsgenerator2023`
- **Optimizer / framework**: `fire_optimizer2006`, `ase2017`, `ovito2010`

Example excerpt for a structure that used ORB + phonons:

```bibtex
@article{phonopy2015,
    author = {Togo, Atsushi and Tanaka, Isao},
    title = {{First principles phonon calculations in materials science}},
    journal = {Scripta Materialia},
    volume = {108},
    pages = {1--5},
    year = {2015},
    doi = {10.1016/j.scriptamat.2015.07.021}
}

@misc{orb2024,
    author = {{Orbital Materials}},
    title = {{ORB Models: Universal Neural Network Potentials for Atomistic Simulations}},
    year = {2024},
    url = {https://github.com/orbital-materials/orb-models}
}
```

## User Notification

When export is activated, users see a notification:

```
**Generating export package for structure abc12345...**

**Structure files:**
  - structure.cif (CIF format)
  - POSCAR (VASP format)
  - structure.lammps (LAMMPS data)
  - structure.xyz (XYZ format)

**CSV data files** (if computed):
  - elastic_stiffness_tensor.csv
  - finite_temp_properties.csv
  - thermal_conductivity.csv
  - rdf_data.csv
  - elate/ (one 2D + 3D pair per ELATE property)

**Report:**
  - report.pdf (Publication-ready)
  - references.bib (BibTeX citations)
```

## Implementation Details

### Module Structure

```
src/
├── core/
│   ├── structure_export.py    # Format conversion + per-property CSV writers
│   ├── report_generator.py    # PDF generation (reportlab)
│   └── references.py          # ComputationalMethod definitions, BIBTEX_REFERENCES, generate_bibtex_file
├── tools/
│   └── generate_report.py     # The `generate_report` tool — orchestrates display + export + PDF
```

### Dependencies

- `reportlab>=4.0.0`: PDF generation
- `ase`: Structure format conversion (CIF, VASP, LAMMPS, XYZ)

### Data Flow

1. User asks to view, report, or export a structure
2. Agent calls `generate_report(structure_ref=…)`
3. Tool resolves the ref → UUID and reads atoms + metadata + property arrays from the database
4. In-chat: structural-analysis chart, RDF, elastic tables/heatmap, ELATE tables + 24 directional plots, QHA / κ plots
5. To disk under `structures/{uuid}/`:
   - structure files via `export_all_structure_formats` (CIF / VASP / LAMMPS / XYZ)
   - per-property CSVs via `export_*_csv` helpers in `src/core/structure_export.py`
6. PDF report via `report_generator.py` (reportlab)
7. `references.bib` via `references.generate_bibtex_file`, scoped to the methods that actually ran
8. Final task list shown to the user with download links

## History & Data Model

Originally there were two separate tools — `display_structure_report` (in-chat visualization) and `export_structure_data` (file export). They were merged into a single `generate_report` tool that handles both in the same task list, so users get the visual analysis and the downloadable bundle in one call (`src/tools/generate_report.py` line 4 still notes the merge).

Calculation tools (QHA, elastic, thermal conductivity) no longer write CSVs or HTML directly — they store their results in the structure's database row, and `generate_report` reads from there. Relevant `data`-field keys:

- `qha_1d_properties`, `qha_2d_properties` (set in `src/tools/anharmonic_properties.py`)
- `thermal_conductivity`
- `elastic_stiffness_tensor_voigt_GPa`, `elate_properties` (set in `src/tools/elastic_properties.py`)

This keeps the calculation pipeline clean and means the same structure can be re-reported any number of times without re-running the underlying calculation.

## Troubleshooting

### Missing Data in Export

If CSV files are empty or not generated:
1. Check if the property was computed (QHA, elastic, etc.)
2. Verify the structure UUID is correct
3. Check database has the required data fields

### PDF Generation Errors

If PDF fails to generate:
1. Ensure `reportlab` is installed: `pip install reportlab>=4.0.0`
2. Check for missing data in database
3. Verify image paths are correct (if including visualizations)

### BibTeX Issues

If references.bib is incomplete:
1. Check which calculation types were performed
2. Verify the references module has all required entries
