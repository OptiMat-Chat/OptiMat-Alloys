# ELATE Elastic Anisotropy Analysis

**Feature Added**: January 2025

## Overview

OptiMat Alloys integrates [MechElastic ELATE](https://github.com/romerogroup/MechElastic) for rigorous elastic tensor analysis, providing comprehensive anisotropy characterization for materials.

### Capabilities

- **Voigt/Reuss/Hill averaging** for all elastic moduli (polycrystalline behavior)
- **Universal Anisotropy Index** and other anisotropy measures
- **Directional property variations** (min/max values with crystal orientations)
- **Auxetic behavior detection** (negative Poisson's ratio)
- **Ductility prediction** via Pugh ratio (K/G)
- **Acoustic wave speed calculations** (shear and compression waves)

### Modules

- **Core**: `src/core/elate_analysis.py` - ElasticAnisotropyAnalyzer class
- **Visualization**: `src/visualization/elate_plots.py` - Tables and Plotly charts

## Integration Points

### 1. Calculate Elastic Properties Tool

**Location**: `src/tools/elastic_properties.py` (`calculate_elastic_properties` tool)

Automatically computes ELATE properties after elastic tensor calculation:

- **Requirements**: Density must be available in database (for wave speeds)
- **Output**: Quick anisotropy summary displayed to user
  - Universal Anisotropy Index with classification
  - Pugh Ratio (ductility: brittle vs. ductile)
  - Auxetic behavior flag (if negative Poisson's ratio detected)
- **Storage**: Full `elate_properties` dictionary saved to database
- **Graceful degradation**: Skips if density unavailable

### 2. Generate Report Tool

**Location**: `src/tools/generate_report.py` (`generate_report` tool)

Displays comprehensive ELATE visualizations if elastic tensor was previously computed:

#### Five Markdown Tables (Displayed as Separate Messages)

Each table is displayed as a separate message with emoji headers for clear organization:

1. **📊 Elastic Property Comparison** (Voigt, Reuss, Hill)
   - Bulk modulus (K)
   - Shear modulus (G)
   - Young's modulus (E)
   - Poisson's ratio (ν)

2. **🔍 Anisotropy Measures**
   - Universal Anisotropy Index (AU)
   - Shear anisotropy
   - Young's modulus anisotropy
   - Poisson's ratio anisotropy

3. **📈 Directional Property Ranges**
   - Min/max Young's modulus with anisotropy ratio
   - Min/max Poisson's ratio (auxetic detection)
   - Min/max shear modulus with ratio

4. **⚙️ Mechanical Behavior Indicators**
   - Pugh ratio (K/G) with ductility classification

5. **🔊 Wave Speed Ranges**
   - Min/max shear wave speeds
   - Min/max compression wave speeds

#### Twenty-Four Interactive Plotly Charts (2D Projections + 3D Surfaces)

For each property, both 2D projections and a 3D surface plot are displayed:

1. **Young's Modulus Directional Dependence**
   - 2D: XY, XZ, YZ plane projections (polar plots)
   - 3D: Semi-transparent spherical surface plot

2. **Shear Modulus Directional Dependence**
   - 2D: XY, XZ, YZ plane projections (polar plots)
   - 3D: Semi-transparent spherical surface plot

3. **Poisson's Ratio Directional Dependence**
   - 2D: XY, XZ, YZ plane projections (polar plots)
   - 3D: Semi-transparent spherical surface plot

4. **Linear Compressibility Directional Dependence**
   - 2D: XY, XZ, YZ plane projections (polar plots)
   - 3D: Semi-transparent spherical surface plot

5. **Shear Wave Speed Directional Dependence** (if density available)
   - 2D: XY, XZ, YZ plane projections (polar plots)
   - 3D: Semi-transparent spherical surface plot

6. **Compression Wave Speed Directional Dependence** (if density available)
   - 2D: XY, XZ, YZ plane projections (polar plots)
   - 3D: Semi-transparent spherical surface plot

**3D Plot Features**:
- Semi-transparent surfaces (opacity=0.7) for better depth perception
- Turbo colorscale for property magnitude
- Enhanced lighting for improved visibility
- Full-page display mode for maximum interactivity

## Usage Example

### Calculate Elastic Tensor

```python
User: "Calculate elastic stiffness tensor for structure ID 1"

# Agent Output:
# Task 3: Calculating derived elastic moduli ✓
# Task 3.5: Computing comprehensive anisotropy analysis (ELATE) ✓
#
# Anisotropy Analysis (ELATE):
# - Universal Anisotropy Index: 0.244 (Weakly Anisotropic)
# - Pugh Ratio (K/G): 1.47 (Brittle)
# - Auxetic behavior: No
```

### Generate Visual Report

```python
User: "Generate visual report for structure ID 1"

# Agent Output:
# [5 comprehensive tables with anisotropy data - displayed as separate messages]
# [24 interactive Plotly charts - 6 properties × (3 2D projections + 1 3D surface)]
```

## Code Example

### Computing ELATE Properties

```python
from src.core.elate_analysis import compute_elate_properties

# Compute ELATE properties from elastic tensor
C_voigt = np.array([...])  # 6x6 stiffness tensor (GPa)
density_g_cm3 = 2.33  # Material density

props = compute_elate_properties(C_voigt, density_g_cm3)

# Access properties
print(f"Universal Anisotropy: {props['universal_anisotropy_index']:.3f}")
print(f"Pugh Ratio: {props['pugh_ratio_hill']:.2f}")
print(f"Young's Range: {props['min_youngs_modulus_GPa']:.1f} - {props['max_youngs_modulus_GPa']:.1f} GPa")
```

## Anisotropy Classifications

### Universal Anisotropy Index (AU)

| Range | Classification |
|-------|----------------|
| AU < 0.1 | Nearly Isotropic |
| 0.1 ≤ AU < 1.0 | Weakly Anisotropic |
| 1.0 ≤ AU < 5.0 | Moderately Anisotropic |
| AU ≥ 5.0 or ∞ | Highly Anisotropic |

**Interpretation**: AU quantifies deviation from isotropic behavior. AU = 0 means perfectly isotropic (e.g., ideal cubic crystal or polycrystal). Higher values indicate stronger directional dependence of elastic properties.

### Pugh Ratio (K/G)

| Pugh Ratio | Ductility |
|------------|-----------|
| K/G ≤ 1.75 | Brittle |
| K/G > 1.75 | Ductile |

**Interpretation**: Pugh ratio predicts mechanical behavior. High K/G means material resists volume change more than shape change → ductile. Low K/G means material resists shape change more → brittle.

## Key Properties Computed

### Averaging Methods (Polycrystalline Behavior)

```python
# Bulk modulus (resistance to volume change)
voigt_bulk_modulus_GPa          # Upper bound (uniform strain assumption)
reuss_bulk_modulus_GPa          # Lower bound (uniform stress assumption)
hill_bulk_modulus_GPa           # Average (most accurate for polycrystals)

# Shear modulus (resistance to shape change)
voigt_shear_modulus_GPa
reuss_shear_modulus_GPa
hill_shear_modulus_GPa

# Young's modulus (stiffness in tension/compression)
voigt_youngs_modulus_GPa
reuss_youngs_modulus_GPa
hill_youngs_modulus_GPa

# Poisson's ratio (lateral strain vs. axial strain)
voigt_poisson_ratio
reuss_poisson_ratio
hill_poisson_ratio
```

### Anisotropy Measures

```python
universal_anisotropy_index      # 0 = isotropic, >0 = anisotropic
shear_anisotropy                # Shear modulus variation
youngs_anisotropy               # Young's modulus variation
poisson_anisotropy              # Poisson ratio variation
bulk_modulus_anisotropy         # Bulk modulus variation
```

### Directional Ranges (Single Crystal Behavior)

```python
# Stiffness direction dependence
min_youngs_modulus_GPa          # Minimum stiffness direction
max_youngs_modulus_GPa          # Maximum stiffness direction

# Lateral contraction behavior
min_poisson_ratio               # Minimum lateral strain ratio
max_poisson_ratio               # Maximum lateral strain ratio
has_auxetic_behavior            # True if min_poisson < 0 (exotic behavior)

# Compressibility
min_linear_compressibility_TPa_inv
max_linear_compressibility_TPa_inv

# Shear resistance
min_shear_modulus_GPa
max_shear_modulus_GPa
```

### Mechanical Properties

```python
pugh_ratio_hill                 # K/G ratio (ductility indicator)
```

### Acoustic Wave Speeds

```python
min_shear_wave_speed_m_s        # Slowest shear wave
max_shear_wave_speed_m_s        # Fastest shear wave
min_compression_wave_speed_m_s  # Slowest compression (longitudinal) wave
max_compression_wave_speed_m_s  # Fastest compression (longitudinal) wave
```

## Database Storage

ELATE properties are saved in the `data` field of database entries.

### Storing Properties

```python
# Properties automatically stored during calculate_elastic_properties
# No manual intervention required
```

### Retrieving Properties

```python
from src.storage.database import create_structure_database

db = create_structure_database()
row = db._get_db().get(id=structure_id)

if "elate_properties" in row.data:
    elate_props = row.data["elate_properties"]

    # Access any property
    print(f"AU: {elate_props['universal_anisotropy_index']:.3f}")
    print(f"Pugh: {elate_props['pugh_ratio_hill']:.2f}")
    print(f"Ductility: {'Ductile' if elate_props['pugh_ratio_hill'] > 1.75 else 'Brittle'}")

    # Check for auxetic behavior
    if elate_props['has_auxetic_behavior']:
        print("Material exhibits auxetic behavior (negative Poisson's ratio)")
```

## Testing

Manual validation of the ELATE pipeline can be done end-to-end through the Chainlit UI:

1. Start the app: `chainlit run run_chat.py`
2. Generate or load a structure (e.g. *"Generate FCC Cu"*).
3. Compute the elastic tensor: *"Calculate elastic stiffness tensor for structure 1"* — the tool runs `compute_elate_properties()` and stores the result under `row.data["elate_properties"]`.
4. Render the visualizations: *"Generate visual report for structure 1"* — produces 5 tables and 24 directional plots.
5. Spot-check the stored fields against the **Key Properties Computed** section above by inspecting the row in `structures/database.db` (see `docs/MAINTENANCE.md` → "Query Database Stats" for the ASE/Python recipe).

There is currently no automated unit/integration suite for ELATE in this repository; the standalone test report at `tests/ELATE_AUTONOMOUS_TEST_REPORT.md` and the autonomous-test driver at `tests/run_elate_autonomous.py` are the closest historical artifacts.

## Code References

### Core Module

- **ElasticAnisotropyAnalyzer class**: `src/core/elate_analysis.py`
- **compute_elate_properties() function**: `src/core/elate_analysis.py`

### Visualization Module

- **create_anisotropy_summary_table()**: `src/visualization/elate_plots.py`
  - Original combined table function (still available for backwards compatibility)
- **create_anisotropy_tables_separated()**: `src/visualization/elate_plots.py`
  - Returns 5 separate tables as a dictionary
  - Used by `generate_report` for cleaner UI organization
- **plot_directional_property_2d_projections()**: `src/visualization/elate_plots.py`
  - Supports: YOUNG, SHEAR, POISSON, BULK, LC, SHEAR_SPEED, COMPRESSION_SPEED
  - Returns list of 3 figures (XY, XZ, YZ planes)
  - Uses real ELATE calculations via `elate.elas.Young([theta, phi])` etc.
- **plot_directional_property_3d()**: `src/visualization/elate_plots.py`
  - Supports: YOUNG, SHEAR, POISSON, BULK, LC, SHEAR_SPEED, COMPRESSION_SPEED
  - Returns single 3D figure with semi-transparent surface (opacity=0.7)
  - Uses real ELATE calculations with 60×120 angular mesh
  - Enhanced lighting for better visibility with transparency

### Tool Integration

- **ELATE analysis in `calculate_elastic_properties`**: `src/tools/elastic_properties.py` (computes comprehensive properties)
- **ELATE visualizations in `generate_report`**: `src/tools/generate_report.py` (displays 5 tables + 24 directional plots)

## Known Limitations

### Density Requirement

ELATE requires density for wave speed calculations. If density is not available in the database:
- Wave speed calculations are skipped
- All other ELATE properties are still computed
- User is notified that density is missing

**Workaround**: Ensure structures have density calculated before computing elastic tensor.

### ~~Simplified Directional Data~~ ✓ FIXED

~~Current 2D/3D visualizations use simplified directional data (sampled at key angles) for demonstration purposes.~~

**FIXED (January 2025)**: All visualizations now use real ELATE directional property calculations at each angle, replacing previous synthetic approximations. The elastic tensor is properly evaluated at 360 angles (2D) and 60×120 angular mesh (3D).

### ELATE Library Warnings

The ELATE library may emit warnings for highly anisotropic materials (e.g., AU > 100). This is expected behavior and does not indicate errors.

**Example**:
```
RuntimeWarning: divide by zero encountered in log10
```

This occurs when computing logarithmic scales for visualization and is handled gracefully.

## Interpretation Guide

### Reading Anisotropy Results

1. **Check Universal Anisotropy Index (AU)**
   - AU < 0.1: Material behaves nearly isotropically (safe to use isotropic approximations)
   - 0.1 ≤ AU < 1.0: Mild anisotropy (consider directional effects in critical applications)
   - 1.0 ≤ AU < 5.0: Significant anisotropy (directional effects important)
   - AU ≥ 5.0: Extreme anisotropy (directional effects dominate)

2. **Assess Ductility (Pugh Ratio)**
   - K/G ≤ 1.75: Brittle behavior expected
   - K/G > 1.75: Ductile behavior expected

3. **Look for Auxetic Behavior**
   - `has_auxetic_behavior = True`: Material expands laterally when stretched (rare, useful)
   - Applications: Shock absorbers, fasteners, biomedical implants

4. **Examine Directional Ranges**
   - Large anisotropy ratios (e.g., E_max/E_min > 2) indicate strong directional dependence
   - Critical for single crystal applications or textured polycrystals

### Practical Applications

- **Material Selection**: Compare AU values to choose isotropic vs. anisotropic materials
- **Design Optimization**: Orient single crystals along high-stiffness directions
- **Failure Prediction**: Low Pugh ratio materials require care in loading
- **Novel Materials**: Identify auxetic candidates for unconventional applications

## See Also

- [MechElastic GitHub](https://github.com/romerogroup/MechElastic) - Upstream ELATE library
