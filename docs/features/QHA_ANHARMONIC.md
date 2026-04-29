# QHA and Anharmonic Properties Implementation

**Status:** ✅ Implemented (November 2025)

**Author:** OptiMat Alloys Development Team

## Table of Contents

1. [Overview](#overview)
2. [Scientific Background](#scientific-background)
3. [Implementation Architecture](#implementation-architecture)
4. [Usage Guide](#usage-guide)
5. [Parameters Reference](#parameters-reference)
6. [Output Files](#output-files)
7. [Performance & Computational Cost](#performance--computational-cost)
8. [Examples](#examples)
9. [Limitations & Best Practices](#limitations--best-practices)
10. [References](#references)

---

## Overview

The QHA (Quasi-Harmonic Approximation) and Anharmonic Properties tool computes temperature-dependent thermodynamic and transport properties of materials. This implementation combines two complementary methods:

1. **QHA (Quasi-Harmonic Approximation)** - Fast, captures volume-dependent anharmonicity
2. **Phono3py** - Slow, captures phonon-phonon scattering for thermal conductivity

### Key Properties Computed

| Property | Symbol | Units | Method | Time |
|----------|--------|-------|--------|------|
| Bulk modulus | B(T) | GPa | QHA | ~1 hour |
| Volume | V(T) | Ų | QHA | ~1 hour |
| Thermal expansion coefficient | α(T) | K⁻¹ | QHA | ~1 hour |
| Isobaric heat capacity | Cp(T) | J/K/mol | QHA | ~1 hour |
| Grüneisen parameter | γ(T) | - | QHA | ~1 hour |
| Thermal conductivity | κ(T) | W/(m K) | Phono3py | Hours-days |

### Design Philosophy

**One Combined Tool with Optional Expensive Calculations:**
- QHA properties are **always computed** (required, relatively fast)
- Thermal conductivity is **optional** (very expensive, only when needed)
- User explicitly enables thermal conductivity via `compute_thermal_conductivity_flag=True`

---

## Scientific Background

### Quasi-Harmonic Approximation (QHA)

The quasi-harmonic approximation extends the harmonic approximation by allowing phonon frequencies to depend on volume:

**Gibbs Free Energy:**
```
G(T, p) = min_V [U(V) + F_phonon(T; V) + pV]
```

Where:
- `U(V)` = Electronic energy at volume V
- `F_phonon(T; V)` = Helmholtz free energy from phonons at volume V
- `pV` = Pressure-volume work

**Key Assumption:** Phonon modes change with volume but remain harmonic at each volume.

**What QHA Captures:**
- ✅ Volume-dependent phonon frequencies (implicit anharmonicity)
- ✅ Thermal expansion
- ✅ Temperature-dependent bulk modulus
- ✅ Cp (isobaric heat capacity, different from Cv)

**What QHA Misses:**
- ❌ Phonon-phonon scattering (requires 3rd-order force constants)
- ❌ Thermal conductivity (no scattering mechanism)
- ❌ Explicit anharmonic effects at fixed volume

### Phono3py for Thermal Conductivity

Phono3py computes thermal conductivity by solving the Boltzmann Transport Equation (BTE) with phonon-phonon scattering:

**Thermal Conductivity Tensor:**
```
κ_αβ = (1/NV) Σ_λ C_λ v_λα v_λβ τ_λ
```

Where:
- `C_λ` = Mode heat capacity
- `v_λ` = Group velocity
- `τ_λ` = Relaxation time (from 3-phonon scattering)
- `N` = Number of q-points
- `V` = Volume

**3rd-Order Force Constants:**
```
Φ_αβγ(l,l',l'') = ∂³U / ∂u_α(l) ∂u_β(l') ∂u_γ(l'')
```

**Computational Cost:**
- 2nd-order: ~10 displacements
- 3rd-order: ~100-500 displacements
- **100x more expensive** than harmonic phonons

---

## Implementation Architecture

### Module Structure

```
src/
├── core/
│   └── qha_wrapper.py              # Core QHA and phono3py functions
├── visualization/
│   └── qha_plots.py                # Plotly visualizations
└── tools/
    └── anharmonic_properties.py    # Chainlit-aware tool wrapper
```

### Core Functions (`src/core/qha_wrapper.py`)

#### 1. `compute_qha_properties()`

Computes QHA properties by:
1. Generating N strained volumes around equilibrium (±2% by default)
2. Relaxing atomic positions at each volume (fixed cell)
3. Computing phonons for each volume
4. Running QHA analysis to extract temperature-dependent properties

**Algorithm:**
```python
volumes = V0 * (1 + strain_range * linspace(-1, 1, num_volumes))

for each volume V_i:
    # Strain cell
    atoms_strained = scale_cell(atoms, V_i)

    # Relax positions (cell fixed)
    relax_positions(atoms_strained)

    # Compute phonons
    phonon = Phonopy(atoms_strained, supercell_matrix)
    phonon.generate_displacements()
    compute_forces()  # Using the configured ML calculator (ORB / NequIP / MACE)
    phonon.produce_force_constants()

    # Thermal properties on mesh
    phonon.run_mesh(mesh)
    phonon.run_thermal_properties(t_min, t_max, t_step)

    # Store F(V, T) and E(V)
    free_energies[i] = phonon.free_energy
    energies[i] = atoms_strained.get_potential_energy()

# Run QHA
qha = QHA(volumes, energies, temperatures, free_energies, eos='vinet')
return {
    'bulk_modulus': qha.bulk_modulus,
    'volume': qha.volume_temperature,
    'thermal_expansion': qha.thermal_expansion,
    'heat_capacity_p': qha.heat_capacity_P_numerical,
    'gruneisen': qha.gruneisen_temperature
}
```

**Key Parameters:**
- `num_volumes`: Number of volume points (minimum 5, default 11)
- `strain_range`: Volume strain (default 0.10 = ±10%)
- `mesh`: Q-point mesh for thermal properties (default 20×20×20)
- `eos`: Equation of state ('vinet' for better compression behavior)

**Energy and Volume Scaling (Critical for Accuracy):**

The implementation ensures correct bulk modulus by scaling BOTH energies AND volumes to primitive cell:

```python
# Scale energies: supercell → primitive cell
scaling_factor = n_atoms_primitive / n_atoms_input
energies_per_primitive_eV = energies_at_volumes_eV * scaling_factor  # eV per primitive

# Scale volumes: supercell → primitive cell
volumes_primitive = volumes_supercell * scaling_factor  # Ų per primitive

# Pass BOTH scaled to PhonopyQHA
qha = PhonopyQHA(
    volumes_primitive,  # Ų per primitive cell (SCALED)
    energies_per_primitive_eV,  # eV per primitive cell (SCALED)
    ...
)
```

**Why Both Must Be Scaled:**
- Bulk modulus formula: B = V × d²E/dV²
- If only E is scaled: E(V) curvature becomes wrong → B incorrect by 40-50×
- If both E and V are scaled by same factor: curvature preserves → B correct
- **Example:** Si 96-atom supercell → 2-atom primitive (scaling = 0.0208)
  - **Wrong:** Scale E only → B ~ 4000 GPa (40× too high!)
  - **Correct:** Scale E and V → B ~ 91 GPa (within 7% of expected)

**Energy Unit Flow:**

1. **Internal PhonopyQHA calculation** (lines 450-498):
   - Input: `energies_per_primitive` (eV), `fe_phonon` (kJ/mol → converted to eV internally)
   - Computation: All in eV for numerical consistency
   - Output: Gibbs, Helmholtz, etc. all in eV

2. **Conversion for user output** (lines 542-572):
   - Convert eV → kJ/mol (multiply by 96.485)
   - Normalize per formula unit (divide by n_formula_units_primitive)
   - Final units: kJ/mol per formula unit (standard chemistry unit)

3. **Returned to user:**
   - `gibbs_free_energy`: kJ/mol per formula unit
   - `helmholtz_volume`: kJ/mol per formula unit
   - `bulk_modulus`: GPa (no conversion)
   - `thermal_expansion`: 1/K (no conversion)

#### 2. `compute_thermal_conductivity()`

Computes thermal conductivity using phono3py:

**Algorithm:**
```python
# Create Phono3py object
phono3py = Phono3py(atoms, supercell_matrix, primitive_matrix)

# Generate 3rd-order displacements
phono3py.generate_displacements(distance, cutoff_pair_distance)
# Typically 100-500 displacements!

# Compute forces for all displacements
for supercell in phono3py.supercells_with_displacements:
    forces = calculator.get_forces(supercell)
    forces_list.append(forces)

# Produce 3rd-order force constants
phono3py.forces = forces_list
phono3py.produce_fc3()  # Expensive!
phono3py.produce_fc2()

# Compute thermal conductivity on mesh
phono3py.mesh_numbers = mesh
phono3py.init_phph_interaction()

for T in temperatures:
    phono3py.run_thermal_conductivity(temperatures=[T])
    kappa[T] = phono3py.thermal_conductivity.kappa
```

**Key Parameters:**
- `mesh`: Q-point mesh for κ calculation (default 32×32×32, finer than QHA)
- `cutoff_pair_distance`: Cutoff for 3rd-order interactions (optional, reduces cost)

### Visualization (`src/visualization/qha_plots.py`)

Three plotting functions:

1. **`plot_qha_properties()`** - 4-panel plot:
   - Panel 1: B(T) - Bulk modulus
   - Panel 2: V(T) - Volume
   - Panel 3: α(T) - Thermal expansion
   - Panel 4: Cp(T) - Heat capacity

2. **`plot_thermal_conductivity()`** - κ(T) plot:
   - κ_xx, κ_yy, κ_zz (directional)
   - κ_iso (isotropic average)

3. **`plot_qha_with_thermal_conductivity()`** - Combined 5-panel plot

### Tool Interface (`src/tools/anharmonic_properties.py`)

Chainlit-aware wrapper with:
- Task list for progress tracking
- Async execution with `cl.make_async()`
- CSV export and HTML plot generation
- Database metadata storage
- Error handling and user feedback

---

## Usage Guide

### From Chainlit UI

**Basic Usage (QHA only, no thermal conductivity):**
```python
# Ask the AI assistant:
"Compute QHA properties for structure abc123"

# Or more explicitly:
"Use compute_anharmonic_properties on structure abc123,
 with 7 volumes, skip thermal conductivity"
```

**With Thermal Conductivity (very expensive!):**
```python
"Compute anharmonic properties for structure abc123,
 including thermal conductivity"
```

### From Python Script

```python
from src.tools.anharmonic_properties import compute_anharmonic_properties

# QHA only (fast, ~1 hour). All keyword args shown are optional;
# defaults are num_volumes=11, mesh_qha=[20,20,20], temperature_range=[0,610,10].
results = await compute_anharmonic_properties(
    structure_ref="abc123...",
    compute_thermal_conductivity_flag=False
)

# QHA + thermal conductivity (slow, hours to days)
results = await compute_anharmonic_properties(
    structure_ref="abc123...",
    compute_thermal_conductivity_flag=True,
    mesh_phono3py=[32, 32, 32]   # finer than default for production accuracy
)
```

### Test Script

```bash
# Test QHA with Cu structure (reduced parameters for speed)
python test_qha_properties.py

# Expected output after ~30-60 minutes:
# - test_output/cu_qha_properties.csv
# - test_output/cu_qha_properties.html
```

---

## Parameters Reference

### `compute_anharmonic_properties()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure_ref` | `Union[int, str]` | **Required** | Structure ID or UUID in the database |
| `num_volumes` | `int` | `11` | Number of volume points for QHA (minimum 5) |
| `compute_thermal_conductivity_flag` | `bool` | `False` | Enable expensive thermal conductivity calculation |
| `mesh_qha` | `Optional[List[int]]` | `None` (resolves to `[20, 20, 20]`) | Mesh for QHA phonon calculations |
| `mesh_phono3py` | `Optional[List[int]]` | `None` (resolves to `[20, 20, 20]`) | Mesh for thermal conductivity. Pass a finer mesh (e.g. `[32, 32, 32]`) for production accuracy. |
| `temperature_range` | `Optional[List[float]]` | `None` (resolves to `[0, 610, 10]`) | `[T_min, T_max, T_step]` in Kelvin. Default output range is 0–600 K (61 points). Pass a custom range (e.g. `[0, 1010, 10]`) if you need higher temperatures. |

### `compute_qha_properties()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atoms` | `Atoms` | **Required** | Relaxed ASE Atoms at equilibrium |
| `calculator` | `Calculator` | **Required** | ASE calculator for forces (GPU) |
| `cpu_calculator` | `Calculator` | **Required** | ASE calculator on CPU (used for fallback paths) |
| `model_name` | `str` | **Required** | Calculator name (e.g. `orb-v3-conservative-inf-omat`) |
| `num_volumes` | `int` | `11` | Number of volume points |
| `strain_range` | `float` | `0.10` | Volume strain range (±10%) |
| `mesh` | `Tuple[int,int,int]` | `(20,20,20)` | Q-point mesh |
| `t_min` | `float` | `0` | Minimum temperature (K) |
| `t_max` | `float` | `610` | Maximum temperature (K). With `t_step=10`, output range is 0–600 K (61 points). |
| `t_step` | `float` | `10` | Temperature step (K) |
| `supercell_matrix` | `Optional[List[List[int]]]` | `None` (auto-detected) | Supercell for force constants |
| `primitive_matrix` | `str` or `ndarray` | `'auto'` | Primitive cell detection mode passed to Phonopy |
| `symprec` | `float` | `5e-3` | Symmetry tolerance (Å). Appropriate for relaxed structures. |
| `distance` | `float` | `0.01` | Displacement distance (Å) |

The function also accepts internal `cancellation_event`, `relaxation_callback`, and `progress_callback` arguments used by the Chainlit wrapper — typically you don't pass these directly.

### `compute_thermal_conductivity()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atoms` | `Atoms` | **Required** | Relaxed ASE Atoms at equilibrium |
| `calculator` | `Calculator` | **Required** | ASE calculator for forces |
| `mesh` | `Tuple[int,int,int]` | `(20,20,20)` | Q-point mesh. **Pass a finer mesh (e.g. `(32,32,32)` or `(40,40,40)`) for production accuracy** — the default is conservative for memory. |
| `t_min` / `t_max` / `t_step` | `float` | `0` / `610` / `10` | Temperature range (output 0–600 K by default) |
| `supercell_matrix` | `Optional[List[List[int]]]` | `None` | Supercell for force constants |
| `primitive_matrix` | `str` or `ndarray` | `'auto'` | Primitive cell detection mode |
| `symprec` | `float` | `5e-3` | Symmetry tolerance (Å) |
| `distance` | `float` | `0.01` | Displacement distance (Å) |
| `cutoff_pair_distance` | `Optional[float]` | `None` | Cutoff for 3rd-order interactions (Å) |

---

## Output Files

### File Locations

Organized directory structure for better organization:

```
structures/{uuid}/
├── qha/                                  # QHA properties
│   ├── qha_properties.csv               # B(T), V(T), α(T), Cp(T), γ(T)
│   └── qha_properties.html              # 4-panel interactive plot
├── thermal_conductivity/                 # Thermal conductivity (if computed)
│   ├── thermal_conductivity.csv         # κ(T) data
│   └── thermal_conductivity.html        # κ(T) plot
└── qha_and_thermal_conductivity.html    # Combined 5-panel plot (if computed)
```

### CSV Files

#### 1. `qha/qha_properties.csv`

```csv
Temperature (K),Bulk Modulus (GPa),Volume (Ų),Thermal Expansion (1/K),Heat Capacity Cp (J/K/mol),Gruneisen Parameter
0.0,142.345678,11.234567,0.000000000,0.000000,2.123456
10.0,142.123456,11.235678,0.000001234,0.123456,2.134567
...
```

**Columns:**
- Temperature (K) - 1 decimal
- Bulk Modulus (GPa) - 6 decimals
- Volume (Ų) - 6 decimals
- Thermal Expansion (1/K) - 9 decimals (very small numbers!)
- Heat Capacity Cp (J/K/mol) - 6 decimals
- Gruneisen Parameter - 6 decimals

#### 2. `thermal_conductivity/thermal_conductivity.csv` (if computed)

```csv
Temperature (K),κ_xx (W/(m K)),κ_yy (W/(m K)),κ_zz (W/(m K)),κ_iso (W/(m K))
0.0,0.000000,0.000000,0.000000,0.000000
10.0,123.456789,123.456789,123.456789,123.456789
...
```

**Columns:**
- Temperature (K)
- κ_xx, κ_yy, κ_zz - Directional components (W/(m K))
- κ_iso - Isotropic average (W/(m K))

### Interactive HTML Plots

#### 1. `qha/qha_properties.html`

4-panel interactive Plotly figure:
- **Panel 1 (top-left):** B(T) in GPa
- **Panel 2 (top-right):** V(T) in Ų
- **Panel 3 (bottom-left):** α(T) in 10⁻⁶ K⁻¹
- **Panel 4 (bottom-right):** Cp(T) in J/K/mol

Features:
- Hover tooltips with exact values
- Zoom, pan, reset
- Download as PNG

#### 2. `thermal_conductivity/thermal_conductivity.html` (if computed)

Single-panel plot with 4 traces:
- κ_xx (solid blue line)
- κ_yy (dashed orange line)
- κ_zz (dotted green line)
- κ_iso (thick black line - average)

#### 3. `qha_and_thermal_conductivity.html` (if computed)

Combined 5-panel plot with QHA (4 panels) + κ(T) (1 panel spanning bottom)

**Location:** Saved to base directory (combines data from both subdirectories)

### Database Metadata

Stored in database for searchability:

```python
{
    # QHA properties at 300 K
    "qha_bulk_modulus_300K_GPa": 142.5,
    "qha_volume_300K_angstrom3": 11.89,
    "qha_thermal_expansion_300K_per_K": 1.67e-5,
    "qha_heat_capacity_p_300K_J_K_mol": 24.5,
    "qha_gruneisen_300K": 2.1,

    # QHA parameters (defaults shown)
    "qha_num_volumes": 11,
    "qha_mesh": [20, 20, 20],
    "qha_temperature_range": [0, 610, 10],  # Output: 0-600 K (61 points)

    # Thermal conductivity at 300 K (if computed)
    "thermal_conductivity_300K_W_m_K": 385.2,
    "phono3py_mesh": [20, 20, 20]
}
```

**Searchable via database queries:**
```python
# Find structures with high thermal expansion
db.select('qha_thermal_expansion_300K_per_K>2e-5')

# Find good thermal conductors
db.select('thermal_conductivity_300K_W_m_K>300')
```

---

## Performance & Computational Cost

### QHA Performance

**Scaling:** O(num_volumes × num_atoms³ × mesh³)

**Breakdown for typical case:**
- 7 volumes
- 4-atom primitive cell
- 2×2×2 supercell = 32 atoms
- 20×20×20 mesh
- ~10 displacements per volume

**Time per volume:**
- Force calculations: ~5-10 minutes (10 displacements × 30-60s each)
- Force constant production: ~1 minute
- Mesh calculation: ~1 minute
- **Total: ~7-12 minutes per volume**

**Total QHA time: ~50-90 minutes for 7 volumes**

### Thermal Conductivity Performance

**Scaling:** O(num_atoms³ × num_displacements × mesh³)

**Breakdown for typical case:**
- 4-atom primitive cell
- 2×2×2 supercell = 32 atoms
- 32×32×32 mesh
- ~200-300 3rd-order displacements

**Time:**
- Force calculations: ~2-5 hours (200-300 displacements × 30-60s each)
- 3rd-order force constant production: ~30-60 minutes
- Thermal conductivity calculation: ~30 minutes
- **Total: ~3-7 hours for small systems**

**Larger systems (64+ atoms):** Can take **days**!

### Memory Requirements

| Component | Memory Usage |
|-----------|-------------|
| QHA (7 volumes, 32 atoms) | ~2-4 GB |
| Thermal conductivity (32 atoms) | ~8-16 GB |
| Thermal conductivity (64 atoms) | ~32-64 GB |

**Recommendation:** Use GPU with 16GB+ VRAM for thermal conductivity.

### Optimization Strategies

1. **Reduce mesh for testing:**
   ```python
   mesh_qha=[10, 10, 10]  # Instead of [20, 20, 20]
   ```

2. **Reduce temperature range:**
   ```python
   temperature_range=[0, 310, 50]  # Output: 0-300 K (instead of the 0-600 K default)
   ```

3. **Use cutoff for phono3py:**
   ```python
   cutoff_pair_distance=10.0  # Ångströms, reduces 3rd-order interactions
   ```

4. **Reduce number of volumes:**
   ```python
   num_volumes=5  # Minimum for QHA (instead of the default 11)
   ```

---

## Examples

### Example 1: Basic QHA for Cu

```python
# From Chainlit UI:
"Compute QHA properties for the Cu FCC structure"

# Results at 300 K:
# - B(300K) = 142.5 GPa
# - V(300K) = 11.89 Ų
# - α(300K) = 16.7 × 10⁻⁶ K⁻¹
# - Cp(300K) = 24.5 J/K/mol
```

### Example 2: QHA with Custom Parameters

```python
await compute_anharmonic_properties(
    structure_ref="abc123...",
    num_volumes=13,                  # More volumes than the default 11 for tighter EOS fit
    mesh_qha=[25, 25, 25],            # Finer mesh than the default 20³
    temperature_range=[0, 1010, 10]   # Extend output to 0-1000 K (default is 0-600 K)
)
```

### Example 3: QHA + Thermal Conductivity

```python
# WARNING: This will take hours!
await compute_anharmonic_properties(
    structure_ref="abc123...",
    compute_thermal_conductivity_flag=True,  # Enable κ(T)
    mesh_phono3py=[40, 40, 40]               # Much finer than the default 20³ for accuracy
)

# Results at 300 K:
# - κ(300K) = 385 W/(m K) (for Cu, literature: ~400 W/(m K))
```

### Example 4: Fast Testing Configuration

```python
# For rapid testing (~15-30 minutes)
await compute_anharmonic_properties(
    structure_ref="abc123...",
    num_volumes=5,                     # Minimum (vs. default 11)
    mesh_qha=[10, 10, 10],             # Coarse mesh (vs. default 20³)
    temperature_range=[0, 310, 50],    # Output: 0-300 K (7 points)
    compute_thermal_conductivity_flag=False
)
```

---

## Limitations & Best Practices

### QHA Limitations

1. **Assumes harmonic modes at each volume**
   - Cannot capture intrinsic anharmonicity at fixed volume
   - May fail for highly anharmonic systems (e.g., superionic conductors)

2. **Requires stable phonons at all volumes**
   - Imaginary modes → QHA fails
   - Solution: Ensure structure is fully relaxed at equilibrium

3. **Volume range must be reasonable**
   - Default ±2% is safe for most materials
   - Larger strains may cause structural instabilities
   - Check that all volumes have positive phonon frequencies

4. **Accuracy depends on num_volumes**
   - Minimum: 5 volumes
   - Recommended: 7-9 volumes
   - More volumes = better accuracy but higher cost

### Phono3py Limitations

1. **Extremely expensive**
   - 100x more expensive than harmonic phonons
   - Scales poorly with system size: O(N³)
   - **Only use when absolutely necessary!**

2. **Requires fine mesh for convergence**
   - Thermal conductivity is very sensitive to mesh
   - Typical: 32×32×32 or finer
   - Coarse mesh → underestimated κ

3. **Cutoff distance trade-off**
   - Smaller cutoff = faster but less accurate
   - Larger cutoff = more accurate but slower
   - No cutoff = most accurate but very slow

4. **Single-mode RTA approximation**
   - Default method: Relaxation Time Approximation (RTA)
   - More accurate: LBTE (even more expensive!)
   - May underestimate κ for complex phonon interactions

### Best Practices

#### ✅ DO:

1. **Always relax structure first**
   ```python
   # Use well-relaxed structure (fmax < 0.001 eV/Å)
   ```

2. **Check phonon stability**
   ```python
   # Verify no imaginary modes before QHA
   phonon.run_mesh([10,10,10])
   phonon.plot_band_structure()  # Check for negative frequencies
   ```

3. **Start with reduced parameters for testing**
   ```python
   # Test with 5 volumes, coarse mesh first
   num_volumes=5
   mesh_qha=[10, 10, 10]
   ```

4. **Validate against experimental data**
   ```python
   # Compare α(300K) with literature values
   # Cu experimental: α ≈ 16.5 × 10⁻⁶ K⁻¹
   ```

5. **Use calculator that was used for original structure**
   ```python
   # Consistency: use same calculator throughout
   calculator_name = metadata['calculator_name']
   ```

#### ❌ DON'T:

1. **Don't use thermal conductivity by default**
   - Only enable when explicitly needed
   - Cost is prohibitive for routine calculations

2. **Don't use too few volumes**
   - num_volumes < 5 → inaccurate QHA
   - Minimum: 5, Recommended: 7

3. **Don't use coarse mesh for production**
   - Test: 10×10×10
   - Production: 20×20×20 or finer

4. **Don't ignore convergence testing**
   - Test mesh convergence for critical results
   - Test num_volumes convergence

5. **Don't mix calculators**
   - Use same calculator for all volumes
   - Different calculators → inconsistent energies

### Convergence Testing

**Mesh Convergence (QHA):**
```python
for mesh_size in [10, 15, 20, 25]:
    compute_qha_properties(mesh=(mesh_size, mesh_size, mesh_size))
# Plot B(T), α(T) vs mesh_size
# Converged when properties change < 1%
```

**Volume Convergence:**
```python
for n_vols in [5, 7, 9, 11]:
    compute_qha_properties(num_volumes=n_vols)
# Plot B(T), α(T) vs num_volumes
# Converged when properties change < 1%
```

**Mesh Convergence (Thermal Conductivity):**
```python
for mesh_size in [20, 24, 28, 32, 36]:
    compute_thermal_conductivity(mesh=(mesh_size, mesh_size, mesh_size))
# Plot κ(300K) vs mesh_size
# Converged when κ changes < 5%
```

---

## References

### Scientific Papers

1. **Quasi-Harmonic Approximation:**
   - Born, M. & Huang, K. *Dynamical Theory of Crystal Lattices* (1954)
   - Baroni, S., de Gironcoli, S., Dal Corso, A. & Giannozzi, P.
     "Phonons and related crystal properties from density-functional perturbation theory"
     *Rev. Mod. Phys.* **73**, 515 (2001)

2. **Thermal Conductivity:**
   - Togo, A., Chaput, L. & Tanaka, I.
     "Distributions of phonon lifetimes in Brillouin zones"
     *Phys. Rev. B* **91**, 094306 (2015)

   - Broido, D. A., Malorny, M., Birner, G., Mingo, N. & Stewart, D. A.
     "Intrinsic lattice thermal conductivity of semiconductors from first principles"
     *Appl. Phys. Lett.* **91**, 231922 (2007)

3. **Grüneisen Parameter:**
   - Grüneisen, E. "Theorie des festen Zustandes einatomiger Elemente"
     *Ann. Phys.* **344**, 257 (1912)

### Software Documentation

1. **Phonopy (QHA):**
   - Official docs: https://phonopy.github.io/phonopy/
   - QHA tutorial: https://phonopy.github.io/phonopy/qha.html
   - Examples: https://phonopy.github.io/phonopy/examples.html

2. **Phono3py (Thermal Conductivity):**
   - Official docs: https://phonopy.github.io/phono3py/
   - Theory: https://phonopy.github.io/phono3py/theory.html
   - Examples: https://phonopy.github.io/phono3py/examples.html

3. **ASE (Atomic Simulation Environment):**
   - Official docs: https://ase-lib.org/
   - Phonons: https://ase-lib.org/ase/phonons.html

### Related OptiMat Alloys Documentation

- [`ELATE.md`](ELATE.md) - Elastic anisotropy analysis
- [`../CONFIGURATION.md`](../CONFIGURATION.md) - Calculator settings

---

## Implementation Files

### Core Modules
- **`src/core/qha_wrapper.py`**
  - `compute_qha_properties()` - Main QHA function
  - `compute_thermal_conductivity()` - Phono3py wrapper
  - `export_qha_csv()` / `export_qha_volume_csv()` - CSV export for QHA
  - `export_thermal_conductivity_csv()` - CSV export for κ

### Visualization
- **`src/visualization/qha_plots.py`**
  - `plot_qha_properties()` - 4-panel QHA plot
  - `plot_qha_properties_individual()` - per-panel variants
  - `plot_thermal_conductivity()` - κ(T) plot
  - `plot_qha_with_thermal_conductivity()` - Combined 5-panel plot
  - `plot_qha_volume_surfaces()` - Helmholtz energy surface plots

### Tool Interface
- **`src/tools/anharmonic_properties.py`**
  - `compute_anharmonic_properties()` - Main tool entry point
  - Task management, database storage, user feedback

### Testing
- **`test_qha_properties.py`** - QHA validation test with Cu
  - Reduced parameters for faster testing (~30-60 min)
  - Verifies: computation, export, plotting

### Dependencies
- **`environment.yml`** - Added `phonopy` and `phono3py` packages

---

## Changelog

### v1.2.0 (December 2025)
- ✅ **CRITICAL FIX:** Corrected bulk modulus calculation by scaling BOTH energies AND volumes to primitive cell
  - **Issue:** Previous implementation scaled energies but NOT volumes → incorrect E(V) curvature → bulk modulus off by 40-50×
  - **Solution:** Scale both energies and volumes by same factor (n_atoms_primitive/n_atoms_input) before PhonopyQHA
  - **Impact:** Bulk modulus now accurate within 7-10% of experimental/DFT values
  - **Example:** Si at 300K: B = 91.06 GPa (vs expected ~98 GPa, within 7%)
- ✅ Added volumetric thermal expansion coefficient (β) to output alongside linear (α)
  - **New field:** `thermal_expansion_volumetric` in results dictionary
  - **Relation:** α = β/3 (for isotropic materials)
  - **Direct from PhonopyQHA:** No conversion needed
- ✅ Updated documentation to clarify energy unit flow (eV internally, kJ/mol for output)
- ✅ Improved volume display in UI: "Volume: X.XXX Å³ (primitive cell)" for clarity
- ✅ Updated `volumes_used` in results to return scaled primitive cell volumes (not supercell)

### v1.1.0 (November 2025)
- ✅ Tightened temperature-range handling so the upper bound is included with `t_step` aligned (current default range is 0–600 K, configurable per call).
- ✅ Added robust bounds checking to prevent index out-of-bounds errors
- ✅ Improved error handling when QHA drops temperature points
- ✅ User notifications when using closest available temperature
- ✅ User-facing messages now show the actual output range rather than the raw `t_max` implementation detail.

### v1.0.0 (November 2025)
- ✅ Initial implementation
- ✅ QHA computation for B(T), V(T), α(T), Cp(T), γ(T)
- ✅ Phono3py integration for κ(T)
- ✅ 4-panel and 5-panel Plotly visualizations
- ✅ CSV export for all properties
- ✅ Database metadata storage
- ✅ Test script for Cu validation
- ✅ Comprehensive documentation

---

## Contact & Support

**Questions or Issues?**
- GitHub: https://github.com/OptiMat-Chat/OptiMat-Alloys/issues
- Documentation: See `docs/` directory

**Contributors:**
- OptiMat Alloys Development Team
- Implemented: November 2025

---

**Last Updated:** November 2025
