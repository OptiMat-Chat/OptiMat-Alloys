# Element Support Testing Guide

This guide explains how to test and expand element support in OptiMat Alloys for different universal ML potential calculators (ORB, MACE, NequIP).

## Overview

OptiMat Alloys can theoretically work with any element in the periodic table, but practical support depends on the training data of the calculator models. The element testing framework systematically tests all 118 elements to determine which ones are actually supported.

## Pre-Computed Test Results

> **Note:** The element-testing harness (`scripts/test_element_support.py`) has been removed from this repo. The pre-computed results below are checked into `data/element_support/` and remain canonical. To regenerate, recover the script from git history (`git log --diff-filter=D --name-only -- scripts/test_element_support.py`).

For each element (H through Og), the harness creates a simple FCC/BCC/SC unit cell with a radii-based lattice constant, attempts a single-point energy calculation, and records success/failure. Each calculator is tested across all 118 elements × 3 structures.

### Output Files

Three calculator-specific JSON files are checked in:

```
data/element_support/element_support_orb_v3_conservative_inf_omat.json
data/element_support/element_support_orb_v3_direct_20_omat.json
data/element_support/element_support_mace_mpa_0_medium.json
```

The `mace-omat-0-small` and `mace-omat-0-medium` calculators (also wired up in the UI; see CONFIGURATION.md) have not yet been independently tested for element support. As a working assumption they share the MACE-MPA-0 element coverage, since they come from the same architecture family. A NequIP element-support JSON has not been generated either — the NequIP coverage table below is sourced from upstream documentation.

### File Format

```json
{
  "_metadata": {
    "calculator": "orb-v3-direct-20-omat",
    "tested_at": "2025-10-24T21:10:44",
    "total_elements": 118,
    "supported_count": 117,
    "in_progress": false,
    "version": "1.0"
  },
  "Cu": {
    "supported": true,
    "structures": {
      "fcc": {
        "supported": true,
        "energy": -3.721,
        "lattice_constant": 3.615,
        "num_atoms": 4
      },
      "bcc": {...},
      "sc": {...}
    }
  },
  "He": {
    "supported": false,
    "structures": {
      "fcc": {
        "supported": false,
        "error": "Element not in training data"
      }
    }
  }
}
```

## Using Element Support Data

### In Python Code

```python
from src.core.element_testing import get_supported_elements, load_element_support

# Get list of supported elements
elements = get_supported_elements("orb-v3-direct-20-omat")
print(f"Supported elements: {len(elements)}")

# Get detailed support info
support = load_element_support("orb-v3-direct-20-omat")
if support.get("Cu"):
    print("Copper is supported!")
```

### In Reference Data Generation

The `precompute_and_save()` function automatically uses element support data:

```python
from src.core.reference_data import precompute_and_save

# Will use supported elements from testing results
precompute_and_save(
    calculator="orb-v3-direct-20-omat",
    fmax=0.005
)
```

## Atomic Radii Database

### Overview

The atomic radii database (`data/radii/atomic_radii.json`) provides comprehensive radii data for all 119 elements, combining:
- **Metallic radii** (48 core metals) - most accurate for metallic bonding
- **Covalent radii** (119 elements from ASE) - fallback for other 69 elements and all nonmetals/semiconductors

### Database Schema

```json
{
  "Cu": {
    "atomic_number": 29,
    "atomic_mass": 63.546,
    "covalent_radius": 1.32,
    "metallic_radius": 1.32,
    "preferred_radius": 1.32,
    "radius_type": "metallic"
  },
  "Si": {
    "atomic_number": 14,
    "atomic_mass": 28.085,
    "covalent_radius": 1.11,
    "metallic_radius": null,
    "preferred_radius": 1.11,
    "radius_type": "covalent"
  }
}
```

### Using the Radii Database

```python
from src.core.radii_database import get_radii_database

db = get_radii_database()

# Get preferred radius (metallic if available, else covalent)
r_cu = db.get_radius('Cu')  # Returns 1.32 Å

# Get specific radius type
r_cov = db.get_radius('Cu', radius_type='covalent')

# Check if metallic radius available
if db.has_metallic_radius('Cu'):
    print("Copper has metallic radius data")

# Get complete element info
info = db.get_element_info('Cu')
print(info['atomic_number'])  # 29
```

### Regenerating the Database

```python
from src.core.radii_database import RadiiDatabase

# Rebuild from sources and save
RadiiDatabase.save_database("data/radii/atomic_radii.json")
```

## Validated Element Support

### Test Results Summary (October 2025)

Comprehensive testing of all 118 elements (H through Og) reveals extensive ORB model support:

- **Total Validated**: 117/118 elements (99.2%)
- **Both calculators show identical support** (orb-v3-direct-20-omat and orb-v3-conservative-inf-omat)
- **Only unsupported**: Oganesson (Og, Z=118) - fails with "Class values must be smaller than num_classes"

### Supported Elements by Category

**Metals (48 original validated elements):**
- Li, Be, Na, Mg, Al, K, Ca, Rb, Sr, Cs, Ba (alkali & alkaline earth)
- Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn (3d transition metals)
- Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd (4d transition metals)
- Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg (5d transition metals)
- Ga, In, Sn, Tl, Pb, Bi (post-transition metals)
- Si, Ge (metalloids)

**Newly Validated (+69 elements):**

- **Nonmetals & Metalloids (15)**: H, B, C, N, O, F, P, S, Cl, As, Se, Br, Te, I, At
  - Surprisingly, bulk metallic structures work even for molecular elements

- **Noble Gases (6)**: He, Ne, Ar, Kr, Xe, Rn
  - Tested successfully despite lacking natural metallic structures

- **Lanthanides (15)**: La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu
  - Complete lanthanide series supported (including radioactive Pm)

- **Actinides (15)**: Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr
  - Complete actinide series supported (all radioactive elements)

- **Transactinides (14)**: Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts
  - Superheavy synthetic elements (Z=104-117) all validated

- **Additional Post-Transition Metals (4)**: Sb, Po, Ra, Fr

### Key Findings

1. **Nonmetals work**: Light nonmetals (H, C, N, O, F) successfully tested despite being molecular in nature
2. **Noble gases work**: All noble gases validated despite lacking metallic structures
3. **Radioactive elements work**: Tc, Pm, Po, At, Fr, Ra, Ac all supported
4. **Transuranium elements work**: U, Np, Pu, Am and heavier actinides all validated
5. **Superheavy elements work**: Transactinides Rf through Ts (Z=104-117) all validated
6. **Only Oganesson fails**: Z=118 not in ORB training data (model limitation)

### Why Previous Predictions Were Wrong

Original documentation predicted many elements would be unsupported based on assumptions:
- **Assumed**: Noble gases lack metallic structures → Would fail
- **Reality**: ORB models handle forced metallic configurations
- **Assumed**: Light nonmetals are molecular → Would fail
- **Reality**: Models successfully calculate bulk energies for artificial structures
- **Assumed**: Transuranics lack DFT training data → Would fail
- **Reality**: OMat24 training dataset includes these elements

The ORB models demonstrate remarkable generalization across the periodic table.

## Multi-Calculator Element Support Comparison

OptiMat Alloys supports three universal ML potential families with varying element coverage:

| Calculator Family | Supported Elements | Coverage | Source |
|-------------------|-------------------|----------|--------|
| **ORB** (Orbital Materials) | 117/118 | 99.2% | Internal testing (Oct 2025) |
| **MACE** (Foundation) | 89/118 | 75.4% | **Internal testing (Jan 2026)** |
| **NequIP** (Foundation) | 86 | 72.9% | nequip.net documentation |

### ORB Models (117 elements)

ORB models trained on OMat24 (100M+ DFT calculations) support nearly all elements:
- **Supported**: H (Z=1) through Ts (Z=117)
- **Unsupported**: Only Oganesson (Og, Z=118)
- **Notable**: Includes all lanthanides, actinides, and superheavy elements (Rf-Ts)

### MACE Models (89 elements) ✓ Validated

MACE foundation models support **89/118 elements** (75.4% coverage). This has been **empirically validated** through systematic testing (January 2026).

**Models and training data**:
- `mace-mpa-0-medium`: MPTrj + sAlex (general materials, Matbench SOTA)
- `mace-omat-0-small/medium`: OMAT dataset (best for phonon calculations)

**Validated Supported Elements (89 total)**:

Ac, Ag, Al, Ar, As, Au, B, Ba, Be, Bi, Br, C, Ca, Cd, Ce, Cl, Co, Cr, Cs, Cu, Dy, Er, Eu, F, Fe, Ga, Gd, Ge, H, He, Hf, Hg, Ho, I, In, Ir, K, Kr, La, Li, Lu, Mg, Mn, Mo, N, Na, Nb, Nd, Ne, Ni, Np, O, Os, P, Pa, Pb, Pd, Pm, Pr, Pt, Pu, Rb, Re, Rh, Ru, S, Sb, Sc, Se, Si, Sm, Sn, Sr, Ta, Tb, Tc, Te, Th, Ti, Tl, Tm, U, V, W, Xe, Y, Yb, Zn, Zr

**Key Finding**: **Noble gases ARE supported** (He, Ne, Ar, Kr, Xe) - contrary to previous inference based on training data analysis. The MACE models successfully generalize to these elements despite minimal training examples.

**Not supported (29 elements)**:

- **Late actinides (Z≥95)**: Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr
- **Transactinides (Z≥104)**: Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, Og
- **Radioactive elements**: Po, At, Rn, Fr, Ra

**Test details**:
- Tested: `mace-mpa-0-medium` on CUDA
- Structures: FCC, BCC, SC
- Results file: `data/element_support/element_support_mace_mpa_0_medium.json`
- Date: January 2026

**References**:
- [MACE Docs](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html) - States "89 elements"
- [Matbench Discovery MPtrj](https://matbench-discovery.materialsproject.org/data/mptrj) - Training data element counts

### NequIP Models (86 elements)

NequIP foundation models from [nequip.net](https://nequip.net) cover elements H through Pu:

**Supported Elements (86 total)**:
H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Ac, Th, Pa, U, Np, Pu

**Not supported (Z=95-118)**: Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr, Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, Og

- **Training data**: OMat24 + MPtrj + sAlex (112.8M structures for oam models)
- **Best for**: Highest accuracy calculations (`nequip-oam-xl` has F1=0.906)
- **Models**: `nequip-oam-l`, `nequip-oam-xl`, `nequip-mp-l`

### Practical Implications

**For most materials research**: All three calculator families support the commonly used elements (main group, transition metals, lanthanides).

**For actinide/transactinide research**: Use ORB models, which uniquely support Am-Ts (Z=95-117).

**For calculator benchmarking**: Use the `recompute_structure` tool to compare results across calculators on the same atomic configuration.

## Interpreting Test Results

### Success Criteria

An element is marked as "supported" if:
- Energy calculation succeeds in at least one structure
- No CUDA errors or memory issues
- Reasonable energy value (not NaN or inf)

### Common Failure Modes

1. **"Element not in training data"**
   - ORB model wasn't trained on this element
   - Cannot be fixed without retraining

2. **"CUDA out of memory"**
   - Try with `--device cpu`
   - Or test with smaller structure (sc instead of diamond)

3. **"Cannot determine atomic radius"**
   - Element not in radii database
   - Add manually to `data/radii/atomic_radii.json`

4. **"Convergence failure"**
   - Element may need special treatment
   - Try different structure or lattice constant

## Reference Data Precomputation

Once element-support data is available, reference energies and lattice constants can be precomputed for any new calculator using `precompute_and_save()`:

```python
from src.core.reference_data import precompute_and_save
from src.storage.cache import get_reference_cache

cache = get_reference_cache('orb-v3-direct-20-omat')
precompute_and_save(
    calculator='orb-v3-direct-20-omat',
    fmax=0.005,
    cache=cache
)
```

**Performance**: ~5 structures × 100–500 optimization steps per element, ~8–16 hours for 117 elements. Run overnight on GPU.

If precomputation fails for an element, check that the element is marked supported in the relevant `data/element_support/element_support_*.json` file and that the radii database (`data/radii/atomic_radii.json`) includes it.

## Best Practices

1. **Save results to version control** — the JSON files are lightweight.
2. **Add new reference data when introducing a new calculator** — required for that calculator to work.
3. **Document supported elements** in the project README.
4. **Validate critical elements** before large computational campaigns.

## API Reference

```python
# Element support data (src/core/element_testing.py)
load_element_support(calculator, base_dir)
get_supported_elements(calculator, base_dir)

# Radii database (src/core/radii_database.py)
RadiiDatabase.get_radius(element, radius_type)
RadiiDatabase.has_metallic_radius(element)
RadiiDatabase.build_database()
RadiiDatabase.save_database(output_file)
get_radii_database(database_file=None)
```

## Further Reading

- [CONFIGURATION.md](CONFIGURATION.md) - Calculator configuration
