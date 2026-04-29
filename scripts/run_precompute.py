#!/usr/bin/env python3
"""
Run reference data precomputation for all supported elements.

This script will:
1. Load supported elements from element testing results (117 elements)
2. Relax each element in 5 structures (sc, bcc, fcc, hcp, diamond)
3. Save lattice constants and energies to reference data files

Expected runtime: 8-16 hours per calculator (117 elements × 5 structures)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.reference_data import precompute_and_save
from src.storage.cache import get_reference_cache


def main():
    """Run precomputation for both calculators."""

    calculators = [
        "orb-v3-direct-20-omat",
        "orb-v3-conservative-inf-omat"
    ]

    print("=" * 80)
    print("REFERENCE DATA PRECOMPUTATION - 117 ELEMENTS")
    print("=" * 80)
    print()
    print("This will precompute lattice constants and energies for all 117 supported elements.")
    print("Estimated time: 8-16 hours per calculator (32 hours total)")
    print()

    for calc_idx, calculator in enumerate(calculators, 1):
        print(f"\n{'=' * 80}")
        print(f"CALCULATOR {calc_idx}/{len(calculators)}: {calculator}")
        print(f"{'=' * 80}\n")

        # Create cache
        cache = get_reference_cache(calculator)

        # Run precomputation (will automatically use 117 elements from element testing)
        try:
            precompute_and_save(
                hydrostatic_cell_relaxation=True,
                optimizer="FIRE",
                fmax=0.005,  # Tighter convergence than default
                calculator=calculator,
                cache=cache
            )
            print(f"\n✓ Completed reference data for {calculator}")
        except Exception as e:
            print(f"\n✗ Error during precomputation for {calculator}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 80)
    print("\nReference data files updated:")
    print("  - data/reference/lattice_constants_orb_v3_direct_20_omat.json")
    print("  - data/reference/energies_per_atom_orb_v3_direct_20_omat.json")
    print("  - data/reference/lattice_constants_orb_v3_conservative_inf_omat.json")
    print("  - data/reference/energies_per_atom_orb_v3_conservative_inf_omat.json")
    print("\nBackup of original 48-element data:")
    print("  - data/reference/backup_48_elements_20251024/")
    print()


if __name__ == "__main__":
    main()
