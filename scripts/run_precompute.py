#!/usr/bin/env python3
"""
Reference data precomputation — stage 2 of the reference-data pipeline.

Relaxes every supported element in 5 structures (sc, bcc, fcc, hcp, diamond)
for one or more calculators, and saves the resulting lattice constants and
energies to data/reference/. These are the reference energies used to compute
formation energies.

Stage 1 (scripts/test_element_support.py) determines which elements a
calculator supports. If its results are missing, this script falls back to the
48 elements that already have reference data.

Usage:
    python scripts/run_precompute.py --calculator orb-v3-direct-20-omat
    python scripts/run_precompute.py --calculator mace-mpa-0-medium orb-v3-direct-20-omat

Runtime is roughly 8-16 hours per calculator, so --calculator is required:
running this by accident is expensive.

NequIP models are not offered here. This script loads calculators through
load_calculator(), which serves the main environment only; NequIP runs in a
separate environment via calculator_service.
"""

import argparse
import sys
from pathlib import Path
from typing import get_args

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.calculators import SupportedModel
from src.core.reference_data import precompute_and_save
from src.storage.cache import get_reference_cache


def _supported_calculators():
    """Flatten SupportedModel (a Union of Literals) into a list of names.

    Derived from the registry rather than hardcoded so this script cannot drift
    out of sync with the calculators the app actually loads.
    """
    names = []
    for member in get_args(SupportedModel):
        names.extend(get_args(member) or [member])
    return names


def main():
    """Run reference data precomputation for the requested calculators."""
    parser = argparse.ArgumentParser(
        description="Precompute reference lattice constants and energies (ORB, MACE)"
    )
    parser.add_argument(
        "--calculator",
        type=str,
        nargs="+",
        required=True,
        choices=_supported_calculators(),
        help="One or more calculators to precompute",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.005,
        help="Force convergence criterion (default: 0.005, tighter than the app default)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="FIRE",
        help="ASE optimizer to use (default: FIRE)",
    )

    args = parser.parse_args()
    calculators = args.calculator

    print("=" * 80)
    print("REFERENCE DATA PRECOMPUTATION")
    print("=" * 80)
    print()
    print("Relaxes every supported element in 5 structures (sc, bcc, fcc, hcp, diamond).")
    print("The element count is per-calculator: it comes from stage 1 results if")
    print("present, otherwise the 48 elements that already have reference data.")
    print(f"Estimated time: 8-16 hours per calculator ({len(calculators)} requested)")
    print()

    completed = []
    for calc_idx, calculator in enumerate(calculators, 1):
        print(f"\n{'=' * 80}")
        print(f"CALCULATOR {calc_idx}/{len(calculators)}: {calculator}")
        print(f"{'=' * 80}\n")

        # Create cache
        cache = get_reference_cache(calculator)

        # Run precomputation (element list resolved inside precompute_and_save)
        try:
            precompute_and_save(
                hydrostatic_cell_relaxation=True,
                optimizer=args.optimizer,
                fmax=args.fmax,
                calculator=calculator,
                cache=cache,
            )
            print(f"\n✓ Completed reference data for {calculator}")
            completed.append(calculator)
        except Exception as e:
            print(f"\n✗ Error during precomputation for {calculator}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 80)

    if completed:
        print("\nReference data files updated:")
        for calculator in completed:
            safe_name = calculator.replace("-", "_")
            print(f"  - data/reference/lattice_constants_{safe_name}.json")
            print(f"  - data/reference/energies_per_atom_{safe_name}.json")
    else:
        print("\nNo calculators completed successfully.")

    failed = [c for c in calculators if c not in completed]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")

    print()
    return 0 if completed else 1


if __name__ == "__main__":
    sys.exit(main())
