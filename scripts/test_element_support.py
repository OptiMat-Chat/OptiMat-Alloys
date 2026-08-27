#!/usr/bin/env python
"""
Element support testing — stage 1 of the reference-data pipeline.

Sweeps the periodic table for one calculator, building each element in a few
simple structures and recording whether the calculator produces an energy at
all. Results land in data/element_support/element_support_<calculator>.json.

Stage 2 (scripts/run_precompute.py) then takes the supported elements and
relaxes them properly to produce the reference energies and lattice constants
used for formation energies.

Usage:
    python scripts/test_element_support.py --calculator mace-mpa-0-medium
    python scripts/test_element_support.py --calculator orb-v3-direct-20-omat --device cpu
    python scripts/test_element_support.py --calculator mace-omat-0-small --summary

Runtime is 2-4 hours on a GPU. Results save incrementally, so interrupting and
rerunning is safe.

NequIP models are not offered here. This script loads calculators through
load_calculator(), which serves the main environment only; NequIP runs in a
separate environment via calculator_service.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import get_args, Union

from src.core.calculators import SupportedModel
from src.core.element_testing import test_all_elements, print_support_summary


def _supported_calculators():
    """Flatten SupportedModel (a Union of Literals) into a list of names.

    Derived from the registry rather than hardcoded so this script cannot
    drift out of sync with the calculators the app actually loads.
    """
    names = []
    for member in get_args(SupportedModel):
        names.extend(get_args(member) or [member])
    return names


def main():
    """Run element support testing."""
    parser = argparse.ArgumentParser(
        description="Test element support for universal ML potential calculators (ORB, MACE)"
    )

    parser.add_argument(
        "--calculator",
        type=str,
        required=True,
        choices=_supported_calculators(),
        help="Calculator to test"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for testing (default: cuda)"
    )

    parser.add_argument(
        "--structures",
        type=str,
        nargs="+",
        default=["fcc", "bcc", "sc"],
        choices=["sc", "bcc", "fcc", "hcp", "diamond"],
        help="Structures to test (default: fcc bcc sc)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of existing results instead of running tests"
    )

    args = parser.parse_args()

    # If summary mode, print existing results
    if args.summary:
        try:
            print_support_summary(args.calculator)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # Run tests
    print(f"\n{'='*70}")
    print(f"ELEMENT SUPPORT TESTING")
    print(f"{'='*70}")
    print(f"Calculator: {args.calculator}")
    print(f"Device: {args.device}")
    print(f"Structures: {', '.join(args.structures)}")
    print(f"{'='*70}\n")

    print("⚠️  WARNING: This will test 119 elements and may take 2-4 hours!")
    print("             Results are saved incrementally to avoid data loss.")
    print("             You can safely interrupt and resume later.\n")

    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Run tests
    try:
        results = test_all_elements(
            calculator=args.calculator,
            structures=args.structures,
            device=args.device,
            output_file=args.output,
            max_workers=1  # Sequential for GPU safety
        )

        # Print summary
        print("\n")
        print_support_summary(args.calculator)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results have been saved.")
        sys.exit(1)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
