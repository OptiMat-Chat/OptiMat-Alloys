"""
Database Search Tool

Searches global database for existing alloy structures with composition-aware filtering.
"""

from typing import Annotated, List, Dict, Literal, Optional
import chainlit as cl
import math
from datetime import datetime

# Utilities
from src.utils.composition_parser import parse_composition_string

# Storage modules
from src.storage.database import create_structure_database


def format_composition_display(composition: str) -> str:
    """
    Format composition string for display in Chainlit Markdown tables.

    Removes decimal places from atomic percentages for cleaner display.
    Example: "Cu50.0Ag50.0" -> "Cu50Ag50"

    Args:
        composition: Composition string like "Cu50.0Ag50.0"

    Returns:
        Cleaned composition string like "Cu50Ag50"

    Example:
        >>> format_composition_display("Cu50.0Ag50.0")
        "Cu50Ag50"
    """
    import re

    # Pattern: Element symbol followed by number with optional decimal
    pattern = r'([A-Z][a-z]?)(\d+)\.0'

    # Replace "50.0" with "50" (remove trailing .0)
    return re.sub(pattern, r'\1\2', composition)


@cl.step(type="tool")  # type: ignore
async def search_database(
    structure_ref: Annotated[Optional[str], "Direct lookup by ID (e.g. '111') or UUID — bypasses all filters"] = None,
    elements: Annotated[Optional[List[str]], "Element symbols (or use composition_string)"] = None,
    target_fractions: Annotated[Optional[List[float]], "Atomic fractions (or use composition_string)"] = None,
    composition_string: Annotated[str, "e.g., 'Ag75Cu25' or 'Cu-Ag'"] = "",
    structure: Annotated[str, "Crystal structure filter (empty=none)"] = "",
    min_atoms: Annotated[int, "Min atoms (0=none)"] = 0,
    max_atoms: Annotated[int, "Max atoms (0=none)"] = 0,
    composition_tolerance: Annotated[float, "Composition tolerance as a FRACTION, not a percent: 0.05 means +/-5 at.%. Default 0.05 is right for almost every query — OMIT this argument unless the user asks for a wider or narrower composition window. Do NOT pass 5 for '5%'; values above 0.5 are rejected because they would match every composition."] = 0.05,
    include_higher_order: Annotated[bool, "Allow elements beyond those named. True when the user asks for alloys CONTAINING an element — 'Ir-containing alloys', 'alloys with Ti', 'Cu-based alloys', 'anything with Ag and Cu'. False when they name an exact system — 'Cu-Ag', 'CoCrFeNi', 'pure Cu' — where extra elements would be wrong."] = False,
    stable_only: Annotated[bool, "Structurally stable only (≥90% match)"] = False,
    phonon_stable_only: Annotated[bool, "Dynamically stable (no imaginary modes)"] = False,
    has_qha_data: Annotated[bool, "Has QHA data"] = None,
    calculator_name: Annotated[Optional[str], "Calculator filter: 'mace', 'orb', 'nequip', or full name"] = None,
    sort_by: Annotated[Literal["auto", "newest", "oldest", "composition_proximity", "relevance"], "Sort order"] = "auto",
    limit: Annotated[int, "Max results (default 10, max 50)"] = 10
) -> Annotated[Dict, "Search results with UUIDs and metadata."]:
    """
    Search global database for existing alloy structures with composition-aware filtering.

    Use structure_ref for direct ID/UUID lookup (bypasses all filters).
    Use composition_string (e.g., 'Cu-Ag', 'Ag75Cu25') for composition queries.
    Sort: 'auto' (relevance scoring), 'newest', 'oldest', 'composition_proximity', 'relevance'.

    Returns structure UUIDs and metadata.
    """

    db = create_structure_database()

    # Enforce maximum limit to prevent UI flooding
    MAX_LIMIT = 50
    # Above this many results, QHA curves are downsampled to keep the payload sane.
    QHA_FULL_CURVE_MAX_RESULTS = 3
    if limit > MAX_LIMIT:
        limit = MAX_LIMIT

    # Initialised before the branch below: the direct-lookup path skips the
    # composition block where these are normally assigned, and the search
    # confirmation message reads them unconditionally further down.
    searched_elements: List[str] = []
    searched_fractions = None

    # Direct ID/UUID lookup — bypass all filters
    if structure_ref is not None and str(structure_ref).strip():
        try:
            structure_uuid = db.resolve_to_uuid(str(structure_ref).strip())
            row = db._get_row_by_uuid(structure_uuid)
            results = [row]
            sort_method = "direct_lookup"
        except (KeyError, ValueError):
            await cl.Message(content=f"Structure '{structure_ref}' not found in database.").send()
            return {
                "total_found": 0, "num_returned": 0,
                "structure_uuids": [], "results": [],
                "sort_method": "direct_lookup"
            }
    else:
        # Composition-based search path (existing logic)
        sort_method = None  # Will be determined below

    # Parse composition_string if provided (takes precedence over elements/target_fractions)
    if sort_method != "direct_lookup" and composition_string:
        parsed = parse_composition_string(composition_string)
        if parsed:
            elements = parsed.elements
            # Use fractions only if they were actually parsed (not just equal distribution for elements-only input)
            if parsed.format_detected != "elements only (equal parts)":
                target_fractions = parsed.fractions
            else:
                # For elements-only input like "Cu-Ag", don't set target_fractions
                # This allows finding all Cu-Ag structures regardless of composition
                target_fractions = None
        else:
            # If parsing fails, treat as warning but don't raise error for search
            await cl.Message(content=f"⚠️ Could not parse '{composition_string}'. Proceeding with other filters.").send()

    # Composition-based search path (skipped for direct_lookup)
    if sort_method != "direct_lookup":
        # Convert None to empty collections
        if elements is None:
            elements = []
        if target_fractions is None:
            target_fractions = []

        # Store what was actually searched (for model confirmation in response)
        # This prevents the model from hallucinating about which elements were queried
        searched_elements = list(elements) if elements else []
        searched_fractions = list(target_fractions) if target_fractions else None
        query_type = "composition_specific" if target_fractions else ("element_only" if elements else "all_structures")

        # Validate target_fractions if provided
        if target_fractions:  # Non-empty list
            if not elements:
                raise ValueError("target_fractions requires elements parameter to be specified.")
            if len(elements) != len(target_fractions):
                raise ValueError(f"Length of elements ({len(elements)}) must equal length of target_fractions ({len(target_fractions)}).")

            # Convert to dict internally for range filtering
            target_composition = dict(zip(elements, target_fractions))
        else:
            target_composition = {}

        # Note: include_higher_order is only meaningful when elements are specified
        # When no elements specified, this parameter is ignored (no filter added)

        # Build query filters (ASE database syntax)
        filters = []

        # Element-based filtering (contains all specified elements)
        if elements:  # Non-empty list
            # Require all specified elements present
            for elem in elements:
                filters.append(f'{elem}_fraction>0')

            # Restrict to exactly these elements (no additional elements)
            # Note: Inverted logic for Ollama compatibility. Ollama sends false for
            # unspecified bools, so we use include_higher_order=False as default.
            if not include_higher_order:
                filters.append(f'num_elements={len(elements)}')

        # Guard the tolerance before it reaches the query. A percent-shaped value
        # (e.g. 5 or 10 for "5%"/"10%") would widen every window to [0, 1] and
        # silently void the composition filter — the search would still succeed,
        # just against every composition. Fail loudly instead.
        if composition_tolerance is not None and not (0.0 <= composition_tolerance <= 0.5):
            await cl.Message(content=(
                f"⚠️ `composition_tolerance={composition_tolerance}` is out of range. "
                f"It is a FRACTION (0.05 = ±5 at.%), not a percent. "
                f"Using the default 0.05 instead."
            )).send()
            composition_tolerance = 0.05

        # Composition range filtering
        if target_composition:  # Non-empty dict
            for elem, frac in target_composition.items():
                min_frac = max(0.0, frac - composition_tolerance)
                max_frac = min(1.0, frac + composition_tolerance)
                filters.append(f'{elem}_fraction>={min_frac}')
                filters.append(f'{elem}_fraction<={max_frac}')

        # Structure filtering
        if structure:  # Non-empty string
            filters.append(f'target_structure={structure}')

        # Atom count filtering
        if min_atoms > 0:  # Greater than 0
            filters.append(f'optimized_num_atoms>={min_atoms}')
        if max_atoms > 0:  # Greater than 0
            filters.append(f'optimized_num_atoms<={max_atoms}')

        # Stability filtering (structural - PTM based)
        # Note: Only filter when explicitly True. Ollama passes False instead of None,
        # so we treat False as "no filter" for compatibility.
        if stable_only is True:
            filters.append('is_structurally_stable=1')  # SQLite stores booleans as integers (0/1)

        # Finite temperature (QHA) data filtering
        # Note: Only filter when True. Ollama passes False instead of None,
        # so we treat False as "no filter" for compatibility.
        if has_qha_data is True:
            filters.append('has_qha_data=1')

        # Phonon/dynamical stability filtering (no imaginary modes from QHA)
        # Note: Only filter when explicitly True for Ollama compatibility
        if phonon_stable_only is True:
            filters.append('has_qha_data=1')
            filters.append('qha_dynamically_stable=1')

        # Calculator filtering (supports shorthands like 'mace', 'orb', 'nequip')
        if calculator_name:
            CALCULATOR_SHORTHANDS = {
                "mace": "mace-omat-0-medium",
                "mace-omat": "mace-omat-0-medium",
                "mace-mpa": "mace-mpa-0-medium",
                "mace-omat-small": "mace-omat-0-small",
                "orb": "orb-v3-conservative-inf-omat",
                "orb-conservative": "orb-v3-conservative-inf-omat",
                "nequip": "nequip-oam-l",
                "nequip-xl": "nequip-oam-xl",
                "nequip-mp": "nequip-mp-l",
            }
            resolved = CALCULATOR_SHORTHANDS.get(calculator_name.lower().strip(), calculator_name)
            filters.append(f'calculator_name={resolved}')

        # Execute query
        query_string = ','.join(filters) if filters else None

        try:
            if query_string:
                results = list(db._get_db().select(query_string))
            else:
                results = list(db._get_db().select())
        except Exception as e:
            await cl.Message(content=f"❌ Database query failed: {e}").send()
            return {
                "total_found": 0, "num_returned": 0, "structure_uuids": [], "results": [], "sort_method": "auto"
            }

        # Determine sort method (auto-select based on query type)
        if sort_by == "auto":
            sort_method = "relevance"
        else:
            sort_method = sort_by

        # Apply sorting
        if results:
            if sort_method == "relevance":
                # Relevance scoring: property completeness + composition proximity + recency
                has_fractions = bool(target_fractions)

                # Normalize recency
                if len(results) > 1:
                    ids = [r.id for r in results]
                    min_id, max_id = min(ids), max(ids)
                    id_range = max_id - min_id or 1
                else:
                    min_id, id_range = (results[0].id if results else 0), 1

                # Normalize composition distance
                if has_fractions:
                    distances = []
                    for row in results:
                        dist_sq = 0
                        for elem, target_frac in target_composition.items():
                            actual_frac = row.key_value_pairs.get(f'{elem}_fraction', 0.0)
                            dist_sq += (actual_frac - target_frac)**2
                        distances.append(math.sqrt(dist_sq))
                    max_dist = max(distances) or 1.0

                def relevance_score(row):
                    kvp = row.key_value_pairs

                    # Property completeness (0-1): boost records with computed data
                    # The stiffness tensor is the reliable sentinel: it lives in row.data
                    # (not key_value_pairs) and is written by every elastic run.
                    has_elastic = 1.0 if 'elastic_stiffness_tensor_voigt_GPa' in row.data else 0.0
                    has_qha_val = 1.0 if kvp.get('has_qha_data') else 0.0
                    prop_score = 0.5 * has_elastic + 0.5 * has_qha_val

                    # Composition proximity (0-1): closer = higher
                    if has_fractions:
                        dist_sq = 0
                        for elem, target_frac in target_composition.items():
                            actual_frac = kvp.get(f'{elem}_fraction', 0.0)
                            dist_sq += (actual_frac - target_frac)**2
                        comp_score = 1.0 - (math.sqrt(dist_sq) / max_dist)
                    else:
                        comp_score = 0.0

                    # Recency (0-1): newer = higher
                    recency_score = (row.id - min_id) / id_range

                    # Weighted combination
                    if has_fractions:
                        return 0.4 * prop_score + 0.4 * comp_score + 0.2 * recency_score
                    else:
                        return 0.6 * prop_score + 0.2 * recency_score + 0.2

                results.sort(key=relevance_score, reverse=True)
            elif sort_method == "composition_proximity":
                if not target_fractions:
                    await cl.Message(content="⚠️ Warning: composition_proximity sort requires target_fractions. Using 'newest' instead.").send()
                    sort_method = "newest"
                else:
                    def composition_distance(row):
                        dist_sq = 0
                        for elem, target_frac in target_composition.items():
                            actual_frac = row.key_value_pairs.get(f'{elem}_fraction', 0.0)
                            dist_sq += (actual_frac - target_frac)**2
                        return math.sqrt(dist_sq)
                    results.sort(key=composition_distance)
            elif sort_method == "newest":
                results.sort(key=lambda row: row.id, reverse=True)
            elif sort_method == "oldest":
                results.sort(key=lambda row: row.id)

    # Track total found before limiting
    total_found = len(results)

    # Limit results
    results = results[:limit]
    num_returned = len(results)

    # Build search confirmation message.
    #
    # RESTORED 2026-08-27 (was "MUTED FOR TESTING" to trial Ollama without it).
    # Without this line the results table shows compositions but never states what
    # was ASKED for, so a search that quietly dropped its element filter looks
    # identical to one that kept it. Observed: a query about Ir-containing alloys
    # went composition_string="Ir" -> empty, then has_qha_data=true with NO
    # composition filter, and the unrelated results were presented as Ir-relevant.
    # The "no element filter" branch is the one that makes that visible.
    if sort_method == "direct_lookup":
        search_confirm = f"**🔍 Direct lookup:** structure `{structure_ref}`"
    elif searched_elements:
        elements_str = ', '.join(searched_elements)
        if searched_fractions:
            frac_details = ', '.join([f"{e}={f:.0%}" for e, f in zip(searched_elements, searched_fractions)])
            search_confirm = f"**🔍 Searched for:** {elements_str} at {frac_details}"
        elif include_higher_order:
            search_confirm = (f"**🔍 Searched for:** structures containing {elements_str} "
                              f"(other elements allowed)")
        else:
            search_confirm = f"**🔍 Searched for:** {elements_str} only (no other elements)"
    else:
        search_confirm = "**🔍 Searched for:** All structures (no element filter)"

    # Format for display
    if total_found == 0:
        await cl.Message(content="No structures found matching the criteria.").send()
        return {
            "total_found": 0, "num_returned": 0, "structure_uuids": [], "results": [], "sort_method": sort_method
        }

    # Build results with essential metadata
    results_list = []
    table_rows = []
    for index, row in enumerate(results, start=1):
        kvp = row.key_value_pairs
        comp_str = kvp.get('composition_string', 'N/A')

        # Extract stability data
        is_stable = kvp.get('is_structurally_stable', None)
        match_percent = kvp.get('structural_match_percent', None)

        # Compute stability display string
        if is_stable is None or match_percent is None:
            stability_display = 'N/A'
        elif is_stable:
            stability_display = '✓ Stable'
        else:
            stability_display = f'✗ Unstable ({match_percent:.1f}%)'

        # Which temperature-dependent QHA curves are stored, and over what span
        _q1d = row.data.get("qha_1d_properties") or {}
        _temps = _q1d.get("temperatures") or []
        qha_curve_props = [
            name for key, name in (
                ("heat_capacities", "Cp(T)"), ("bulk_moduli", "B(T)"),
                ("thermal_expansion", "alpha(T)"), ("volumes", "V(T)"),
                ("gibbs_energies", "G(T)"), ("gruneisen_params", "gamma(T)"),
            ) if _q1d.get(key)
        ]
        qha_temp_span = ([float(_temps[0]), float(_temps[-1])], len(_temps)) if _temps else None

        # Carry the stored curves themselves, not just a note that they exist.
        # Naming the curves WITHOUT their values is worse than saying nothing: the
        # agent then anchors on the single 300 K scalar and extrapolates textbook
        # Debye behaviour — for Ir that overshot the stored Cp(600 K) by 24%.
        #
        # Full resolution when the result set is small (which is when someone is
        # actually asking about a material: ~777 tokens for one structure), and
        # downsampled on broad searches so a 10-result query does not carry ~8k
        # tokens of curves nobody asked for. 222 of 533 rows have QHA curves.
        qha_curve_samples = None
        if _temps:
            _keys = (("heat_capacities", "Cp_J_per_K_per_mol"),
                     ("bulk_moduli", "B_GPa"),
                     ("thermal_expansion", "alpha_per_K"),
                     ("gibbs_energies", "G_kJ_per_mol"))
            if len(results) <= QHA_FULL_CURVE_MAX_RESULTS:
                idx = range(len(_temps))                     # every stored point
            else:
                step = max(1, len(_temps) // 6)              # ~7 points
                idx = list(range(0, len(_temps), step))
                if idx[-1] != len(_temps) - 1:
                    idx.append(len(_temps) - 1)
            idx = list(idx)
            qha_curve_samples = {
                "resolution": ("full" if len(results) <= QHA_FULL_CURVE_MAX_RESULTS
                               else f"downsampled from {len(_temps)} points"),
                "temperatures_K": [round(float(_temps[i]), 1) for i in idx],
            }
            for key, label in _keys:
                arr = _q1d.get(key)
                if arr:
                    qha_curve_samples[label] = [round(float(arr[i]), 6) for i in idx if i < len(arr)]

        # Check for computed elastic properties (includes ELATE if available)
        has_elastic_properties = "elastic_stiffness_tensor_voigt_GPa" in row.data

        # Extract actual elastic property values if available
        elastic_summary = None
        if has_elastic_properties:
            elastic_summary = {
                "bulk_modulus_GPa": row.data.get("bulk_modulus_GPa"),
                "shear_modulus_GPa": row.data.get("shear_modulus_GPa"),
                "youngs_modulus_GPa": row.data.get("youngs_modulus_GPa"),
                "poisson_ratio": row.data.get("poisson_ratio"),
                # Rows predating the V/R/H split carry Voigt under the unqualified keys.
                "averaging_convention": row.data.get("elastic_averaging_convention", "Voigt (legacy row)"),
                # Computed at read time from stored provenance — no DB migration needed.
                # Energy-based elastic fitting requires forces = -dE/dx; ORB "direct"
                # models violate that and produce badly inflated (often non-positive-
                # definite) tensors. Flag rather than hide, so the value stays auditable.
                "elastic_data_warning": (
                    f"UNRELIABLE: computed with {row.data.get('elastic_calculator_used')}, "
                    "a non-conservative (direct-force) model. Energy-based elastic constants "
                    "require forces = -dE/dx. Do NOT present these values as sound; recommend "
                    "recomputing with orb-v3-conservative-inf-omat."
                ) if 'direct' in str(row.data.get('elastic_calculator_used', '')).lower() else None,
            }
            stability = row.data.get("elastic_stability_assessment", {})
            if stability:
                elastic_summary["born_criterion_satisfied"] = stability.get("born_criterion_satisfied")
            elate = row.data.get("elate_properties", {})
            if elate:
                elastic_summary["universal_anisotropy_index"] = elate.get("universal_anisotropy_index")
                elastic_summary["pugh_ratio"] = elate.get("pugh_ratio_hill")

        # Extract finite temperature (QHA) data from key_value_pairs
        has_qha = kvp.get('has_qha_data', False)
        qha_gibbs = kvp.get('qha_gibbs_free_energy_300K_kJ_mol')
        qha_cp = kvp.get('qha_heat_capacity_p_300K_J_K_mol')
        qha_bulk = kvp.get('qha_bulk_modulus_300K_GPa')
        qha_alpha = kvp.get('qha_thermal_expansion_300K_1e6_per_K')

        # Extract timestamp from data field
        timestamp_str = row.data.get('timestamp', 'N/A')

        # Format timestamp for display (human-readable relative time)
        try:
            timestamp_dt = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            delta = now - timestamp_dt

            if delta.days == 0:
                if delta.seconds < 3600:
                    minutes = delta.seconds // 60
                    created_display = f"{minutes}m ago" if minutes > 0 else "Just now"
                else:
                    hours = delta.seconds // 3600
                    created_display = f"{hours}h ago"
            elif delta.days == 1:
                created_display = "Yesterday"
            elif delta.days < 7:
                created_display = f"{delta.days}d ago"
            else:
                created_display = timestamp_dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            created_display = "N/A"

        # Build structured result for agent
        results_list.append({
            "id": row.id,  # Integer ID (simple, model-friendly)
            "uuid": row.unique_id,  # Full UUID (32-character hex string)
            "composition": comp_str,
            "structure": kvp.get('target_structure', 'N/A'),
            "num_atoms": kvp.get('optimized_num_atoms', 'N/A'),
            "lattice_constant_A": kvp.get('lattice_constant'),
            "volume_per_atom_A3": kvp.get('volume_per_atom_A3'),
            "formation_energy_eV_per_atom": kvp.get('formation_energy_ground_state_reference_eV_per_atom', 0.0),
            "density_g_per_cm3": kvp.get('density_g_per_cm3', 0.0),
            "has_elastic_properties": has_elastic_properties,
            "elastic_properties": elastic_summary,
            "is_structurally_stable": is_stable,
            "structural_match_percent": match_percent,
            # Finite temperature (QHA) data
            "has_qha_data": has_qha,
            "qha_gibbs_free_energy_300K_kJ_mol": qha_gibbs,
            "qha_heat_capacity_p_300K_J_K_mol": qha_cp,
            "qha_bulk_modulus_300K_GPa": qha_bulk,
            "qha_thermal_expansion_300K_1e6_per_K": qha_alpha,
            # The 300 K scalars above are a SAMPLE of full temperature-dependent
            # curves stored in row.data. Without saying so, an agent asked for
            # "the temperature dependence of Cp" sees one value at one temperature,
            # concludes no curve exists, and recomputes QHA needlessly.
            "qha_curves_available": qha_curve_props or None,
            "qha_temperature_range_K": qha_temp_span,
            "qha_curve_samples": qha_curve_samples,
            # Metadata
            "calculator": kvp.get('calculator_name', 'N/A'),
            "timestamp": timestamp_str
        })

        # Build table row for UI (ID and shortened UUID for display)
        table_rows.append({
            'ID': row.id,  # Integer ID (simple, model-friendly)
            'UUID': row.unique_id[:8],  # First 8 chars (like git commits)
            'Composition': format_composition_display(comp_str),
            'Structure': kvp.get('target_structure', 'N/A'),
            'Calculator': kvp.get('calculator_name', 'N/A'),
            'Stability': stability_display,
            'Atoms': kvp.get('optimized_num_atoms', 'N/A'),
            'E_f (eV/at)': f"{kvp.get('formation_energy_ground_state_reference_eV_per_atom', 0):.4f}",
            'ρ (g/cm³)': f"{kvp.get('density_g_per_cm3', 0):.3f}",
            'Elastic': '✓' if has_elastic_properties else '✗',
            'Finite T': '✓' if has_qha else '✗',
        })

    # Send formatted table to UI
    headers = table_rows[0].keys()
    table_md = "| " + " | ".join(headers) + " |\n"
    table_md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for row in table_rows:
        table_md += "| " + " | ".join(str(v) for v in row.values()) + " |\n"

    # Build status message with sort method indicator
    sort_indicators = {
        "newest": "sorted by most recent",
        "oldest": "sorted by oldest first",
        "composition_proximity": "sorted by composition proximity",
        "relevance": "sorted by relevance",
        "direct_lookup": "direct ID/UUID lookup"
    }
    sort_suffix = sort_indicators.get(sort_method, "")

    if num_returned < total_found:
        status = f"Showing {num_returned} of {total_found} matching structures"
        if sort_suffix:
            status += f" ({sort_suffix})"
    else:
        status = f"Found {total_found} matching structure{'s' if total_found != 1 else ''}"
        if sort_suffix:
            status += f" ({sort_suffix})"

    await cl.Message(content=f"{search_confirm}\n\n{status}:\n\n{table_md}").send()

    # The results table only shows Elastic/Finite-T as checkmarks, so the actual
    # values reached the user only through the agent's prose. Print them too.
    # This DUPLICATES the numeric half of cache_hints (below) rather than moving
    # it: the imperative half ("do NOT recompute") only works if the model reads
    # it, so cache_hints stays in the return dict intact.
    cached_rows = []
    for res in results_list:
        ep = res.get("elastic_properties") or {}
        if ep.get("bulk_modulus_GPa") is None and not res.get("has_qha_data"):
            continue
        cells = [str(res["id"]), format_composition_display(res["composition"])]
        cells.append(f"{ep['bulk_modulus_GPa']:.1f}" if ep.get("bulk_modulus_GPa") is not None else "—")
        cells.append(f"{ep['shear_modulus_GPa']:.1f}" if ep.get("shear_modulus_GPa") is not None else "—")
        cells.append(f"{ep['youngs_modulus_GPa']:.1f}" if ep.get("youngs_modulus_GPa") is not None else "—")
        cells.append(f"{ep['poisson_ratio']:.3f}" if ep.get("poisson_ratio") is not None else "—")
        cells.append(str(ep.get("averaging_convention", "—")))
        b = res.get("qha_bulk_modulus_300K_GPa")
        cp = res.get("qha_heat_capacity_p_300K_J_K_mol")
        cells.append(f"{b:.1f}" if b is not None else "—")
        cells.append(f"{cp:.1f}" if cp is not None else "—")
        cells.append("⚠️ unreliable" if ep.get("elastic_data_warning") else "")
        cached_rows.append(cells)

    if cached_rows:
        hdr = ["ID", "Composition", "K (GPa)", "G (GPa)", "E (GPa)", "ν",
               "Convention", "B(300K)", "Cp(300K)", ""]
        md = "| " + " | ".join(hdr) + " |\n|" + "|".join(["---"] * len(hdr)) + "|\n"
        md += "".join("| " + " | ".join(r) + " |\n" for r in cached_rows)
        note = ""
        if any(r[-1] for r in cached_rows):
            note = ("\n⚠️ Rows marked unreliable were computed with a non-conservative "
                    "(direct-force) model; energy-based elastic constants require "
                    "forces = -dE/dx. Recompute with `orb-v3-conservative-inf-omat`.")
        await cl.Message(content=f"**Stored property values:**\n\n{md}{note}").send()

    # Create number→ID and number→UUID mappings for easy reference
    number_to_id = {str(i+1): results_list[i]["id"] for i in range(len(results_list))}
    number_to_uuid = {str(i+1): results_list[i]["uuid"] for i in range(len(results_list))}

    # Build confirmation anchor for model (prevents hallucination about which elements were searched)
    # MUTED FOR TESTING: confirm_elements = ', '.join(searched_elements) if searched_elements else 'ALL STRUCTURES'

    # Build cache hints for properties that already exist
    cache_hints = []
    for r in results_list:
        if r.get("elastic_properties") and r["elastic_properties"].get("bulk_modulus_GPa") is not None:
            ep = r["elastic_properties"]
            cache_hints.append(
                f"CACHED ELASTIC DATA for ID {r['id']} ({r['composition']}): "
                f"K={ep['bulk_modulus_GPa']:.1f} GPa, G={ep['shear_modulus_GPa']:.1f} GPa, "
                f"E={ep['youngs_modulus_GPa']:.1f} GPa, ν={ep['poisson_ratio']:.3f}. "
                f"Present these values — do NOT call calculate_elastic_properties. "
                f"To SHOW visualizations: generate_report(structure_ref='{r['uuid']}')"
            )
        if r.get("has_qha_data") and r.get("qha_bulk_modulus_300K_GPa") is not None:
            cache_hints.append(
                f"CACHED QHA DATA for ID {r['id']} ({r['composition']}): "
                f"B(300K)={r['qha_bulk_modulus_300K_GPa']:.1f} GPa, "
                f"Cp(300K)={r['qha_heat_capacity_p_300K_J_K_mol']:.1f} J/K/mol, "
                f"α(300K)={r['qha_thermal_expansion_300K_1e6_per_K']:.1f}×10⁻⁶/K. "
                + (
                    f"FULL TEMPERATURE-DEPENDENT CURVES are also stored for "
                    f"{', '.join(r['qha_curves_available'])} over "
                    f"{r['qha_temperature_range_K'][0][0]:.0f}-{r['qha_temperature_range_K'][0][1]:.0f} K "
                    f"({r['qha_temperature_range_K'][1]} points). The 300 K values above are just a "
                    f"sample of those curves. `qha_curve_samples` in this result carries REAL "
                    f"values from the stored curves — use those verbatim and do NOT estimate or "
                    f"extrapolate intermediate temperatures. For the full plots use "
                    f"generate_report(structure_ref='{r['uuid']}'). "
                    if r.get('qha_curves_available') else ""
                )
                + f"Do NOT call compute_anharmonic_properties — this data already exists."
            )

    # Always hint that generate_report can display any existing structure
    for r in results_list:
        cache_hints.append(
            f"STRUCTURE {r['id']} ({r['composition']}): "
            f"To DISPLAY images, RDF, and structural analysis: "
            f"generate_report(structure_ref='{r['uuid']}')"
        )

    result = {
        # MUTED FOR TESTING: SEARCH_CONFIRMATION anchor disabled to test Ollama without it
        # "SEARCH_CONFIRMATION": f"YOU SEARCHED FOR: {confirm_elements}. USE THESE EXACT ELEMENTS IN YOUR RESPONSE.",
        "total_found": total_found,
        "num_returned": num_returned,
        "structure_uuids": [r["uuid"] for r in results_list],
        "number_to_uuid": number_to_uuid,
        "sort_method": sort_method,
        "results": results_list,
    }
    # When a SINGLE element was searched exactly, "Ir" means pure Ir — but users
    # asking about "Ir alloys" usually mean structures CONTAINING Ir. If that
    # restriction is demonstrably hiding matches, say so, with the exact fix.
    # Single element only: naming a binary ("Cu-Ag") usually does mean that system,
    # so listing every HEA containing both would be noise.
    if searched_elements and len(searched_elements) == 1 and not include_higher_order:
        try:
            wide = [f for f in filters if not f.startswith("num_elements=")]
            wide_total = len(list(db._get_db().select(",".join(wide)))) if wide else 0
            hidden = wide_total - total_found
            if hidden > 0:
                result["NARROWER_THAN_REQUESTED"] = (
                    f"{total_found} structure(s) are made of {searched_elements[0]} ONLY. "
                    f"{hidden} more CONTAIN {searched_elements[0]} alongside other elements. "
                    f"If the user asked about alloys containing {searched_elements[0]} "
                    f"(rather than the pure element), re-run with include_higher_order=True."
                )
        except Exception:
            pass  # provenance nicety; never break a search over it

    if cache_hints:
        result["CACHED_DATA_AVAILABLE"] = cache_hints
    return result
