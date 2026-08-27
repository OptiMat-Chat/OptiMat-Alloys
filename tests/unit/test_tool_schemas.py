"""Golden tests: Agent Framework tool schemas must reproduce the exact
parameter description strings the AutoGen agent exposed to the model.

The maps below are copied VERBATIM from the `Annotated[T, "..."]` metadata in
src/tools/*.py. If a test fails, fix the tool's *type* annotation only —
never the description string (prompt texts are frozen for the migration).
"""

import hashlib

import pytest

from agent_framework import FunctionTool, tool

from src.tools import (
    generate_alloy_supercell,
    search_database,
    generate_report,
    calculate_elastic_properties,
    visualize_database_statistics,
    compute_anharmonic_properties,
    recompute_structure,
)

GOLDEN_PARAM_DESCRIPTIONS = {
    "generate_alloy_supercell": {
        "structure": "Lattice structure",
        "elements": "Element symbols (or use composition_string)",
        "target_fractions": "Atomic fractions, sum=1 (or use composition_string)",
        "composition_string": "e.g., 'Ag75Cu25' or 'Cu-Ag' (takes precedence)",
        "sqs_iterations": "SQS iterations (0=disabled)",
        "lattice_constant": "Lattice constant Å (0=auto-estimate)",
        "fmax": "Force convergence eV/Å (default 0.001)",
    },
    "search_database": {
        "structure_ref": "Direct lookup by ID (e.g. '111') or UUID — bypasses all filters",
        "elements": "Element symbols (or use composition_string)",
        "target_fractions": "Atomic fractions (or use composition_string)",
        "composition_string": "e.g., 'Ag75Cu25' or 'Cu-Ag'",
        "structure": "Crystal structure filter (empty=none)",
        "min_atoms": "Min atoms (0=none)",
        "max_atoms": "Max atoms (0=none)",
        "composition_tolerance": "Composition tolerance as a FRACTION, not a percent: 0.05 means +/-5 at.%. Default 0.05 is right for almost every query — OMIT this argument unless the user asks for a wider or narrower composition window. Do NOT pass 5 for '5%'; values above 0.5 are rejected because they would match every composition.",
        "include_higher_order": "Allow elements beyond those named. True when the user asks for alloys CONTAINING an element — 'Ir-containing alloys', 'alloys with Ti', 'Cu-based alloys', 'anything with Ag and Cu'. False when they name an exact system — 'Cu-Ag', 'CoCrFeNi', 'pure Cu' — where extra elements would be wrong.",
        "stable_only": "Structurally stable only (≥90% match)",
        "phonon_stable_only": "Dynamically stable (no imaginary modes)",
        "has_qha_data": "Has QHA data",
        "calculator_name": "Calculator filter: 'mace', 'orb', 'nequip', or full name",
        "sort_by": "Sort order",
        "limit": "Max results (default 10, max 50)",
    },
    "generate_report": {
        "structure_ref": "Structure ID (local) or UUID (global)",
        "include_rdf": "Include RDF analysis",
        "rdf_cutoff": "RDF cutoff Å",
    },
    "calculate_elastic_properties": {
        "structure_ref": "Structure ID (local) or UUID (global)",
        "epsilon": "Strain magnitude as a FRACTION (not percent). Default 0.01 = 1%. Valid range: 0.01–0.02. Do NOT pass values ≥0.1 — this is a strain applied to the cell for finite differences, not a percentage.",
    },
    "visualize_database_statistics": {
        "chart_types": "Chart types (default: all)",
    },
    "compute_anharmonic_properties": {
        "structure_ref": "Structure ID (local) or UUID (global)",
        "num_volumes": "Volume points for QHA (default 11)",
        "compute_thermal_conductivity_flag": "Enable κ(T) calculation (expensive)",
        "mesh_qha": "QHA phonon mesh [nx,ny,nz] (default [20,20,20])",
        "mesh_phono3py": "Thermal conductivity mesh [nx,ny,nz] (default [20,20,20])",
        "temperature_range": "[Tmin, Tmax, Tstep] in K, Tmax INCLUSIVE — you get exactly the range you ask for. OMIT this argument unless the user gave explicit temperatures; the tool applies a safe default.",
    },
    "recompute_structure": {
        "source_ref": "Structure ID (local) or UUID (global) to recompute",
        "calculator": "Calculator name (default: session setting)",
        "fmax": "Force convergence eV/Å (default 0.001)",
    },
}

# SHA-256 of each tool's docstring (the description the model sees), pinned
# at migration time. An empty docstring hashes to the well-known empty-string
# digest — generate_alloy_supercell had no docstring in the AutoGen era and
# must keep none until prompt optimization.
GOLDEN_DESCRIPTION_SHA256 = {
    "generate_alloy_supercell": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "search_database": "9e181083dcc452c10c1614e726d6f7c8c604092d2cb1013638d53a6678c0c75c",
    "generate_report": "e2d1b405378ae5601a404ba168a90b04d02d29adc782dc080db6377922068d88",
    "calculate_elastic_properties": "ae608a8dc80b7be81813dce8139c01d3e9dcb364acf4d9b568d1748c4b0cea26",
    "visualize_database_statistics": "fcbe20a4ac4fd32bbc7fd1e08a13649d3a0857123711704be286a8c4728ccc02",
    "compute_anharmonic_properties": "004f6e69d0e51f672bdcc9167716d6b6d9b3aeacc197ab201a0e1c4f97fc6f23",
    "recompute_structure": "48ff9059f5db71288efd53c4c4b5850b3b8f5d82be84181eeee5b6dfe6bec89c",
}

# Sentinel: parameter has no default (it is required).
NO_DEFAULT = object()

# Complete default map — every parameter of every tool. Schema-value parity
# (1e-2 == 0.01 is acceptable); None defaults are asserted as an explicit
# "default": null key in the schema, not a missing key.
GOLDEN_PARAM_DEFAULTS = {
    "generate_alloy_supercell": {
        "structure": NO_DEFAULT,
        "elements": None,
        "target_fractions": None,
        "composition_string": "",
        "sqs_iterations": 1000000,
        "lattice_constant": 0.0,
        "fmax": 0.001,
    },
    "search_database": {
        "structure_ref": None,
        "elements": None,
        "target_fractions": None,
        "composition_string": "",
        "structure": "",
        "min_atoms": 0,
        "max_atoms": 0,
        "composition_tolerance": 0.05,
        "include_higher_order": False,
        "stable_only": False,
        "phonon_stable_only": False,
        "has_qha_data": None,
        "calculator_name": None,
        "sort_by": "auto",
        "limit": 10,
    },
    "generate_report": {
        "structure_ref": NO_DEFAULT,
        "include_rdf": True,
        "rdf_cutoff": 10.0,
    },
    "calculate_elastic_properties": {
        "structure_ref": NO_DEFAULT,
        "epsilon": 0.01,
    },
    "visualize_database_statistics": {
        "chart_types": None,
    },
    "compute_anharmonic_properties": {
        "structure_ref": NO_DEFAULT,
        "num_volumes": 11,
        "compute_thermal_conductivity_flag": False,
        "mesh_qha": None,
        "mesh_phono3py": None,
        "temperature_range": None,
    },
    "recompute_structure": {
        "source_ref": NO_DEFAULT,
        "calculator": None,
        "fmax": 0.001,
    },
}

GOLDEN_REQUIRED = {
    "generate_alloy_supercell": ["structure"],
    "search_database": [],
    "generate_report": ["structure_ref"],
    "calculate_elastic_properties": ["structure_ref"],
    "visualize_database_statistics": [],
    "compute_anharmonic_properties": ["structure_ref"],
    "recompute_structure": ["source_ref"],
}

ALL_TOOLS = {
    "generate_alloy_supercell": generate_alloy_supercell,
    "search_database": search_database,
    "generate_report": generate_report,
    "calculate_elastic_properties": calculate_elastic_properties,
    "visualize_database_statistics": visualize_database_statistics,
    "compute_anharmonic_properties": compute_anharmonic_properties,
    "recompute_structure": recompute_structure,
}


def _as_function_tool(fn) -> FunctionTool:
    """Convert a tool exactly as the Agent does with plain callables."""
    converted = tool(fn)
    assert isinstance(converted, FunctionTool)
    return converted


@pytest.mark.parametrize("tool_name", sorted(ALL_TOOLS))
def test_tool_converts_and_keeps_name_and_docstring(tool_name):
    ft = _as_function_tool(ALL_TOOLS[tool_name])
    assert ft.name == tool_name
    # The description the model sees must derive from the function docstring…
    assert ft.description == (ALL_TOOLS[tool_name].__doc__ or "")
    # …and that docstring is pinned byte-identically to the AutoGen-era text.
    digest = hashlib.sha256(ft.description.encode("utf-8")).hexdigest()
    assert digest == GOLDEN_DESCRIPTION_SHA256[tool_name], (
        f"{tool_name}: docstring text changed — prompt texts are frozen "
        f"for the migration (stage 2 is prompt optimization)"
    )


@pytest.mark.parametrize("tool_name", sorted(ALL_TOOLS))
def test_tool_schema_has_exactly_the_golden_params(tool_name):
    props = _as_function_tool(ALL_TOOLS[tool_name]).parameters()["properties"]
    assert set(props) == set(GOLDEN_PARAM_DESCRIPTIONS[tool_name])


@pytest.mark.parametrize("tool_name", sorted(ALL_TOOLS))
def test_tool_param_descriptions_are_byte_identical(tool_name):
    props = _as_function_tool(ALL_TOOLS[tool_name]).parameters()["properties"]
    for param, expected in GOLDEN_PARAM_DESCRIPTIONS[tool_name].items():
        assert props[param].get("description") == expected, (
            f"{tool_name}.{param}: schema description diverged from the "
            f"frozen AutoGen-era annotation"
        )


@pytest.mark.parametrize("tool_name", sorted(GOLDEN_PARAM_DEFAULTS))
def test_tool_param_defaults_survive(tool_name):
    schema = _as_function_tool(ALL_TOOLS[tool_name]).parameters()
    props = schema["properties"]
    assert set(GOLDEN_PARAM_DEFAULTS[tool_name]) == set(props), (
        f"{tool_name}: default map out of sync with signature"
    )
    for param, expected in GOLDEN_PARAM_DEFAULTS[tool_name].items():
        if expected is NO_DEFAULT:
            assert "default" not in props[param], (
                f"{tool_name}.{param}: required param grew a default"
            )
        else:
            assert "default" in props[param], (
                f"{tool_name}.{param}: default disappeared from schema"
            )
            assert props[param]["default"] == expected, (
                f"{tool_name}.{param}: default changed"
            )


@pytest.mark.parametrize("tool_name", sorted(GOLDEN_REQUIRED))
def test_tool_required_params_unchanged(tool_name):
    schema = _as_function_tool(ALL_TOOLS[tool_name]).parameters()
    assert sorted(schema.get("required", [])) == sorted(GOLDEN_REQUIRED[tool_name])
