"""Tests for standardized diagnostic definitions and local discovery."""

from pathlib import Path

import pytest

from csfdata.catalogue.diagnostics import DiagnosticField, DiagnosticRequirement
from csfdata_analysis.diagnostics import (
    DIAGNOSTICS,
    EvaluationChoice,
    SCALAR_DIAGNOSTICS,
    ScalarDiagnostic,
    TimeSeriesDiagnostic,
    TIME_SERIES_DIAGNOSTICS,
    diagnostic_directories,
    diagnostics_in_dependency_order,
    load_diagnostics,
)


def test_diagnostic_records_its_declared_schema() -> None:
    """A diagnostic definition is importable without AMUSE installed."""
    diagnostic = TimeSeriesDiagnostic(
        name="example",
        version=1,
        description="Example diagnostic used only for tests.",
        evaluate=lambda particles: {"example": {"value": 1.0}},
        evaluation_choices=(
            EvaluationChoice(
                name="example",
                metadata={"method": "example"},
                outputs={"value": "pc"},
            ),
        ),
        field_descriptions={"value": "Example output value."},
    )

    assert diagnostic.name == "example"
    assert diagnostic.version == 1
    assert diagnostic.evaluation_choices[0].outputs == {"value": "pc"}


def test_scalar_diagnostic_declares_fields_and_requirements() -> None:
    """Scalar templates publish their dependency graph without a calculation."""
    diagnostic = ScalarDiagnostic(
        name="example_summary",
        version=1,
        description="Example scalar diagnostic used only for tests.",
        evaluate=lambda simulation, choices: {"example_value": 0.0},
        fields=(DiagnosticField("example_value", "Example scalar value.", "km/s"),),
        requires=(DiagnosticRequirement("example_series", "v1"),),
    )

    definition = diagnostic.catalogue_definition()

    assert definition.kind == "scalar"
    assert definition.relative_path.as_posix() == "derived/scalar_diagnostics.yaml"
    assert definition.requires == (DiagnosticRequirement("example_series", "v1"),)


def test_diagnostic_loader_uses_explicit_module_tuples() -> None:
    """Declared module tuples provide every registered diagnostic definition."""
    loaded = load_diagnostics()

    assert loaded == DIAGNOSTICS
    assert ("lagrangian_radii", "v1") in loaded
    assert ("expansion_rate", "v1") in loaded
    assert TIME_SERIES_DIAGNOSTICS["lagrangian_radii"] is loaded[
        ("lagrangian_radii", "v1")
    ]
    assert SCALAR_DIAGNOSTICS["expansion_rate"] is loaded[("expansion_rate", "v1")]


def test_diagnostic_order_places_requirements_first() -> None:
    """The registry order makes dependent catalogue updates safe."""
    ordered = diagnostics_in_dependency_order(DIAGNOSTICS)
    identities = [(diagnostic.name, diagnostic.version) for diagnostic in ordered]

    assert identities.index(("lagrangian_radii", 1)) < identities.index(("expansion_rate", 1))


def test_local_diagnostic_directory_loads_through_analysis_configuration(
    tmp_path: Path,
) -> None:
    """A lite analysis configuration adds a trusted local diagnostic module."""
    local_directory = tmp_path / "local_diagnostics"
    local_directory.mkdir()
    (local_directory / "local_radius.py").write_text(
        "from csfdata_analysis.diagnostics import EvaluationChoice, TimeSeriesDiagnostic\n"
        "\n"
        "LOCAL_RADIUS_V1 = TimeSeriesDiagnostic(\n"
        "    name='local_radius',\n"
        "    version=1,\n"
        "    description='Local test diagnostic.',\n"
        "    evaluate=lambda particles: {'origin': {'radius': 1.0}},\n"
        "    evaluation_choices=(EvaluationChoice('origin', {}, {'radius': 'pc'}),),\n"
        "    field_descriptions={'radius': 'Local radius.'},\n"
        ")\n"
        "\n"
        "DIAGNOSTICS = (LOCAL_RADIUS_V1,)\n",
        encoding="utf-8",
    )
    (tmp_path / "analysis.yaml").write_text(
        "schema_version: 1\n"
        "diagnostic_directories:\n"
        "  - local_diagnostics\n",
        encoding="utf-8",
    )

    directories = diagnostic_directories(tmp_path)
    loaded = load_diagnostics(directories)

    assert directories == (local_directory,)
    assert ("local_radius", "v1") in loaded


def test_local_diagnostic_directory_rejects_invalid_tuple_contents(tmp_path: Path) -> None:
    """A local module reports its path and invalid object type clearly."""
    (tmp_path / "invalid_diagnostic.py").write_text(
        "DIAGNOSTICS = (42,)\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid_diagnostic.py DIAGNOSTICS contains int",
    ):
        load_diagnostics((tmp_path,))
