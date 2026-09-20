"""Tests for standardized diagnostic definitions."""

from csfdata.catalogue.diagnostics import DiagnosticField, DiagnosticRequirement
from csfdata_analysis.diagnostics import (
    EvaluationChoice,
    ScalarDiagnostic,
    TimeSeriesDiagnostic,
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
        fields=(DiagnosticField("example_value", "Example scalar value.", "km/s"),),
        requires=(DiagnosticRequirement("example_series", "v1"),),
    )

    definition = diagnostic.catalogue_definition()

    assert definition.kind == "scalar"
    assert definition.relative_path.as_posix() == "derived/scalar_diagnostics.yaml"
    assert definition.requires == (DiagnosticRequirement("example_series", "v1"),)
