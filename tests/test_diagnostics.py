"""Tests for standardized diagnostic definitions."""

from csfdata_analysis.diagnostics import Diagnostic, DiagnosticChoice


def test_diagnostic_records_its_declared_schema() -> None:
    """A diagnostic definition is importable without AMUSE installed."""
    diagnostic = Diagnostic(
        name="example",
        version=1,
        evaluate=lambda particles: {"example": {"value": 1.0}},
        choices=(
            DiagnosticChoice(
                name="example",
                metadata={"method": "example"},
                outputs={"value": "pc"},
            ),
        ),
    )

    assert diagnostic.name == "example"
    assert diagnostic.version == 1
    assert diagnostic.choices[0].outputs == {"value": "pc"}
