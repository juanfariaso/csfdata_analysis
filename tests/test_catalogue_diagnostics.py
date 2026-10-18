"""Tests for automatic analysis diagnostic registration in core CSFData."""

from pathlib import Path

import pytest

from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.diagnostics import read_collection_diagnostics
from csfdata_analysis.catalogue_runner import compute_collection
from csfdata_analysis.choices import CENTER, Choice
from csfdata_analysis.diagnostics import ensure_collection_diagnostics
from csfdata_analysis.diagnostics.base import EvaluationChoice, TimeSeriesDiagnostic
from csfdata_analysis.diagnostics.time_series.bound_fraction import (
    BINDING_POTENTIAL,
    BOUND_FRACTION_V1,
)
from csfdata_analysis.diagnostics.time_series.lagrangian_radii import (
    LAGRANGIAN_RADII_V1,
)


def test_ensure_collection_diagnostics_writes_lagrangian_radii_definition(
    tmp_path: Path,
) -> None:
    """Automatic registration writes choices and the Lagrangian-radii schema."""
    collection_root = tmp_path / "example-collection"
    collection_root.mkdir()

    path = ensure_collection_diagnostics(collection_root, (LAGRANGIAN_RADII_V1,))

    published = read_collection_diagnostics(path)
    assert path == collection_root / "diagnostics.yaml"
    assert published.choices[0].name == "center"
    assert published.choices[0].values == ("origin", "stellar_com")
    assert published.choices[0].default == "stellar_com"
    diagnostic = published.diagnostics[0]
    assert diagnostic.name == "lagrangian_radii"
    assert diagnostic.version == "v1"
    assert diagnostic.kind == "time_series"
    assert diagnostic.relative_path.as_posix() == (
        "derived/diagnostics/lagrangian_radii/v1/series.h5"
    )
    assert diagnostic.choices == ("center",)
    assert next(field for field in diagnostic.fields if field.name == "r_l50").unit == "pc"


def test_ensure_collection_diagnostics_writes_diagnostic_local_choice(
    tmp_path: Path,
) -> None:
    """A diagnostic module may publish a choice absent from the shared registry."""
    collection_root = tmp_path / "example-collection"
    collection_root.mkdir()

    path = ensure_collection_diagnostics(collection_root, (BOUND_FRACTION_V1,))

    published = read_collection_diagnostics(path)
    choice = next(
        candidate
        for candidate in published.choices
        if candidate.name == BINDING_POTENTIAL.name
    )
    assert choice.values == ("stellar_self_potential",)
    assert choice.default == "stellar_self_potential"
    diagnostic = next(
        candidate
        for candidate in published.diagnostics
        if candidate.name == BOUND_FRACTION_V1.name
    )
    assert diagnostic.choices == (BINDING_POTENTIAL.name,)


def test_local_choice_cannot_conflict_with_a_shared_choice(tmp_path: Path) -> None:
    """A module cannot silently redefine a package-wide scientific choice."""
    conflicting_center = Choice(
        name=CENTER.name,
        description=CENTER.description,
        values=("origin",),
        default="origin",
    )
    diagnostic = TimeSeriesDiagnostic(
        name="conflicting_choice",
        version=1,
        description="Test-only diagnostic with a conflicting local choice.",
        evaluate=lambda particles: {"origin": {"value": 1.0}},
        evaluation_choices=(
            EvaluationChoice("origin", {}, {"value": "1"}),
        ),
        field_descriptions={"value": "Test value."},
        choice_names=(CENTER.name,),
        choice_definitions=(conflicting_center,),
    )

    with pytest.raises(
        ValueError,
        match="Diagnostic-local choice conflicts with registered shared analysis choice",
    ):
        ensure_collection_diagnostics(tmp_path, (diagnostic,))


def test_collection_runner_registers_an_existing_diagnostic(tmp_path: Path) -> None:
    """A skipped complete result still makes its collection definition available."""
    collection_root = tmp_path / "example-collection"
    simulation_root = collection_root / "simulations" / "0001"
    output_path = (
        simulation_root
        / "derived"
        / "diagnostics"
        / "lagrangian_radii"
        / "v1"
        / "series.h5"
    )
    output_path.parent.mkdir(parents=True)
    output_path.touch()
    simulation = CatalogueSimulation(
        collection_id="example-collection",
        simulation_id="0001",
        path=simulation_root,
        importer="dcaf",
    )

    results = compute_collection((simulation,), "lagrangian_radii")

    assert results[0].status == "skipped"
    assert read_collection_diagnostics(collection_root / "diagnostics.yaml").diagnostics
