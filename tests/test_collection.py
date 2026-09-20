"""Tests for catalogue-wide analysis coordination."""

from pathlib import Path

from csfdata.catalogue import CatalogueSimulation
from csfdata_analysis.catalogue_runner import (
    clear_time_series,
    compute_catalogue_simulation,
    compute_collection,
)
from csfdata_analysis.diagnostics import TIME_SERIES_DIAGNOSTICS
from csfdata_analysis.runner import time_series_path


def test_collection_runner_reports_unsupported_importers_without_stopping() -> None:
    """One unsupported simulation becomes a failed result owned by the parent."""
    simulation = CatalogueSimulation(
        collection_id="mhd-grid-v1",
        simulation_id="0001",
        path=Path("/catalogue/collections/mhd-grid-v1/simulations/0001"),
        importer="mhd",
    )

    result = compute_catalogue_simulation(simulation, "lagrangian_radii")
    results = compute_collection((simulation,), "lagrangian_radii")

    assert result.status == "failed"
    assert result.error == "Unsupported catalogue importer: mhd"
    assert results == (result,)


def test_clear_time_series_removes_only_selected_lite_output(tmp_path: Path) -> None:
    """Derived clearing removes only the requested diagnostic file in a lite root."""
    lite_catalogue = tmp_path / "lite"
    lite_catalogue.mkdir()
    (lite_catalogue / "lite.yaml").touch()
    simulation = CatalogueSimulation(
        collection_id="dcaf-grid-v1",
        simulation_id="0001",
        path=lite_catalogue / "collections" / "dcaf-grid-v1" / "simulations" / "0001",
        importer="dcaf",
    )
    diagnostic = TIME_SERIES_DIAGNOSTICS["lagrangian_radii"]
    output_path = time_series_path(simulation.path, diagnostic)
    output_path.parent.mkdir(parents=True)
    output_path.touch()

    removed_paths = clear_time_series(lite_catalogue, (simulation,), diagnostic.name)

    assert removed_paths == (output_path,)
    assert not output_path.exists()
