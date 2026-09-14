"""Tests for catalogue-wide analysis coordination."""

from pathlib import Path

from csfdata.catalogue import CatalogueSimulation
from csfdata_analysis.collection import compute_catalogue_simulation, compute_collection


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
