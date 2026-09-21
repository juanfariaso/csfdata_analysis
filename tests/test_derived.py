"""Tests for safe import of lite-catalogue derived data."""

from pathlib import Path

import h5py

from csfdata.catalogue.configuration import (
    ConfigurationParameter,
    SimulationConfiguration,
    write_simulation_configuration,
)
from csfdata.catalogue.lite import file_sha256, import_lite_collection
from csfdata.catalogue.metadata import SimulationMetadata, write_simulation_metadata
from csfdata.catalogue.registry import index_catalogue
from csfdata.catalogue.snapshots import refresh_snapshot_times
from csfdata_analysis.derived import import_derived


def test_import_derived_skips_existing_and_overwrites_explicitly(tmp_path: Path) -> None:
    full = tmp_path / "full"
    simulation = full / "collections" / "grid" / "simulations" / "0001"
    simulation.mkdir(parents=True)
    (simulation.parent.parent / "collection.yaml").write_text(
        """schema_version: 1
id: grid
importer: dcaf
config_schema_version: 1
required_parameters: []
optional_parameters: []
""",
        encoding="utf-8",
    )
    write_simulation_metadata(
        SimulationMetadata("0001", "grid", "dcaf", "host", tmp_path / "source", Path("run"), "2026-01-01T00:00:00Z"),
        simulation / "metadata.yaml",
    )
    write_simulation_configuration(
        SimulationConfiguration((ConfigurationParameter("tff", 1.0, "Myr", "explicit"),), ()),
        simulation / "config.yaml",
    )
    index_catalogue(full)
    refresh_snapshot_times(full, "grid")
    lite = tmp_path / "lite"
    import_lite_collection(full, "grid", lite)
    lite_simulation = lite / "collections" / "grid" / "simulations" / "0001"
    result = lite_simulation / "derived" / "diagnostics" / "test" / "v1" / "series.h5"
    result.parent.mkdir(parents=True)
    with h5py.File(result, "w") as output:
        output.attrs["complete"] = True
        output.attrs["format_schema_version"] = 2
        output.attrs["collection_id"] = "grid"
        output.attrs["simulation_id"] = "0001"
        output.attrs["config_sha256"] = file_sha256(lite_simulation / "config.yaml")
        output.attrs["diagnostic_name"] = "test"
        output.attrs["diagnostic_version"] = 1
    first = import_derived(lite, full)
    destination = simulation / "derived" / "diagnostics" / "test" / "v1" / "series.h5"
    assert first.copied_paths == (destination,)
    assert destination.is_file()
    assert import_derived(lite, full).skipped_paths == (destination,)
    assert import_derived(lite, full, overwrite=True).copied_paths == (destination,)
