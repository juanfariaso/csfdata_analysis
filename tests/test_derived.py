"""Tests for safe import of lite-catalogue derived data."""

from pathlib import Path, PurePosixPath
import shutil

import h5py

from csfdata.catalogue.configuration import (
    ConfigurationParameter,
    SimulationConfiguration,
    write_simulation_configuration,
)
from csfdata.catalogue.diagnostics import (
    CollectionDiagnostics,
    DiagnosticDefinition,
    DiagnosticField,
    ScalarDiagnosticResult,
    ScalarValue,
    SimulationScalarDiagnostics,
    write_collection_diagnostics,
    write_simulation_scalar_diagnostics,
)
from csfdata.catalogue.lite import file_sha256, import_lite_collection
from csfdata.catalogue.metadata import SimulationMetadata, write_simulation_metadata
from csfdata.catalogue.registry import find_simulations, index_catalogue
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


def test_import_derived_accepts_a_full_catalogue_source(tmp_path: Path) -> None:
    """A full source imports every compatible collection by default."""
    source = tmp_path / "source"
    simulation = source / "collections" / "grid" / "simulations" / "0001"
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
        SimulationMetadata("0001", "grid", "dcaf", "host", tmp_path / "run", Path("run"), "2026-01-01T00:00:00Z"),
        simulation / "metadata.yaml",
    )
    write_simulation_configuration(
        SimulationConfiguration((ConfigurationParameter("tff", 1.0, "Myr", "explicit"),), ()),
        simulation / "config.yaml",
    )
    destination = tmp_path / "destination"
    shutil.copytree(source, destination)

    result = simulation / "derived" / "diagnostics" / "test" / "v1" / "series.h5"
    result.parent.mkdir(parents=True)
    with h5py.File(result, "w") as output:
        output.attrs["complete"] = True
        output.attrs["format_schema_version"] = 2
        output.attrs["collection_id"] = "grid"
        output.attrs["simulation_id"] = "0001"
        output.attrs["config_sha256"] = file_sha256(simulation / "config.yaml")
        output.attrs["diagnostic_name"] = "test"
        output.attrs["diagnostic_version"] = 1

    report = import_derived(source, destination)

    assert report.copied_paths == (
        destination / "collections" / "grid" / "simulations" / "0001" / result.relative_to(simulation),
    )


def test_import_derived_merges_scalar_results_for_indexing(tmp_path: Path) -> None:
    """Scalar values and their declarations reach the destination together."""
    source = tmp_path / "source"
    simulation = source / "collections" / "grid" / "simulations" / "0001"
    simulation.mkdir(parents=True)
    collection = simulation.parent.parent
    (collection / "collection.yaml").write_text(
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
        SimulationMetadata("0001", "grid", "dcaf", "host", tmp_path / "run", Path("run"), "2026-01-01T00:00:00Z"),
        simulation / "metadata.yaml",
    )
    write_simulation_configuration(
        SimulationConfiguration((ConfigurationParameter("tff", 1.0, "Myr", "explicit"),), ()),
        simulation / "config.yaml",
    )
    destination = tmp_path / "destination"
    shutil.copytree(source, destination)

    diagnostics = CollectionDiagnostics(
        (),
        (
            DiagnosticDefinition(
                "test_scalar",
                "v1",
                "scalar",
                "Test scalar diagnostic.",
                PurePosixPath("derived/scalar_diagnostics.yaml"),
                (DiagnosticField("test_value", "Test queryable value.", "1"),),
            ),
        ),
    )
    write_collection_diagnostics(diagnostics, collection / "diagnostics.yaml")
    scalar_path = simulation / "derived" / "scalar_diagnostics.yaml"
    scalar_path.parent.mkdir()
    write_simulation_scalar_diagnostics(
        SimulationScalarDiagnostics(
            (
                ScalarDiagnosticResult(
                    "test_scalar",
                    "v1",
                    (),
                    (ScalarValue("test_value", 2.0, "1"),),
                ),
            )
        ),
        diagnostics,
        scalar_path,
    )

    report = import_derived(source, destination)
    destination_collection = destination / "collections" / "grid"
    index_catalogue(destination)

    assert destination_collection.joinpath("diagnostics.yaml").is_file()
    assert report.copied_paths == (
        destination_collection / "simulations" / "0001" / "derived" / "scalar_diagnostics.yaml",
    )
    assert len(find_simulations(destination, filters={"test_value": 2.0})) == 1
