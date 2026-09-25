"""Tests for the researcher-facing data interfaces."""

from pathlib import Path, PurePosixPath
import math
import sqlite3

import h5py
from pytest import approx, raises

import csfdata.catalogue.diagnostics as catalogue_diagnostics
from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.configuration import (
    ConfigurationParameter,
    SimulationConfiguration,
    write_simulation_configuration,
)
from csfdata.catalogue.diagnostics import (
    ChoiceDefinition,
    CollectionDiagnostics,
    DiagnosticDefinition,
    DiagnosticField,
    ScalarDiagnosticResult,
    ScalarValue,
    SimulationScalarDiagnostics,
    collection_diagnostics_path,
    simulation_scalar_diagnostics_path,
    write_collection_diagnostics,
    write_simulation_scalar_diagnostics,
)
from csfdata_analysis.datamodel.scalars import load_collection_scalars
from csfdata_analysis.datamodel.series import (
    aggregate_time_series,
    interpolate_time_series,
    load_collection_time_series,
    load_time_series,
)
from csfdata_analysis.datamodel.slices import select_snapshot_slice, select_time_series_slice
from csfdata_analysis.datamodel.simulations import SimulationSet, load_simulations


_MYR_IN_SECONDS = 365.25 * 24.0 * 60.0 * 60.0 * 1.0e6


def test_load_align_aggregate_and_slice_multiple_time_series(tmp_path: Path) -> None:
    """Multiple diagnostic fields load, align, summarize, and slice consistently."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    for simulation, radii, velocity in (
        (first, (1.0, 3.0), (2.0, 6.0)),
        (second, (2.0, 6.0), (4.0, 12.0)),
    ):
        _write_time_series(simulation, "radii", {"r50": radii, "n_stars": (10.0, 20.0)})
        _write_time_series(simulation, "velocity", {"mean_vr": velocity})

    data = load_time_series(
        (first, second),
        {
            ("radii", "v1"): {
                "choices": {"center": "stellar_com"},
                "fields": ("r50", "n_stars"),
            },
            ("velocity", "v1"): {
                "choices": {"center": "stellar_com"},
                "fields": ("mean_vr",),
            },
        },
    )
    aligned = interpolate_time_series(data, (0.0, 1.0, 3.0))
    summary = aggregate_time_series(aligned)
    sliced = select_time_series_slice(data, 0.5, normalization="tff")

    radii = data[("radii", "v1")]
    assert set(radii["tff"]) == {1.0, 2.0}
    assert list(radii.columns[-3:]) == ["time", "r50", "n_stars"]
    first_values = list(aligned[("radii", "v1")][lambda table: table["simulation_id"] == "0001"]["r50"])
    assert first_values[:2] == [1.0, 2.0]
    assert math.isnan(first_values[2])
    first_summary = summary[("radii", "v1")].iloc[1]
    assert first_summary["n_simulations"] == 2
    assert first_summary["r50_mean"] == approx(3.0)
    assert first_summary["r50_std"] == approx(2.0**0.5)
    assert summary[("radii", "v1")].iloc[2]["n_simulations"] == 0
    assert list(sliced[("velocity", "v1")]["time"]) == [0.5, 1.0]
    assert list(sliced[("velocity", "v1")]["mean_vr"]) == [3.0, 8.0]


def test_load_one_simulation_merges_default_and_overridden_diagnostics(tmp_path: Path) -> None:
    """One simulation loads one shared table with resolved diagnostic choices."""
    simulation = _write_simulation(tmp_path, "0001", 1.0)
    _write_time_series(simulation, "radii", {"r50": (1.0, 3.0), "n_stars": (10.0, 20.0)})
    _write_time_series(simulation, "velocity", {"mean_vr": (2.0, 6.0)})
    velocity_path = simulation.path / "derived/diagnostics/velocity/v1/series.h5"
    with h5py.File(velocity_path, "a") as output:
        origin = output["choices"].create_group("origin")
        origin.create_dataset("mean_vr", data=(4.0, 8.0))

    data = load_time_series(
        simulation,
        ("radii", "velocity"),
        choices={"center": "stellar_com"},
        diagnostic_choices={"velocity": {"center": "origin"}},
        fields={"velocity": ("mean_vr",)},
    )

    assert list(data.columns) == ["time", "r50", "n_stars", "mean_vr"]
    assert list(data["mean_vr"]) == [4.0, 8.0]
    assert data.attrs["diagnostics"] == (("radii", "v1"), ("velocity", "v1"))
    assert data.attrs["choices"] == {
        "radii": {"center": "stellar_com"},
        "velocity": {"center": "origin"},
    }
    assert data.attrs["fields"] == {
        "radii": ("r50", "n_stars"),
        "velocity": ("mean_vr",),
    }


def test_load_align_and_aggregate_a_collection_with_concise_selection(tmp_path: Path) -> None:
    """Collection tables use concise selections and retain grouping metadata."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    _write_time_series(first, "radii", {"r50": (1.0, 3.0), "n_stars": (10.0, 20.0)})
    _write_time_series(second, "radii", {"r50": (2.0, 6.0), "n_stars": (10.0, 20.0)})

    series = load_collection_time_series(
        (first, second),
        "radii",
        fields={"radii": ("r50",)},
    )
    aligned = interpolate_time_series(series, (0.0, 1.0, 3.0))
    summary = aggregate_time_series(aligned, group_by=("tff",))

    assert list(series.columns) == ["collection_id", "simulation_id", "tff", "time", "r50"]
    assert series.attrs["data_fields"] == ("r50",)
    first_values = list(aligned[aligned["simulation_id"] == "0001"]["r50"])
    assert first_values[:2] == [1.0, 2.0]
    assert math.isnan(first_values[2])
    assert summary.iloc[1]["r50_mean"] == approx(2.0)
    assert summary.iloc[1]["n_simulations"] == 1


def test_collection_loading_reads_each_schema_once(tmp_path: Path, monkeypatch) -> None:
    """One collection schema is parsed once during a concise collection load."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    for simulation in (first, second):
        _write_time_series(simulation, "radii", {"r50": (1.0, 3.0), "n_stars": (10.0, 20.0)})

    calls = 0
    read_collection_diagnostics = catalogue_diagnostics.read_collection_diagnostics

    def count_schema_reads(path: Path):
        """Count collection-schema reads while preserving normal parsing."""
        nonlocal calls
        calls += 1
        return read_collection_diagnostics(path)

    monkeypatch.setattr(catalogue_diagnostics, "read_collection_diagnostics", count_schema_reads)

    load_collection_time_series((first, second), "radii")

    assert calls == 1


def test_load_collection_scalars_adds_configuration_and_selected_fields(tmp_path: Path) -> None:
    """Scalar tables have one row per simulation and remain plot-ready."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    _write_scalar(first, "expansion_rate", 0.12)
    _write_scalar(second, "expansion_rate", 0.25)

    table = load_collection_scalars(
        (first, second),
        "expansion_rate",
        choices={"center": "stellar_com"},
        fields={"expansion_rate": ("dRdt",)},
    )

    assert list(table.columns) == ["collection_id", "simulation_id", "tff", "dRdt"]
    assert list(table["dRdt"]) == [0.12, 0.25]
    assert table.attrs["diagnostics"] == (("expansion_rate", "v1"),)
    assert table.attrs["choices"] == {"expansion_rate": {"center": "stellar_com"}}
    assert table.attrs["fields"] == {"expansion_rate": ("dRdt",)}


def test_simulation_set_wraps_selected_simulations_and_data_views(tmp_path: Path) -> None:
    """One read-only selection supports sequence access and data-view wrappers."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    for simulation in (first, second):
        _write_time_series(simulation, "radii", {"r50": (1.0, 3.0), "n_stars": (10.0, 20.0)})
    _write_scalar(first, "expansion_rate", 0.12)
    _write_scalar(second, "expansion_rate", 0.25)
    selection = SimulationSet(tmp_path / "catalogue", (first, second))

    assert len(selection) == 2
    assert selection[0] == first
    assert selection[:1] == SimulationSet(tmp_path / "catalogue", (first,))
    assert list(selection.parameters()["tff"]) == [1.0, 2.0]
    assert list(selection.series("radii", fields={"radii": ("r50",)})["r50"]) == [1.0, 3.0, 1.0, 3.0]
    assert list(selection.scalars("expansion_rate")["dRdt"]) == [0.12, 0.25]


def test_simulation_set_inventory_counts_declared_diagnostic_results(tmp_path: Path) -> None:
    """Inventory reports fields and indexed coverage without loading values."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    for simulation in (first, second):
        _write_time_series(simulation, "radii", {"r50": (1.0, 3.0), "n_stars": (10.0, 20.0)})
    _write_scalar(first, "expansion_rate", 0.12)
    _write_inventory_registry(tmp_path / "catalogue", first, second)
    selection = SimulationSet(tmp_path / "catalogue", (first, second))

    inventory = selection.inventory()
    data = inventory.data.set_index("diagnostic")

    assert inventory.diagnostics.series == ("radii", "velocity")
    assert inventory.diagnostics.scalar == ("expansion_rate",)
    assert inventory.fields["radii"] == ("r50", "n_stars")
    assert inventory.versions["radii"] == ("v1",)
    assert inventory.counts.loc[("time_series", "radii", "v1"), "available"] == 2
    assert data.loc["velocity", "available"] == 0
    assert data.loc["expansion_rate", "available"] == 1
    assert data.loc["expansion_rate", "total"] == 2


def test_simulation_set_inventory_requires_an_index(tmp_path: Path) -> None:
    """Inventory refuses to inspect result files when the registry is missing."""
    simulation = _write_simulation(tmp_path, "0001", 1.0)
    selection = SimulationSet(tmp_path / "catalogue", (simulation,))

    with raises(FileNotFoundError, match="csfdata index-catalogue"):
        selection.inventory()


def test_load_simulations_returns_a_simulation_set(tmp_path: Path, monkeypatch) -> None:
    """The public selection function wraps the core registry result."""
    simulation = _write_simulation(tmp_path, "0001", 1.0)

    def selected_simulations(*_args, **_kwargs):
        """Supply a known registry result without creating a SQLite fixture."""
        return (simulation,)

    monkeypatch.setattr(
        "csfdata_analysis.datamodel.simulations.find_simulations",
        selected_simulations,
    )

    selection = load_simulations(tmp_path / "catalogue")

    assert selection == SimulationSet(tmp_path / "catalogue", (simulation,))


def test_collection_scalar_loading_reads_each_schema_once(tmp_path: Path, monkeypatch) -> None:
    """A scalar collection load caches its shared diagnostic schema."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    _write_scalar(first, "expansion_rate", 0.12)
    _write_scalar(second, "expansion_rate", 0.25)

    calls = 0
    read_collection_diagnostics = catalogue_diagnostics.read_collection_diagnostics

    def count_schema_reads(path: Path):
        """Count collection-schema reads while preserving normal parsing."""
        nonlocal calls
        calls += 1
        return read_collection_diagnostics(path)

    monkeypatch.setattr(catalogue_diagnostics, "read_collection_diagnostics", count_schema_reads)

    load_collection_scalars((first, second), "expansion_rate")

    assert calls == 1


def test_select_snapshot_slice_uses_a_normalized_time(tmp_path: Path) -> None:
    """A normalized raw request records the nearest real snapshot and its offset."""
    simulation = _write_simulation(tmp_path, "0001", 2.0)
    raw_output = simulation.path / "raw/dcaf_output"
    raw_output.mkdir(parents=True)
    for index, time in enumerate((1.0, 3.0)):
        with h5py.File(raw_output / f"stars_{index:03}.amuse", "w") as snapshot:
            group = snapshot.create_group("data/0000000001")
            group.attrs["model_time"] = time * _MYR_IN_SECONDS

    slice_data = select_snapshot_slice((simulation,), 1.4, normalization="tff")

    assert slice_data.iloc[0]["target_time"] == 2.8
    assert slice_data.iloc[0]["snapshot_time"] == 3.0
    assert slice_data.iloc[0]["time_offset"] == approx(0.2)


def _write_simulation(root: Path, simulation_id: str, tff: float) -> CatalogueSimulation:
    """Create one minimal simulation with two published time-series schemas."""
    collection_root = root / "catalogue" / "collections" / "grid"
    diagnostics_path = collection_diagnostics_path(collection_root)
    if not diagnostics_path.exists():
        diagnostics_path.parent.mkdir(parents=True)
        write_collection_diagnostics(
            CollectionDiagnostics(
                choices=(
                    ChoiceDefinition(
                        "center",
                        "Reference centre for position-dependent measurements.",
                        ("stellar_com",),
                        "stellar_com",
                    ),
                ),
                diagnostics=(
                    DiagnosticDefinition(
                        "radii",
                        "v1",
                        "time_series",
                        "Synthetic radius results.",
                        PurePosixPath("derived/diagnostics/radii/v1/series.h5"),
                        (
                            DiagnosticField("r50", "Synthetic half-number radius.", "pc"),
                            DiagnosticField("n_stars", "Synthetic star count.", "1"),
                        ),
                        ("center",),
                    ),
                    DiagnosticDefinition(
                        "velocity",
                        "v1",
                        "time_series",
                        "Synthetic radial velocity results.",
                        PurePosixPath("derived/diagnostics/velocity/v1/series.h5"),
                        (DiagnosticField("mean_vr", "Synthetic mean radial velocity.", "km/s"),),
                        ("center",),
                    ),
                    DiagnosticDefinition(
                        "expansion_rate",
                        "v1",
                        "scalar",
                        "Synthetic expansion-rate fit.",
                        PurePosixPath("derived/scalar_diagnostics.yaml"),
                        (DiagnosticField("dRdt", "Synthetic expansion rate.", "km/s"),),
                        ("center",),
                    ),
                ),
            ),
            diagnostics_path,
        )
    path = collection_root / "simulations" / simulation_id
    path.mkdir(parents=True)
    write_simulation_configuration(
        SimulationConfiguration((ConfigurationParameter("tff", tff, "Myr", "explicit"),), ()),
        path / "config.yaml",
    )
    return CatalogueSimulation("grid", simulation_id, path, "dcaf")


def _write_time_series(
    simulation: CatalogueSimulation,
    diagnostic: str,
    fields: dict[str, tuple[float, float]],
) -> None:
    """Write one completed synthetic choice-specific time series for a test."""
    path = simulation.path / "derived" / "diagnostics" / diagnostic / "v1" / "series.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as output:
        output.attrs["complete"] = True
        output.attrs["format_schema_version"] = 2
        output.attrs["diagnostic_name"] = diagnostic
        output.attrs["diagnostic_version"] = 1
        output.create_dataset("time", data=(0.0, 2.0))
        choice = output.create_group("choices").create_group("stellar_com")
        for name, values in fields.items():
            choice.create_dataset(name, data=values)


def _write_scalar(
    simulation: CatalogueSimulation,
    diagnostic: str,
    value: float,
) -> None:
    """Write one synthetic scalar result using the collection schema."""
    diagnostics = simulation.diagnostics.scalar.collection_diagnostics
    path = simulation_scalar_diagnostics_path(simulation.path)
    path.parent.mkdir(exist_ok=True)
    write_simulation_scalar_diagnostics(
        SimulationScalarDiagnostics(
            (
                ScalarDiagnosticResult(
                    diagnostic,
                    "v1",
                    (("center", "stellar_com"),),
                    (ScalarValue("dRdt", value, "km/s"),),
                ),
            )
        ),
        diagnostics,
        path,
    )


def _write_inventory_registry(
    catalogue_root: Path,
    first: CatalogueSimulation,
    second: CatalogueSimulation,
) -> None:
    """Create the minimal indexed diagnostic coverage needed by inventory tests."""
    with sqlite3.connect(catalogue_root / "registry.sqlite") as connection:
        connection.executescript(
            """
            CREATE TABLE derived_values (
                collection_id TEXT,
                simulation_id TEXT,
                name TEXT,
                diagnostic_name TEXT,
                diagnostic_version TEXT,
                choice_key TEXT
            );
            CREATE TABLE time_series_products (
                collection_id TEXT,
                simulation_id TEXT,
                diagnostic_name TEXT,
                diagnostic_version TEXT
            );
            """
        )
        connection.executemany(
            "INSERT INTO time_series_products VALUES (?, ?, ?, ?)",
            (
                ("grid", first.simulation_id, "radii", "v1"),
                ("grid", second.simulation_id, "radii", "v1"),
            ),
        )
        connection.execute(
            "INSERT INTO derived_values VALUES (?, ?, ?, ?, ?, ?)",
            ("grid", first.simulation_id, "dRdt", "expansion_rate", "v1", '{"center":"stellar_com"}'),
        )
