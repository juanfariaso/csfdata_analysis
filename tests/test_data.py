"""Tests for the researcher-facing data interfaces."""

from pathlib import Path, PurePosixPath
from dataclasses import replace
import math
import sqlite3

import h5py
import pandas
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
from csfdata.catalogue.snapshots import SnapshotTime, SnapshotTimeInventory
from csfdata_analysis.datamodel.scalars import DataScalars, load_collection_scalars
from csfdata_analysis.datamodel.series import (
    DataSeries,
    aggregate_time_series,
    interpolate_time_series,
    load_collection_time_series,
    load_time_series,
)
from csfdata_analysis.datamodel.slices import select_snapshot_slice, select_time_slice
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
    sliced = select_time_slice(data, 0.5, normalization="tff")

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


def test_scalar_loading_fills_missing_results_with_nan_by_default(tmp_path: Path) -> None:
    """Scalar tables retain selected simulations with unavailable values as NaN."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    _write_scalar(first, "expansion_rate", 0.12)
    selection = SimulationSet(tmp_path / "catalogue", (first, second))

    scalars = selection.scalars("expansion_rate", fields={"expansion_rate": ("dRdt",)})
    table = scalars.dataframe

    assert list(table["simulation_id"]) == ["0001", "0002"]
    assert table.loc[table["simulation_id"] == "0001", "dRdt"].item() == 0.12
    assert math.isnan(table.loc[table["simulation_id"] == "0002", "dRdt"].item())


def test_load_collection_scalars_expands_scalar_choice_into_long_table(tmp_path: Path) -> None:
    """Expanded scalar choices become explicit columns rather than field names."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    collection_root = first.path.parent.parent
    schema_path = collection_diagnostics_path(collection_root)
    schema = catalogue_diagnostics.read_collection_diagnostics(schema_path)
    write_collection_diagnostics(
        CollectionDiagnostics(
            schema.choices
            + (
                ChoiceDefinition(
                    "lagrangian_radius",
                    "Synthetic selected Lagrangian radius.",
                    ("r_l10", "r_l50"),
                    "r_l50",
                ),
            ),
            tuple(
                replace(definition, choices=("center", "lagrangian_radius"))
                if definition.name == "expansion_rate"
                else definition
                for definition in schema.diagnostics
            ),
        ),
        schema_path,
    )
    for simulation, values in ((first, (0.1, 0.5)), (second, (0.2, 0.6))):
        scalar_path = simulation_scalar_diagnostics_path(simulation.path)
        scalar_path.parent.mkdir(exist_ok=True)
        write_simulation_scalar_diagnostics(
            SimulationScalarDiagnostics(
                tuple(
                    ScalarDiagnosticResult(
                        "expansion_rate",
                        "v1",
                        (("center", "stellar_com"), ("lagrangian_radius", radius)),
                        (ScalarValue("dRdt", value, "km/s"),),
                    )
                    for radius, value in zip(("r_l10", "r_l50"), values, strict=True)
                )
            ),
            catalogue_diagnostics.read_collection_diagnostics(schema_path),
            scalar_path,
        )

    table = load_collection_scalars(
        (first, second),
        "expansion_rate",
        choices={"center": "stellar_com"},
        expand_choices={"expansion_rate": ("lagrangian_radius",)},
        fields={"expansion_rate": ("dRdt",)},
    )

    assert list(table["lagrangian_radius"]) == ["r_l10", "r_l10", "r_l50", "r_l50"]
    assert list(table["dRdt"]) == [0.1, 0.2, 0.5, 0.6]
    assert table.attrs["expanded_choices"] == {
        "expansion_rate": ("lagrangian_radius",)
    }


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
    parameters = selection.parameters()
    series = selection.series("radii", fields={"radii": ("r50",)})
    scalars = selection.scalars("expansion_rate")
    assert str(parameters["collection_id"].dtype) == "category"
    assert str(parameters["simulation_id"].dtype) == "category"
    assert isinstance(series, DataSeries)
    assert isinstance(scalars, DataScalars)
    assert str(series.dataframe["simulation_id"].dtype) == "category"
    assert str(scalars.dataframe["simulation_id"].dtype) == "category"
    assert list(parameters["tff"]) == [1.0, 2.0]
    assert list(series.dataframe["r50"]) == [1.0, 3.0, 1.0, 3.0]
    assert list(scalars.dataframe["dRdt"]) == [0.12, 0.25]
    assert series.diagnostics == ("radii",)
    assert scalars.diagnostics == ("expansion_rate",)
    assert "DataSeries(rows=4" in repr(series)
    assert "DataScalars(rows=2" in repr(scalars)


def test_data_views_select_exact_list_and_range_filters() -> None:
    """Research-facing views retain their type and metadata after selection."""
    table = pandas.DataFrame(
        {
            "collection_id": ["grid", "grid", "grid", "grid"],
            "simulation_id": ["0001", "0001", "0002", "0002"],
            "tff": [0.5, 0.5, 1.0, 1.0],
            "time": [0.0, 1.0, 0.0, 1.0],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    table.attrs["diagnostics"] = (("example", "v1"),)
    table.attrs["fields"] = {"example": ("value",)}
    series = DataSeries(table)
    scalars = DataScalars(table.drop(columns="time"))

    selected_series = series.select(
        {"simulation_id": ["0001"], "time": (0.5, None)}
    )
    selected_scalars = scalars.select(
        {"tff": (0.5, 1.0), "value": [1.0, 4.0]}
    )

    assert isinstance(selected_series, DataSeries)
    assert isinstance(selected_scalars, DataScalars)
    assert list(selected_series.dataframe["value"]) == [2.0]
    assert list(selected_scalars.dataframe["value"]) == [1.0, 4.0]
    assert selected_series.fields == {"example": ("value",)}
    assert selected_scalars.diagnostics == ("example",)


def test_data_views_report_invalid_filter_columns_and_ranges() -> None:
    """Selection mistakes identify the unsupported column or range."""
    table = pandas.DataFrame(
        {
            "collection_id": ["grid"],
            "simulation_id": ["0001"],
            "time": [0.0],
        }
    )

    with raises(KeyError, match="Unknown DataSeries filter column 'sfe'"):
        DataSeries(table).select({"sfe": 0.1})
    with raises(ValueError, match="exactly two bounds"):
        DataScalars(table.drop(columns="time")).select({"simulation_id": ("0001",)})


def test_time_series_aligns_before_seed_aggregation(tmp_path: Path) -> None:
    """Series aggregation requires and preserves one explicit shared time grid."""
    collection_root = tmp_path / "catalogue" / "collections" / "grid"
    collection_root.mkdir(parents=True)
    (collection_root / "collection.yaml").write_text(
        """schema_version: 1
id: grid
importer: dcaf
config_schema_version: 1
required_parameters: [tff, seed_index]
optional_parameters: []
grid_axes:
  tff: [0.5]
  seed_index: [0, 1]
lite:
  include: []
""",
        encoding="utf-8",
    )
    table = pandas.DataFrame(
        {
            "collection_id": ["grid"] * 5,
            "simulation_id": ["0001", "0001", "0002", "0002", "0002"],
            "tff": [0.5] * 5,
            "seed_index": [0, 0, 1, 1, 1],
            "time": [0.0, 2.0, 0.0, 1.0, 2.0],
            "radius": [0.0, 2.0, 0.0, 2.0, 4.0],
        }
    )
    table.attrs["catalogue_root"] = str(tmp_path / "catalogue")
    table.attrs["diagnostics"] = (("radius", "v1"),)
    table.attrs["fields"] = {"radius": ("radius",)}
    table.attrs["data_fields"] = ("radius",)
    series = DataSeries(table)

    with raises(ValueError, match="Call align_time"):
        series.aggregate_over()

    aligned = series.align_time((0.0, 1.0, 2.0))
    summary = aligned.aggregate_over()
    middle = summary.loc[summary["time"] == 1.0].iloc[0]

    assert isinstance(aligned, DataSeries)
    assert aligned.dataframe.attrs["aligned_times"] == (0.0, 1.0, 2.0)
    assert middle["radius_mean"] == approx(1.5)
    assert middle["radius_median"] == approx(1.5)
    assert middle["radius_std"] == approx(2.0**-0.5)
    assert middle["n_simulations"] == 2
    assert "seed_index" not in summary.columns


def test_scalar_aggregation_discovers_grid_axes_and_preserves_expanded_choices(
    tmp_path: Path,
) -> None:
    """Seed aggregation uses the collection grid and keeps diagnostic choices."""
    collection_root = tmp_path / "catalogue" / "collections" / "grid"
    collection_root.mkdir(parents=True)
    (collection_root / "collection.yaml").write_text(
        """schema_version: 1
id: grid
importer: dcaf
config_schema_version: 1
required_parameters: [tff, seed_index]
optional_parameters: []
grid_axes:
  tff: [0.5, 1.0]
  seed_index: [0, 1]
lite:
  include: []
""",
        encoding="utf-8",
    )
    table = pandas.DataFrame(
        {
            "collection_id": ["grid"] * 8,
            "simulation_id": ["0001", "0001", "0002", "0002", "0003", "0003", "0004", "0004"],
            "tff": [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            "seed_index": [0, 0, 1, 1, 0, 0, 1, 1],
            "lagrangian_number_radius": ["r_n10", "r_n50"] * 4,
            "dRdt": [0.1, 0.3, 0.3, 0.5, 0.2, 0.4, float("nan"), 0.8],
        }
    )
    table.attrs["catalogue_root"] = str(tmp_path / "catalogue")
    table.attrs["diagnostics"] = (("number_radius_expansion_rate", "v1"),)
    table.attrs["fields"] = {"number_radius_expansion_rate": ("dRdt",)}
    table.attrs["expanded_choices"] = {
        "number_radius_expansion_rate": ("lagrangian_number_radius",)
    }
    values = DataScalars(table)

    summary = values.aggregate_over()

    first = summary.loc[
        (summary["tff"] == 0.5)
        & (summary["lagrangian_number_radius"] == "r_n10")
    ].iloc[0]
    incomplete = summary.loc[
        (summary["tff"] == 1.0)
        & (summary["lagrangian_number_radius"] == "r_n10")
    ].iloc[0]
    assert len(summary) == 4
    assert first["dRdt_mean"] == approx(0.2)
    assert first["dRdt_median"] == approx(0.2)
    assert first["dRdt_std"] == approx(2.0**0.5 * 0.1)
    assert first["n_simulations"] == 2
    assert incomplete["dRdt_mean"] == approx(0.2)
    assert incomplete["n_simulations"] == 1
    assert "seed_index" not in summary.columns
    assert list(summary["lagrangian_number_radius"][:2]) == ["r_n10", "r_n50"]


def test_scalar_aggregation_accepts_explicit_grid_axes() -> None:
    """Explicit axes allow aggregation without catalogue provenance."""
    table = pandas.DataFrame(
        {
            "collection_id": ["grid", "grid"],
            "simulation_id": ["0001", "0002"],
            "tff": [0.5, 0.5],
            "seed_index": [0, 1],
            "dRdt": [0.1, 0.3],
        }
    )
    table.attrs["fields"] = {"expansion_rate": ("dRdt",)}

    summary = DataScalars(table).aggregate_over(grid_axes=("tff", "seed_index"))

    assert list(summary["tff"]) == [0.5]
    assert list(summary["dRdt_mean"]) == [approx(0.2)]
    assert list(summary["n_simulations"]) == [2]


def test_scalar_aggregation_accepts_named_and_custom_statistics() -> None:
    """Statistic names define columns and custom reducers receive clean values."""
    table = pandas.DataFrame(
        {
            "collection_id": ["grid", "grid", "grid"],
            "simulation_id": ["0001", "0002", "0003"],
            "seed_index": [0, 1, 2],
            "dRdt": [0.1, float("nan"), 0.5],
        }
    )
    table.attrs["fields"] = {"expansion_rate": ("dRdt",)}

    summary = DataScalars(table).aggregate_over(
        statistics={
            "minimum": "min",
            "span": lambda values: values.max() - values.min(),
        },
        grid_axes=("seed_index",),
    )

    assert list(summary["dRdt_minimum"]) == [0.1]
    assert list(summary["dRdt_span"]) == [0.4]
    assert list(summary["n_simulations"]) == [2]


def test_simulation_set_compiles_series_and_scalar_fields(tmp_path: Path) -> None:
    """A compiled table repeats one scalar result over each simulation series."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    for simulation in (first, second):
        _write_time_series(simulation, "radii", {"r50": (1.0, 3.0), "n_stars": (10.0, 20.0)})
    _write_scalar(first, "expansion_rate", 0.12)
    _write_scalar(second, "expansion_rate", 0.25)
    selection = SimulationSet(tmp_path / "catalogue", (first, second))

    series = selection.compile_dataframe(
        series={"radii": ("r50",)},
        scalars={"expansion_rate": ("dRdt",)},
    )
    data = series.dataframe

    assert isinstance(series, DataSeries)
    assert list(data["dRdt"]) == [0.12, 0.12, 0.25, 0.25]
    assert data.attrs["scalar_diagnostics"] == (("expansion_rate", "v1"),)
    assert data.attrs["scalar_fields"] == {"expansion_rate": ("dRdt",)}


def test_simulation_set_slices_existing_dataframe_rows(tmp_path: Path) -> None:
    """DataFrame slices allow subsets, masks, and nearest normalized times."""
    first = _write_simulation(tmp_path, "0001", 2.0)
    second = _write_simulation(tmp_path, "0002", 1.0)
    selection = SimulationSet(tmp_path / "catalogue", (first, second))
    data = pandas.DataFrame(
        {
            "collection_id": ["grid", "grid", "grid"],
            "simulation_id": ["0001", "0001", "0001"],
            "time": [1.0, 2.0, 3.0],
            "value": [10.0, 20.0, 30.0],
            "eligible": [False, True, True],
        }
    )
    data.attrs["example"] = "preserved"

    nearest = selection.slice_dataframe(data, time=1.4, normalization="tff")
    first_masked = selection.slice_dataframe(
        data,
        time="first",
        mask_field="eligible",
    )
    last_masked = selection.slice_dataframe(
        data,
        time="last",
        mask_field="eligible",
    )

    assert list(nearest["time"]) == [3.0]
    assert list(first_masked["time"]) == [2.0]
    assert list(last_masked["time"]) == [3.0]
    assert nearest.attrs["example"] == "preserved"


def test_simulation_set_dataframe_slice_rejects_unknown_simulations(tmp_path: Path) -> None:
    """A DataFrame may be a subset but cannot introduce unrelated models."""
    simulation = _write_simulation(tmp_path, "0001", 1.0)
    selection = SimulationSet(tmp_path / "catalogue", (simulation,))
    data = pandas.DataFrame(
        {
            "collection_id": ["grid"],
            "simulation_id": ["outside"],
            "time": [1.0],
        }
    )

    with raises(ValueError, match="grid/outside"):
        selection.slice_dataframe(data, time="last")


def test_select_time_slice_returns_one_compiled_row_per_simulation(tmp_path: Path) -> None:
    """A compiled time slice keeps invariant scalar values ready for comparison."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    for simulation in (first, second):
        _write_time_series(simulation, "radii", {"r50": (1.0, 3.0), "n_stars": (10.0, 20.0)})
    _write_scalar(first, "expansion_rate", 0.12)
    _write_scalar(second, "expansion_rate", 0.25)
    selection = SimulationSet(tmp_path / "catalogue", (first, second))
    data = selection.compile_dataframe(
        series={"radii": ("r50",)},
        scalars={"expansion_rate": ("dRdt",)},
    ).dataframe

    selected = select_time_slice(data, 0.5, normalization="tff")

    assert list(selected["time"]) == [0.5, 1.0]
    assert list(selected["r50"]) == [1.5, 2.0]
    assert list(selected["dRdt"]) == [0.12, 0.25]


def test_compiled_dataframe_keeps_series_with_missing_scalar_as_nan(tmp_path: Path) -> None:
    """Default compiled tables retain series rows whose scalar fit is unavailable."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    for simulation in (first, second):
        _write_time_series(simulation, "radii", {"r50": (1.0, 3.0), "n_stars": (10.0, 20.0)})
    _write_scalar(first, "expansion_rate", 0.12)
    selection = SimulationSet(tmp_path / "catalogue", (first, second))

    data = selection.compile_dataframe(
        series={"radii": ("r50",)},
        scalars={"expansion_rate": ("dRdt",)},
    ).dataframe

    assert list(data["dRdt"][:2]) == [0.12, 0.12]
    assert all(math.isnan(value) for value in data["dRdt"][2:])


def test_series_reports_an_unknown_requested_field_before_scanning_simulations(tmp_path: Path) -> None:
    """A field typo is not reported as missing data for every simulation."""
    simulation = _write_simulation(tmp_path, "0001", 1.0)
    _write_time_series(simulation, "velocity", {"mean_vr": (2.0, 6.0)})
    selection = SimulationSet(tmp_path / "catalogue", (simulation,))

    with raises(ValueError, match="Unknown fields for time-series diagnostic 'velocity': mean_vrr"):
        selection.series("velocity", fields={"velocity": ("mean_vrr",)})


def test_scalars_reports_an_unknown_requested_field_before_scanning_simulations(tmp_path: Path) -> None:
    """A scalar field typo is not reported as missing data for every simulation."""
    simulation = _write_simulation(tmp_path, "0001", 1.0)
    _write_scalar(simulation, "expansion_rate", 0.12)
    selection = SimulationSet(tmp_path / "catalogue", (simulation,))

    with raises(ValueError, match="Unknown fields for scalar diagnostic 'expansion_rate': drdt"):
        selection.scalars("expansion_rate", fields={"expansion_rate": ("drdt",)})


def test_simulation_set_scalars_reads_indexed_values(tmp_path: Path) -> None:
    """Indexed selections read scalar and configuration values from SQLite."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    _write_inventory_registry(
        tmp_path / "catalogue",
        first,
        second,
        include_second_scalar=True,
    )
    selection = SimulationSet(tmp_path / "catalogue", (first, second))

    values = selection.scalars("expansion_rate").dataframe

    assert list(values["tff"]) == [1.0, 2.0]
    assert list(values["dRdt"]) == [0.12, 0.25]


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


def test_snapshot_slice_selects_ordered_and_masked_snapshot_times(tmp_path: Path) -> None:
    """First, last, and masked snapshot selection use each model's time order."""
    simulation = _write_simulation(tmp_path, "0001", 1.0)
    raw_output = simulation.path / "raw/dcaf_output"
    raw_output.mkdir(parents=True)
    for index, time in enumerate((1.0, 3.0)):
        with h5py.File(raw_output / f"stars_{index:03}.amuse", "w") as snapshot:
            group = snapshot.create_group("data/0000000001")
            group.attrs["model_time"] = time * _MYR_IN_SECONDS

    first = select_snapshot_slice((simulation,), "first")
    last = select_snapshot_slice((simulation,), "last")
    from_data = select_snapshot_slice(
        (simulation,),
        "last",
        data=pandas.DataFrame(
            {
                "collection_id": ["grid"],
                "simulation_id": ["0001"],
                "time": [1.0],
            }
        ),
    )
    masked = select_snapshot_slice(
        (simulation,),
        "first",
        data=pandas.DataFrame(
            {
                "collection_id": ["grid", "grid"],
                "simulation_id": ["0001", "0001"],
                "time": [1.0, 3.0],
                "all_stars_present": [False, True],
            }
        ),
        mask_field="all_stars_present",
    )

    assert first.iloc[0]["snapshot_time"] == 1.0
    assert last.iloc[0]["snapshot_time"] == 3.0
    assert from_data.iloc[0]["snapshot_time"] == 1.0
    assert masked.iloc[0]["snapshot_time"] == 3.0


def test_simulation_set_gets_snapshot_paths_from_time_or_dataframe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Snapshot paths resolve direct and prepared DataFrame time requests."""
    simulation = _write_simulation(tmp_path, "0001", 2.0)
    raw_output = simulation.path / "raw/dcaf_output"
    raw_output.mkdir(parents=True)
    paths = []
    for index in range(2):
        path = raw_output / f"stars_{index:03}.amuse"
        path.touch()
        paths.append(path)
    inventory = SnapshotTimeInventory(
        "grid",
        "test-digest",
        {
            "0001": (
                SnapshotTime(paths[0].relative_to(simulation.path), 1.0, 0),
                SnapshotTime(paths[1].relative_to(simulation.path), 3.0, 0),
            )
        },
        {},
    )
    monkeypatch.setattr(
        "csfdata_analysis.datamodel.simulations.read_snapshot_times",
        lambda _root, _collection, _simulation_ids: inventory,
    )
    selection = SimulationSet(tmp_path / "catalogue", (simulation,))

    normalized = selection.get_snapshot_paths(time=1.4, normalization="tff", tolerance=0.5)
    last = selection.get_snapshot_paths(time="last")
    last_with_parameters = selection.get_snapshot_paths(
        time="last",
        include_parameters=True,
    )
    prepared = selection.get_snapshot_paths(
        pandas.DataFrame(
            {
                "collection_id": ["grid"],
                "simulation_id": ["0001"],
                "time": [1.1],
            }
        ),
        tolerance=0.2,
    )

    assert normalized.iloc[0]["requested_time"] == 2.8
    assert normalized.iloc[0]["snapshot_time"] == 3.0
    assert "tff" not in last.columns
    assert last_with_parameters.iloc[0]["tff"] == 2.0
    assert last.iloc[0]["local_path"] == paths[1]
    assert last.iloc[0]["source_catalogue_root"] == tmp_path / "catalogue"
    assert last.iloc[0]["source_path"] == Path(
        "collections/grid/simulations/0001/raw/dcaf_output/stars_001.amuse"
    )
    assert prepared.iloc[0]["local_path"] == paths[0]
    assert prepared.iloc[0]["time_offset"] == approx(-0.1)

    assert last.iloc[0]["simulation_id"] == "0001"


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
                        ("origin", "stellar_com"),
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
    include_second_scalar: bool = False,
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
                choice_key TEXT,
                value_type TEXT,
                numeric_value REAL,
                text_value TEXT
            );
            CREATE TABLE parameter_values (
                collection_id TEXT,
                simulation_id TEXT,
                name TEXT,
                value_type TEXT,
                numeric_value REAL,
                text_value TEXT
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
            "INSERT INTO parameter_values VALUES (?, ?, ?, ?, ?, ?)",
            (
                ("grid", first.simulation_id, "tff", "number", 1.0, None),
                ("grid", second.simulation_id, "tff", "number", 2.0, None),
            ),
        )
        scalar_rows = [
            (
                "grid",
                first.simulation_id,
                "dRdt",
                "expansion_rate",
                "v1",
                '{"center":"stellar_com"}',
                "number",
                0.12,
                None,
            )
        ]
        if include_second_scalar:
            scalar_rows.append(
                (
                    "grid",
                    second.simulation_id,
                    "dRdt",
                    "expansion_rate",
                    "v1",
                    '{"center":"stellar_com"}',
                    "number",
                    0.25,
                    None,
                )
            )
        connection.executemany(
            "INSERT INTO derived_values VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            scalar_rows,
        )
        connection.executemany(
            "INSERT INTO time_series_products VALUES (?, ?, ?, ?)",
            (
                ("grid", first.simulation_id, "radii", "v1"),
                ("grid", second.simulation_id, "radii", "v1"),
            ),
        )
