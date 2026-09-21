"""Tests for the researcher-facing data interfaces."""

from pathlib import Path, PurePosixPath
import math

import h5py
from pytest import approx

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
    collection_diagnostics_path,
    write_collection_diagnostics,
)
from csfdata_analysis.data.series import (
    aggregate_time_series,
    interpolate_time_series,
    load_time_series,
)
from csfdata_analysis.data.slices import select_snapshot_slice, select_time_series_slice


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
