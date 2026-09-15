"""Tests for the researcher-facing data interfaces."""

from pathlib import Path
import math

import h5py
from pytest import approx

from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.configuration import (
    ConfigurationParameter,
    SimulationConfiguration,
    write_simulation_configuration,
)
from csfdata_analysis.data.series import interpolate_time_series, load_time_series
from csfdata_analysis.data.slices import select_snapshot_slice


_MYR_IN_SECONDS = 365.25 * 24.0 * 60.0 * 60.0 * 1.0e6


def test_load_and_interpolate_time_series(tmp_path: Path) -> None:
    """Stored series become a labelled table and align without extrapolation."""
    first = _write_simulation(tmp_path, "0001", 1.0)
    second = _write_simulation(tmp_path, "0002", 2.0)
    for simulation, values in ((first, (1.0, 3.0)), (second, (2.0, 6.0))):
        result = simulation.path / "derived/diagnostics/test/v1/series.h5"
        result.parent.mkdir(parents=True)
        with h5py.File(result, "w") as output:
            output.attrs["complete"] = True
            output.attrs["diagnostic_name"] = "test"
            output.attrs["diagnostic_version"] = 1
            output.create_dataset("time_myr", data=(0.0, 2.0))
            output.create_group("choices").create_group("choice").create_dataset("value", data=values)

    data = load_time_series((first, second), "test", 1, "choice", "value")
    aligned = interpolate_time_series(data, (0.0, 1.0, 3.0), "value")

    assert set(data["tff"]) == {1.0, 2.0}
    first_values = list(aligned[aligned["simulation_id"] == "0001"]["value"])
    assert first_values[:2] == [1.0, 2.0]
    assert math.isnan(first_values[2])
    assert aligned[aligned["simulation_id"] == "0002"].iloc[1]["value"] == 4.0


def test_select_snapshot_slice_uses_a_normalized_time(tmp_path: Path) -> None:
    """A normalized request records the nearest real snapshot and its offset."""
    simulation = _write_simulation(tmp_path, "0001", 2.0)
    raw_output = simulation.path / "raw/dcaf_output"
    raw_output.mkdir(parents=True)
    for index, time_myr in enumerate((1.0, 3.0)):
        with h5py.File(raw_output / f"stars_{index:03}.amuse", "w") as snapshot:
            group = snapshot.create_group("data/0000000001")
            group.attrs["model_time"] = time_myr * _MYR_IN_SECONDS

    slice_data = select_snapshot_slice((simulation,), 1.4, normalization="tff")

    assert slice_data.iloc[0]["target_time_myr"] == 2.8
    assert slice_data.iloc[0]["snapshot_time_myr"] == 3.0
    assert slice_data.iloc[0]["time_offset_myr"] == approx(0.2)


def _write_simulation(root: Path, simulation_id: str, tff: float) -> CatalogueSimulation:
    """Create one minimal catalogue simulation used by data-interface tests."""
    path = root / simulation_id
    path.mkdir()
    write_simulation_configuration(
        SimulationConfiguration((ConfigurationParameter("tff", tff, "Myr", "explicit"),), ()),
        path / "config.yaml",
    )
    return CatalogueSimulation("grid", simulation_id, path, "dcaf")
