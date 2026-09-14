"""Tests for per-simulation diagnostic execution."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
from amuse.datamodel import Particles
from amuse.units import units

from csfdata_analysis.diagnostics import Diagnostic, DiagnosticChoice
from csfdata_analysis.runner import compute_time_series


def test_compute_time_series_writes_a_versioned_time_series(tmp_path: Path) -> None:
    """One simulation produces one unit-labelled HDF5 time series."""
    simulation_root = tmp_path / "0001"
    raw_root = simulation_root / "raw"
    snapshot_path = raw_root / "dcaf_output" / "stars_0001.amuse"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.touch()
    adapter = SimpleNamespace(
        run_root=raw_root,
        snapshot_paths=lambda: (snapshot_path,),
        snapshot_time=lambda path: 2.5,
    )
    diagnostic = Diagnostic(
        name="example_radius",
        version=1,
        evaluate=lambda particles: {"origin": {"r_l50": 3.0 | units.pc}},
        choices=(
            DiagnosticChoice(
                name="origin",
                metadata={"method": "coordinate_origin"},
                outputs={"r_l50": "pc"},
            ),
        ),
    )

    report = compute_time_series(
        simulation_root,
        diagnostic,
        adapter,
        lambda path: Particles(),
    )

    assert report.snapshot_count == 1
    with h5py.File(report.output_path) as result:
        assert result.attrs["diagnostic_name"] == "example_radius"
        assert result.attrs["diagnostic_version"] == 1
        assert result["time_myr"][:].tolist() == [2.5]
        assert result["snapshot_id"][:].tolist() == [b"dcaf_output/stars_0001.amuse"]
        assert result.attrs["format_schema_version"] == 1
        assert result["choices/origin"].attrs["method"] == "coordinate_origin"
        assert result["choices/origin/r_l50"][:].tolist() == [3.0]
        assert result["choices/origin/r_l50"].attrs["unit"] == "pc"


def test_compute_time_series_never_replaces_completed_output(tmp_path: Path) -> None:
    """Re-running one diagnostic version fails instead of overwriting it."""
    simulation_root = tmp_path / "0001"
    raw_root = simulation_root / "raw"
    snapshot_path = raw_root / "dcaf_output" / "stars_0001.amuse"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.touch()
    adapter = SimpleNamespace(
        run_root=raw_root,
        snapshot_paths=lambda: (snapshot_path,),
        snapshot_time=lambda path: 2.5,
    )
    diagnostic = Diagnostic(
        name="example_radius",
        version=1,
        evaluate=lambda particles: {"origin": {"r_l50": 3.0 | units.pc}},
        choices=(
            DiagnosticChoice(
                name="origin",
                metadata={"method": "coordinate_origin"},
                outputs={"r_l50": "pc"},
            ),
        ),
    )

    compute_time_series(simulation_root, diagnostic, adapter, lambda path: Particles())

    with pytest.raises(FileExistsError, match="already exists"):
        compute_time_series(simulation_root, diagnostic, adapter, lambda path: Particles())
