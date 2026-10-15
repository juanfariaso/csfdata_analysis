"""Tests for per-simulation diagnostic execution."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from amuse.datamodel import Particles
from amuse.units import units

from csfdata_analysis.diagnostics import (
    EvaluationChoice,
    ParticleField,
    ParticleSnapshotResult,
    SnapshotDiagnosticResult,
    TimeSeriesDiagnostic,
)
from csfdata_analysis._particle_series import read_particle_series
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
    diagnostic = TimeSeriesDiagnostic(
        name="example_radius",
        version=1,
        description="Example radius diagnostic used only for tests.",
        evaluate=lambda particles: {
            "origin": {
                "r_l50": 3.0 | units.pc,
                "n_l50": 4 | units.none,
            }
        },
        evaluation_choices=(
            EvaluationChoice(
                name="origin",
                metadata={"method": "coordinate_origin"},
                outputs={"r_l50": "pc", "n_l50": "1"},
            ),
        ),
        field_descriptions={
            "r_l50": "Example half-mass radius.",
            "n_l50": "Example count inside the half-mass radius.",
        },
    )

    report = compute_time_series(
        simulation_root,
        diagnostic,
        adapter,
        lambda path: Particles(),
    )

    assert report.snapshot_count == 1
    assert report.particle_output_path is None
    with h5py.File(report.output_path) as result:
        assert result.attrs["diagnostic_name"] == "example_radius"
        assert result.attrs["diagnostic_version"] == 1
        assert result["time"][:].tolist() == [2.5]
        assert result["snapshot_id"][:].tolist() == [b"dcaf_output/stars_0001.amuse"]
        assert result.attrs["format_schema_version"] == 2
        assert result["choices/origin"].attrs["method"] == "coordinate_origin"
        assert result["choices/origin/r_l50"][:].tolist() == [3.0]
        assert result["choices/origin/r_l50"].attrs["unit"] == "pc"
        assert result["choices/origin/n_l50"][:].tolist() == [4.0]
        assert result["choices/origin/n_l50"].attrs["unit"] == "1"


def test_compute_time_series_never_replaces_completed_output(tmp_path: Path) -> None:
    """Re-running requires explicit permission before it replaces output."""
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
    diagnostic = TimeSeriesDiagnostic(
        name="example_radius",
        version=1,
        description="Example radius diagnostic used only for tests.",
        evaluate=lambda particles: {"origin": {"r_l50": 3.0 | units.pc}},
        evaluation_choices=(
            EvaluationChoice(
                name="origin",
                metadata={"method": "coordinate_origin"},
                outputs={"r_l50": "pc"},
            ),
        ),
        field_descriptions={"r_l50": "Example half-mass radius."},
    )

    compute_time_series(simulation_root, diagnostic, adapter, lambda path: Particles())

    with pytest.raises(FileExistsError, match="already exists"):
        compute_time_series(simulation_root, diagnostic, adapter, lambda path: Particles())

    report = compute_time_series(
        simulation_root,
        diagnostic,
        adapter,
        lambda path: Particles(),
        overwrite=True,
    )

    assert report.output_path.is_file()


def test_compute_time_series_writes_optional_particle_rows(tmp_path: Path) -> None:
    """Particle diagnostics stream a long table beside the scalar series."""
    simulation_root = tmp_path / "0001"
    raw_root = simulation_root / "raw"
    snapshot_paths = tuple(
        raw_root / "dcaf_output" / f"stars_{number:04d}.amuse"
        for number in (1, 2)
    )
    for path in snapshot_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    adapter = SimpleNamespace(
        run_root=raw_root,
        snapshot_paths=lambda: snapshot_paths,
        snapshot_time=lambda path: float(snapshot_paths.index(path)),
    )
    returned = iter(
        (
            SnapshotDiagnosticResult(
                series={"canonical": {"n_present": 2 | units.none}},
                particles={
                    "canonical": ParticleSnapshotResult(
                        particle_key=np.array([10, 11]),
                        values={
                            "label": np.array([1, 0]),
                            "density": np.array([2.0, 3.0]) | units.MSun / units.pc**3,
                            "origin": np.array(["initial", "initial"]),
                        },
                    )
                },
            ),
            SnapshotDiagnosticResult(
                series={"canonical": {"n_present": 2 | units.none}},
                particles={
                    "canonical": ParticleSnapshotResult(
                        particle_key=np.array([10, 12]),
                        values={
                            "label": np.array([1, -1]),
                            "density": np.array([4.0, 5.0]) | units.MSun / units.pc**3,
                            "origin": np.array(["evolved", "new"]),
                        },
                    )
                },
            ),
        )
    )
    diagnostic = TimeSeriesDiagnostic(
        name="example_membership",
        version=1,
        description="Example particle-level diagnostic used only for tests.",
        evaluate=lambda particles: next(returned),
        evaluation_choices=(
            EvaluationChoice(
                name="canonical",
                metadata={"method": "test"},
                outputs={"n_present": "1"},
                particle_fields=(
                    ParticleField(
                        "label",
                        "Example classification label.",
                        "int8",
                        metadata={"undetermined": -1},
                    ),
                    ParticleField(
                        "density",
                        "Example particle density.",
                        "float64",
                        "Msun/pc^3",
                    ),
                    ParticleField("origin", "Example text value.", "string"),
                ),
            ),
        ),
        field_descriptions={"n_present": "Number of present particles."},
    )

    report = compute_time_series(
        simulation_root,
        diagnostic,
        adapter,
        lambda path: Particles(),
    )

    assert report.particle_output_path == report.output_path.with_name("particles.h5")
    stored = read_particle_series(report.particle_output_path)
    choice = stored.choices["canonical"]
    assert choice.snapshot_id.tolist() == [
        "dcaf_output/stars_0001.amuse",
        "dcaf_output/stars_0001.amuse",
        "dcaf_output/stars_0002.amuse",
        "dcaf_output/stars_0002.amuse",
    ]
    assert choice.particle_key.tolist() == [10, 11, 10, 12]
    assert choice.values["label"].tolist() == [1, 0, 1, -1]
    assert choice.values["density"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert choice.values["origin"].tolist() == ["initial", "initial", "evolved", "new"]
    with h5py.File(report.output_path) as scalar_file, h5py.File(
        report.particle_output_path
    ) as particle_file:
        assert scalar_file.attrs["particle_series_file"] == "particles.h5"
        assert scalar_file.attrs["series_id"] == particle_file.attrs["series_id"]
        assert particle_file["choices/canonical/density"].attrs["unit"] == "Msun/pc^3"
        assert particle_file["choices/canonical/label"].attrs["undetermined"] == -1


def test_invalid_particle_rows_leave_no_completed_product(tmp_path: Path) -> None:
    """A field/key length mismatch cannot publish either member of the bundle."""
    simulation_root = tmp_path / "0001"
    raw_root = simulation_root / "raw"
    snapshot_path = raw_root / "stars.amuse"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.touch()
    adapter = SimpleNamespace(
        run_root=raw_root,
        snapshot_paths=lambda: (snapshot_path,),
        snapshot_time=lambda path: 1.0,
    )
    diagnostic = TimeSeriesDiagnostic(
        name="invalid_membership",
        version=1,
        description="Invalid particle diagnostic used only for tests.",
        evaluate=lambda particles: SnapshotDiagnosticResult(
            series={"canonical": {"count": 2 | units.none}},
            particles={
                "canonical": ParticleSnapshotResult(
                    particle_key=np.array([10, 11]),
                    values={"label": np.array([1])},
                )
            },
        ),
        evaluation_choices=(
            EvaluationChoice(
                "canonical",
                {},
                {"count": "1"},
                (ParticleField("label", "Example label.", "int8"),),
            ),
        ),
        field_descriptions={"count": "Example count."},
    )

    with pytest.raises(ValueError, match="2 particle keys"):
        compute_time_series(
            simulation_root,
            diagnostic,
            adapter,
            lambda path: Particles(),
        )

    result_directory = simulation_root / "derived" / "diagnostics" / "invalid_membership" / "v1"
    assert not (result_directory / "series.h5").exists()
    assert not (result_directory / "particles.h5").exists()
