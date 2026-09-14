"""Built-in readiness checks for registered analysis diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h5py
from amuse.datamodel import Particles
from amuse.io import write_set_to_file
from amuse.units import units

from csfdata_analysis.diagnostics import DIAGNOSTICS
from csfdata_analysis.readers import read_stars
from csfdata_analysis.runner import compute_time_series


@dataclass(frozen=True)
class DiagnosticCheck:
    """Result of testing one registered diagnostic on standard particle data.

    Args:
        name: Registered diagnostic name.
        version: Registered diagnostic version.
        error: Failure reason, or ``None`` when the diagnostic passed.
    """

    name: str
    version: int
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Return whether the diagnostic passed its readiness check.

        Returns:
            ``True`` when the diagnostic produced its expected versioned HDF5
            output without an error.
        """
        return self.error is None


def test_diagnostics() -> tuple[DiagnosticCheck, ...]:
    """Run every registered diagnostic on one representative AMUSE snapshot.

    Returns:
        One readiness-check result for each diagnostic in ``DIAGNOSTICS``.

    Notes:
        This is a technical readiness check. It confirms a diagnostic can read
        standard stellar particles, satisfy its declared result schema, and
        write the correct versioned HDF5 file. It does not independently prove
        the scientific correctness of a diagnostic's formula.
    """
    particles = Particles(4)
    particles.mass = [1.0, 2.0, 3.0, 4.0] | units.MSun
    particles.x = [0.0, 1.0, 2.0, 3.0] | units.pc
    particles.y = [0.0, 1.0, 0.0, -1.0] | units.pc
    particles.z = [0.0, 0.0, 1.0, 1.0] | units.pc
    particles.vx = [0.0, 1.0, 0.0, -1.0] | units.kms
    particles.vy = [1.0, 0.0, -1.0, 0.0] | units.kms
    particles.vz = [0.0, 0.0, 1.0, -1.0] | units.kms

    checks: list[DiagnosticCheck] = []
    with TemporaryDirectory() as directory:
        for diagnostic in DIAGNOSTICS.values():
            simulation_root = Path(directory) / diagnostic.name
            raw_root = simulation_root / "raw"
            snapshot_path = raw_root / "dcaf_output" / "stars_000.amuse"
            snapshot_path.parent.mkdir(parents=True)
            write_set_to_file(particles, str(snapshot_path), "amuse")
            adapter = SimpleNamespace(
                run_root=raw_root,
                snapshot_paths=lambda: (snapshot_path,),
                snapshot_time=lambda path: 1.0,
            )
            try:
                report = compute_time_series(simulation_root, diagnostic, adapter, read_stars)
                expected_path = (
                    simulation_root
                    / "derived"
                    / "diagnostics"
                    / diagnostic.name
                    / f"v{diagnostic.version}"
                    / "series.h5"
                )
                if report.output_path != expected_path:
                    raise ValueError(f"Output path differs from declared version: {report.output_path}")
                with h5py.File(report.output_path) as output_file:
                    if output_file.attrs["diagnostic_name"] != diagnostic.name:
                        raise ValueError("Output diagnostic name differs from declaration.")
                    if output_file.attrs["diagnostic_version"] != diagnostic.version:
                        raise ValueError("Output diagnostic version differs from declaration.")
                    for choice in diagnostic.choices:
                        for output_name, unit in choice.outputs.items():
                            dataset = output_file[f"choices/{choice.name}/{output_name}"]
                            if dataset.shape != (1,) or dataset.attrs["unit"] != unit:
                                raise ValueError(
                                    f"Output {choice.name}/{output_name} differs from declaration."
                                )
            except Exception as error:
                checks.append(
                    DiagnosticCheck(diagnostic.name, diagnostic.version, f"{type(error).__name__}: {error}")
                )
            else:
                checks.append(DiagnosticCheck(diagnostic.name, diagnostic.version))
    return tuple(checks)
