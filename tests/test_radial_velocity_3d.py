"""Tests for standardized three-dimensional radial-velocity statistics."""

import pytest
from amuse.datamodel import Particles
from amuse.units import units

from csfdata_analysis.diagnostics.time_series.radial_velocity_3d import (
    measure_radial_velocity_3d,
)


def test_measure_radial_velocity_3d_records_mean_and_median_statistics() -> None:
    """Radially expanding stars have the expected all-star velocity statistics."""
    particles = Particles(4)
    particles.mass = [1.0, 1.0, 1.0, 1.0] | units.MSun
    particles.x = [-2.0, -1.0, 1.0, 2.0] | units.pc
    particles.y = 0.0 | units.pc
    particles.z = 0.0 | units.pc
    particles.vx = [-2.0, -1.0, 1.0, 2.0] | units.kms
    particles.vy = 0.0 | units.kms
    particles.vz = 0.0 | units.kms

    results = measure_radial_velocity_3d(particles)

    for center in ("origin", "stellar_com"):
        measurements = results[center]
        assert measurements["n_vr"].value_in(units.none) == 4
        assert measurements["mean_vr"].value_in(units.kms) == pytest.approx(1.5)
        assert measurements["median_vr"].value_in(units.kms) == pytest.approx(1.5)
        assert measurements["sigma_vr"].value_in(units.kms) == pytest.approx(0.5)
        assert measurements["mean_vr_over_sigma_vr"].value_in(units.none) == pytest.approx(3.0)
        assert measurements["median_vr_over_sigma_vr"].value_in(units.none) == pytest.approx(3.0)
        assert measurements["mean_vr_l90"].value_in(units.kms) == pytest.approx(1.5)
        assert measurements["n_vr_l90"].value_in(units.none) == 4
