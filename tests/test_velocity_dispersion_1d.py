"""Tests for standardized component velocity dispersions."""

import math

import pytest
from amuse.datamodel import Particles
from amuse.units import units

from csfdata_analysis.diagnostics.time_series.velocity_dispersion_1d import (
    measure_velocity_dispersion_1d,
)


def test_measure_velocity_dispersion_1d_records_mass_and_number_regions() -> None:
    """Component dispersions use all, mass, and number Lagrangian selections."""
    particles = Particles(4)
    particles.mass = [1.0, 1.0, 1.0, 1.0] | units.MSun
    particles.x = [0.0, 1.0, 2.0, 3.0] | units.pc
    particles.y = 0.0 | units.pc
    particles.z = 0.0 | units.pc
    particles.vx = [0.0, 1.0, 2.0, 3.0] | units.kms
    particles.vy = [0.0, 2.0, 4.0, 6.0] | units.kms
    particles.vz = [3.0, 3.0, 3.0, 3.0] | units.kms

    measurements = measure_velocity_dispersion_1d(particles)["origin"]

    assert measurements["n_sigma"].value_in(units.none) == 4
    assert measurements["sigma_vx"].value_in(units.kms) == pytest.approx(math.sqrt(1.25))
    assert measurements["sigma_vy"].value_in(units.kms) == pytest.approx(math.sqrt(5.0))
    assert measurements["sigma_vz"].value_in(units.kms) == pytest.approx(0.0)
    assert measurements["n_sigma_l50"].value_in(units.none) == 2
    assert math.isnan(measurements["sigma_vx_l50"].value_in(units.kms))
    assert measurements["n_sigma_l90"].value_in(units.none) == 4
    assert measurements["sigma_vx_l90"].value_in(units.kms) == pytest.approx(math.sqrt(1.25))
    assert measurements["n_sigma_n90"].value_in(units.none) == 4
    assert measurements["sigma_vy_n90"].value_in(units.kms) == pytest.approx(math.sqrt(5.0))
