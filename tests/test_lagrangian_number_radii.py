"""Tests for standardized stellar number-based Lagrangian radii."""

import pytest
from amuse.datamodel import Particles
from amuse.units import units

from csfdata_analysis.diagnostics.time_series.lagrangian_number_radii import (
    measure_lagrangian_number_radii,
)


def test_measure_lagrangian_number_radii_records_expected_boundaries() -> None:
    """Number fractions use ordered stars and include their boundary members."""
    particles = Particles(4)
    particles.mass = [1.0, 1.0, 1.0, 1.0] | units.MSun
    particles.x = [0.0, 1.0, 2.0, 3.0] | units.pc
    particles.y = 0.0 | units.pc
    particles.z = 0.0 | units.pc

    results = measure_lagrangian_number_radii(particles)

    assert results["origin"]["r_n50"].value_in(units.pc) == pytest.approx(1.0)
    assert results["origin"]["n_n50"].value_in(units.none) == 2
    assert results["origin"]["r_n95"].value_in(units.pc) == pytest.approx(3.0)
    assert results["origin"]["n_n95"].value_in(units.none) == 4
    assert results["stellar_com"]["centre_x"].value_in(units.pc) == pytest.approx(1.5)
    assert results["stellar_com"]["r_n50"].value_in(units.pc) == pytest.approx(0.5)
    assert results["stellar_com"]["n_n50"].value_in(units.none) == 2
