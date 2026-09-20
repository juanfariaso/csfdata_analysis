"""Tests for standardized stellar Lagrangian radii."""

import pytest
from amuse.datamodel import Particles
from amuse.units import units

from csfdata_analysis.diagnostics.time_series.lagrangian_radii import (
    measure_lagrangian_radii,
)


def test_measure_lagrangian_radii_records_origin_and_stellar_com() -> None:
    """The two standard centres produce their expected half-mass radii."""
    particles = Particles(4)
    particles.mass = [1.0, 1.0, 1.0, 1.0] | units.MSun
    particles.x = [0.0, 1.0, 2.0, 3.0] | units.pc
    particles.y = 0.0 | units.pc
    particles.z = 0.0 | units.pc

    results = measure_lagrangian_radii(particles)

    assert results["origin"]["r_l50"].value_in(units.pc) == pytest.approx(1.0)
    assert results["origin"]["stellar_mass"].value_in(units.MSun) == pytest.approx(4.0)
    assert results["origin"]["n_stars"].value_in(units.none) == 4
    assert results["origin"]["n_l50"].value_in(units.none) == 2
    assert results["stellar_com"]["centre_x"].value_in(units.pc) == pytest.approx(1.5)
    assert results["stellar_com"]["r_l50"].value_in(units.pc) == pytest.approx(0.5)
    assert results["stellar_com"]["n_l50"].value_in(units.none) == 2
