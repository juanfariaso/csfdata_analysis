"""Tests for standardized three-dimensional kappa expansion slopes."""

import numpy as np
import pytest
from amuse.datamodel import Particles
from amuse.units import units

from csfdata_analysis.diagnostics.time_series.kappa_3d import measure_kappa_3d


def test_measure_kappa_3d_records_outer_coordinate_slope_and_moments() -> None:
    """A linear x expansion gives the expected x slope outside the number r50."""
    positions = np.concatenate((-np.arange(1.0, 21.0), np.arange(1.0, 21.0)))
    particles = Particles(len(positions))
    particles.mass = 1.0 | units.MSun
    particles.x = positions | units.pc
    particles.y = 0.0 | units.pc
    particles.z = 0.0 | units.pc
    particles.vx = 2.0 * positions | units.kms
    particles.vy = 0.0 | units.kms
    particles.vz = 0.0 | units.kms

    results = measure_kappa_3d(particles)

    for center in ("origin", "stellar_com"):
        measurements = results[center]
        assert measurements["n_total"].value_in(units.none) == 40
        assert measurements["n_used"].value_in(units.none) == 20
        assert measurements["kappa_x"].value_in(units.kms / units.pc) == pytest.approx(2.0)
        assert measurements["r2_x"].value_in(units.none) == pytest.approx(1.0)
        assert measurements["cov_x_vx"].value_in(units.pc * units.kms) > 0.0
        assert measurements["var_x"].value_in(units.pc**2) > 0.0
        assert measurements["var_vx"].value_in(units.kms**2) > 0.0
        assert np.isnan(measurements["kappa_y"].value_in(units.kms / units.pc))
