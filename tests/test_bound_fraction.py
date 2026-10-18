"""Tests for the instantaneous bound-fraction diagnostic."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from amuse.datamodel import Particles
from amuse.units import units
from csfdata.catalogue import CatalogueSimulation

import csfdata_analysis.diagnostics.time_series.bound_fraction as bound_fraction_module
from csfdata_analysis.diagnostics import ensure_collection_diagnostics
from csfdata_analysis.diagnostics.time_series.bound_fraction import (
    BOUND_FRACTION_V1,
    calculate_bound_fraction,
    measure_bound_fraction,
)
from csfdata_analysis.runner import compute_time_series


def two_star_system(speed: float = 0.0) -> Particles:
    """Return two equal stars separated by one parsec in their COM frame."""
    particles = Particles(2)
    particles.mass = [1.0, 1.0] | units.MSun
    particles.x = [-0.5, 0.5] | units.pc
    particles.y = 0.0 | units.pc
    particles.z = 0.0 | units.pc
    particles.vx = [-speed, speed] | units.kms
    particles.vy = 0.0 | units.kms
    particles.vz = 0.0 | units.kms
    return particles


def test_stationary_pair_is_bound() -> None:
    """A stationary equal-mass pair has negative specific energy."""
    result = calculate_bound_fraction(two_star_system())

    assert result.bound.tolist() == [1, 1]
    assert result.n_bound == 2
    assert result.bound_number_fraction == pytest.approx(1.0)
    assert result.bound_mass_fraction == pytest.approx(1.0)
    assert result.n_iterations == 1
    assert result.converged is True
    assert result.centre_position.value_in(units.pc).tolist() == pytest.approx(
        [0.0, 0.0, 0.0]
    )


def test_fast_pair_converges_to_an_empty_bound_set() -> None:
    """Stars moving above their mutual escape speed become self-unbound."""
    result = calculate_bound_fraction(two_star_system(speed=1.0))

    assert result.bound.tolist() == [0, 0]
    assert result.n_bound == 0
    assert result.bound_mass.value_in(units.MSun) == pytest.approx(0.0)
    assert result.n_iterations == 2
    assert result.converged is True
    assert np.all(np.isnan(result.centre_position.value_in(units.pc)))


def test_invalid_particle_is_labelled_and_excluded_from_fractions() -> None:
    """An invalid present particle receives -1 without changing denominators."""
    particles = Particles(3)
    particles.mass = [1.0, 1.0, np.nan] | units.MSun
    particles.x = [-0.5, 0.5, 0.0] | units.pc
    particles.y = 0.0 | units.pc
    particles.z = 0.0 | units.pc
    particles.velocity = [0.0, 0.0, 0.0] | units.kms

    result = calculate_bound_fraction(particles)

    assert result.bound.tolist() == [1, 1, -1]
    assert result.n_present == 3
    assert result.n_valid == 2
    assert result.bound_number_fraction == pytest.approx(1.0)
    assert result.stellar_mass.value_in(units.MSun) == pytest.approx(2.0)


def test_potential_is_calculated_once_before_membership_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Membership updates reuse the fixed potential of all valid stars."""
    original = bound_fraction_module.gravitational_potential
    call_count = 0

    def counted_potential(*args: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return original(*args)

    monkeypatch.setattr(
        bound_fraction_module,
        "gravitational_potential",
        counted_potential,
    )

    result = calculate_bound_fraction(two_star_system(speed=1.0))

    assert result.n_iterations == 2
    assert call_count == 1


def test_iteration_limit_returns_latest_membership_without_claiming_convergence() -> None:
    """The safety ceiling preserves the latest labels and reports its status."""
    result = calculate_bound_fraction(
        two_star_system(speed=1.0),
        max_iterations=1,
    )

    assert result.bound.tolist() == [0, 0]
    assert result.n_iterations == 1
    assert result.converged is False


def test_measurement_exposes_scalars_and_labels_for_storage() -> None:
    """The registered evaluator returns aligned scalar and particle results."""
    particles = two_star_system()

    measured = measure_bound_fraction(particles)
    scalars = measured.series["stellar_self_potential"]
    particle_result = measured.particles["stellar_self_potential"]

    assert scalars["n_present"].value_in(units.none) == 2
    assert scalars["n_bound"].value_in(units.none) == 2
    assert scalars["bound_mass_fraction"].value_in(units.none) == pytest.approx(1.0)
    assert particle_result.particle_key.tolist() == particles.key.tolist()
    assert particle_result.values["bound"].tolist() == [1, 1]


def test_registered_diagnostic_writes_scalar_and_particle_products(
    tmp_path: Path,
) -> None:
    """The complete version-1 definition passes the generic runner contract."""
    collection_root = tmp_path / "example-collection"
    simulation_root = collection_root / "simulations" / "0001"
    raw_root = simulation_root / "raw"
    snapshot_path = raw_root / "stars.amuse"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.touch()
    adapter = SimpleNamespace(
        run_root=raw_root,
        snapshot_paths=lambda: (snapshot_path,),
        snapshot_time=lambda path: 1.0,
    )

    report = compute_time_series(
        simulation_root,
        BOUND_FRACTION_V1,
        adapter,
        lambda path: two_star_system(),
    )

    assert report.particle_output_path is not None
    with h5py.File(report.output_path) as scalar_file:
        assert scalar_file[
            "choices/stellar_self_potential/bound_number_fraction"
        ][0] == pytest.approx(1.0)
    with h5py.File(report.particle_output_path) as particle_file:
        assert particle_file[
            "choices/stellar_self_potential/bound"
        ][:].tolist() == [1, 1]

    ensure_collection_diagnostics(collection_root, (BOUND_FRACTION_V1,))
    simulation = CatalogueSimulation(
        collection_id="example-collection",
        simulation_id="0001",
        path=simulation_root,
        importer="dcaf",
    )
    stored = simulation.diagnostics.time_series[("bound_fraction", "v1")].read(
        choices={"binding_potential": "stellar_self_potential"},
        fields=("bound_number_fraction",),
    )
    assert stored["bound_number_fraction"][0] == pytest.approx(1.0)
