"""Three-dimensional stellar radial-velocity statistics through time."""

from __future__ import annotations

import numpy as np
from amuse.datamodel import Particles
from amuse.units import units
from amuse.units.quantities import Quantity

from csfdata_analysis.diagnostics.base import EvaluationChoice, TimeSeriesDiagnostic
from csfdata_analysis.diagnostics.time_series.lagrangian_radii import LAGRANGIAN_FRACTIONS


RADIAL_VELOCITY_OUTPUTS = {
    "n_vr": "1",
    "mean_vr": "km/s",
    "median_vr": "km/s",
    "sigma_vr": "km/s",
    "mean_vr_over_sigma_vr": "1",
    "median_vr_over_sigma_vr": "1",
    **{
        f"{name}_{suffix}": unit
        for suffix, _ in LAGRANGIAN_FRACTIONS
        for name, unit in (
            ("n_vr", "1"),
            ("mean_vr", "km/s"),
            ("median_vr", "km/s"),
            ("sigma_vr", "km/s"),
            ("mean_vr_over_sigma_vr", "1"),
            ("median_vr_over_sigma_vr", "1"),
        )
    },
}
"""Canonical fields and units for each radial-velocity centre choice."""


def measure_radial_velocity_3d(
    particles: Particles,
) -> dict[str, dict[str, Quantity]]:
    """Measure outward stellar velocities for all and Lagrangian subsets.

    For each centre, the calculation subtracts the mean bulk velocity of the
    selected stars before projecting each remaining velocity onto its radial
    direction. It reports the mean and median radial velocity, their shared
    radial-velocity dispersion, and both values normalized by that dispersion.

    Args:
        particles: Stellar AMUSE particles with finite positions in pc,
            velocities in km/s, and positive finite masses.

    Returns:
        Results grouped as ``"origin"`` and ``"stellar_com"``. Each group
        contains all-star fields and fields for every standard cumulative
        stellar-mass Lagrangian region.

    Raises:
        ValueError: If no particles have finite phase-space coordinates and
            positive finite masses.

    Notes:
        A region with fewer than three stars away from its selected centre
        stores ``NaN`` velocity statistics. Lagrangian membership follows the
        legacy first-paper convention: radius-sort the valid stars and include
        every star through the first cumulative-mass crossing.
    """
    # Convert once into canonical units, then apply one physically valid star
    # selection shared by centre definitions and all Lagrangian subsets.
    positions = np.column_stack(
        (
            particles.x.value_in(units.pc),
            particles.y.value_in(units.pc),
            particles.z.value_in(units.pc),
        )
    )
    velocities = np.column_stack(
        (
            particles.vx.value_in(units.kms),
            particles.vy.value_in(units.kms),
            particles.vz.value_in(units.kms),
        )
    )
    masses = np.asarray(particles.mass.value_in(units.MSun), dtype=float)
    valid = (
        np.all(np.isfinite(positions), axis=1)
        & np.all(np.isfinite(velocities), axis=1)
        & np.isfinite(masses)
        & (masses > 0.0)
    )
    if not np.any(valid):
        raise ValueError(
            "Radial velocity requires finite positions, velocities, and positive masses."
        )
    positions = positions[valid]
    velocities = velocities[valid]
    masses = masses[valid]
    total_mass = float(np.sum(masses))
    centres = {
        "origin": np.zeros(3),
        "stellar_com": np.sum(positions * masses[:, None], axis=0) / total_mass,
    }

    results: dict[str, dict[str, Quantity]] = {}
    for choice_name, centre in centres.items():
        # Define cumulative Lagrangian subsets from centre-relative radii
        # before calculating their independent bulk and radial velocities.
        relative_positions = positions - centre
        radii = np.sqrt(np.sum(relative_positions**2, axis=1))
        order = np.argsort(radii)
        cumulative_mass = np.cumsum(masses[order])
        selections = [("", np.arange(len(positions)))]
        for suffix, fraction in LAGRANGIAN_FRACTIONS:
            crossing = int(np.searchsorted(cumulative_mass, fraction * total_mass, side="left"))
            selections.append((f"_{suffix}", order[: crossing + 1]))

        measurements: dict[str, Quantity] = {}
        for suffix, indices in selections:
            selected_positions = relative_positions[indices]
            selected_velocities = velocities[indices]
            selected_radii = radii[indices]
            usable = selected_radii > 0.0
            n_vr = int(np.sum(usable))
            if n_vr < 3:
                mean_vr = np.nan
                median_vr = np.nan
                sigma_vr = np.nan
            else:
                # Each region removes its own bulk motion, matching the legacy
                # first-paper statistic and isolating internal expansion.
                centred_velocities = selected_velocities[usable] - np.mean(
                    selected_velocities[usable], axis=0
                )
                radial_velocity = np.sum(
                    selected_positions[usable] * centred_velocities,
                    axis=1,
                ) / selected_radii[usable]
                mean_vr = float(np.mean(radial_velocity))
                median_vr = float(np.median(radial_velocity))
                sigma_vr = float(np.std(radial_velocity))
            if not np.isfinite(sigma_vr) or sigma_vr <= 0.0:
                mean_ratio = np.nan
                median_ratio = np.nan
            else:
                mean_ratio = mean_vr / sigma_vr
                median_ratio = median_vr / sigma_vr
            measurements.update(
                {
                    f"n_vr{suffix}": n_vr | units.none,
                    f"mean_vr{suffix}": mean_vr | units.kms,
                    f"median_vr{suffix}": median_vr | units.kms,
                    f"sigma_vr{suffix}": sigma_vr | units.kms,
                    f"mean_vr_over_sigma_vr{suffix}": mean_ratio | units.none,
                    f"median_vr_over_sigma_vr{suffix}": median_ratio | units.none,
                }
            )
        results[choice_name] = measurements
    return results


RADIAL_VELOCITY_3D_V1 = TimeSeriesDiagnostic(
    name="radial_velocity_3d",
    version=1,
    description=(
        "Mean and median stellar radial velocities, radial dispersions, and "
        "normalized expansion measures through time."
    ),
    evaluate=measure_radial_velocity_3d,
    evaluation_choices=(
        EvaluationChoice(
            name="origin",
            metadata={},
            outputs=RADIAL_VELOCITY_OUTPUTS,
        ),
        EvaluationChoice(
            name="stellar_com",
            metadata={},
            outputs=RADIAL_VELOCITY_OUTPUTS,
        ),
    ),
    field_descriptions={
        "n_vr": "Number of stars contributing to radial-velocity statistics.",
        "mean_vr": "Mean signed three-dimensional stellar radial velocity.",
        "median_vr": "Median signed three-dimensional stellar radial velocity.",
        "sigma_vr": "Standard deviation of stellar radial velocities.",
        "mean_vr_over_sigma_vr": "Mean radial velocity divided by its dispersion.",
        "median_vr_over_sigma_vr": "Median radial velocity divided by its dispersion.",
        **{
            f"{name}_{suffix}": f"{description} within the {fraction:.0%} stellar-mass Lagrangian region."
            for suffix, fraction in LAGRANGIAN_FRACTIONS
            for name, description in (
                ("n_vr", "Number of stars contributing to radial-velocity statistics"),
                ("mean_vr", "Mean signed three-dimensional stellar radial velocity"),
                ("median_vr", "Median signed three-dimensional stellar radial velocity"),
                ("sigma_vr", "Standard deviation of stellar radial velocities"),
                ("mean_vr_over_sigma_vr", "Mean radial velocity divided by its dispersion"),
                ("median_vr_over_sigma_vr", "Median radial velocity divided by its dispersion"),
            )
        },
    },
    choice_names=("center",),
)
"""Three-dimensional radial-velocity diagnostic using standard centre choices."""


DIAGNOSTICS = (RADIAL_VELOCITY_3D_V1,)
"""Every supported three-dimensional radial-velocity diagnostic version."""
