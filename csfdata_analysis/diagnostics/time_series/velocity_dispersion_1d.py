"""One-dimensional stellar velocity dispersions in Lagrangian regions."""

from __future__ import annotations

import numpy as np
from amuse.datamodel import Particles
from amuse.units import units
from amuse.units.quantities import Quantity

from csfdata_analysis.diagnostics.base import EvaluationChoice, TimeSeriesDiagnostic
from csfdata_analysis.diagnostics.time_series.lagrangian_radii import LAGRANGIAN_FRACTIONS


VELOCITY_DISPERSION_OUTPUTS = {
    "n_sigma": "1",
    "sigma_vx": "km/s",
    "sigma_vy": "km/s",
    "sigma_vz": "km/s",
    **{
        f"{name}_{region}{suffix[1:]}": unit
        for region in ("l", "n")
        for suffix, _ in LAGRANGIAN_FRACTIONS
        for name, unit in (
            ("n_sigma", "1"),
            ("sigma_vx", "km/s"),
            ("sigma_vy", "km/s"),
            ("sigma_vz", "km/s"),
        )
    },
}
"""Canonical HDF5 fields and units for each centre choice."""


def measure_velocity_dispersion_1d(
    particles: Particles,
) -> dict[str, dict[str, Quantity]]:
    """Measure component velocity dispersions in stellar Lagrangian regions.

    The diagnostic calculates unweighted population standard deviations of
    ``vx``, ``vy``, and ``vz`` for all valid stars, mass-defined Lagrangian
    regions, and number-defined Lagrangian regions. Each dispersion is
    therefore independent of the selected region's bulk velocity.

    Args:
        particles: Stellar AMUSE particles with finite positions in pc,
            velocities in km/s, and positive finite masses.

    Returns:
        Results grouped as ``"origin"`` and ``"stellar_com"``. Each group
        contains all-star values, mass-region fields ending in ``_lXX``, and
        number-region fields ending in ``_nXX``.

    Raises:
        ValueError: If no particle has finite phase-space coordinates and
            positive finite mass.

    Notes:
        Regions with fewer than three stars store ``NaN`` component
        dispersions. Boundary membership includes every star at or within the
        corresponding radius.
    """
    # Convert and validate once so every region uses one consistent stellar
    # sample, including the regions defined by either mass or star number.
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
            "Velocity dispersion requires finite positions, velocities, and positive stellar masses."
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
        # Calculate both region families from the centre-relative ordering.
        # The selections are then passed directly to NumPy's component-wise
        # standard deviation, which removes the regional mean automatically.
        radii = np.sqrt(np.sum((positions - centre) ** 2, axis=1))
        order = np.argsort(radii)
        sorted_radii = radii[order]
        cumulative_mass = np.cumsum(masses[order])
        selections = [("", np.arange(len(positions)))]
        for suffix, fraction in LAGRANGIAN_FRACTIONS:
            mass_index = int(np.searchsorted(cumulative_mass, fraction * total_mass, side="left"))
            mass_radius = sorted_radii[mass_index]
            selections.append((f"_l{suffix[1:]}", np.flatnonzero(radii <= mass_radius)))

            number_count = max(1, int(np.ceil(fraction * len(sorted_radii))))
            number_radius = sorted_radii[number_count - 1]
            selections.append((f"_n{suffix[1:]}", np.flatnonzero(radii <= number_radius)))

        measurements: dict[str, Quantity] = {}
        for suffix, indices in selections:
            count = len(indices)
            if count < 3:
                dispersion = np.full(3, np.nan)
            else:
                dispersion = np.std(velocities[indices], axis=0)
            measurements.update(
                {
                    f"n_sigma{suffix}": count | units.none,
                    f"sigma_vx{suffix}": float(dispersion[0]) | units.kms,
                    f"sigma_vy{suffix}": float(dispersion[1]) | units.kms,
                    f"sigma_vz{suffix}": float(dispersion[2]) | units.kms,
                }
            )
        results[choice_name] = measurements
    return results


VELOCITY_DISPERSION_1D_V1 = TimeSeriesDiagnostic(
    name="velocity_dispersion_1d",
    version=1,
    description=(
        "Unweighted one-dimensional stellar velocity dispersions for all "
        "stars and mass- and number-defined Lagrangian regions through time."
    ),
    evaluate=measure_velocity_dispersion_1d,
    evaluation_choices=(
        EvaluationChoice("origin", {}, VELOCITY_DISPERSION_OUTPUTS),
        EvaluationChoice("stellar_com", {}, VELOCITY_DISPERSION_OUTPUTS),
    ),
    field_descriptions={
        "n_sigma": "Number of valid stellar particles contributing to the all-star dispersion.",
        "sigma_vx": "Population standard deviation of stellar vx.",
        "sigma_vy": "Population standard deviation of stellar vy.",
        "sigma_vz": "Population standard deviation of stellar vz.",
        **{
            f"{name}_{region}{suffix[1:]}": description.format(
                region=("mass" if region == "l" else "number"),
                fraction=fraction,
            )
            for region in ("l", "n")
            for suffix, fraction in LAGRANGIAN_FRACTIONS
            for name, description in (
                ("n_sigma", "Number of stars within the {fraction:.0%} {region} Lagrangian region."),
                ("sigma_vx", "Population standard deviation of vx within the {fraction:.0%} {region} Lagrangian region."),
                ("sigma_vy", "Population standard deviation of vy within the {fraction:.0%} {region} Lagrangian region."),
                ("sigma_vz", "Population standard deviation of vz within the {fraction:.0%} {region} Lagrangian region."),
            )
        },
    },
    choice_names=("center",),
)
"""Component velocity-dispersion diagnostic using standard version-1 choices."""


DIAGNOSTICS = (VELOCITY_DISPERSION_1D_V1,)
"""Every supported one-dimensional velocity-dispersion diagnostic version in this module."""
