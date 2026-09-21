"""Mass-weighted stellar Lagrangian radii and enclosed star counts."""

from __future__ import annotations

import numpy as np
from amuse.datamodel import Particles
from amuse.units import units
from amuse.units.quantities import Quantity

from csfdata_analysis.diagnostics.base import TimeSeriesDiagnostic, EvaluationChoice


LAGRANGIAN_FRACTIONS = (
    ("l01", 0.01),
    ("l05", 0.05),
    ("l10", 0.10),
    ("l20", 0.20),
    ("l30", 0.30),
    ("l40", 0.40),
    ("l50", 0.50),
    ("l60", 0.60),
    ("l70", 0.70),
    ("l80", 0.80),
    ("l90", 0.90),
    ("l95", 0.95),
)
"""Standard stellar-mass fractions used by the Lagrangian-radii diagnostic."""


LAGRANGIAN_OUTPUTS = {
    "centre_x": "pc",
    "centre_y": "pc",
    "centre_z": "pc",
    "stellar_mass": "Msun",
    "n_stars": "1",
    **{f"r_{suffix}": "pc" for suffix, _ in LAGRANGIAN_FRACTIONS},
    **{f"n_{suffix}": "1" for suffix, _ in LAGRANGIAN_FRACTIONS},
}
"""Canonical HDF5 fields and units for each centre choice."""


def measure_lagrangian_radii(
    particles: Particles,
) -> dict[str, dict[str, Quantity]]:
    """Measure stellar Lagrangian radii and enclosed star counts.

    The calculation sorts all stars by distance from each centre and linearly
    interpolates the radius enclosing each cumulative stellar-mass fraction.
    It then counts the same valid stellar particles at or within each measured
    radius.

    Args:
        particles: Stellar AMUSE particles with finite positions in pc and
            positive finite masses.

    Returns:
        Dict[str, Dict[str, Quantity]]: Results grouped as ``"origin"`` and
            ``"stellar_com"``. Each group contains its centre coordinates,
            total stellar mass, number of valid stars, radii enclosing the
            standard stellar-mass fractions, and the number of stars within
            each corresponding radius.

    Raises:
        ValueError: If the particle set is empty or lacks finite positive-mass
            particles with finite positions.
    """
    # Convert once into canonical units so validation and sorting use finite
    # floating-point arrays.
    positions = np.column_stack(
        (
            particles.x.value_in(units.pc),
            particles.y.value_in(units.pc),
            particles.z.value_in(units.pc),
        )
    )
    masses = np.asarray(particles.mass.value_in(units.MSun), dtype=float)
    valid = np.all(np.isfinite(positions), axis=1) & np.isfinite(masses) & (
        masses > 0.0
    )
    if not np.any(valid):
        raise ValueError("Lagrangian radii require finite positions and positive stellar masses.")

    # Restrict every derived value to the same physically valid star sample.
    positions = positions[valid]
    masses = masses[valid]
    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("Lagrangian radii require a positive finite total stellar mass.")

    centres = {
        "origin": np.zeros(3),
        "stellar_com": np.sum(positions * masses[:, None], axis=0) / total_mass,
    }
    results: dict[str, dict[str, Quantity]] = {}
    for choice_name, centre in centres.items():
        # The shared radial ordering aligns the cumulative-mass radii with the
        # counts enclosed by those interpolated radii.
        radii = np.sqrt(np.sum((positions - centre) ** 2, axis=1))
        order = np.argsort(radii)
        radii = radii[order]
        cumulative_mass = np.cumsum(masses[order])
        measurements: dict[str, Quantity] = {
            "centre_x": float(centre[0]) | units.pc,
            "centre_y": float(centre[1]) | units.pc,
            "centre_z": float(centre[2]) | units.pc,
            "stellar_mass": total_mass | units.MSun,
            "n_stars": len(masses) | units.none,
        }
        for suffix, fraction in LAGRANGIAN_FRACTIONS:
            enclosed_radius = float(
                np.interp(fraction * total_mass, cumulative_mass, radii)
            )
            measurements[f"r_{suffix}"] = enclosed_radius | units.pc
            # ``side='right'`` includes every star at the interpolated radius.
            measurements[f"n_{suffix}"] = (
                np.searchsorted(radii, enclosed_radius, side="right") | units.none
            )
        results[choice_name] = measurements
    return results


LAGRANGIAN_RADII_V1 = TimeSeriesDiagnostic(
    name="lagrangian_radii",
    version=1,
    description=(
        "Mass-weighted stellar Lagrangian radii, total stellar properties, "
        "and enclosed stellar counts through time."
    ),
    evaluate=measure_lagrangian_radii,
    evaluation_choices=(
        EvaluationChoice(
            name="origin",
            metadata={
                "method": "coordinate_origin",
                "particle_selection": "finite_positive_mass_stars",
            },
            outputs=LAGRANGIAN_OUTPUTS,
        ),
        EvaluationChoice(
            name="stellar_com",
            metadata={
                "method": "mass_weighted_stellar_center_of_mass",
                "particle_selection": "finite_positive_mass_stars",
            },
            outputs=LAGRANGIAN_OUTPUTS,
        ),
    ),
    field_descriptions={
        "centre_x": "X coordinate of the selected stellar centre.",
        "centre_y": "Y coordinate of the selected stellar centre.",
        "centre_z": "Z coordinate of the selected stellar centre.",
        "stellar_mass": "Total mass of valid stellar particles.",
        "n_stars": "Total number of valid stellar particles.",
        **{
            f"r_{suffix}": f"Radius enclosing {fraction:.0%} of stellar mass."
            for suffix, fraction in LAGRANGIAN_FRACTIONS
        },
        **{
            f"n_{suffix}": f"Number of valid stellar particles within or at r_{suffix}."
            for suffix, _ in LAGRANGIAN_FRACTIONS
        },
    },
    choice_names=("center",),
)
"""Lagrangian-radii diagnostic using the standard version-1 choices."""


DIAGNOSTICS = (LAGRANGIAN_RADII_V1,)
"""Every supported Lagrangian-radii diagnostic version in this module."""
