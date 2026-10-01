"""Stellar number-weighted Lagrangian radii and enclosed star counts."""

from __future__ import annotations

import numpy as np
from amuse.datamodel import Particles
from amuse.units import units
from amuse.units.quantities import Quantity

from csfdata_analysis.diagnostics.base import EvaluationChoice, TimeSeriesDiagnostic
from csfdata_analysis.diagnostics.time_series.lagrangian_radii import LAGRANGIAN_FRACTIONS


LAGRANGIAN_NUMBER_OUTPUTS = {
    "centre_x": "pc",
    "centre_y": "pc",
    "centre_z": "pc",
    "n_stars": "1",
    **{f"r_n{suffix[1:]}": "pc" for suffix, _ in LAGRANGIAN_FRACTIONS},
    **{f"n_n{suffix[1:]}": "1" for suffix, _ in LAGRANGIAN_FRACTIONS},
}
"""Canonical HDF5 fields and units for each centre choice."""


def measure_lagrangian_number_radii(
    particles: Particles,
) -> dict[str, dict[str, Quantity]]:
    """Measure stellar Lagrangian radii defined by enclosed star number.

    The calculation sorts valid stellar particles by distance from each centre.
    For every standard fraction, the radius is the distance to the required
    ordered star. All stars at that boundary radius are included, so the
    reported enclosed count can be greater than the nominal target count.

    Args:
        particles: Stellar AMUSE particles with finite positions in pc and
            positive finite masses.

    Returns:
        Results grouped as ``"origin"`` and ``"stellar_com"``. Each group
        contains centre coordinates, the number of valid stars, number-based
        Lagrangian radii, and actual enclosed star counts.

    Raises:
        ValueError: If no particle has finite positions and positive finite
            mass.
    """
    # Use the same valid particle definition as mass Lagrangian radii so the
    # two diagnostic families describe a consistent stellar sample.
    positions = np.column_stack(
        (
            particles.x.value_in(units.pc),
            particles.y.value_in(units.pc),
            particles.z.value_in(units.pc),
        )
    )
    masses = np.asarray(particles.mass.value_in(units.MSun), dtype=float)
    valid = np.all(np.isfinite(positions), axis=1) & np.isfinite(masses) & (masses > 0.0)
    if not np.any(valid):
        raise ValueError(
            "Lagrangian number radii require finite positions and positive stellar masses."
        )

    positions = positions[valid]
    masses = masses[valid]
    total_mass = float(np.sum(masses))
    centres = {
        "origin": np.zeros(3),
        "stellar_com": np.sum(positions * masses[:, None], axis=0) / total_mass,
    }
    results: dict[str, dict[str, Quantity]] = {}
    for choice_name, centre in centres.items():
        # The ordered radius selects the nominal number fraction. ``side=right``
        # then retains every star physically at or inside that boundary.
        radii = np.sqrt(np.sum((positions - centre) ** 2, axis=1))
        sorted_radii = np.sort(radii)
        measurements: dict[str, Quantity] = {
            "centre_x": float(centre[0]) | units.pc,
            "centre_y": float(centre[1]) | units.pc,
            "centre_z": float(centre[2]) | units.pc,
            "n_stars": len(positions) | units.none,
        }
        for suffix, fraction in LAGRANGIAN_FRACTIONS:
            target_count = max(1, int(np.ceil(fraction * len(sorted_radii))))
            enclosed_radius = float(sorted_radii[target_count - 1])
            number_suffix = suffix[1:]
            measurements[f"r_n{number_suffix}"] = enclosed_radius | units.pc
            measurements[f"n_n{number_suffix}"] = (
                np.searchsorted(sorted_radii, enclosed_radius, side="right") | units.none
            )
        results[choice_name] = measurements
    return results


LAGRANGIAN_NUMBER_RADII_V1 = TimeSeriesDiagnostic(
    name="lagrangian_number_radii",
    version=1,
    description=(
        "Stellar Lagrangian radii defined by enclosed star-number fractions "
        "and their actual enclosed counts through time."
    ),
    evaluate=measure_lagrangian_number_radii,
    evaluation_choices=(
        EvaluationChoice("origin", {}, LAGRANGIAN_NUMBER_OUTPUTS),
        EvaluationChoice("stellar_com", {}, LAGRANGIAN_NUMBER_OUTPUTS),
    ),
    field_descriptions={
        "centre_x": "X coordinate of the selected stellar centre.",
        "centre_y": "Y coordinate of the selected stellar centre.",
        "centre_z": "Z coordinate of the selected stellar centre.",
        "n_stars": "Number of valid stellar particles.",
        **{
            f"r_n{suffix[1:]}": (
                f"Radius enclosing at least {fraction:.0%} of valid stellar particles."
            )
            for suffix, fraction in LAGRANGIAN_FRACTIONS
        },
        **{
            f"n_n{suffix[1:]}": (
                f"Number of valid stellar particles within or at r_n{suffix[1:]}."
            )
            for suffix, _ in LAGRANGIAN_FRACTIONS
        },
    },
    choice_names=("center",),
)
"""Number-based Lagrangian-radii diagnostic using standard version-1 choices."""


DIAGNOSTICS = (LAGRANGIAN_NUMBER_RADII_V1,)
"""Every supported number-based Lagrangian-radii diagnostic version in this module."""
