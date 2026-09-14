"""Mass-weighted stellar Lagrangian radii."""

from __future__ import annotations

import numpy as np
from amuse.datamodel import Particles
from amuse.units import units
from amuse.units.quantities import Quantity

from csfdata_analysis.diagnostics.base import Diagnostic, DiagnosticChoice


def measure_lagrangian_radii(
    particles: Particles,
) -> dict[str, dict[str, Quantity]]:
    """Measure stellar Lagrangian radii for standard centre choices.

    The calculation sorts all stars by distance from each centre and linearly
    interpolates the radius enclosing each cumulative stellar-mass fraction.

    Args:
        particles: Stellar AMUSE particles with finite positions in pc and
            positive finite masses.

    Returns:
        Dict[str, Dict[str, Quantity]]: Results grouped as ``"origin"`` and
            ``"stellar_com"``. Each group contains its centre coordinates and
            radii enclosing 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, and 95
            percent of stellar mass.

    Raises:
        ValueError: If the particle set is empty or lacks finite positive-mass
            particles with finite positions.
    """
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
        radii = np.sqrt(np.sum((positions - centre) ** 2, axis=1))
        order = np.argsort(radii)
        radii = radii[order]
        cumulative_mass = np.cumsum(masses[order])
        results[choice_name] = {
            "centre_x": float(centre[0]) | units.pc,
            "centre_y": float(centre[1]) | units.pc,
            "centre_z": float(centre[2]) | units.pc,
            "r_l01": float(np.interp(0.01 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l05": float(np.interp(0.05 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l10": float(np.interp(0.10 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l20": float(np.interp(0.20 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l30": float(np.interp(0.30 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l40": float(np.interp(0.40 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l50": float(np.interp(0.50 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l60": float(np.interp(0.60 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l70": float(np.interp(0.70 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l80": float(np.interp(0.80 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l90": float(np.interp(0.90 * total_mass, cumulative_mass, radii))
            | units.pc,
            "r_l95": float(np.interp(0.95 * total_mass, cumulative_mass, radii))
            | units.pc,
        }
    return results


LAGRANGIAN_RADII_V1 = Diagnostic(
    name="lagrangian_radii",
    version=1,
    evaluate=measure_lagrangian_radii,
    choices=(
        DiagnosticChoice(
            name="origin",
            metadata={
                "method": "coordinate_origin",
                "particle_selection": "finite_positive_mass_stars",
            },
            outputs={
                "centre_x": "pc",
                "centre_y": "pc",
                "centre_z": "pc",
                "r_l01": "pc",
                "r_l05": "pc",
                "r_l10": "pc",
                "r_l20": "pc",
                "r_l30": "pc",
                "r_l40": "pc",
                "r_l50": "pc",
                "r_l60": "pc",
                "r_l70": "pc",
                "r_l80": "pc",
                "r_l90": "pc",
                "r_l95": "pc",
            },
        ),
        DiagnosticChoice(
            name="stellar_com",
            metadata={
                "method": "mass_weighted_stellar_center_of_mass",
                "particle_selection": "finite_positive_mass_stars",
            },
            outputs={
                "centre_x": "pc",
                "centre_y": "pc",
                "centre_z": "pc",
                "r_l01": "pc",
                "r_l05": "pc",
                "r_l10": "pc",
                "r_l20": "pc",
                "r_l30": "pc",
                "r_l40": "pc",
                "r_l50": "pc",
                "r_l60": "pc",
                "r_l70": "pc",
                "r_l80": "pc",
                "r_l90": "pc",
                "r_l95": "pc",
            },
        ),
    ),
)
"""Lagrangian-radii diagnostic using the standard version-1 choices."""
