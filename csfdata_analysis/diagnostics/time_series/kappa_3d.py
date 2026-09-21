"""Three-dimensional observer-style stellar expansion slopes through time."""

from __future__ import annotations

import numpy as np
from amuse.datamodel import Particles
from amuse.units import units
from amuse.units.quantities import Quantity

from csfdata_analysis.diagnostics.base import EvaluationChoice, TimeSeriesDiagnostic


KAPPA_OUTPUTS = {
    "r50": "pc",
    "n_total": "1",
    "n_used": "1",
    **{f"kappa_{axis}": "km/s/pc" for axis in ("x", "y", "z")},
    **{f"r2_{axis}": "1" for axis in ("x", "y", "z")},
    **{f"cov_{axis}_v{axis}": "pc*km/s" for axis in ("x", "y", "z")},
    **{f"var_{axis}": "pc^2" for axis in ("x", "y", "z")},
    **{f"var_v{axis}": "km^2/s^2" for axis in ("x", "y", "z")},
    "sigma3d_residual_outer": "km/s",
    "sigma1d_residual_outer": "km/s",
    "sigma3d_residual_all": "km/s",
    "sigma1d_residual_all": "km/s",
}
"""Canonical fields and units for each three-dimensional kappa centre choice."""


def measure_kappa_3d(
    particles: Particles,
) -> dict[str, dict[str, Quantity]]:
    """Measure coordinate-wise stellar expansion slopes outside ``r50``.

    For each centre, the calculation subtracts the mass-weighted bulk velocity,
    defines ``r50`` as the 50th percentile of valid stellar distances by
    number, and fits each velocity coordinate against its matching position
    coordinate using stars strictly outside that radius.

    Args:
        particles: Stellar AMUSE particles with finite positions in pc,
            velocities in km/s, and positive finite masses.

    Returns:
        Results grouped as ``"origin"`` and ``"stellar_com"``. Each group
        contains the half-number radius, sample counts, coordinate slopes, fit
        diagnostics, covariance and variance terms, and residual dispersions.

    Raises:
        ValueError: If no particles have finite phase-space coordinates and
            positive finite masses.

    Notes:
        Fewer than 20 stars outside ``r50`` produces ``NaN`` fit quantities
        while still recording ``r50``, ``n_total``, and ``n_used``. The
        ``stellar_com`` choice preserves the first-paper kappa convention.
    """
    # Convert once into canonical units, then retain one valid star sample for
    # every centre and every fit quantity in this snapshot.
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
        raise ValueError("Kappa requires finite positions, velocities, and positive masses.")
    positions = positions[valid]
    velocities = velocities[valid]
    masses = masses[valid]
    total_mass = float(np.sum(masses))
    bulk_velocity = np.sum(velocities * masses[:, None], axis=0) / total_mass
    centres = {
        "origin": np.zeros(3),
        "stellar_com": np.sum(positions * masses[:, None], axis=0) / total_mass,
    }

    results: dict[str, dict[str, Quantity]] = {}
    for choice_name, centre in centres.items():
        # Centre-relative radii select the outer half by number, while the same
        # global bulk-velocity correction is used for all fitted coordinates.
        relative_positions = positions - centre
        centred_velocities = velocities - bulk_velocity
        radii = np.sqrt(np.sum(relative_positions**2, axis=1))
        r50 = float(np.percentile(radii, 50.0))
        outer = radii > r50
        n_total = len(relative_positions)
        n_used = int(np.sum(outer))
        measurements: dict[str, Quantity] = {
            "r50": r50 | units.pc,
            "n_total": n_total | units.none,
            "n_used": n_used | units.none,
        }
        kappa = np.full(3, np.nan)
        r2 = np.full(3, np.nan)
        covariance = np.full(3, np.nan)
        position_variance = np.full(3, np.nan)
        velocity_variance = np.full(3, np.nan)

        if n_used >= 20:
            # Fit each Cartesian component independently, retaining all
            # covariance and variance terms for later alternative estimators.
            for axis in range(3):
                position = relative_positions[outer, axis]
                velocity = centred_velocities[outer, axis]
                position_offset = position - np.mean(position)
                velocity_offset = velocity - np.mean(velocity)
                position_variance[axis] = float(np.mean(position_offset**2))
                velocity_variance[axis] = float(np.mean(velocity_offset**2))
                covariance[axis] = float(np.mean(position_offset * velocity_offset))
                if position_variance[axis] > 0.0:
                    kappa[axis] = covariance[axis] / position_variance[axis]
                    fitted_velocity = np.mean(velocity) + kappa[axis] * position_offset
                    squared_total = float(np.sum(velocity_offset**2))
                    squared_residuals = float(np.sum((velocity - fitted_velocity) ** 2))
                    if squared_total > 0.0:
                        r2[axis] = 1.0 - squared_residuals / squared_total

        for axis, label in enumerate(("x", "y", "z")):
            measurements[f"kappa_{label}"] = kappa[axis] | (units.kms / units.pc)
            measurements[f"r2_{label}"] = r2[axis] | units.none
            measurements[f"cov_{label}_v{label}"] = covariance[axis] | (units.pc * units.kms)
            measurements[f"var_{label}"] = position_variance[axis] | units.pc**2
            measurements[f"var_v{label}"] = velocity_variance[axis] | units.kms**2

        if n_used >= 20:
            # A coordinate with no positional spread has no meaningful slope.
            # Preserve that NaN but calculate residual dispersions from the
            # remaining fitted coordinates rather than discarding the snapshot.
            fitted_axes = np.isfinite(kappa)
            if np.any(fitted_axes):
                residual_outer = (
                    centred_velocities[outer][:, fitted_axes]
                    - relative_positions[outer][:, fitted_axes] * kappa[fitted_axes]
                )
                residual_all = (
                    centred_velocities[:, fitted_axes]
                    - relative_positions[:, fitted_axes] * kappa[fitted_axes]
                )
                outer_variance = np.var(residual_outer, axis=0)
                all_variance = np.var(residual_all, axis=0)
                measurements["sigma3d_residual_outer"] = float(
                    np.sqrt(np.sum(outer_variance))
                ) | units.kms
                measurements["sigma1d_residual_outer"] = float(
                    np.sqrt(np.mean(outer_variance))
                ) | units.kms
                measurements["sigma3d_residual_all"] = float(
                    np.sqrt(np.sum(all_variance))
                ) | units.kms
                measurements["sigma1d_residual_all"] = float(
                    np.sqrt(np.mean(all_variance))
                ) | units.kms
            else:
                measurements["sigma3d_residual_outer"] = np.nan | units.kms
                measurements["sigma1d_residual_outer"] = np.nan | units.kms
                measurements["sigma3d_residual_all"] = np.nan | units.kms
                measurements["sigma1d_residual_all"] = np.nan | units.kms
        else:
            measurements["sigma3d_residual_outer"] = np.nan | units.kms
            measurements["sigma1d_residual_outer"] = np.nan | units.kms
            measurements["sigma3d_residual_all"] = np.nan | units.kms
            measurements["sigma1d_residual_all"] = np.nan | units.kms
        results[choice_name] = measurements
    return results


KAPPA_3D_V1 = TimeSeriesDiagnostic(
    name="kappa_3d",
    version=1,
    description=(
        "Coordinate-wise stellar expansion slopes and residual dispersions "
        "outside the half-number radius through time."
    ),
    evaluate=measure_kappa_3d,
    evaluation_choices=(
        EvaluationChoice(name="origin", metadata={}, outputs=KAPPA_OUTPUTS),
        EvaluationChoice(name="stellar_com", metadata={}, outputs=KAPPA_OUTPUTS),
    ),
    field_descriptions={
        "r50": "Half-number radius of valid stars for the selected centre.",
        "n_total": "Number of valid stars before the half-number-radius cut.",
        "n_used": "Number of valid stars strictly outside r50 used by the fits.",
        **{
            f"kappa_{axis}": f"Slope of v{axis} versus {axis} outside r50."
            for axis in ("x", "y", "z")
        },
        **{
            f"r2_{axis}": f"Coefficient of determination for v{axis} versus {axis}."
            for axis in ("x", "y", "z")
        },
        **{
            f"cov_{axis}_v{axis}": f"Covariance of {axis} and v{axis} outside r50."
            for axis in ("x", "y", "z")
        },
        **{
            f"var_{axis}": f"Variance of {axis} outside r50."
            for axis in ("x", "y", "z")
        },
        **{
            f"var_v{axis}": f"Variance of v{axis} outside r50."
            for axis in ("x", "y", "z")
        },
        "sigma3d_residual_outer": "Three-dimensional residual velocity dispersion outside r50.",
        "sigma1d_residual_outer": "One-dimensional residual velocity dispersion outside r50.",
        "sigma3d_residual_all": "Three-dimensional residual velocity dispersion of all valid stars.",
        "sigma1d_residual_all": "One-dimensional residual velocity dispersion of all valid stars.",
    },
    choice_names=("center",),
)
"""Three-dimensional kappa diagnostic using standard centre choices."""


DIAGNOSTICS = (KAPPA_3D_V1,)
"""Every supported three-dimensional kappa diagnostic version."""
