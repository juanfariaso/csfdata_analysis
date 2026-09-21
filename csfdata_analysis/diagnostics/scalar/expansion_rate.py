"""Late-time expansion-rate scalar diagnostic."""

from __future__ import annotations

import numpy as np
from amuse.units import units
from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.diagnostics import (
    DiagnosticField,
    DiagnosticRequirement,
)
from csfdata_analysis.diagnostics.base import ScalarDiagnostic


def fit_expansion_rate(
    time: np.ndarray,
    radius: np.ndarray,
    n_stars: np.ndarray,
    min_points: int = 6,
) -> dict[str, float]:
    """Fit the late-time expansion of one Lagrangian-radius series.

    The fit begins at the first minimum radius after the final stellar count is
    reached. This excludes the early phase in which star formation is still
    changing the stellar system.

    Args:
        time: Snapshot times in Myr.
        radius: One selected Lagrangian-radius series in pc, for example
            ``r_l50`` from the ``stellar_com`` group.
        n_stars: Total valid stellar count at the same snapshots.
        min_points: Minimum number of snapshots required in the fitted interval.

    Returns:
        Scalar diagnostic values with canonical units:

        - ``dRdt``: Fitted expansion rate in km/s.
        - ``fit_t0``: Time of the post-star-formation radius minimum in Myr.
        - ``fit_r0``: Radius of the fitted line at ``fit_t0`` in pc.
        - ``r_min``: Radius at the fit start in pc.
        - ``r2``: Coefficient of determination.
        - ``nrmse_iqr``: Root-mean-square residual normalized by the radius
            interquartile range.
        - ``n_points``: Number of snapshots in the fitted interval.
        - ``final_time``: Final fitted snapshot time in Myr.

    Raises:
        ValueError: If the inputs are not one-dimensional, have inconsistent
            lengths, contain insufficient usable snapshots, or do not provide
            enough post-minimum snapshots for the fit.
    """
    time = np.asarray(time, dtype=float)
    radius = np.asarray(radius, dtype=float)
    counts = np.asarray(n_stars, dtype=float)

    if time.ndim != 1 or radius.ndim != 1 or counts.ndim != 1:
        raise ValueError("Expansion-rate inputs must be one-dimensional arrays.")
    if len(time) != len(radius) or len(time) != len(counts):
        raise ValueError("Expansion-rate inputs must have the same length.")
    if min_points < 2:
        raise ValueError("min_points must be at least 2.")

    # Keep only complete measurements, then enforce the time order assumed by
    # the star-formation and late-expansion definitions.
    usable = np.isfinite(time) & np.isfinite(radius) & np.isfinite(counts) & (counts >= 0)
    time = time[usable]
    radius = radius[usable]
    counts = counts[usable]

    if len(time) < min_points:
        raise ValueError("Not enough finite snapshots to fit an expansion rate.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Expansion-rate snapshot times must be strictly increasing.")

    # The highest recorded count is the completed stellar population. The first
    # snapshot reaching it marks the earliest time after star formation ends.
    final_star_count = np.max(counts)
    post_formation = counts == final_star_count
    if not np.any(post_formation):
        raise ValueError("Could not identify the final stellar count.")

    post_time = time[post_formation]
    post_radius = radius[post_formation]
    minimum_index = int(np.argmin(post_radius))
    fit_start_time = float(post_time[minimum_index])
    r_min = float(post_radius[minimum_index])

    # Fit from the earliest post-formation minimum through the final snapshot.
    fit_mask = time >= fit_start_time
    fit_time = time[fit_mask]
    fit_radius = radius[fit_mask]
    if len(fit_time) < min_points:
        raise ValueError(
            "Not enough snapshots from the post-star-formation minimum to the final time."
        )

    slope, intercept = np.polyfit(fit_time, fit_radius, 1)
    fitted_radius = intercept + slope * fit_time
    residuals = fit_radius - fitted_radius

    squared_residuals = float(np.sum(residuals**2))
    squared_total = float(np.sum((fit_radius - np.mean(fit_radius)) ** 2))
    r2 = 1.0 if squared_total == 0.0 else 1.0 - squared_residuals / squared_total

    rmse = float(np.sqrt(squared_residuals / len(fit_radius)))
    lower_quartile, upper_quartile = np.percentile(fit_radius, [25.0, 75.0])
    iqr = float(upper_quartile - lower_quartile)
    if iqr == 0.0:
        nrmse_iqr = 0.0 if rmse == 0.0 else float("inf")
    else:
        nrmse_iqr = rmse / iqr

    dRdt = float(
        (slope | (units.pc / units.Myr)).value_in(units.kms)
    )
    fit_radius_at_start = float(intercept + slope * fit_start_time)

    return {
        "dRdt": dRdt,
        "fit_t0": fit_start_time,
        "fit_r0": fit_radius_at_start,
        "r_min": r_min,
        "r2": float(r2),
        "nrmse_iqr": float(nrmse_iqr),
        "n_points": float(len(fit_time)),
        "final_time": float(fit_time[-1]),
    }


def compute_expansion_rate(
    simulation: CatalogueSimulation,
    choices: dict[str, str],
) -> dict[str, float | int]:
    """Compute one choice-specific expansion-rate result for a simulation.

    Args:
        simulation: Catalogue simulation containing completed
            ``lagrangian_radii v1`` results.
        choices: Concrete ``center`` and ``lagrangian_radius`` selections.

    Returns:
        Scalar expansion-rate fields in their declared canonical units.

    Raises:
        KeyError: If the required Lagrangian-radii result or selected choice is
            unavailable.
        ValueError: If the choices do not match the diagnostic contract.
    """
    if set(choices) != {"center", "lagrangian_radius"}:
        raise ValueError("Expansion rate requires center and lagrangian_radius choices.")
    center = choices["center"]
    radius_name = choices["lagrangian_radius"]
    # The core simulation interface resolves paths, validates completion, and
    # returns only the selected stored datasets without exposing HDF5 layout.
    radii = simulation.diagnostics.time_series[("lagrangian_radii", "v1")]
    data = radii.read(
        choices={"center": center},
        fields=(radius_name, "n_stars"),
    )

    return fit_expansion_rate(
        np.asarray(data["time"], dtype=float),
        np.asarray(data[radius_name], dtype=float),
        np.asarray(data["n_stars"], dtype=float),
    )



EXPANSION_RATE_V1 = ScalarDiagnostic(
    name="expansion_rate",
    version=1,
    description="Late-time linear expansion rates fitted to Lagrangian radii.",
    evaluate=compute_expansion_rate,
    fields=(
        DiagnosticField("dRdt", "Fitted late-time radius change.", "km/s"),
        DiagnosticField("fit_t0", "Selected fit reference time.", "Myr"),
        DiagnosticField("fit_r0", "Fitted radius at the reference time.", "pc"),
        DiagnosticField("r_min", "Radius at the fit start.", "pc"),
        DiagnosticField("r2", "Coefficient of determination.", "1"),
        DiagnosticField("nrmse_iqr", "Normalized fit residual.", "1"),
        DiagnosticField("n_points", "Snapshots included in the fit.", "1"),
        DiagnosticField("final_time", "Final time included in the fit.", "Myr"),
    ),
    choice_names=("center", "lagrangian_radius"),
    requires=(DiagnosticRequirement("lagrangian_radii", "v1"),),
)


DIAGNOSTICS = (EXPANSION_RATE_V1,)
"""Every supported expansion-rate diagnostic version in this module."""
