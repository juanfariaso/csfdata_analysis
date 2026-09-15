"""Load and align completed diagnostic time series as Pandas tables."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy
import pandas

from csfdata.catalogue import CatalogueSimulation
from csfdata_analysis.data.loader import configuration_values


def load_time_series(
    simulations: Sequence[CatalogueSimulation],
    diagnostic: str,
    version: int,
    choice: str,
    output: str,
    allow_missing: bool = False,
) -> pandas.DataFrame:
    """Load one completed diagnostic output for selected simulations.

    Args:
        simulations: Catalogue simulations selected by ``load_simulations``.
        diagnostic: Stored diagnostic name, such as ``"lagrangian_radii"``.
        version: Stored diagnostic version.
        choice: Stored scientific choice, such as ``"stellar_com"``.
        output: Stored output column, such as ``"r_l50"``.
        allow_missing: Whether to omit simulations without this completed
            result. By default, missing results raise an error.

    Returns:
        Tidy table with collection and simulation IDs, known configuration
        parameters, ``time_myr``, and a column named after ``output``.

    Raises:
        ValueError: If a result is missing, incomplete, or has a different
            diagnostic schema than requested.
    """
    rows: list[dict[str, str | int | float | bool]] = []
    missing: list[str] = []
    for simulation in simulations:
        result_path = (
            simulation.path
            / "derived"
            / "diagnostics"
            / diagnostic
            / f"v{version}"
            / "series.h5"
        )
        if not result_path.is_file():
            missing.append(f"{simulation.collection_id}/{simulation.simulation_id}")
            continue
        with h5py.File(result_path, "r") as result_file:
            if result_file.attrs.get("complete") != True:
                missing.append(f"{simulation.collection_id}/{simulation.simulation_id}")
                continue
            if (
                result_file.attrs.get("diagnostic_name") != diagnostic
                or int(result_file.attrs.get("diagnostic_version", -1)) != version
            ):
                raise ValueError(f"Diagnostic metadata differs: {result_path}")
            try:
                times = result_file["time_myr"][:]
                values = result_file["choices"][choice][output][:]
            except KeyError as error:
                raise ValueError(f"Stored choice or output is unavailable: {result_path}") from error
        if len(times) != len(values):
            raise ValueError(f"Stored time and output lengths differ: {result_path}")
        parameters = configuration_values(simulation)
        for time_myr, value in zip(times, values, strict=True):
            rows.append(
                {
                    "collection_id": simulation.collection_id,
                    "simulation_id": simulation.simulation_id,
                    **parameters,
                    "time_myr": float(time_myr),
                    output: float(value),
                }
            )
    if missing and not allow_missing:
        raise ValueError(
            "Selected simulations have no completed requested result: " + ", ".join(missing)
        )
    return pandas.DataFrame(rows)


def interpolate_time_series(
    data: pandas.DataFrame,
    times_myr: Sequence[float],
    output: str,
) -> pandas.DataFrame:
    """Linearly interpolate each simulation's time series onto a common grid.

    Args:
        data: Table returned by ``load_time_series``.
        times_myr: Explicit target model times in Myr.
        output: Name of the loaded output column to interpolate.

    Returns:
        Table with one row per simulation and target time. Values outside an
        individual simulation's stored time range are ``NaN``; no extrapolation
        is performed.

    Raises:
        ValueError: If required columns are absent, target times are not
            strictly increasing, or a simulation has duplicate stored times.
    """
    required_columns = {"collection_id", "simulation_id", "time_myr", output}
    if not required_columns.issubset(data.columns):
        missing = ", ".join(sorted(required_columns - set(data.columns)))
        raise ValueError(f"Time-series table is missing required columns: {missing}")
    targets = numpy.asarray(times_myr, dtype=float)
    if targets.ndim != 1 or len(targets) == 0 or numpy.any(numpy.diff(targets) <= 0):
        raise ValueError("times_myr must be one non-empty strictly increasing sequence.")
    rows: list[dict[str, str | int | float | bool]] = []
    group_columns = ["collection_id", "simulation_id"]
    for _, group in data.groupby(group_columns, sort=False):
        ordered = group.sort_values("time_myr")
        source_times = ordered["time_myr"].to_numpy(dtype=float)
        if numpy.any(numpy.diff(source_times) <= 0):
            label = f"{ordered.iloc[0]['collection_id']}/{ordered.iloc[0]['simulation_id']}"
            raise ValueError(f"Simulation has duplicate or unordered times: {label}")
        source_values = ordered[output].to_numpy(dtype=float)
        interpolated = numpy.interp(targets, source_times, source_values)
        interpolated[(targets < source_times[0]) | (targets > source_times[-1])] = numpy.nan
        metadata = ordered.iloc[0].drop(labels=["time_myr", output]).to_dict()
        for target, value in zip(targets, interpolated, strict=True):
            rows.append({**metadata, "time_myr": float(target), output: float(value)})
    return pandas.DataFrame(rows)
