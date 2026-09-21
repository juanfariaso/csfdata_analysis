"""Load, align, and summarize completed diagnostic time series."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy
import pandas

from csfdata.catalogue import CatalogueSimulation
from csfdata_analysis.data.loader import configuration_values


DiagnosticKey = tuple[str, str]
"""Stable ``(name, version)`` identity of one time-series diagnostic."""


def load_time_series(
    simulations: Sequence[CatalogueSimulation],
    diagnostics: Mapping[DiagnosticKey, Mapping[str, object]],
    allow_missing: bool = False,
) -> dict[DiagnosticKey, pandas.DataFrame]:
    """Load selected fields from one or more completed diagnostics.

    Args:
        simulations: Catalogue simulations selected by ``load_simulations``.
        diagnostics: Mapping from ``(name, version)`` to a selection mapping.
            Each selection defines ``"choices"`` as an exact choice mapping
            and ``"fields"`` as a non-empty sequence of output names.
        allow_missing: Whether to omit simulations without a completed selected
            diagnostic. By default, any missing selected result raises an error.

    Returns:
        One wide Pandas table per requested diagnostic. Every table includes
        IDs, canonical configuration values, selected choices, ``time_myr``,
        and its requested fields.

    Raises:
        ValueError: If a selection is malformed or a requested result is
            unavailable for any selected simulation.

    Notes:
        Tables retain ``fields`` and ``choices`` in ``DataFrame.attrs`` so
        later alignment, slicing, and aggregation operate only on diagnostic
        columns rather than configuration metadata.
    """
    if not diagnostics:
        raise ValueError("diagnostics must contain at least one diagnostic selection.")

    tables: dict[DiagnosticKey, pandas.DataFrame] = {}
    for identity, selection in diagnostics.items():
        # Validate each request before reading data so a malformed selection
        # cannot create a partial table that looks scientifically valid.
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or not all(isinstance(value, str) and value for value in identity)
        ):
            raise ValueError("Diagnostic identities must be non-empty (name, version) tuples.")
        if not isinstance(selection, Mapping):
            raise ValueError(f"Diagnostic selection for {identity!r} must be a mapping.")
        choices = selection.get("choices")
        fields = selection.get("fields")
        if not isinstance(choices, Mapping) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in choices.items()
        ):
            raise ValueError(f"Diagnostic selection for {identity!r} needs string choices.")
        if (
            not isinstance(fields, Sequence)
            or isinstance(fields, str)
            or not fields
            or not all(isinstance(field, str) and field for field in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ValueError(
                f"Diagnostic selection for {identity!r} needs unique non-empty fields."
            )

        rows: list[dict[str, str | int | float | bool]] = []
        missing: list[str] = []
        for simulation in simulations:
            try:
                result = simulation.diagnostics.time_series[identity]
                values = result.read(choices, fields)
            except (KeyError, ValueError):
                missing.append(f"{simulation.collection_id}/{simulation.simulation_id}")
                continue
            times = values.pop("time_myr")
            lengths = {len(times), *(len(values[field]) for field in fields)}
            if len(lengths) != 1:
                raise ValueError(
                    f"Stored time-series lengths differ: "
                    f"{simulation.collection_id}/{simulation.simulation_id} {identity!r}."
                )

            # Attach invariant simulation metadata to every time row, making
            # the returned table immediately usable for grouping and plotting.
            parameters = configuration_values(simulation)
            for index, time_myr in enumerate(times):
                rows.append(
                    {
                        "collection_id": simulation.collection_id,
                        "simulation_id": simulation.simulation_id,
                        **parameters,
                        **dict(choices),
                        "time_myr": float(time_myr),
                        **{field: float(values[field][index]) for field in fields},
                    }
                )
        if missing and not allow_missing:
            raise ValueError(
                f"Selected simulations have no completed {identity[0]} {identity[1]} result: "
                + ", ".join(missing)
            )
        table = pandas.DataFrame(rows)
        table.attrs["fields"] = tuple(fields)
        table.attrs["choices"] = dict(choices)
        tables[identity] = table
    return tables


def interpolate_time_series(
    data: Mapping[DiagnosticKey, pandas.DataFrame],
    times_myr: Sequence[float],
) -> dict[DiagnosticKey, pandas.DataFrame]:
    """Linearly interpolate selected diagnostics onto one physical-time grid.

    Args:
        data: Diagnostic tables returned by :func:`load_time_series`.
        times_myr: Explicit target model times in Myr.

    Returns:
        One aligned table per diagnostic. Values outside an individual
        simulation's stored range are ``NaN``; the function never extrapolates.

    Raises:
        ValueError: If tables lack their stored field metadata, times are not a
            strictly increasing non-empty sequence, or one simulation has
            duplicate or unordered source times.
    """
    targets = numpy.asarray(times_myr, dtype=float)
    if targets.ndim != 1 or len(targets) == 0 or numpy.any(numpy.diff(targets) <= 0):
        raise ValueError("times_myr must be one non-empty strictly increasing sequence.")

    aligned_tables: dict[DiagnosticKey, pandas.DataFrame] = {}
    for identity, table in data.items():
        fields = table.attrs.get("fields")
        if not isinstance(fields, tuple) or not fields:
            raise ValueError(f"Diagnostic table {identity!r} has no declared fields metadata.")
        required_columns = {"collection_id", "simulation_id", "time_myr", *fields}
        if not required_columns.issubset(table.columns):
            missing = ", ".join(sorted(required_columns - set(table.columns)))
            raise ValueError(f"Diagnostic table {identity!r} is missing columns: {missing}")

        rows: list[dict[str, str | int | float | bool]] = []
        for _, group in table.groupby(["collection_id", "simulation_id"], sort=False):
            # A unique ascending source grid is required for one well-defined
            # linear interpolation per simulation and output field.
            ordered = group.sort_values("time_myr")
            source_times = ordered["time_myr"].to_numpy(dtype=float)
            if numpy.any(numpy.diff(source_times) <= 0):
                label = f"{ordered.iloc[0]['collection_id']}/{ordered.iloc[0]['simulation_id']}"
                raise ValueError(f"Simulation has duplicate or unordered times: {label}")
            metadata = ordered.iloc[0].drop(labels=["time_myr", *fields]).to_dict()
            interpolated_fields = {}
            for field in fields:
                values = ordered[field].to_numpy(dtype=float)
                interpolated = numpy.interp(targets, source_times, values)
                interpolated[(targets < source_times[0]) | (targets > source_times[-1])] = numpy.nan
                interpolated_fields[field] = interpolated
            for index, target in enumerate(targets):
                rows.append(
                    {
                        **metadata,
                        "time_myr": float(target),
                        **{field: float(interpolated_fields[field][index]) for field in fields},
                    }
                )
        aligned = pandas.DataFrame(rows)
        aligned.attrs.update(table.attrs)
        aligned_tables[identity] = aligned
    return aligned_tables


def aggregate_time_series(
    data: Mapping[DiagnosticKey, pandas.DataFrame],
    group_by: Sequence[str] = (),
) -> dict[DiagnosticKey, pandas.DataFrame]:
    """Calculate ensemble mean and standard deviation at every stored time.

    Args:
        data: Aligned diagnostic tables returned by
            :func:`interpolate_time_series`.
        group_by: Optional configuration columns that define separate
            ensembles, such as ``("tff", "sfe")``.

    Returns:
        One summary table per diagnostic. Each has grouping columns,
        ``time_myr``, ``n_simulations``, and ``<field>_mean`` and
        ``<field>_std`` columns.

    Raises:
        ValueError: If a requested grouping column or declared field is absent.

    Notes:
        Standard deviations use sample normalization (``ddof=1``), and are
        ``NaN`` for groups with fewer than two finite values.
    """
    summaries: dict[DiagnosticKey, pandas.DataFrame] = {}
    for identity, table in data.items():
        fields = table.attrs.get("fields")
        if not isinstance(fields, tuple) or not fields:
            raise ValueError(f"Diagnostic table {identity!r} has no declared fields metadata.")
        columns = {"collection_id", "time_myr", *group_by, *fields}
        if not columns.issubset(table.columns):
            missing = ", ".join(sorted(columns - set(table.columns)))
            raise ValueError(f"Diagnostic table {identity!r} is missing columns: {missing}")

        # Group only by stable metadata and model time. A simulation contributes
        # only when it has finite values for every requested output field.
        grouping = ["collection_id", *group_by, "time_myr"]
        rows: list[dict[str, str | int | float | bool]] = []
        for values, group in table.groupby(grouping, sort=False, dropna=False):
            group_values = values if isinstance(values, tuple) else (values,)
            row = dict(zip(grouping, group_values, strict=True))
            contributing = group.dropna(subset=list(fields))
            row["n_simulations"] = int(contributing["simulation_id"].nunique())
            for field in fields:
                finite_values = contributing[field]
                row[f"{field}_mean"] = float(finite_values.mean()) if not finite_values.empty else numpy.nan
                row[f"{field}_std"] = (
                    float(finite_values.std(ddof=1))
                    if len(finite_values) > 1
                    else numpy.nan
                )
            rows.append(row)
        summary = pandas.DataFrame(rows)
        summary.attrs.update(table.attrs)
        summaries[identity] = summary
    return summaries
