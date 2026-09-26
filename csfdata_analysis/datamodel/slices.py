"""Select raw snapshots and derived diagnostic values at specified times."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy
import pandas

from csfdata.adapters.dcaf import DcafAdapter
from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.configuration import read_simulation_configuration
from csfdata_analysis.datamodel.simulations import source_simulation
from csfdata_analysis.datamodel.series import DiagnosticKey


def select_time_slice(
    data: pandas.DataFrame | dict[DiagnosticKey, pandas.DataFrame],
    time: float,
    normalization: str | None = None,
) -> pandas.DataFrame | dict[DiagnosticKey, pandas.DataFrame]:
    """Interpolate every selected simulation at one requested time.

    Args:
        data: A compiled long DataFrame returned by
            :meth:`SimulationSet.compile_dataframe`, or legacy diagnostic
            tables returned by :func:`load_time_series`.
        time: Physical target time in Myr, or a dimensionless multiplier when
            ``normalization`` is supplied.
        normalization: Optional Myr-valued configuration column, such as
            ``"tff"``. Each simulation's physical target becomes
            ``time * normalization``.

    Returns:
        For a compiled DataFrame, one flat row per selected simulation. For
        legacy input, one row per simulation in every diagnostic table. Rows
        include the requested physical time, interpolated time-series fields,
        and unchanged configuration or scalar columns. Values outside the
        stored time range are ``NaN``; no extrapolation is performed.

    Raises:
        ValueError: If a table lacks field metadata, a required normalization
            column, or unique ascending simulation times.
    """
    collection_table = isinstance(data, pandas.DataFrame)
    tables = {None: data} if collection_table else data
    slices: dict[DiagnosticKey | None, pandas.DataFrame] = {}
    for identity, table in tables.items():
        fields = table.attrs.get("data_fields" if collection_table else "fields")
        if not isinstance(fields, tuple) or not fields:
            raise ValueError(f"Diagnostic table {identity!r} has no declared fields metadata.")
        required = {"collection_id", "simulation_id", "time", *fields}
        if normalization is not None:
            required.add(normalization)
        if not required.issubset(table.columns):
            missing = ", ".join(sorted(required - set(table.columns)))
            raise ValueError(f"Diagnostic table {identity!r} is missing columns: {missing}")

        rows: list[dict[str, str | int | float | bool]] = []
        for _, group in table.groupby(
            ["collection_id", "simulation_id"],
            sort=False,
            observed=True,
        ):
            # Each simulation may use a different normalized target, but its
            # source diagnostic still supplies the unique interpolation grid.
            ordered = group.sort_values("time")
            source_times = ordered["time"].to_numpy(dtype=float)
            label = f"{ordered.iloc[0]['collection_id']}/{ordered.iloc[0]['simulation_id']}"
            if numpy.any(numpy.diff(source_times) <= 0):
                raise ValueError(f"Simulation has duplicate or unordered times: {label}")
            target = float(time)
            if normalization is not None:
                scale = ordered.iloc[0][normalization]
                if not isinstance(scale, (int, float)) or not numpy.isfinite(scale):
                    raise ValueError(f"Simulation has invalid normalization {normalization!r}: {label}")
                target *= float(scale)
            metadata = ordered.iloc[0].drop(labels=["time", *fields]).to_dict()
            row = {**metadata, "time": target}
            for field in fields:
                values = ordered[field].to_numpy(dtype=float)
                row[field] = (
                    numpy.nan
                    if target < source_times[0] or target > source_times[-1]
                    else float(numpy.interp(target, source_times, values))
                )
            rows.append(row)
        selected = pandas.DataFrame(rows)
        selected["collection_id"] = selected["collection_id"].astype("category")
        selected["simulation_id"] = selected["simulation_id"].astype("category")
        selected.attrs.update(table.attrs)
        slices[identity] = selected
    return slices[None] if collection_table else slices


def select_snapshot_slice(
    simulations: Sequence[CatalogueSimulation],
    time: float | Literal["first", "last"],
    normalization: str | None = None,
    data: pandas.DataFrame | None = None,
    mask_field: str | None = None,
) -> pandas.DataFrame:
    """Select stored raw snapshots by time, order, or a Boolean time-series mask.

    Args:
        simulations: Catalogue simulations selected by ``load_simulations``.
        time: Physical target time in Myr, ``"first"``, or ``"last"``. A
            numeric value is a dimensionless multiplier when ``normalization``
            is supplied.
        normalization: Optional canonical configuration parameter in Myr, such
            as ``"tff"``. Each numeric target becomes ``time * normalization``.
        data: Optional time-series table containing ``collection_id``,
            ``simulation_id``, and ``time``. It supplies candidate times when
            provided; all its rows are eligible unless ``mask_field`` is set.
        mask_field: Optional Boolean column in ``data``. Only rows where this
            field is ``True`` remain candidate times for a simulation.

    Returns:
        Table containing each simulation ID, selected snapshot path, selected
        physical time, stored snapshot time, and signed time offset in Myr.

    Raises:
        ValueError: If the selection is malformed, supplied data has no
            eligible time for a selected simulation, a simulation is not
            D-CAF, has no snapshots, or lacks a known Myr-valued normalization
            parameter.
    """
    if isinstance(time, str):
        if time not in {"first", "last"}:
            raise ValueError("time must be a number, 'first', or 'last'.")
        if normalization is not None:
            raise ValueError("normalization applies only when time is numeric.")
    elif not isinstance(time, (int, float)) or isinstance(time, bool):
        raise ValueError("time must be a number, 'first', or 'last'.")
    if data is not None:
        required_columns = {"collection_id", "simulation_id", "time"}
        if mask_field is not None:
            required_columns.add(mask_field)
        if not required_columns.issubset(data.columns):
            missing = ", ".join(sorted(required_columns - set(data.columns)))
            raise ValueError(f"Snapshot selection data is missing columns: {missing}")
    if mask_field is not None:
        if data is None:
            raise ValueError("mask_field requires a time-series data table.")
        if not pandas.api.types.is_bool_dtype(data[mask_field]):
            raise ValueError(f"mask_field {mask_field!r} must contain Boolean values.")

    rows: list[dict[str, str | float]] = []
    for simulation in simulations:
        if simulation.importer != "dcaf":
            raise ValueError(f"Unsupported catalogue importer: {simulation.importer}")
        target_time: float | None = None
        if isinstance(time, (int, float)):
            target_time = float(time)
        if normalization is not None:
            parameter = read_simulation_configuration(simulation.path / "config.yaml").parameter(
                normalization
            )
            if parameter is None or not parameter.is_known or parameter.unit != "Myr":
                raise ValueError(
                    f"Simulation lacks a known Myr normalization parameter {normalization!r}: "
                    f"{simulation.collection_id}/{simulation.simulation_id}"
                )
            target_time *= float(parameter.value)
        raw_simulation = source_simulation(simulation)
        adapter = DcafAdapter(raw_simulation / "raw")
        snapshots = adapter.snapshot_paths()
        if not snapshots:
            raise ValueError(f"Simulation has no snapshots: {raw_simulation}")
        snapshot_times = [adapter.snapshot_time(path) for path in snapshots]
        if any(snapshot_time is None for snapshot_time in snapshot_times):
            raise ValueError(f"Simulation has a snapshot without a model time: {raw_simulation}")
        if data is not None:
            candidates = data.loc[
                (data["collection_id"] == simulation.collection_id)
                & (data["simulation_id"] == simulation.simulation_id)
            ].sort_values("time")
            if mask_field is not None:
                candidates = candidates.loc[candidates[mask_field].fillna(False)]
            if candidates.empty:
                label = f"{simulation.collection_id}/{simulation.simulation_id}"
                raise ValueError(f"Snapshot selection has no eligible time: {label}")
            candidate_times = candidates["time"].to_numpy(dtype=float)
            if not numpy.all(numpy.isfinite(candidate_times)):
                label = f"{simulation.collection_id}/{simulation.simulation_id}"
                raise ValueError(f"Snapshot selection has invalid times: {label}")
            if time == "first":
                target_time = float(candidate_times[0])
            elif time == "last":
                target_time = float(candidate_times[-1])
            else:
                target_time = float(candidate_times[numpy.argmin(abs(candidate_times - target_time))])
        elif time == "first":
            target_time = min(float(snapshot_time) for snapshot_time in snapshot_times)
        elif time == "last":
            target_time = max(float(snapshot_time) for snapshot_time in snapshot_times)
        selected_path, selected_time = min(
            zip(snapshots, snapshot_times, strict=True),
            key=lambda item: abs(float(item[1]) - target_time),
        )
        rows.append(
            {
                "collection_id": simulation.collection_id,
                "simulation_id": simulation.simulation_id,
                "snapshot_path": str(selected_path),
                "target_time": target_time,
                "snapshot_time": float(selected_time),
                "time_offset": float(selected_time) - target_time,
            }
        )
    table = pandas.DataFrame(rows)
    table["collection_id"] = table["collection_id"].astype("category")
    table["simulation_id"] = table["simulation_id"].astype("category")
    return table
