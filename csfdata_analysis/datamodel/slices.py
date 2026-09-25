"""Select raw snapshots and derived diagnostic values at specified times."""

from __future__ import annotations

from collections.abc import Sequence

import numpy
import pandas

from csfdata.adapters.dcaf import DcafAdapter
from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.configuration import read_simulation_configuration
from csfdata_analysis.datamodel.simulations import source_simulation
from csfdata_analysis.datamodel.series import DiagnosticKey


def select_time_series_slice(
    data: dict[DiagnosticKey, pandas.DataFrame],
    time: float,
    normalization: str | None = None,
) -> dict[DiagnosticKey, pandas.DataFrame]:
    """Interpolate diagnostic values for every simulation at one requested time.

    Args:
        data: Diagnostic tables returned by :func:`load_time_series`.
        time: Physical target time in Myr, or a dimensionless multiplier when
            ``normalization`` is supplied.
        normalization: Optional Myr-valued configuration column, such as
            ``"tff"``. Each simulation's physical target becomes
            ``time * normalization``.

    Returns:
        One row per source simulation in every diagnostic table. Rows include
        the requested physical time and linearly interpolated selected fields.
        Values outside the stored time range are ``NaN``; no extrapolation is
        performed.

    Raises:
        ValueError: If a table lacks field metadata, a required normalization
            column, or unique ascending simulation times.
    """
    slices: dict[DiagnosticKey, pandas.DataFrame] = {}
    for identity, table in data.items():
        fields = table.attrs.get("fields")
        if not isinstance(fields, tuple) or not fields:
            raise ValueError(f"Diagnostic table {identity!r} has no declared fields metadata.")
        required = {"collection_id", "simulation_id", "time", *fields}
        if normalization is not None:
            required.add(normalization)
        if not required.issubset(table.columns):
            missing = ", ".join(sorted(required - set(table.columns)))
            raise ValueError(f"Diagnostic table {identity!r} is missing columns: {missing}")

        rows: list[dict[str, str | int | float | bool]] = []
        for _, group in table.groupby(["collection_id", "simulation_id"], sort=False):
            # Each simulation may use a different normalized target, but its
            # source diagnostic still supplies the unique interpolation grid.
            ordered = group.sort_values("time")
            source_times = ordered["time"].to_numpy(dtype=float)
            if numpy.any(numpy.diff(source_times) <= 0):
                label = f"{ordered.iloc[0]['collection_id']}/{ordered.iloc[0]['simulation_id']}"
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
        selected.attrs.update(table.attrs)
        slices[identity] = selected
    return slices


def select_snapshot_slice(
    simulations: Sequence[CatalogueSimulation],
    time: float,
    normalization: str | None = None,
) -> pandas.DataFrame:
    """Select the nearest stored snapshot for every requested simulation.

    Args:
        simulations: Catalogue simulations selected by ``load_simulations``.
        time: Physical target time in Myr, or a dimensionless multiplier when
            ``normalization`` is supplied.
        normalization: Optional canonical configuration parameter in Myr, such
            as ``"tff"``. Each target becomes ``time * normalization``.

    Returns:
        Table containing each simulation ID, selected snapshot path, requested
        physical time, stored snapshot time, and signed time offset in Myr.

    Raises:
        ValueError: If a simulation is not D-CAF, has no snapshots, or lacks a
            known Myr-valued normalization parameter.
    """
    rows: list[dict[str, str | float]] = []
    for simulation in simulations:
        if simulation.importer != "dcaf":
            raise ValueError(f"Unsupported catalogue importer: {simulation.importer}")
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
    return pandas.DataFrame(rows)
