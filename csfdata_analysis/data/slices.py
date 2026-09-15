"""Select nearest raw snapshots at physical or normalized model times."""

from __future__ import annotations

from collections.abc import Sequence

import pandas

from csfdata.adapters.dcaf import DcafAdapter
from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.configuration import read_simulation_configuration
from csfdata_analysis.data.loader import source_simulation


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
                "target_time_myr": target_time,
                "snapshot_time_myr": float(selected_time),
                "time_offset_myr": float(selected_time) - target_time,
            }
        )
    return pandas.DataFrame(rows)
