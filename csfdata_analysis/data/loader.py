"""Load selected catalogue simulations and their canonical parameters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from csfdata.catalogue import CatalogueSimulation, file_sha256, find_simulations, is_lite_catalogue, read_lite_source
from csfdata.catalogue.configuration import read_simulation_configuration
from csfdata.catalogue.metadata import read_simulation_metadata


def load_simulations(
    catalogue_root: Path | str,
    collection_id: str | None = None,
    filters: Mapping[str, str | int | float | bool | tuple[float | None, float | None]] | None = None,
) -> tuple[CatalogueSimulation, ...]:
    """Load catalogue simulations selected by collection and parameter filters.

    Args:
        catalogue_root: Indexed full or lite catalogue root path.
        collection_id: Optional collection ID to select.
        filters: Optional exact or inclusive-range configuration filters.

    Returns:
        Selected simulation references ordered by collection and simulation ID.
    """
    return find_simulations(Path(catalogue_root), collection_id, filters)


def configuration_values(simulation: CatalogueSimulation) -> dict[str, str | int | float | bool]:
    """Return every known scalar canonical parameter for one simulation.

    Args:
        simulation: Catalogue simulation whose top-level ``config.yaml`` is read.

    Returns:
        Mapping from parameter names to known scalar values.
    """
    configuration = read_simulation_configuration(simulation.path / "config.yaml")
    return {
        parameter.name: parameter.value
        for parameter in (*configuration.parameters, *configuration.code_parameters)
        if parameter.is_known and parameter.value is not None
    }


def source_simulation(simulation: CatalogueSimulation) -> Path:
    """Return the simulation directory that owns raw snapshots for analysis.

    Args:
        simulation: Simulation selected from a full or lite catalogue.

    Returns:
        Full-catalogue simulation directory containing ``raw/`` snapshots.

    Raises:
        FileNotFoundError: If a lite catalogue's recorded source is unavailable.
        ValueError: If the recorded source collection, metadata, or canonical
            configuration differs from the lite export.
    """
    catalogue_root = simulation.path.parents[3]
    if not is_lite_catalogue(catalogue_root):
        return simulation.path
    source = read_lite_source(catalogue_root)
    if source.collection_id != simulation.collection_id:
        raise ValueError("Lite catalogue collection does not match selected simulation.")
    source_collection = source.catalogue_root / "collections" / source.collection_id
    if file_sha256(source_collection / "collection.yaml") != source.collection_sha256:
        raise ValueError("Source collection configuration differs from the lite export.")
    source_root = source_collection / "simulations" / simulation.simulation_id
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source simulation is unavailable: {source_root}")
    if read_simulation_metadata(source_root / "metadata.yaml") != read_simulation_metadata(
        simulation.path / "metadata.yaml"
    ):
        raise ValueError("Source simulation metadata differs from the lite export.")
    if file_sha256(source_root / "config.yaml") != file_sha256(simulation.path / "config.yaml"):
        raise ValueError("Source simulation configuration differs from the lite export.")
    return source_root
