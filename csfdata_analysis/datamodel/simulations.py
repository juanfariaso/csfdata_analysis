"""Represent selected core simulations and bridge them to analysis views."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import pandas
from csfdata.catalogue import CatalogueSimulation, file_sha256, find_simulations, is_lite_catalogue, read_lite_source
from csfdata.catalogue.configuration import read_simulation_configuration
from csfdata.catalogue.metadata import read_simulation_metadata


@dataclass(frozen=True)
class SimulationSet(Sequence[CatalogueSimulation]):
    """One ordered, read-only selection of catalogue simulations.

    Args:
        catalogue_root: Root directory of the source full or lite catalogue.
        simulations: Selected core simulation records in stable order.

    Notes:
        This class does not duplicate or load stored data. It behaves like a
        sequence of :class:`csfdata.catalogue.CatalogueSimulation` objects and
        provides convenient wrappers for Pandas-oriented analysis views.
    """

    catalogue_root: Path
    simulations: tuple[CatalogueSimulation, ...]

    def __len__(self) -> int:
        """Return the number of selected simulations.

        Returns:
            Number of contained core simulation records.
        """
        return len(self.simulations)

    @overload
    def __getitem__(self, index: int) -> CatalogueSimulation:
        ...

    @overload
    def __getitem__(self, index: slice) -> SimulationSet:
        ...

    def __getitem__(self, index: int | slice) -> CatalogueSimulation | SimulationSet:
        """Return one simulation or a smaller selection by position.

        Args:
            index: Integer position or ordinary Python slice.

        Returns:
            One core simulation for an integer, or a new selection retaining
            the same catalogue root for a slice.
        """
        selected = self.simulations[index]
        if isinstance(index, slice):
            return SimulationSet(self.catalogue_root, selected)
        return selected

    def __repr__(self) -> str:
        """Return a compact interactive description of this selection.

        Returns:
            Catalogue root, number of simulations, and contained collections.
        """
        collections = tuple(dict.fromkeys(simulation.collection_id for simulation in self))
        return (
            "SimulationSet("
            f"catalogue_root={str(self.catalogue_root)!r}, "
            f"simulations={len(self)}, collections={collections!r}"
            ")"
        )

    __str__ = __repr__

    def parameters(self) -> pandas.DataFrame:
        """Return canonical configuration values for every selected simulation.

        Returns:
            One Pandas row per selected simulation with stable IDs and every
            known canonical configuration parameter.
        """
        rows = [
            {
                "collection_id": simulation.collection_id,
                "simulation_id": simulation.simulation_id,
                **configuration_values(simulation),
            }
            for simulation in self
        ]
        return pandas.DataFrame(rows, columns=None if rows else ("collection_id", "simulation_id"))

    def series(
        self,
        diagnostics: str | Sequence[str],
        allow_missing: bool = False,
        *,
        choices: dict[str, str] | None = None,
        diagnostic_choices: dict[str, dict[str, str]] | None = None,
        fields: dict[str, Sequence[str]] | None = None,
    ) -> pandas.DataFrame:
        """Load selected evolving diagnostic fields into one Pandas table.

        Args:
            diagnostics: One time-series diagnostic name or a sequence of names.
            allow_missing: Whether to omit simulations without every selected result.
            choices: Optional global scientific choice selections.
            diagnostic_choices: Optional per-diagnostic choice overrides.
            fields: Optional selected fields by diagnostic name.

        Returns:
            Long time-series table with simulation and configuration columns.
        """
        # Import locally because series.py depends on this foundational module
        # for common configuration access.
        from csfdata_analysis.datamodel.series import load_collection_time_series

        return load_collection_time_series(
            self,
            diagnostics,
            allow_missing,
            choices=choices,
            diagnostic_choices=diagnostic_choices,
            fields=fields,
        )

    def scalars(
        self,
        diagnostics: str | Sequence[str],
        allow_missing: bool = False,
        *,
        choices: dict[str, str] | None = None,
        diagnostic_choices: dict[str, dict[str, str]] | None = None,
        fields: dict[str, Sequence[str]] | None = None,
    ) -> pandas.DataFrame:
        """Load selected scalar diagnostic fields into one Pandas table.

        Args:
            diagnostics: One scalar diagnostic name or a sequence of names.
            allow_missing: Whether to omit simulations without every selected result.
            choices: Optional global scientific choice selections.
            diagnostic_choices: Optional per-diagnostic choice overrides.
            fields: Optional selected fields by diagnostic name.

        Returns:
            One table row per simulation with configuration and scalar fields.
        """
        # Import locally because scalars.py uses the shared configuration
        # access defined below in this foundational module.
        from csfdata_analysis.datamodel.scalars import load_collection_scalars

        return load_collection_scalars(
            self,
            diagnostics,
            allow_missing,
            choices=choices,
            diagnostic_choices=diagnostic_choices,
            fields=fields,
        )

    def snapshot_slice(
        self,
        time: float,
        normalization: str | None = None,
    ) -> pandas.DataFrame:
        """Select the nearest raw snapshot at one time for every simulation.

        Args:
            time: Physical target time in Myr, or a multiplier when normalized.
            normalization: Optional Myr-valued configuration parameter name.

        Returns:
            One selected snapshot path and timing record per simulation.
        """
        # Import locally because slices.py resolves raw source directories
        # through source_simulation below.
        from csfdata_analysis.datamodel.slices import select_snapshot_slice

        return select_snapshot_slice(self, time, normalization)

    def inventory(self) -> pandas.DataFrame:
        """Return declared diagnostics and their availability in this selection.

        Returns:
            One Pandas row per declared diagnostic version with fields, choice
            defaults, completed result count, and selected total.
        """
        # Import locally because inventory inspection builds on the core
        # simulation records owned by this foundational module.
        from csfdata_analysis.datamodel.inventory import diagnostic_inventory

        return diagnostic_inventory(self.catalogue_root, self)


def load_simulations(
    catalogue_root: Path | str,
    collection_id: str | None = None,
    filters: dict[str, str | int | float | bool | tuple[float | None, float | None]] | None = None,
) -> SimulationSet:
    """Load catalogue simulations selected by collection and parameter filters.

    Args:
        catalogue_root: Indexed full or lite catalogue root path.
        collection_id: Optional collection ID to select.
        filters: Optional exact or inclusive-range configuration filters.

    Returns:
        Ordered read-only simulation selection.
    """
    root = Path(catalogue_root)
    return SimulationSet(root, find_simulations(root, collection_id, filters))


def configuration_values(simulation: CatalogueSimulation) -> dict[str, str | int | float | bool]:
    """Return every known scalar canonical parameter for one simulation.

    Args:
        simulation: Catalogue simulation whose top-level ``config.yaml`` is read.

    Returns:
        Dictionary from parameter names to known scalar values.
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
