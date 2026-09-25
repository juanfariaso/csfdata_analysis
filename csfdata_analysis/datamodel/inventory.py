"""Inspect indexed diagnostic availability across simulation selections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

import pandas

from csfdata.catalogue import (
    CatalogueSimulation,
    collection_diagnostics_path,
    read_collection_diagnostics,
)


@dataclass(frozen=True)
class DiagnosticKinds:
    """List the latest registered diagnostics by storage kind.

    Args:
        data: Indexed diagnostic inventory table.

    Notes:
        A diagnostic name may have several registered versions. These views
        expose only the latest version; use :attr:`DiagnosticInventory.versions`
        to inspect every version.
    """

    data: pandas.DataFrame

    @property
    def series(self) -> tuple[str, ...]:
        """Return latest registered time-series diagnostic names.

        Returns:
            Alphabetically ordered time-series diagnostic names.
        """
        return tuple(
            self.data.loc[self.data["kind"] == "time_series", "diagnostic"]
            .drop_duplicates()
            .sort_values()
        )

    @property
    def scalar(self) -> tuple[str, ...]:
        """Return latest registered scalar diagnostic names.

        Returns:
            Alphabetically ordered scalar diagnostic names.
        """
        return tuple(
            self.data.loc[self.data["kind"] == "scalar", "diagnostic"]
            .drop_duplicates()
            .sort_values()
        )


@dataclass(frozen=True)
class DiagnosticInventory:
    """Interactive view of indexed diagnostic availability.

    Args:
        data: One row per declared diagnostic version, with indexed coverage.

    Notes:
        ``data`` remains the complete Pandas representation. The other
        properties are compact exploration views intended for an interactive
        analysis session.
    """

    data: pandas.DataFrame

    @property
    def current(self) -> pandas.DataFrame:
        """Return the latest registered version of every diagnostic kind.

        Returns:
            Inventory rows reduced to the latest version of each diagnostic.
        """
        return (
            self.data.sort_values("version")
            .drop_duplicates(("kind", "diagnostic"), keep="last")
            .reset_index(drop=True)
        )

    @property
    def diagnostics(self) -> DiagnosticKinds:
        """Return diagnostic names grouped by storage kind.

        Returns:
            Latest series and scalar diagnostic names.
        """
        return DiagnosticKinds(self.current)

    @property
    def fields(self) -> dict[str, tuple[str, ...]]:
        """Return output fields by latest diagnostic name.

        Returns:
            Dictionary from diagnostic name to its declared field names.
        """
        return {
            row.diagnostic: row.fields
            for row in self.current.itertuples(index=False)
        }

    @property
    def versions(self) -> dict[str, tuple[str, ...]]:
        """Return every registered version by diagnostic name.

        Returns:
            Dictionary from diagnostic name to registered versions in order.
        """
        return {
            diagnostic: tuple(rows["version"].sort_values())
            for diagnostic, rows in self.data.groupby("diagnostic", sort=True)
        }

    @property
    def counts(self) -> pandas.DataFrame:
        """Return indexed availability counts for current diagnostics.

        Returns:
            Table indexed by kind, diagnostic, and version with ``available``
            and ``total`` columns.
        """
        return self.current.set_index(["kind", "diagnostic", "version"])[
            ["available", "total"]
        ]

    def __repr__(self) -> str:
        """Return a compact summary for interactive inspection.

        Returns:
            Available time-series and scalar diagnostic names.
        """
        return (
            "DiagnosticInventory("
            f"series={self.diagnostics.series!r}, "
            f"scalar={self.diagnostics.scalar!r}"
            ")"
        )

    __str__ = __repr__


def diagnostic_inventory(
    catalogue_root: Path | str,
    simulations: Sequence[CatalogueSimulation],
) -> DiagnosticInventory:
    """Summarize diagnostic availability for one selected simulation set.

    Args:
        catalogue_root: Indexed full or lite catalogue root path.
        simulations: Core simulations, usually a ``SimulationSet`` returned
            by :func:`csfdata_analysis.datamodel.load_simulations`.

    Returns:
        Interactive inventory containing one row per declared ``(collection,
        kind, diagnostic, version)`` in :attr:`DiagnosticInventory.data`.

    Notes:
        The SQLite registry is the source of availability counts. Run
        ``csfdata index-catalogue`` after adding or importing diagnostics.
        Collections with no selected simulations produce no rows because no
        availability denominator exists.
    """
    root = Path(catalogue_root)
    registry_path = root / "registry.sqlite"
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Catalogue has not been indexed: {registry_path}. "
            "Run csfdata index-catalogue first."
        )

    rows: list[dict[str, object]] = []
    simulations_by_collection: dict[str, list[CatalogueSimulation]] = {}
    for simulation in simulations:
        simulations_by_collection.setdefault(simulation.collection_id, []).append(simulation)

    with sqlite3.connect(registry_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required_tables = {"derived_values", "time_series_products"}
        if not required_tables <= tables:
            raise ValueError(
                "Catalogue registry is out of date. Run csfdata index-catalogue to rebuild it."
            )

        for collection_id, selected in simulations_by_collection.items():
            # The collection schema defines fields and defaults once; SQLite
            # supplies coverage without reopening individual result files.
            collection_root = root / "collections" / collection_id
            collection_diagnostics = read_collection_diagnostics(
                collection_diagnostics_path(collection_root)
            )
            choice_defaults = {
                choice.name: choice.default
                for choice in collection_diagnostics.choices
            }
            selected_ids = {simulation.simulation_id for simulation in selected}
            time_series_products = {
                (simulation_id, name, version)
                for simulation_id, name, version in connection.execute(
                    """
                    SELECT simulation_id, diagnostic_name, diagnostic_version
                    FROM time_series_products
                    WHERE collection_id = ?
                    """,
                    (collection_id,),
                )
                if simulation_id in selected_ids
            }
            scalar_values: dict[tuple[str, str, str, str], set[str]] = {}
            for simulation_id, name, version, choice_key, field in connection.execute(
                """
                SELECT simulation_id, diagnostic_name, diagnostic_version, choice_key, name
                FROM derived_values
                WHERE collection_id = ?
                """,
                (collection_id,),
            ):
                if simulation_id in selected_ids:
                    scalar_values.setdefault(
                        (simulation_id, name, version, choice_key), set()
                    ).add(field)

            for definition in collection_diagnostics.diagnostics:
                choices = {
                    name: choice_defaults[name]
                    for name in definition.choices
                }
                if definition.kind == "time_series":
                    available = sum(
                        (simulation.simulation_id, definition.name, definition.version)
                        in time_series_products
                        for simulation in selected
                    )
                else:
                    choice_key = json.dumps(choices, sort_keys=True, separators=(",", ":"))
                    fields = {field.name for field in definition.fields}
                    available = sum(
                        fields
                        <= scalar_values.get(
                            (
                                simulation.simulation_id,
                                definition.name,
                                definition.version,
                                choice_key,
                            ),
                            set(),
                        )
                        for simulation in selected
                    )
                rows.append(
                    {
                        "collection_id": collection_id,
                        "kind": definition.kind,
                        "diagnostic": definition.name,
                        "version": definition.version,
                        "fields": tuple(field.name for field in definition.fields),
                        "choices": choices,
                        "available": available,
                        "total": len(selected),
                    }
                )

    return DiagnosticInventory(
        pandas.DataFrame(
            rows,
            columns=(
                "collection_id",
                "kind",
                "diagnostic",
                "version",
                "fields",
                "choices",
                "available",
                "total",
            ),
        )
    )
