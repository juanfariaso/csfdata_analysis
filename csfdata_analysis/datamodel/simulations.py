"""Represent selected core simulations and bridge them to analysis views."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import socket
from typing import Literal, overload

import pandas
from csfdata.catalogue import CatalogueSimulation, file_sha256, find_simulations, is_lite_catalogue, read_lite_source
from csfdata.catalogue.configuration import read_simulation_configuration
from csfdata.catalogue.metadata import read_simulation_metadata
from csfdata.catalogue.snapshots import read_snapshot_times


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
        table = pandas.DataFrame(rows, columns=None if rows else ("collection_id", "simulation_id"))
        table["collection_id"] = table["collection_id"].astype("category")
        table["simulation_id"] = table["simulation_id"].astype("category")
        return table

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
            catalogue_root=self.catalogue_root,
        )

    def compile_dataframe(
        self,
        *,
        series: dict[str, Sequence[str]],
        scalars: dict[str, Sequence[str]] | None = None,
        choices: dict[str, str] | None = None,
        series_choices: dict[str, dict[str, str]] | None = None,
        scalar_choices: dict[str, dict[str, str]] | None = None,
        missing_scalars: Literal["error", "nan"] = "nan",
    ) -> pandas.DataFrame:
        """Build one time-series table enriched with scalar diagnostic values.

        Args:
            series: Required time-series fields by diagnostic name.
            scalars: Optional scalar fields by diagnostic name.
            choices: Optional global scientific choice selections. Registered
                defaults apply to relevant choices not supplied here.
            series_choices: Optional per-time-series diagnostic overrides.
            scalar_choices: Optional per-scalar diagnostic overrides.
            missing_scalars: ``"nan"`` retains series rows without a scalar
                result and fills their requested scalar fields with ``NaN``.
                ``"error"`` instead requires every requested scalar.

        Returns:
            Long Pandas table containing the selected series fields and, when
            requested, scalar fields repeated for every snapshot of their
            containing simulation.

        Raises:
            ValueError: If no time-series diagnostic is selected or a scalar
                field would collide with an existing time-series column, or
                ``missing_scalars`` is unsupported.
            KeyError: If a selected time-series diagnostic result is unavailable,
                or ``missing_scalars`` is ``"error"`` and a scalar is absent.

        Notes:
            The table preserves time-series metadata in ``DataFrame.attrs``.
            Scalar identities, choices, and fields are recorded separately as
            ``scalar_diagnostics``, ``scalar_choices``, and ``scalar_fields``.
        """
        if not series:
            raise ValueError("compile_dataframe requires at least one time-series diagnostic.")
        if missing_scalars not in {"error", "nan"}:
            raise ValueError("missing_scalars must be 'error' or 'nan'.")

        series_table = self.series(
            tuple(series),
            choices=choices,
            diagnostic_choices=series_choices,
            fields=series,
        )
        if not scalars:
            return series_table

        scalar_table = self.scalars(
            tuple(scalars),
            allow_missing=missing_scalars == "nan",
            choices=choices,
            diagnostic_choices=scalar_choices,
            fields=scalars,
        )
        scalar_fields = tuple(
            field
            for fields in scalar_table.attrs["fields"].values()
            for field in fields
        )
        duplicate_fields = set(series_table.columns) & set(scalar_fields)
        if duplicate_fields:
            fields_text = ", ".join(sorted(duplicate_fields))
            raise ValueError(f"Scalar fields collide with time-series columns: {fields_text}.")

        # A scalar result has one row per simulation. Joining on stable IDs
        # repeats it safely over each of that simulation's snapshot rows.
        compiled = series_table.merge(
            scalar_table[["collection_id", "simulation_id", *scalar_fields]],
            on=["collection_id", "simulation_id"],
            how="left",
            validate="many_to_one",
        )
        compiled.attrs.update(series_table.attrs)
        compiled.attrs["scalar_diagnostics"] = scalar_table.attrs["diagnostics"]
        compiled.attrs["scalar_choices"] = scalar_table.attrs["choices"]
        compiled.attrs["scalar_fields"] = scalar_table.attrs["fields"]
        compiled["collection_id"] = compiled["collection_id"].astype("category")
        compiled["simulation_id"] = compiled["simulation_id"].astype("category")
        return compiled

    def slice_dataframe(
        self,
        data: pandas.DataFrame,
        time: float | Literal["first", "last"],
        normalization: str | None = None,
        mask_field: str | None = None,
    ) -> pandas.DataFrame:
        """Select one existing time row for each represented simulation.

        Args:
            data: Time-series table containing ``collection_id``,
                ``simulation_id``, and physical ``time`` columns. It may
                represent any subset of this simulation set.
            time: Physical target time in Myr, ``"first"``, or ``"last"``.
                Numeric targets select the closest existing row without
                interpolation.
            normalization: Optional known Myr-valued configuration parameter.
                A numeric ``time`` is multiplied by this value independently
                for every represented simulation.
            mask_field: Optional Boolean column. False and missing values are
                discarded before selecting a row.

        Returns:
            One original DataFrame row for every represented simulation with
            at least one eligible row. Input columns and attributes are
            preserved.

        Raises:
            ValueError: If required columns or arguments are invalid, the
                DataFrame contains simulations outside this set, times are not
                finite numbers, or a normalization parameter is unavailable.

        Notes:
            Simulations in this set that do not occur in ``data`` are allowed.
            The set defines the permitted simulation identities, not a required
            row count.
        """
        required = {"collection_id", "simulation_id", "time"}
        if mask_field is not None:
            required.add(mask_field)
        if not required.issubset(data.columns):
            missing = ", ".join(sorted(required - set(data.columns)))
            raise ValueError(f"DataFrame slice is missing columns: {missing}")
        if isinstance(time, str):
            if time not in {"first", "last"}:
                raise ValueError("time must be a number, 'first', or 'last'.")
            if normalization is not None:
                raise ValueError("normalization applies only when time is numeric.")
        elif (
            not isinstance(time, (int, float))
            or isinstance(time, bool)
            or not math.isfinite(float(time))
        ):
            raise ValueError("time must be a finite number, 'first', or 'last'.")
        if mask_field is not None and not pandas.api.types.is_bool_dtype(data[mask_field]):
            raise ValueError(f"mask_field {mask_field!r} must contain Boolean values.")

        selected_simulations = {
            (simulation.collection_id, simulation.simulation_id): simulation
            for simulation in self
        }
        provided_ids = set(
            data[["collection_id", "simulation_id"]].itertuples(
                index=False,
                name=None,
            )
        )
        unknown_ids = provided_ids - set(selected_simulations)
        if unknown_ids:
            unknown = ", ".join(
                f"{collection}/{simulation}"
                for collection, simulation in sorted(unknown_ids)
            )
            raise ValueError(f"DataFrame contains simulations outside this set: {unknown}")

        eligible = data.copy().reset_index(drop=True)
        eligible.attrs.update(data.attrs)
        eligible["collection_id"] = eligible["collection_id"].astype("category")
        eligible["simulation_id"] = eligible["simulation_id"].astype("category")
        if mask_field is not None:
            eligible = eligible.loc[eligible[mask_field].fillna(False)].copy()
        if eligible.empty:
            return eligible

        selected_rows: list[pandas.DataFrame] = []
        for identity, group in eligible.groupby(
            ["collection_id", "simulation_id"],
            sort=False,
            observed=True,
        ):
            times = pandas.to_numeric(group["time"], errors="coerce")
            if times.isna().any() or not all(math.isfinite(float(value)) for value in times):
                raise ValueError(
                    f"Simulation has invalid time values: {identity[0]}/{identity[1]}"
                )
            if time == "first":
                selected_index = times.idxmin()
            elif time == "last":
                selected_index = times.idxmax()
            else:
                target = float(time)
                if normalization is not None:
                    simulation = selected_simulations[identity]
                    parameter = read_simulation_configuration(
                        simulation.path / "config.yaml"
                    ).parameter(normalization)
                    if parameter is None or not parameter.is_known or parameter.unit != "Myr":
                        raise ValueError(
                            f"Simulation lacks a known Myr normalization parameter "
                            f"{normalization!r}: {identity[0]}/{identity[1]}"
                        )
                    target *= float(parameter.value)
                selected_index = (times - target).abs().idxmin()
            selected_rows.append(eligible.loc[[selected_index]])

        selected = pandas.concat(selected_rows, ignore_index=True)
        selected.attrs.update(data.attrs)
        selected["collection_id"] = selected["collection_id"].astype("category")
        selected["simulation_id"] = selected["simulation_id"].astype("category")
        return selected

    def snapshot_slice(
        self,
        time: float | Literal["first", "last"],
        normalization: str | None = None,
        data: pandas.DataFrame | None = None,
        mask_field: str | None = None,
    ) -> pandas.DataFrame:
        """Select raw snapshots by time, order, or a Boolean time-series mask.

        Args:
            time: Physical target time in Myr, ``"first"``, or ``"last"``.
            normalization: Optional Myr-valued parameter for numeric times.
            data: Optional time-series table used for a masked selection.
            mask_field: Optional Boolean ``data`` column that limits eligible
                times before selecting the raw snapshot.

        Returns:
            One selected snapshot path and timing record per simulation.
        """
        # Import locally because slices.py resolves raw source directories
        # through source_simulation below.
        from csfdata_analysis.datamodel.slices import select_snapshot_slice

        return select_snapshot_slice(self, time, normalization, data, mask_field)

    def get_snapshot_paths(
        self,
        data: pandas.DataFrame | None = None,
        *,
        time: float | Literal["first", "last"] | None = None,
        tolerance: float | None = None,
        normalization: str | None = None,
    ) -> pandas.DataFrame:
        """Return raw snapshot paths for explicit model-time requests.

        Args:
            data: Optional table containing ``collection_id``,
                ``simulation_id``, and physical ``time`` columns. Every unique
                model-time row is resolved to a snapshot.
            time: Direct physical time in Myr, ``"first"``, or ``"last"`` for
                every simulation in this set. It cannot be combined with
                ``data``.
            tolerance: Optional maximum absolute difference in Myr between a
                requested time and the nearest stored snapshot.
            normalization: Optional known Myr-valued configuration parameter.
                A numeric ``time`` is multiplied by this value independently
                for every simulation.

        Returns:
            One Pandas row per requested model time, containing stable IDs,
            requested and stored times, their offset, iterable local and source
            path lists, source provenance, and any selection issue.

        Raises:
            ValueError: If the selection arguments or input table are invalid,
                or the table references simulations outside this set.
            FileNotFoundError: If a collection snapshot inventory is absent.

        Notes:
            Selection uses the collection ``snapshot-times.yaml`` inventory
            and does not reopen particle files. In a lite catalogue, a locally
            imported snapshot is preferred before the recorded source path.
        """
        if (data is None) == (time is None):
            raise ValueError("Provide exactly one of data or time.")
        if tolerance is not None and (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or tolerance < 0
        ):
            raise ValueError("tolerance must be a non-negative number or None.")
        if data is not None:
            required = {"collection_id", "simulation_id", "time"}
            if not required.issubset(data.columns):
                missing = ", ".join(sorted(required - set(data.columns)))
                raise ValueError(f"Snapshot request data is missing columns: {missing}")
            if normalization is not None:
                raise ValueError("normalization cannot be combined with data.")
            requests = data[["collection_id", "simulation_id", "time"]].drop_duplicates()
        else:
            if isinstance(time, str):
                if time not in {"first", "last"}:
                    raise ValueError("time must be a number, 'first', or 'last'.")
                if normalization is not None:
                    raise ValueError("normalization applies only when time is numeric.")
            elif not isinstance(time, (int, float)) or isinstance(time, bool):
                raise ValueError("time must be a number, 'first', or 'last'.")
            requests = None

        selected_simulations = {
            (simulation.collection_id, simulation.simulation_id): simulation
            for simulation in self
        }
        if requests is not None:
            requested_ids = set(
                requests[["collection_id", "simulation_id"]].itertuples(
                    index=False,
                    name=None,
                )
            )
            unknown_ids = requested_ids - set(selected_simulations)
            if unknown_ids:
                unknown = ", ".join(f"{collection}/{simulation}" for collection, simulation in sorted(unknown_ids))
                raise ValueError(f"Snapshot request contains simulations outside this set: {unknown}")

        catalogue_root = self.catalogue_root.resolve()
        source_hostname = socket.gethostname()
        source_catalogue_root = catalogue_root
        if is_lite_catalogue(catalogue_root):
            source = read_lite_source(catalogue_root)
            source_hostname = source.hostname
            source_catalogue_root = source.catalogue_root

        inventories = {}
        rows: list[dict[str, object]] = []
        request_rows = (
            requests.itertuples(index=False, name=None)
            if requests is not None
            else (
                (simulation.collection_id, simulation.simulation_id, time)
                for simulation in self
            )
        )
        for collection_id, simulation_id, requested in request_rows:
            simulation = selected_simulations[(collection_id, simulation_id)]
            if collection_id not in inventories:
                inventories[collection_id] = read_snapshot_times(
                    self.catalogue_root,
                    collection_id,
                )
            inventory = inventories[collection_id]
            records = tuple(
                sorted(
                    inventory.snapshots.get(simulation_id, ()),
                    key=lambda record: record.time,
                )
            )
            if not records:
                issue = "; ".join(inventory.issues.get(simulation_id, ()))
                rows.append(
                    {
                        "collection_id": collection_id,
                        "simulation_id": simulation_id,
                        "requested_time": None if isinstance(requested, str) else float(requested),
                        "snapshot_time": None,
                        "time_offset": None,
                        "local_paths": [],
                        "source_hostname": source_hostname,
                        "source_catalogue_root": source_catalogue_root,
                        "source_paths": [],
                        "issue": issue or "No readable snapshots are recorded in the inventory.",
                    }
                )
                continue

            if requested == "first":
                target = records[0].time
            elif requested == "last":
                target = records[-1].time
            else:
                target = float(requested)
                if normalization is not None:
                    parameter = read_simulation_configuration(
                        simulation.path / "config.yaml"
                    ).parameter(normalization)
                    if parameter is None or not parameter.is_known or parameter.unit != "Myr":
                        rows.append(
                            {
                                "collection_id": collection_id,
                                "simulation_id": simulation_id,
                                "requested_time": target,
                                "snapshot_time": None,
                                "time_offset": None,
                                "local_paths": [],
                                "source_hostname": source_hostname,
                                "source_catalogue_root": source_catalogue_root,
                                "source_paths": [],
                                "issue": f"Missing known Myr normalization parameter: {normalization}",
                            }
                        )
                        continue
                    target *= float(parameter.value)

            selected = min(records, key=lambda record: abs(record.time - target))
            offset = selected.time - target
            source_path = (
                Path("collections")
                / collection_id
                / "simulations"
                / simulation_id
                / selected.path
            )
            local_path = simulation.path / selected.path
            if not local_path.is_file():
                local_path = source_catalogue_root / source_path

            issue = None
            local_paths: list[Path] = []
            source_paths: list[Path] = [source_path]
            if tolerance is not None and abs(offset) > tolerance:
                issue = f"Nearest snapshot is outside the {float(tolerance):g} Myr tolerance."
                source_paths = []
            elif local_path.is_file():
                local_paths = [local_path]
            else:
                issue = "Selected snapshot is not available on the local filesystem."
            rows.append(
                {
                    "collection_id": collection_id,
                    "simulation_id": simulation_id,
                    "requested_time": target,
                    "snapshot_time": selected.time,
                    "time_offset": offset,
                    "local_paths": local_paths,
                    "source_hostname": source_hostname,
                    "source_catalogue_root": source_catalogue_root,
                    "source_paths": source_paths,
                    "issue": issue,
                }
            )
        table = pandas.DataFrame(rows)
        table["collection_id"] = table["collection_id"].astype("category")
        table["simulation_id"] = table["simulation_id"].astype("category")
        return table

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
