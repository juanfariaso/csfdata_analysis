"""Load selected scalar diagnostics into tables for filtering and plotting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
import json
from pathlib import Path
import sqlite3

import pandas

from csfdata.catalogue import CatalogueSimulation, SimulationDiagnosticResults, read_collection
from csfdata.catalogue.diagnostics import collection_diagnostics_path, read_collection_diagnostics
from csfdata_analysis.datamodel.simulations import configuration_values


class DataScalars:
    """Research-facing view of one scalar diagnostic table.

    Args:
        dataframe: Pandas table containing ``collection_id`` and
            ``simulation_id``. It may contain one row per simulation or one
            row per simulation and expanded diagnostic choice.

    Attributes:
        dataframe: The ordinary Pandas table containing the loaded values.

    Notes:
        This class uses composition rather than subclassing
        :class:`pandas.DataFrame`. Access the table through ``dataframe`` for
        normal Pandas filtering, grouping, plotting, and merging.
    """

    def __init__(self, dataframe: pandas.DataFrame) -> None:
        if not isinstance(dataframe, pandas.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        required = {"collection_id", "simulation_id"}
        if not required <= set(dataframe.columns):
            missing = ", ".join(sorted(required - set(dataframe.columns)))
            raise ValueError(f"A scalar DataFrame is missing columns: {missing}.")
        self.dataframe = dataframe

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return diagnostic names represented by this table."""
        return tuple(
            identity[0] if isinstance(identity, tuple) else str(identity)
            for identity in self.dataframe.attrs.get("diagnostics", ())
        )

    @property
    def fields(self) -> dict[str, tuple[str, ...]]:
        """Return selected output fields grouped by diagnostic name."""
        return dict(self.dataframe.attrs.get("fields", {}))

    @property
    def choices(self) -> dict[str, dict[str, str]]:
        """Return fixed scientific choices grouped by diagnostic name."""
        return {
            name: dict(values)
            for name, values in self.dataframe.attrs.get("choices", {}).items()
        }

    @property
    def expanded_choices(self) -> dict[str, tuple[str, ...]]:
        """Return choices retained as ordinary table columns."""
        return dict(self.dataframe.attrs.get("expanded_choices", {}))

    def __len__(self) -> int:
        """Return the number of scalar rows in the table."""
        return len(self.dataframe)

    def select(self, filters: dict[str, object]) -> DataScalars:
        """Select rows using simulation parameters or diagnostic values.

        Args:
            filters: Column selections. A scalar requires an exact match, a
                list accepts any listed value, and a two-item tuple defines an
                inclusive ``(minimum, maximum)`` range. Either range bound may
                be ``None``.

        Returns:
            A new scalar-data view containing the matching rows and the same
            diagnostic metadata. An unmatched selection returns an empty view.

        Raises:
            KeyError: If a selected column is absent.
            ValueError: If ``filters`` is malformed or a tuple does not contain
                exactly two range bounds.
        """
        if not isinstance(filters, dict) or not all(
            isinstance(name, str) and name for name in filters
        ):
            raise ValueError("filters must be a dictionary with non-empty string keys.")

        mask = pandas.Series(True, index=self.dataframe.index, dtype=bool)
        for name, selection in filters.items():
            if name not in self.dataframe.columns:
                available = ", ".join(str(column) for column in self.dataframe.columns)
                raise KeyError(
                    f"Unknown DataScalars filter column {name!r}. Available: {available}."
                )
            values = self.dataframe[name]
            if isinstance(selection, list):
                condition = values.isin(selection)
            elif isinstance(selection, tuple):
                if len(selection) != 2:
                    raise ValueError(
                        f"Range filter {name!r} must contain exactly two bounds."
                    )
                minimum, maximum = selection
                condition = pandas.Series(True, index=self.dataframe.index, dtype=bool)
                if minimum is not None:
                    condition &= values >= minimum
                if maximum is not None:
                    condition &= values <= maximum
            else:
                condition = values == selection
            mask &= condition.fillna(False)

        selected = self.dataframe.loc[mask].copy()
        selected.attrs.update(self.dataframe.attrs)
        return DataScalars(selected)

    def aggregate_over(
        self,
        over: str | Sequence[str] = "seed_index",
        *,
        statistics: dict[str, str | Callable[[pandas.Series], object]] | None = None,
        grid_axes: Sequence[str] | None = None,
    ) -> pandas.DataFrame:
        """Aggregate loaded scalar fields over selected grid axes.

        Args:
            over: One grid axis or a sequence of axes to average over. The
                default combines stochastic realizations identified by
                ``seed_index``.
            statistics: Optional output suffixes mapped to Pandas reducer names
                or custom functions. The default calculates ``mean``,
                ``median``, and ``std``. Custom functions receive the
                non-missing values from one ensemble group.
            grid_axes: Optional ordered grid-axis names. When omitted, axes are
                read from each collection's ``collection.yaml`` using the
                catalogue root recorded when this view was loaded.

        Returns:
            One ordinary Pandas table per remaining grid-point and expanded
            diagnostic-choice combination. It contains ``n_simulations`` and
            one ``<field>_<statistic>`` column for every loaded diagnostic
            field and selected statistic.

        Raises:
            FileNotFoundError: If automatic grid discovery cannot read a
                collection configuration.
            KeyError: If a selected field or grid axis is absent from the
                scalar table.
            ValueError: If an argument is malformed, no grid definition is
                available, an averaged axis is not part of the grid, or a
                selected field is not numeric.

        Notes:
            Aggregation is performed independently for every collection.
            Expanded diagnostic choices remain grouping coordinates and are
            never averaged implicitly. Output rows preserve the first-observed
            grid and expanded-choice order for direct line plotting.
        """
        if isinstance(over, str):
            averaged_axes = (over,)
        elif (
            isinstance(over, Sequence)
            and not isinstance(over, str)
            and over
            and all(isinstance(name, str) and name for name in over)
        ):
            averaged_axes = tuple(over)
        else:
            raise ValueError("over must be a non-empty grid-axis name or sequence.")
        if len(set(averaged_axes)) != len(averaged_axes):
            raise ValueError("over must not contain duplicate grid-axis names.")

        if statistics is None:
            selected_statistics: dict[
                str,
                str | Callable[[pandas.Series], object],
            ] = {
                "mean": "mean",
                "median": "median",
                "std": "std",
            }
        elif not isinstance(statistics, dict) or not statistics or not all(
            isinstance(name, str)
            and name
            and (
                (isinstance(reducer, str) and bool(reducer))
                or callable(reducer)
            )
            for name, reducer in statistics.items()
        ):
            raise ValueError(
                "statistics must map non-empty output names to Pandas reducer "
                "names or callable functions."
            )
        else:
            selected_statistics = dict(statistics)

        selected_fields = tuple(
            field
            for diagnostic_fields in self.fields.values()
            for field in diagnostic_fields
        )
        if not selected_fields or len(set(selected_fields)) != len(selected_fields):
            raise ValueError(
                "DataScalars must declare non-empty unique fields before aggregation."
            )

        if grid_axes is not None and (
            not isinstance(grid_axes, Sequence)
            or isinstance(grid_axes, str)
            or not grid_axes
            or not all(isinstance(name, str) and name for name in grid_axes)
        ):
            raise ValueError("grid_axes must be a non-empty sequence of axis names.")
        explicit_grid_axes = tuple(grid_axes) if grid_axes is not None else None
        if explicit_grid_axes is not None and len(set(explicit_grid_axes)) != len(explicit_grid_axes):
            raise ValueError("grid_axes must not contain duplicate names.")

        missing_fields = set(selected_fields) - set(self.dataframe.columns)
        if missing_fields:
            names = ", ".join(sorted(missing_fields))
            raise KeyError(f"Scalar aggregation fields are missing: {names}.")
        nonnumeric_fields = tuple(
            field
            for field in selected_fields
            if not pandas.api.types.is_numeric_dtype(self.dataframe[field])
        )
        if nonnumeric_fields:
            names = ", ".join(nonnumeric_fields)
            raise ValueError(f"Scalar aggregation fields must be numeric: {names}.")

        expanded_axes = tuple(
            choice
            for diagnostic_choices in self.expanded_choices.values()
            for choice in diagnostic_choices
        )
        catalogue_root = self.dataframe.attrs.get("catalogue_root")
        summaries: list[pandas.DataFrame] = []
        for collection_id, collection_rows in self.dataframe.groupby(
            "collection_id",
            sort=False,
            observed=True,
        ):
            if explicit_grid_axes is None:
                if catalogue_root is None:
                    raise ValueError(
                        "Grid axes are unavailable. Pass grid_axes explicitly or load "
                        "the values through SimulationSet.scalars()."
                    )
                collection = read_collection(catalogue_root, str(collection_id))
                collection_grid_axes = tuple(name for name, _ in collection.grid_axes)
                if not collection_grid_axes:
                    raise ValueError(
                        f"Collection does not declare grid_axes: {collection_id}."
                    )
            else:
                collection_grid_axes = explicit_grid_axes

            unknown_averaged_axes = set(averaged_axes) - set(collection_grid_axes)
            if unknown_averaged_axes:
                names = ", ".join(sorted(unknown_averaged_axes))
                raise ValueError(
                    f"Averaged axes are not declared for collection {collection_id}: {names}."
                )
            grouping_axes = tuple(
                name for name in collection_grid_axes if name not in averaged_axes
            )
            grouping_columns = tuple(
                dict.fromkeys(("collection_id", *grouping_axes, *expanded_axes))
            )
            missing_columns = set(grouping_columns) - set(collection_rows.columns)
            if missing_columns:
                names = ", ".join(sorted(missing_columns))
                raise KeyError(
                    f"Grid or expanded-choice columns are missing for collection "
                    f"{collection_id}: {names}."
                )

            # Count only simulations with every requested scalar field while
            # retaining all-NaN groups in the resulting grid summary.
            working = collection_rows.copy()
            complete = working[list(selected_fields)].notna().all(axis=1)
            working["_contributing_simulation"] = working["simulation_id"].where(complete)
            aggregations: dict[
                str,
                tuple[str, str | Callable[[pandas.Series], object]],
            ] = {
                "n_simulations": ("_contributing_simulation", "nunique")
            }
            for field in selected_fields:
                for statistic, reducer in selected_statistics.items():
                    if callable(reducer):
                        aggregations[f"{field}_{statistic}"] = (
                            field,
                            lambda values, function=reducer: function(values.dropna()),
                        )
                    else:
                        aggregations[f"{field}_{statistic}"] = (field, reducer)
            summary = working.groupby(
                list(grouping_columns),
                sort=False,
                observed=True,
                dropna=False,
                as_index=False,
            ).agg(**aggregations)
            summaries.append(summary)

        if not summaries:
            columns = ["collection_id", "n_simulations"]
            columns.extend(
                f"{field}_{statistic}"
                for field in selected_fields
                for statistic in selected_statistics
            )
            return pandas.DataFrame(columns=columns)

        result = pandas.concat(summaries, ignore_index=True)
        result["collection_id"] = result["collection_id"].astype("category")
        result.attrs["aggregated_over"] = averaged_axes
        result.attrs["fields"] = selected_fields
        result.attrs["statistics"] = tuple(selected_statistics)
        return result

    def __repr__(self) -> str:
        """Return a compact interactive description of the loaded scalars."""
        simulation_count = len(
            self.dataframe[["collection_id", "simulation_id"]].drop_duplicates()
        )
        return (
            f"DataScalars(rows={len(self)}, simulations={simulation_count}, "
            f"diagnostics={self.diagnostics!r}, fields={self.fields!r}, "
            f"expanded_choices={self.expanded_choices!r})"
        )

    __str__ = __repr__


def load_collection_scalars(
    simulations: Sequence[CatalogueSimulation],
    diagnostics: str | Sequence[str],
    allow_missing: bool = True,
    *,
    choices: dict[str, str] | None = None,
    diagnostic_choices: dict[str, dict[str, str]] | None = None,
    expand_choices: dict[str, Sequence[str]] | None = None,
    fields: dict[str, Sequence[str]] | None = None,
    catalogue_root: Path | str | None = None,
) -> pandas.DataFrame:
    """Load scalar diagnostic values from selected catalogue simulations.

    Args:
        simulations: Catalogue simulations selected by ``load_simulations``.
        diagnostics: One scalar diagnostic name or a sequence of names. Each
            name resolves to its latest collection-declared version.
        allow_missing: Whether to retain simulations without every selected
            scalar result, filling requested scalar fields with ``NaN``. Set
            this to ``False`` to require a completed result for every selected
            simulation.
        choices: Optional global choice values. Registered defaults are used
            for relevant choices not supplied here.
        diagnostic_choices: Optional per-diagnostic choice overrides. These
            take precedence over ``choices``.
        expand_choices: Optional choice names to retain as DataFrame columns.
            This currently supports one scalar diagnostic at a time. Each
            selected combination produces one row per simulation.
        fields: Optional selected fields by diagnostic name. Omitted entries
            load every declared field for that diagnostic.
        catalogue_root: Optional indexed catalogue root. When its
            ``registry.sqlite`` exists, scalar values and configuration values
            are read from SQLite instead of per-simulation YAML files.

    Returns:
        One Pandas row per selected simulation, or per selected simulation and
        expanded choice combination. Rows contain stable IDs, canonical
        configuration values, expanded choice columns, and selected scalar
        fields. The table attrs record diagnostic versions, fixed choices,
        expanded choices, and fields.

    Raises:
        ValueError: If selections are malformed, scalar fields collide with
            each other or configuration values, or simulations resolve to
            incompatible scalar schemas.
        KeyError: If a selected scalar result is unavailable and
            ``allow_missing`` is ``False``.
    """
    if isinstance(diagnostics, str):
        names = (diagnostics,)
    elif (
        isinstance(diagnostics, Sequence)
        and not isinstance(diagnostics, dict)
        and all(isinstance(name, str) and name for name in diagnostics)
    ):
        names = tuple(diagnostics)
    else:
        raise ValueError("diagnostics must contain one or more non-empty diagnostic names.")
    if not names or len(set(names)) != len(names):
        raise ValueError("diagnostics must contain non-empty unique names.")
    if expand_choices is not None and (
        not isinstance(expand_choices, dict)
        or not all(
            isinstance(name, str)
            and isinstance(choice_names, Sequence)
            and not isinstance(choice_names, str)
            and choice_names
            and all(isinstance(choice_name, str) and choice_name for choice_name in choice_names)
            and len(set(choice_names)) == len(choice_names)
            for name, choice_names in expand_choices.items()
        )
    ):
        raise ValueError(
            "expand_choices must map diagnostic names to non-empty unique choice-name sequences."
        )
    if expand_choices is not None and set(expand_choices) - set(names):
        raise ValueError("expand_choices contains diagnostics that were not selected.")
    if expand_choices:
        if len(names) != 1:
            raise ValueError("expand_choices currently supports one scalar diagnostic at a time.")
        if not simulations:
            raise ValueError("No simulations were selected.")

        # Read the registered choice values once, then reuse the normal scalar
        # loader for every concrete combination. This keeps indexed and direct
        # file reads on exactly the same validation path.
        diagnostic_name = names[0]
        collection_diagnostics = simulations[0].diagnostics.scalar.collection_diagnostics
        definition = next(
            (
                candidate
                for candidate in collection_diagnostics.diagnostics
                if candidate.name == diagnostic_name and candidate.kind == "scalar"
            ),
            None,
        )
        if definition is None:
            raise ValueError(f"Unknown scalar diagnostic: {diagnostic_name}.")
        expanded_names = tuple(expand_choices[diagnostic_name])
        unknown_expanded = set(expanded_names) - set(definition.choices)
        if unknown_expanded:
            choices_text = ", ".join(sorted(unknown_expanded))
            raise ValueError(
                f"{diagnostic_name} has irrelevant expanded choices: {choices_text}."
            )
        fixed_global = set(choices or {}) & set(expanded_names)
        fixed_diagnostic = set((diagnostic_choices or {}).get(diagnostic_name, {})) & set(expanded_names)
        if fixed_global or fixed_diagnostic:
            choices_text = ", ".join(sorted(fixed_global | fixed_diagnostic))
            raise ValueError(
                f"Expanded choices cannot also be fixed: {choices_text}."
            )
        choices_by_name = {
            choice.name: choice
            for choice in collection_diagnostics.choices
        }
        tables = []
        for values in product(*(choices_by_name[name].values for name in expanded_names)):
            overrides = dict((diagnostic_choices or {}).get(diagnostic_name, {}))
            overrides.update(dict(zip(expanded_names, values, strict=True)))
            try:
                table = load_collection_scalars(
                    simulations,
                    diagnostic_name,
                    allow_missing=allow_missing,
                    choices=choices,
                    diagnostic_choices={diagnostic_name: overrides},
                    fields=fields,
                    catalogue_root=catalogue_root,
                )
            except ValueError as error:
                # A fully absent choice combination is expected when callers
                # allow incomplete scalar coverage; leave it absent from the
                # long table so Pandas can represent it as NaN after a pivot.
                if allow_missing and str(error) == "No selected simulations have completed scalar results.":
                    continue
                raise
            for choice_name, value in zip(expanded_names, values, strict=True):
                table[choice_name] = value
            tables.append(table)

        if not tables:
            raise ValueError("No selected simulations have completed scalar results.")
        expanded = pandas.concat(tables, ignore_index=True)
        expanded["collection_id"] = expanded["collection_id"].astype("category")
        expanded["simulation_id"] = expanded["simulation_id"].astype("category")
        expanded.attrs["diagnostics"] = tables[0].attrs["diagnostics"]
        expanded.attrs["choices"] = {
            diagnostic_name: {
                name: value
                for name, value in tables[0].attrs["choices"][diagnostic_name].items()
                if name not in expanded_names
            }
        }
        expanded.attrs["expanded_choices"] = {diagnostic_name: expanded_names}
        expanded.attrs["fields"] = tables[0].attrs["fields"]
        if "catalogue_root" in tables[0].attrs:
            expanded.attrs["catalogue_root"] = tables[0].attrs["catalogue_root"]
        return expanded
    if choices is not None and (
        not isinstance(choices, dict)
        or not all(isinstance(name, str) and isinstance(value, str) for name, value in choices.items())
    ):
        raise ValueError("choices must be a dictionary of string names and values.")
    if diagnostic_choices is not None and (
        not isinstance(diagnostic_choices, dict)
        or not all(
            isinstance(name, str)
            and isinstance(values, dict)
            and all(isinstance(choice, str) and isinstance(value, str) for choice, value in values.items())
            for name, values in diagnostic_choices.items()
        )
    ):
        raise ValueError("diagnostic_choices must map diagnostic names to string choice dictionaries.")
    if fields is not None and (
        not isinstance(fields, dict)
        or not all(
            isinstance(name, str)
            and isinstance(selected, Sequence)
            and not isinstance(selected, str)
            and selected
            and all(isinstance(field, str) and field for field in selected)
            and len(set(selected)) == len(selected)
            for name, selected in fields.items()
        )
    ):
        raise ValueError("fields must map diagnostic names to non-empty unique field sequences.")
    if diagnostic_choices is not None and set(diagnostic_choices) - set(names):
        raise ValueError("diagnostic_choices contains diagnostics that were not selected.")
    if fields is not None and set(fields) - set(names):
        raise ValueError("fields contains diagnostics that were not selected.")

    root = Path(catalogue_root) if catalogue_root is not None else None
    registry_path = root / "registry.sqlite" if root is not None else None
    if registry_path is not None and registry_path.is_file():
        simulations_by_collection: dict[str, list[CatalogueSimulation]] = {}
        for simulation in simulations:
            simulations_by_collection.setdefault(simulation.collection_id, []).append(simulation)

        rows: list[dict[str, str | int | float | bool]] = []
        reference: tuple[
            tuple[tuple[str, str], ...],
            dict[str, dict[str, str]],
            dict[str, tuple[str, ...]],
        ] | None = None
        with sqlite3.connect(registry_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not {"parameter_values", "derived_values"} <= tables:
                raise ValueError(
                    "Catalogue registry is out of date. Run csfdata index-catalogue to rebuild it."
                )

            for collection_id, selected in simulations_by_collection.items():
                collection_diagnostics = read_collection_diagnostics(
                    collection_diagnostics_path(root / "collections" / collection_id)
                )
                collection_choices = {
                    choice.name: choice
                    for choice in collection_diagnostics.choices
                }
                unknown_choices = set(choices or {}) - set(collection_choices)
                if unknown_choices:
                    names_text = ", ".join(sorted(unknown_choices))
                    raise ValueError(f"choices contains unregistered names: {names_text}.")
                definitions = {
                    definition.name: definition
                    for definition in collection_diagnostics.diagnostics
                    if definition.kind == "scalar"
                }
                unknown_diagnostics = set(names) - set(definitions)
                if unknown_diagnostics:
                    unknown_text = ", ".join(sorted(unknown_diagnostics))
                    available_text = ", ".join(sorted(definitions)) or "none"
                    raise ValueError(
                        f"Unknown scalar diagnostics: {unknown_text}. "
                        f"Available: {available_text}."
                    )

                selected_ids = {simulation.simulation_id for simulation in selected}
                configurations: dict[str, dict[str, str | int | float | bool]] = {
                    simulation_id: {}
                    for simulation_id in selected_ids
                }
                for simulation_id, name, value_type, numeric_value, text_value in connection.execute(
                    """
                    SELECT simulation_id, name, value_type, numeric_value, text_value
                    FROM parameter_values
                    WHERE collection_id = ?
                    """,
                    (collection_id,),
                ):
                    if simulation_id not in configurations:
                        continue
                    configurations[simulation_id][name] = (
                        float(numeric_value)
                        if value_type == "number"
                        else bool(numeric_value)
                        if value_type == "boolean"
                        else text_value
                    )

                selected_definitions: list[tuple[object, dict[str, str], tuple[str, ...]]] = []
                for name in names:
                    definition = definitions[name]
                    selected_choices = {
                        choice_name: collection_choices[choice_name].default
                        for choice_name in definition.choices
                    }
                    selected_choices.update(
                        {
                            choice_name: value
                            for choice_name, value in (choices or {}).items()
                            if choice_name in definition.choices
                        }
                    )
                    overrides = (diagnostic_choices or {}).get(name, {})
                    unknown_overrides = set(overrides) - set(definition.choices)
                    if unknown_overrides:
                        choices_text = ", ".join(sorted(unknown_overrides))
                        raise ValueError(f"{name} has irrelevant diagnostic choices: {choices_text}.")
                    selected_choices.update(overrides)
                    invalid_choices = {
                        choice_name: value
                        for choice_name, value in selected_choices.items()
                        if value not in collection_choices[choice_name].values
                    }
                    if invalid_choices:
                        choices_text = ", ".join(
                            f"{choice_name}={value!r}"
                            for choice_name, value in invalid_choices.items()
                        )
                        available_text = ", ".join(
                            f"{choice_name}={collection_choices[choice_name].values!r}"
                            for choice_name in invalid_choices
                        )
                        raise ValueError(
                            f"{name} has unavailable choice values: {choices_text}. "
                            f"Available: {available_text}."
                        )
                    selected_fields = tuple(
                        (fields or {}).get(name, tuple(field.name for field in definition.fields))
                    )
                    unknown_fields = set(selected_fields) - {
                        field.name for field in definition.fields
                    }
                    if unknown_fields:
                        fields_text = ", ".join(sorted(unknown_fields))
                        available_text = ", ".join(field.name for field in definition.fields)
                        raise ValueError(
                            f"Unknown fields for scalar diagnostic {name!r}: {fields_text}. "
                            f"Available: {available_text}."
                        )
                    selected_definitions.append((definition, selected_choices, selected_fields))

                indexed_values: dict[tuple[str, str, str, str], dict[str, str | int | float | bool]] = {}
                for definition, selected_choices, _ in selected_definitions:
                    choice_key = json.dumps(
                        selected_choices,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for simulation_id, field, value_type, numeric_value, text_value in connection.execute(
                        """
                        SELECT simulation_id, name, value_type, numeric_value, text_value
                        FROM derived_values
                        WHERE collection_id = ?
                          AND diagnostic_name = ?
                          AND diagnostic_version = ?
                          AND choice_key = ?
                        """,
                        (collection_id, definition.name, definition.version, choice_key),
                    ):
                        if simulation_id not in selected_ids:
                            continue
                        indexed_values.setdefault(
                            (simulation_id, definition.name, definition.version, choice_key),
                            {},
                        )[field] = (
                            float(numeric_value)
                            if value_type == "number"
                            else bool(numeric_value)
                            if value_type == "boolean"
                            else text_value
                        )

                resolved_diagnostics = tuple(
                    (definition.name, definition.version)
                    for definition, _, _ in selected_definitions
                )
                resolved_choices = {
                    definition.name: selected_choices
                    for definition, selected_choices, _ in selected_definitions
                }
                resolved_fields = {
                    definition.name: selected_fields
                    for definition, _, selected_fields in selected_definitions
                }
                resolved = (resolved_diagnostics, resolved_choices, resolved_fields)
                if reference is None:
                    reference = resolved
                elif resolved != reference:
                    raise ValueError(
                        "Selected simulations resolve to different scalar diagnostic versions, "
                        "choices, or fields."
                    )

                missing: list[str] = []
                for simulation in selected:
                    values: dict[str, str | int | float | bool] = {}
                    for definition, selected_choices, selected_fields in selected_definitions:
                        choice_key = json.dumps(
                            selected_choices,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        stored = indexed_values.get(
                            (simulation.simulation_id, definition.name, definition.version, choice_key),
                            {},
                        )
                        if not set(selected_fields) <= set(stored):
                            missing.append(f"{simulation.collection_id}/{simulation.simulation_id}")
                            break
                        values.update({field: stored[field] for field in selected_fields})
                    else:
                        rows.append(
                            {
                                "collection_id": simulation.collection_id,
                                "simulation_id": simulation.simulation_id,
                                **configurations[simulation.simulation_id],
                                **values,
                            }
                        )

                if missing and not allow_missing:
                    raise KeyError(
                        "Selected simulations have no completed scalar result: "
                        + ", ".join(missing)
                    )
                if missing:
                    missing_keys = {
                        tuple(item.split("/", maxsplit=1))
                        for item in missing
                    }
                    missing_fields = {
                        field: float("nan")
                        for _, _, selected_fields in selected_definitions
                        for field in selected_fields
                    }
                    for simulation in selected:
                        if (simulation.collection_id, simulation.simulation_id) in missing_keys:
                            rows.append(
                                {
                                    "collection_id": simulation.collection_id,
                                    "simulation_id": simulation.simulation_id,
                                    **configurations[simulation.simulation_id],
                                    **missing_fields,
                                }
                            )

        if not rows:
            raise ValueError("No selected simulations have completed scalar results.")
        table = pandas.DataFrame(rows)
        table["collection_id"] = table["collection_id"].astype("category")
        table["simulation_id"] = table["simulation_id"].astype("category")
        table.attrs["diagnostics"] = reference[0] if reference is not None else ()
        table.attrs["choices"] = reference[1] if reference is not None else {}
        table.attrs["fields"] = reference[2] if reference is not None else {}
        table.attrs["catalogue_root"] = str(root)
        return table

    rows: list[dict[str, str | int | float | bool]] = []
    missing: list[str] = []
    schemas = {}
    reference: tuple[
        tuple[tuple[str, str], ...],
        dict[str, dict[str, str]],
        dict[str, tuple[str, ...]],
    ] | None = None

    for simulation in simulations:
        # Parse each collection schema once. Scalar files must still be read
        # per simulation because they contain its actual fitted values.
        collection_root = simulation.path.parent.parent
        if collection_root not in schemas:
            results = simulation.diagnostics.scalar
            schemas[collection_root] = results.collection_diagnostics
        else:
            results = SimulationDiagnosticResults(
                simulation.path,
                schemas[collection_root],
                "scalar",
            )
        collection_choices = {
            choice.name: choice
            for choice in results.collection_diagnostics.choices
        }
        unknown_choices = set(choices or {}) - set(collection_choices)
        if unknown_choices:
            names_text = ", ".join(sorted(unknown_choices))
            raise ValueError(f"choices contains unregistered names: {names_text}.")
        definitions = {
            definition.name: definition
            for definition in results.collection_diagnostics.diagnostics
            if definition.kind == "scalar"
        }
        unknown_diagnostics = set(names) - set(definitions)
        if unknown_diagnostics:
            unknown_text = ", ".join(sorted(unknown_diagnostics))
            available_text = ", ".join(sorted(definitions)) or "none"
            raise ValueError(
                f"Unknown scalar diagnostics: {unknown_text}. "
                f"Available: {available_text}."
            )

        values: dict[str, str | int | float | bool] = {}
        resolved_diagnostics: list[tuple[str, str]] = []
        resolved_choices: dict[str, dict[str, str]] = {}
        resolved_fields: dict[str, tuple[str, ...]] = {}
        try:
            for name in names:
                definition = definitions[name]
                selected_choices = {
                    choice_name: collection_choices[choice_name].default
                    for choice_name in definition.choices
                }
                selected_choices.update(
                    {
                        choice_name: value
                        for choice_name, value in (choices or {}).items()
                        if choice_name in definition.choices
                    }
                )
                overrides = (diagnostic_choices or {}).get(name, {})
                unknown_overrides = set(overrides) - set(definition.choices)
                if unknown_overrides:
                    choices_text = ", ".join(sorted(unknown_overrides))
                    raise ValueError(f"{name} has irrelevant diagnostic choices: {choices_text}.")
                selected_choices.update(overrides)
                invalid_choices = {
                    choice_name: value
                    for choice_name, value in selected_choices.items()
                    if value not in collection_choices[choice_name].values
                }
                if invalid_choices:
                    choices_text = ", ".join(
                        f"{choice_name}={value!r}"
                        for choice_name, value in invalid_choices.items()
                    )
                    available_text = ", ".join(
                        f"{choice_name}={collection_choices[choice_name].values!r}"
                        for choice_name in invalid_choices
                    )
                    raise ValueError(
                        f"{name} has unavailable choice values: {choices_text}. "
                        f"Available: {available_text}."
                    )
                selected_fields = tuple(
                    (fields or {}).get(name, tuple(field.name for field in definition.fields))
                )
                unknown_fields = set(selected_fields) - {field.name for field in definition.fields}
                if unknown_fields:
                    fields_text = ", ".join(sorted(unknown_fields))
                    available_text = ", ".join(field.name for field in definition.fields)
                    raise ValueError(
                        f"Unknown fields for scalar diagnostic {name!r}: {fields_text}. "
                        f"Available: {available_text}."
                    )
                duplicate_fields = set(values) & set(selected_fields)
                if duplicate_fields:
                    fields_text = ", ".join(sorted(duplicate_fields))
                    raise ValueError(f"Selected scalar fields collide: {fields_text}.")

                result = results[name]
                values.update(result.read(selected_choices, selected_fields))
                resolved_diagnostics.append((result.definition.name, result.definition.version))
                resolved_choices[result.definition.name] = selected_choices
                resolved_fields[result.definition.name] = selected_fields
        except KeyError:
            missing.append(f"{simulation.collection_id}/{simulation.simulation_id}")
            continue

        configuration = configuration_values(simulation)
        duplicate_fields = set(values) & set(configuration)
        if duplicate_fields:
            fields_text = ", ".join(sorted(duplicate_fields))
            raise ValueError(f"Scalar fields collide with configuration values: {fields_text}.")
        resolved = (
            tuple(resolved_diagnostics),
            resolved_choices,
            resolved_fields,
        )
        if reference is None:
            reference = resolved
        elif resolved != reference:
            raise ValueError(
                "Selected simulations resolve to different scalar diagnostic versions, "
                "choices, or fields."
            )
        rows.append(
            {
                "collection_id": simulation.collection_id,
                "simulation_id": simulation.simulation_id,
                **configuration,
                **values,
            }
        )

    if missing and not allow_missing:
        raise KeyError("Selected simulations have no completed scalar result: " + ", ".join(missing))
    if missing and reference is not None:
        missing_keys = {
            tuple(item.split("/", maxsplit=1))
            for item in missing
        }
        missing_fields = {
            field: float("nan")
            for selected_fields in reference[2].values()
            for field in selected_fields
        }
        for simulation in simulations:
            if (simulation.collection_id, simulation.simulation_id) in missing_keys:
                rows.append(
                    {
                        "collection_id": simulation.collection_id,
                        "simulation_id": simulation.simulation_id,
                        **configuration_values(simulation),
                        **missing_fields,
                    }
                )
    if not rows:
        raise ValueError("No selected simulations have completed scalar results.")

    table = pandas.DataFrame(rows)
    table["collection_id"] = table["collection_id"].astype("category")
    table["simulation_id"] = table["simulation_id"].astype("category")
    table.attrs["diagnostics"] = reference[0] if reference is not None else ()
    table.attrs["choices"] = reference[1] if reference is not None else {}
    table.attrs["fields"] = reference[2] if reference is not None else {}
    if root is not None:
        table.attrs["catalogue_root"] = str(root)
    return table
