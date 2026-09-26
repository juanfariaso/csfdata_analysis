"""Load, align, and summarize completed diagnostic time series."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy
import pandas

from csfdata.catalogue import CatalogueSimulation, CollectionDiagnostics, SimulationDiagnosticResults
from csfdata_analysis.datamodel.simulations import configuration_values


DiagnosticKey = tuple[str, str]
"""Stable ``(name, version)`` identity of one time-series diagnostic."""


def load_time_series(
    simulation_or_simulations: CatalogueSimulation | Sequence[CatalogueSimulation],
    diagnostics: str
    | Sequence[str]
    | dict[DiagnosticKey, dict[str, object]],
    allow_missing: bool = False,
    *,
    choices: dict[str, str] | None = None,
    diagnostic_choices: dict[str, dict[str, str]] | None = None,
    fields: dict[str, Sequence[str]] | None = None,
) -> pandas.DataFrame | dict[DiagnosticKey, pandas.DataFrame]:
    """Load time-series data from one simulation or a legacy simulation sequence.

    Args:
        simulation_or_simulations: One catalogue simulation for the concise
            single-simulation interface, or a sequence for the existing
            collection-level interface.
        diagnostics: For one simulation, one diagnostic name or a non-empty
            sequence of names. For a simulation sequence, the existing
            dictionary of exact ``(name, version)`` selections.
        allow_missing: Existing collection-level option to omit simulations
            without completed selected diagnostics. It does not apply to one
            simulation.
        choices: Optional global choice values. Applicable values override
            each diagnostic's registered defaults; irrelevant registered
            choices are ignored.
        diagnostic_choices: Optional per-diagnostic choice overrides. These
            take precedence over ``choices`` for the named diagnostic.
        fields: Optional fields by diagnostic name. Omitted diagnostics load
            every declared field.

    Returns:
        For one simulation, one Pandas table with a shared ``time`` column and
        the requested diagnostic fields. For a simulation sequence, the legacy
        dictionary of one wide table per exact diagnostic identity.

    Raises:
        KeyError: If a named diagnostic, selected field, or choice combination
            is unavailable.
        ValueError: If an input selection is malformed, diagnostic time grids
            differ, output fields collide, or legacy-only options are used for
            one simulation.

    Notes:
        A name resolves to the latest version declared in the collection. The
        resulting single-simulation table records exact identities, resolved
        choices, and fields in ``DataFrame.attrs`` for later inspection.
    """
    if not isinstance(simulation_or_simulations, CatalogueSimulation):
        if choices is not None or diagnostic_choices is not None or fields is not None:
            raise ValueError(
                "choices, diagnostic_choices, and fields apply only when loading one simulation."
            )
        if not isinstance(diagnostics, dict):
            raise ValueError(
                "Loading multiple simulations requires exact diagnostic selection dictionaries."
            )
        return load_collection_time_series(
            simulation_or_simulations,
            diagnostics,
            allow_missing,
        )

    if allow_missing:
        raise ValueError("allow_missing applies only when loading multiple simulations.")
    simulation = simulation_or_simulations
    return _load_time_series_from_results(
        simulation,
        diagnostics,
        choices,
        diagnostic_choices,
        fields,
        simulation.diagnostics.time_series,
    )


def _load_time_series_from_results(
    simulation: CatalogueSimulation,
    diagnostics: str | Sequence[str] | dict[DiagnosticKey, dict[str, object]],
    choices: dict[str, str] | None,
    diagnostic_choices: dict[str, dict[str, str]] | None,
    fields: dict[str, Sequence[str]] | None,
    results: SimulationDiagnosticResults,
) -> pandas.DataFrame:
    """Load one simulation using an already resolved time-series schema.

    Args:
        simulation: Catalogue simulation whose result files are read.
        diagnostics: One name or a non-empty sequence of diagnostic names.
        choices: Optional global choice values.
        diagnostic_choices: Optional per-diagnostic choice overrides.
        fields: Optional fields by diagnostic name.
        results: Time-series view using the correct collection diagnostics.

    Returns:
        One Pandas table with a shared ``time`` column and selected fields.

    Raises:
        KeyError: If a selected result, field, or choice combination is absent.
        ValueError: If a selection is malformed, fields collide, or diagnostic
            time coordinates differ.

    Notes:
        ``load_collection_time_series`` supplies this view from a call-scoped
        collection-schema cache. The public one-simulation API supplies its
        ordinary lazy view, so both paths retain identical semantics.
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
        raise ValueError("A single simulation requires one or more non-empty diagnostic names.")
    if not names or len(set(names)) != len(names):
        raise ValueError("Diagnostic names must be non-empty and unique.")
    if choices is not None and (
        not isinstance(choices, dict)
        or not all(isinstance(name, str) and isinstance(value, str) for name, value in choices.items())
    ):
        raise ValueError("choices must be a dictionary of string names and values.")
    if diagnostic_choices is not None and (
        not isinstance(diagnostic_choices, dict)
        or not all(
            isinstance(name, str)
            and isinstance(value, dict)
            and all(isinstance(choice, str) and isinstance(selected, str) for choice, selected in value.items())
            for name, value in diagnostic_choices.items()
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

    # Validate global choices against the collection before ignoring values
    # that are irrelevant to an individual selected diagnostic.
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
        if definition.kind == "time_series"
    }
    unknown_diagnostics = set(names) - set(definitions)
    if unknown_diagnostics:
        unknown_text = ", ".join(sorted(unknown_diagnostics))
        available_text = ", ".join(sorted(definitions)) or "none"
        raise ValueError(
            f"Unknown time-series diagnostics: {unknown_text}. "
            f"Available: {available_text}."
        )

    columns: dict[str, numpy.ndarray] = {}
    reference_times: numpy.ndarray | None = None
    resolved_diagnostics: list[tuple[str, str]] = []
    resolved_choices: dict[str, dict[str, str]] = {}
    resolved_fields: dict[str, tuple[str, ...]] = {}
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
        selected_fields = tuple((fields or {}).get(name, tuple(field.name for field in definition.fields)))
        unknown_fields = set(selected_fields) - {field.name for field in definition.fields}
        if unknown_fields:
            fields_text = ", ".join(sorted(unknown_fields))
            available_text = ", ".join(field.name for field in definition.fields)
            raise ValueError(
                f"Unknown fields for time-series diagnostic {name!r}: {fields_text}. "
                f"Available: {available_text}."
            )
        duplicate_fields = set(columns) & set(selected_fields)
        if duplicate_fields:
            fields_text = ", ".join(sorted(duplicate_fields))
            raise ValueError(f"Selected diagnostic fields collide: {fields_text}.")

        # Name lookup deliberately resolves the latest collection-declared
        # version, while the stored identity remains attached to the table.
        result = results[name]
        values = result.read(selected_choices, selected_fields)
        times = values.pop("time")
        if reference_times is None:
            reference_times = times
            columns["time"] = times
        elif not numpy.array_equal(reference_times, times):
            raise ValueError(
                f"Diagnostic time grid differs from {resolved_diagnostics[0]!r}: "
                f"{result.definition.name!r}."
            )
        columns.update(values)
        resolved_diagnostics.append((result.definition.name, result.definition.version))
        resolved_choices[result.definition.name] = selected_choices
        resolved_fields[result.definition.name] = selected_fields

    table = pandas.DataFrame(columns)
    table.attrs["diagnostics"] = tuple(resolved_diagnostics)
    table.attrs["choices"] = resolved_choices
    table.attrs["fields"] = resolved_fields
    return table


def load_collection_time_series(
    simulations: Sequence[CatalogueSimulation],
    diagnostics: str | Sequence[str] | dict[DiagnosticKey, dict[str, object]],
    allow_missing: bool = False,
    *,
    choices: dict[str, str] | None = None,
    diagnostic_choices: dict[str, dict[str, str]] | None = None,
    fields: dict[str, Sequence[str]] | None = None,
) -> pandas.DataFrame | dict[DiagnosticKey, pandas.DataFrame]:
    """Load completed time-series diagnostics from selected simulations.

    Args:
        simulations: Catalogue simulations selected by ``load_simulations``.
        diagnostics: One name or a sequence of names for the current concise
            interface. A legacy dictionary of exact ``(name, version)``
            selections remains supported for backward compatibility.
        allow_missing: Whether to omit simulations without a completed selected
            result. By default, any missing selected result raises an error.
        choices: Optional global choices for the concise interface.
        diagnostic_choices: Optional per-diagnostic concise-interface choice
            overrides.
        fields: Optional fields by diagnostic name for the concise interface.

    Returns:
        For names, one long Pandas table with IDs, canonical configuration
        values, ``time``, and selected fields. For a legacy dictionary, one
        wide table per exact diagnostic identity.

    Raises:
        KeyError: If a requested result is unavailable and ``allow_missing``
            is ``False``.
        ValueError: If a selection is malformed or selected simulations resolve
            to incompatible diagnostic schemas.

    Notes:
        The concise interface delegates one-simulation reads to
        :func:`load_time_series`, so it uses the same latest-version and
        choice-resolution rules. Its returned table retains resolved details
        in ``DataFrame.attrs`` for later alignment and aggregation.
    """
    if not isinstance(diagnostics, dict):
        if not simulations:
            raise ValueError("simulations must contain at least one simulation.")

        tables: list[pandas.DataFrame] = []
        missing: list[str] = []
        reference_attrs: dict[str, object] | None = None
        schemas: dict[Path, CollectionDiagnostics] = {}
        for simulation in simulations:
            collection_root = simulation.path.parent.parent
            if collection_root not in schemas:
                # Read the collection schema once, then reuse this parsed
                # definition for every later simulation in the collection.
                results = simulation.diagnostics.time_series
                schemas[collection_root] = results.collection_diagnostics
            else:
                results = SimulationDiagnosticResults(
                    simulation.path,
                    schemas[collection_root],
                    "time_series",
                )
            try:
                table = _load_time_series_from_results(
                    simulation,
                    diagnostics,
                    choices,
                    diagnostic_choices,
                    fields,
                    results,
                )
            except KeyError:
                missing.append(f"{simulation.collection_id}/{simulation.simulation_id}")
                continue

            # Every table must describe the same scientific measurements before
            # their rows can share one collection-level DataFrame.
            if reference_attrs is None:
                reference_attrs = dict(table.attrs)
            elif any(
                table.attrs.get(name) != reference_attrs.get(name)
                for name in ("diagnostics", "choices", "fields")
            ):
                raise ValueError(
                    "Selected simulations resolve to different diagnostic versions, "
                    "choices, or fields."
                )

            # Repeat immutable catalogue metadata on each output row so later
            # grouping, interpolation, and plotting never need another lookup.
            metadata = {
                "collection_id": simulation.collection_id,
                "simulation_id": simulation.simulation_id,
                **configuration_values(simulation),
            }
            tables.append(
                pandas.DataFrame(
                    {
                        **metadata,
                        **{column: table[column] for column in table.columns},
                    }
                )
            )
        if missing and not allow_missing:
            raise KeyError("Selected simulations have no completed result: " + ", ".join(missing))
        if not tables:
            raise ValueError("No selected simulations have completed diagnostic results.")

        collection_table = pandas.concat(tables, ignore_index=True)
        collection_table["collection_id"] = collection_table["collection_id"].astype("category")
        collection_table["simulation_id"] = collection_table["simulation_id"].astype("category")
        collection_table.attrs.update(reference_attrs or {})
        collection_table.attrs["data_fields"] = tuple(
            field
            for fields_for_diagnostic in collection_table.attrs["fields"].values()
            for field in fields_for_diagnostic
        )
        return collection_table

    if choices is not None or diagnostic_choices is not None or fields is not None:
        raise ValueError(
            "choices, diagnostic_choices, and fields by name require diagnostic names, not legacy selections."
        )
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
        if not isinstance(selection, dict):
            raise ValueError(f"Diagnostic selection for {identity!r} must be a dictionary.")
        choices = selection.get("choices")
        fields = selection.get("fields")
        if not isinstance(choices, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in choices.items()
        ):
            raise ValueError(f"Diagnostic selection for {identity!r} needs a string choices dictionary.")
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
            times = values.pop("time")
            lengths = {len(times), *(len(values[field]) for field in fields)}
            if len(lengths) != 1:
                raise ValueError(
                    f"Stored time-series lengths differ: "
                    f"{simulation.collection_id}/{simulation.simulation_id} {identity!r}."
                )

            # Attach invariant simulation metadata to every time row, making
            # the returned table immediately usable for grouping and plotting.
            parameters = configuration_values(simulation)
            for index, time in enumerate(times):
                rows.append(
                    {
                        "collection_id": simulation.collection_id,
                        "simulation_id": simulation.simulation_id,
                        **parameters,
                        **dict(choices),
                        "time": float(time),
                        **{field: float(values[field][index]) for field in fields},
                    }
                )
        if missing and not allow_missing:
            raise ValueError(
                f"Selected simulations have no completed {identity[0]} {identity[1]} result: "
                + ", ".join(missing)
            )
        table = pandas.DataFrame(rows)
        table["collection_id"] = table["collection_id"].astype("category")
        table["simulation_id"] = table["simulation_id"].astype("category")
        table.attrs["fields"] = tuple(fields)
        table.attrs["choices"] = dict(choices)
        tables[identity] = table
    return tables


def interpolate_time_series(
    data: pandas.DataFrame | dict[DiagnosticKey, pandas.DataFrame],
    times: Sequence[float],
) -> pandas.DataFrame | dict[DiagnosticKey, pandas.DataFrame]:
    """Linearly interpolate selected diagnostics onto one physical-time grid.

    Args:
        data: Long collection table returned by
            :func:`load_collection_time_series`, or legacy diagnostic tables.
        times: Explicit target model times in Myr.

    Returns:
        An aligned long collection table or one aligned legacy table per
        diagnostic. Values outside an individual simulation's stored range are
        ``NaN``; the function never extrapolates.

    Raises:
        ValueError: If data lacks stored field metadata, times are not a
            strictly increasing non-empty sequence, or one simulation has
            duplicate or unordered source times.
    """
    targets = numpy.asarray(times, dtype=float)
    if targets.ndim != 1 or len(targets) == 0 or numpy.any(numpy.diff(targets) <= 0):
        raise ValueError("times must be one non-empty strictly increasing sequence.")

    collection_table = isinstance(data, pandas.DataFrame)
    tables = {None: data} if collection_table else data
    aligned_tables: dict[DiagnosticKey | None, pandas.DataFrame] = {}
    for identity, table in tables.items():
        fields = table.attrs.get("data_fields" if collection_table else "fields")
        if not isinstance(fields, tuple) or not fields:
            raise ValueError(f"Diagnostic table {identity!r} has no declared fields metadata.")
        required_columns = {"collection_id", "simulation_id", "time", *fields}
        if not required_columns.issubset(table.columns):
            missing = ", ".join(sorted(required_columns - set(table.columns)))
            raise ValueError(f"Diagnostic table {identity!r} is missing columns: {missing}")

        rows: list[dict[str, str | int | float | bool]] = []
        for _, group in table.groupby(
            ["collection_id", "simulation_id"],
            sort=False,
            observed=True,
        ):
            # A unique ascending source grid is required for one well-defined
            # linear interpolation per simulation and output field.
            ordered = group.sort_values("time")
            source_times = ordered["time"].to_numpy(dtype=float)
            if numpy.any(numpy.diff(source_times) <= 0):
                label = f"{ordered.iloc[0]['collection_id']}/{ordered.iloc[0]['simulation_id']}"
                raise ValueError(f"Simulation has duplicate or unordered times: {label}")
            metadata = ordered.iloc[0].drop(labels=["time", *fields]).to_dict()
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
                        "time": float(target),
                        **{field: float(interpolated_fields[field][index]) for field in fields},
                    }
                )
        aligned = pandas.DataFrame(rows)
        aligned["collection_id"] = aligned["collection_id"].astype("category")
        aligned["simulation_id"] = aligned["simulation_id"].astype("category")
        aligned.attrs.update(table.attrs)
        aligned_tables[identity] = aligned
    return aligned_tables[None] if collection_table else aligned_tables


def aggregate_time_series(
    data: pandas.DataFrame | dict[DiagnosticKey, pandas.DataFrame],
    group_by: Sequence[str] = (),
) -> pandas.DataFrame | dict[DiagnosticKey, pandas.DataFrame]:
    """Calculate ensemble mean and standard deviation at every stored time.

    Args:
        data: Aligned long collection table or legacy diagnostic tables
            returned by :func:`interpolate_time_series`.
        group_by: Optional configuration columns that define separate
            ensembles, such as ``("tff", "sfe")``.

    Returns:
        One summary table, or one legacy summary table per diagnostic. Each
        has grouping columns, ``time``, ``n_simulations``, and
        ``<field>_mean`` and ``<field>_std`` columns.

    Raises:
        ValueError: If a requested grouping column or declared field is absent.

    Notes:
        Standard deviations use sample normalization (``ddof=1``), and are
        ``NaN`` for groups with fewer than two finite values.
    """
    collection_table = isinstance(data, pandas.DataFrame)
    tables = {None: data} if collection_table else data
    summaries: dict[DiagnosticKey | None, pandas.DataFrame] = {}
    for identity, table in tables.items():
        fields = table.attrs.get("data_fields" if collection_table else "fields")
        if not isinstance(fields, tuple) or not fields:
            raise ValueError(f"Diagnostic table {identity!r} has no declared fields metadata.")
        columns = {"collection_id", "time", *group_by, *fields}
        if not columns.issubset(table.columns):
            missing = ", ".join(sorted(columns - set(table.columns)))
            raise ValueError(f"Diagnostic table {identity!r} is missing columns: {missing}")

        # Group only by stable metadata and model time. A simulation contributes
        # only when it has finite values for every requested output field.
        grouping = ["collection_id", *group_by, "time"]
        rows: list[dict[str, str | int | float | bool]] = []
        for values, group in table.groupby(
            grouping,
            sort=False,
            dropna=False,
            observed=True,
        ):
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
        if "collection_id" in summary:
            summary["collection_id"] = summary["collection_id"].astype("category")
        if "simulation_id" in summary:
            summary["simulation_id"] = summary["simulation_id"].astype("category")
        summary.attrs.update(table.attrs)
        # Preserve the scientific grouping decision so plotting can reject a
        # figure that would otherwise mix unresolved model parameters.
        summary.attrs["group_by"] = tuple(group_by)
        summaries[identity] = summary
    return summaries[None] if collection_table else summaries
