"""Load selected scalar diagnostics into tables for filtering and plotting."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sqlite3

import pandas

from csfdata.catalogue import CatalogueSimulation, SimulationDiagnosticResults
from csfdata.catalogue.diagnostics import collection_diagnostics_path, read_collection_diagnostics
from csfdata_analysis.datamodel.simulations import configuration_values


def load_collection_scalars(
    simulations: Sequence[CatalogueSimulation],
    diagnostics: str | Sequence[str],
    allow_missing: bool = False,
    *,
    choices: dict[str, str] | None = None,
    diagnostic_choices: dict[str, dict[str, str]] | None = None,
    fields: dict[str, Sequence[str]] | None = None,
    catalogue_root: Path | str | None = None,
) -> pandas.DataFrame:
    """Load scalar diagnostic values from selected catalogue simulations.

    Args:
        simulations: Catalogue simulations selected by ``load_simulations``.
        diagnostics: One scalar diagnostic name or a sequence of names. Each
            name resolves to its latest collection-declared version.
        allow_missing: Whether to omit simulations without every selected
            scalar result. By default, missing results raise an error.
        choices: Optional global choice values. Registered defaults are used
            for relevant choices not supplied here.
        diagnostic_choices: Optional per-diagnostic choice overrides. These
            take precedence over ``choices``.
        fields: Optional selected fields by diagnostic name. Omitted entries
            load every declared field for that diagnostic.
        catalogue_root: Optional indexed catalogue root. When its
            ``registry.sqlite`` exists, scalar values and configuration values
            are read from SQLite instead of per-simulation YAML files.

    Returns:
        One Pandas row per selected simulation, containing collection and
        simulation IDs, canonical configuration values, and selected scalar
        fields. The table attrs record the resolved diagnostic versions,
        choices, and fields.

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

        if not rows:
            raise ValueError("No selected simulations have completed scalar results.")
        table = pandas.DataFrame(rows)
        table["collection_id"] = table["collection_id"].astype("category")
        table["simulation_id"] = table["simulation_id"].astype("category")
        table.attrs["diagnostics"] = reference[0] if reference is not None else ()
        table.attrs["choices"] = reference[1] if reference is not None else {}
        table.attrs["fields"] = reference[2] if reference is not None else {}
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
    if not rows:
        raise ValueError("No selected simulations have completed scalar results.")

    table = pandas.DataFrame(rows)
    table["collection_id"] = table["collection_id"].astype("category")
    table["simulation_id"] = table["simulation_id"].astype("category")
    table.attrs["diagnostics"] = reference[0] if reference is not None else ()
    table.attrs["choices"] = reference[1] if reference is not None else {}
    table.attrs["fields"] = reference[2] if reference is not None else {}
    return table
