"""Run one standardized diagnostic over one imported simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from uuid import uuid4

import h5py
from amuse.datamodel import Particles
from amuse.units import units

from csfdata.adapters.base import SimulationAdapter
from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.diagnostics import (
    ScalarDiagnosticResult,
    ScalarValue,
    SimulationScalarDiagnostics,
    read_collection_diagnostics,
    read_simulation_scalar_diagnostics,
    simulation_scalar_diagnostics_path,
    write_simulation_scalar_diagnostics,
)
from csfdata_analysis.diagnostics import (
    ScalarDiagnostic,
    TimeSeriesDiagnostic,
    diagnostic_directories,
    ensure_collection_diagnostics,
    load_diagnostics,
)


@dataclass(frozen=True)
class SimulationReport:
    """Describe one successfully written diagnostic time series.

    Args:
        simulation_root: Imported simulation directory that was analysed.
        output_path: Final path of the created HDF5 time series.
        snapshot_count: Number of snapshots evaluated successfully.
        first_time_myr: First snapshot model time in Myr.
        last_time_myr: Final snapshot model time in Myr.
        dry_run: Whether this report describes a plan that created no file.
    """

    simulation_root: Path
    output_path: Path
    snapshot_count: int
    first_time_myr: float
    last_time_myr: float
    dry_run: bool


@dataclass(frozen=True)
class ScalarReport:
    """Describe one scalar-diagnostic calculation for one simulation.

    Args:
        simulation_root: Imported simulation directory that was analysed.
        output_path: Shared scalar-diagnostics YAML path.
        results: Stored results for every declared choice combination.
        written_count: Number of new or replaced results written in this call.
    """

    simulation_root: Path
    output_path: Path
    results: tuple[ScalarDiagnosticResult, ...]
    written_count: int


def time_series_path(simulation_root: Path, diagnostic: TimeSeriesDiagnostic) -> Path:
    """Return the standard output path for one diagnostic time series.

    Args:
        simulation_root: Imported simulation directory containing ``derived/``.
        diagnostic: Versioned diagnostic whose path is required.

    Returns:
        Path: Destination ``series.h5`` path. The path may not exist yet.
    """
    return (
        simulation_root
        / "derived"
        / "diagnostics"
        / diagnostic.name
        / f"v{diagnostic.version}"
        / "series.h5"
    )


def compute_time_series(
    simulation_root: Path,
    diagnostic: TimeSeriesDiagnostic,
    adapter: SimulationAdapter,
    read_snapshot: Callable[[Path], Particles],
    dry_run: bool = False,
    overwrite: bool = False,
    output_path: Path | None = None,
    identity: Mapping[str, str | int] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> SimulationReport:
    """Compute one diagnostic time series for one imported simulation.

    Args:
        simulation_root: Imported simulation directory containing ``raw/`` and
            ``derived/`` directories.
        diagnostic: Versioned diagnostic to evaluate.
        adapter: Format-specific adapter initialized with this simulation's
            ``raw/`` directory.
        read_snapshot: Function that converts one adapter snapshot path into
            AMUSE particles.
        dry_run: Whether to validate the snapshot plan without reading AMUSE
            particles or creating output.
        overwrite: Whether a completed output at the destination may be
            atomically replaced after a successful calculation.
        output_path: Optional alternate final HDF5 destination. This allows a
            lite catalogue to store results while raw inputs remain elsewhere.
        identity: Optional immutable simulation identity written as HDF5
            attributes for a later derived-data import.
        progress: Optional callback invoked after each evaluated snapshot. It
            receives the one-based completed count, total count, and source
            snapshot ID.

    Returns:
        SimulationReport: Location, time range, and snapshot count of the
            completed time series or dry-run plan.

    Raises:
        FileExistsError: If this diagnostic version already has a completed
            time series for the simulation and ``overwrite`` is ``False``.
        ValueError: If snapshots have no model time, a snapshot lies outside
            the adapter's raw directory, diagnostic choices are invalid, a
            diagnostic returns unexpected parameter names, or a returned value
            is not a scalar in a supported canonical unit.

    Notes:
        The function evaluates all snapshots before creating a temporary HDF5
        file. It then atomically renames that file into ``derived/``. Raw input
        data and completed time series are never modified.
    """
    raw_root = simulation_root / "raw"
    if adapter.run_root != raw_root:
        raise ValueError("The adapter must be initialized with the simulation raw directory.")

    destination = output_path or time_series_path(simulation_root, diagnostic)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Time-series diagnostic output already exists: {destination}")

    snapshot_paths = adapter.snapshot_paths()
    if not snapshot_paths:
        raise ValueError(f"No snapshots found for simulation: {simulation_root}")

    times_myr = []
    snapshot_ids = []
    for snapshot_number, snapshot_path in enumerate(snapshot_paths, start=1):
        try:
            snapshot_id = str(snapshot_path.relative_to(raw_root))
        except ValueError as error:
            raise ValueError(
                f"Snapshot path is outside the simulation raw directory: {snapshot_path}"
            ) from error
        time_myr = adapter.snapshot_time(snapshot_path)
        if time_myr is None:
            raise ValueError(f"Snapshot has no readable model time: {snapshot_id}")
        times_myr.append(time_myr)
        snapshot_ids.append(snapshot_id)

    if dry_run:
        return SimulationReport(
            simulation_root=simulation_root,
            output_path=destination,
            snapshot_count=len(snapshot_paths),
            first_time_myr=times_myr[0],
            last_time_myr=times_myr[-1],
            dry_run=True,
        )

    if not diagnostic.evaluation_choices:
        raise ValueError(
            f"Time-series diagnostic {diagnostic.name!r} must declare at least one choice."
        )
    choices_by_name = {
        choice.name: choice for choice in diagnostic.evaluation_choices
    }
    if len(choices_by_name) != len(diagnostic.evaluation_choices):
        raise ValueError(
            f"Time-series diagnostic {diagnostic.name!r} declares duplicate choice names."
        )

    canonical_units = {
        "Myr": units.Myr,
        "pc": units.pc,
        "Msun": units.MSun,
        "km/s": units.kms,
        "1": units.none,
    }
    unsupported_units = sorted(
        {
            unit_name
            for choice in diagnostic.evaluation_choices
            for unit_name in choice.outputs.values()
            if unit_name not in canonical_units
        }
    )
    if unsupported_units:
        raise ValueError(
            "Time-series diagnostic declares unsupported canonical units: "
            f"{', '.join(unsupported_units)}."
        )

    values = {
        choice.name: {name: [] for name in choice.outputs}
        for choice in diagnostic.evaluation_choices
    }
    expected_choice_names = set(choices_by_name)
    for snapshot_path in snapshot_paths:
        measurements = diagnostic.evaluate(read_snapshot(snapshot_path))
        if set(measurements) != expected_choice_names:
            raise ValueError(
                f"Time-series diagnostic {diagnostic.name!r} returned choices "
                f"{sorted(measurements)}, "
                f"but declares {sorted(expected_choice_names)}."
            )

        for choice_name, choice in choices_by_name.items():
            if set(measurements[choice_name]) != set(choice.outputs):
                raise ValueError(
                    f"Time-series diagnostic {diagnostic.name!r} choice "
                    f"{choice_name!r} returned "
                    f"{sorted(measurements[choice_name])}, but declares "
                    f"{sorted(choice.outputs)}."
                )
            for name, unit_name in choice.outputs.items():
                try:
                    values[choice_name][name].append(
                        float(measurements[choice_name][name].value_in(canonical_units[unit_name]))
                    )
                except (AttributeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Time-series diagnostic {diagnostic.name!r} choice "
                        f"{choice_name!r} returned a "
                        f"non-scalar or incompatible value for {name!r}; expected {unit_name}."
                    ) from error
        if progress is not None:
            progress(snapshot_number, len(snapshot_paths), snapshot_ids[snapshot_number - 1])

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_path = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    with h5py.File(staging_path, "w") as output_file:
        output_file.attrs["format_schema_version"] = 1
        output_file.attrs["diagnostic_name"] = diagnostic.name
        output_file.attrs["diagnostic_version"] = diagnostic.version
        for name, value in (identity or {}).items():
            output_file.attrs[name] = value
        output_file.create_dataset("time_myr", data=times_myr)
        output_file.create_dataset(
            "snapshot_id",
            data=snapshot_ids,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        choice_group = output_file.create_group("choices")
        for choice in diagnostic.evaluation_choices:
            output_choice_group = choice_group.create_group(choice.name)
            for name, value in choice.metadata.items():
                output_choice_group.attrs[name] = value
            for name, unit_name in choice.outputs.items():
                dataset = output_choice_group.create_dataset(name, data=values[choice.name][name])
                dataset.attrs["unit"] = unit_name
        output_file.attrs["complete"] = True

    # The rename replaces an old completed file only after every snapshot was
    # evaluated and the replacement HDF5 file was fully written.
    staging_path.replace(destination)
    return SimulationReport(
        simulation_root=simulation_root,
        output_path=destination,
        snapshot_count=len(snapshot_paths),
        first_time_myr=times_myr[0],
        last_time_myr=times_myr[-1],
        dry_run=False,
    )


def compute_scalar_diagnostic(
    simulation: CatalogueSimulation,
    diagnostic: ScalarDiagnostic,
    overwrite: bool = False,
    available_diagnostics: Mapping[
        tuple[str, str], TimeSeriesDiagnostic | ScalarDiagnostic
    ] | None = None,
) -> ScalarReport:
    """Compute every missing choice combination of one scalar diagnostic.

    Args:
        simulation: Indexed catalogue simulation where scalar results are
            stored and whose published diagnostics can be read.
        diagnostic: Versioned scalar diagnostic to evaluate.
        overwrite: Whether existing results for this diagnostic version and
            choice combination may be replaced.
        available_diagnostics: Optional built-in and trusted local diagnostic
            registry used to resolve requirements. ``None`` loads directories
            configured in the collection's local ``analysis.yaml``.

    Returns:
        Report containing stored results for the requested diagnostic version
        and its declared choice combinations, including already present values.

    Raises:
        FileNotFoundError: If a declared required diagnostic product is absent.
        ValueError: If collection definitions conflict, a requirement is not
            published, the evaluator returns the wrong fields, or output
            values have unsupported types.

    Notes:
        This operation owns scalar-result storage but not the scientific
        calculation. It registers the declared collection schema before
        computing so the core writer can validate every generated result.
    """
    simulation_root = simulation.path
    collection_root = simulation.diagnostics.collection_root
    registry = available_diagnostics
    if registry is None:
        configuration_root = collection_root
        if collection_root.parent.name == "collections":
            # Collection metadata lives below the catalogue root, while local
            # diagnostic directories are configured once for that catalogue.
            configuration_root = collection_root.parent.parent
        registry = load_diagnostics(diagnostic_directories(configuration_root))
    ensure_collection_diagnostics(collection_root, (diagnostic,), registry)
    collection_diagnostics = read_collection_diagnostics(
        collection_root / "diagnostics.yaml"
    )

    # Confirm declared prerequisite files exist before evaluating any choices.
    for requirement in diagnostic.requires:
        definition = next(
            (
                candidate
                for candidate in collection_diagnostics.diagnostics
                if candidate.name == requirement.name
                and candidate.version == requirement.version
            ),
            None,
        )
        if definition is None:
            raise ValueError(
                "Scalar diagnostic requirement is not published: "
                f"{requirement.name} {requirement.version}."
            )
        required_path = simulation_root / definition.relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                "Scalar diagnostic requirement is missing: "
                f"{requirement.name} {requirement.version} at {required_path}."
            )

    output_path = simulation_scalar_diagnostics_path(simulation_root)
    existing = (
        read_simulation_scalar_diagnostics(output_path, collection_diagnostics)
        if output_path.is_file()
        else SimulationScalarDiagnostics(())
    )
    existing_by_choices = {
        result.choices: result
        for result in existing.results
        if result.diagnostic_name == diagnostic.name
        and result.diagnostic_version == f"v{diagnostic.version}"
    }
    choices_by_name = {choice.name: choice for choice in collection_diagnostics.choices}
    choice_values = [
        choices_by_name[choice_name].values for choice_name in diagnostic.choice_names
    ]
    choice_combinations = tuple(product(*choice_values)) if choice_values else ((),)
    new_results = []

    for combination in choice_combinations:
        selected_choices = tuple(zip(diagnostic.choice_names, combination, strict=True))
        if selected_choices in existing_by_choices and not overwrite:
            continue
        calculated_values = diagnostic.evaluate(simulation, dict(selected_choices))
        declared_fields = {field.name: field for field in diagnostic.fields}
        if set(calculated_values) != set(declared_fields):
            raise ValueError(
                f"Scalar diagnostic {diagnostic.name!r} returned "
                f"{sorted(calculated_values)}, but declares {sorted(declared_fields)}."
            )
        for name, value in calculated_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Scalar diagnostic {diagnostic.name!r} returned a non-numeric "
                    f"value for {name!r}."
                )
        new_results.append(
            ScalarDiagnosticResult(
                diagnostic.name,
                f"v{diagnostic.version}",
                selected_choices,
                tuple(
                    ScalarValue(field.name, calculated_values[field.name], field.unit)
                    for field in diagnostic.fields
                ),
            )
        )

    if new_results:
        # Retain unrelated diagnostics and replace only explicitly requested
        # choice combinations after every new value has been validated.
        new_keys = {result.choices for result in new_results}
        retained_results = tuple(
            result
            for result in existing.results
            if not (
                result.diagnostic_name == diagnostic.name
                and result.diagnostic_version == f"v{diagnostic.version}"
                and result.choices in new_keys
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_simulation_scalar_diagnostics(
            SimulationScalarDiagnostics(retained_results + tuple(new_results)),
            collection_diagnostics,
            output_path,
        )
        existing_by_choices.update({result.choices: result for result in new_results})

    results = tuple(
        existing_by_choices[tuple(zip(diagnostic.choice_names, combination, strict=True))]
        for combination in choice_combinations
    )
    return ScalarReport(
        simulation_root,
        output_path,
        results,
        len(new_results),
    )
