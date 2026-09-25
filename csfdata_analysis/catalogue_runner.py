"""Parallel execution of standardized diagnostics over catalogue selections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from csfdata.adapters.dcaf import DcafAdapter
from csfdata.catalogue import CatalogueSimulation, file_sha256, is_lite_catalogue
from csfdata_analysis.datamodel.simulations import source_simulation
from csfdata_analysis.diagnostics import (
    TIME_SERIES_DIAGNOSTICS,
    ensure_collection_diagnostics,
    load_diagnostics,
    scalar_diagnostic,
    time_series_diagnostic,
)
from csfdata_analysis.readers import read_stars
from csfdata_analysis.runner import (
    ScalarReport,
    SimulationReport,
    compute_scalar_diagnostic,
    compute_time_series,
    time_series_path,
)


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of analysing one selected catalogue simulation.

    Args:
        simulation: Indexed catalogue simulation selected for analysis.
        status: ``"complete"``, ``"ready"``, ``"skipped"``, or ``"failed"``.
        report: Per-simulation time-series or scalar report for completed work.
        error: Human-readable failure reason, if the status is ``"failed"``.

    Notes:
        ``"ready"`` is used by a time-series dry run. ``"skipped"`` means
        the requested completed diagnostic result was left unchanged.
    """

    simulation: CatalogueSimulation
    status: str
    report: SimulationReport | ScalarReport | None = None
    error: str | None = None


def compute_catalogue_simulation(
    simulation: CatalogueSimulation,
    diagnostic_name: str,
    dry_run: bool = False,
    overwrite: bool = False,
    diagnostic_directories: tuple[Path | str, ...] = (),
    diagnostic_version: str | None = None,
    update: bool = False,
) -> SimulationResult:
    """Compute one registered diagnostic for one catalogue simulation.

    Args:
        simulation: Indexed simulation selected by ``csfdata``.
        diagnostic_name: Registered diagnostic name, such as
            ``"lagrangian_radii"``.
        dry_run: Whether to inspect the snapshot plan without writing output.
        overwrite: Whether a completed selected diagnostic may be recomputed.
        diagnostic_directories: Trusted local directories containing additional
            diagnostic modules. Each worker loads these directories itself.
        diagnostic_version: Optional exact diagnostic version. ``None`` uses
            the highest registered version for ``diagnostic_name``.
        update: Whether incomplete or outdated existing results should be
            replaced while current completed results remain untouched.

    Returns:
        A result describing completed, ready, skipped, or failed work.

    Notes:
        This function is intentionally module-level so it can run in a process
        worker. It currently supports catalogue simulations imported with the
        D-CAF adapter.
    """
    try:
        diagnostic = time_series_diagnostic(
            diagnostic_name,
            diagnostic_directories,
            diagnostic_version,
        )
    except ValueError:
        return SimulationResult(
            simulation,
            "failed",
            error=f"Unknown diagnostic: {diagnostic_name}",
        )
    if simulation.importer != "dcaf":
        return SimulationResult(
            simulation,
            "failed",
            error=f"Unsupported catalogue importer: {simulation.importer}",
        )
    destination = time_series_path(simulation.path, diagnostic)
    replace_existing = overwrite
    if destination.exists() and not overwrite and not update:
        return SimulationResult(simulation, "skipped")
    if destination.exists() and update and not overwrite:
        try:
            simulation.diagnostics.time_series[(diagnostic.name, f"v{diagnostic.version}")]
        except (KeyError, OSError, ValueError):
            # An older schema remains in place until a fully computed result
            # can atomically replace it.
            replace_existing = True
        else:
            return SimulationResult(simulation, "skipped")
    try:
        input_simulation = source_simulation(simulation)
        report = compute_time_series(
            input_simulation,
            diagnostic,
            DcafAdapter(input_simulation / "raw"),
            read_stars,
            dry_run=dry_run,
            overwrite=replace_existing,
            output_path=destination,
            identity={
                "collection_id": simulation.collection_id,
                "simulation_id": simulation.simulation_id,
                "config_sha256": file_sha256(simulation.path / "config.yaml"),
            },
        )
    except Exception as error:
        return SimulationResult(
            simulation,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )
    return SimulationResult(simulation, "ready" if dry_run else "complete", report=report)


def compute_collection(
    simulations: Sequence[CatalogueSimulation],
    diagnostic_name: str,
    workers: int = 1,
    dry_run: bool = False,
    overwrite: bool = False,
    on_result: Callable[[SimulationResult], None] | None = None,
    diagnostic_directories: tuple[Path | str, ...] = (),
    diagnostic_version: str | None = None,
    update: bool = False,
) -> tuple[SimulationResult, ...]:
    """Compute one diagnostic for every selected catalogue simulation.

    Args:
        simulations: Indexed catalogue simulations to analyse.
        diagnostic_name: Registered diagnostic name to compute.
        workers: Maximum number of simulations to analyse concurrently.
        dry_run: Whether to inspect each simulation without reading AMUSE
            particles or writing output files.
        overwrite: Whether completed selected diagnostics may be recomputed.
        on_result: Optional parent-process callback called once for each
            completed, skipped, ready, or failed simulation result.
        diagnostic_directories: Trusted local directories containing additional
            diagnostic modules.
        diagnostic_version: Optional exact diagnostic version. ``None`` uses
            the highest registered version for ``diagnostic_name``.
        update: Whether incomplete or outdated existing results should be
            replaced while current completed results remain untouched.

    Returns:
        Results in the same order as ``simulations``.

    Raises:
        ValueError: If ``workers`` is less than one.

    Notes:
        Parallelism is per simulation, never per snapshot. Each worker writes
        only its own simulation's derived file, while the parent process owns
        progress reporting through ``on_result``.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    try:
        diagnostic = time_series_diagnostic(
            diagnostic_name,
            diagnostic_directories,
            diagnostic_version,
        )
    except ValueError:
        diagnostic = None
    if not dry_run and diagnostic is not None:
        # Register the complete scientific contract before workers write data,
        # so every completed result can be validated against this collection.
        collection_roots = {
            simulation.path.parent.parent
            for simulation in simulations
            if simulation.importer == "dcaf"
        }
        available_diagnostics = load_diagnostics(diagnostic_directories)
        for collection_root in collection_roots:
            ensure_collection_diagnostics(
                collection_root,
                (diagnostic,),
                available_diagnostics,
            )
    if workers == 1:
        results = []
        for simulation in simulations:
            result = compute_catalogue_simulation(
                simulation,
                diagnostic_name,
                dry_run,
                overwrite,
                diagnostic_directories,
                diagnostic_version,
                update,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
        result_tuple = tuple(results)
    else:
        results_by_simulation: dict[CatalogueSimulation, SimulationResult] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    compute_catalogue_simulation,
                    simulation,
                    diagnostic_name,
                    dry_run,
                    overwrite,
                    diagnostic_directories,
                    diagnostic_version,
                    update,
                ): simulation
                for simulation in simulations
            }
            for future in as_completed(futures):
                simulation = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = SimulationResult(
                        simulation,
                        "failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                results_by_simulation[simulation] = result
                if on_result is not None:
                    on_result(result)
        result_tuple = tuple(results_by_simulation[simulation] for simulation in simulations)

    return result_tuple


def compute_scalar_catalogue_simulation(
    simulation: CatalogueSimulation,
    diagnostic_name: str,
    overwrite: bool = False,
    diagnostic_directories: tuple[Path | str, ...] = (),
    diagnostic_version: str | None = None,
    update: bool = False,
) -> SimulationResult:
    """Compute one registered scalar diagnostic for one catalogue simulation.

    Args:
        simulation: Indexed simulation selected by ``csfdata``.
        diagnostic_name: Registered scalar diagnostic name.
        overwrite: Whether existing choice-specific results may be replaced.
        diagnostic_directories: Trusted local directories containing additional
            diagnostic modules. Each worker loads these directories itself.
        diagnostic_version: Optional exact diagnostic version. ``None`` uses
            the highest registered version for ``diagnostic_name``.
        update: Whether incomplete existing scalar results should be replaced
            while current completed results remain untouched.

    Returns:
        A result describing completed, skipped, or failed work.
    """
    try:
        diagnostic = scalar_diagnostic(
            diagnostic_name,
            diagnostic_directories,
            diagnostic_version,
        )
        registry = load_diagnostics(diagnostic_directories)
        replace_existing = overwrite
        if update and not overwrite:
            try:
                simulation.diagnostics.scalar[(diagnostic.name, f"v{diagnostic.version}")]
            except (KeyError, OSError, ValueError):
                replace_existing = True
            else:
                return SimulationResult(simulation, "skipped")
        report = compute_scalar_diagnostic(
            simulation,
            diagnostic,
            replace_existing,
            registry,
        )
    except Exception as error:
        return SimulationResult(
            simulation,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )
    return SimulationResult(
        simulation,
        "complete" if report.written_count else "skipped",
        report=report,
    )


def compute_scalar_collection(
    simulations: Sequence[CatalogueSimulation],
    diagnostic_name: str,
    workers: int = 1,
    overwrite: bool = False,
    on_result: Callable[[SimulationResult], None] | None = None,
    diagnostic_directories: tuple[Path | str, ...] = (),
    diagnostic_version: str | None = None,
    update: bool = False,
) -> tuple[SimulationResult, ...]:
    """Compute one scalar diagnostic for every selected catalogue simulation.

    Args:
        simulations: Indexed catalogue simulations to analyse.
        diagnostic_name: Registered scalar diagnostic name to compute.
        workers: Maximum number of simulations to analyse concurrently.
        overwrite: Whether existing choice-specific results may be recomputed.
        on_result: Optional parent-process callback called once for each
            completed, skipped, or failed simulation result.
        diagnostic_directories: Trusted local directories containing additional
            diagnostic modules.
        diagnostic_version: Optional exact diagnostic version. ``None`` uses
            the highest registered version for ``diagnostic_name``.
        update: Whether incomplete existing scalar results should be replaced
            while current completed results remain untouched.

    Returns:
        Results in the same order as ``simulations``.

    Raises:
        ValueError: If ``workers`` is less than one or the diagnostic name is
            not registered as a scalar diagnostic.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    diagnostic = scalar_diagnostic(
        diagnostic_name,
        diagnostic_directories,
        diagnostic_version,
    )
    registry = load_diagnostics(diagnostic_directories)

    # Register once in the parent before independent workers validate and
    # write their own simulation-level scalar result files.
    collection_roots = {simulation.path.parent.parent for simulation in simulations}
    for collection_root in collection_roots:
        ensure_collection_diagnostics(collection_root, (diagnostic,), registry)

    if workers == 1:
        results = []
        for simulation in simulations:
            result = compute_scalar_catalogue_simulation(
                simulation,
                diagnostic_name,
                overwrite,
                diagnostic_directories,
                diagnostic_version,
                update,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
        return tuple(results)

    results_by_simulation: dict[CatalogueSimulation, SimulationResult] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                compute_scalar_catalogue_simulation,
                simulation,
                diagnostic_name,
                overwrite,
                diagnostic_directories,
                diagnostic_version,
                update,
            ): simulation
            for simulation in simulations
        }
        for future in as_completed(futures):
            simulation = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = SimulationResult(
                    simulation,
                    "failed",
                    error=f"{type(error).__name__}: {error}",
                )
            results_by_simulation[simulation] = result
            if on_result is not None:
                on_result(result)
    return tuple(results_by_simulation[simulation] for simulation in simulations)


def clear_time_series(
    lite_catalogue: Path,
    simulations: Sequence[CatalogueSimulation],
    diagnostic_name: str,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, ...]:
    """Remove one selected time-series diagnostic from a lite catalogue.

    Args:
        lite_catalogue: Root directory of the lite catalogue that owns the
            selected derived data.
        simulations: Indexed simulations whose selected diagnostic is removed.
        diagnostic_name: Registered time-series diagnostic to remove.
        progress: Optional callback invoked after each selected simulation is
            inspected. It receives the completed count, total count, and
            ``collection_id/simulation_id`` label.

    Returns:
        Tuple[Path, ...]: Exact ``series.h5`` files removed from the lite
            catalogue. Simulations without that file are omitted.

    Raises:
        ValueError: If the root is not a lite catalogue, a diagnostic is
            unknown, or a selected simulation does not belong to the root.

    Notes:
        This intentionally removes only the exact versioned HDF5 result file.
        Empty parent directories are preserved because they are harmless and
        make the diagnostic's standard location visible during inspection.
    """
    lite_root = lite_catalogue.resolve()
    if not is_lite_catalogue(lite_root):
        raise ValueError(f"Derived data may be cleared only from a lite catalogue: {lite_root}")
    diagnostic = TIME_SERIES_DIAGNOSTICS.get(diagnostic_name)
    if diagnostic is None:
        raise ValueError(f"Unknown time-series diagnostic: {diagnostic_name}")

    removed_paths = []
    for simulation_number, simulation in enumerate(simulations, start=1):
        # Selection comes from an index, so verify its path before any removal.
        try:
            simulation.path.resolve().relative_to(lite_root)
        except ValueError as error:
            raise ValueError(
                f"Selected simulation is outside the lite catalogue: {simulation.path}"
            ) from error
        path = time_series_path(simulation.path, diagnostic)
        if path.is_file():
            path.unlink()
            removed_paths.append(path)
        if progress is not None:
            progress(
                simulation_number,
                len(simulations),
                f"{simulation.collection_id}/{simulation.simulation_id}",
            )
    return tuple(removed_paths)
