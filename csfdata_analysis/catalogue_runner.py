"""Parallel execution of standardized diagnostics over catalogue selections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from csfdata.adapters.dcaf import DcafAdapter
from csfdata.catalogue import CatalogueSimulation, file_sha256, is_lite_catalogue
from csfdata_analysis.data.loader import source_simulation
from csfdata_analysis.diagnostics import (
    TIME_SERIES_DIAGNOSTICS,
    publish_collection_diagnostics,
)
from csfdata_analysis.readers import read_stars
from csfdata_analysis.runner import SimulationReport, compute_time_series, time_series_path


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of analysing one selected catalogue simulation.

    Args:
        simulation: Indexed catalogue simulation selected for analysis.
        status: ``"complete"``, ``"ready"``, ``"skipped"``, or ``"failed"``.
        report: Per-simulation time-series report for complete or ready work.
        error: Human-readable failure reason, if the status is ``"failed"``.

    Notes:
        ``"ready"`` is used by a dry run. ``"skipped"`` means a completed
        time series already exists and was deliberately left unchanged.
    """

    simulation: CatalogueSimulation
    status: str
    report: SimulationReport | None = None
    error: str | None = None


def compute_catalogue_simulation(
    simulation: CatalogueSimulation,
    diagnostic_name: str,
    dry_run: bool = False,
    overwrite: bool = False,
) -> SimulationResult:
    """Compute one registered diagnostic for one catalogue simulation.

    Args:
        simulation: Indexed simulation selected by ``csfdata``.
        diagnostic_name: Registered diagnostic name, such as
            ``"lagrangian_radii"``.
        dry_run: Whether to inspect the snapshot plan without writing output.
        overwrite: Whether a completed selected diagnostic may be recomputed.

    Returns:
        A result describing completed, ready, skipped, or failed work.

    Notes:
        This function is intentionally module-level so it can run in a process
        worker. It currently supports catalogue simulations imported with the
        D-CAF adapter.
    """
    diagnostic = TIME_SERIES_DIAGNOSTICS.get(diagnostic_name)
    if diagnostic is None:
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
    if destination.exists() and not overwrite:
        return SimulationResult(simulation, "skipped")
    try:
        input_simulation = source_simulation(simulation)
        report = compute_time_series(
            input_simulation,
            diagnostic,
            DcafAdapter(input_simulation / "raw"),
            read_stars,
            dry_run=dry_run,
            overwrite=overwrite,
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
    if workers == 1:
        results = []
        for simulation in simulations:
            result = compute_catalogue_simulation(
                simulation, diagnostic_name, dry_run, overwrite
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

    diagnostic = TIME_SERIES_DIAGNOSTICS.get(diagnostic_name)
    if not dry_run and diagnostic is not None:
        # Publish only collections with a completed file or an already complete
        # file; a dry run or completely failed selection must not advertise data.
        collection_roots = {
            result.simulation.path.parent.parent
            for result in result_tuple
            if result.status in {"complete", "skipped"}
        }
        for collection_root in collection_roots:
            publish_collection_diagnostics(collection_root, (diagnostic,))
    return result_tuple


def clear_time_series(
    lite_catalogue: Path,
    simulations: Sequence[CatalogueSimulation],
    diagnostic_name: str,
) -> tuple[Path, ...]:
    """Remove one selected time-series diagnostic from a lite catalogue.

    Args:
        lite_catalogue: Root directory of the lite catalogue that owns the
            selected derived data.
        simulations: Indexed simulations whose selected diagnostic is removed.
        diagnostic_name: Registered time-series diagnostic to remove.

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
    for simulation in simulations:
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
    return tuple(removed_paths)
