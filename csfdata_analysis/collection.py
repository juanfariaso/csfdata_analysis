"""Parallel execution of standardized diagnostics over catalogue selections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from csfdata.adapters.dcaf import DcafAdapter
from csfdata.catalogue import CatalogueSimulation
from csfdata_analysis.diagnostics import DIAGNOSTICS
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
) -> SimulationResult:
    """Compute one registered diagnostic for one catalogue simulation.

    Args:
        simulation: Indexed simulation selected by ``csfdata``.
        diagnostic_name: Registered diagnostic name, such as
            ``"lagrangian_radii"``.
        dry_run: Whether to inspect the snapshot plan without writing output.

    Returns:
        A result describing completed, ready, skipped, or failed work.

    Notes:
        This function is intentionally module-level so it can run in a process
        worker. It currently supports catalogue simulations imported with the
        D-CAF adapter.
    """
    diagnostic = DIAGNOSTICS.get(diagnostic_name)
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
    if time_series_path(simulation.path, diagnostic).exists():
        return SimulationResult(simulation, "skipped")
    try:
        report = compute_time_series(
            simulation.path,
            diagnostic,
            DcafAdapter(simulation.path / "raw"),
            read_stars,
            dry_run=dry_run,
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
    on_result: Callable[[SimulationResult], None] | None = None,
) -> tuple[SimulationResult, ...]:
    """Compute one diagnostic for every selected catalogue simulation.

    Args:
        simulations: Indexed catalogue simulations to analyse.
        diagnostic_name: Registered diagnostic name to compute.
        workers: Maximum number of simulations to analyse concurrently.
        dry_run: Whether to inspect each simulation without reading AMUSE
            particles or writing output files.
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
            result = compute_catalogue_simulation(simulation, diagnostic_name, dry_run)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return tuple(results)

    results_by_simulation: dict[CatalogueSimulation, SimulationResult] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                compute_catalogue_simulation,
                simulation,
                diagnostic_name,
                dry_run,
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
