"""Command-line interface for CSFData analysis diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from csfdata.adapters.dcaf import DcafAdapter
from csfdata.catalogue import find_simulations
from csfdata_analysis.checks import test_diagnostics
from csfdata_analysis.collection import compute_collection
from csfdata_analysis.diagnostics import DIAGNOSTICS
from csfdata_analysis.readers import read_stars
from csfdata_analysis.runner import compute_time_series


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CSFData analysis command-line interface.

    Args:
        arguments: Command arguments excluding the program name. ``None`` uses
            arguments supplied by the shell.

    Returns:
        int: Zero when the requested command completes successfully.
    """
    parser = argparse.ArgumentParser(prog="csfdata-analysis")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("diagnostics", help="List available diagnostic defaults.")
    subcommands.add_parser(
        "test-diagnostics",
        help="Run the built-in readiness check for every registered diagnostic.",
    )
    compute_parser = subcommands.add_parser(
        "compute",
        help="Compute one diagnostic for indexed catalogue simulations.",
    )
    compute_parser.add_argument("diagnostic", choices=sorted(DIAGNOSTICS))
    compute_parser.add_argument(
        "--catalogue",
        type=Path,
        required=True,
        help="Indexed catalogue root.",
    )
    compute_parser.add_argument(
        "--filter",
        action="append",
        required=True,
        help="Selection filter such as collection=dcaf-grid-v1 or tff=0.5:3.0.",
    )
    compute_parser.add_argument("--workers", type=int, default=1)
    compute_parser.add_argument("--dry-run", action="store_true")
    simulation_parser = subcommands.add_parser(
        "compute-simulation",
        help="Compute one diagnostic for one imported simulation directory.",
    )
    simulation_parser.add_argument("simulation_root", type=Path)
    simulation_parser.add_argument("diagnostic", choices=sorted(DIAGNOSTICS))
    simulation_parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(arguments)

    if options.command == "diagnostics":
        for name, diagnostic in DIAGNOSTICS.items():
            print(f"{name} v{diagnostic.version}")
        return 0

    if options.command == "test-diagnostics":
        checks = test_diagnostics()
        for check in checks:
            if check.passed:
                print(f"OK: {check.name} v{check.version}")
            else:
                print(f"FAILED: {check.name} v{check.version}: {check.error}")
        print(f"Passed: {sum(check.passed for check in checks)}/{len(checks)}")
        return 1 if any(not check.passed for check in checks) else 0

    if options.command == "compute":
        try:
            collection_id, filters = parse_filters(options.filter)
            simulations = find_simulations(
                options.catalogue,
                collection_id=collection_id,
                filters=filters,
            )
            if not simulations:
                print("No simulations matched the supplied filters.")
                return 0
            completed = 0

            def show_result(result) -> None:
                """Print one parent-owned collection-analysis progress result."""
                nonlocal completed
                completed += 1
                label = f"{result.simulation.collection_id}/{result.simulation.simulation_id}"
                if result.error is None:
                    print(f"[{completed:>4}/{len(simulations)}] {result.status}: {label}")
                else:
                    print(f"[{completed:>4}/{len(simulations)}] failed: {label}: {result.error}")

            results = compute_collection(
                simulations,
                options.diagnostic,
                workers=options.workers,
                dry_run=options.dry_run,
                on_result=show_result,
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            parser.error(str(error))
        counts = {status: sum(result.status == status for result in results) for status in (
            "complete", "ready", "skipped", "failed"
        )}
        print(f"Selected: {len(results)}")
        print(f"Complete: {counts['complete']}")
        print(f"Ready: {counts['ready']}")
        print(f"Skipped: {counts['skipped']}")
        print(f"Failed: {counts['failed']}")
        return 1 if counts["failed"] else 0

    simulation_root = options.simulation_root.resolve()
    adapter = DcafAdapter(simulation_root / "raw")
    if not adapter.is_simulation():
        parser.error(f"{simulation_root / 'raw'} is not a D-CAF simulation directory.")
    diagnostic = DIAGNOSTICS[options.diagnostic]
    report = compute_time_series(
        simulation_root, diagnostic, adapter, read_stars, dry_run=options.dry_run
    )
    print(f"Diagnostic: {diagnostic.name} v{diagnostic.version}")
    print(f"Simulation: {report.simulation_root}")
    print(f"Snapshots: {report.snapshot_count}")
    print(f"Time range: {report.first_time_myr:g} to {report.last_time_myr:g} Myr")
    print(f"Output: {report.output_path}")
    if report.dry_run:
        print("Dry run: no particle data was read and no output was created.")
    return 0


def parse_filters(
    values: Sequence[str],
) -> tuple[str | None, dict[str, str | float | bool | tuple[float | None, float | None]]]:
    """Parse command-line catalogue filters into registry query values.

    Args:
        values: Repeated ``NAME=VALUE`` or ``NAME=LOWER:UPPER`` strings.

    Returns:
        The optional ``collection`` filter and all remaining parameter filters.

    Raises:
        ValueError: If a filter is malformed, repeated, or has an invalid range.

    Notes:
        Numeric values become floats. ``true`` and ``false`` become booleans.
        A blank range bound, such as ``tff=:3.0``, means unbounded.
    """
    collection_id = None
    filters: dict[str, str | float | bool | tuple[float | None, float | None]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Filter must use NAME=VALUE syntax: {value}")
        name, raw_value = value.split("=", maxsplit=1)
        if not name or not raw_value:
            raise ValueError(f"Filter must have a name and value: {value}")
        if name == "collection":
            if collection_id is not None:
                raise ValueError("Collection filter may be specified only once.")
            collection_id = raw_value
            continue
        if name in filters:
            raise ValueError(f"Filter may be specified only once: {name}")
        if ":" in raw_value:
            lower_text, upper_text = raw_value.split(":", maxsplit=1)
            try:
                lower = float(lower_text) if lower_text else None
                upper = float(upper_text) if upper_text else None
            except ValueError as error:
                raise ValueError(f"Range filter has invalid bounds: {value}") from error
            filters[name] = (lower, upper)
        elif raw_value.lower() in {"true", "false"}:
            filters[name] = raw_value.lower() == "true"
        else:
            try:
                filters[name] = float(raw_value)
            except ValueError:
                filters[name] = raw_value
    return collection_id, filters


if __name__ == "__main__":
    raise SystemExit(main())
