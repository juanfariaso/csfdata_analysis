"""Command-line interface for CSFData analysis diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from csfdata.adapters.dcaf import DcafAdapter
from csfdata.catalogue import find_simulations
from csfdata_analysis.catalogue_runner import clear_time_series, compute_collection
from csfdata_analysis.derived import import_derived
from csfdata_analysis.diagnostics import TIME_SERIES_DIAGNOSTICS
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
    compute_parser = subcommands.add_parser(
        "compute",
        help="Compute one diagnostic for indexed catalogue simulations.",
    )
    compute_parser.add_argument("diagnostic", choices=sorted(TIME_SERIES_DIAGNOSTICS))
    compute_parser.add_argument(
        "--catalogue",
        type=Path,
        required=True,
        help="Indexed catalogue root.",
    )
    compute_parser.add_argument(
        "--filter",
        action="append",
        help="Selection filter such as collection=dcaf-grid-v1 or tff=0.5:3.0.",
    )
    compute_parser.add_argument("--workers", type=int, default=1)
    compute_parser.add_argument("--dry-run", action="store_true")
    compute_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace completed selected diagnostic files after success.",
    )
    compute_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Start immediately without showing the selection and asking for confirmation.",
    )
    simulation_parser = subcommands.add_parser(
        "compute-simulation",
        help="Compute one diagnostic for one imported simulation directory.",
    )
    simulation_parser.add_argument("simulation_root", type=Path)
    simulation_parser.add_argument("diagnostic", choices=sorted(TIME_SERIES_DIAGNOSTICS))
    simulation_parser.add_argument("--dry-run", action="store_true")
    simulation_parser.add_argument("--overwrite", action="store_true")
    clear_parser = subcommands.add_parser(
        "clear-derived",
        help="Remove one selected time-series diagnostic from a lite catalogue.",
    )
    clear_parser.add_argument("diagnostic", choices=sorted(TIME_SERIES_DIAGNOSTICS))
    clear_parser.add_argument("--catalogue", type=Path, required=True)
    clear_parser.add_argument(
        "--filter",
        action="append",
        help="Selection filter such as collection=dcaf-grid-v1 or tff=0.5:3.0.",
    )
    clear_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Remove immediately without showing the selection and asking for confirmation.",
    )
    import_parser = subcommands.add_parser(
        "import-derived",
        help="Safely import completed derived results from one lite catalogue.",
    )
    import_parser.add_argument("lite_catalogue", type=Path)
    import_parser.add_argument("--catalogue", type=Path, required=True)
    import_parser.add_argument("--overwrite", action="store_true")
    import_parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(arguments)

    if options.command == "diagnostics":
        for name, diagnostic in TIME_SERIES_DIAGNOSTICS.items():
            print(f"{name} v{diagnostic.version}")
        return 0

    if options.command == "compute":
        try:
            collection_id, filters = parse_filters(options.filter or ())
            simulations = find_simulations(
                options.catalogue,
                collection_id=collection_id,
                filters=filters,
            )
            if not simulations:
                print("No simulations matched the supplied filters.")
                return 0
            counts_by_collection: dict[str, int] = {}
            for simulation in simulations:
                counts_by_collection[simulation.collection_id] = (
                    counts_by_collection.get(simulation.collection_id, 0) + 1
                )
            print("Selected collections:")
            for selected_collection, count in counts_by_collection.items():
                print(f"  {selected_collection}: {count} simulations")
            print(f"Total simulations: {len(simulations)}")
            if not options.no_prompt:
                try:
                    answer = input("Compute this diagnostic? [y/N] ").strip().lower()
                except EOFError as error:
                    raise ValueError("No confirmation input available. Re-run with --no-prompt.") from error
                if answer not in {"y", "yes"}:
                    print("Cancelled.")
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
                overwrite=options.overwrite,
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

    if options.command == "clear-derived":
        try:
            collection_id, filters = parse_filters(options.filter or ())
            simulations = find_simulations(
                options.catalogue,
                collection_id=collection_id,
                filters=filters,
            )
            if not simulations:
                print("No simulations matched the supplied filters.")
                return 0
            print(f"Diagnostic: {options.diagnostic}")
            print(f"Selected simulations: {len(simulations)}")
            if not options.no_prompt:
                try:
                    answer = input("Remove these derived results? [y/N] ").strip().lower()
                except EOFError as error:
                    raise ValueError("No confirmation input available. Re-run with --no-prompt.") from error
                if answer not in {"y", "yes"}:
                    print("Cancelled.")
                    return 0
            removed_paths = clear_time_series(
                options.catalogue,
                simulations,
                options.diagnostic,
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            clear_parser.error(str(error))
        print(f"Removed: {len(removed_paths)}")
        return 0

    if options.command == "import-derived":
        return import_derived_command(
            options.lite_catalogue,
            options.catalogue,
            overwrite=options.overwrite,
            dry_run=options.dry_run,
            parser=import_parser,
        )

    simulation_root = options.simulation_root.resolve()
    adapter = DcafAdapter(simulation_root / "raw")
    if not adapter.is_simulation():
        parser.error(f"{simulation_root / 'raw'} is not a D-CAF simulation directory.")
    diagnostic = TIME_SERIES_DIAGNOSTICS[options.diagnostic]
    report = compute_time_series(
        simulation_root,
        diagnostic,
        adapter,
        read_stars,
        dry_run=options.dry_run,
        overwrite=options.overwrite,
    )
    print(f"Time-series diagnostic: {diagnostic.name} v{diagnostic.version}")
    print(f"Simulation: {report.simulation_root}")
    print(f"Snapshots: {report.snapshot_count}")
    print(f"Time range: {report.first_time_myr:g} to {report.last_time_myr:g} Myr")
    print(f"Output: {report.output_path}")
    if report.dry_run:
        print("Dry run: no particle data was read and no output was created.")
    return 0


def import_derived_command(
    lite_catalogue: Path,
    destination_catalogue: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    """Import complete results from one lite catalogue into its full source.

    Args:
        lite_catalogue: Working lite catalogue containing derived HDF5 files.
        destination_catalogue: Recorded full source catalogue to receive results.
        overwrite: Whether existing destination files may be replaced.
        dry_run: Whether to report actions without copying results.
        parser: Optional CLI parser used to present validation errors.

    Returns:
        Zero after reporting copied, skipped, and unsafe results.
    """
    try:
        report = import_derived(
            lite_catalogue,
            destination_catalogue,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        if parser is None:
            raise
        parser.error(str(error))
    print(f"Copied: {len(report.copied_paths)}")
    print(f"Skipped existing: {len(report.skipped_paths)}")
    print(f"Issues: {len(report.issues)}")
    for issue in report.issues:
        print(f"  - {issue}")
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
