"""Create a portable snapshot-slice manifest and rsync file list."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from csfdata.catalogue import is_lite_catalogue, read_lite_source
from csfdata_analysis.cli import parse_filters
from csfdata_analysis.data.loader import load_simulations, source_simulation
from csfdata_analysis.data.slices import select_snapshot_slice


def create_slice_manifest(
    catalogue: Path | str,
    collection_id: str,
    output_directory: Path | str,
    time: float,
    normalization: str | None = None,
    filters: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Write one snapshot-slice manifest and its exact rsync file list.

    Args:
        catalogue: Full or lite catalogue used to select simulations.
        collection_id: Single collection from which snapshots are selected.
        output_directory: New directory for ``selection.yaml`` and
            ``rsync-files.txt``.
        time: Physical target time in Myr, or a dimensionless multiplier when
            ``normalization`` is supplied.
        normalization: Optional Myr-valued canonical parameter, such as
            ``"tff"``.
        filters: Repeated ``NAME=VALUE`` or ``NAME=LOWER:UPPER`` parameter
            filters, excluding the collection ID.

    Returns:
        Paths to the written YAML manifest and rsync file list.

    Raises:
        FileExistsError: If the output directory or either output file exists.
        ValueError: If a filter is invalid, no simulations match, or a selected
            snapshot cannot be represented relative to the source catalogue.

    Notes:
        The output file list contains only source-relative catalogue files. Run
        ``rsync`` separately to copy these files into a local filtered
        catalogue while preserving the normal catalogue directory layout.
    """
    catalogue_root = Path(catalogue).resolve()
    output_root = Path(output_directory).resolve()
    if output_root.exists():
        raise FileExistsError(f"Manifest output directory already exists: {output_root}")
    _, parsed_filters = parse_filters(filters)
    simulations = load_simulations(catalogue_root, collection_id, parsed_filters)
    if not simulations:
        raise ValueError("No simulations matched the requested collection and filters.")
    slices = select_snapshot_slice(simulations, time, normalization)
    source_catalogue = (
        read_lite_source(catalogue_root).catalogue_root
        if is_lite_catalogue(catalogue_root)
        else catalogue_root
    )
    files = {Path("collections") / collection_id / "collection.yaml"}
    entries: list[dict[str, str | float]] = []
    for simulation, (_, slice_row) in zip(simulations, slices.iterrows(), strict=True):
        source_root = source_simulation(simulation)
        try:
            simulation_relative_path = source_root.relative_to(source_catalogue)
            snapshot_relative_path = Path(str(slice_row["snapshot_path"])).relative_to(
                source_catalogue
            )
        except ValueError as error:
            raise ValueError(
                f"Selected snapshot is outside the source catalogue: {slice_row['snapshot_path']}"
            ) from error
        files.update(
            {
                simulation_relative_path / "metadata.yaml",
                simulation_relative_path / "config.yaml",
                snapshot_relative_path,
            }
        )
        entries.append(
            {
                "simulation_id": simulation.simulation_id,
                "target_time": float(slice_row["target_time"]),
                "snapshot_time": float(slice_row["snapshot_time"]),
                "time_offset": float(slice_row["time_offset"]),
                "snapshot_relative_path": str(snapshot_relative_path),
            }
        )

    output_root.mkdir(parents=True)
    manifest_path = output_root / "selection.yaml"
    file_list_path = output_root / "rsync-files.txt"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source": {
                    "catalogue_root": str(source_catalogue),
                    "selection_catalogue": str(catalogue_root),
                    "collection_id": collection_id,
                },
                "slice": {
                    "time": time,
                    "normalization": normalization,
                    "method": "nearest",
                },
                "filters": list(filters),
                "simulations": entries,
            },
            allow_unicode=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    file_list_path.write_text(
        "".join(f"{path}\n" for path in sorted(files)),
        encoding="utf-8",
    )
    return manifest_path, file_list_path


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the snapshot-slice manifest script.

    Args:
        arguments: Script arguments without the executable name. ``None``
            reads arguments supplied by the shell.

    Returns:
        Zero after writing the manifest and rsync file list.
    """
    parser = argparse.ArgumentParser(
        description="Create a snapshot-slice manifest and rsync file list."
    )
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--normalization")
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("output", type=Path)
    options = parser.parse_args(arguments)
    manifest_path, file_list_path = create_slice_manifest(
        options.catalogue,
        options.collection,
        options.output,
        options.time,
        options.normalization,
        options.filter,
    )
    print(f"Manifest: {manifest_path}")
    print(f"Rsync file list: {file_list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
