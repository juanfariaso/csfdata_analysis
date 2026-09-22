"""Safely import completed time-series products between catalogues."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil
from uuid import uuid4

import h5py

from csfdata.catalogue import file_sha256, is_lite_catalogue, read_lite_source
from csfdata.catalogue.metadata import read_simulation_metadata


@dataclass(frozen=True)
class DerivedImportReport:
    """Summary of one derived-data import attempt.

    Args:
        copied_paths: Destination files copied into the full catalogue.
        skipped_paths: Existing destination files left unchanged.
        issues: Human-readable missing, incomplete, or incompatible results.
    """

    copied_paths: tuple[Path, ...]
    skipped_paths: tuple[Path, ...]
    issues: tuple[str, ...]


def import_derived(
    source_catalogue: Path,
    destination_catalogue: Path,
    collection_id: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> DerivedImportReport:
    """Import completed HDF5 diagnostics between compatible catalogues.

    Args:
        source_catalogue: Full or lite catalogue containing derived results.
        destination_catalogue: Catalogue that receives matching results.
        collection_id: Optional one collection ID to import. ``None`` imports
            every collection present in ``source_catalogue``.
        overwrite: Whether completed destination results may be replaced.
        dry_run: Whether to report actions without copying files.
        progress: Optional callback invoked after each source simulation is
            inspected. It receives the completed count, total count, and
            ``collection_id/simulation_id`` label.

    Returns:
        Copied and skipped paths plus every result that could not be imported.

    Raises:
        ValueError: If source and destination are incompatible, the requested
            collection is absent, or a lite source does not target its recorded
            full catalogue.
        FileNotFoundError: If required source or destination collection files
            are absent.

    Notes:
        The first catalogue argument is always the source and the second is
        always the destination. A lite source remains restricted to the full
        catalogue recorded in its provenance file. Only final ``series.h5``
        files carrying the required complete identity attributes are eligible.
        Existing results are skipped by default; ``overwrite=True`` is required
        to replace them.
    """
    source_catalogue = source_catalogue.resolve()
    destination_catalogue = destination_catalogue.resolve()
    if source_catalogue == destination_catalogue:
        raise ValueError("Source and destination catalogues must be different.")
    source_collections_root = source_catalogue / "collections"
    destination_collections_root = destination_catalogue / "collections"
    if not source_collections_root.is_dir():
        raise FileNotFoundError(f"Source catalogue has no collections directory: {source_collections_root}")
    if not destination_collections_root.is_dir():
        raise FileNotFoundError(
            f"Destination catalogue has no collections directory: {destination_collections_root}"
        )

    if is_lite_catalogue(source_catalogue):
        lite_source = read_lite_source(source_catalogue)
        if destination_catalogue != lite_source.catalogue_root:
            raise ValueError(
                "Destination catalogue does not match the source recorded by the lite import."
            )
        if collection_id is not None and collection_id != lite_source.collection_id:
            raise ValueError(
                "Requested collection does not match the collection recorded by the lite import."
            )
        collection_ids = (lite_source.collection_id,)
    elif collection_id is not None:
        collection_ids = (collection_id,)
    else:
        collection_ids = tuple(
            sorted(path.name for path in source_collections_root.iterdir() if path.is_dir())
        )

    copied_paths: list[Path] = []
    skipped_paths: list[Path] = []
    issues: list[str] = []
    collections = []
    for current_collection_id in collection_ids:
        source_collection = source_collections_root / current_collection_id
        destination_collection = destination_collections_root / current_collection_id
        source_configuration = source_collection / "collection.yaml"
        destination_configuration = destination_collection / "collection.yaml"
        if not source_configuration.is_file():
            raise FileNotFoundError(f"Source collection configuration is missing: {source_configuration}")
        if not destination_configuration.is_file():
            raise FileNotFoundError(
                f"Destination collection configuration is missing: {destination_configuration}"
            )
        if file_sha256(source_configuration) != file_sha256(destination_configuration):
            raise ValueError(
                f"Collection configuration differs: {current_collection_id}."
            )
        source_simulations = source_collection / "simulations"
        if not source_simulations.is_dir():
            raise FileNotFoundError(f"Source collection has no simulations directory: {source_simulations}")
        collections.append(
            (
                current_collection_id,
                destination_collection,
                tuple(sorted(path for path in source_simulations.iterdir() if path.is_dir())),
            )
        )

    total = sum(len(simulations) for _, _, simulations in collections)
    completed = 0
    for current_collection_id, destination_collection, simulations in collections:
        for source_simulation in simulations:
            completed += 1
            label = f"{current_collection_id}/{source_simulation.name}"
            destination_simulation = (
                destination_collection / "simulations" / source_simulation.name
            )
            if not destination_simulation.is_dir():
                issues.append(f"Missing destination simulation: {label}")
            elif read_simulation_metadata(source_simulation / "metadata.yaml") != read_simulation_metadata(
                destination_simulation / "metadata.yaml"
            ):
                issues.append(f"Simulation metadata differs: {label}")
            else:
                configuration_sha256 = file_sha256(source_simulation / "config.yaml")
                if configuration_sha256 != file_sha256(destination_simulation / "config.yaml"):
                    issues.append(f"Simulation configuration differs: {label}")
                else:
                    result_paths = tuple(
                        source_simulation.glob("derived/diagnostics/*/v*/series.h5")
                    )
                    if not result_paths:
                        issues.append(f"No completed derived result: {label}")
                    for result_path in result_paths:
                        error = validate_derived_result(
                            result_path,
                            current_collection_id,
                            source_simulation.name,
                            configuration_sha256,
                        )
                        if error is not None:
                            issues.append(error)
                            continue
                        relative_path = result_path.relative_to(source_simulation)
                        destination_path = destination_simulation / relative_path
                        if destination_path.exists() and not overwrite:
                            skipped_paths.append(destination_path)
                            continue
                        if not dry_run:
                            destination_path.parent.mkdir(parents=True, exist_ok=True)
                            staging_path = (
                                destination_path.parent
                                / f".{destination_path.name}.{uuid4().hex}.tmp"
                            )
                            shutil.copy2(result_path, staging_path)
                            staging_path.replace(destination_path)
                        copied_paths.append(destination_path)
            if progress is not None:
                progress(completed, total, label)
    return DerivedImportReport(tuple(copied_paths), tuple(skipped_paths), tuple(issues))


def validate_derived_result(
    result_path: Path,
    collection_id: str,
    simulation_id: str,
    config_sha256: str,
) -> str | None:
    """Validate the completion and identity attributes of one HDF5 result.

    Args:
        result_path: Final ``series.h5`` file from a lite catalogue.
        collection_id: Expected parent collection ID.
        simulation_id: Expected parent simulation ID.
        config_sha256: Expected hash of the copied canonical configuration.

    Returns:
        ``None`` when the result is safe to import, otherwise an issue string.
    """
    try:
        with h5py.File(result_path, "r") as output_file:
            attributes = output_file.attrs
            if attributes.get("complete") != True:
                return f"Incomplete derived result: {result_path}"
            expected = {
                "collection_id": collection_id,
                "simulation_id": simulation_id,
                "config_sha256": config_sha256,
            }
            for name, value in expected.items():
                if attributes.get(name) != value:
                    return f"Derived result identity differs: {result_path} ({name})"
            diagnostic_name = attributes.get("diagnostic_name")
            diagnostic_version = attributes.get("diagnostic_version")
            if not isinstance(diagnostic_name, str):
                return f"Derived result diagnostic metadata is invalid: {result_path}"
            try:
                parsed_version = int(diagnostic_version)
            except (TypeError, ValueError):
                return f"Derived result diagnostic metadata is invalid: {result_path}"
            if (
                result_path.parent.parent.name != diagnostic_name
                or result_path.parent.name != f"v{parsed_version}"
            ):
                return f"Derived result path differs from diagnostic metadata: {result_path}"
    except (OSError, ValueError):
        return f"Unreadable derived result: {result_path}"
    return None
