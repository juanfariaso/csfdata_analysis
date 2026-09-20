"""Safely import completed derived products from a lite catalogue."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from uuid import uuid4

import h5py

from csfdata.catalogue import file_sha256, read_lite_source
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
    lite_catalogue: Path,
    destination_catalogue: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> DerivedImportReport:
    """Import completed HDF5 diagnostics from one lite catalogue.

    Args:
        lite_catalogue: Lite collection created by ``csfdata import-lite``.
        destination_catalogue: Full catalogue that owns the matching collection.
        overwrite: Whether completed destination results may be replaced.
        dry_run: Whether to report actions without copying files.

    Returns:
        Copied and skipped paths plus every result that could not be imported.

    Raises:
        ValueError: If the destination is not the source catalogue recorded by
            the lite import or its collection definition has changed.
        FileNotFoundError: If required lite or destination collection files are absent.

    Notes:
        Only final ``series.h5`` files carrying the required complete identity
        attributes are eligible. Existing results are skipped by default;
        ``overwrite=True`` is required to replace them.
    """
    lite_catalogue = lite_catalogue.resolve()
    destination_catalogue = destination_catalogue.resolve()
    source = read_lite_source(lite_catalogue)
    if destination_catalogue != source.catalogue_root:
        raise ValueError(
            "Destination catalogue does not match the source recorded by the lite import."
        )
    lite_collection = lite_catalogue / "collections" / source.collection_id
    destination_collection = destination_catalogue / "collections" / source.collection_id
    if file_sha256(destination_collection / "collection.yaml") != source.collection_sha256:
        raise ValueError("Destination collection configuration differs from the lite import.")

    copied_paths: list[Path] = []
    skipped_paths: list[Path] = []
    issues: list[str] = []
    lite_simulations = lite_collection / "simulations"
    for lite_simulation in sorted(path for path in lite_simulations.iterdir() if path.is_dir()):
        destination_simulation = destination_collection / "simulations" / lite_simulation.name
        if not destination_simulation.is_dir():
            issues.append(f"Missing destination simulation: {source.collection_id}/{lite_simulation.name}")
            continue
        if read_simulation_metadata(lite_simulation / "metadata.yaml") != read_simulation_metadata(
            destination_simulation / "metadata.yaml"
        ):
            issues.append(f"Simulation metadata differs: {source.collection_id}/{lite_simulation.name}")
            continue
        configuration_sha256 = file_sha256(lite_simulation / "config.yaml")
        if configuration_sha256 != file_sha256(destination_simulation / "config.yaml"):
            issues.append(f"Simulation configuration differs: {source.collection_id}/{lite_simulation.name}")
            continue
        result_paths = tuple(lite_simulation.glob("derived/diagnostics/*/v*/series.h5"))
        if not result_paths:
            issues.append(f"No completed derived result: {source.collection_id}/{lite_simulation.name}")
        for result_path in result_paths:
            error = validate_derived_result(
                result_path,
                source.collection_id,
                lite_simulation.name,
                configuration_sha256,
            )
            if error is not None:
                issues.append(error)
                continue
            relative_path = result_path.relative_to(lite_simulation)
            destination_path = destination_simulation / relative_path
            if destination_path.exists() and not overwrite:
                skipped_paths.append(destination_path)
                continue
            if not dry_run:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                staging_path = destination_path.parent / f".{destination_path.name}.{uuid4().hex}.tmp"
                shutil.copy2(result_path, staging_path)
                staging_path.replace(destination_path)
            copied_paths.append(destination_path)
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
