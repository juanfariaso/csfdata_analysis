"""Standardized diagnostic definitions and implementations."""

from pathlib import Path

from csfdata.catalogue.diagnostics import (
    ChoiceDefinition,
    CollectionDiagnostics,
    collection_diagnostics_path,
    read_collection_diagnostics,
    write_collection_diagnostics,
)
from csfdata_analysis.choices import CHOICES
from csfdata_analysis.diagnostics.base import (
    EvaluationChoice,
    ScalarDiagnostic,
    TimeSeriesDiagnostic,
)
from csfdata_analysis.diagnostics.time_series.lagrangian_radii import LAGRANGIAN_RADII_V1

TIME_SERIES_DIAGNOSTICS = {"lagrangian_radii": LAGRANGIAN_RADII_V1}
"""Time-series diagnostics selected by name through the command-line interface."""


def publish_collection_diagnostics(
    collection_root: Path,
    diagnostics: tuple[TimeSeriesDiagnostic | ScalarDiagnostic, ...],
) -> Path:
    """Publish analysis definitions used by a collection to core CSFData.

    Args:
        collection_root: Destination collection directory below ``collections/``.
        diagnostics: Diagnostics with completed or already-present results.

    Returns:
        The updated collection ``diagnostics.yaml`` path.

    Raises:
        ValueError: If a diagnostic requests a choice that the analysis package
            has not registered.
        OSError: If the published definitions cannot be read or written.

    Notes:
        Existing unrelated definitions remain available. A newly supplied
        definition replaces only the matching diagnostic name and version.
    """
    path = collection_diagnostics_path(collection_root)
    existing = read_collection_diagnostics(path) if path.exists() else CollectionDiagnostics((), ())

    # Start with existing declarations so publishing one new diagnostic never
    # removes valid results produced by an earlier analysis command.
    definitions_by_key = {
        (definition.name, definition.version): definition
        for definition in existing.diagnostics
    }
    for diagnostic in diagnostics:
        definition = diagnostic.catalogue_definition()
        definitions_by_key[(definition.name, definition.version)] = definition

    # Keep earlier choice declarations, then refresh every choice used by the
    # supplied diagnostics from the analysis package's human-maintained source.
    choices_by_name = {choice.name: choice for choice in existing.choices}
    for diagnostic in diagnostics:
        for choice_name in diagnostic.choice_names:
            choice = CHOICES.get(choice_name)
            if choice is None:
                raise ValueError(f"Diagnostic uses an unregistered choice: {choice_name}.")
            choices_by_name[choice_name] = ChoiceDefinition(
                choice.name,
                choice.description,
                choice.values,
                choice.default,
            )

    # Core validation confirms every retained and newly declared choice exists.
    published = CollectionDiagnostics(
        tuple(choices_by_name.values()),
        tuple(definitions_by_key.values()),
    )
    write_collection_diagnostics(published, path)
    return path


__all__ = [
    "EvaluationChoice",
    "ScalarDiagnostic",
    "TIME_SERIES_DIAGNOSTICS",
    "TimeSeriesDiagnostic",
    "publish_collection_diagnostics",
]
