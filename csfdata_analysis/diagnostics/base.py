"""Definitions for standardized time-series and scalar diagnostics.

Time-series diagnostics evaluate one AMUSE particle set per snapshot. Summary
diagnostics combine completed diagnostic products into scalar simulation values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from amuse.datamodel import Particles
from amuse.units.quantities import Quantity
from csfdata.catalogue.diagnostics import (
    DiagnosticDefinition,
    DiagnosticField,
    DiagnosticRequirement,
)


SnapshotEvaluator = Callable[[Particles], Mapping[str, Mapping[str, Quantity]]]
"""Type of a function that evaluates all choices for one AMUSE particle set."""


@dataclass(frozen=True)
class EvaluationChoice:
    """One concrete result group evaluated by a diagnostic.

    Args:
        name: Stable result-group identifier, such as ``"stellar_com"``.
        metadata: Method details written as HDF5 attributes for this choice.
        outputs: Mapping from standardized output names to their canonical unit
            strings, such as ``{"r_l50": "pc"}``.
    """

    name: str
    metadata: Mapping[str, str]
    outputs: Mapping[str, str]


@dataclass(frozen=True)
class TimeSeriesDiagnostic:
    """A versioned definition for a standardized snapshot diagnostic.

    Args:
        name: Stable diagnostic identifier, such as ``"lagrangian_radii"``.
        version: Scientific and output-schema version. Create a new version
            when results, parameter meanings, units, or output columns change.
        description: Human-readable scientific purpose of the diagnostic.
        evaluate: Function that receives one AMUSE :class:`Particles` set and
            returns scalar AMUSE quantities grouped by declared choice name.
        evaluation_choices: Concrete HDF5 result groups and their output
            schemas, such as ``"origin"`` and ``"stellar_com"``.
        field_descriptions: Human-readable descriptions keyed by output name.
        choice_names: Global scientific choices that affect this diagnostic,
            such as ``"center"``. Their values and defaults are registered in
            :mod:`csfdata_analysis.choices`.
        requires: Completed diagnostic versions required before this diagnostic
            can be calculated.

    Notes:
        A diagnostic definition contains no simulation paths, file I/O, or
        parallel-execution behavior. Those belong to the snapshot reader and
        runner. Each evaluation choice is the declared schema that the runner
        uses to validate every returned measurement. It is deliberately
        separate from ``choice_names``: evaluation groups are stored values,
        while global choices are user-facing selection dimensions.
    """

    name: str
    version: int
    description: str
    evaluate: SnapshotEvaluator
    evaluation_choices: tuple[EvaluationChoice, ...]
    field_descriptions: Mapping[str, str]
    choice_names: tuple[str, ...] = ()
    requires: tuple[DiagnosticRequirement, ...] = ()

    def catalogue_definition(self) -> DiagnosticDefinition:
        """Return this diagnostic's portable core-catalogue declaration.

        Returns:
            A time-series diagnostic definition ready to publish in a
            collection's ``diagnostics.yaml``.

        Raises:
            ValueError: If evaluation groups disagree about a field's unit or
                if output-field descriptions do not exactly match the schema.
        """
        # Combine all evaluation groups because they describe one shared
        # time-series file whose choice groups must expose compatible fields.
        units_by_field: dict[str, str] = {}
        for evaluation_choice in self.evaluation_choices:
            for field_name, unit in evaluation_choice.outputs.items():
                previous_unit = units_by_field.setdefault(field_name, unit)
                if previous_unit != unit:
                    raise ValueError(
                        f"Diagnostic {self.name!r} declares conflicting units for "
                        f"{field_name!r}."
                    )

        # Publishing incomplete documentation would make later queries opaque.
        if set(units_by_field) != set(self.field_descriptions):
            raise ValueError(
                f"Diagnostic {self.name!r} field descriptions must match its outputs."
            )
        fields = tuple(
            DiagnosticField(field_name, self.field_descriptions[field_name], unit)
            for field_name, unit in units_by_field.items()
        )

        # The runner and declaration use the same standard versioned location.
        return DiagnosticDefinition(
            name=self.name,
            version=f"v{self.version}",
            kind="time_series",
            description=self.description,
            relative_path=(
                PurePosixPath("derived")
                / "diagnostics"
                / self.name
                / f"v{self.version}"
                / "series.h5"
            ),
            fields=fields,
            choices=self.choice_names,
            requires=self.requires,
        )


@dataclass(frozen=True)
class ScalarDiagnostic:
    """Metadata template for one simulation-level scalar diagnostic.

    Args:
        name: Stable diagnostic identifier.
        version: Scientific and output-schema version.
        description: Human-readable scientific purpose of the diagnostic.
        fields: Scalar output fields written into
            ``derived/scalar_diagnostics.yaml``.
        choice_names: Global scientific choices that affect the scalar result.
        requires: Completed time-series or scalar diagnostic versions required
            before this scalar diagnostic can be calculated.

    Notes:
        This template deliberately contains no scientific calculation yet. A
        concrete module later pairs it with a calculation that reads the
        declared requirements and writes its values through the core summary
        API.
    """

    name: str
    version: int
    description: str
    fields: tuple[DiagnosticField, ...]
    choice_names: tuple[str, ...] = ()
    requires: tuple[DiagnosticRequirement, ...] = ()

    def catalogue_definition(self) -> DiagnosticDefinition:
        """Return this summary diagnostic's portable catalogue declaration.

        Returns:
            A summary diagnostic definition ready for collection publication.
        """
        # All scalar diagnostics share one YAML file; name and version select
        # their individual result section inside that file.
        return DiagnosticDefinition(
            name=self.name,
            version=f"v{self.version}",
            kind="scalar",
            description=self.description,
            relative_path=PurePosixPath("derived/scalar_diagnostics.yaml"),
            fields=self.fields,
            choices=self.choice_names,
            requires=self.requires,
        )
