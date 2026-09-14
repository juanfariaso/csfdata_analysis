"""Definitions for standardized snapshot diagnostics.

A diagnostic evaluates one AMUSE particle set and returns named scalar AMUSE
quantities. The runner later combines those measurements into a time series.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from amuse.datamodel import Particles
from amuse.units.quantities import Quantity


SnapshotEvaluator = Callable[[Particles], Mapping[str, Mapping[str, Quantity]]]
"""Type of a function that evaluates all choices for one AMUSE particle set."""


@dataclass(frozen=True)
class DiagnosticChoice:
    """One scientifically defined choice within a diagnostic.

    Args:
        name: Stable choice identifier, such as ``"stellar_com"``.
        metadata: Method details written as HDF5 attributes for this choice.
        outputs: Mapping from standardized output names to their canonical unit
            strings, such as ``{"r_l50": "pc"}``.
    """

    name: str
    metadata: Mapping[str, str]
    outputs: Mapping[str, str]


@dataclass(frozen=True)
class Diagnostic:
    """A versioned definition for a standardized snapshot diagnostic.

    Args:
        name: Stable diagnostic identifier, such as ``"lagrangian_radii"``.
        version: Scientific and output-schema version. Create a new version
            when results, parameter meanings, units, or output columns change.
        evaluate: Function that receives one AMUSE :class:`Particles` set and
            returns scalar AMUSE quantities grouped by declared choice name.
        choices: Scientifically defined choices and their output schemas.

    Notes:
        A diagnostic definition contains no simulation paths, file I/O, or
        parallel-execution behavior. Those belong to the snapshot reader and
        runner. Each choice is the declared schema that the runner will use to
        validate every returned measurement.
    """

    name: str
    version: int
    evaluate: SnapshotEvaluator
    choices: tuple[DiagnosticChoice, ...]
