"""Registered scientific choices used by standardized derived parameters.

This module is the human-maintained source for choice names, allowed values,
and defaults used while calculating new analysis results. Catalogue collections
later receive a serialized copy in ``diagnostics.yaml`` so CSFData can query
existing results without importing this package.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Choice:
    """One named scientific choice with its allowed values and default.

    Args:
        name: Stable machine-readable choice name used in code and catalogue
            data, such as ``"center"``.
        description: Human-readable scientific meaning of the choice.
        values: Allowed stable values for this choice.
        default: Value used by a parameter when neither a parameter-specific
            nor global query choice supplies one.

    Raises:
        ValueError: If the name is empty, allowed values are duplicated or
            empty, or the default is not one of the allowed values.
    """

    name: str
    description: str
    values: tuple[str, ...]
    default: str

    def __post_init__(self) -> None:
        """Validate one immutable choice definition at module import time."""
        if not self.name:
            raise ValueError("Choice names must not be empty.")
        if not self.values or any(not value for value in self.values):
            raise ValueError(f"Choice {self.name!r} must define non-empty values.")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Choice {self.name!r} defines duplicate values.")
        if self.default not in self.values:
            raise ValueError(
                f"Choice {self.name!r} default must be one of its allowed values."
            )


CENTER = Choice(
    name="center",
    description="Reference centre used for position-dependent measurements.",
    values=("origin", "stellar_com"),
    default="stellar_com",
)
"""Registered centre definitions for position-dependent derived parameters."""


LAGRANGIAN_RADIUS = Choice(
    name="lagrangian_radius",
    description="Stellar Lagrangian radius used by a derived measurement.",
    values=(
        "r_l01",
        "r_l05",
        "r_l10",
        "r_l20",
        "r_l30",
        "r_l40",
        "r_l50",
        "r_l60",
        "r_l70",
        "r_l80",
        "r_l90",
        "r_l95",
    ),
    default="r_l50",
)
"""Registered stellar Lagrangian-radius choices for derived parameters."""


CHOICES = {
    CENTER.name: CENTER,
    LAGRANGIAN_RADIUS.name: LAGRANGIAN_RADIUS,
}
"""Registered choices keyed by their stable machine-readable names."""
