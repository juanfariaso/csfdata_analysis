"""Standardized diagnostic definitions and implementations."""

from csfdata_analysis.diagnostics.base import Diagnostic, DiagnosticChoice
from csfdata_analysis.diagnostics.lagrangian_radii import LAGRANGIAN_RADII_V1

DIAGNOSTICS = {"lagrangian_radii": LAGRANGIAN_RADII_V1}
"""Diagnostics selected by name through the command-line interface."""

__all__ = ["Diagnostic", "DiagnosticChoice", "DIAGNOSTICS"]
