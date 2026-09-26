"""Pandas-oriented views of stored CSFData catalogue simulations.

This package is the researcher-facing bridge between the durable core
catalogue and convenient analysis tables. Its modules keep simulation access,
evolving series, scalar values, and fixed-time selections separate.
"""

from csfdata_analysis.datamodel.simulations import (
    SimulationSet,
    configuration_values,
    load_simulations,
    source_simulation,
)
from csfdata_analysis.datamodel.scalars import load_collection_scalars
from csfdata_analysis.datamodel.inventory import (
    DiagnosticInventory,
    DiagnosticKinds,
    diagnostic_inventory,
)
from csfdata_analysis.datamodel.series import (
    aggregate_time_series,
    interpolate_time_series,
    load_collection_time_series,
    load_time_series,
)
from csfdata_analysis.datamodel.slices import (
    select_snapshot_slice,
    select_time_slice,
)

__all__ = [
    "aggregate_time_series",
    "SimulationSet",
    "configuration_values",
    "DiagnosticInventory",
    "DiagnosticKinds",
    "diagnostic_inventory",
    "interpolate_time_series",
    "load_collection_scalars",
    "load_collection_time_series",
    "load_simulations",
    "load_time_series",
    "select_snapshot_slice",
    "select_time_slice",
    "source_simulation",
]
