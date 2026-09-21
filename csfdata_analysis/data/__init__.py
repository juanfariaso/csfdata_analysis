"""Selection, time-series, and snapshot-slice data interfaces."""

from csfdata_analysis.data.loader import configuration_values, load_simulations, source_simulation
from csfdata_analysis.data.series import aggregate_time_series, interpolate_time_series, load_time_series
from csfdata_analysis.data.slices import select_snapshot_slice, select_time_series_slice

__all__ = [
    "aggregate_time_series",
    "configuration_values",
    "interpolate_time_series",
    "load_simulations",
    "load_time_series",
    "select_snapshot_slice",
    "select_time_series_slice",
    "source_simulation",
]
