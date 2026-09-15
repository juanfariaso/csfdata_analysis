# Data Interfaces

`csfdata_analysis.data` provides small, explicit interfaces for selecting
catalogue simulations, loading derived time series into Pandas, and selecting
raw snapshot slices. It does not create diagnostics or move files.

## Full And Lite Catalogues

Every data function receives one catalogue root path. With a full catalogue,
that path contains both the index and the raw simulation data. With a lite
catalogue, it contains the copied index, metadata, configuration, and derived
products, but no raw snapshots.

A lite catalogue's `lite.yaml` records the absolute path of the full source
catalogue used when it was exported. You do not provide that source path again:

```python
simulations = load_simulations("/path/to/lite-catalogue")
```

`load_time_series` reads derived HDF5 files directly from the supplied
catalogue. `select_snapshot_slice` needs raw snapshots, so it reads `lite.yaml`
and automatically resolves the matching simulation in the recorded source
catalogue. If that source path is unavailable, raw-snapshot operations stop
safely rather than guessing a different location.

## Load Simulations

Use the core catalogue index to select lightweight simulation references:

```python
from csfdata_analysis.data.loader import load_simulations

simulations = load_simulations(
    "/path/to/catalogue",
    collection_id="dcaf-tff-grid-v1",
    filters={"Mstars": 1000, "tff": 1.0, "sfe": 0.1},
)
```

Do not include `seed_index` when selecting an ensemble that should be averaged
across seeds. The returned objects contain catalogue IDs and paths; they do
not load snapshot particles or HDF5 values.

## Load A Time Series

Load one stored diagnostic column as a tidy Pandas DataFrame:

```python
from csfdata_analysis.data.series import load_time_series

data = load_time_series(
    simulations,
    diagnostic="lagrangian_radii",
    version=1,
    choice="stellar_com",
    output="r_l50",
)
```

The table includes `collection_id`, `simulation_id`, all known canonical
configuration parameters, `time_myr`, and the requested output column. By
default, the function stops if any selected simulation lacks a completed
matching result. Use `allow_missing=True` only for an explicitly partial
exploratory selection.

## Align And Summarize

Different simulations may have different snapshot times. Align them explicitly
before calculating an ensemble mean or standard deviation:

```python
import numpy as np

from csfdata_analysis.data.series import interpolate_time_series

aligned = interpolate_time_series(
    data,
    times_myr=np.arange(0.0, 30.0, 0.1),
    output="r_l50",
)

summary = (
    aligned.groupby("time_myr")["r_l50"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
```

Interpolation is linear and occurs independently for every simulation. The
function never extrapolates: target times outside a simulation's stored range
receive `NaN`, and the `count` column therefore shows how many simulations
contribute at each time.

## Select Snapshot Slices

Select the nearest stored raw snapshot for every simulation without loading
AMUSE particles:

```python
from csfdata_analysis.data.slices import select_snapshot_slice

slices = select_snapshot_slice(
    simulations,
    time=1.5,
    normalization="tff",
)
```

With `normalization="tff"`, each simulation's physical target is
`1.5 * tff`. The returned DataFrame records the requested physical time, the
chosen snapshot path and time, and the signed time offset. This makes the
nearest-snapshot approximation visible before any later particle analysis.
