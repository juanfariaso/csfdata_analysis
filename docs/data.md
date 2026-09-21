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
catalogue used when it was imported. You do not provide that source path again:

```python
simulations = load_simulations("/path/to/lite-catalogue")
```

`load_time_series` uses each simulation's lazy `diagnostics` interface to read
only requested derived fields. `select_snapshot_slice` needs raw snapshots, so
it reads `lite.yaml` and automatically resolves the matching simulation in the
recorded source catalogue. If that source path is unavailable, raw-snapshot
operations stop safely rather than guessing a different location.

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

## Load Time Series

Load one or more fields from one or more stored diagnostics. The return value
is a dictionary keyed by `(diagnostic_name, version)`, with one wide Pandas
table per diagnostic:

```python
from csfdata_analysis.data.series import load_time_series

data = load_time_series(
    simulations,
    diagnostics={
        ("lagrangian_radii", "v1"): {
            "choices": {"center": "stellar_com"},
            "fields": ("r_l50", "n_l50"),
        },
        ("radial_velocity_3d", "v1"): {
            "choices": {"center": "stellar_com"},
            "fields": ("mean_vr", "sigma_vr"),
        },
    },
)

radii = data[("lagrangian_radii", "v1")]
velocity = data[("radial_velocity_3d", "v1")]
```

Every table includes `collection_id`, `simulation_id`, all known canonical
configuration parameters, selected scientific choices, `time`, and its
requested output fields. By default, the function stops if any selected
simulation lacks a completed matching result. Use `allow_missing=True` only
for an explicitly partial exploratory selection.

## Align And Summarize

Different simulations may have different snapshot times. Align them explicitly
before calculating an ensemble mean or standard deviation:

```python
import numpy as np

from csfdata_analysis.data.series import aggregate_time_series, interpolate_time_series

aligned = interpolate_time_series(
    data,
    times=np.arange(0.0, 30.0, 0.1),
)

summary = aggregate_time_series(aligned)
radii_summary = summary[("lagrangian_radii", "v1")]
```

Interpolation is linear and occurs independently for every simulation. The
function never extrapolates: target times outside a simulation's stored range
receive `NaN`. Every summary table contains `n_simulations` plus
`<field>_mean` and `<field>_std` columns. Use `group_by=("tff", "sfe")` with
`aggregate_time_series` to produce separate ensembles for configuration-value
combinations. `n_simulations` counts simulations with finite values for every
field selected in that diagnostic table.

For a seed ensemble, omit `seed_index` from the initial `load_simulations`
filters and from `group_by`. Include every parameter that defines the physical
model in `group_by`; rows that differ only in seed are then averaged together.

## Select Diagnostic Slices

Select linearly interpolated derived values at one time. A normalization makes
the requested time relative to a canonical Myr-valued configuration parameter:

```python
from csfdata_analysis.data.slices import select_time_series_slice

slices = select_time_series_slice(
    data,
    time=1.5,
    normalization="tff",
)

radii_slice = slices[("lagrangian_radii", "v1")]
```

Every resulting row contains one simulation's requested physical time and
interpolated output fields. Values outside its stored time range are `NaN`.

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

## Create A Transfer Manifest

The repository stores a script that turns one collection selection and slice
rule into a portable YAML manifest plus an exact rsync file list:

```bash
python scripts/create_slice_manifest.py \
  --catalogue /path/to/source-or-lite-catalogue \
  --collection dcaf-tff-grid-v1 \
  --filter tff=1.0 \
  --time 1.5 \
  --normalization tff \
  /path/to/manifest-directory
```

It writes `selection.yaml`, which records the selected simulations and actual
nearest snapshot times, and `rsync-files.txt`, which contains the exact paths
below the full source catalogue. Use the latter with `rsync --files-from` to
copy only the selected metadata, configurations, and raw snapshots while
preserving the standard catalogue layout.
