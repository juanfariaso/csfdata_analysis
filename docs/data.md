# Data Interfaces

This page describes two compatible data-access layers:

- **Core CSFData** provides lightweight catalogue simulation objects and lazy,
  dependency-light diagnostic reads as NumPy arrays.
- **CSFData Analysis** builds on those objects to provide Pandas tables,
  time-series interpolation and aggregation, and raw snapshot selection.

Neither layer creates diagnostics or moves files. Use the core layer when a
script needs direct access to one stored result; use the analysis layer for
interactive plotting and working with multiple diagnostics or simulations.

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

## Core Diagnostic Access

The returned simulation object exposes lazy stored diagnostics through core
CSFData. A diagnostic name selects its latest collection-declared version,
currently ``v1``. This reads NumPy arrays directly and does not require
Pandas:

```python
simulation = simulations[0]

radii = simulation.diagnostics.time_series["lagrangian_radii"]
raw = radii.read(
    choices={"center": "stellar_com"},
    fields=("r_l50", "n_l50"),
)

time = raw["time"]
half_mass_radius = raw["r_l50"]
```

Omit ``fields`` to read every result field. The exact version remains available
when needed for comparison or reproducibility:

```python
radii_v1 = simulation.diagnostics.time_series[("lagrangian_radii", "v1")]
```

## Load Time Series

The CSFData Analysis layer loads one or more stored diagnostics from one
simulation into one Pandas table. All fields are loaded by default. Diagnostic
names resolve to their latest collection-declared versions, currently ``v1``.
Choices use their registered defaults unless the analysis makes an explicit
scientific selection:

```python
from csfdata_analysis.data.series import load_time_series

data = load_time_series(
    simulations[0],
    diagnostics=("lagrangian_radii", "radial_velocity_3d"),
    choices={"center": "stellar_com"},
)

data.plot(x="time", y="r_l50")
```

The table has one shared ``time`` column and the selected output fields. The
loader verifies that every requested diagnostic has exactly the same stored
time coordinate and fails rather than silently interpolating a mismatch. Its
``attrs`` record the resolved diagnostic versions, choices, and fields.

Use ``fields`` only when reading every result field would be unnecessary:

```python
data = load_time_series(
    simulations[0],
    diagnostics=("lagrangian_radii", "radial_velocity_3d"),
    choices={"center": "stellar_com"},
    diagnostic_choices={"lagrangian_radii": {"center": "origin"}},
    fields={
        "lagrangian_radii": ("r_l50", "n_l50"),
        "radial_velocity_3d": ("mean_vr",),
    },
)
```

Global choices apply to every diagnostic that uses them. A
diagnostic-specific choice overrides the global value for that one diagnostic.
Globally declared choices that do not affect a diagnostic are ignored; a typo
or an irrelevant diagnostic-specific choice raises an error.

## Load A Collection

``load_collection_time_series`` uses the same names, global choices,
diagnostic-specific overrides, and optional field selection as
``load_time_series``. It returns one long Pandas table: each row is one stored
time from one simulation, with stable IDs and canonical configuration values.
Use it before interpolation and aggregation across simulations:

```python
from csfdata_analysis.data.series import load_collection_time_series

data = load_collection_time_series(
    simulations,
    diagnostics="lagrangian_radii",
    choices={"center": "stellar_com"},
    fields={"lagrangian_radii": ("r_l50",)},
)
```

The older exact-version dictionary form is still accepted for compatibility,
but new analysis should use the concise form above.

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
```

Interpolation is linear and occurs independently for every simulation. The
function never extrapolates: target times outside a simulation's stored range
receive `NaN`. The summary table contains `n_simulations` plus
`<field>_mean` and `<field>_std` columns. Use `group_by=("tff", "sfe")` with
`aggregate_time_series` to produce separate ensembles for configuration-value
combinations. `n_simulations` counts simulations with finite values for every
selected field.

For a seed ensemble, omit `seed_index` from the initial `load_simulations`
filters and from `group_by`. Include every parameter that defines the physical
model in `group_by`; rows that differ only in seed are then averaged together.

## Select Diagnostic Slices

`select_time_series_slice` currently uses the legacy dictionary-of-diagnostic-
tables interface. It selects linearly interpolated derived values at one time.
A normalization makes the requested time relative to a canonical Myr-valued
configuration parameter:

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

For the concise collection DataFrame interface, first use
`interpolate_time_series` with an explicit time grid, then select the desired
rows with ordinary Pandas filtering.

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
