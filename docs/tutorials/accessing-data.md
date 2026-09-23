# Accessing The Data

This page is a short reference for reading already calculated diagnostic data
and selecting raw snapshot paths. It does not compute diagnostics or load
particle data into memory.

## Inspect One Simulation

Select simulations through the catalogue index with
[`load_simulations`](../reference/csfdata_analysis/data/loader.md#csfdata_analysis.data.loader.load_simulations),
then inspect the first result:

```python
from csfdata_analysis.data import load_simulations

simulations = load_simulations(
    "/path/to/catalogue",
    collection_id="collection-id",
    filters={"tff": 1.0},
)
simulation = simulations[0]

print(simulation.diagnostics)
```

[`simulation.diagnostics`](https://juanfariaso.github.io/csfdata/reference/csfdata/catalogue/registry/#csfdata.catalogue.registry.CatalogueSimulation.diagnostics)
is lazy: printing it shows a compact inventory of available time-series and
scalar diagnostic versions, fields, default choices, and stored choice values
without loading their numerical arrays.

## Select Raw Snapshot Slices

First select the simulations through ordinary catalogue filters. Then pass that
explicit selection to
[`select_snapshot_slice`](../reference/csfdata_analysis/data/slices.md#csfdata_analysis.data.slices.select_snapshot_slice):

```python
from csfdata_analysis.data import load_simulations, select_snapshot_slice

simulations = load_simulations(
    "/path/to/catalogue",
    collection_id="collection-id",
    filters={
        "tff": 1.0,
        "sfe": 0.1,
        "fret_max": (2.0, None),
    },
)

snapshot_slice = select_snapshot_slice(
    simulations,
    time=1.5,
    normalization="tff",
)
```

With `normalization="tff"`, the target time is `1.5 * tff` separately for
each simulation. `snapshot_slice` is a Pandas table containing the selected
`snapshot_path`, requested physical time, actual snapshot time, and its offset.
It only identifies files; it does not read AMUSE particles.

### Work With One Snapshot

To work with particles from one particular simulation, include its `seed_index`
in the filters so the selection contains one row. Check the actual selected
time, then use [`read_stars`](../reference/csfdata_analysis/readers/dcaf.md#csfdata_analysis.readers.dcaf.read_stars)
to load that one D-CAF snapshot:

```python
from pathlib import Path

from csfdata_analysis.data import load_simulations, select_snapshot_slice
from csfdata_analysis.readers import read_stars

simulations = load_simulations(
    "/path/to/catalogue",
    collection_id="collection-id",
    filters={"tff": 1.0, "sfe": 0.1, "seed_index": 3},
)
snapshot = select_snapshot_slice(simulations, time=10.0).iloc[0]

print(snapshot["snapshot_time"])
print(snapshot["time_offset"])

stars = read_stars(Path(snapshot["snapshot_path"]))
```

`stars` is an AMUSE `Particles` object for the selected stored file. This is
appropriate for an exploratory calculation; it does not create or register a
catalogue diagnostic.

## Read A Time Series

Select a diagnostic by name to use its latest collection-declared version,
inspect it, then request one stored choice and the fields needed for the
analysis. The explicit `(name, version)` form remains available when needed:

```python
radii = simulation.diagnostics.time_series["lagrangian_radii"]
print(radii)

data = radii.read(
    choices={"center": "stellar_com"},
    fields=("r_l50", "n_l50"),
)

time = data["time"]
half_mass_radius = data["r_l50"]
```

Time-series reads return NumPy arrays. A
[`SimulationDiagnostic`](https://juanfariaso.github.io/csfdata/reference/csfdata/catalogue/diagnostics/#csfdata.catalogue.diagnostics.SimulationDiagnostic)
provides `fields`, `choices`, and `available_choices` to inspect what is
available before reading.

For plotting data from one simulation, use the analysis-layer
[`load_time_series`](../reference/csfdata_analysis/data/series.md#csfdata_analysis.data.series.load_time_series)
function. It loads all requested diagnostic fields by default into one Pandas
table with a shared ``time`` column:

```python
from csfdata_analysis.data import load_time_series

data = load_time_series(
    simulation,
    diagnostics=("lagrangian_radii", "radial_velocity_3d"),
    choices={"center": "stellar_com"},
)

data.plot(x="time", y="r_l50")
```

## Read A Scalar Diagnostic

Scalar diagnostics use the same `(name, version)` identity, but return
canonical Python scalar values:

```python
rates = simulation.diagnostics.scalar[("expansion_rate", "v1")]
print(rates)

values = rates.read(choices={"center": "stellar_com"})
expansion_rate = values["dRdt"]
```

## Work Across A Collection

For plotting or statistics across many simulations,
[`load_collection_time_series`](../reference/csfdata_analysis/data/series.md#csfdata_analysis.data.series.load_collection_time_series)
loads the desired diagnostic field into a Pandas table,
[`interpolate_time_series`](../reference/csfdata_analysis/data/series.md#csfdata_analysis.data.series.interpolate_time_series)
aligns its output times, and
[`aggregate_time_series`](../reference/csfdata_analysis/data/series.md#csfdata_analysis.data.series.aggregate_time_series)
summarizes it. This example averages simulations that share physical parameters
but differ in random seed. It deliberately does not filter on `seed_index`:

```python
import numpy as np

from csfdata_analysis.data import (
    aggregate_time_series,
    interpolate_time_series,
    load_collection_time_series,
    load_simulations,
)

simulations = load_simulations(
    "/path/to/catalogue",
    collection_id="collection-id",
)

series = load_collection_time_series(
    simulations,
    diagnostics="lagrangian_radii",
    choices={"center": "stellar_com"},
    fields={"lagrangian_radii": ("r_l50",)},
)

aligned = interpolate_time_series(
    series,
    times=np.arange(0.0, 30.0, 0.1),
)

summary = aggregate_time_series(
    aligned,
    group_by=("tff", "sfe", "fret_max", "texp_over_tff", "Mstars"),
)

```

Each summary row represents one exact combination of the listed parameters at
one model time. `n_simulations`, `<field>_mean`, and `<field>_std` therefore
describe the seed ensemble. Include another parameter in `group_by` when it
defines a physically distinct model; leave it out only when it should be
averaged over, as with `seed_index`.
