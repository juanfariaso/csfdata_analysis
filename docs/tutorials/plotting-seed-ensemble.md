# Plotting A Seed Ensemble

This tutorial plots a time-series diagnostic for one model while averaging
simulations that differ only in `seed_index`. It keeps the model parameters
explicit, aligns the individual stored time grids, and then plots the mean
with its seed-to-seed standard deviation.

## Inspect The Grid

Collections can declare their expected grid axes in `collection.yaml`. Inspect
the declared values and the indexed coverage before selecting models:

```bash
csfdata catalogue-summary /path/to/catalogue --collection collection-id
```

The `Grid coverage` section lists the expected number of combinations and any
missing combinations. The `Simulation parameters` section lists the values
currently indexed. `grid_axes` is optional: if a collection does not declare
it, the same summary still reports the values that are actually available.

## Load, Align, And Average

Select the collection, load the desired diagnostic, and interpolate every seed
onto one explicit physical-time grid. Do not filter by `seed_index`: leaving
it out selects the full seed ensemble.

```python
import numpy as np

from csfdata_analysis.datamodel import (
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
    group_by=("Mstars", "tff", "sfe", "fret_max", "texp_over_tff"),
)
```

Every `summary` row now represents one model-parameter combination at one
shared time. It contains `r_l50_mean`, `r_l50_std`, and `n_simulations`.
The listed `group_by` parameters must include every parameter that defines a
physically distinct model. `seed_index` is omitted so those realizations are
averaged together.

## Plot One Model Family

Fix the parameters that should not vary, then use one remaining model
parameter as the line colour:

```python
import matplotlib.pyplot as plt

from csfdata_analysis.plotting import plot_time_series

ax = plot_time_series(
    summary,
    y="r_l50",
    fixed={
        "Mstars": 1000,
        "sfe": 0.1,
        "fret_max": 2.0,
        "texp_over_tff": 0.5,
    },
    color_by="tff",
)

ax.set_title("Half-mass-radius evolution")
plt.show(block=False)
```

This produces one line per `tff` value. Each line is the seed-ensemble mean
and its shaded band is one standard deviation. `plot_time_series` rejects a
plot that would silently mix another varying model parameter; add that
parameter to `fixed` or choose it as `color_by`.
