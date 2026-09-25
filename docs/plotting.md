# Plotting

`csfdata_analysis.plotting` provides small plotting helpers for aggregated
time-series tables. These helpers never select simulations, align time grids,
or calculate statistics. Those scientific choices remain explicit in the data
workflow before plotting.

## Plot A Model Series

Load a collection, align its time coordinates, and average the seed ensemble:

```python
import numpy as np

from csfdata_analysis.datamodel import (
    aggregate_time_series,
    interpolate_time_series,
    load_collection_time_series,
    load_simulations,
)
from csfdata_analysis.plotting import plot_time_series

simulations = load_simulations(
    "/path/to/catalogue",
    collection_id="collection-id",
)
series = load_collection_time_series(
    simulations,
    diagnostics="lagrangian_radii",
    fields={"lagrangian_radii": ("r_l50",)},
)
aligned = interpolate_time_series(series, times=np.arange(0.0, 30.0, 0.1))
summary = aggregate_time_series(
    aligned,
    group_by=("Mstars", "tff", "sfe", "fret_max", "texp_over_tff"),
)
```

Then fix every model parameter except the one represented by colour:

```python
import matplotlib.pyplot as plt

ax = plot_time_series(
    summary,
    y="r_l50",
    x="time",
    fixed={
        "Mstars": 1000,
        "sfe": 0.1,
        "fret_max": 2.0,
        "texp_over_tff": 0.5,
    },
    color_by="tff",
)

plt.show(block=False)
```

The function draws one mean line and one standard-deviation band for each
value of `color_by`, then returns its Matplotlib `Axes` for further adjustment.
It raises an error when another grouped model parameter still varies, rather
than silently mixing different models in one line.

`x` defaults to `"time"`. Pass the name of another existing summary column to
use it as the horizontal axis, for example `x="time_over_tff"` when that
normalized coordinate has been prepared before aggregation.
