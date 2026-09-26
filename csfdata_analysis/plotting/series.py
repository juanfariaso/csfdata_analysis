"""Plot grouped, aggregated diagnostic time series."""

from __future__ import annotations

import matplotlib.pyplot as pyplot
from matplotlib.axes import Axes
import pandas


def plot_time_series(
    summary: pandas.DataFrame,
    y: str,
    x: str = "time",
    fixed: dict[str, str | int | float | bool] | None = None,
    color_by: str | None = None,
    ax: Axes | None = None,
    show_std: bool = True,
) -> Axes:
    """Plot one aggregated time-series field for selected model groups.

    Args:
        summary: Long summary table returned by ``aggregate_time_series``.
        y: Base diagnostic field name, such as ``"r_l50"``. The table must
            contain ``<y>_mean`` and ``<y>_std`` columns.
        x: Existing summary column used for the horizontal axis. Defaults to
            ``"time"``.
        fixed: Exact values for model parameters that should not vary in the
            figure. Parameters not fixed or assigned to ``color_by`` must have
            only one value in the selected rows.
        color_by: Optional grouped model parameter represented by one line and
            one shaded uncertainty band per value.
        ax: Optional existing Matplotlib axes. When omitted, create one.
        show_std: Whether to shade the stored one-standard-deviation interval.

    Returns:
        Matplotlib axes containing the mean lines and optional uncertainty
        bands. The function does not call ``show`` or save a figure.

    Raises:
        TypeError: If ``summary`` is not a Pandas DataFrame.
        ValueError: If summary metadata or columns are missing, filters select
            no rows, or unresolved model parameters would mix distinct models.

    Notes:
        Use the returned axes for titles, limits, annotations, and saving.
        ``matplotlib.pyplot.show(block=False)`` remains an explicit caller
        choice for interactive local sessions.
    """
    if not isinstance(summary, pandas.DataFrame):
        raise TypeError("summary must be a Pandas DataFrame.")
    if not isinstance(y, str) or not y:
        raise ValueError("y must be one non-empty diagnostic field name.")
    if not isinstance(x, str) or not x:
        raise ValueError("x must be one non-empty summary column name.")
    data_fields = summary.attrs.get("data_fields")
    group_by = summary.attrs.get("group_by")
    if not isinstance(data_fields, tuple) or y not in data_fields:
        raise ValueError(f"Summary does not declare diagnostic field {y!r}.")
    if not isinstance(group_by, tuple) or not all(isinstance(name, str) for name in group_by):
        raise ValueError("Summary has no grouping metadata. Run aggregate_time_series first.")
    required_columns = {"collection_id", x, f"{y}_mean", f"{y}_std", *group_by}
    if not required_columns.issubset(summary.columns):
        missing = ", ".join(sorted(required_columns - set(summary.columns)))
        raise ValueError(f"Summary is missing columns: {missing}.")
    if not isinstance(fixed, (dict, type(None))) or not all(
        isinstance(name, str) and isinstance(value, (str, int, float, bool))
        for name, value in (fixed or {}).items()
    ):
        raise ValueError("fixed must be a dictionary of parameter names and scalar values.")

    # Validate the plot selection before drawing so no figure silently mixes
    # physically distinct models that differ outside the requested colour axis.
    model_parameters = ("collection_id", *group_by)
    unknown_fixed = set(fixed or {}) - set(model_parameters)
    if unknown_fixed:
        names = ", ".join(sorted(unknown_fixed))
        raise ValueError(f"fixed contains parameters outside the grouped model: {names}.")
    if color_by is not None and color_by not in model_parameters:
        raise ValueError(f"color_by is not a grouped model parameter: {color_by!r}.")
    selected = summary
    for name, value in (fixed or {}).items():
        selected = selected[selected[name] == value]
    if selected.empty:
        raise ValueError("fixed selects no summary rows.")
    unresolved = [
        name
        for name in model_parameters
        if name != color_by
        and name not in (fixed or {})
        and selected[name].nunique(dropna=False) > 1
    ]
    if unresolved:
        names = ", ".join(unresolved)
        raise ValueError(f"Plot would mix unresolved model parameters: {names}.")

    # Draw one mean line for each requested model value. A shared colormap
    # makes every standard-deviation band visibly belong to its mean line.
    axis = ax if ax is not None else pyplot.subplots()[1]
    groups = (
        [(None, selected)]
        if color_by is None
        else list(selected.groupby(color_by, sort=True, observed=True))
    )
    colors = pyplot.colormaps["viridis"].resampled(len(groups))
    for index, (value, group) in enumerate(groups):
        ordered = group.sort_values(x)
        label = None if color_by is None else f"{color_by} = {value}"
        color = colors(index)
        axis.plot(ordered[x], ordered[f"{y}_mean"], color=color, label=label)
        if show_std:
            axis.fill_between(
                ordered[x],
                ordered[f"{y}_mean"] - ordered[f"{y}_std"],
                ordered[f"{y}_mean"] + ordered[f"{y}_std"],
                color=color,
                alpha=0.2,
            )
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    if color_by is not None:
        axis.legend(title=color_by)
    return axis
