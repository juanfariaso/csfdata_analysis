"""Tests for concise grouped time-series plots."""

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
import pandas
import pytest

from csfdata_analysis.plotting import plot_time_series


def test_plot_time_series_colours_one_model_parameter() -> None:
    """One line and uncertainty band are drawn for every colour value."""
    summary = pandas.DataFrame(
        {
            "collection_id": ("grid", "grid", "grid", "grid"),
            "tff": (0.5, 0.5, 1.0, 1.0),
            "sfe": (0.1, 0.1, 0.1, 0.1),
            "time": (0.0, 1.0, 0.0, 1.0),
            "time_over_tff": (0.0, 2.0, 0.0, 1.0),
            "r_l50_mean": (1.0, 2.0, 1.5, 3.0),
            "r_l50_std": (0.1, 0.2, 0.2, 0.3),
        }
    )
    summary.attrs["data_fields"] = ("r_l50",)
    summary.attrs["group_by"] = ("tff", "sfe")

    axis = plot_time_series(
        summary,
        y="r_l50",
        fixed={"sfe": 0.1},
        color_by="tff",
    )

    assert len(axis.lines) == 2
    assert len(axis.collections) == 2
    assert [line.get_label() for line in axis.lines] == ["tff = 0.5", "tff = 1.0"]
    pyplot.close(axis.figure)


def test_plot_time_series_uses_requested_horizontal_column() -> None:
    """A caller can plot against an existing summary column other than time."""
    summary = pandas.DataFrame(
        {
            "collection_id": ("grid", "grid"),
            "tff": (1.0, 1.0),
            "time": (0.0, 1.0),
            "time_over_tff": (0.0, 2.0),
            "r_l50_mean": (1.0, 2.0),
            "r_l50_std": (0.1, 0.2),
        }
    )
    summary.attrs["data_fields"] = ("r_l50",)
    summary.attrs["group_by"] = ("tff",)

    axis = plot_time_series(summary, y="r_l50", x="time_over_tff")

    assert axis.get_xlabel() == "time_over_tff"
    assert axis.lines[0].get_xdata().tolist() == [0.0, 2.0]
    pyplot.close(axis.figure)


def test_plot_time_series_rejects_unresolved_model_parameters() -> None:
    """A figure cannot silently mix models that differ outside its colour axis."""
    summary = pandas.DataFrame(
        {
            "collection_id": ("grid", "grid"),
            "tff": (1.0, 1.0),
            "sfe": (0.1, 0.3),
            "time": (0.0, 0.0),
            "r_l50_mean": (1.0, 2.0),
            "r_l50_std": (0.1, 0.2),
        }
    )
    summary.attrs["data_fields"] = ("r_l50",)
    summary.attrs["group_by"] = ("tff", "sfe")

    with pytest.raises(ValueError, match="unresolved model parameters: sfe"):
        plot_time_series(summary, y="r_l50", color_by="tff")
