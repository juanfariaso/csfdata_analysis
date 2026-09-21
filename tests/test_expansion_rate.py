"""Tests for the standardized late-time expansion-rate diagnostic."""

from pathlib import Path

import h5py
import numpy as np
from amuse.units import units

from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.diagnostics import (
    read_collection_diagnostics,
    read_simulation_scalar_diagnostics,
    simulation_scalar_diagnostics_path,
)
from csfdata_analysis.diagnostics.scalar.expansion_rate import (
    EXPANSION_RATE_V1,
    compute_expansion_rate,
    fit_expansion_rate,
)
from csfdata_analysis.diagnostics.time_series.lagrangian_radii import (
    LAGRANGIAN_FRACTIONS,
)
from csfdata_analysis.catalogue_runner import compute_scalar_collection
from csfdata_analysis.runner import compute_scalar_diagnostic


def test_fit_expansion_rate_starts_after_star_formation() -> None:
    """The fit begins at the post-formation radius minimum, not an earlier one."""
    result = fit_expansion_rate(
        np.arange(8, dtype=float),
        np.asarray((0.2, 0.4, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0)),
        np.asarray((1, 2, 3, 4, 5, 5, 5, 5), dtype=float),
        min_points=4,
    )

    assert result["fit_t0"] == 4.0
    assert result["r_min"] == 2.0
    assert np.isclose(result["fit_r0"], 2.0)
    assert result["n_points"] == 4.0
    assert np.isclose(
        result["dRdt"],
        (1 | (units.pc / units.Myr)).value_in(units.kms),
    )
    assert np.isclose(result["r2"], 1.0)


def test_scalar_runner_registers_and_writes_every_choice(tmp_path: Path) -> None:
    """A scalar run registers its contract and writes every declared choice result."""
    collection_root = tmp_path / "collections" / "example-grid"
    simulation_root = collection_root / "simulations" / "0000"
    series_path = (
        simulation_root
        / "derived"
        / "diagnostics"
        / "lagrangian_radii"
        / "v1"
        / "series.h5"
    )
    series_path.parent.mkdir(parents=True)
    with h5py.File(series_path, "w") as series_file:
        series_file.attrs["complete"] = True
        series_file.attrs["format_schema_version"] = 2
        series_file.attrs["diagnostic_name"] = "lagrangian_radii"
        series_file.attrs["diagnostic_version"] = 1
        series_file.create_dataset("time", data=np.arange(10, dtype=float))
        choices_group = series_file.create_group("choices")
        for center in ("origin", "stellar_com"):
            group = choices_group.create_group(center)
            group.create_dataset(
                "n_stars",
                data=np.asarray((1, 2, 3, 4, 5, 5, 5, 5, 5, 5)),
            )
            for suffix, _ in LAGRANGIAN_FRACTIONS:
                group.create_dataset(
                    f"r_{suffix}",
                    data=np.asarray((0.2, 0.4, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
                )

    simulation = CatalogueSimulation("example-grid", "0000", simulation_root, "dcaf")
    report = compute_scalar_diagnostic(
        simulation,
        EXPANSION_RATE_V1,
    )
    direct_result = compute_expansion_rate(
        simulation,
        {"center": "stellar_com", "lagrangian_radius": "r_l50"},
    )
    collection_diagnostics = read_collection_diagnostics(
        collection_root / "diagnostics.yaml"
    )
    stored = read_simulation_scalar_diagnostics(
        simulation_scalar_diagnostics_path(simulation_root),
        collection_diagnostics,
    )

    assert direct_result["dRdt"] == report.results[0].values[0].value
    assert len(report.results) == 24
    assert report.written_count == 24
    assert len(stored.results) == 24
    assert {(definition.name, definition.version) for definition in collection_diagnostics.diagnostics} == {
        ("lagrangian_radii", "v1"),
        ("expansion_rate", "v1"),
    }
    assert {choice.name for choice in collection_diagnostics.choices} == {
        "center",
        "lagrangian_radius",
    }

    results = compute_scalar_collection(
        (simulation,),
        "expansion_rate",
    )

    assert results[0].status == "skipped"
