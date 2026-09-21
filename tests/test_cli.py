"""Tests for analysis command-line filter parsing and confirmation."""

from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

import csfdata_analysis.cli as cli
from csfdata.catalogue import CatalogueSimulation
from csfdata_analysis.catalogue_runner import SimulationResult
from csfdata_analysis.cli import parse_filters


def test_parse_filters_handles_collection_scalars_and_ranges() -> None:
    """Registry filters preserve their intended comparison types."""
    collection_id, filters = parse_filters(
        ("collection=dcaf-grid-v1", "tff=0.5:3.0", "sfe=0.3", "bound=true")
    )

    assert collection_id == "dcaf-grid-v1"
    assert filters == {"tff": (0.5, 3.0), "sfe": 0.3, "bound": True}


def test_diagnostics_lists_types_versions_and_evaluator_summaries(
    capsys: CaptureFixture[str],
) -> None:
    """The diagnostics command describes every built-in executable diagnostic."""
    assert cli.main(["diagnostics"]) == 0

    output = capsys.readouterr().out
    assert "Time-series diagnostics:" in output
    assert "lagrangian_radii v1: Measure stellar Lagrangian radii" in output
    assert "Scalar diagnostics:" in output
    assert "expansion_rate v1: Compute one choice-specific expansion-rate result" in output


def test_compute_lists_unfiltered_collections_before_confirmation(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """An unfiltered run shows its complete selection before it can start."""
    monkeypatch.setattr(
        cli,
        "find_simulations",
        lambda *args, **kwargs: (
            CatalogueSimulation("first", "0001", Path("/tmp/first"), "dcaf"),
            CatalogueSimulation("second", "0001", Path("/tmp/second"), "dcaf"),
            CatalogueSimulation("second", "0002", Path("/tmp/third"), "dcaf"),
        ),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    assert cli.main(["compute", "lagrangian_radii", "--catalogue", "/tmp/catalogue"]) == 0

    output = capsys.readouterr().out
    assert "first: 1 simulations" in output
    assert "second: 2 simulations" in output
    assert "Total simulations: 3" in output
    assert "Cancelled." in output


def test_compute_dispatches_scalar_diagnostics_to_scalar_runner(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """The shared compute command selects the scalar collection runner."""
    simulation = CatalogueSimulation("example-grid", "0000", Path("/tmp/example"), "dcaf")
    received: dict[str, object] = {}
    monkeypatch.setattr(cli, "find_simulations", lambda *args, **kwargs: (simulation,))

    def compute_scalar(simulations, diagnostic_name, **kwargs):
        """Record scalar dispatch without evaluating the fixture simulation."""
        received["simulations"] = simulations
        received["diagnostic_name"] = diagnostic_name
        received.update(kwargs)
        return (SimulationResult(simulation, "complete"),)

    monkeypatch.setattr(cli, "compute_scalar_collection", compute_scalar)

    assert (
        cli.main(
            [
                "compute",
                "expansion_rate",
                "--catalogue",
                "/tmp/catalogue",
                "--no-prompt",
            ]
        )
        == 0
    )

    assert received["simulations"] == (simulation,)
    assert received["diagnostic_name"] == "expansion_rate"
    assert received["workers"] == 1
    assert "Complete: 1" in capsys.readouterr().out


def test_clear_derived_requires_confirmation_before_removing(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """Derived clearing shows its selection before it can remove files."""
    monkeypatch.setattr(
        cli,
        "find_simulations",
        lambda *args, **kwargs: (
            CatalogueSimulation("first", "0001", Path("/tmp/first"), "dcaf"),
        ),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    assert (
        cli.main(
            [
                "clear-derived",
                "lagrangian_radii",
                "--catalogue",
                "/tmp/catalogue",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Diagnostic: lagrangian_radii" in output
    assert "Selected simulations: 1" in output
    assert "Cancelled." in output
