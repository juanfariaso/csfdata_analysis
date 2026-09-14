"""Tests for analysis command-line filter parsing and confirmation."""

from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

import csfdata_analysis.cli as cli
from csfdata.catalogue import CatalogueSimulation
from csfdata_analysis.cli import parse_filters


def test_parse_filters_handles_collection_scalars_and_ranges() -> None:
    """Registry filters preserve their intended comparison types."""
    collection_id, filters = parse_filters(
        ("collection=dcaf-grid-v1", "tff=0.5:3.0", "sfe=0.3", "bound=true")
    )

    assert collection_id == "dcaf-grid-v1"
    assert filters == {"tff": (0.5, 3.0), "sfe": 0.3, "bound": True}


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
