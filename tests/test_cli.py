"""Tests for analysis command-line filter parsing."""

from csfdata_analysis.cli import parse_filters


def test_parse_filters_handles_collection_scalars_and_ranges() -> None:
    """Registry filters preserve their intended comparison types."""
    collection_id, filters = parse_filters(
        ("collection=dcaf-grid-v1", "tff=0.5:3.0", "sfe=0.3", "bound=true")
    )

    assert collection_id == "dcaf-grid-v1"
    assert filters == {"tff": (0.5, 3.0), "sfe": 0.3, "bound": True}
