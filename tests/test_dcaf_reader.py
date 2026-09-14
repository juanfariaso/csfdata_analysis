"""Tests for D-CAF raw-output readers."""

from pathlib import Path

import pytest

from csfdata_analysis.readers import read_stars


def test_read_stars_rejects_a_missing_snapshot() -> None:
    """Missing input is reported before AMUSE is imported."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_stars(Path("missing-stars.amuse"))
