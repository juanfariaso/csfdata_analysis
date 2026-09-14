"""Readers for D-CAF simulation output."""

from __future__ import annotations

from pathlib import Path

from amuse.datamodel import Particles
from amuse.io import read_set_from_file


def read_stars(path: Path) -> Particles:
    """Read one D-CAF stellar snapshot into AMUSE particles.

    Args:
        path: Path to a D-CAF ``stars_*.amuse`` snapshot file.

    Returns:
        Particles: Stellar particles stored in the snapshot.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"D-CAF stellar snapshot does not exist: {path}")

    return read_set_from_file(str(path), "amuse")
