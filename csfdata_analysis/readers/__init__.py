"""Readers that convert simulation output into standard analysis inputs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from amuse.datamodel import Particles
from amuse.units import units

from csfdata.adapters.base import SimulationAdapter

from csfdata_analysis.readers.dcaf import read_stars


def read_snapshot_with_ages(
    path: Path,
    snapshot_time: float,
    adapter: SimulationAdapter,
    read_snapshot: Callable[[Path], Particles],
) -> Particles:
    """Read one snapshot and attach format-specific stellar ages.

    Args:
        path: Native snapshot path.
        snapshot_time: Snapshot model time in canonical Myr.
        adapter: Initialized simulation-format adapter for the snapshot's run.
        read_snapshot: Format-specific function that returns AMUSE particles.

    Returns:
        Particles with an ``age`` attribute in Myr reconstructed from the
        adapter's formation-time mapping.

    Raises:
        NotImplementedError: If the adapter cannot recover stellar ages.
        KeyError: If any snapshot particle lacks a formation-time record.
        ValueError: If a formation time is later than the snapshot time.

    Notes:
        The dependency-light adapter owns format-specific formation-time
        recovery. This wrapper performs the AMUSE-specific age assignment.
    """
    stars = read_snapshot(path)
    formation_times = adapter.stellar_formation_times()

    # Match native AMUSE keys rather than particle order because snapshots may
    # contain different subsets as stars form over time.
    particle_keys = tuple(int(key) for key in stars.key)
    missing_keys = tuple(key for key in particle_keys if key not in formation_times)
    if missing_keys:
        displayed = ", ".join(str(key) for key in missing_keys[:10])
        suffix = "" if len(missing_keys) <= 10 else ", ..."
        raise KeyError(
            f"Snapshot {path} has {len(missing_keys)} particle key(s) without "
            f"formation times: {displayed}{suffix}."
        )

    ages = [snapshot_time - formation_times[key] for key in particle_keys]
    if any(age < -1.0e-6 for age in ages):
        raise ValueError(
            f"Snapshot {path} contains a star formed after its model time "
            f"{snapshot_time:g} Myr."
        )
    # D-CAF logs formation times to six decimals, so clamp only sub-micro-Myr
    # negative roundoff at the snapshot boundary to zero.
    stars.age = [max(age, 0.0) for age in ages] | units.Myr
    return stars


__all__ = ["read_snapshot_with_ages", "read_stars"]
