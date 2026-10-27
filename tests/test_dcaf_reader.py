"""Tests for D-CAF raw-output readers."""

from pathlib import Path

import pytest
import h5py
from amuse.datamodel import Particles
from amuse.io import write_set_to_file
from amuse.units import units

from csfdata_analysis.readers import read_stars, read_stars_with_ages


def test_read_stars_rejects_a_missing_snapshot() -> None:
    """Missing input is reported before AMUSE is imported."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_stars(Path("missing-stars.amuse"))


def test_read_stars_with_ages_resolves_dcaf_context(tmp_path: Path) -> None:
    """The public D-CAF reader derives its adapter and snapshot time."""
    output_root = tmp_path / "run" / "dcaf_output"
    output_root.mkdir(parents=True)
    snapshot_path = output_root / "stars_000.amuse"
    stars = Particles(keys=(101, 102))
    stars.mass = [1.0, 2.0] | units.MSun
    stars.collection_attributes.model_time = 2.0 | units.Myr
    write_set_to_file(stars, str(snapshot_path), "amuse")
    # D-CAF stores the collection model time as an SI value in this HDF5
    # attribute; reproduce that native representation in the synthetic file.
    with h5py.File(snapshot_path, "r+") as snapshot_file:
        group_name = sorted(snapshot_file["data"].keys())[0]
        snapshot_file[f"data/{group_name}"].attrs["model_time"] = (
            2.0 | units.Myr
        ).value_in(units.s)
    (output_root / "dcaf.log").write_text(
        "2026-01-01 [INFO] [NEW_STAR] 0.500000 101 1.0\n"
        "2026-01-01 [INFO] [NEW_STAR] 1.250000 102 2.0\n",
        encoding="utf-8",
    )

    loaded = read_stars_with_ages(snapshot_path)

    assert loaded.key.tolist() == [101, 102]
    assert loaded.age.value_in(units.Myr).tolist() == pytest.approx([1.5, 0.75])
