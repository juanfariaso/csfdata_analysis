"""Internal HDF5 storage for particle-level diagnostic time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from csfdata_analysis.diagnostics.base import EvaluationChoice


PARTICLE_FORMAT_SCHEMA_VERSION = 1
"""Current schema version for ``particles.h5`` diagnostic sidecars."""


@dataclass(frozen=True)
class StoredParticleChoice:
    """Materialized columns for one stored diagnostic choice."""

    snapshot_id: np.ndarray
    particle_key: np.ndarray
    values: dict[str, np.ndarray]


@dataclass(frozen=True)
class StoredParticleSeries:
    """Materialized contents of one completed ``particles.h5`` file."""

    diagnostic_name: str
    diagnostic_version: int
    series_id: str
    choices: dict[str, StoredParticleChoice]


def particle_series_path(series_path: Path) -> Path:
    """Return the particle sidecar beside a diagnostic ``series.h5`` path."""
    return series_path.with_name("particles.h5")


def hdf5_dtype(dtype: str) -> np.dtype[Any] | h5py.Datatype:
    """Translate one declared particle dtype to a stable HDF5 dtype."""
    if dtype == "string":
        return h5py.string_dtype(encoding="utf-8")
    try:
        parsed = np.dtype(dtype)
    except TypeError as error:
        raise ValueError(f"Unsupported particle field dtype: {dtype!r}.") from error
    if parsed.kind not in "biuf":
        raise ValueError(f"Unsupported particle field dtype: {dtype!r}.")
    return parsed


class ParticleSeriesWriter:
    """Append validated particle rows to a temporary HDF5 sidecar."""

    def __init__(
        self,
        path: Path,
        diagnostic_name: str,
        diagnostic_version: int,
        series_id: str,
        choices: tuple[EvaluationChoice, ...],
        identity: dict[str, str | int],
    ) -> None:
        self.path = path
        self._file = h5py.File(path, "w")
        try:
            self._file.attrs["format_schema_version"] = PARTICLE_FORMAT_SCHEMA_VERSION
            self._file.attrs["diagnostic_name"] = diagnostic_name
            self._file.attrs["diagnostic_version"] = diagnostic_version
            self._file.attrs["series_id"] = series_id
            for name, value in identity.items():
                self._file.attrs[name] = value

            choices_group = self._file.create_group("choices")
            for choice in choices:
                if not choice.particle_fields:
                    continue
                choice_group = choices_group.create_group(choice.name)
                for name, value in choice.metadata.items():
                    choice_group.attrs[name] = value
                choice_group.create_dataset(
                    "snapshot_id",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=True,
                    compression="gzip",
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                choice_group.create_dataset(
                    "particle_key",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=True,
                    compression="gzip",
                    shuffle=True,
                    dtype=np.uint64,
                )
                for field in choice.particle_fields:
                    dataset = choice_group.create_dataset(
                        field.name,
                        shape=(0,),
                        maxshape=(None,),
                        chunks=True,
                        compression="gzip",
                        shuffle=field.dtype != "string",
                        dtype=hdf5_dtype(field.dtype),
                    )
                    dataset.attrs["description"] = field.description
                    dataset.attrs["dtype"] = field.dtype
                    if field.unit is not None:
                        dataset.attrs["unit"] = field.unit
                    for name, value in field.metadata.items():
                        dataset.attrs[name] = value
        except Exception:
            self._file.close()
            raise

    def append(
        self,
        choice_name: str,
        snapshot_id: str,
        particle_key: np.ndarray,
        values: dict[str, np.ndarray],
    ) -> None:
        """Append one snapshot's rows for one declared choice."""
        choice_group = self._file[f"choices/{choice_name}"]
        start = len(choice_group["particle_key"])
        stop = start + len(particle_key)
        for name in ("snapshot_id", "particle_key", *values):
            dataset = choice_group[name]
            dataset.resize((stop,))
        choice_group["snapshot_id"][start:stop] = np.full(
            len(particle_key), snapshot_id, dtype=object
        )
        choice_group["particle_key"][start:stop] = particle_key
        for name, value in values.items():
            choice_group[name][start:stop] = value

    def complete(self) -> None:
        """Mark the sidecar complete and close it."""
        self._file.attrs["complete"] = True
        self._file.close()

    def close(self) -> None:
        """Close an incomplete sidecar after a failed calculation."""
        if self._file:
            self._file.close()


def read_particle_series(path: Path) -> StoredParticleSeries:
    """Read one completed particle sidecar for internal round-trip checks."""
    with h5py.File(path, "r") as input_file:
        if input_file.attrs.get("complete") != True:
            raise ValueError(f"Particle series is incomplete: {path}")
        if input_file.attrs.get("format_schema_version") != PARTICLE_FORMAT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported particle-series schema: {path}")
        choices = {}
        for choice_name, choice_group in input_file["choices"].items():
            values = {
                name: dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:]
                for name, dataset in choice_group.items()
                if name not in {"snapshot_id", "particle_key"}
            }
            snapshot_id = choice_group["snapshot_id"].asstr()[:]
            particle_key = choice_group["particle_key"][:]
            row_count = len(particle_key)
            if len(snapshot_id) != row_count or any(
                len(value) != row_count for value in values.values()
            ):
                raise ValueError(f"Particle-series column lengths differ: {path}")
            choices[choice_name] = StoredParticleChoice(
                snapshot_id,
                particle_key,
                values,
            )
        return StoredParticleSeries(
            str(input_file.attrs["diagnostic_name"]),
            int(input_file.attrs["diagnostic_version"]),
            str(input_file.attrs["series_id"]),
            choices,
        )
