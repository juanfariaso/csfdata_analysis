# Runner

The runner evaluates one diagnostic over every snapshot of one imported
simulation. It is the safe per-simulation unit that later collection-level
parallel execution will use.

```python
report = compute_time_series(
    simulation_root=simulation_root,
    diagnostic=lagrangian_radii,
    adapter=dcaf_adapter,
    read_snapshot=read_stars,
)
```

The adapter must be initialized with `simulation_root / "raw"`. The runner
uses it to obtain snapshot paths and their model times, while `read_stars`
converts D-CAF output to AMUSE particles.

On success, the runner writes exactly one HDF5 file:

```text
derived/diagnostics/<diagnostic-name>/v<version>/series.h5
```

It contains `time`, `snapshot_id`, and one group per diagnostic choice.
Each choice group contains its declared output datasets and method metadata.
Every output dataset stores its canonical unit as an HDF5 attribute.

The runner never modifies `raw/` or replaces a completed time series. It
writes a temporary file first and atomically renames it only after all
snapshots have been evaluated successfully.

For a lite catalogue, the runner reads snapshots from the recorded full
catalogue source but receives an alternate output path in the lite catalogue.
The resulting HDF5 file records the collection ID, simulation ID, canonical
configuration hash, and a completion flag so it can later be imported safely.
