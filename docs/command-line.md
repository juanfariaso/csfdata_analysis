# Command Line

## Catalogue Time-Series Diagnostic

Run a standard diagnostic for every indexed simulation selected by filters:

```bash
csfdata analysis compute lagrangian_radii \
  --catalogue /path/to/catalogue \
  --filter collection=dcaf-grid-v1 \
  --workers 12
```

The catalogue must first have a current `registry.sqlite`, built with
`csfdata index-catalogue`. The runner uses one worker per simulation, writes
each successful result to that simulation's versioned `derived/` directory,
and skips a completed `series.h5` file by default. Re-running the command
therefore retries only missing or failed work.

Before starting, the command lists each selected collection and its number of
selected simulations, then asks for confirmation. Omit `--filter` to select
every indexed collection. In a non-interactive batch job, use `--no-prompt`:

```bash
csfdata analysis compute lagrangian_radii \
  --catalogue /path/to/catalogue \
  --workers 12 \
  --no-prompt
```

Use repeated filters for a subset. Numeric ranges use inclusive `LOWER:UPPER`
syntax:

```bash
csfdata analysis compute lagrangian_radii \
  --catalogue /path/to/catalogue \
  --filter collection=dcaf-grid-v1 \
  --filter tff=0.5:3.0 \
  --filter sfe=0.3 \
  --workers 4 \
  --dry-run
```

`--dry-run` checks the snapshot plan but does not read AMUSE particles or
write analysis files. The parent process prints one final result per
simulation, followed by complete, ready, skipped, and failed totals.

To replace an earlier result with the same diagnostic version, use
`--overwrite`. Every replacement is first written to a temporary HDF5 file and
atomically installed only after the full calculation succeeds:

```bash
csfdata analysis compute lagrangian_radii \
  --catalogue /path/to/lite-catalogue \
  --filter collection=dcaf-grid-v1 \
  --workers 12 \
  --overwrite
```

## Clear Derived Results

Remove a time-series diagnostic without recomputing it immediately:

```bash
csfdata analysis clear-derived lagrangian_radii \
  --catalogue /path/to/lite-catalogue \
  --filter collection=dcaf-grid-v1
```

This command is intentionally restricted to lite catalogues. It removes only
the selected diagnostic's versioned `series.h5` file for matching simulations;
other diagnostics and all raw data are preserved. The command shows the number
of selected simulations and asks for confirmation. Use `--no-prompt` only in a
non-interactive job after checking the selection.

## Lite Catalogue Time-Series Diagnostic

For a lite catalogue created by `csfdata import-lite`, omit the collection
filter. The lite catalogue contains exactly one collection and records the
full catalogue that owns its raw snapshots:

```bash
csfdata analysis compute lagrangian_radii \
  --catalogue /path/to/lite-catalogue \
  --workers 12
```

The command checks that the recorded source collection, simulation metadata,
and canonical configuration still match before reading each raw snapshot. It
writes results only inside the lite catalogue.

## Import Derived Results

After computation, run this from a node that can write to the full catalogue:

```bash
csfdata analysis import-derived /path/to/lite-catalogue \
  --catalogue /path/to/source-catalogue
```

Only completed HDF5 files with matching collection, simulation, and
configuration identities are copied. Existing destination files are skipped by
default. Use `--dry-run` to inspect actions, or `--overwrite` to explicitly
replace existing compatible results.

## One Simulation

Use the direct command while developing or debugging one imported simulation:

```bash
csfdata analysis compute-simulation \
  /path/to/catalogue/collections/dcaf-grid-v1/simulations/0001 \
  lagrangian_radii \
  --dry-run
```

```bash
csfdata-analysis diagnostics
csfdata-analysis compute SIMULATION_ROOT lagrangian_radii --dry-run
csfdata-analysis compute SIMULATION_ROOT lagrangian_radii
```

`SIMULATION_ROOT` contains `raw/` and `derived/`. A dry run validates
snapshots, times, and the output path without reading particles or writing
files. The compute command writes the default diagnostic version and does not
replace existing results.
