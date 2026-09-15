# Command Line

## Test Diagnostics

Run the built-in readiness check for every registered diagnostic:

```bash
csfdata analysis test-diagnostics
```

It creates one temporary representative AMUSE stellar snapshot, runs every
registered diagnostic through the normal reader and HDF5 writer, and checks the
declared outputs and versioned destination. It prints one `OK` or `FAILED` line
per diagnostic and exits with a failure status when any diagnostic fails.

## Catalogue Diagnostic

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
and never replaces a completed `series.h5` file. Re-running the command skips
completed simulations and retries only missing or failed work.

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

## Lite Catalogue Diagnostic

For a lite catalogue created by `csfdata export-lite`, omit the collection
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
