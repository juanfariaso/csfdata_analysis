# Particle-Series Format

## Purpose

A time-series diagnostic may produce derived values for individual particles
as well as snapshot-level scalars. These values are stored beside the scalar
series:

```text
derived/diagnostics/<diagnostic-name>/v<version>/
├── series.h5
└── particles.h5
```

`series.h5` remains the completion marker. It names the sidecar and carries a
`series_id` that must match `particles.h5`. Scalar-only diagnostics retain the
existing one-file format unchanged.

## Long-Table Layout

Each evaluation choice is a column-oriented long table:

```text
particles.h5
└── choices/
    └── <choice>/
        ├── snapshot_id
        ├── particle_key
        └── <declared particle fields>
```

Every dataset in one choice has the same length. One row identifies one
particle in one source snapshot. Snapshot identifiers are repeated so that
snapshots may contain different particles without padding a rectangular
particle-by-time array.

A particle absent from a snapshot has no row for that snapshot. A diagnostic
may reserve a field value such as `-1` for a particle that is present but
unclassified; that meaning must be declared in the field metadata.

## Fields

Every particle field declares:

- a stable name;
- a human-readable description;
- a stored numeric, Boolean, or UTF-8 string dtype;
- an optional canonical AMUSE unit;
- optional semantic metadata, such as the meaning of classification labels.

Particle keys must be non-negative, unique within one snapshot and choice, and
stable wherever a diagnostic intends to follow the same physical particle
through time.

## Safety

Particle rows stream into a temporary HDF5 file while snapshots are evaluated.
The runner publishes `particles.h5` first and `series.h5` last only after both
files are complete and a round-trip check succeeds. Catalogue-derived imports
validate and transfer the two files as one bundle.

Changing particle fields, dtypes, units, label meanings, or particle-selection
rules requires a new diagnostic version.
