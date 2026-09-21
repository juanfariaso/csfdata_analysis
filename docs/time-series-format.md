# Time-Series Format

## Purpose

Each completed time-series diagnostic writes one HDF5 file for one simulation and
one diagnostic version. The file contains only derived scalar measurements and
their provenance. It never duplicates raw particle data.

```text
derived/diagnostics/<diagnostic-name>/v<version>/series.h5
```

This is deliberately the file boundary: one file per simulation, diagnostic,
and version. A collection is never stored in one shared HDF5 file, and a
snapshot never receives its own result file.

## Root Datasets

Every time-series file contains these root datasets, with one row per source
snapshot:

| Dataset | Meaning |
| --- | --- |
| `time` | Physical model time in Myr. |
| `snapshot_id` | Source path relative to the imported simulation's `raw/` directory. |

The root attributes identify the diagnostic, its version, and this HDF5
format's schema version. The current `format_schema_version` is `2`; files
written by the earlier time-series schema must be recomputed before this
interface reads them.

## Choices

Scientifically meaningful methodological choices are stored as named HDF5
groups inside the same time-series file. They are not separate files.

```text
series.h5
├── time
├── snapshot_id
└── choices/
    ├── origin/
    │   ├── centre_x
    │   ├── centre_y
    │   ├── centre_z
    │   ├── r_l01
    │   └── ...
    └── stellar_com/
        ├── centre_x
        ├── centre_y
        ├── centre_z
        ├── r_l01
        └── ...
```

Each choice group records its method as HDF5 attributes. For a centre-based
choice, the calculated centre coordinates are stored for every snapshot. This
makes the location used by each measurement inspectable and reproducible.

For example, the Lagrangian-radii diagnostic can define:

| Choice | Method | Centre coordinates |
| --- | --- | --- |
| `origin` | D-CAF coordinate origin | Always `(0, 0, 0)` pc. |
| `stellar_com` | Mass-weighted stellar centre of mass | Calculated from finite, positive-mass stars with finite positions. |

Choice names and their exact methods are part of the diagnostic version. New
or changed choice definitions require a new diagnostic version.

## Units And Values

Every output dataset has a `unit` HDF5 attribute. The initial canonical units
are `Myr`, `pc`, `Msun`, `km/s`, and the dimensionless unit `1` for counts.
Diagnostics return AMUSE quantities; the runner converts them to their
declared canonical units before writing.

Every choice must return every declared output for every snapshot. If a
measurement is unavailable or incompatible, the diagnostic fails for that
simulation rather than silently creating a partially complete final file.

## Safety And Versioning

- `v1` is immutable once written. Changed results, columns, units, or methods
  require `v2`.
- The runner evaluates all snapshots before writing the final file.
- A temporary HDF5 file is atomically renamed into place only on success.
- Existing completed files are never overwritten.
- Raw data is never changed and can regenerate any derived time series.

### Names And Versions

The diagnostic name and version are part of the catalogue data contract. They
choose the result path, for example `lagrangian_radii/v1/series.h5`. Choice
names and output names are also part of the contract because they choose HDF5
groups and datasets.

Local Python variable names and the evaluator function name do not change the
stored format. A descriptive constant such as `LAGRANGIAN_RADII_V1` is useful
for code readability, but the authoritative identity is
`TimeSeriesDiagnostic(name="lagrangian_radii", version=1, ...)`.

Adding an output parameter after a diagnostic has been used changes its schema
and requires a new diagnostic version.
