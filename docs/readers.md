# Readers

Readers convert code-specific raw output into standard inputs for diagnostics.

## D-CAF Stellar Snapshots

`read_stars` reads one D-CAF `stars_*.amuse` file and returns AMUSE
`Particles`. Diagnostics then operate on those particles without knowing the
source file format.

```python
from pathlib import Path

from csfdata_analysis.readers import read_stars

particles = read_stars(Path("raw/dcaf_output/stars_0001.amuse"))
```

## Stellar Ages

[`read_stars_with_ages`](reference/csfdata_analysis/readers/index.md#csfdata_analysis.readers.read_stars_with_ages)
is the public D-CAF reader for age-dependent analysis. It derives the simulation
root and snapshot time from the snapshot path, constructs the D-CAF adapter,
and returns particles with the standard `age` attribute in Myr:

```python
from pathlib import Path

from csfdata_analysis.readers import read_stars_with_ages

particles = read_stars_with_ages(
    Path("raw/dcaf_output/stars_0001.amuse")
)
```

Internally,
[`read_snapshot_with_ages`](reference/csfdata_analysis/readers/index.md#csfdata_analysis.readers.read_snapshot_with_ages)
combines a format-specific adapter and snapshot reader. Collection runners use
this generic bridge so other simulation formats can provide their own formation
times without placing AMUSE inside the core `csfdata` package.

A time-series diagnostic requests this automatically by declaring:

```python
DIAGNOSTIC = TimeSeriesDiagnostic(
    ...,
    requires_stellar_ages=True,
)
```

An adapter that cannot recover formation times raises `NotImplementedError`;
ages are never silently guessed.

### D-CAF Formation Times

`csfdata.adapters.dcaf.DcafAdapter.stellar_formation_times` reads records
written by D-CAF in this form:

```text
[NEW_STAR] <formation-time> <particle-key> ...
```

The adapter scans `dcaf.log` in every numbered `dcaf_output*` segment and
returns `{particle_key: formation_time}` in Myr. It streams each log once and
caches the mapping for subsequent snapshots. Files named `dcaf.log.failed` are
ignored because they represent incomplete attempts rather than accepted run
history.

The reader matches this mapping against the AMUSE keys in each snapshot. It
raises a clear error when a key is missing, when duplicate records disagree,
or when a recorded formation time is later than the snapshot. Formation times
are logged to six decimal places, so negative age differences smaller than
`1e-6` Myr are treated as rounding and set to zero.

### Validate One D-CAF Simulation

The following check reads one snapshot and verifies that every star has an
unambiguous formation record.

```python
from pathlib import Path

from amuse.units import units
from csfdata_analysis.readers import read_stars_with_ages

snapshot = Path("/path/to/simulation/raw/dcaf_output/stars_0001.amuse")
stars = read_stars_with_ages(snapshot)

print(f"Snapshot stars: {len(stars)}")
print(
    "Age range: "
    f"{stars.age.min().value_in(units.Myr):g} to "
    f"{stars.age.max().value_in(units.Myr):g} Myr"
)
```

Successful completion confirms that the final snapshot keys are fully covered.
It does not prove that earlier snapshots are covered; diagnostics requesting
ages perform the same validation independently for every snapshot they read.

AMUSE is a required runtime dependency of `csfdata_analysis`. It must be
available in the active Python environment before importing its reader or
diagnostic modules.
