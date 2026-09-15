# Diagnostics

A diagnostic is a versioned definition of scalar measurements calculated from
one AMUSE particle set. The runner evaluates the same diagnostic for every
snapshot of a simulation and writes one standardized time series.

## Definition

```python
from csfdata_analysis.diagnostics import Diagnostic, DiagnosticChoice


lagrangian_radii = Diagnostic(
    name="lagrangian_radii",
    version=1,
    evaluate=measure_lagrangian_radii,
    choices=(
        DiagnosticChoice(
            name="stellar_com",
            metadata={"method": "mass_weighted_stellar_center_of_mass"},
            outputs={"r_l10": "pc", "r_l50": "pc", "r_l90": "pc"},
        ),
    ),
)
```

The evaluation function receives AMUSE particles and returns scalar AMUSE
quantities with exactly the declared output names:

```python
def measure_lagrangian_radii(particles):
    return {
        "stellar_com": {
            "r_l10": r10,
            "r_l50": r50,
            "r_l90": r90,
        },
    }
```

The runner, rather than the function, provides the model time and snapshot
identity, validates returned values and units, and handles output files,
parallelism, provenance, and progress reporting. Each choice becomes one HDF5
group in the diagnostic time series.

## Versioning

Keep a diagnostic version when a change cannot alter its scientific results.
Create a new version when the calculation, output columns, units, settings
meaning, or parameter definitions change. Existing results are never
overwritten; version 2 writes beside version 1.
