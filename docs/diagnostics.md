# Diagnostics

## Published Catalogue Definitions

Before a diagnostic calculation writes results, `csfdata_analysis` ensures the
collection's `diagnostics.yaml` contains its definition, requirements, and
registered scientific choices. This small file is the interface to core
`csfdata`: it declares each available diagnostic version, its result path,
fields, units, and choices. It never duplicates the numerical HDF5 or YAML
results.

For example, the `lagrangian_radii` diagnostic publishes `center` as its
choice, with `stellar_com` as the default. Its concrete HDF5 result groups,
`origin` and `stellar_com`, are evaluation choices: they are the stored values
of the global `center` choice rather than a second independent choice system.
For each centre, it records total stellar mass, the number of valid stars,
mass-weighted Lagrangian radii, and the number of valid stars within each
corresponding radius. Thus `r_l50` is the half-mass radius and `n_l50` is the
number of valid stars within or at that radius.

A diagnostic is a versioned definition of scalar measurements calculated from
one AMUSE particle set. The runner evaluates the same diagnostic for every
snapshot of a simulation and writes one standardized time series.

## Module Layout

Concrete diagnostic implementations are separated by their stored result type:

```text
diagnostics/
  base.py
  time_series/
  scalar/
```

`time_series/` is for diagnostics evaluated at each snapshot and stored in
versioned HDF5 files. `scalar/` is for scalar diagnostics calculated once per
simulation and stored in `derived/scalar_diagnostics.yaml`. Each module declares any
completed diagnostic versions it requires, so dependency chains are visible in
the published collection definition. Every concrete module must also declare a
non-empty `DIAGNOSTICS` tuple containing its supported versions. The analysis
package loads only these explicit tuples and stops with an error if a module is
missing one or contains an object of the wrong diagnostic type.

## Definition

```python
from csfdata_analysis.diagnostics import TimeSeriesDiagnostic, EvaluationChoice


lagrangian_radii = TimeSeriesDiagnostic(
    name="lagrangian_radii",
    version=1,
    description="Mass-weighted stellar Lagrangian radii through time.",
    evaluate=measure_lagrangian_radii,
    evaluation_choices=(
        EvaluationChoice(
            name="stellar_com",
            metadata={"method": "mass_weighted_stellar_center_of_mass"},
            outputs={"r_l10": "pc", "r_l50": "pc", "r_l90": "pc"},
        ),
    ),
    field_descriptions={
        "r_l10": "Radius enclosing 10 percent of stellar mass.",
        "r_l50": "Radius enclosing 50 percent of stellar mass.",
        "r_l90": "Radius enclosing 90 percent of stellar mass.",
    },
    choice_names=("center",),
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
parallelism, provenance, and progress reporting. Each evaluation choice becomes
one HDF5 group in the diagnostic time series.

## Versioning

Keep a diagnostic version when a change cannot alter its scientific results.
Create a new version when the calculation, output columns, units, settings
meaning, or parameter definitions change. Existing results are never
overwritten; version 2 writes beside version 1.
