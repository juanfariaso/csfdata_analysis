# Adding A Diagnostic

This guide describes the standard workflow for adding a snapshot-based,
time-evolution diagnostic to `csfdata_analysis`.

## 1. Define The Scientific Contract

Before writing code, decide and document:

- The AMUSE particle attributes the calculation requires.
- The scalar quantities it returns and their canonical units.
- Every scientifically meaningful choice, such as a centre definition.
- The exact method and particle selection for each choice.
- The diagnostic version, starting at `1`.

Do not put model time, source paths, HDF5 handling, parallelism, or progress
reporting in the scientific function. The runner owns those responsibilities.

## 2. Create One Diagnostic Module

Create a module below `csfdata_analysis/diagnostics/`, for example:

```text
csfdata_analysis/diagnostics/lagrangian_radii.py
```

It contains two public items:

- One function that evaluates all choices for one AMUSE `Particles` set.
- One versioned `Diagnostic` instance that declares the choices and schemas.

## 3. Write The Measurement Function

The function receives AMUSE particles and returns AMUSE scalar quantities
grouped by choice name:

```python
def measure_example(particles: Particles) -> dict[str, dict[str, Quantity]]:
    return {
        "choice_name": {
            "measurement": value | units.pc,
        },
    }
```

Returned choice names and output names must exactly match the `Diagnostic`
definition. Use AMUSE quantities, not bare floats, so the runner can validate
and convert each value to its declared canonical unit.

## 4. Declare The Versioned Schema

```python
EXAMPLE_V1 = Diagnostic(
    name="example",
    version=1,
    evaluate=measure_example,
    choices=(
        DiagnosticChoice(
            name="choice_name",
            metadata={"method": "documented_method"},
            outputs={"measurement": "pc"},
        ),
    ),
)
```

`metadata` becomes HDF5 attributes on the choice group. Record enough detail
there to identify the scientific method and particle selection.

## 5. Test The Calculation

Every diagnostic registered in `DIAGNOSTICS` is automatically run by the
generic readiness test against a standard temporary AMUSE particle snapshot.
It checks that the diagnostic can read representative stellar data and write
every declared output through the normal HDF5 runner:

```bash
csfdata analysis test-diagnostics
```

Run this test whenever a diagnostic is added or its descriptor changes. A pass
means the function is technically ready for collection execution: its particle
requirements are satisfied by the standard test snapshot, its output matches
its declared schema, and the runner can write it successfully.

This general test does not independently prove a scientific formula is correct.
Add a focused numerical test later when a diagnostic needs a scientific
regression check.

## 6. Run And Inspect One Simulation

For a direct Python inspection, import the diagnostic and run it explicitly:

```python
from csfdata.adapters.dcaf import DcafAdapter
from csfdata_analysis.diagnostics.lagrangian_radii import LAGRANGIAN_RADII_V1
from csfdata_analysis.readers import read_stars
from csfdata_analysis.runner import compute_time_series

simulation_root = ...
report = compute_time_series(
    simulation_root,
    LAGRANGIAN_RADII_V1,
    DcafAdapter(simulation_root / "raw"),
    read_stars,
)
```

Inspect the generated `series.h5`, especially its choice metadata, centre
coordinates, units, and time sequence. A completed output is never overwritten
by a repeat run.


## 7. Version Changes

Keep the version only when a change cannot alter scientific results. Create a
new version when a calculation, choice definition, particle selection, output
name, unit, or output layout changes. Version 2 writes to `v2/` beside the
existing `v1/` result.
