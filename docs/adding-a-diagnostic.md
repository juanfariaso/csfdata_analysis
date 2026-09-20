# Adding A Time-Series Diagnostic

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

## 2. Create One Time-Series Diagnostic Module

Create a module below `csfdata_analysis/diagnostics/`, for example:

```text
csfdata_analysis/diagnostics/time_series/lagrangian_radii.py
```

It contains two public items:

- One function that evaluates all choices for one AMUSE `Particles` set.
- One versioned `TimeSeriesDiagnostic` instance that declares choices and schemas.

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

Returned choice names and output names must exactly match the `TimeSeriesDiagnostic`
definition. Use AMUSE quantities, not bare floats, so the runner can validate
and convert each value to its declared canonical unit.

## 4. Declare The Versioned Schema

```python
EXAMPLE_V1 = TimeSeriesDiagnostic(
    name="example",
    version=1,
    description="Scientific purpose of this diagnostic.",
    evaluate=measure_example,
    evaluation_choices=(
        EvaluationChoice(
            name="choice_name",
            metadata={"method": "documented_method"},
            outputs={"measurement": "pc"},
        ),
    ),
    field_descriptions={"measurement": "Meaning of the measured value."},
    choice_names=("center",),
)
```

`metadata` becomes HDF5 attributes on the evaluation group. Record enough detail
there to identify the scientific method and particle selection.

`choice_names` declares the global scientific choices that affect the result.
Their allowed values and defaults come from `csfdata_analysis.choices` and are
published to the collection's `diagnostics.yaml` after a successful run.

## 5. Test The Calculation

Create a focused test file for the diagnostic, such as
`tests/test_example.py`. Define only the AMUSE particles required by that
calculation, then assert its scientifically meaningful outputs and units. This
keeps the test data and expected results beside the diagnostic they verify,
instead of forcing every diagnostic through one artificial generic snapshot.

Run the diagnostic test directly while developing:

```bash
python -m pytest tests/test_example.py
```

## 6. Run And Inspect One Simulation

For a direct Python inspection, import the diagnostic and run it explicitly:

```python
from csfdata.adapters.dcaf import DcafAdapter
from csfdata_analysis.diagnostics.time_series.lagrangian_radii import (
    LAGRANGIAN_RADII_V1,
)
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
