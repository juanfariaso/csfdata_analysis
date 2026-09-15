# Parameters And Versions

This tutorial uses `lagrangian_radii` to show how to add a standardized output
parameter and how to create a new diagnostic version.

Here, a *parameter* means one derived output dataset, such as `r_l50`. It does
not mean a simulation input in `config.yaml`.

## Add An Output Parameter

Assume we decide to include the radius enclosing 99 percent of stellar mass.
The output name is `r_l99`; its canonical unit remains `pc` in the diagnostic
schema and HDF5 attribute, not in the name.

1. Add `"r_l99": "pc"` to the `outputs` mapping of every relevant
   `DiagnosticChoice`.
2. Update `measure_lagrangian_radii` so every choice returns an AMUSE quantity
   named `r_l99`.
3. Add a focused test for the new value.
4. Update the diagnostic docstring and any manual documentation that lists its
   outputs.
5. Create a new diagnostic version before running it on catalogue data.

The runner requires each choice to return exactly the outputs it declares. It
will fail clearly if the implementation and schema disagree.

## Create Version 2

Do not edit `LAGRANGIAN_RADII_V1` after it has produced catalogue results.
Instead, retain it and add a version-2 definition:

```python
LAGRANGIAN_RADII_V2 = Diagnostic(
    name="lagrangian_radii",
    version=2,
    evaluate=measure_lagrangian_radii_v2,
    choices=(...),
)
```

`measure_lagrangian_radii_v2` can reuse version-1 logic, but it must implement
the new scientific definition, such as the additional `r_l99` output or a new
centre method.

Running version 2 creates a sibling result:

```text
derived/diagnostics/lagrangian_radii/v1/series.h5
derived/diagnostics/lagrangian_radii/v2/series.h5
```

Version 1 stays available for reproducing earlier work. Researchers choose the
version explicitly when comparing or querying results.

## When A New Version Is Required

Create a new version whenever any of these can change scientific interpretation
or stored values:

- Calculation algorithm.
- Output parameter names, units, or meanings.
- Choice name, centre method, or particle selection.
- Output layout.

Refactoring that provably leaves results and the HDF5 schema unchanged can stay
within the existing diagnostic version.

