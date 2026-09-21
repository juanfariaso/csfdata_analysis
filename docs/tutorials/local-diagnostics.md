# Running Local Diagnostics

Local diagnostics support analysis in a research project without modifying or
installing code in `csfdata_analysis`. They use the same definitions, result
files, choices, and collection registration as built-in diagnostics.

Only use trusted local directories. Loading a diagnostic imports and executes
its Python file.

## 1. Configure The Lite Catalogue

Create `analysis.yaml` at the root of the local lite catalogue:

```text
lite-catalogue/
  analysis.yaml
  lite.yaml
  collections/
```

```yaml
schema_version: 1

diagnostic_directories:
  - /path/to/project/csfdata_diagnostics
```

`analysis.yaml` is local machine configuration. It is not imported into the
full catalogue and is not scientific metadata. Relative paths are permitted
and are resolved relative to `analysis.yaml`.

## 2. Define Local Diagnostics

Every `.py` file in a configured diagnostic directory must define a non-empty
`DIAGNOSTICS` tuple. The loader stops with an error if a file omits that tuple,
contains an unsupported object, or duplicates an existing diagnostic name and
version.

One module may contain both time-series and scalar diagnostics. For example,
create:

```text
/path/to/project/csfdata_diagnostics/maximum_radius.py
```

```python
from amuse.datamodel import Particles
from csfdata.catalogue import CatalogueSimulation
from csfdata.catalogue.diagnostics import DiagnosticField, DiagnosticRequirement

from csfdata_analysis.diagnostics import (
    EvaluationChoice,
    ScalarDiagnostic,
    TimeSeriesDiagnostic,
)


def measure_maximum_radius(particles: Particles):
    """Measure the largest stellar distance from the coordinate origin.

    Args:
        particles: AMUSE stellar particles for one simulation snapshot.

    Returns:
        One result dictionary for every declared evaluation choice.
    """
    radii = (particles.x**2 + particles.y**2 + particles.z**2).sqrt()
    return {
        "origin": {
            "maximum_radius": radii.max(),
        },
    }


MAXIMUM_RADIUS_V1 = TimeSeriesDiagnostic(
    name="maximum_radius",
    version=1,
    description="Largest stellar distance from the coordinate origin.",
    evaluate=measure_maximum_radius,
    evaluation_choices=(
        EvaluationChoice(
            name="origin",
            metadata={},
            outputs={"maximum_radius": "pc"},
        ),
    ),
    field_descriptions={
        "maximum_radius": "Largest stellar distance from the coordinate origin.",
    },
    choice_names=(),  # Optional: registered scientific choices affecting this result.
    requires=(),  # Optional: completed diagnostic versions required first.
)


def measure_final_maximum_radius(
    simulation: CatalogueSimulation,
    choices: dict[str, str],
) -> dict[str, float]:
    """Read one completed time series and return its final scalar value.

    Args:
        simulation: Catalogue simulation whose derived results are available.
        choices: Concrete registered choices selected by the scalar runner.
            This example has no choice dimensions, so the dictionary is empty.

    Returns:
        The declared scalar output in its canonical unit.
    """
    maximum_radius = simulation.diagnostics.time_series[("maximum_radius", "v1")]
    data = maximum_radius.read(fields=("maximum_radius",))
    return {"final_maximum_radius": float(data["maximum_radius"][-1])}


FINAL_MAXIMUM_RADIUS_V1 = ScalarDiagnostic(
    name="final_maximum_radius",
    version=1,
    description="Final maximum stellar radius from the origin.",
    evaluate=measure_final_maximum_radius,
    fields=(
        DiagnosticField(
            "final_maximum_radius",
            "Maximum stellar radius at the final stored time.",
            "pc",
        ),
    ),
    choice_names=(),  # Optional: registered scientific choices affecting this result.
    requires=(
        DiagnosticRequirement("maximum_radius", "v1"),
    ),  # Optional: completed diagnostics required before this evaluator runs.
)


DIAGNOSTICS = (MAXIMUM_RADIUS_V1, FINAL_MAXIMUM_RADIUS_V1)
```

The individual variable names are only for human readability. The loader reads
`DIAGNOSTICS` and validates its contents. The two evaluator signatures are
intentionally different:

- A `TimeSeriesDiagnostic` evaluator receives `particles` for one snapshot and
  returns all declared evaluation-choice results.
- A `ScalarDiagnostic` evaluator receives `simulation` and `choices`, then
  returns one dictionary of declared scalar fields.

`choice_names` and `requires` are optional definition arguments for both
types. Omit them when the diagnostic has no registered scientific choices or
no diagnostic dependencies. A scalar diagnostic that reads a time-series
result should declare that dependency in `requires`.

## 3. Run Them Across A Collection

Run the time-series diagnostic first from a machine with access to the lite
catalogue and its configured diagnostic directory:

```bash
csfdata analysis compute maximum_radius \
  --catalogue /path/to/lite-catalogue \
  --filter collection=example-collection \
  --workers 4
```

The runner loads the local module in the parent process and in each worker. It
registers the selected diagnostic in the collection's `diagnostics.yaml` before
writing one HDF5 series per simulation:

```text
collections/<collection-id>/simulations/<simulation-id>/
derived/diagnostics/maximum_radius/v1/series.h5
```

Then run the scalar diagnostic. Its declared requirement makes the command
fail clearly if `maximum_radius v1` is incomplete for any selected simulation:

```bash
csfdata analysis compute final_maximum_radius \
  --catalogue /path/to/lite-catalogue \
  --filter collection=example-collection \
  --workers 4
```

The scalar runner writes the standard `derived/scalar_diagnostics.yaml` file.
Once scalar results are available, rebuild the core index to make the default
choice values available to catalogue queries:

```bash
csfdata index-catalogue /path/to/lite-catalogue --collection example-collection
```

## 4. Promote A Diagnostic

Keep diagnostics local while they are exploratory or project-specific. Move a
diagnostic into `csfdata_analysis` when it needs maintained documentation,
shared tests, or stable reuse by other projects. The technical procedure is in
[Adding Diagnostics](../adding-a-diagnostic.md).
