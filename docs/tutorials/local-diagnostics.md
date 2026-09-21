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

## 2. Define A Local Time-Series Diagnostic

Every `.py` file in a configured diagnostic directory must define a non-empty
`DIAGNOSTICS` tuple. The loader stops with an error if a file omits that tuple,
contains an unsupported object, or duplicates an existing diagnostic name and
version.

For example, create:

```text
/path/to/project/csfdata_diagnostics/maximum_radius.py
```

```python
from amuse.datamodel import Particles
from amuse.units import units

from csfdata_analysis.diagnostics import EvaluationChoice, TimeSeriesDiagnostic


def measure_maximum_radius(particles: Particles):
    """Measure the largest stellar distance from the coordinate origin."""
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
            metadata={"center": "origin"},
            outputs={"maximum_radius": "pc"},
        ),
    ),
    field_descriptions={
        "maximum_radius": "Largest stellar distance from the coordinate origin.",
    },
)

DIAGNOSTICS = (MAXIMUM_RADIUS_V1,)
```

The individual variable name `MAXIMUM_RADIUS_V1` is only for human readability.
The loader reads `DIAGNOSTICS` and validates its contents.

## 3. Run It Across A Collection

Run the standard collection command from a machine with access to the lite
catalogue and its configured diagnostic directory:

```bash
csfdata analysis compute maximum_radius \
  --catalogue /path/to/lite-catalogue \
  --filter collection=example-collection \
  --workers 4
```

The runner loads the local module in the parent process and in each worker. It
registers the selected diagnostic in the collection's `diagnostics.yaml` before
writing any results. Results use the standard location:

```text
collections/<collection-id>/simulations/<simulation-id>/
derived/diagnostics/maximum_radius/v1/series.h5
```

## 4. Local Scalar Diagnostics

Local scalar diagnostics use the same `DIAGNOSTICS` convention. Their evaluator
receives the imported simulation path and one concrete choice mapping, then
returns the declared scalar fields. The generic scalar runner writes the
standard `derived/scalar_diagnostics.yaml` format and automatically reads the
same local `analysis.yaml` configuration. Run it with the same collection
command:

```bash
csfdata analysis compute maximum_radius \
  --catalogue /path/to/lite-catalogue \
  --workers 4
```

Once scalar results are available, rebuild the core index to make the default
choice values available to catalogue queries:

```bash
csfdata index-catalogue /path/to/lite-catalogue --collection example-collection
```

## 5. Promote A Diagnostic

Keep diagnostics local while they are exploratory or project-specific. Move a
diagnostic into `csfdata_analysis` when it needs maintained documentation,
shared tests, or stable reuse by other projects. The technical procedure is in
[Adding Diagnostics](../adding-a-diagnostic.md).
