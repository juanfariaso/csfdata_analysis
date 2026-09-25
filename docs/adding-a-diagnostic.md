# Adding Diagnostics

`csfdata_analysis` supports two standardized diagnostic types:

- A time-series diagnostic evaluates every snapshot and writes one versioned
  HDF5 series per simulation through
  [`TimeSeriesDiagnostic`](reference/csfdata_analysis/diagnostics/base.md#csfdata_analysis.diagnostics.base.TimeSeriesDiagnostic).
- A scalar diagnostic reads completed products and writes versioned
  simulation-level values through
  [`ScalarDiagnostic`](reference/csfdata_analysis/diagnostics/base.md#csfdata_analysis.diagnostics.base.ScalarDiagnostic).

Standardized diagnostics are versioned, published to the collection, stored in
the catalogue, and available to the normal SQL query interface. A local custom
function is appropriate for exploratory work that does not need those
guarantees.

## Time-Series Diagnostics

### 1. Define The Contract

Define the required AMUSE particle attributes, output fields and canonical
units, scientific choices, method, particle selection, and initial version.
The evaluator does not manage paths, model times, HDF5 files, parallelism, or
progress reporting.

<details>
<summary>Lagrangian-radii implementation</summary>

Location:

```text
csfdata_analysis/diagnostics/time_series/lagrangian_radii.py
```

The `lagrangian_radii v1` contract requires stellar particles with finite
positions and positive finite masses. It calculates the `origin` and
`stellar_com` groups and stores centre coordinates, total stellar mass, valid
star count, Lagrangian radii, and enclosed star counts.

</details>

### 2. Create The Diagnostic Module

Place a time-series diagnostic below
`csfdata_analysis/diagnostics/time_series/`. The module contains an evaluator
for one AMUSE `Particles` object, one or more versioned diagnostic definitions,
and a required `DIAGNOSTICS` tuple. The loader reads only that tuple.

<details>
<summary>Lagrangian-radii implementation</summary>

Location:

```text
csfdata_analysis/diagnostics/time_series/lagrangian_radii.py
```

```python
def measure_lagrangian_radii(particles: Particles) -> dict[str, dict[str, Quantity]]:
    ...


LAGRANGIAN_RADII_V1 = TimeSeriesDiagnostic(...)

DIAGNOSTICS = (LAGRANGIAN_RADII_V1,)
```

</details>

### 3. Implement The Evaluator

The evaluator receives one AMUSE `Particles` object and returns AMUSE scalar
quantities grouped by evaluation choice. Returned group and field names must
match the corresponding
[`TimeSeriesDiagnostic`](reference/csfdata_analysis/diagnostics/base.md#csfdata_analysis.diagnostics.base.TimeSeriesDiagnostic)
definition exactly.

<details>
<summary>Lagrangian-radii implementation</summary>

Location:

```text
csfdata_analysis/diagnostics/time_series/lagrangian_radii.py
```

```python
return {
    "origin": {
        "r_l50": radius | units.pc,
        "n_l50": count | units.none,
        "stellar_mass": total_mass,
    },
    "stellar_com": {
        "r_l50": radius | units.pc,
        "n_l50": count | units.none,
        "stellar_mass": total_mass,
    },
}
```

</details>

### 4. Declare The Schema

Declare a versioned
[`TimeSeriesDiagnostic`](reference/csfdata_analysis/diagnostics/base.md#csfdata_analysis.diagnostics.base.TimeSeriesDiagnostic)
with one
[`EvaluationChoice`](reference/csfdata_analysis/diagnostics/base.md#csfdata_analysis.diagnostics.base.EvaluationChoice)
per stored evaluation group. The `metadata` dictionary becomes HDF5 attributes.
`choice_names` references registered global choices from
[`csfdata_analysis.choices`](reference/csfdata_analysis/choices.md).

<details>
<summary>Lagrangian-radii implementation</summary>

Location:

```text
csfdata_analysis/diagnostics/time_series/lagrangian_radii.py
```

```python
LAGRANGIAN_RADII_V1 = TimeSeriesDiagnostic(
    name="lagrangian_radii",
    version=1,
    description="Stellar Lagrangian radii and enclosed star counts.",
    evaluate=measure_lagrangian_radii,
    evaluation_choices=(
        EvaluationChoice("origin", {"center": "origin"}, LAGRANGIAN_OUTPUTS),
        EvaluationChoice("stellar_com", {"center": "stellar_com"}, LAGRANGIAN_OUTPUTS),
    ),
    field_descriptions={
        "centre_x": "X coordinate of the selected stellar centre.",
        "centre_y": "Y coordinate of the selected stellar centre.",
        "centre_z": "Z coordinate of the selected stellar centre.",
        "stellar_mass": "Total mass of valid stellar particles.",
        "n_stars": "Total number of valid stellar particles.",
        **{
            f"r_{suffix}": f"Radius enclosing {fraction:.0%} of stellar mass."
            for suffix, fraction in LAGRANGIAN_FRACTIONS
        },
        **{
            f"n_{suffix}": f"Number of valid stellar particles within or at r_{suffix}."
            for suffix, _ in LAGRANGIAN_FRACTIONS
        },
    },
    choice_names=("center",),
)

DIAGNOSTICS = (LAGRANGIAN_RADII_V1,)
```

</details>

### 5. Test The Evaluator

Create a focused test containing only the AMUSE particles required by the
diagnostic. Assert the scientific outputs and units.

<details>
<summary>Lagrangian-radii implementation</summary>

Location:

```text
tests/test_lagrangian_radii.py
```

```bash
python -m pytest tests/test_lagrangian_radii.py
```

The test verifies both centre groups and checks `r_l50`, `stellar_mass`,
`n_stars`, and `n_l50` for four equal-mass particles.

</details>

### 6. Run The Diagnostic

[`compute_time_series`](reference/csfdata_analysis/runner.md#csfdata_analysis.runner.compute_time_series)
writes one completed `series.h5` file for a simulation. The collection command
runs the same diagnostic across a selected collection.

<details>
<summary>Lagrangian-radii implementation</summary>

```bash
csfdata analysis compute lagrangian_radii \
  --catalogue /path/to/catalogue \
  --filter collection=example-collection \
  --workers 4
```

Output location:

```text
collections/<collection-id>/simulations/<simulation-id>/
derived/diagnostics/lagrangian_radii/v1/series.h5
```

</details>

### 7. Version Changes

Create a new version when the method, particle selection, choice definition,
field name, unit, or output layout changes. A new version is stored beside the
older output as `v2/`, `v3/`, and so on.

<details>
<summary>Lagrangian-radii implementation</summary>

Location of the next incompatible version:

```text
csfdata_analysis/diagnostics/time_series/lagrangian_radii.py
```

```python
LAGRANGIAN_RADII_V2 = TimeSeriesDiagnostic(
    name="lagrangian_radii",
    version=2,
    ...
)
```

</details>

## Scalar Diagnostics

Scalar diagnostics read completed diagnostic products and write compact values
to `derived/scalar_diagnostics.yaml`. Their default-choice fields are indexed
with simulation configuration parameters and can be used in the same query.
Their evaluator signature is:

```python
def evaluate(
    simulation: CatalogueSimulation,
    choices: dict[str, str],
) -> dict[str, float | int]:
    ...
```

The simulation object provides the lazy `simulation.diagnostics` interface for
reading required time-series and scalar products without constructing internal
paths.

### 1. Define Inputs, Choices, And Fields

Declare every required product, every scientific choice, and every scalar
field with a canonical unit. A scalar result is written separately for each
choice combination.

<details>
<summary>Expansion-rate implementation</summary>

Location:

```text
csfdata_analysis/diagnostics/scalar/expansion_rate.py
```

`expansion_rate v1` reads
`derived/diagnostics/lagrangian_radii/v1/series.h5`. It evaluates `center`
and `lagrangian_radius` choices. Its fields are `dRdt`, `fit_t0`, `fit_r0`,
`r_min`, `r2`, `nrmse_iqr`, `n_points`, and `final_time`.

</details>

### 2. Declare The Definition And Evaluator

Declare a versioned
[`ScalarDiagnostic`](reference/csfdata_analysis/diagnostics/base.md#csfdata_analysis.diagnostics.base.ScalarDiagnostic)
with fields, choices, and required diagnostic versions. The module must expose
every supported version through its `DIAGNOSTICS` tuple.

<details>
<summary>Expansion-rate implementation</summary>

Location:

```text
csfdata_analysis/diagnostics/scalar/expansion_rate.py
```

```python
EXPANSION_RATE_V1 = ScalarDiagnostic(
    name="expansion_rate",
    version=1,
    description="Late-time linear expansion rates fitted to Lagrangian radii.",
    evaluate=compute_expansion_rate,
    fields=(
        DiagnosticField("dRdt", "Fitted late-time radius change.", "km/s"),
        DiagnosticField("fit_t0", "Selected fit reference time.", "Myr"),
        DiagnosticField("fit_r0", "Fitted radius at the reference time.", "pc"),
        DiagnosticField("r_min", "Radius at the fit start.", "pc"),
        DiagnosticField("r2", "Coefficient of determination.", "1"),
        DiagnosticField("nrmse_iqr", "Normalized fit residual.", "1"),
        DiagnosticField("n_points", "Snapshots included in the fit.", "1"),
        DiagnosticField("final_time", "Final time included in the fit.", "Myr"),
    ),
    choice_names=("center", "lagrangian_radius"),
    requires=(DiagnosticRequirement("lagrangian_radii", "v1"),),
)

DIAGNOSTICS = (EXPANSION_RATE_V1,)
```

</details>

### 3. Automatic Collection Registration

The generic scalar runner automatically registers the requested diagnostic,
its required diagnostic versions, and its registered choices in the
collection's `diagnostics.yaml`. Existing matching definitions are preserved;
conflicting name/version definitions stop the operation.

<details>
<summary>Expansion-rate implementation</summary>

```text
csfdata_analysis/runner.py
```

```python
compute_scalar_diagnostic(
    simulation,
    EXPANSION_RATE_V1,
)
```

</details>

### 4. Compute And Write Results

The generic scalar runner enumerates every declared choice combination, calls
the evaluator, validates its values, and writes results to
`derived/scalar_diagnostics.yaml`. Existing results are skipped unless
explicitly overwritten.

<details>
<summary>Expansion-rate implementation</summary>

```text
csfdata_analysis/runner.py
```

```python
results = compute_scalar_diagnostic(
    simulation,
    EXPANSION_RATE_V1,
)
```

</details>

### 5. Reindex And Query

Rebuild the SQL index after writing scalar results. Default-choice scalar
fields are then available with configuration parameters.

<details>
<summary>Expansion-rate implementation</summary>

```bash
csfdata index-catalogue /path/to/catalogue --collection <collection-id>
```

```python
find_simulations(
    "/path/to/catalogue",
    collection_id="example-collection",
    filters={"tff": 1.0, "dRdt": (0.1, None)},
)
```

</details>

## Local Custom Functions

Local custom functions are ordinary Python functions stored outside the
`csfdata_analysis` package, for example in a research project's
`analysis/custom_diagnostics.py`. They are not registered, versioned,
published, indexed, or written into the catalogue automatically.

### 1. Define The Function

The function accepts an AMUSE `Particles` object and returns the values needed
by the local analysis.

<details>
<summary>Local custom-function implementation</summary>

Location:

```text
analysis/custom_diagnostics.py
```

```python
from amuse.datamodel import Particles
from amuse.units import units


def maximum_radius(particles: Particles):
    """Return the largest stellar distance from the origin."""
    radii = (particles.x**2 + particles.y**2 + particles.z**2).sqrt()
    return radii.max().value_in(units.pc)
```

</details>

### 2. Select And Load Snapshots

Use the catalogue data interface to select simulations and a common snapshot
slice. [`read_stars`](reference/csfdata_analysis/readers/dcaf.md#csfdata_analysis.readers.dcaf.read_stars)
loads each selected D-CAF stellar snapshot into AMUSE particles.

<details>
<summary>Local custom-function implementation</summary>

Location:

```text
analysis/run_custom_diagnostic.py
```

```python
from pathlib import Path

from csfdata_analysis.datamodel import load_simulations, select_snapshot_slice
from csfdata_analysis.readers import read_stars

from analysis.custom_diagnostics import maximum_radius


simulations = load_simulations(
    "/path/to/catalogue",
    collection_id="example-collection",
    filters={"tff": 1.0},
)
slices = select_snapshot_slice(simulations, time=10.0)

rows = []
for snapshot in slices.itertuples():
    particles = read_stars(Path(snapshot.snapshot_path))
    rows.append(
        {
            "simulation_id": snapshot.simulation_id,
            "time": snapshot.snapshot_time,
            "maximum_radius": maximum_radius(particles),
        }
    )
```

</details>

Store and plot local results in the research project. Promote a local function
to a standardized diagnostic when it requires versioned catalogue storage,
reproducible choices, or catalogue-wide querying.
