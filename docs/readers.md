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

AMUSE is a required runtime dependency of `csfdata_analysis`. It must be
available in the active Python environment before importing its reader or
diagnostic modules.
