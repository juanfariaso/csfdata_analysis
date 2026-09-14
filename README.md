# CSFData Analysis

`csfdata_analysis` is an optional, dependent package for producing derived
analysis products from catalogues managed by
[CSFData](../csfdata/README.md).

The core `csfdata` package remains responsible for importing, validating, and
preserving raw simulation data. This package will read an imported simulation
and write reproducible, versioned products under its `derived/` directory.

AMUSE is a required runtime dependency, provided by the `amuse-framework`
package.

## Development

Install the backbone package first, then install this repository in editable
mode:

```bash
cd /path/to/csfdata
pip install -e .

cd /path/to/csfdata_analysis
pip install -e .
```

The design boundary and planned workflow are described in
[docs/architecture.md](docs/architecture.md).
