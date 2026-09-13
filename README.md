# CSFData Analysis

`csfdata_analysis` is an optional, dependent package for producing derived
analysis products from catalogues managed by
[CSFData](../csfdata/README.md).

The core `csfdata` package remains responsible for importing, validating, and
preserving raw simulation data. This package will read an imported simulation
and write reproducible, versioned products under its `derived/` directory.

Analysis dependencies such as AMUSE are intentionally not installed yet. They
will be added only when we implement an analysis product that requires them.

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
