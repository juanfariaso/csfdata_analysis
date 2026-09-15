# CSFData Analysis

CSFData Analysis is the optional, science-dependency-aware layer for deriving
reproducible products from catalogues managed by CSFData.

## Architecture

The analysis package creates versioned derived products without changing raw
simulation data. It is installed separately because it may require scientific
dependencies such as AMUSE and Pandas.

`csfdata` owns:

- Catalogue structure, collection metadata, validation, and import.
- The untouched `raw/` simulation data.
- Canonical simulation configuration and fast catalogue queries.

`csfdata_analysis` owns:

- Format-specific readers for raw simulation output.
- Versioned standardized diagnostics and their HDF5 schemas.
- Parallel diagnostic computation and safe derived-data import.
- Research-facing data interfaces for time series and snapshot selections.

Derived products live below each simulation's `derived/` directory. They
record their diagnostic name, version, choices, units, simulation identity,
and configuration fingerprint. Raw files are never modified.

## Documentation

- [CSFData Catalogue Documentation](https://juanfariaso.github.io/csfdata/)
  describes collections, validation, import, indexing, and lite catalogues.
- [Data Interfaces](data.md) describes loading time series into Pandas and
  selecting normalized snapshot slices.
- [Diagnostics](diagnostics.md) describes standardized analysis products.
- [Command Line](command-line.md) documents collection-wide computation and
  safe derived-data import.
