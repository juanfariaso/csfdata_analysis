# Architecture

## Purpose

This package is the optional analysis layer for CSFData catalogues. It creates
derived scientific products without changing the imported raw simulation data.

## Boundary With CSFData

`csfdata` owns:

- Catalogue structure and collection metadata.
- Simulation validation and import.
- The untouched `raw/` simulation data.
- The public metadata and configuration conventions used by add-ons.

`csfdata_analysis` will own:

- Code-specific readers needed to interpret raw simulation output.
- Standard derived products shared across simulation codes where possible.
- Product provenance, schemas, and analysis versions.
- Optional science dependencies such as AMUSE.

## Product Rule

Every generated product will live below the imported simulation's `derived/`
directory. It will record its product name, schema version, analysis settings,
input provenance, and creation time. Raw files are never modified.

The first analysis product, its data columns, on-disk format, and required
dependencies remain to be decided before implementation.

## Plug-In Registration

This package registers itself as `csfdata_analysis` in the `csfdata.analysis`
Python entry-point group. The lightweight backbone discovers installed add-ons
through `from csfdata import analysis`; it does not import their optional
science dependencies unless an add-on is explicitly loaded.
