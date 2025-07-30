Transition Metal Oxidation State
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/openforcefield/tmos/workflows/CI/badge.svg)](https://github.com/openforcefield/tmos/actions?query=workflow%3ACI)


This package provides tools for analyzing and determining the oxidation state of transition metal complexes from molecular representations. The core functionality is provided by the [`sanitize_complex`](tmos/sanitize.py#L1) function, which processes a molecular structure and outputs a sanitized version suitable for oxidation state assignment.

Key utilities in the [`tmos/utils`](tmos/utils) directory include:
- [`mol_to_smiles`](tmos/utils/smiles.py#L1): Converts molecular objects to SMILES strings for easy representation and interoperability.
- [`mol_from_smiles`](tmos/utils/smiles.py#L20): Generates molecular objects from SMILES strings, enabling flexible input formats.
- Additional helper functions for molecular manipulation and data handling.

Together, these tools enable automated, reproducible workflows for transition metal chemistry analysis.

### Copyright

Copyright (c) 2025, Jennifer A Clark


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
