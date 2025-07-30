Transition Metal Oxidation State (TMOS)
=======================================
[//]: # (Badges)
[![Documentation Status](https://readthedocs.org/projects/tmos/badge/?version=latest)](https://tmos.readthedocs.io/en/latest/?badge=latest)
[![GitHub Actions Build Status](https://github.com/openforcefield/tmos/workflows/CI/badge.svg)](https://github.com/openforcefield/tmos/actions?query=workflow%3ACI)

**This is a set of informal tools for internal use. We make no guarantees of versioning, functionality, or support.**

This package provides tools for analyzing and determining the oxidation state of transition metal complexes from molecular representations. The core functionality is provided by the [`sanitize_complex`](tmos/sanitize.py#L1) function, which processes a molecular structure and outputs a sanitized version suitable for oxidation state assignment.

Key utilities are available in the [`tmos/utils`](tmos/utils) directory, others include:
- [`mol_to_smiles`](tmos/utils/smiles.py#L1): Converts molecular objects to SMILES strings for easy representation and interoperability.
- [`mol_from_smiles`](tmos/utils/smiles.py#L20): Generates molecular objects from SMILES strings, enabling flexible input formats.

As RDKit Molecule class instances of transition metal complexes must be handled carefully.

Together, these tools enable automated, reproducible workflows for transition metal chemistry analysis.

## Installation

* Step 1: Download the main branch from our GitHub page as a zip file, or clone it to your working directory with:

    ``git clone https://github.com/openforcefield/tmos``

* Step 2 (Optional): If you are using conda and you want to create a new environment for this package you may install with:

    ``conda env create -f requirements.yaml``

* Step 3: Install package with:

    ``pip install tmos/.``

    or change directories and run

    ``pip install .``

    Adding the flag ``-e`` will allow you to make changes that will be functional without reinstallation.

* Step 4: Initialize pre-commits (for developers)

    ``pre-commit install``

### Copyright

Copyright (c) 2025, Jennifer A Clark


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
