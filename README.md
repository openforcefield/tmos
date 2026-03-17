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

### Requirements

Python ≥ 3.10 is required. Core dependencies (`rdkit`, `openbabel`, `networkx`, `qcelemental`, `periodictable`, `numpy`, `loguru`) are declared in `pyproject.toml` and installed automatically by `pip`.

> **Note**: `rdkit` and `openbabel` have compiled C/C++ extensions. Installing via **conda/mamba is strongly recommended** to avoid build issues. Pure-`pip` installs may work on some platforms but are not guaranteed.

### Option A — conda / micromamba (recommended)

Clone the repository and create the bundled environment (includes all optional dependencies):

```bash
git clone https://github.com/openforcefield/tmos
cd tmos
micromamba env create -f requirements.yaml   # or: conda env create -f requirements.yaml
micromamba activate tmos                     # or: conda activate tmos
pip install -e .
```

### Option B — pip only

```bash
git clone https://github.com/openforcefield/tmos
cd tmos
pip install .
```

Add `-e` for an editable / development install.

### Optional dependencies

Several features require extra packages that are *not* installed by default:

| Feature | Extra | Install command |
|---|---|---|
| Geometry (pymatgen, posym) | `geometry` | `pip install "tmos[geometry]"` |
| 3-D visualization (py3Dmol) | `viz` | `pip install "tmos[viz]"` |
| Documentation build | `docs` | `pip install "tmos[docs]"` |
| Tests | `test` | `pip install "tmos[test]"` |

Install multiple extras at once: `pip install "tmos[geometry,viz,test]"`.

**rylm** is required for `geometry mode="rylm"` and is not on PyPI. Install it directly from GitHub:

```bash
pip install git+https://github.com/chrisiacovella/rylm.git
```

If an optional dependency is absent, the relevant function raises a clear `ImportError` with the exact install command.

### Developer setup

```bash
pip install "tmos[geometry,viz,test]"
pip install git+https://github.com/chrisiacovella/rylm.git
pre-commit install
```

### Copyright

Copyright (c) 2025, Jennifer A Clark


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
