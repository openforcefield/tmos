Transition Metal Oxidation State (TMOS)
=======================================
[//]: # (Badges)
[![Documentation Status](https://readthedocs.org/projects/tmos/badge/?version=latest)](https://tmos.readthedocs.io/en/latest/?badge=latest)
[![GitHub Actions Build Status](https://github.com/openforcefield/tmos/workflows/CI/badge.svg)](https://github.com/openforcefield/tmos/actions?query=workflow%3ACI)

**This is a set of informal tools for internal use. We make no guarantees of versioning, functionality, or support.**

`tmos` predicts the oxidation state, formal charge, and d-electron count of a
transition metal complex from its 3-D structure.  Starting from atomic symbols
and Cartesian coordinates it builds an RDKit molecule with correct bond orders and scores all (ligand assignment, oxidation state) pairs and returns a ranked list of `ComplexState` objects

## Quick start

### Transition metal complex (from XYZ data)

```python
import tmos
import tmos.build_rdmol

# symbols : list[str]   — element symbols
# geometry: np.ndarray  — coordinates in Bohr (QCElemental convention)
# molecular_charge: int — net charge of the complex
rdmol_draft = tmos.build_rdmol.xyz_to_rdkit(input["symbols"], input["geometry"])
results = tmos.sanitize_complex(rdmol_draft, target_charge=input["molecular_charge"])

best = results[0]
print(best.metal.oxidation_state)   # e.g. 2
print(best.metal.electron_count)    # e.g. 16
print(best.ligands.summary)         # "6 ligand(s), 4L/2X donors, total charge=-2"
print(best.complex.smiles)          # canonical SMILES of the reassembled complex
```

`xyz_to_rdkit` expects coordinates in **Å**.

`sanitize_complex` returns a list of `ComplexState` objects sorted by score
ascending (lower is better). Scores below 1000 indicate a chemically
plausible state (valid oxidation state, consistent net charge).
The total score is a weighted sum of penalty terms available via
`state.score_components`.

Each `ComplexState` has five sub-objects:

| Attribute | Type | Key fields |
|---|---|---|
| `state.score` | `int` | Lower is better; < 1000 = plausible |
| `state.metal` | `MetalInfo` | `oxidation_state`, `electron_count`, `charge` |
| `state.ligands` | `LigandSummary` | `number_Ltype_connectors`, `number_Xtype_connectors`, `total_charge`, `ligand_info` |
| `state.complex` | `ComplexInfo` | `rdmol`, `smiles`, `formula`, `geometry_type` |
| `state.score_components` | `ScoreComponents` | Penalty breakdown |

### Organic / non-metal molecule (from XYZ data)

For molecules without a transition metal, use `determine_bonds` directly after
building the connectivity graph:

```python
import tmos.build_rdmol

rdmol = tmos.build_rdmol.xyz_to_rdkit(
    input["symbols"], input["geometry"]
)
rdmol = tmos.build_rdmol.determine_bonds(rdmol, charge=input["molecular_charge"])
```

### Useful `sanitize_complex` options

| Parameter | Default | Purpose |
|---|---|---|
| `target_charge` | `0` | Expected net charge of the complex |
| `score_cutoff` | `1000` | Discard states with score ≥ this value (`None` keeps all) |
| `n_results` | `5` | Maximum states returned |
| `geometry_method` | `"angles"` | Geometry backend: `"angles"`, `"posym"`, `"pymatgen"`, `"rylm"` |
| `add_hydrogens` | `False` | Add explicit H atoms before processing |

## Installation

### Requirements

Python ≥ 3.10 is required. Core dependencies (`rdkit`, `openbabel`, `networkx`,
`qcelemental`, `periodictable`, `numpy`, `loguru`) are declared in
`pyproject.toml` and installed automatically by `pip`.

> **Note**: `rdkit` and `openbabel` have compiled C/C++ extensions. Installing
> via **conda/mamba is strongly recommended** to avoid build issues.
> Pure-`pip` installs may work on some platforms but are not guaranteed.

### Option A — conda / micromamba (recommended)

Clone the repository and create the bundled environment (includes all optional
dependencies):

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

**rylm** is required for `geometry_method="rylm"` and is not on PyPI:

```bash
pip install git+https://github.com/chrisiacovella/rylm.git
```

If an optional dependency is absent, the relevant function raises a clear
`ImportError` with the exact install command.

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
