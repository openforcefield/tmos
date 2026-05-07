"""Integration tests for sanitize_complex on nonhaptic (monodentate) metal carbonyl complexes.

These tests call sanitize_complex end-to-end with synthetic 3D coordinates, requiring
correct bond detection, ligand assignment, and oxidation-state scoring to pass.
The carbonylate anion cases (Fe(-2), Co(-1)) specifically exercise the negative
oxidation-state support added to reference_values.py and the
negative_charge_with_xtype_penalty added to ScoreComponents.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from tmos import tmos as tmos_module


# ---------------------------------------------------------------------------
# Molecule builders
# ---------------------------------------------------------------------------


def _mol_from_symbols_positions(symbols, positions):
    """Build an RDKit mol with atoms and a 3D conformer but no bonds.

    prepare_complex will add bonds from coordinates via detect_additional_bonds.
    """
    mol = Chem.RWMol()
    for sym in symbols:
        atom = Chem.Atom(sym)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        mol.AddAtom(atom)
    conf = Chem.Conformer(len(symbols))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(*pos))
    mol.AddConformer(conf, assignId=True)
    return mol.GetMol()


def _octahedral_carbonyl_mol(metal_symbol, mc_dist, co_dist=1.15):
    """Return an RDKit mol for octahedral M(CO)6 placed at (0.1, 0.1, 0.1).

    Ligands are arranged along ±x, ±y, ±z axes; no atom coordinate sums to 0
    so find_missing_coords does not raise with the default value=0.
    """
    origin = np.array([0.1, 0.1, 0.1])
    axes = [
        np.array([1, 0, 0]),
        np.array([-1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, -1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 0, -1]),
    ]
    symbols = [metal_symbol]
    positions = [origin.tolist()]
    for ax in axes:
        symbols += ["C", "O"]
        positions += [
            (origin + ax * mc_dist).tolist(),
            (origin + ax * (mc_dist + co_dist)).tolist(),
        ]
    return _mol_from_symbols_positions(symbols, positions)


def _tetrahedral_carbonyl_mol(metal_symbol, mc_dist, co_dist=1.15):
    """Return an RDKit mol for tetrahedral M(CO)4 placed at (0.1, 0.1, 0.1).

    Four ligands are arranged at the tetrahedral vertices of an inscribed cube.
    """
    origin = np.array([0.1, 0.1, 0.1])
    raw_dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    dirs = raw_dirs / np.linalg.norm(raw_dirs, axis=1, keepdims=True)
    symbols = [metal_symbol]
    positions = [origin.tolist()]
    for d in dirs:
        symbols += ["C", "O"]
        positions += [
            (origin + d * mc_dist).tolist(),
            (origin + d * (mc_dist + co_dist)).tolist(),
        ]
    return _mol_from_symbols_positions(symbols, positions)


def _square_pyramidal_carbonyl_mol(metal_symbol, mc_dist, co_dist=1.15):
    """Return an RDKit mol for square pyramidal M(CO)5 placed at (0.1, 0.1, 0.1).

    Four equatorial ligands along ±x, ±y; one apical along +z.
    """
    origin = np.array([0.1, 0.1, 0.1])
    axes = [
        np.array([1, 0, 0]),
        np.array([-1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, -1, 0]),
        np.array([0, 0, 1]),
    ]
    symbols = [metal_symbol]
    positions = [origin.tolist()]
    for ax in axes:
        symbols += ["C", "O"]
        positions += [
            (origin + ax * mc_dist).tolist(),
            (origin + ax * (mc_dist + co_dist)).tolist(),
        ]
    return _mol_from_symbols_positions(symbols, positions)


# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metal, n_coord, mc_dist, target_charge, expected_os",
    [
        # Neutral Group-6 hexacarbonyls: M(0) d6 18e, octahedral
        ("Cr", 6, 1.92, 0, 0),  # [Cr(CO)6]
        ("Mo", 6, 2.06, 0, 0),  # [Mo(CO)6]
        ("W", 6, 2.06, 0, 0),  # [W(CO)6]
        # Neutral Group-10 tetracarbonyl: Ni(0) d10 18e, tetrahedral
        ("Ni", 4, 1.83, 0, 0),  # [Ni(CO)4]
        # Carbonylate anions: negative metal oxidation states
        ("Fe", 4, 1.76, -2, -2),  # [Fe(CO)4]2-, Fe(-II) d10 18e
        ("Co", 4, 1.76, -1, -1),  # [Co(CO)4]-, Co(-I) d10 18e
        ("Cr", 5, 1.92, -2, -2),  # [Cr(CO)5]2-, Cr(-II) d10 18e
    ],
    ids=[
        "Cr(CO)6",
        "Mo(CO)6",
        "W(CO)6",
        "Ni(CO)4",
        "Fe(CO)4_2minus",
        "Co(CO)4_minus",
        "Cr(CO)5_2minus",
    ],
)
def test_carbonyl_oxidation_state(metal, n_coord, mc_dist, target_charge, expected_os):
    """sanitize_complex assigns the correct oxidation state for M(CO)n complexes."""
    if n_coord == 6:
        mol = _octahedral_carbonyl_mol(metal, mc_dist)
    elif n_coord == 5:
        mol = _square_pyramidal_carbonyl_mol(metal, mc_dist)
    else:
        mol = _tetrahedral_carbonyl_mol(metal, mc_dist)

    results = tmos_module.sanitize_complex(
        mol, target_charge=target_charge, score_cutoff=None, n_results=5
    )

    assert len(results) > 0, f"No states returned for {metal}(CO){n_coord}"
    best = results[0]

    assert best.metal.symbol == metal
    assert best.metal.oxidation_state == expected_os, (
        f"{metal}(CO){n_coord} target_charge={target_charge}: "
        f"expected OS={expected_os}, got OS={best.metal.oxidation_state}. "
        f"score={best.score}, {best.score_components.summary}"
    )


@pytest.mark.parametrize(
    "metal, n_coord, mc_dist, target_charge",
    [
        ("Fe", 4, 1.76, -2),  # [Fe(CO)4]2-
        ("Co", 4, 1.76, -1),  # [Co(CO)4]-
    ],
    ids=["Fe(CO)4_2minus", "Co(CO)4_minus"],
)
def test_carbonylate_anion_has_no_xtype_penalty(metal, n_coord, mc_dist, target_charge):
    """Carbonylate anions have all-L-type CO: negative_charge_with_xtype_penalty must be 0."""
    mol = _tetrahedral_carbonyl_mol(metal, mc_dist)

    results = tmos_module.sanitize_complex(
        mol, target_charge=target_charge, score_cutoff=None, n_results=5
    )

    assert len(results) > 0
    best = results[0]
    assert best.score_components.negative_charge_with_xtype_penalty == 0, (
        f"{metal}(CO){n_coord} should have no X-type penalty but got "
        f"negative_charge_with_xtype_penalty="
        f"{best.score_components.negative_charge_with_xtype_penalty}. "
        f"{best.score_components.summary}"
    )
