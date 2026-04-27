"""Integration and unit tests for haptic (η ≥ 2) bond support.

Tests cover:
- _find_haptic_groups: unit tests for the connectivity-detection helper.
- get_ligand_attributes: haptic_groups, effective_l_count, effective_x_count
    fields are populated correctly for η2–η6 ligands.
- sanitize_complex: end-to-end oxidation-state scoring for Zeise's salt (η2),
    a representative η3 allyl complex, an η6 arene complex, and the CCD
    benzene-without-H edge case.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from tmos import tmos as tmos_module
from tmos.tmos import LigandInfo, _find_haptic_groups


# ---------------------------------------------------------------------------
# Helpers shared with test_nonhaptic_complexes
# ---------------------------------------------------------------------------


def _mol_from_symbols_positions(symbols, positions):
    """Build an RDKit mol with atoms and a 3D conformer but no bonds."""
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


# ---------------------------------------------------------------------------
# Molecule builders
# ---------------------------------------------------------------------------


def _zeises_salt_mol(pt_c_dist=2.02, c_c_dist=1.34, pt_cl_dist=2.32):
    """Return an RDKit mol for Zeise's salt [PtCl3(C2H4)]^- geometry.

    Pt is at origin (offset to avoid zero-sum), the ethylene C=C lies along
    the x-axis, and three Cl ligands are at -y, +z, -z positions.
    """
    origin = np.array([0.1, 0.1, 0.1])
    # Ethylene: two C atoms symmetrically around Pt in the x-direction
    c1 = origin + np.array([pt_c_dist, -c_c_dist / 2, 0])
    c2 = origin + np.array([pt_c_dist, c_c_dist / 2, 0])
    cl1 = origin + np.array([-pt_cl_dist, 0, 0])
    cl2 = origin + np.array([0, pt_cl_dist, 0])
    cl3 = origin + np.array([0, 0, pt_cl_dist])
    # H atoms on the ethylene carbons (simplified positions)
    h1a = c1 + np.array([0.5, -0.5, 0.5])
    h1b = c1 + np.array([0.5, -0.5, -0.5])
    h2a = c2 + np.array([0.5, 0.5, 0.5])
    h2b = c2 + np.array([0.5, 0.5, -0.5])

    symbols = ["Pt", "C", "C", "Cl", "Cl", "Cl", "H", "H", "H", "H"]
    positions = [
        origin.tolist(),
        c1.tolist(),
        c2.tolist(),
        cl1.tolist(),
        cl2.tolist(),
        cl3.tolist(),
        h1a.tolist(),
        h1b.tolist(),
        h2a.tolist(),
        h2b.tolist(),
    ]
    return _mol_from_symbols_positions(symbols, positions)


def _allyl_metal_mol(metal="Mo", metal_dist=2.1, cc_dist=1.40):
    """Return an RDKit mol for a simplified η3-allyl complex M(allyl).

    Three allyl C atoms are arranged in a shallow arc; the metal is below the
    midpoint of the arc.
    """
    origin = np.array([0.1, 0.1, 0.1])
    # Allyl C atoms: C1-C2-C3 along x-axis, slightly curved
    c1 = origin + np.array([-cc_dist, metal_dist, 0])
    c2 = origin + np.array([0, metal_dist * 0.9, 0])
    c3 = origin + np.array([cc_dist, metal_dist, 0])
    # H atoms: one per terminal C, two on central C
    h1 = c1 + np.array([-0.5, 0.5, 0.5])
    h3 = c3 + np.array([0.5, 0.5, 0.5])
    h2a = c2 + np.array([0, 0.5, 0.5])
    h2b = c2 + np.array([0, 0.5, -0.5])

    symbols = [metal, "C", "C", "C", "H", "H", "H", "H"]
    positions = [
        origin.tolist(),
        c1.tolist(),
        c2.tolist(),
        c3.tolist(),
        h1.tolist(),
        h3.tolist(),
        h2a.tolist(),
        h2b.tolist(),
    ]
    return _mol_from_symbols_positions(symbols, positions)


def _benzene_complex_mol(metal="Cr", mring_dist=1.72, c_c_dist=1.40, add_h=True):
    """Return an RDKit mol for an η6-benzene complex M(C6H6).

    Six benzene C atoms are arranged in a regular hexagon; the metal is placed
    directly below the ring centroid.
    """
    origin = np.array([0.1, 0.1, 0.1])
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    c_positions = [
        origin + np.array([c_c_dist * np.cos(a), c_c_dist * np.sin(a), mring_dist])
        for a in angles
    ]
    symbols = [metal] + ["C"] * 6
    positions = [origin.tolist()] + [c.tolist() for c in c_positions]
    if add_h:
        h_positions = [
            origin
            + np.array(
                [
                    (c_c_dist + 1.08) * np.cos(a),
                    (c_c_dist + 1.08) * np.sin(a),
                    mring_dist,
                ]
            )
            for a in angles
        ]
        symbols += ["H"] * 6
        positions += [h.tolist() for h in h_positions]
    return _mol_from_symbols_positions(symbols, positions)


# ---------------------------------------------------------------------------
# Unit tests for _find_haptic_groups
# ---------------------------------------------------------------------------


def _simple_linear_mol_with_bonds(n_atoms, bond_pairs):
    """Build a minimal RWMol with n_atoms and the given bond pairs."""
    mol = Chem.RWMol()
    for _ in range(n_atoms):
        a = Chem.Atom("C")
        a.SetNoImplicit(True)
        mol.AddAtom(a)
    for i, j in bond_pairs:
        mol.AddBond(i, j, Chem.BondType.SINGLE)
    return mol.GetMol()


class TestFindHapticGroups:
    def test_single_coordinating_atom_is_singleton(self):
        mol = _simple_linear_mol_with_bonds(3, [(0, 1), (1, 2)])
        groups = _find_haptic_groups(mol, [1])
        assert groups == [[1]]

    def test_two_bonded_coordinating_atoms_form_one_group(self):
        # 0-1-2, atoms 0 and 1 coordinate metal
        mol = _simple_linear_mol_with_bonds(3, [(0, 1), (1, 2)])
        groups = _find_haptic_groups(mol, [0, 1])
        assert groups == [[0, 1]]

    def test_two_unbonded_coordinating_atoms_are_separate(self):
        # 0-2, 1-2: atoms 0 and 1 both bond to 2 but not to each other
        mol = _simple_linear_mol_with_bonds(3, [(0, 2), (1, 2)])
        groups = _find_haptic_groups(mol, [0, 1])
        assert groups == [[0], [1]]

    def test_three_bonded_in_a_row(self):
        # 0-1-2-3: atoms 0,1,2 coordinate
        mol = _simple_linear_mol_with_bonds(4, [(0, 1), (1, 2), (2, 3)])
        groups = _find_haptic_groups(mol, [0, 1, 2])
        assert groups == [[0, 1, 2]]

    def test_ring_of_five(self):
        # Five atoms in a ring; all coordinate
        mol = _simple_linear_mol_with_bonds(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
        groups = _find_haptic_groups(mol, [0, 1, 2, 3, 4])
        assert groups == [[0, 1, 2, 3, 4]]

    def test_mixed_haptic_and_sigma(self):
        # 0-1-2-3-4: atoms 1,2,3 are bonded in a chain, atom 5 is isolated
        mol = _simple_linear_mol_with_bonds(6, [(1, 2), (2, 3)])
        groups = _find_haptic_groups(mol, [1, 2, 3, 5])
        # 1,2,3 form one group; 5 is a singleton
        assert sorted([sorted(g) for g in groups]) == [[1, 2, 3], [5]]


# ---------------------------------------------------------------------------
# Unit tests for get_ligand_attributes haptic fields
# ---------------------------------------------------------------------------


def _make_cp_ring_frag_mol():
    """Build a cyclopentadienyl (Cp, η5-C5H5) fragment with dummy metal-bond markers.

    Returns (mol, metal_coordinating_indices) for use with get_ligand_attributes.
    The 5 ring C atoms are indices 0–4 (with __original_index 0–4), 5 H atoms
    are indices 5–9, and 5 dummy I atoms (one per ring C) are indices 10–14
    with __original_index == -1.
    """
    mol = Chem.RWMol()
    n_ring = 5
    r_ring = 1.22  # Å, regular pentagon with C-C ≈ 1.43 Å
    r_h = 2.30  # Å, H atoms further out

    # Ring C atoms
    for i in range(n_ring):
        a = Chem.Atom("C")
        a.SetNoImplicit(True)
        a.SetIntProp("__original_index", i)
        mol.AddAtom(a)

    # Ring H atoms (one per C)
    for i in range(n_ring):
        a = Chem.Atom("H")
        a.SetNoImplicit(True)
        a.SetIntProp("__original_index", n_ring + i)
        mol.AddAtom(a)

    # Dummy I atoms (one per C, marking former M–C bonds)
    for _ in range(n_ring):
        a = Chem.Atom("I")
        a.SetNoImplicit(True)
        a.SetIntProp("__original_index", -1)
        mol.AddAtom(a)

    # Ring bonds
    for i in range(n_ring):
        mol.AddBond(i, (i + 1) % n_ring, Chem.BondType.SINGLE)

    # C–H bonds
    for i in range(n_ring):
        mol.AddBond(i, n_ring + i, Chem.BondType.SINGLE)

    # C–dummy bonds
    for i in range(n_ring):
        mol.AddBond(i, 2 * n_ring + i, Chem.BondType.SINGLE)

    # 3D conformer: ring in xy-plane at z=0, dummies below at z=-2
    angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, angle in enumerate(angles):
        conf.SetAtomPosition(
            i, Point3D(r_ring * np.cos(angle), r_ring * np.sin(angle), 0.0)
        )
        conf.SetAtomPosition(
            n_ring + i, Point3D(r_h * np.cos(angle), r_h * np.sin(angle), 0.0)
        )
        conf.SetAtomPosition(
            2 * n_ring + i,
            Point3D(r_ring * np.cos(angle), r_ring * np.sin(angle), -2.0),
        )
    mol.AddConformer(conf, assignId=True)

    return mol.GetMol(), list(range(n_ring))


def _make_ligand_frag_with_chain(n_chain, add_h=True):
    """Build a C_n chain with dummy metal-placeholder atoms at each end.

    Returns (mol, metal_coordinating_indices) suitable for get_ligand_attributes.
    The n_chain C atoms are indexed 0..n_chain-1 and connected in a chain.
    Dummy atoms (representing the severed metal bond) are added as atom n_chain
    and n_chain+1 attached to C0 and C_{n-1}.
    """
    mol = Chem.RWMol()
    c_indices = []
    for i in range(n_chain):
        a = Chem.Atom("C")
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(1 if add_h else 0)
        a.SetIntProp("__original_index", i)
        mol.AddAtom(a)
        c_indices.append(i)
    # Chain bonds
    for i in range(n_chain - 1):
        mol.AddBond(i, i + 1, Chem.BondType.SINGLE)

    # Dummy atom at C0 (original_index = -1)
    d0 = mol.AddAtom(Chem.Atom("I"))
    mol.GetAtomWithIdx(d0).SetIntProp("__original_index", -1)
    mol.AddBond(0, d0, Chem.BondType.SINGLE)

    # Dummy atom at C_{n-1}
    dn = mol.AddAtom(Chem.Atom("I"))
    mol.GetAtomWithIdx(dn).SetIntProp("__original_index", -1)
    mol.AddBond(n_chain - 1, dn, Chem.BondType.SINGLE)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(n_chain):
        conf.SetAtomPosition(i, Point3D(float(i), 0.0, 0.0))
    conf.SetAtomPosition(d0, Point3D(-1.0, 0.0, 0.0))
    conf.SetAtomPosition(dn, Point3D(float(n_chain), 0.0, 0.0))
    mol.AddConformer(conf, assignId=True)

    metal_coordinating_indices = c_indices  # all C atoms coordinate
    return mol.GetMol(), metal_coordinating_indices


class TestGetLigandAttributesHapticFields:
    def test_eta2_effective_counts(self):
        """η2 group → effective_l_count=1, effective_x_count=0."""
        mol, coord_idx = _make_ligand_frag_with_chain(2)
        candidates = tmos_module.get_ligand_attributes(mol, coord_idx)
        best = candidates[0]
        assert best.haptic_groups is not None
        assert len(best.haptic_groups) == 1
        assert len(best.haptic_groups[0]) == 2
        assert best.effective_l_count == 1
        assert best.effective_x_count == 0

    def test_eta3_effective_counts(self):
        """η3 group → effective_l_count=1, effective_x_count=1."""
        mol, coord_idx = _make_ligand_frag_with_chain(3)
        candidates = tmos_module.get_ligand_attributes(mol, coord_idx)
        best = candidates[0]
        assert best.haptic_groups is not None
        assert len(best.haptic_groups) == 1
        assert best.effective_l_count == 1
        assert best.effective_x_count == 1

    def test_eta4_effective_counts(self):
        """η4 group → effective_l_count=2, effective_x_count=0."""
        mol, coord_idx = _make_ligand_frag_with_chain(4)
        candidates = tmos_module.get_ligand_attributes(mol, coord_idx)
        best = candidates[0]
        assert best.haptic_groups is not None
        assert best.effective_l_count == 2
        assert best.effective_x_count == 0

    def test_non_haptic_has_expected_effective_counts(self):
        """Single σ-donor → effective_l_count == 1, effective_x_count == 0 (L) or 1 (X)."""
        # Build NH3 with actual H atoms in the graph so OpenBabel can sanitize the
        # L-type candidate (isolated N with only NumExplicitHs would not sanitize).
        mol = Chem.RWMol()
        n_idx = mol.AddAtom(Chem.Atom("N"))
        mol.GetAtomWithIdx(n_idx).SetNoImplicit(True)
        mol.GetAtomWithIdx(n_idx).SetIntProp("__original_index", 0)
        h_indices = []
        for _ in range(3):
            h = Chem.Atom("H")
            h.SetNoImplicit(True)
            hi = mol.AddAtom(h)
            mol.GetAtomWithIdx(hi).SetIntProp("__original_index", hi)
            h_indices.append(hi)
            mol.AddBond(n_idx, hi, Chem.BondType.SINGLE)
        d_idx = mol.AddAtom(Chem.Atom("I"))
        mol.GetAtomWithIdx(d_idx).SetNoImplicit(True)
        mol.GetAtomWithIdx(d_idx).SetIntProp("__original_index", -1)
        mol.AddBond(n_idx, d_idx, Chem.BondType.SINGLE)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(n_idx, Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(h_indices[0], Point3D(1.0, 0.0, 0.0))
        conf.SetAtomPosition(h_indices[1], Point3D(-0.5, 0.87, 0.0))
        conf.SetAtomPosition(h_indices[2], Point3D(-0.5, -0.87, 0.0))
        conf.SetAtomPosition(d_idx, Point3D(0.0, 0.0, 2.0))
        mol.AddConformer(conf, assignId=True)
        candidates = tmos_module.get_ligand_attributes(mol.GetMol(), [0])
        best = candidates[0]
        assert best.haptic_groups == []
        # NH3 is an L-type donor
        assert best.effective_l_count == 1
        assert best.effective_x_count == 0


class TestCpLigandCharge:
    """Tests that the η5-cyclopentadienyl (Cp) ligand is perceived as Cp⁻ (charge = -1)."""

    def test_cp_ring_haptic_group_detected(self):
        """Cp fragment: one η5 haptic group of size 5 is detected."""
        mol, coord_idx = _make_cp_ring_frag_mol()
        candidates = tmos_module.get_ligand_attributes(mol, coord_idx)
        assert candidates, "No candidates returned for Cp fragment"
        best = candidates[0]
        assert best.haptic_groups is not None
        assert (
            len(best.haptic_groups) == 1
        ), f"Expected 1 haptic group, got {len(best.haptic_groups)}"
        assert (
            len(best.haptic_groups[0]) == 5
        ), f"Expected η5 group, got size {len(best.haptic_groups[0])}"

    def test_cp_ring_effective_cbc_counts(self):
        """η5-Cp: CBC gives effective_l_count=2, effective_x_count=1 (L₂X per Cp⁻)."""
        mol, coord_idx = _make_cp_ring_frag_mol()
        candidates = tmos_module.get_ligand_attributes(mol, coord_idx)
        best = candidates[0]
        assert (
            best.effective_l_count == 2
        ), f"Expected effective_l_count=2, got {best.effective_l_count}"
        assert (
            best.effective_x_count == 1
        ), f"Expected effective_x_count=1, got {best.effective_x_count}"

    def test_cp_ring_ligand_charge_is_negative_one(self):
        """After bond-order perception, isolated Cp fragment must carry charge -1 (Cp⁻)."""
        mol, coord_idx = _make_cp_ring_frag_mol()
        candidates = tmos_module.get_ligand_attributes(mol, coord_idx)
        best = candidates[0]
        assert best.total_charge == -1, (
            f"Expected Cp ligand charge -1 (Cp⁻), got {best.total_charge}. "
            "OpenBabel may not have correctly perceived the cyclopentadienyl anion."
        )


# ---------------------------------------------------------------------------
# Integration tests via sanitize_complex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder_kwargs, target_charge, expected_os, description",
    [
        # η6-benzene on Cr: [Cr(C6H6)]^0 → Cr(0) if 3 L from η6, no X
        # but alone Cr has no 18e without additional ligands, so we test
        # Cr(C6H6)(CO)3 by using the benzene builder alone and accepting
        # that the top state is scored (Cr may show OS=0 as allowed)
        # For a simpler check: the η6 group should expose effective_l_count=3
        (
            {"metal": "Cr", "mring_dist": 1.72, "add_h": True},
            0,
            None,  # We only check the haptic fields, not OS
            "Cr(η6-C6H6): η6 group detected, effective_l_count=3",
        ),
    ],
)
def test_benzene_haptic_fields(builder_kwargs, target_charge, expected_os, description):
    """η6-benzene complex: haptic_groups has one group of size 6, effective_l_count=3."""
    mol = _benzene_complex_mol(**builder_kwargs)
    results = tmos_module.sanitize_complex(
        mol, target_charge=target_charge, score_cutoff=None, n_results=5
    )
    assert len(results) > 0, f"No states for {description}"
    best = results[0]
    all_lig_info = best.ligands.ligand_info

    assert best.ligands.haptic_group_counts == {
        6: 1
    }, f"Expected one η6 group, got {best.ligands.haptic_group_counts} in {description}"
    benzene_lig = next(
        li
        for li in all_lig_info
        if li.haptic_groups and any(len(g) == 6 for g in li.haptic_groups)
    )
    assert benzene_lig.effective_l_count == 3
    assert benzene_lig.effective_x_count == 0


def test_benzene_without_H_haptic_fields():
    """CCD benzene-without-H edge case: η6 group still detected with no H atoms."""
    mol = _benzene_complex_mol(metal="Cr", mring_dist=1.72, add_h=False)
    results = tmos_module.sanitize_complex(
        mol, target_charge=0, score_cutoff=None, n_results=5
    )
    assert len(results) > 0, "No states for H-less benzene complex"
    best = results[0]
    all_lig_info = best.ligands.ligand_info
    assert best.ligands.haptic_group_counts == {
        6: 1
    }, f"Expected one η6 group, got {best.ligands.haptic_group_counts}"
    benzene_lig = next(
        li
        for li in all_lig_info
        if li.haptic_groups and any(len(g) == 6 for g in li.haptic_groups)
    )
    assert benzene_lig.effective_l_count == 3


def test_reformed_complex_haptic_bonds_are_dative():
    """After reform_metal_complex, haptic C–metal bonds should be DATIVE (bond_type_dict[0])."""
    mol = _benzene_complex_mol(metal="Cr", mring_dist=1.72, add_h=True)
    results = tmos_module.sanitize_complex(
        mol, target_charge=0, score_cutoff=None, n_results=1
    )
    assert len(results) > 0
    tmc_mol = results[0].complex.rdmol
    assert tmc_mol is not None

    cr_idx = next(a.GetIdx() for a in tmc_mol.GetAtoms() if a.GetSymbol() == "Cr")
    cr_bonds = tmc_mol.GetAtomWithIdx(cr_idx).GetBonds()
    c_bonds = [
        b
        for b in cr_bonds
        if (b.GetBeginAtom().GetSymbol() == "C" or b.GetEndAtom().GetSymbol() == "C")
    ]
    assert len(c_bonds) == 6, "Expected 6 Cr–C bonds"
    for b in c_bonds:
        assert (
            b.GetBondType() == Chem.BondType.DATIVE
        ), f"Expected DATIVE for Cr–C haptic bond, got {b.GetBondTypeAsDouble()}"


def test_fe_haptic_complex_expected_state_summary():
    """Regression test for provided Fe haptic complex scoring and assignment."""
    mol = _fe_haptic_complex_mol()
    results = tmos_module.sanitize_complex(
        mol, target_charge=0, score_cutoff=None, n_results=5
    )
    assert len(results) > 0, "No states returned for provided Fe haptic complex"

    best = results[0]
    assert best.score == 3

    assert best.metal is not None
    assert best.metal.symbol == "Fe"
    assert best.metal.oxidation_state == 0
    assert best.metal.charge == 0
    assert best.metal.electron_count == 18

    assert best.ligands is not None
    assert len(best.ligands.ligand_info) == 4
    assert best.ligands.number_Ltype_connectors == 5
    assert best.ligands.number_Xtype_connectors == 0
    assert best.ligands.total_charge == 0
    assert best.ligands.haptic_group_counts == {4: 1}

    assert best.complex is not None
    assert best.complex.formula == "C24Fe1H16O4"
    assert best.complex.charge == 0
    assert best.complex.geometry_type == "Trigonal Bipyramidal"
    assert best.complex.number_metal_connections == 5

    assert best.score_components is not None
    assert best.score_components.oxidation_membership_penalty == 0
    assert best.score_components.negative_charge_with_xtype_penalty == 0
    assert best.score_components.charge_consistency_penalty == 0
    assert best.score_components.electron_count_penalty == 0
    assert best.score_components.residual_valence_penalty == 3


def test_fe_charge0_complex_has_two_haptic_bonds():
    """Provided neutral Fe geometry should contain one η4 haptic interaction.

    The four-membered carbocyclic ring (atoms 11-12-24-16) coordinates as η4,
    contributing two CBC L-type bonds (η4//2 = 2L) — the same two-haptic-bond
    count the test name refers to.  Combined with 3 CO ligands this gives
    Fe(CO)3(η4-ring): Fe(0), 18e, Trigonal Bipyramidal — consistent with the
    18-electron rule.
    """
    mol = _fe_charge0_two_haptic_mol()
    results = tmos_module.sanitize_complex(
        mol, target_charge=0, score_cutoff=None, n_results=5
    )
    assert len(results) > 0, "No states returned for provided neutral Fe geometry"

    best = results[0]
    assert best.predicted_complex_charge == 0
    assert best.ligands.haptic_group_counts == {4: 1}, (
        f"Expected one η4 haptic group (four-membered ring, 2 L-type CBC bonds), "
        f"got {best.ligands.haptic_group_counts}"
    )
    assert (
        best.metal.electron_count == 18
    ), f"Fe(CO)3(η4-ring) should satisfy the 18e rule, got {best.metal.electron_count}e"
    assert (
        best.complex.number_metal_connections == 5
    ), f"3 CO + η4(2L) = 5 effective connections, got {best.complex.number_metal_connections}"


# ---------------------------------------------------------------------------
# Builder for anionic η3-allyl complex
# ---------------------------------------------------------------------------


def _cr_co4_allyl_mol(cr_allyl_dist=2.1, cr_co_dist=1.85, cc_dist=1.40):
    """Return geometry for [Cr(CO)₄(η3-allyl)]⁻.

    The η3-allyl sits above Cr (+y direction); four CO ligands fill the
    remaining coordination sites along −y, −x, +x, +z.  All Cr–C(allyl)
    distances are ≤ 2.53 Å, within the custom connectivity threshold for
    transition-metal pairs (base 2.16 Å + 2 × 0.20 Å tolerance ≈ 2.67 Å).
    """
    origin = np.array([0.1, 0.1, 0.1])
    # η3-allyl: C1-C2-C3 in a shallow arc above Cr
    c1 = origin + np.array([-cc_dist, cr_allyl_dist, 0.0])
    c2 = origin + np.array([0.0, cr_allyl_dist * 0.9, 0.0])
    c3 = origin + np.array([cc_dist, cr_allyl_dist, 0.0])
    h1 = c1 + np.array([-0.5, 0.5, 0.5])
    h3 = c3 + np.array([0.5, 0.5, 0.5])
    h2a = c2 + np.array([0.0, 0.5, 0.5])
    h2b = c2 + np.array([0.0, 0.5, -0.5])
    # 4 CO ligands (Cr–C ≈ 1.85 Å, C–O ≈ 1.15 Å)
    co1_c = origin + np.array([0.0, -cr_co_dist, 0.0])
    co1_o = origin + np.array([0.0, -(cr_co_dist + 1.15), 0.0])
    co2_c = origin + np.array([-cr_co_dist, 0.0, 0.0])
    co2_o = origin + np.array([-(cr_co_dist + 1.15), 0.0, 0.0])
    co3_c = origin + np.array([cr_co_dist, 0.0, 0.0])
    co3_o = origin + np.array([cr_co_dist + 1.15, 0.0, 0.0])
    co4_c = origin + np.array([0.0, 0.0, cr_co_dist])
    co4_o = origin + np.array([0.0, 0.0, cr_co_dist + 1.15])

    symbols = [
        "Cr",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",  # allyl + H
        "C",
        "O",
        "C",
        "O",
        "C",
        "O",
        "C",
        "O",  # 4 CO
    ]
    positions = [
        origin.tolist(),
        c1.tolist(),
        c2.tolist(),
        c3.tolist(),
        h1.tolist(),
        h3.tolist(),
        h2a.tolist(),
        h2b.tolist(),
        co1_c.tolist(),
        co1_o.tolist(),
        co2_c.tolist(),
        co2_o.tolist(),
        co3_c.tolist(),
        co3_o.tolist(),
        co4_c.tolist(),
        co4_o.tolist(),
    ]
    return _mol_from_symbols_positions(symbols, positions)


def _fe_haptic_complex_mol():
    """Return the provided Fe haptic-complex geometry as an RDKit mol."""
    symbols = [
        "Fe",
        "C",
        "O",
        "C",
        "O",
        "C",
        "O",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "O",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
    ]
    positions = [
        [1.42744563, 1.76994327, 4.18351898],
        [-0.23246823, 1.89425041, 4.76866406],
        [-1.28864229, 1.92768361, 5.21395634],
        [1.90687878, 2.66643802, 5.62812272],
        [2.18321123, 3.19494913, 6.60784366],
        [1.81872122, 0.0660151, 4.61762634],
        [2.02512649, -1.0253736, 4.88562677],
        [3.75636955, 0.80126597, 2.36040862],
        [3.46178351, 2.04989495, 3.01426014],
        [2.47546031, 2.98729894, 2.60780034],
        [1.19063302, 2.59677071, 2.19762747],
        [0.81773348, 1.20771061, 2.17130835],
        [1.5310376, 0.17736905, 1.30311261],
        [2.97211646, -0.03294371, 1.59904421],
        [3.77360711, -1.16216909, 1.28859565],
        [5.00564007, -0.93025215, 1.8748484],
        [4.98503543, 0.25900195, 2.52871533],
        [3.31670231, -2.35117747, 0.56907232],
        [2.69919934, -2.23508091, -0.67239296],
        [2.24451222, -3.36033851, -1.33599958],
        [2.39178495, -4.61233786, -0.76401004],
        [2.99859431, -4.73558139, 0.47571349],
        [3.45797026, -3.61434494, 1.1397527],
        [6.24386652, -1.68293379, 1.8704795],
        [7.19875881, -1.44251208, 2.86159222],
        [8.38572551, -2.1474111, 2.87669522],
        [8.64275629, -3.10116136, 1.90514669],
        [7.70730268, -3.33659637, 0.91004249],
        [6.51909329, -2.63334362, 0.88520882],
        [4.33881784, 2.46892087, 3.50090118],
        [2.69067928, 4.04519831, 2.72231351],
        [0.46339499, 3.36604649, 1.95344145],
        [-0.25723442, 1.07733678, 2.02472238],
        [1.40984797, 0.50346289, 0.26100438],
        [1.0176989, -0.78479846, 1.39312762],
        [2.59516307, -1.26016082, -1.12593972],
        [1.77376451, -3.25931246, -2.30321269],
        [2.03473064, -5.49003197, -1.28221366],
        [3.11217506, -5.71029839, 0.92707567],
        [3.92977083, -3.70668821, 2.10699264],
        [6.99318812, -0.70198818, 3.61977837],
        [9.11409513, -1.95470146, 3.65088683],
        [9.56979379, -3.65483259, 1.92000364],
        [7.90831997, -4.07074408, 0.14356593],
        [5.80453846, -2.80864251, 0.09627161],
    ]
    return _mol_from_symbols_positions(symbols, positions)


def _fe_charge0_two_haptic_mol():
    """Return the provided Fe geometry expected to have one η2 interaction."""
    symbols = [
        "Fe",
        "O",
        "O",
        "C",
        "H",
        "C",
        "C",
        "H",
        "H",
        "C",
        "O",
        "C",
        "C",
        "C",
        "H",
        "H",
        "C",
        "C",
        "H",
        "H",
        "C",
        "H",
        "C",
        "H",
        "C",
        "C",
        "H",
        "H",
        "C",
        "H",
        "C",
        "H",
        "C",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "C",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
    ]
    positions = [
        [11.76748918, 7.15277021, 2.13016044],
        [9.56931533, 9.06487914, 2.27648657],
        [12.57804175, 7.75697118, -0.6051279],
        [8.7184299, 4.70504729, 0.23163294],
        [8.05615261, 5.09445745, -0.5515537],
        [10.39532348, 8.26730679, 2.21954498],
        [9.03076546, 3.4901625, 2.88011221],
        [9.70384556, 3.13855238, 3.66557458],
        [8.01195003, 3.24237816, 3.1895362],
        [12.23947566, 7.47844912, 0.45700713],
        [13.55595855, 9.23184087, 3.11275827],
        [10.58258985, 5.38061037, 2.75813093],
        [11.65885941, 5.73778456, 3.6743194],
        [13.10062831, 6.4618303, 5.58514059],
        [13.05436574, 6.58814456, 6.67034171],
        [13.03509031, 7.45409184, 5.13487273],
        [11.52474725, 4.99535562, 1.76355054],
        [10.75758278, 2.77316666, 1.06595304],
        [10.82942485, 2.07095257, 0.23115642],
        [11.43634215, 2.42109564, 1.846281],
        [14.45929894, 5.82944735, 5.26025663],
        [15.21997804, 6.46254918, 5.73371339],
        [13.45162698, 3.57938405, 5.20141101],
        [13.47777456, 2.56913592, 5.63153229],
        [12.63472059, 5.33883484, 2.65395102],
        [14.830379, 5.73642216, 3.77535686],
        [14.77193658, 6.72481067, 3.31481848],
        [15.87208095, 5.40830985, 3.72244106],
        [9.13645761, 5.02464402, 2.76588093],
        [8.65029386, 5.47476716, 3.63929307],
        [9.31216233, 2.72759591, 1.57874556],
        [9.08416607, 1.67226583, 1.77898432],
        [11.22463985, 4.15639854, 0.569477],
        [12.13528991, 4.02472637, -0.02650234],
        [13.79014623, 3.43086744, 3.71330141],
        [13.0172177, 2.83929835, 3.21745262],
        [14.72518214, 2.86795982, 3.64619693],
        [8.42545771, 5.51951136, 1.49704876],
        [7.3470368, 5.4732001, 1.67179249],
        [8.68987537, 6.56562703, 1.32727157],
        [11.88176837, 5.63984648, 5.14610858],
        [10.99728867, 6.01937534, 5.67293201],
        [13.9768765, 4.76014389, 2.95481623],
        [14.49859598, 4.54808827, 2.01318811],
        [10.14758495, 4.80423214, -0.31434266],
        [10.40537462, 5.85392174, -0.47511624],
        [10.16017114, 4.30562938, -1.28737172],
        [12.06687606, 4.15538222, 5.5202559],
        [11.91200748, 4.05558696, 6.59807576],
        [11.29726406, 3.56009103, 5.02373948],
        [12.89231518, 8.34595899, 2.79947488],
        [14.51799255, 4.43508508, 5.87525933],
        [15.50582325, 3.99514982, 5.72283504],
        [14.33516446, 4.48725659, 6.95065868],
        [8.37947921, 3.2406945, 0.48784613],
        [7.33888616, 3.14270746, 0.80415077],
        [8.50983198, 2.65654696, -0.42551142],
    ]
    return _mol_from_symbols_positions(symbols, positions)


# ---------------------------------------------------------------------------
# Tests for negative_xtype_penalty with haptic X contributions
# ---------------------------------------------------------------------------


class TestNegativeXtypePenaltyHapticGroups:
    """The negative_charge_with_xtype_penalty must not fire for haptic-X contributors.

    η3-allyl contributes effective_x_count=1 via η%2 == 1.  Before the fix
    this caused anionic allyl complexes to be incorrectly filtered out
    (score ≥ score_cutoff=1000).  The fix tracks σ-X (non-haptic) connectors
    separately and only applies the penalty to those.

    [Cr(CO)₄(η3-allyl)]⁻ is used as the test case:
    - allyl η3 → haptic_x=1, sigma_x=0, total_charge=0 (neutral radical)
    - target_charge=−1 → metal_charge=−1, Cr(0) oxidation state
    - Cr(0) is valid for Cr; 18e; score should be 0 (with fix) vs 1000 (without)
    """

    @pytest.fixture(scope="class")
    def cr_allyl_results(self):
        mol = _cr_co4_allyl_mol()
        return tmos_module.sanitize_complex(
            mol, target_charge=-1, score_cutoff=1000, n_results=5
        )

    def test_anionic_allyl_not_filtered(self, cr_allyl_results):
        """sanitize_complex returns at least one state for [Cr(CO)₄(η3-allyl)]⁻."""
        assert len(cr_allyl_results) > 0, (
            "No states survived score_cutoff=1000 for [Cr(CO)₄(η3-allyl)]⁻. "
            "The negative_xtype_penalty may be incorrectly firing for the haptic X "
            "from η3-allyl (η%2 == 1 is a CBC bookkeeping artifact, not a σ-donor)."
        )

    def test_anionic_allyl_charge_consistent(self, cr_allyl_results):
        """Best state for [Cr(CO)₄(η3-allyl)]⁻ has predicted_complex_charge == -1."""
        assert len(cr_allyl_results) > 0
        assert cr_allyl_results[0].predicted_complex_charge == -1

    def test_anionic_allyl_no_xtype_penalty(self, cr_allyl_results):
        """Best state has negative_charge_with_xtype_penalty == 0 (haptic X excluded)."""
        assert len(cr_allyl_results) > 0
        assert (
            cr_allyl_results[0].score_components.negative_charge_with_xtype_penalty == 0
        )


class TestNegativeXtypePenaltyUnit:
    """Unit-level verification of the sigma-X-only penalty rule via _score_and_flatten_states.

    The integration test above uses a geometry-derived complex where full η3 perception
    depends on the connectivity algorithm.  This class bypasses geometry by constructing
    combo dicts directly, letting us verify the exact code path in
    ``_score_and_flatten_states`` that distinguishes haptic X from sigma X.
    """

    @staticmethod
    def _make_cr_mol():
        """Minimal single-atom Cr RDKit mol for score-only tests."""
        mol = Chem.RWMol()
        atom = Chem.Atom("Cr")
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
        return mol.GetMol()

    def test_haptic_x_no_penalty_for_negative_metal(self):
        """x_type_connectors=[] with effective_x_count=1 (η3-allyl CBC η%2): penalty must be 0.

        Scenario: [Cr(CO)₄(η3-allyl)]⁻, n_L=5, n_X=1 (haptic remainder),
        sigma_X=0, total_ligand_charge=0, target=-1.
        CBC: charge=-1, n_X=1, n_L=5 → nel=18, Cr(0).
        Haptic atoms are always L-type; x_type_connectors is empty for the allyl,
        so n_sigma_xtype=0 and the penalty does not fire.
        """
        cr_mol = self._make_cr_mol()
        # η3-allyl: all 3 carbons are L-type (DATIVE), none in x_type_connectors.
        # effective_x_count=1 is the CBC η%2 bookkeeping remainder, not a σ-bond.
        allyl_lig = LigandInfo(
            index=0,
            rdmol=None,
            smiles=None,
            chemical_formula=None,
            candidate_id="test-allyl",
            total_charge=0,
            hanging_bonds=0,
            charged_atoms={},
            l_type_connectors=[1, 2, 3, 4, 5],  # η3-allyl (as 1 L) + 4 CO
            x_type_connectors=[],  # no sigma-X donors
            haptic_groups=[[1, 2, 3]],
            effective_l_count=5,
            effective_x_count=1,
        )
        combo = {
            "ligand_info": [allyl_lig],
            "candidate_ids": ["test-allyl"],
            "number_Ltype_connectors": 5,
            "number_Xtype_connectors": 1,
            "total_ligand_charge": 0,
        }
        results = tmos_module._score_and_flatten_states(
            cr_mol, [combo], target_complex_charge=-1
        )
        # Cr(0): oxidation_state = n_X + charge = 1 + (-1) = 0; nel = 6+1+10+1=18
        cr0 = next((s for s in results if s.metal.oxidation_state == 0), None)
        assert (
            cr0 is not None
        ), "Cr(0) state must exist for [Cr(CO)₄(η3-allyl)]⁻ scenario"
        assert cr0.score_components.negative_charge_with_xtype_penalty == 0, (
            "Haptic-only X (n_sigma_xtype=0) must not trigger negative_xtype_penalty "
            "even when metal_charge < 0"
        )
        assert cr0.predicted_complex_charge == -1
        assert cr0.metal.electron_count == 18

    def test_sigma_x_still_penalizes_negative_metal(self):
        """x_type_connectors=[idx] with effective_x_count=1 (genuine σ-X): penalty must be 1.

        Scenario: hypothetical Cr(-1) with one sigma X donor (e.g. Cl).
        Cl is not haptic, so it appears in x_type_connectors; n_sigma_xtype=1
        and the penalty must still fire.
        """
        cr_mol = self._make_cr_mol()
        # Genuine σ-X donor (e.g. Cl): appears in x_type_connectors.
        sigma_lig = LigandInfo(
            index=0,
            rdmol=None,
            smiles=None,
            chemical_formula=None,
            candidate_id="test-sigma",
            total_charge=0,
            hanging_bonds=0,
            charged_atoms={},
            l_type_connectors=[1, 2, 3, 4, 5],  # 5 L (e.g. CO ligands)
            x_type_connectors=[10],  # one genuine sigma-X donor
            haptic_groups=[],
            effective_l_count=5,
            effective_x_count=1,
        )
        combo = {
            "ligand_info": [sigma_lig],
            "candidate_ids": ["test-sigma"],
            "number_Ltype_connectors": 5,
            "number_Xtype_connectors": 1,
            "total_ligand_charge": 0,
        }
        results = tmos_module._score_and_flatten_states(
            cr_mol, [combo], target_complex_charge=-1
        )
        cr0 = next((s for s in results if s.metal.oxidation_state == 0), None)
        assert cr0 is not None
        assert cr0.score_components.negative_charge_with_xtype_penalty == 1, (
            "A genuine sigma X donor (n_sigma_xtype=1) with metal_charge < 0 "
            "must still trigger the penalty"
        )
