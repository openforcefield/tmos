"""Regression tests for N-S/N-O donor bond detection on saturated ring amines.

Covers the class of false-negative bonds where a lone-pair donor (N, exocyclic
S or O, e.g. a sulfonamide/sulfonyl substituent) attached to a saturated ring
nitrogen (pyrrolidine/piperidine/azetidine-type, as opposed to an aromatic or
otherwise conjugated ring nitrogen such as pyridine) was not detected by
``xyz_to_rdkit``/``determine_connectivity`` even though the bond distance fell
well inside the allowed window. See ``_is_conjugated_ring_nitrogen`` in
``tmos.build_rdmol`` for the underlying fix.

Each case is checked three ways:
  1. Every N-S/N-O bond present in the reference SMILES is present in the
     connectivity-only graph produced by ``xyz_to_rdkit`` (the direct
     regression check for the bug this module targets).
  2. The full heavy-atom bonding graph of the bond-order-assigned molecule
     is isomorphic to the reference SMILES graph (mirrors the pattern used
     in ``test_error_check4.py``).
  3. The bond-order-assigned molecule is a resonance structure of the
     reference SMILES (adapted from
     ``Test_QM40/qm40_pipeline.py::is_resonance_equivalent_targeted``).
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolHash

from tmos import build_rdmol
from tmos.graph_mapping import mol_to_graph
from tmos.tests.diagnostic import analyze_graph_mismatch, format_graph_mismatch_report

with open(Path(__file__).parent / "n-s_molecules.json") as f:
    TEST_CASES = json.load(f)


# ---------------------------------------------------------------------------
# Resonance-equivalence helpers (adapted from Test_QM40/qm40_pipeline.py;
# rewritten to depend only on RDKit, since openff.toolkit is not a tmos
# dependency).
# ---------------------------------------------------------------------------


def _canonical_smiles_variants_for_resonance(mol: Chem.rdchem.Mol) -> set[str]:
    """Return canonical aromatic and kekule SMILES variants for a molecule."""
    if mol is None:
        return set()

    variants = set()
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)

    for sanitize in (False, True):
        mm = Chem.Mol(m)
        if sanitize:
            try:
                Chem.SanitizeMol(mm)
            except Exception:
                pass
        variants.add(Chem.MolToSmiles(mm, canonical=True))
        try:
            variants.add(Chem.MolToSmiles(mm, canonical=True, kekuleSmiles=True))
        except Exception:
            pass

    return variants


def _mesomer_hash(mol: Chem.rdchem.Mol) -> str:
    """Return RDKit mesomer hash (resonance-invariant) for a molecule copy."""
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass

    m_no_h = Chem.RemoveHs(m)
    return rdMolHash.MolHash(m_no_h, rdMolHash.HashFunction.Mesomer)


def _resonance_form_suppliers(ref_mol: Chem.rdchem.Mol):
    """Yield resonance suppliers from multiple reference preparations."""
    flags = (
        Chem.ResonanceFlags.KEKULE_ALL
        | Chem.ResonanceFlags.ALLOW_CHARGE_SEPARATION
        | Chem.ResonanceFlags.UNCONSTRAINED_CATIONS
        | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS
    )

    prepared_refs = []

    ref_raw = Chem.Mol(ref_mol)
    Chem.RemoveStereochemistry(ref_raw)
    prepared_refs.append(ref_raw)

    ref_sanitized = Chem.Mol(ref_mol)
    Chem.RemoveStereochemistry(ref_sanitized)
    try:
        Chem.SanitizeMol(ref_sanitized)
    except Exception:
        pass
    prepared_refs.append(ref_sanitized)

    seen = set()
    for candidate in prepared_refs:
        key = tuple(sorted(_canonical_smiles_variants_for_resonance(candidate)))
        if key in seen:
            continue
        seen.add(key)
        try:
            supplier = Chem.ResonanceMolSupplier(candidate, flags)
        except Exception:
            continue
        if supplier is not None:
            yield supplier


def is_resonance_equivalent(
    ref_mol: Chem.rdchem.Mol, test_mol: Chem.rdchem.Mol
) -> bool:
    """Return True only if test matches resonance-equivalent chemistry of reference."""
    if ref_mol is None or test_mol is None:
        return False

    ref = Chem.Mol(ref_mol)
    test = Chem.Mol(test_mol)

    Chem.RemoveStereochemistry(ref)
    Chem.RemoveStereochemistry(test)

    if ref.GetNumAtoms() != test.GetNumAtoms():
        return False
    if Chem.GetFormalCharge(ref) != Chem.GetFormalCharge(test):
        return False

    try:
        if _mesomer_hash(ref) == _mesomer_hash(test):
            return True
    except Exception:
        pass

    test_variants = _canonical_smiles_variants_for_resonance(test)
    if not test_variants:
        return False

    for supplier in _resonance_form_suppliers(ref):
        for idx in range(len(supplier)):
            ref_res = supplier[idx]
            if ref_res is None:
                continue
            if _canonical_smiles_variants_for_resonance(ref_res) & test_variants:
                return True

    return bool(_canonical_smiles_variants_for_resonance(ref) & test_variants)


# ---------------------------------------------------------------------------
# Heavy-atom graph isomorphism helper (mirrors test_error_check4.py).
# ---------------------------------------------------------------------------


def _check_graph_consistent(rdmol: Chem.rdchem.Mol, ref_smiles: str) -> None:
    """Assert the heavy-atom graph of *rdmol* is isomorphic to the reference SMILES graph."""
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if ref_mol is None:
        ref_mol = Chem.MolFromSmiles(ref_smiles, sanitize=False)
    assert ref_mol is not None, f"Invalid reference SMILES: {ref_smiles}"

    g_actual = mol_to_graph(rdmol, remove_hydrogens=True)
    g_ref = mol_to_graph(ref_mol, remove_hydrogens=True)

    n_actual, n_ref = g_actual.number_of_nodes(), g_ref.number_of_nodes()
    assert (
        n_actual == n_ref
    ), f"Heavy-atom count mismatch: got {n_actual}, expected {n_ref}"

    node_match = nx.algorithms.isomorphism.categorical_node_match("symbol", None)
    gm = nx.algorithms.isomorphism.GraphMatcher(g_actual, g_ref, node_match=node_match)
    if gm.is_isomorphic():
        return

    report = analyze_graph_mismatch(rdmol, ref_smiles)
    formatted = format_graph_mismatch_report(report)
    e_actual, e_ref = g_actual.number_of_edges(), g_ref.number_of_edges()
    raise AssertionError(
        f"Heavy-atom graph mismatch: "
        f"got {e_actual} edges, expected {e_ref}.\n{formatted}"
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [pytest.param(key, entry, id=key) for key, entry in TEST_CASES.items()]


@pytest.mark.parametrize(("key", "entry"), CASES)
def test_ns_no_donor_bonds_detected(key: str, entry: dict) -> None:
    """The connectivity-only graph must contain every reference N-S/N-O bond."""
    symbols = entry["symbols"]
    coordinates = entry["coordinates"]
    ref_smiles = entry["smiles"]

    ref_mol = Chem.MolFromSmiles(ref_smiles)
    assert ref_mol is not None, f"{key}: invalid reference SMILES: {ref_smiles}"

    ref_donor_pairs = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in ref_mol.GetBonds()
        if {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
        in ({"N", "S"}, {"N", "O"})
    }
    if not ref_donor_pairs:
        pytest.skip(f"{key}: no N-S/N-O bonds in reference SMILES")

    rdmol = build_rdmol.xyz_to_rdkit(
        symbols, coordinates, method="custom", ignore_scale=True
    )
    detected_pairs = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in rdmol.GetBonds()
    }

    missing = ref_donor_pairs - detected_pairs
    assert not missing, (
        f"{key}: missing N-S/N-O bonds at heavy-atom indices "
        f"{sorted(missing)} "
        f"({[(symbols[i], symbols[j]) for i, j in sorted(missing)]})"
    )


@pytest.mark.parametrize(("key", "entry"), CASES)
def test_ns_full_pipeline_graph_and_resonance_match_reference(
    key: str, entry: dict
) -> None:
    """The bond-order-assigned molecule must match the reference graph and
    be a resonance structure of it."""
    symbols = entry["symbols"]
    coordinates = entry["coordinates"]
    ref_smiles = entry["smiles"]

    ref_mol = Chem.MolFromSmiles(ref_smiles)
    assert ref_mol is not None, f"{key}: invalid reference SMILES: {ref_smiles}"

    rdmol = build_rdmol.xyz_to_rdkit(
        symbols, coordinates, method="custom", ignore_scale=True
    )
    rdmol = build_rdmol.determine_bonds(
        rdmol, charge=Chem.GetFormalCharge(ref_mol), custom_cleanup=True
    )

    _check_graph_consistent(rdmol, ref_smiles)

    ref_mol_h = Chem.AddHs(ref_mol)
    assert is_resonance_equivalent(ref_mol_h, rdmol), (
        f"{key}: detected molecule "
        f"({Chem.MolToSmiles(rdmol)}) is not a resonance structure of the "
        f"reference SMILES ({ref_smiles})"
    )
