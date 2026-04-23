"""
Unit and regression test for the tmos package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest
import numpy as np
from rdkit import Chem

from tmos import tmos as tmos_module


def test_tmos_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "tmos" in sys.modules


def _build_test_ligand() -> tuple[Chem.rdchem.Mol, list[int]]:
    """Create a tiny ligand fragment with one coordinating atom and one dummy atom."""
    ligand = Chem.MolFromSmiles("[CH3][I]")
    ligand = Chem.AddHs(ligand)
    coordinating_indices = []
    for atom in ligand.GetAtoms():
        if atom.GetSymbol() == "I":
            atom.SetIntProp("__original_index", -1)
        else:
            atom.SetIntProp("__original_index", 42)
            if atom.GetSymbol() == "C":
                coordinating_indices.append(atom.GetIdx())
    return ligand, coordinating_indices


@pytest.fixture
def patch_ligand_enumeration(monkeypatch):
    """Patch heavy chemistry paths to test enumeration/control flow deterministically."""

    monkeypatch.setattr(
        tmos_module,
        "check_ligand_exception",
        lambda ligand_mol, metal_indices: (None, []),
    )
    monkeypatch.setattr(
        tmos_module,
        "is_coordinate_ring",
        lambda ligand_mol, metal_indices: False,
    )

    def fake_sanitize_ligand(ligand_mol, delete_list=None, **kwargs):
        result = Chem.RWMol(Chem.Mol(ligand_mol))
        result.SetIntProp("_test_deleted", len(delete_list or []))
        return result

    monkeypatch.setattr(tmos_module, "sanitize_ligand", fake_sanitize_ligand)

    def fake_assess_atoms(mol):
        if mol.GetIntProp("_test_deleted") == 1:
            return -1, 0, {0: {"formal_charge": -1}}
        return 0, 1, {}

    monkeypatch.setattr(tmos_module.brd, "assess_atoms", fake_assess_atoms)


def test_get_ligand_attributes_returns_all_candidates_with_deterministic_ids(
    patch_ligand_enumeration,
):
    ligand, coordinating_indices = _build_test_ligand()

    candidates_1 = tmos_module.get_ligand_attributes(
        ligand,
        coordinating_indices,
        return_all_candidates=True,
    )
    candidates_2 = tmos_module.get_ligand_attributes(
        ligand,
        coordinating_indices,
        return_all_candidates=True,
    )

    assert len(candidates_1) == 2
    assert [c["candidate_id"] for c in candidates_1] == [
        c["candidate_id"] for c in candidates_2
    ]
    assert all(c["candidate_id"].startswith("ligcand-") for c in candidates_1)
    assert all(
        "L-type connectors" in c and "X-type connectors" in c for c in candidates_1
    )

    connector_patterns = {
        (tuple(sorted(c["L-type connectors"])), tuple(sorted(c["X-type connectors"])))
        for c in candidates_1
    }
    assert connector_patterns == {((42,), ()), ((), (42,))}


def test_get_ligand_attributes_default_returns_best_candidate(patch_ligand_enumeration):
    ligand, coordinating_indices = _build_test_ligand()

    best = tmos_module.get_ligand_attributes(ligand, coordinating_indices)

    assert best["candidate_id"].startswith("ligcand-")
    assert best["hanging_bonds"] == 0
    assert best["L-type connectors"] == [42]
    assert best["X-type connectors"] == []


def test_enumerate_ligand_combinations_cartesian_totals():
    ligand_a1 = {
        "candidate_id": "ligcand-a1",
        "L-type connectors": [10],
        "X-type connectors": [],
        "total_charge": 0,
    }
    ligand_a2 = {
        "candidate_id": "ligcand-a2",
        "L-type connectors": [],
        "X-type connectors": [10],
        "total_charge": -1,
    }
    ligand_b1 = {
        "candidate_id": "ligcand-b1",
        "L-type connectors": [20, 21],
        "X-type connectors": [],
        "total_charge": 0,
    }

    combinations_out = tmos_module._enumerate_ligand_combinations(
        [[ligand_a1, ligand_a2], [ligand_b1]]
    )

    assert len(combinations_out) == 2
    assert combinations_out[0]["candidate_ids"] == ["ligcand-a1", "ligcand-b1"]
    assert combinations_out[0]["number_Ltype_connectors"] == 3
    assert combinations_out[0]["number_Xtype_connectors"] == 0
    assert combinations_out[0]["total_ligand_charge"] == 0

    assert combinations_out[1]["candidate_ids"] == ["ligcand-a2", "ligcand-b1"]
    assert combinations_out[1]["number_Ltype_connectors"] == 2
    assert combinations_out[1]["number_Xtype_connectors"] == 1
    assert combinations_out[1]["total_ligand_charge"] == -1


def test_enumerate_ligand_combinations_symmetry_reduction():
    ligand_1_a = {
        "candidate_id": "ligcand-1a",
        "chemical_formula": "C1H1N1",
        "smiles": "[CH]=[N]",
        "total_charge": 0,
        "hanging_bonds": 0,
        "charged_atoms": {},
        "L-type connectors": [10],
        "X-type connectors": [],
    }
    ligand_1_b = {
        "candidate_id": "ligcand-1b",
        "chemical_formula": "C1H1N1",
        "smiles": "[CH][N-]",
        "total_charge": -1,
        "hanging_bonds": 0,
        "charged_atoms": {0: {"formal_charge": -1}},
        "L-type connectors": [],
        "X-type connectors": [10],
    }
    ligand_2_a = {
        "candidate_id": "ligcand-2a",
        "chemical_formula": "C1H1N1",
        "smiles": "[CH]=[N]",
        "total_charge": 0,
        "hanging_bonds": 0,
        "charged_atoms": {},
        "L-type connectors": [20],
        "X-type connectors": [],
    }
    ligand_2_b = {
        "candidate_id": "ligcand-2b",
        "chemical_formula": "C1H1N1",
        "smiles": "[CH][N-]",
        "total_charge": -1,
        "hanging_bonds": 0,
        "charged_atoms": {0: {"formal_charge": -1}},
        "L-type connectors": [],
        "X-type connectors": [20],
    }

    combinations_out = tmos_module._enumerate_ligand_combinations(
        [[ligand_1_a, ligand_1_b], [ligand_2_a, ligand_2_b]]
    )

    assert len(combinations_out) == 3
    assert all(x["dedupe_group_ids"] == ["g0", "g0"] for x in combinations_out)
    combo_signatures = {
        (
            x["number_Ltype_connectors"],
            x["number_Xtype_connectors"],
            x["total_ligand_charge"],
        )
        for x in combinations_out
    }
    assert combo_signatures == {(2, 0, 0), (1, 1, -1), (0, 2, -2)}


def test_score_ligand_combinations_with_metal_prefers_charge_consistency(monkeypatch):
    tm_mol = Chem.MolFromSmiles("[Fe]")

    # Keep metal-state options deterministic for scoring assertions.
    monkeypatch.setattr(
        tmos_module,
        "get_tm_attributes",
        lambda *_args, **_kwargs: ([2], np.array([2]), np.array([18])),
    )

    combinations_in = [
        {
            "ligand_info": [{"hanging_bonds": 0}],
            "candidate_ids": ["ligcand-a"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 2,
            "total_ligand_charge": -2,
            "dedupe_group_ids": ["g0"],
            "dedupe_key": ("k0",),
        },
        {
            "ligand_info": [{"hanging_bonds": 0}],
            "candidate_ids": ["ligcand-b"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 2,
            "total_ligand_charge": 0,
            "dedupe_group_ids": ["g0"],
            "dedupe_key": ("k1",),
        },
    ]

    scored = tmos_module._score_ligand_combinations_with_metal(tm_mol, combinations_in)

    assert scored[0]["candidate_ids"] == ["ligcand-a"]
    assert scored[0]["metal_scoring"]["best_state"]["predicted_complex_charge"] == 0
    assert (
        scored[0]["metal_scoring"]["best_score"]
        < scored[1]["metal_scoring"]["best_score"]
    )


def test_score_ligand_combinations_with_metal_penalizes_residual_valence(monkeypatch):
    tm_mol = Chem.MolFromSmiles("[Fe]")

    monkeypatch.setattr(
        tmos_module,
        "get_tm_attributes",
        lambda *_args, **_kwargs: ([2], np.array([2]), np.array([18])),
    )

    combinations_in = [
        {
            "ligand_info": [{"hanging_bonds": 0}],
            "candidate_ids": ["ligcand-clean"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 2,
            "total_ligand_charge": -2,
            "dedupe_group_ids": ["g0"],
            "dedupe_key": ("k0",),
        },
        {
            "ligand_info": [{"hanging_bonds": 3}],
            "candidate_ids": ["ligcand-residual"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 2,
            "total_ligand_charge": -2,
            "dedupe_group_ids": ["g0"],
            "dedupe_key": ("k1",),
        },
    ]

    scored = tmos_module._score_ligand_combinations_with_metal(tm_mol, combinations_in)

    assert scored[0]["candidate_ids"] == ["ligcand-clean"]
    assert (
        scored[0]["metal_scoring"]["best_score"]
        < scored[1]["metal_scoring"]["best_score"]
    )


def test_sanitize_complex_default_output_unchanged_without_all_candidates(monkeypatch):
    complex_mol = Chem.MolFromSmiles("[Fe]")
    ligand_fragment = Chem.MolFromSmiles("[CH3][I]")
    metal_fragment = Chem.MolFromSmiles("[Fe]")

    for atom in ligand_fragment.GetAtoms():
        if atom.GetSymbol() == "I":
            atom.SetIntProp("__original_index", -1)
        else:
            atom.SetIntProp("__original_index", 42)
    metal_fragment.GetAtomWithIdx(0).SetIntProp("__original_index", 0)

    monkeypatch.setattr(
        tmos_module, "prepare_complex", lambda *args, **kwargs: complex_mol
    )
    monkeypatch.setattr(tmos_module, "find_metal_index", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        tmos_module,
        "cleave_mol_from_index",
        lambda *_args, **_kwargs: ([ligand_fragment, metal_fragment], [42]),
    )
    monkeypatch.setattr(
        tmos_module,
        "get_geometry_from_mol",
        lambda *_args, **kwargs: ("square-planar", 4, {})
        if kwargs.get("mode") == "angles"
        else ("square-planar", 4, {}),
    )

    ligand_best = {
        "candidate_id": "ligcand-best",
        "rdmol": ligand_fragment,
        "smiles": "[CH3][I]",
        "chemical_formula": "C1H3I1",
        "L-type connectors": [42],
        "X-type connectors": [],
        "total_charge": 0,
        "hanging_bonds": 0,
        "charged_atoms": {},
    }
    monkeypatch.setattr(
        tmos_module,
        "get_ligand_attributes",
        lambda *_args, **kwargs: [ligand_best]
        if kwargs.get("return_all_candidates")
        else ligand_best,
    )

    combinations_out = [
        {
            "ligand_info": [ligand_best],
            "candidate_ids": ["ligcand-best"],
            "number_Ltype_connectors": 1,
            "number_Xtype_connectors": 0,
            "total_ligand_charge": 0,
            "dedupe_group_ids": ["g0"],
            "dedupe_key": ("k0",),
        }
    ]
    scored_out = [
        {
            **combinations_out[0],
            "metal_scoring": {
                "target_complex_charge": 0,
                "target_electron_count": 18,
                "state_scores": [
                    {
                        "oxidation_state": 2,
                        "tm_charge": 2,
                        "tm_electron_count": 18,
                        "predicted_complex_charge": 2,
                        "score_components": {
                            "oxidation_membership_penalty": 0,
                            "charge_consistency_penalty": 2,
                            "electron_count_penalty": 0,
                            "residual_valence_penalty": 0,
                        },
                        "score": 200,
                    }
                ],
                "best_state": {
                    "oxidation_state": 2,
                    "tm_charge": 2,
                    "tm_electron_count": 18,
                    "predicted_complex_charge": 2,
                    "score_components": {},
                    "score": 200,
                },
                "best_score": 200,
            },
        }
    ]
    monkeypatch.setattr(
        tmos_module,
        "_enumerate_ligand_combinations",
        lambda *_args, **_kwargs: combinations_out,
    )
    monkeypatch.setattr(
        tmos_module,
        "_score_ligand_combinations_with_metal",
        lambda *_args, **_kwargs: scored_out,
    )
    monkeypatch.setattr(
        tmos_module,
        "reform_metal_complex",
        lambda *_args, **_kwargs: Chem.MolFromSmiles("[Fe]"),
    )
    monkeypatch.setattr(tmos_module, "mol_to_smiles", lambda *_args, **_kwargs: "[Fe]")
    monkeypatch.setattr(
        tmos_module,
        "get_molecular_formula",
        lambda *_args, **_kwargs: "Fe1",
    )

    outputs = tmos_module.sanitize_complex(complex_mol)

    assert "__all_candidates__" not in outputs
    assert len(outputs) >= 1


def test_sanitize_complex_return_all_candidates_payload(monkeypatch):
    complex_mol = Chem.MolFromSmiles("[Fe]")
    ligand_fragment = Chem.MolFromSmiles("[CH3][I]")
    metal_fragment = Chem.MolFromSmiles("[Fe]")

    for atom in ligand_fragment.GetAtoms():
        if atom.GetSymbol() == "I":
            atom.SetIntProp("__original_index", -1)
        else:
            atom.SetIntProp("__original_index", 42)
    metal_fragment.GetAtomWithIdx(0).SetIntProp("__original_index", 0)

    monkeypatch.setattr(
        tmos_module, "prepare_complex", lambda *args, **kwargs: complex_mol
    )
    monkeypatch.setattr(tmos_module, "find_metal_index", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        tmos_module,
        "cleave_mol_from_index",
        lambda *_args, **_kwargs: ([ligand_fragment, metal_fragment], [42]),
    )
    monkeypatch.setattr(
        tmos_module,
        "get_geometry_from_mol",
        lambda *_args, **kwargs: ("square-planar", 4, {})
        if kwargs.get("mode") == "angles"
        else ("square-planar", 4, {}),
    )

    ligand_best = {
        "candidate_id": "ligcand-best",
        "rdmol": ligand_fragment,
        "smiles": "[CH3][I]",
        "chemical_formula": "C1H3I1",
        "L-type connectors": [42],
        "X-type connectors": [],
        "total_charge": 0,
        "hanging_bonds": 0,
        "charged_atoms": {},
    }
    monkeypatch.setattr(
        tmos_module,
        "get_ligand_attributes",
        lambda *_args, **kwargs: [ligand_best]
        if kwargs.get("return_all_candidates")
        else ligand_best,
    )

    combinations_out = [
        {
            "ligand_info": [ligand_best],
            "candidate_ids": ["ligcand-best"],
            "number_Ltype_connectors": 1,
            "number_Xtype_connectors": 0,
            "total_ligand_charge": 0,
            "dedupe_group_ids": ["g0"],
            "dedupe_key": ("k0",),
        }
    ]
    scored_out = [
        {
            **combinations_out[0],
            "metal_scoring": {
                "target_complex_charge": 0,
                "target_electron_count": 18,
                "state_scores": [
                    {
                        "oxidation_state": 2,
                        "tm_charge": 2,
                        "tm_electron_count": 18,
                        "predicted_complex_charge": 2,
                        "score_components": {
                            "oxidation_membership_penalty": 0,
                            "charge_consistency_penalty": 2,
                            "electron_count_penalty": 0,
                            "residual_valence_penalty": 0,
                        },
                        "score": 200,
                    }
                ],
                "best_state": {
                    "oxidation_state": 2,
                    "tm_charge": 2,
                    "tm_electron_count": 18,
                    "predicted_complex_charge": 2,
                    "score_components": {},
                    "score": 200,
                },
                "best_score": 200,
            },
        }
    ]
    monkeypatch.setattr(
        tmos_module,
        "_enumerate_ligand_combinations",
        lambda *_args, **_kwargs: combinations_out,
    )
    monkeypatch.setattr(
        tmos_module,
        "_score_ligand_combinations_with_metal",
        lambda *_args, **_kwargs: scored_out,
    )
    monkeypatch.setattr(
        tmos_module,
        "reform_metal_complex",
        lambda *_args, **_kwargs: Chem.MolFromSmiles("[Fe]"),
    )
    monkeypatch.setattr(tmos_module, "mol_to_smiles", lambda *_args, **_kwargs: "[Fe]")
    monkeypatch.setattr(
        tmos_module,
        "get_molecular_formula",
        lambda *_args, **_kwargs: "Fe1",
    )

    outputs = tmos_module.sanitize_complex(complex_mol, return_all_candidates=True)

    assert "__all_candidates__" in outputs
    all_candidates = outputs["__all_candidates__"]
    assert all_candidates["count"] == 1
    assert len(all_candidates["candidates"]) == 1
    candidate = all_candidates["candidates"][0]
    assert candidate["ligand_candidate_ids"] == ["ligcand-best"]
    assert "score_components" in candidate


def test_enumerate_ligand_combinations_non_equivalent_remain_distinct():
    ligand_1_a = {
        "candidate_id": "ligcand-1a",
        "chemical_formula": "C1H1N1",
        "smiles": "[CH]=[N]",
        "total_charge": 0,
        "hanging_bonds": 0,
        "charged_atoms": {},
        "L-type connectors": [10],
        "X-type connectors": [],
    }
    ligand_1_b = {
        "candidate_id": "ligcand-1b",
        "chemical_formula": "C1H1N1",
        "smiles": "[CH][N-]",
        "total_charge": -1,
        "hanging_bonds": 0,
        "charged_atoms": {0: {"formal_charge": -1}},
        "L-type connectors": [],
        "X-type connectors": [10],
    }
    ligand_2_a = {
        "candidate_id": "ligcand-2a",
        "chemical_formula": "C1H1O1",
        "smiles": "[CH]=[O]",
        "total_charge": 0,
        "hanging_bonds": 0,
        "charged_atoms": {},
        "L-type connectors": [20],
        "X-type connectors": [],
    }
    ligand_2_b = {
        "candidate_id": "ligcand-2b",
        "chemical_formula": "C1H1O1",
        "smiles": "[CH][O-]",
        "total_charge": -1,
        "hanging_bonds": 0,
        "charged_atoms": {0: {"formal_charge": -1}},
        "L-type connectors": [],
        "X-type connectors": [20],
    }

    combinations_out = tmos_module._enumerate_ligand_combinations(
        [[ligand_1_a, ligand_1_b], [ligand_2_a, ligand_2_b]]
    )

    assert len(combinations_out) == 4
    assert all(x["dedupe_group_ids"] == ["g0", "g1"] for x in combinations_out)


def test_sanitize_complex_all_candidates_retains_charged_ligand_assignment(monkeypatch):
    complex_mol = Chem.MolFromSmiles("[Fe]")
    ligand_fragment = Chem.MolFromSmiles("[CH3][I]")
    metal_fragment = Chem.MolFromSmiles("[Fe]")

    for atom in ligand_fragment.GetAtoms():
        if atom.GetSymbol() == "I":
            atom.SetIntProp("__original_index", -1)
        else:
            atom.SetIntProp("__original_index", 42)
    metal_fragment.GetAtomWithIdx(0).SetIntProp("__original_index", 0)

    monkeypatch.setattr(
        tmos_module, "prepare_complex", lambda *args, **kwargs: complex_mol
    )
    monkeypatch.setattr(tmos_module, "find_metal_index", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        tmos_module,
        "cleave_mol_from_index",
        lambda *_args, **_kwargs: ([ligand_fragment, metal_fragment], [42]),
    )
    monkeypatch.setattr(
        tmos_module,
        "get_geometry_from_mol",
        lambda *_args, **kwargs: ("square-planar", 4, {})
        if kwargs.get("mode") == "angles"
        else ("square-planar", 4, {}),
    )

    ligand_neutral = {
        "candidate_id": "ligcand-neutral",
        "rdmol": ligand_fragment,
        "smiles": "[CH3][I]",
        "chemical_formula": "C1H3I1",
        "L-type connectors": [42],
        "X-type connectors": [],
        "total_charge": 0,
        "hanging_bonds": 0,
        "charged_atoms": {},
    }
    ligand_charged = {
        "candidate_id": "ligcand-charged",
        "rdmol": ligand_fragment,
        "smiles": "[CH2-][I]",
        "chemical_formula": "C1H2I1",
        "L-type connectors": [],
        "X-type connectors": [42],
        "total_charge": -1,
        "hanging_bonds": 0,
        "charged_atoms": {0: {"formal_charge": -1}},
    }
    monkeypatch.setattr(
        tmos_module,
        "get_ligand_attributes",
        lambda *_args, **kwargs: [ligand_neutral, ligand_charged]
        if kwargs.get("return_all_candidates")
        else ligand_neutral,
    )

    combinations_out = [
        {
            "ligand_info": [ligand_neutral],
            "candidate_ids": ["ligcand-neutral"],
            "number_Ltype_connectors": 1,
            "number_Xtype_connectors": 0,
            "total_ligand_charge": 0,
            "dedupe_group_ids": ["g0"],
            "dedupe_key": ("k0",),
        },
        {
            "ligand_info": [ligand_charged],
            "candidate_ids": ["ligcand-charged"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 1,
            "total_ligand_charge": -1,
            "dedupe_group_ids": ["g0"],
            "dedupe_key": ("k1",),
        },
    ]
    scored_out = [
        {
            **combinations_out[0],
            "metal_scoring": {
                "target_complex_charge": 0,
                "target_electron_count": 18,
                "state_scores": [
                    {
                        "oxidation_state": 2,
                        "tm_charge": 2,
                        "tm_electron_count": 18,
                        "predicted_complex_charge": 2,
                        "score_components": {
                            "oxidation_membership_penalty": 0,
                            "charge_consistency_penalty": 2,
                            "electron_count_penalty": 0,
                            "residual_valence_penalty": 0,
                        },
                        "score": 200,
                    }
                ],
                "best_state": {
                    "oxidation_state": 2,
                    "tm_charge": 2,
                    "tm_electron_count": 18,
                    "predicted_complex_charge": 2,
                    "score_components": {},
                    "score": 200,
                },
                "best_score": 200,
            },
        },
        {
            **combinations_out[1],
            "metal_scoring": {
                "target_complex_charge": 0,
                "target_electron_count": 18,
                "state_scores": [
                    {
                        "oxidation_state": 3,
                        "tm_charge": 3,
                        "tm_electron_count": 17,
                        "predicted_complex_charge": 2,
                        "score_components": {
                            "oxidation_membership_penalty": 0,
                            "charge_consistency_penalty": 2,
                            "electron_count_penalty": 1,
                            "residual_valence_penalty": 0,
                        },
                        "score": 210,
                    }
                ],
                "best_state": {
                    "oxidation_state": 3,
                    "tm_charge": 3,
                    "tm_electron_count": 17,
                    "predicted_complex_charge": 2,
                    "score_components": {},
                    "score": 210,
                },
                "best_score": 210,
            },
        },
    ]
    monkeypatch.setattr(
        tmos_module,
        "_enumerate_ligand_combinations",
        lambda *_args, **_kwargs: combinations_out,
    )
    monkeypatch.setattr(
        tmos_module,
        "_score_ligand_combinations_with_metal",
        lambda *_args, **_kwargs: scored_out,
    )
    monkeypatch.setattr(
        tmos_module,
        "reform_metal_complex",
        lambda *_args, **_kwargs: Chem.MolFromSmiles("[Fe]"),
    )
    monkeypatch.setattr(tmos_module, "mol_to_smiles", lambda *_args, **_kwargs: "[Fe]")
    monkeypatch.setattr(
        tmos_module,
        "get_molecular_formula",
        lambda *_args, **_kwargs: "Fe1",
    )

    outputs = tmos_module.sanitize_complex(complex_mol, return_all_candidates=True)

    assert "__all_candidates__" in outputs
    all_candidates = outputs["__all_candidates__"]["candidates"]
    candidate_ids = [tuple(x["ligand_candidate_ids"]) for x in all_candidates]
    assert ("ligcand-neutral",) in candidate_ids
    assert ("ligcand-charged",) in candidate_ids
