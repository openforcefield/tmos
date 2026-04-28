"""
Unit and regression test for the tmos package.
"""

# Import package, test suite, and other packages as needed
import sys
import json

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from tmos import tmos as tmos_module
from tmos import utils as utils_module


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

    candidates_1 = tmos_module.get_ligand_attributes(ligand, coordinating_indices)
    candidates_2 = tmos_module.get_ligand_attributes(ligand, coordinating_indices)

    assert len(candidates_1) == 2
    assert [c.candidate_id for c in candidates_1] == [
        c.candidate_id for c in candidates_2
    ]
    assert all(c.candidate_id.startswith("ligcand-") for c in candidates_1)

    connector_patterns = {
        (tuple(sorted(c.l_type_connectors)), tuple(sorted(c.x_type_connectors)))
        for c in candidates_1
    }
    assert connector_patterns == {((42,), ()), ((), (42,))}


def test_get_ligand_attributes_default_returns_best_candidate(patch_ligand_enumeration):
    ligand, coordinating_indices = _build_test_ligand()

    candidates = tmos_module.get_ligand_attributes(ligand, coordinating_indices)
    best = candidates[0]  # best candidate is first (lowest sort key)

    assert best.candidate_id.startswith("ligcand-")
    assert best.hanging_bonds == 1
    assert best.l_type_connectors == []
    assert best.x_type_connectors == [42]


def test_enumerate_ligand_combinations_cartesian_totals():
    ligand_a1 = tmos_module.LigandInfo(
        candidate_id="ligcand-a1",
        l_type_connectors=[10],
        x_type_connectors=[],
        total_charge=0,
        hanging_bonds=0,
    )
    ligand_a2 = tmos_module.LigandInfo(
        candidate_id="ligcand-a2",
        l_type_connectors=[],
        x_type_connectors=[10],
        total_charge=-1,
        hanging_bonds=0,
    )
    ligand_b1 = tmos_module.LigandInfo(
        candidate_id="ligcand-b1",
        l_type_connectors=[20, 21],
        x_type_connectors=[],
        total_charge=0,
        hanging_bonds=0,
    )

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
    ligand_1_a = tmos_module.LigandInfo(
        candidate_id="ligcand-1a",
        smiles="[CH]=[N]",
        chemical_formula="C1H1N1",
        total_charge=0,
        hanging_bonds=0,
        charged_atoms={},
        l_type_connectors=[10],
        x_type_connectors=[],
    )
    ligand_1_b = tmos_module.LigandInfo(
        candidate_id="ligcand-1b",
        smiles="[CH][N-]",
        chemical_formula="C1H1N1",
        total_charge=-1,
        hanging_bonds=0,
        charged_atoms={0: {"formal_charge": -1}},
        l_type_connectors=[],
        x_type_connectors=[10],
    )
    ligand_2_a = tmos_module.LigandInfo(
        candidate_id="ligcand-2a",
        smiles="[CH]=[N]",
        chemical_formula="C1H1N1",
        total_charge=0,
        hanging_bonds=0,
        charged_atoms={},
        l_type_connectors=[20],
        x_type_connectors=[],
    )
    ligand_2_b = tmos_module.LigandInfo(
        candidate_id="ligcand-2b",
        smiles="[CH][N-]",
        chemical_formula="C1H1N1",
        total_charge=-1,
        hanging_bonds=0,
        charged_atoms={0: {"formal_charge": -1}},
        l_type_connectors=[],
        x_type_connectors=[20],
    )

    combinations_out = tmos_module._enumerate_ligand_combinations(
        [[ligand_1_a, ligand_1_b], [ligand_2_a, ligand_2_b]]
    )

    assert (
        len(combinations_out) == 4
    )  # no symmetry reduction; (1a,2b) and (1b,2a) are distinct by candidate_id
    combo_signatures = {
        (
            x["number_Ltype_connectors"],
            x["number_Xtype_connectors"],
            x["total_ligand_charge"],
        )
        for x in combinations_out
    }
    assert combo_signatures == {(2, 0, 0), (1, 1, -1), (0, 2, -2)}


def test_score_and_flatten_states_prefers_charge_consistency(monkeypatch):
    tm_mol = Chem.MolFromSmiles("[Fe]")

    # Keep metal-state options deterministic for scoring assertions.
    monkeypatch.setattr(
        tmos_module,
        "get_tm_attributes",
        lambda *_args, **_kwargs: ([2], np.array([2]), np.array([18])),
    )

    li_a = tmos_module.LigandInfo(
        hanging_bonds=0, l_type_connectors=[], x_type_connectors=[]
    )
    li_b = tmos_module.LigandInfo(
        hanging_bonds=0, l_type_connectors=[], x_type_connectors=[]
    )
    combinations_in = [
        {
            "ligand_info": [li_a],
            "candidate_ids": ["ligcand-a"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 2,
            "total_ligand_charge": -2,
        },
        {
            "ligand_info": [li_b],
            "candidate_ids": ["ligcand-b"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 2,
            "total_ligand_charge": 0,
        },
    ]

    scored = tmos_module._score_and_flatten_states(tm_mol, combinations_in)

    assert scored[0].ligands.candidate_ids == ["ligcand-a"]
    assert scored[0].predicted_complex_charge == 0
    assert scored[0].score < scored[1].score


def test_score_and_flatten_states_penalizes_residual_valence(monkeypatch):
    tm_mol = Chem.MolFromSmiles("[Fe]")

    monkeypatch.setattr(
        tmos_module,
        "get_tm_attributes",
        lambda *_args, **_kwargs: ([2], np.array([2]), np.array([18])),
    )

    li_clean = tmos_module.LigandInfo(
        hanging_bonds=0, l_type_connectors=[], x_type_connectors=[]
    )
    li_residual = tmos_module.LigandInfo(
        hanging_bonds=3, l_type_connectors=[], x_type_connectors=[]
    )
    combinations_in = [
        {
            "ligand_info": [li_clean],
            "candidate_ids": ["ligcand-clean"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 2,
            "total_ligand_charge": -2,
        },
        {
            "ligand_info": [li_residual],
            "candidate_ids": ["ligcand-residual"],
            "number_Ltype_connectors": 0,
            "number_Xtype_connectors": 2,
            "total_ligand_charge": -2,
        },
    ]

    scored = tmos_module._score_and_flatten_states(tm_mol, combinations_in)

    assert scored[0].ligands.candidate_ids == ["ligcand-clean"]
    assert scored[0].score < scored[1].score


def test_sanitize_complex_returns_list_of_complex_states(monkeypatch):
    """sanitize_complex should return a list of ComplexState objects sorted by score."""
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
        lambda *_args, **kwargs: ("square-planar", 4, {}),
    )

    li_best = tmos_module.LigandInfo(
        candidate_id="ligcand-best",
        rdmol=ligand_fragment,
        smiles="[CH3][I]",
        chemical_formula="C1H3I1",
        l_type_connectors=[42],
        x_type_connectors=[],
        total_charge=0,
        hanging_bonds=0,
        charged_atoms={},
    )
    monkeypatch.setattr(
        tmos_module, "get_ligand_attributes", lambda *_args, **_kwargs: [li_best]
    )

    sc = tmos_module.ScoreComponents(
        target_complex_charge=0,
        target_electron_count=18,
        oxidation_membership_penalty=0,
        charge_consistency_penalty=2,
        electron_count_penalty=0,
        residual_valence_penalty=0,
        negative_charge_with_xtype_penalty=0,
    )
    mi = tmos_module.MetalInfo(
        symbol="Fe", oxidation_state=2, charge=2, electron_count=18
    )
    ls = tmos_module.LigandSummary(
        ligand_info=[li_best],
        candidate_ids=["ligcand-best"],
        number_Ltype_connectors=1,
        number_Xtype_connectors=0,
        total_charge=0,
    )
    scored_out = [
        tmos_module.ComplexState(
            score=200,
            score_components=sc,
            predicted_complex_charge=2,
            metal=mi,
            ligands=ls,
        )
    ]

    monkeypatch.setattr(
        tmos_module, "_enumerate_ligand_combinations", lambda *_a, **_k: [{}]
    )
    monkeypatch.setattr(
        tmos_module, "_score_and_flatten_states", lambda *_a, **_k: scored_out
    )
    monkeypatch.setattr(
        tmos_module,
        "reform_metal_complex",
        lambda *_a, **_k: Chem.MolFromSmiles("[Fe]"),
    )
    monkeypatch.setattr(tmos_module, "mol_to_smiles", lambda *_a, **_k: "[Fe]")
    monkeypatch.setattr(tmos_module, "get_molecular_formula", lambda *_a, **_k: "Fe1")

    outputs = tmos_module.sanitize_complex(complex_mol, score_cutoff=None)

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], tmos_module.ComplexState)
    assert outputs[0].score == 200
    assert outputs[0].metal.symbol == "Fe"


def test_enumerate_ligand_combinations_non_equivalent_remain_distinct():
    ligand_1_a = tmos_module.LigandInfo(
        candidate_id="ligcand-1a",
        smiles="[CH]=[N]",
        chemical_formula="C1H1N1",
        total_charge=0,
        hanging_bonds=0,
        charged_atoms={},
        l_type_connectors=[10],
        x_type_connectors=[],
    )
    ligand_1_b = tmos_module.LigandInfo(
        candidate_id="ligcand-1b",
        smiles="[CH][N-]",
        chemical_formula="C1H1N1",
        total_charge=-1,
        hanging_bonds=0,
        charged_atoms={0: {"formal_charge": -1}},
        l_type_connectors=[],
        x_type_connectors=[10],
    )
    ligand_2_a = tmos_module.LigandInfo(
        candidate_id="ligcand-2a",
        smiles="[CH]=[O]",
        chemical_formula="C1H1O1",
        total_charge=0,
        hanging_bonds=0,
        charged_atoms={},
        l_type_connectors=[20],
        x_type_connectors=[],
    )
    ligand_2_b = tmos_module.LigandInfo(
        candidate_id="ligcand-2b",
        smiles="[CH][O-]",
        chemical_formula="C1H1O1",
        total_charge=-1,
        hanging_bonds=0,
        charged_atoms={0: {"formal_charge": -1}},
        l_type_connectors=[],
        x_type_connectors=[20],
    )

    combinations_out = tmos_module._enumerate_ligand_combinations(
        [[ligand_1_a, ligand_1_b], [ligand_2_a, ligand_2_b]]
    )

    assert len(combinations_out) == 4


def test_sanitize_complex_two_candidates_both_retained(monkeypatch):
    """Both ligand assignments should appear when score_cutoff is permissive."""
    complex_mol = Chem.MolFromSmiles("[Fe]")
    ligand_fragment = Chem.MolFromSmiles("[CH3][I]")
    metal_fragment = Chem.MolFromSmiles("[Fe]")

    for atom in ligand_fragment.GetAtoms():
        if atom.GetSymbol() == "I":
            atom.SetIntProp("__original_index", -1)
        else:
            atom.SetIntProp("__original_index", 42)
    metal_fragment.GetAtomWithIdx(0).SetIntProp("__original_index", 0)

    monkeypatch.setattr(tmos_module, "prepare_complex", lambda *a, **k: complex_mol)
    monkeypatch.setattr(tmos_module, "find_metal_index", lambda *a, **k: 0)
    monkeypatch.setattr(
        tmos_module,
        "cleave_mol_from_index",
        lambda *a, **k: ([ligand_fragment, metal_fragment], [42]),
    )
    monkeypatch.setattr(
        tmos_module, "get_geometry_from_mol", lambda *a, **k: ("square-planar", 4, {})
    )

    li_neutral = tmos_module.LigandInfo(
        candidate_id="ligcand-neutral",
        rdmol=ligand_fragment,
        smiles="[CH3][I]",
        chemical_formula="C1H3I1",
        l_type_connectors=[42],
        x_type_connectors=[],
        total_charge=0,
        hanging_bonds=0,
        charged_atoms={},
    )
    li_charged = tmos_module.LigandInfo(
        candidate_id="ligcand-charged",
        rdmol=ligand_fragment,
        smiles="[CH2-][I]",
        chemical_formula="C1H2I1",
        l_type_connectors=[],
        x_type_connectors=[42],
        total_charge=-1,
        hanging_bonds=0,
        charged_atoms={0: {"formal_charge": -1}},
    )
    monkeypatch.setattr(
        tmos_module, "get_ligand_attributes", lambda *a, **k: [li_neutral, li_charged]
    )

    def _state(candidate_id, li, score):
        sc = tmos_module.ScoreComponents(
            target_complex_charge=0,
            target_electron_count=18,
            oxidation_membership_penalty=0,
            charge_consistency_penalty=score // 100,
            electron_count_penalty=0,
            residual_valence_penalty=0,
            negative_charge_with_xtype_penalty=0,
        )
        mi = tmos_module.MetalInfo(
            symbol="Fe", oxidation_state=2, charge=2, electron_count=18
        )
        ls = tmos_module.LigandSummary(
            ligand_info=[li],
            candidate_ids=[candidate_id],
            number_Ltype_connectors=len(li.l_type_connectors),
            number_Xtype_connectors=len(li.x_type_connectors),
            total_charge=li.total_charge,
        )
        return tmos_module.ComplexState(
            score=score,
            score_components=sc,
            predicted_complex_charge=2,
            metal=mi,
            ligands=ls,
        )

    scored_out = [
        _state("ligcand-neutral", li_neutral, 200),
        _state("ligcand-charged", li_charged, 210),
    ]
    monkeypatch.setattr(
        tmos_module, "_enumerate_ligand_combinations", lambda *a, **k: [{}, {}]
    )
    monkeypatch.setattr(
        tmos_module, "_score_and_flatten_states", lambda *a, **k: scored_out
    )
    monkeypatch.setattr(
        tmos_module, "reform_metal_complex", lambda *a, **k: Chem.MolFromSmiles("[Fe]")
    )
    monkeypatch.setattr(tmos_module, "mol_to_smiles", lambda *a, **k: "[Fe]")
    monkeypatch.setattr(tmos_module, "get_molecular_formula", lambda *a, **k: "Fe1")

    outputs = tmos_module.sanitize_complex(
        complex_mol, score_cutoff=None, n_results=None
    )

    candidate_ids = [s.ligands.candidate_ids[0] for s in outputs]
    assert "ligcand-neutral" in candidate_ids
    assert "ligcand-charged" in candidate_ids


def test_sanitize_complex_score_cutoff_filters_states(monkeypatch):
    """States with score >= score_cutoff should be excluded from the output."""
    complex_mol = Chem.MolFromSmiles("[Fe]")
    ligand_fragment = Chem.MolFromSmiles("[CH3][I]")
    metal_fragment = Chem.MolFromSmiles("[Fe]")

    for atom in ligand_fragment.GetAtoms():
        if atom.GetSymbol() == "I":
            atom.SetIntProp("__original_index", -1)
        else:
            atom.SetIntProp("__original_index", 42)
    metal_fragment.GetAtomWithIdx(0).SetIntProp("__original_index", 0)

    monkeypatch.setattr(tmos_module, "prepare_complex", lambda *a, **k: complex_mol)
    monkeypatch.setattr(tmos_module, "find_metal_index", lambda *a, **k: 0)
    monkeypatch.setattr(
        tmos_module,
        "cleave_mol_from_index",
        lambda *a, **k: ([ligand_fragment, metal_fragment], [42]),
    )
    monkeypatch.setattr(
        tmos_module, "get_geometry_from_mol", lambda *a, **k: ("octahedral", 6, {})
    )

    li = tmos_module.LigandInfo(
        candidate_id="ligcand-co",
        rdmol=ligand_fragment,
        smiles="[CH3][I]",
        chemical_formula="C1H3I1",
        l_type_connectors=[42],
        x_type_connectors=[],
        total_charge=0,
        hanging_bonds=0,
        charged_atoms={},
    )
    monkeypatch.setattr(tmos_module, "get_ligand_attributes", lambda *a, **k: [li])
    monkeypatch.setattr(
        tmos_module, "reform_metal_complex", lambda *a, **k: Chem.MolFromSmiles("[Fe]")
    )
    monkeypatch.setattr(tmos_module, "mol_to_smiles", lambda *a, **k: "[Fe]")
    monkeypatch.setattr(tmos_module, "get_molecular_formula", lambda *a, **k: "Fe1")

    def _make_state(score):
        sc = tmos_module.ScoreComponents(
            target_complex_charge=0,
            target_electron_count=18,
            oxidation_membership_penalty=score // 1000,
            charge_consistency_penalty=0,
            electron_count_penalty=0,
            residual_valence_penalty=0,
            negative_charge_with_xtype_penalty=0,
        )
        mi = tmos_module.MetalInfo(
            symbol="Fe", oxidation_state=2, charge=2, electron_count=18
        )
        ls = tmos_module.LigandSummary(
            ligand_info=[li],
            candidate_ids=["ligcand-co"],
            number_Ltype_connectors=1,
            number_Xtype_connectors=0,
            total_charge=0,
        )
        return tmos_module.ComplexState(
            score=score,
            score_components=sc,
            predicted_complex_charge=2,
            metal=mi,
            ligands=ls,
        )

    scored_out = [_make_state(0), _make_state(1000)]
    monkeypatch.setattr(
        tmos_module, "_enumerate_ligand_combinations", lambda *a, **k: [{}]
    )
    monkeypatch.setattr(
        tmos_module, "_score_and_flatten_states", lambda *a, **k: scored_out
    )

    outputs = tmos_module.sanitize_complex(complex_mol, score_cutoff=1000)

    assert len(outputs) == 1
    assert outputs[0].score == 0


def test_complex_state_to_dict_multimetal_compatible_schema():
    """Serialized state includes a metals list for forward compatibility."""
    mol = Chem.MolFromSmiles("C=C")
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Point3D(1.33, 0.0, 0.0))
    mol.AddConformer(conf)

    state = tmos_module.ComplexState(
        score=12,
        score_components=tmos_module.ScoreComponents(
            target_complex_charge=0,
            target_electron_count=18,
            oxidation_membership_penalty=0,
            charge_consistency_penalty=0,
            electron_count_penalty=1,
            residual_valence_penalty=2,
            negative_charge_with_xtype_penalty=0,
            oxidation_state_preference_penalty=0,
            geometry_oxidation_preference_penalty=0,
        ),
        predicted_complex_charge=0,
        metal=tmos_module.MetalInfo(
            symbol="Fe",
            oxidation_state=2,
            charge=2,
            electron_count=18,
        ),
        ligands=tmos_module.LigandSummary(
            ligand_info=[],
            candidate_ids=[],
            number_Ltype_connectors=0,
            number_Xtype_connectors=0,
            total_charge=-2,
        ),
        complex=tmos_module.ComplexInfo(
            rdmol=mol,
            smiles="C=C",
            formula="C2",
            charge=0,
            number_metal_connections=0,
            geometry_type="linear",
        ),
    )

    payload = state.to_dict(include_graph=True, coordinate_units="angstrom")

    assert payload["schema_version"] == 1
    assert isinstance(payload["metals_summary"], dict)
    assert payload["metals_summary"]["metal_info"][0]["symbol"] == "Fe"
    assert payload["metals_summary"]["total_charge"] == 2
    assert payload["metals_summary"]["total_electron_count"] == 18
    assert isinstance(payload["metals"], list)
    assert len(payload["metals"]) == 1
    assert payload["metals"][0]["symbol"] == "Fe"
    assert "metal" not in payload
    assert payload["complex"]["graph"]["atoms"][0]["symbol"] == "C"
    assert payload["complex"]["graph"]["bonds"][0]["order"] == 2.0
    assert payload["complex"]["graph"]["positions"]["units"] == "angstrom"
    assert len(payload["complex"]["graph"]["positions"]["coordinates"]) == 2


def test_complex_state_to_dict_can_skip_graph_payload():
    """Graph export should be optional for lightweight serialization."""
    state = tmos_module.ComplexState(
        score=0,
        score_components=tmos_module.ScoreComponents(
            target_complex_charge=0,
            target_electron_count=18,
            oxidation_membership_penalty=0,
            charge_consistency_penalty=0,
            electron_count_penalty=0,
            residual_valence_penalty=0,
            negative_charge_with_xtype_penalty=0,
        ),
        predicted_complex_charge=0,
        metal=tmos_module.MetalInfo(
            symbol="Fe",
            oxidation_state=2,
            charge=2,
            electron_count=18,
        ),
        ligands=tmos_module.LigandSummary(
            ligand_info=[],
            candidate_ids=[],
            number_Ltype_connectors=0,
            number_Xtype_connectors=0,
            total_charge=-2,
        ),
        complex=tmos_module.ComplexInfo(
            rdmol=Chem.MolFromSmiles("C"),
            smiles="C",
            formula="C1",
            charge=0,
            number_metal_connections=0,
            geometry_type="unknown",
        ),
    )

    payload = state.to_dict(include_graph=False)

    assert "graph" not in payload["complex"]
    assert payload["metals"][0]["symbol"] == "Fe"


def test_save_complex_states_to_json_writes_schema_payload(tmp_path):
    """Utility should write serialized state payloads to disk."""
    mol = Chem.MolFromSmiles("C=C")
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Point3D(1.33, 0.0, 0.0))
    mol.AddConformer(conf)

    state = tmos_module.ComplexState(
        score=12,
        score_components=tmos_module.ScoreComponents(
            target_complex_charge=0,
            target_electron_count=18,
            oxidation_membership_penalty=0,
            charge_consistency_penalty=0,
            electron_count_penalty=1,
            residual_valence_penalty=2,
            negative_charge_with_xtype_penalty=0,
        ),
        predicted_complex_charge=0,
        metal=tmos_module.MetalInfo(
            symbol="Fe",
            oxidation_state=2,
            charge=2,
            electron_count=18,
        ),
        ligands=tmos_module.LigandSummary(
            ligand_info=[],
            candidate_ids=[],
            number_Ltype_connectors=0,
            number_Xtype_connectors=0,
            total_charge=-2,
        ),
        complex=tmos_module.ComplexInfo(
            rdmol=mol,
            smiles="C=C",
            formula="C2",
            charge=0,
            number_metal_connections=0,
            geometry_type="linear",
        ),
    )

    out = tmp_path / "states.json"
    utils_module.save_complex_states_to_json([state], str(out), include_graph=True)

    with out.open() as f:
        payload = json.load(f)

    assert isinstance(payload, list)
    assert payload[0]["metals_summary"]["metal_info"][0]["symbol"] == "Fe"
    assert payload[0]["metals"][0]["symbol"] == "Fe"
    assert payload[0]["complex"]["graph"]["bonds"][0]["order"] == 2.0


def test_save_complex_states_to_json_respects_include_graph_false(tmp_path):
    """Graph payload should be omitted when include_graph is False."""
    state = tmos_module.ComplexState(
        score=0,
        score_components=tmos_module.ScoreComponents(
            target_complex_charge=0,
            target_electron_count=18,
            oxidation_membership_penalty=0,
            charge_consistency_penalty=0,
            electron_count_penalty=0,
            residual_valence_penalty=0,
            negative_charge_with_xtype_penalty=0,
        ),
        predicted_complex_charge=0,
        metal=tmos_module.MetalInfo(
            symbol="Fe",
            oxidation_state=2,
            charge=2,
            electron_count=18,
        ),
        ligands=tmos_module.LigandSummary(
            ligand_info=[],
            candidate_ids=[],
            number_Ltype_connectors=0,
            number_Xtype_connectors=0,
            total_charge=-2,
        ),
        complex=tmos_module.ComplexInfo(
            rdmol=Chem.MolFromSmiles("C"),
            smiles="C",
            formula="C1",
            charge=0,
            number_metal_connections=0,
            geometry_type="unknown",
        ),
    )

    out = tmp_path / "states_no_graph.json"
    utils_module.save_complex_states_to_json([state], str(out), include_graph=False)

    with out.open() as f:
        payload = json.load(f)

    assert "graph" not in payload[0]["complex"]
