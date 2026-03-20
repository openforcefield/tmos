import networkx as nx
import pytest
from rdkit.Chem.rdmolfiles import MolFromSmiles
import json
from pathlib import Path

import tmos.build_rdmol as build_rdmol
from tmos.graph_mapping import mol_to_graph

with open(Path(__file__).parent / "error_check3.json") as f:
    TEST_CASES = json.load(f)

OUTPUT_REFERENCE = [
    {
        "smiles": "[H:32][c:11]1[c:16]([c:19]2[c:20]([c:18]([c:14]1[C:22]([H:39])([H:40])[O:4][c:13]3[c:10]([c:15]([c:9]([c:12]([c:8]3[H:29])[F:1])[H:30])[C:27]4([C:25]([C:23]([O:5][C:24]([C:26]4([H:47])[H:48])([H:43])[H:44])([H:41])[H:42])([H:45])[H:46])[O:6][C:21]([H:36])([H:37])[H:38])[H:31])[H:35])[c:17]([n:2][c:7]([n:3]2)[H:28])[H:34])[H:33]"
    },
    {
        "smiles": "[H:26][c:13]1[c:10]([c:11]([c:14]([c:17]2[c:16]1[n:5][c:9]3[c:12]([c:15]2[H:28])[N:20]([C:7](=[O:1])[N:19]3[H:30])[H:31])[H:27])[O:6][C:21]([H:32])([H:33])[C:23]([H:36])([H:37])[C:24]([H:38])([H:39])[C:22]([H:34])([H:35])[C:8]4=[N:4][N:2]=[N:3][N:18]4[H:29])[H:25]"
    },
    {
        "smiles": "[H:24][C:12]1=[C:13]([C:8](=[N:3][H:20])[N:18]([N:2]=[C:7]1[C:15]2=[C:10]([S:5][C:9](=[C:11]2[H:23])[H:21])[H:22])[C:19]([H:28])([H:29])[C:16]3=[C:14]([C:6](=[O:1])[N:17]([O:4]3)[H:27])[H:26])[H:25]"
    },
    {
        "smiles": "[O:1]=[S:27](=[O:2])([O:5][c:16]1[c:11]([H:32])[c:13]([H:34])[c:18]2[c:17]([c:12]1[H:33])[C:7]([C:26]1([H:45])[C:24]([H:41])([H:42])[C:22]([H:37])([H:38])[N:21]([H:36])[C:23]([H:39])([H:40])[C:25]1([H:43])[H:44])=[C:6]([H:28])[N:20]2[H:35])[c:19]1[c:14]([F:3])[c:9]([H:30])[c:8]([H:29])[c:10]([H:31])[c:15]1[F:4]"
    },
    {
        "smiles": "[O:1]=[C:6]([c:13]1[c:9]([H:21])[c:11]([H:23])[c:14]([N+:16](=[O:3])[O-:4])[c:12]([H:24])[c:10]1[H:22])[C-:15](/[C:8](=[N+:17](/[O:5][C:7](=[O:2])[C:18]([H:26])([H:27])[H:28])[C:19]([H:29])([H:30])[H:31])[H:20])[H:25]"
    },
    {
        "smiles": "[H:30][c:10]1[c:11]([c:15]([c:20]([c:16]([c:12]1[H:32])[H:36])[C:5](=[O:1])[N:24]([H:43])[C:26]2([C:6](=[O:2])[O:4][C:9](=[C:8]([H:28])[H:29])[C:25]([C:27]2([H:47])[c:21]3[c:17]([c:13]([c:19]([c:14]([c:18]3[H:38])[H:34])[N:3]=[C:7]([N:22]([H:39])[H:40])[N:23]([H:41])[H:42])[H:33])[H:37])([H:44])[H:45])[H:46])[H:35])[H:31]"
    },
    {
        "smiles": "[H:21][C:11]1=[C:10]([S:5][C:14](=[C:12]1[H:22])[C:18]([H:27])([H:28])[N:17]2[C:7]3=[N:2][C:6](=[N:1][C:13]3=[C:9]([N:3]=[C:8]2[N:16]([H:25])[H:26])[O:4][H:19])[N:15]([H:23])[H:24])[H:20]"
    },
    {
        "smiles": "[H:31][c:9]1[c:13]([c:19]([c:14]([c:18]([c:12]1[H:34])[O:4][P:27](=[O:1])([O-:2])[O:3][c:17]2[c:10]([c:15]([c:20]([c:16]([c:11]2[H:33])[H:38])[C:26]([C:23]([H:41])([H:42])[H:43])([C:24]([H:44])([H:45])[H:46])[C:25]([H:47])([H:48])[H:49])[H:37])[H:32])[H:36])[C:22]([H:39])([H:40])[N+:21]3=[C:8]([S:5][C:6](=[C:7]3[H:29])[H:28])[H:30])[H:35]"
    },
    {
        "smiles": "[H:23][C:8]1=[C:10]([C:14]([C:15]([C:11]2=[C:9]1[O:5][C@:19]([C:18]([C@:20]2([C:17]([H:33])([H:34])[C:7](=[O:1])[O:4][H:21])[O:6][H:22])([H:35])[H:36])([H:37])[C:16]([H:30])([H:31])[H:32])([C:13]([H:25])([H:26])[H:27])[N+:12](=[O:2])[O-:3])([H:28])[H:29])[H:24]"
    },
    {
        "smiles": "[H:34][c:16]1[c:12]([c:19]([c:13]([c:17]([c:22]1/[C:10](=[C:11](\\[H:29])/[N+:24](=[O:2])[O-:4])/[H:28])[H:35])[H:31])[O:8][S+:25](=[O:3])([c:23]2[c:18]([c:14]([c:21]([c:15]([c:20]2[O:7][H:27])[H:33])[C:9](=[O:1])[O:6][H:26])[H:32])[H:36])[O-:5])[H:30]"
    },
    {
        "smiles": "[H:19][C:9]1([C:11]([S:13]([C:12]([C:10]1([H:21])[O:6][C:8]([H:17])([H:18])[C:4]#[C:3][C:7]([H:15])([H:16])[O:5][H:14])([H:24])[H:25])([O-:1])[O-:2])([H:22])[H:23])[H:20]"
    },
    {
        "smiles": "[H:28][c:11]1[c:12]([c:18]([c:22]2[c:19]([c:13]1[H:30])[C:8](=[C:7]([C:5]#[N:1])[C:6]#[N:2])/[C:10](=[C:9](\\[H:27])/[c:20]3[c:14]([c:16]([c:21]([c:17]([c:15]3[H:32])[H:34])[N:23]([C:24]([H:36])([H:37])[H:38])[C:25]([H:39])([H:40])[H:41])[H:33])[H:31])/[S:26]2([O-:3])[O-:4])[H:35])[H:29]"
    },
]


def _ref_penalty(smiles: str) -> int:
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid reference SMILES: {smiles}")
    return build_rdmol.molecular_penalty(mol)


def _check_graph_consistent(rdmol, ref_smiles: str) -> None:
    """Assert the heavy-atom graph of *rdmol* is isomorphic to the reference SMILES graph."""
    ref_mol = MolFromSmiles(ref_smiles)
    if ref_mol is None:
        ref_mol = MolFromSmiles(ref_smiles, sanitize=False)
    assert ref_mol is not None, f"Invalid reference SMILES: {ref_smiles}"

    g_actual = mol_to_graph(rdmol, remove_hydrogens=True)
    g_ref = mol_to_graph(ref_mol, remove_hydrogens=True)

    n_actual, n_ref = g_actual.number_of_nodes(), g_ref.number_of_nodes()
    assert (
        n_actual == n_ref
    ), f"Heavy-atom count mismatch: got {n_actual}, expected {n_ref}"

    e_actual, e_ref = g_actual.number_of_edges(), g_ref.number_of_edges()
    assert (
        e_actual == e_ref
    ), f"Heavy-atom bond count mismatch: got {e_actual}, expected {e_ref}"

    node_match = nx.algorithms.isomorphism.categorical_node_match("symbol", None)
    gm = nx.algorithms.isomorphism.GraphMatcher(g_actual, g_ref, node_match=node_match)
    assert gm.is_isomorphic(), "Heavy-atom graph is not isomorphic to reference SMILES"


assert len(TEST_CASES) == len(OUTPUT_REFERENCE)


REFERENCE_CASES = [
    pytest.param(case, reference, id=f"structure_{i + 1}")
    for i, (case, reference) in enumerate(zip(TEST_CASES, OUTPUT_REFERENCE))
]


@pytest.mark.parametrize(
    ("case", "reference"),
    REFERENCE_CASES,
)
def test_determine_bonds_penalty(
    case: dict,
    reference: dict,
) -> None:
    rdmol = build_rdmol.xyz_to_rdkit(
        case["symbols"],
        case["coordinates"],
        ignore_scale=True,
    )

    rdmol = build_rdmol.determine_bonds(rdmol, charge=case["charge"])

    radical_atoms = [
        (a.GetIdx(), a.GetAtomicNum())
        for a in rdmol.GetAtoms()
        if a.GetNumRadicalElectrons() > 0
    ]
    assert not radical_atoms, f"Output molecule contains radical atoms: {radical_atoms}"

    _check_graph_consistent(rdmol, reference["smiles"])

    actual_penalty = build_rdmol.molecular_penalty(rdmol)
    reference_penalty = _ref_penalty(reference["smiles"])
    assert (
        actual_penalty <= reference_penalty
    ), f"Penalty increased: {actual_penalty} > {reference_penalty}"
