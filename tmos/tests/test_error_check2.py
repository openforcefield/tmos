import networkx as nx
import pytest
from rdkit.Chem.rdmolfiles import MolFromSmiles
import json
from pathlib import Path

import tmos.build_rdmol as build_rdmol
from tmos.graph_mapping import mol_to_graph

with open(Path(__file__).parent / "error_check2.json") as f:
    TEST_CASES = json.load(f)

OUTPUT_REFERENCE = [
    {
        "smiles": "[H:32][C@:15]1([C:6](=[O:1])[N:9]([C:11]([C:12]([C:13]([C:14]1([H:30])[H:31])([H:28])[H:29])([H:26])[H:27])([H:24])[H:25])[P:16](=[O:2])([N:8]([H:21])[H:22])[N:10]([H:23])[S:17](=[O:3])(=[O:4])[O:5][H:18])[N:7]([H:19])[H:20]"
    },
    {
        "smiles": "[O:1]=[P:21]([O:6][H:23])([O:7][H:24])[C:20]([F:3])([C:19]([c:17]1[c:14]([H:30])[n:5][c:13]([H:29])[c:16]([C:11]2=[C:12]([H:28])[N:18]([H:32])[N:4]=[C:10]2[H:27])[c:15]1[H:31])([H:33])[H:34])[P:22](=[O:2])([O:8][H:25])[O:9][H:26]"
    },
    {
        "smiles": "[H:11][C:3]\\1=[C:4]([C:7](/[C:6]1=[C:5](/[H:13])\\[C:2](=[C:1]([H:8])[H:9])[H:10])([H:14])[H:15])[H:12]"
    },
    {
        "smiles": "[O:1]=[P:11]12[O:5][P:12]3(=[O:2])[O:7][P:13](=[O:3])([O:6]1)[O:10][P:14](=[O:4])([O:8]2)[O:9]3"
    },
    {
        "smiles": "[O:1]=[C:3]1[C:15]([C:5]([H:17])([H:18])[H:19])([C:6]([H:20])([H:21])[H:22])[C:4](=[N:2][C:14]2([H:39])[C:12]([H:35])([H:36])[C:10]([H:31])([H:32])[C:9]([H:29])([H:30])[C:11]([H:33])([H:34])[C:13]2([H:37])[H:38])[C:16]1([C:7]([H:23])([H:24])[H:25])[C:8]([H:26])([H:27])[H:28]"
    },
    {
        "smiles": "[H:18][C:9]([H:19])([H:20])[C:16]([H:32])([C:15]1([C:6](=[O:1])[N:8]([C:7]1=[O:2])[C:14]([H:30])([H:31])[C:13]([F:3])([F:4])[F:5])[C:17]([H:33])([C:11]([H:24])([H:25])[H:26])[C:12]([H:27])([H:28])[H:29])[C:10]([H:21])([H:22])[H:23]"
    },
    {
        "smiles": "[H:21][C:7]1=[C:9]([C:18]2([C:10]1=[C:8]([C:5](=[O:1])[O:4][C:14]2([H:31])[H:32])[C:11]([H:23])([H:24])[O:3][H:20])[C@@:19]3([C:6](=[O:2])[C:15]([C:16]([C:17]3([H:35])[H:36])([C:12]([H:25])([H:26])[H:27])[C:13]([H:28])([H:29])[H:30])([H:33])[H:34])[H:37])[H:22]"
    },
    {"smiles": "[H:10][O:5][S:8](=[O:1])(=[O:2])[S:7][S:9](=[O:3])(=[O:4])[O:6][H:11]"},
    {"smiles": "[H:7][N:5]([H:8])[C:3]1=[N:1][C:4](=[N:2]1)[N:6]([H:9])[H:10]"},
    {
        "smiles": "[O:1]=[P:23]1([N:15]([C:19]([H:35])([H:36])[H:37])[C:20]([H:38])([H:39])[H:40])[N:17]([c:13]2[c:9]([H:31])[c:5]([H:27])[c:3]([H:25])[c:6]([H:28])[c:10]2[H:32])[P:24](=[O:2])([N:16]([C:21]([H:41])([H:42])[H:43])[C:22]([H:44])([H:45])[H:46])[N:18]1[c:14]1[c:11]([H:33])[c:7]([H:29])[c:4]([H:26])[c:8]([H:30])[c:12]1[H:34]"
    },
    {"smiles": "[H:4][N:2]([H:5])[S:1][N:3]([H:6])[H:7]"},
    {
        "smiles": "[H:12][N:7]([H:13])[C:11]([C:9]([F:1])([F:2])[F:3])([C:10]([F:4])([F:5])[F:6])[N:8]([H:14])[H:15]"
    },
    {"smiles": "[H:6][N:3]([H:7])[P:5]([H:10])([N:4]([H:8])[H:9])([F:1])[F:2]"},
    {
        "smiles": "[H:19][c:9]1[c:8]([c:10]([c:12]2[c:13]([c:11]1[H:21])[N:16]([C:17]3([S:4]2)[N:14]([C:6](=[O:2])[C:5](=[O:1])[C:7](=[O:3])[N:15]3[H:23])[H:22])[H:24])[H:20])[H:18]"
    },
    {
        "smiles": "[H:27][c:6]1[c:8]([c:12]([c:17]([c:13]([c:9]1[H:30])[H:34])[C:26]([H:40])([H:41])[n+:24]2[c:20]([c:18]([n+:23]([c:19]([c:21]2[H:38])[H:36])[c:16]3[c:11]([c:7]([c:10]([c:15]4[c:14]3[N:4]=[C:5]([N:22]4[H:39])[C:25]([F:1])([F:2])[F:3])[H:31])[H:28])[H:32])[H:35])[H:37])[H:33])[H:29]"
    },
    {
        "smiles": "[O:1]=[C:6]1[C:8]([C:7]2=[N:2][N:18]([H:28])[N:20]([H:30])[N:19]2[H:29])=[C:9]([N:17]([c:15]2[c:11]([H:24])[c:13]([O:3][C:21]([H:31])([H:32])[H:33])[c:10]([H:23])[c:14]([O:4][C:22]([H:34])([H:35])[H:36])[c:12]2[H:25])[H:27])[S:5][N:16]1[H:26]"
    },
    {"smiles": "[H:9][C:5](=[S:7](=[O:1])=[O:2])[C:6](=[S:8](=[O:3])=[O:4])[H:10]"},
    {
        "smiles": "[H:21][c:7]1[c:6]([c:9]([c:8]([c:11]2[c:10]1[C:3]3=[C:5]([C:13]([C:14]2([H:28])[H:29])([H:26])[H:27])[P+:17]([C:4](=[C:2]3[H:18])[H:19])([C:15]([H:30])([H:31])[H:32])[C:16]([H:33])([H:34])[H:35])[H:22])[O:1][C:12]([H:23])([H:24])[H:25])[H:20]"
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
