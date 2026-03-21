import networkx as nx
import pytest
from rdkit.Chem.rdmolfiles import MolFromSmiles
import json
from pathlib import Path

import tmos.build_rdmol as build_rdmol
from tmos.diagnostic import analyze_graph_mismatch, format_graph_mismatch_report
from tmos.graph_mapping import mol_to_graph

with open(Path(__file__).parent / "error_check4.json") as f:
    TEST_CASES = json.load(f)

OUTPUT_REFERENCE = [
    {
        "smiles": "[H:11][C:4]([H:12])([H:13])[S:7]([C:5]([H:14])([H:15])[H:16])([C:6]([H:17])([H:18])[C:3]([H:9])([H:10])[O:2][H:8])[I:1]",
        "key": "103936110",
    },
    {
        "smiles": "[H:18][c:8]1[c:7]([c:10]([c:9]([c:12]2[c:11]1[C:13]([C:15]([C:14]2([H:22])[H:23])([H:24])[H:25])([H:20])[H:21])[H:19])[O:5][P:16](=[O:2])([C:6](=[O:1])[O-:3])[O-:4])[H:17]",
        "key": "103937032",
    },
    {
        "smiles": "[O:1]=[C:6]([O:5][C:12]1([H:24])[C:10]([H:20])([H:21])[C:8]([H:16])([H:17])[C:7]([H:14])([H:15])[C:9]([H:18])([H:19])[C:11]1([H:22])[H:23])[P:13](=[O:2])([O-:3])[O-:4]",
        "key": "103937033",
    },
    {
        "smiles": "[H:34][c:15]1[c:30]2[c:27]([c:25]([c:17]([c:23]1[I:8])[O-:2])[I:10])[O:12][c:28]3[c:31]([c:16]([c:24]([c:18]([c:26]3[I:11])[O-:3])[I:9])[H:35])[C:33]24[c:32]5[c:29]([c:21]([c:19]([c:20]([c:22]5[Cl:7])[Cl:5])[Cl:4])[Cl:6])[C:14](=[O:1])[O:13]4",
        "key": "104006144",
    },
    {
        "smiles": "[H:26][c:12]1[c:13]([c:15]([c:17]([c:16]([c:14]1[H:28])[H:30])[C:11]2=[C:10]([N:4]=[C:9]([C:8](=[S:3])[N:18]2[H:31])[C@@:24]3([N:19]([C:22]([C:23]([C:21]([S:6]3)([H:36])[H:37])([H:39])[C:20]([H:33])([H:34])[H:35])([H:38])[C:7](=[O:1])[O-:2])[H:32])[H:40])[O:5][H:25])[H:29])[H:27]",
        "key": "104032050",
    },
    {
        "smiles": "[H:33][c:15]1[c:17]([c:24]2[c:23]([c:22]([c:16]1[H:34])[O:6][C:29]([H:39])([H:40])[H:41])[O:5][C:9](=[C:12]2[C:10]3=[C:11]([C:8](=[O:2])[N:27]([C:7]3=[O:1])[H:38])[C:13]4=[C:14]([N:28]([c:26]5[c:25]4[c:18]([c:20]([c:21]([c:19]5[H:37])[I:4])[F:3])[H:36])[C:30]([H:42])([H:43])[H:44])[H:32])[H:31])[H:35]",
        "key": "104114331",
    },
    {"smiles": "[N+:5]([O-:1])([F:2])([F:3])[F:4]", "key": "134989941"},
    {
        "smiles": "[H:21][C:7]1([C:13]2([C:10]([C:16]3([C:11]([C:14]1([C:9]([C:15]([C:8]2([H:23])[H:24])([C:12]3([H:31])[H:32])[I:1])([H:25])[H:26])[H:34])([H:29])[H:30])/[C:5](=[N:2]/[O:3][H:17])/[C:6]([H:19])([H:20])[O:4][H:18])([H:27])[H:28])[H:33])[H:22]",
        "key": "135012195",
    },
    {
        "smiles": "[H:26][c:13]1[c:19]([c:10]([n:7][c:11]([c:20]1[N:22]([H:29])[H:30])/[N:5]=[N:6]/[c:17]2[c:14]([c:12]([c:15]([c:18]([c:16]2[I:4])[C:9](=[O:1])[O:8][H:23])[I:3])[H:25])[I:2])[H:24])[N:21]([H:27])[H:28]",
        "key": "135018521",
    },
    {
        "smiles": "[H:10][N:9]1[C:7](=[C:5]([C:6](=[C:8]1[I:4])[I:2])[I:1])[I:3]",
        "key": "135025755",
    },
    {
        "smiles": "[H:20][c:7]1[c:5]([c:9]([c:6]([c:8]([c:10]1[C:13]([H:25])([H:26])[P:11]([O:4][C:17]([H:35])([H:36])[C:15]([H:30])([H:31])[H:32])[O:1][O:3][C:16]([H:33])([H:34])[C:14]([H:27])([H:28])[H:29])[H:21])[H:19])[O:2][C:12]([H:22])([H:23])[H:24])[H:18]",
        "key": "135027681",
    },
    {
        "smiles": "[H:19][c:3]1[c:5]([c:8]([c:6]([c:4]([c:7]1[C:15]([C:9]([H:23])([H:24])[H:25])([C:10]([H:26])([H:27])[H:28])[C:11]([H:29])([H:30])[H:31])[H:20])[H:22])[P:17](=[O:1])([C:16]([C:12]([H:32])([H:33])[H:34])([C:13]([H:35])([H:36])[H:37])[C:14]([H:38])([H:39])[H:40])[O:2][H:18])[H:21]",
        "key": "135048465",
    },
    {
        "smiles": "[H:23][c:9]1[c:10]([c:14]([c:11]([c:13]2[c:12]1[N:4]=[C:7]3[N:15]2[N:3]=[S:6]=[C:8]3[N:16]4[C:20]([C:18]([O:5][C:19]([C:21]4([H:34])[H:35])([H:30])[H:31])([H:28])[H:29])([H:32])[H:33])[H:25])[S:22](=[O:1])(=[O:2])[N:17]([H:26])[H:27])[H:24]"
    },
    {
        "smiles": "[H:16][c:8]1[c:10]([c:13]([c:11]([c:9]([c:12]1[C:7](=[O:1])[O-:4])[H:17])[H:19])[S:15](=[O:2])(=[O:3])[N:14]([Cl:5])[Cl:6])[H:18]"
    },
    {
        "smiles": "[C:7]1(=[O:1])[N:10]([C:8](=[O:2])[N:12]([C:9](=[O:3])[N:11]1[Cl:5])[Cl:6])[Cl:4]"
    },
    {
        "smiles": "[c:14]12[c:15]([c:13]([c:11]([c:10]([c:12]1[I:5])[I:3])[I:4])[I:6])[C:9](=[O:2])[O:7][C:8]2=[O:1]"
    },
    {
        "smiles": "[H:13][C:6]\\1=[C:7](/[C:11](=[N:4]\\[Cl:2])/[C:9](=[C:8](/[C:10]1=[N:3]/[Cl:1])[H:15])[O:5][C:12]([H:16])([H:17])[H:18])[H:14]"
    },
    {
        "smiles": "[F:1][N:6]([F:2])[C:12]([N:7]([F:3])[F:4])([C:8]([H:14])([H:15])[H:16])[C:11]([C:10]([C:9]([O:5][H:13])([H:17])[H:18])([H:19])[H:20])([H:21])[H:22]"
    },
    {"smiles": "[F:1][N:8]([F:2])[C:11]([F:7])([N:9]([F:3])[F:4])[N:10]([F:5])[F:6]"},
    {"smiles": "[C:7](=[N:6][F:1])([N:8]([F:2])[F:3])[N:9]([F:4])[F:5]"},
    {"smiles": "[N:1]#[C:3][S:4][I:2]"},
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


assert len(TEST_CASES) == len(OUTPUT_REFERENCE)


REFERENCE_CASES = [
    pytest.param(case, reference, id=f"structure_{i + 1}")
    for i, (case, reference) in enumerate(zip(TEST_CASES, OUTPUT_REFERENCE))
    if (i + 1) not in {2, 5, 15, 17}
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
