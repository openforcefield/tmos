"""Graph mismatch diagnostics for molecule-vs-reference comparisons.

This module is used by tests to provide detailed, chemistry-aware context when
heavy-atom graph comparisons fail.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from tmos import build_rdmol


class EdgeThresholdDiagnostic(TypedDict):
    """Per-edge diagnostic data for a missing/extra heavy-atom bond."""

    i: int
    j: int
    symbol_i: str
    symbol_j: str
    distance: float
    min_threshold: float
    max_threshold: float
    rsum: float
    in_window: bool
    rule1: bool
    rule2: bool
    degree_i: int
    degree_j: int
    max_i: int
    max_j: int


class GraphMismatchReport(TypedDict):
    """Structured mismatch report returned by :func:`analyze_graph_mismatch`."""

    alignment: str
    actual_edges: int
    reference_edges: int
    missing: list[EdgeThresholdDiagnostic]
    extra: list[EdgeThresholdDiagnostic]


__all__ = ["analyze_graph_mismatch", "format_graph_mismatch_report"]


def _ref_atom_to_actual_index(atom: Chem.Atom) -> int:
    """Map reference atom to actual atom index using atom-map number when present."""
    amap = atom.GetAtomMapNum()
    if amap > 0:
        return amap - 1
    return atom.GetIdx()


def _heavy_edge_set_from_ref(ref_mol: Mol) -> set[tuple[int, int]]:
    """Return heavy-atom reference edges mapped into actual-index space."""
    edges: set[tuple[int, int]] = set()
    for bond in ref_mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
            continue
        i = _ref_atom_to_actual_index(a)
        j = _ref_atom_to_actual_index(b)
        if i == j:
            continue
        edges.add((i, j) if i < j else (j, i))
    return edges


def _heavy_edge_set_from_actual(actual_mol: Mol) -> set[tuple[int, int]]:
    """Return heavy-atom edge set from actual molecule indices."""
    edges: set[tuple[int, int]] = set()
    for bond in actual_mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
            continue
        i = a.GetIdx()
        j = b.GetIdx()
        edges.add((i, j) if i < j else (j, i))
    return edges


def _pair_diag(actual_mol: Mol, i: int, j: int) -> EdgeThresholdDiagnostic:
    """Build per-edge diagnostics in the style used by graph-mismatch test output."""
    atom_i = actual_mol.GetAtomWithIdx(i)
    atom_j = actual_mol.GetAtomWithIdx(j)
    sym_i = atom_i.GetSymbol()
    sym_j = atom_j.GetSymbol()

    min_t, max_t = build_rdmol._connectivity_distance_window(  # type: ignore[attr-defined]
        sym_i,
        sym_j,
        max_distance_tolerance=0.2,
        min_distance_tolerance=0.45,
    )
    rsum = build_rdmol._get_covalent_radius(sym_i) + build_rdmol._get_covalent_radius(
        sym_j
    )  # type: ignore[attr-defined]

    distance = float("nan")
    if actual_mol.GetNumConformers() > 0:
        conf = actual_mol.GetConformer()
        pi = conf.GetAtomPosition(i)
        pj = conf.GetAtomPosition(j)
        distance = float(
            np.linalg.norm(np.array([pi.x - pj.x, pi.y - pj.y, pi.z - pj.z]))
        )
    in_window = bool(min_t < distance < max_t) if np.isfinite(distance) else False

    degree_i = atom_i.GetDegree()
    degree_j = atom_j.GetDegree()
    max_i = build_rdmol._max_valence_for_connectivity(atom_i.GetAtomicNum(), sym_i)  # type: ignore[attr-defined]
    max_j = build_rdmol._max_valence_for_connectivity(atom_j.GetAtomicNum(), sym_j)  # type: ignore[attr-defined]
    rule1 = degree_i >= max_i or degree_j >= max_j

    i_ring_n_sat = atom_i.GetAtomicNum() == 7 and degree_i == 2 and atom_i.IsInRing()
    j_ring_n_sat = atom_j.GetAtomicNum() == 7 and degree_j == 2 and atom_j.IsInRing()
    i_satisfied = (
        build_rdmol._is_valence_satisfied(degree_i, atom_i.GetAtomicNum(), sym_i)
        or i_ring_n_sat
    )  # type: ignore[attr-defined]
    j_satisfied = (
        build_rdmol._is_valence_satisfied(degree_j, atom_j.GetAtomicNum(), sym_j)
        or j_ring_n_sat
    )  # type: ignore[attr-defined]
    rule2 = bool(i_satisfied and j_satisfied)

    return {
        "i": i,
        "j": j,
        "symbol_i": sym_i,
        "symbol_j": sym_j,
        "distance": distance,
        "min_threshold": min_t,
        "max_threshold": max_t,
        "rsum": rsum,
        "in_window": in_window,
        "rule1": rule1,
        "rule2": rule2,
        "degree_i": degree_i,
        "degree_j": degree_j,
        "max_i": max_i,
        "max_j": max_j,
    }


def analyze_graph_mismatch(
    actual_mol: Mol, reference_smiles: str
) -> GraphMismatchReport:
    """Compare heavy-atom graph of ``actual_mol`` against ``reference_smiles``.

    Returns a structured report that captures missing/extra heavy-atom edges and
    per-edge distance-threshold diagnostics.
    """
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        ref_mol = Chem.MolFromSmiles(reference_smiles, sanitize=False)
    if ref_mol is None:
        raise ValueError(f"Invalid reference SMILES: {reference_smiles}")

    expected = _heavy_edge_set_from_ref(ref_mol)
    actual = _heavy_edge_set_from_actual(actual_mol)

    missing_pairs = sorted(expected - actual)
    extra_pairs = sorted(actual - expected)

    missing = [_pair_diag(actual_mol, i, j) for i, j in missing_pairs]
    extra = [_pair_diag(actual_mol, i, j) for i, j in extra_pairs]

    return {
        "alignment": "atom_maps",
        "actual_edges": len(actual),
        "reference_edges": len(expected),
        "missing": missing,
        "extra": extra,
    }


def _format_edge(e: EdgeThresholdDiagnostic) -> str:
    return (
        f"  {e['i']}-{e['j']} ({e['symbol_i']}-{e['symbol_j']}): "
        f"d={e['distance']:.3f} A, "
        f"window=[{e['min_threshold']:.3f}, {e['max_threshold']:.3f}] A, "
        f"rsum={e['rsum']:.3f} A, "
        f"in_window={'yes' if e['in_window'] else 'no'}, "
        f"rule1={e['rule1']}, rule2={e['rule2']}, "
        f"deg=({e['degree_i']},{e['degree_j']}), "
        f"max=({e['max_i']},{e['max_j']})"
    )


def format_graph_mismatch_report(report: GraphMismatchReport) -> str:
    """Render a human-readable multi-line graph mismatch report."""
    lines = [
        (
            "Graph mismatch diagnostics "
            f"(alignment={report['alignment']}, "
            f"actual_edges={report['actual_edges']}, "
            f"reference_edges={report['reference_edges']})"
        )
    ]

    missing = report["missing"]
    extra = report["extra"]

    lines.append("Missing heavy-atom edges:")
    if missing:
        lines.extend(_format_edge(e) for e in missing)
    else:
        lines.append("  (none)")

    lines.append("Extra heavy-atom edges:")
    if extra:
        lines.extend(_format_edge(e) for e in extra)
    else:
        lines.append("  (none)")

    return "\n".join(lines)
