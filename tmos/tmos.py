"""Sanitize transition metal complexes and predict their electronic properties.

This module is the primary entry point for the ``tmos`` package.  Given an
RDKit molecule representing a transition metal complex (TMC), it

1. Cleaves every ligand from the metal center.
2. Redetermines bond orders for each ligand fragment.
3. Classifies each metal–ligand bond as either an *L-type* (neutral, DATIVE)
   or *X-type* (anionic, SINGLE) donor using the
   `Covalent Bond Classification (CBC) <doi.org/10.1016/0022-328X(95)00508-N>`_ method,
   including full support for haptic (η ≥ 2) ligands.
4. Scores all (ligand assignment, oxidation state) pairs against chemical
   plausibility constraints and returns the ranked results.

The implementation follows the architecture of
`xyz2mol_tm <https://github.com/jensengroup/xyz2mol_tm/>`_, replacing the
Hückel-based bond assignment with an explicit bond-order perception pipeline
and adding custom corrections for ferrocene and other haptic motifs.

Main functions
--------------
:func:`sanitize_complex`
    Top-level entry point.  Accepts a 3-D RDKit molecule and returns a ranked
    list of :class:`ComplexState` objects.
:func:`get_ligand_attributes`
    Analyses a single ligand fragment, enumerating all valid L/X-type
    assignments and detecting haptic groups.
:func:`is_transition_metal`
    Get our metal centers of interest

Output of ``sanitize_complex``
------------------------------
:func:`sanitize_complex` returns a list of :class:`ComplexState` objects sorted
by ``score`` ascending (lower is better).  Each :class:`ComplexState` bundles
four sub-objects:

``state.score`` : int
    Weighted penalty sum.  Scores below 1000 pass all chemical-plausibility
    checks (valid oxidation state, consistent complex charge).

``state.metal`` : :class:`MetalInfo`
    Predicted ``oxidation_state``, formal ``charge``, and d-electron
    ``electron_count`` for the metal center, ``symbol``.

``state.ligands`` : :class:`LigandSummary`
    Aggregated ligand-field data: total L/X connector counts, net ligand
    charge, and a per-ligand :class:`LigandInfo` list.  Each
    :class:`LigandInfo` entry carries the ligand ``smiles``, ``total_charge``,
    L/X connector indices, and, for haptic ligands, ``haptic_groups``,
    ``effective_l_count``, and ``effective_x_count``.

``state.complex`` : :class:`ComplexInfo`
    The fully assembled RDKit molecule (``rdmol``), its canonical ``smiles``,
    molecular ``formula``, net ``charge``, and predicted coordination
    ``geometry_type``.

``state.score_components`` : :class:`ScoreComponents`
    Per-penalty breakdown: oxidation-state membership, charge consistency,
    electron-count deviation, residual valence, and the
    negative-charge-with-X-type contradiction flag.

Example
-------
Typical usage after loading a structure with 3-D coordinates::

    from tmos import sanitize_complex
    results = sanitize_complex(mol)
    best = results[0]
    print(best.metal.oxidation_state)   # e.g. 2
    print(best.ligands.summary)
    print(best.complex.smiles)
"""

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, TypeAlias
from collections import Counter
from collections.abc import Sequence

from loguru import logger
import numpy as np

from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Geometry import Point3D
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

from .utils import get_molecular_formula, configure_logger, suppress_stdout_stderr
from . import build_rdmol as brd
from .reference_values import (
    bond_type_dict,
    METALS,
)
from .geometry import get_geometry_from_mol, get_geometry_from_positions

# Frozenset of atomic numbers treated as metal centers — derived from the METALS registry.
_METALS_ATOMIC_NUMS: frozenset[int] = frozenset(
    m.atomic_number for m in METALS.values()
)

ChargedAtoms: TypeAlias = dict[int, dict[str, object]]


@dataclass
class LigandInfo:
    """Properties of a single sanitized ligand candidate.

    Produced by :func:`get_ligand_attributes`.

    Attributes
    ----------
    index : int
        Position of this candidate in the enumeration produced by
        :func:`get_ligand_attributes`.
    rdmol : object
        RDKit molecule with bond orders assigned.
    smiles : str
        Canonical non-isomeric SMILES.
    chemical_formula : str
        Molecular formula string.
    candidate_id : str
        Content-addressed hash identifier built from SMILES, charge, connector
        lists, haptic groups, and charged-atom information. See :func:`_build_candidate_id`.
    total_charge : int
        Net formal charge of the ligand as isolated from the metal.
    hanging_bonds : int
        Number of unsatisfied valences remaining after bond assignment; used as
        a tie-breaking penalty in scoring.
    charged_atoms : ChargedAtoms
        Mapping from atom index to charge-related properties for every atom
        carrying a non-zero formal charge.
    l_type_connectors : list[int]
        Original-molecule atom indices that bind the metal as neutral (L-type)
        donors.
    x_type_connectors : list[int]
        Original-molecule atom indices that bind the metal as anionic (X-type)
        donors.
    haptic_groups : list[list[int]] or None
        Groups of two or more coordinating atoms that form a haptic (η ≥ 2)
        interaction with the metal. Each inner list contains the
        original-molecule atom indices of one haptic group. Singletons are not
        included. ``None`` before haptic detection has been run.
    effective_l_count : int or None
        Effective L-type donor count for CBC electron counting, accounting for
        haptic groups.  For each haptic group of hapticity η the contribution
        is ``η // 2`` L (plus ``η % 2`` X); non-haptic L atoms each contribute
        1.  ``None`` before haptic detection has been run.
    effective_x_count : int or None
        Effective X-type donor count for CBC electron counting, accounting for
        haptic groups.  Analogous to ``effective_l_count``.  ``None`` before
        haptic detection has been run.
    """

    index: int | None = None
    rdmol: object | None = None
    smiles: str | None = None
    chemical_formula: str | None = None
    candidate_id: str | None = None
    total_charge: int | None = None
    hanging_bonds: int | None = None
    charged_atoms: ChargedAtoms | None = None
    l_type_connectors: list[int] | None = None
    x_type_connectors: list[int] | None = None
    haptic_groups: list[list[int]] | None = None
    effective_l_count: int | None = None
    effective_x_count: int | None = None

    @property
    def summary(self) -> str:
        smiles_str = self.smiles if self.smiles is not None else "?"
        charge = self.total_charge if self.total_charge is not None else 0
        charge_str = f"+{charge}" if charge > 0 else str(charge)
        n_l = (
            self.effective_l_count
            if self.effective_l_count is not None
            else (
                len(self.l_type_connectors) if self.l_type_connectors is not None else 0
            )
        )
        n_x = (
            self.effective_x_count
            if self.effective_x_count is not None
            else (
                len(self.x_type_connectors) if self.x_type_connectors is not None else 0
            )
        )
        hb = self.hanging_bonds if self.hanging_bonds is not None else 0
        l_list = (
            str(self.l_type_connectors) if self.l_type_connectors is not None else "[]"
        )
        x_list = (
            str(self.x_type_connectors) if self.x_type_connectors is not None else "[]"
        )
        n_charged = len(self.charged_atoms) if self.charged_atoms is not None else 0
        haptic_parts = [
            f"η{len(g)} {g}" for g in (self.haptic_groups or []) if len(g) >= 2
        ]
        lines = [
            f"{smiles_str}\ncharge={charge_str}, {n_l}L/{n_x}X connectors, {hb} hanging bond(s)",
            f"  L-type: {l_list}",
            f"  X-type: {x_list}",
            f"  Charged atoms: {n_charged} atom(s)",
        ]
        if haptic_parts:
            lines.append(f"  Haptic: {', '.join(haptic_parts)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize ligand info without embedding RDKit objects."""
        charged_atoms = []
        for atom_idx, props in sorted((self.charged_atoms or {}).items()):
            charged_atoms.append(
                {
                    "atom_index": int(atom_idx),
                    "properties": _json_compatible(props),
                }
            )

        return {
            "index": self.index,
            "smiles": self.smiles,
            "chemical_formula": self.chemical_formula,
            "candidate_id": self.candidate_id,
            "total_charge": self.total_charge,
            "hanging_bonds": self.hanging_bonds,
            "charged_atoms": charged_atoms,
            "l_type_connectors": list(self.l_type_connectors or []),
            "x_type_connectors": list(self.x_type_connectors or []),
            "haptic_groups": [list(g) for g in (self.haptic_groups or [])],
            "effective_l_count": self.effective_l_count,
            "effective_x_count": self.effective_x_count,
        }


@dataclass
class ScoreComponents:
    """Decomposed scoring inputs and penalties for a single :class:`ComplexState`.

    Produced by :func:`_score_and_flatten_states` and stored under
    ``ComplexState.score_components``.

    Attributes
    ----------
    target_complex_charge : int
        Desired net charge of the complex, usually passed from :func:`sanitize_complex`.
    target_electron_count : int
        Desired electron count at the metal center, usually passed from
        :func:`sanitize_complex`.
    oxidation_membership_penalty : int
        1 if the predicted oxidation state is outside the expected set for the
        metal (from ``expected_oxidation_states``), 0 otherwise. Weighted
        ×1000 in the total score.
    charge_consistency_penalty : int
        ``|predicted_complex_charge - target_complex_charge|``. Weighted ×100
        in the total score.
    electron_count_penalty : int
        ``|metal_electron_count - target_electron_count|``. Weighted ×10 in
        the total score.
    residual_valence_penalty : int
        Sum of ``hanging_bonds`` across all ligand candidates in this
        assignment. Weighted ×1 in the total score.
    negative_charge_with_xtype_penalty : int
        1 if ``metal_charge < 0`` and the number of *σ-type* X donors
        (non-haptic X-type connectors) is > 0, 0 otherwise.
        Genuine covalent σ-X bonds (e.g. Cl, H, alkyl) withdraw one electron
        from the ligand toward the metal, making a negative metal formal charge
        physically contradictory.  X contributions from odd-η haptic groups
        (η%2 == 1 bookkeeping remainders, e.g. η3-allyl, η5-Cp) are excluded
        because they arise from CBC π-electron pairing and do not carry
        directional electron-withdrawal character — anionic allyl complexes
        such as [Fe(CO)₃(η3-allyl)]⁻ are valid at Fe(0).
        Weighted ×1000 in the total score.
    oxidation_state_preference_penalty : int
        Per-oxidation-state preference weight drawn from
        ``MetalDefinition.oxidation_state_penalties`` for the predicted OS.
        A weight of 0 means no penalty; higher values deprioritise uncommon
        but chemically valid OS values (e.g. Cu(0), Fe(0)) relative to
        more common ones.  Weighted ×10 in the total score, matching the
        electron-count penalty multiplier so it acts as a tiebreaker.
    geometry_oxidation_preference_penalty : int
        Optional per-geometry oxidation-state preference weight drawn from
        ``MetalDefinition.geometry_properties[geometry]["oxidation_state_penalties"]``.
        This is applied only when a geometry-specific mapping exists for the
        predicted complex geometry. Weighted ×20 in the total score to give
        geometry-derived oxidation hints stronger ranking influence.
    """

    target_complex_charge: int
    target_electron_count: int
    oxidation_membership_penalty: int
    charge_consistency_penalty: int
    electron_count_penalty: int
    residual_valence_penalty: int
    negative_charge_with_xtype_penalty: int
    oxidation_state_preference_penalty: int = 0
    geometry_oxidation_preference_penalty: int = 0

    @property
    def summary(self) -> str:
        return "\n".join(
            [
                f"Score components (target charge={self.target_complex_charge}, target electrons={self.target_electron_count}):",
                f"  oxidation membership: {self.oxidation_membership_penalty} × 1000 = {1000 * self.oxidation_membership_penalty}",
                f"  neg. charge + X-type: {self.negative_charge_with_xtype_penalty} × 1000 = {1000 * self.negative_charge_with_xtype_penalty}",
                f"  charge consistency:   {self.charge_consistency_penalty} × 100 = {100 * self.charge_consistency_penalty}",
                f"  electron count:       {self.electron_count_penalty} × 10 = {10 * self.electron_count_penalty}",
                f"  OS preference:        {self.oxidation_state_preference_penalty} × 10 = {10 * self.oxidation_state_preference_penalty}",
                f"  geom OS preference:   {self.geometry_oxidation_preference_penalty} × 20 = {20 * self.geometry_oxidation_preference_penalty}",
                f"  residual valence:     {self.residual_valence_penalty} × 1 = {self.residual_valence_penalty}",
            ]
        )

    def to_dict(self) -> dict[str, int]:
        """Serialize all score components as plain integers."""
        return {
            "target_complex_charge": int(self.target_complex_charge),
            "target_electron_count": int(self.target_electron_count),
            "oxidation_membership_penalty": int(self.oxidation_membership_penalty),
            "charge_consistency_penalty": int(self.charge_consistency_penalty),
            "electron_count_penalty": int(self.electron_count_penalty),
            "residual_valence_penalty": int(self.residual_valence_penalty),
            "negative_charge_with_xtype_penalty": int(
                self.negative_charge_with_xtype_penalty
            ),
            "oxidation_state_preference_penalty": int(
                self.oxidation_state_preference_penalty
            ),
            "geometry_oxidation_preference_penalty": int(
                self.geometry_oxidation_preference_penalty
            ),
        }


def _normalize_geometry_key(geometry_name: str) -> str:
    """Canonicalize geometry labels for dictionary lookups.

    Examples
    --------
    "Square Planar" -> "square_planar"
    "square-planar" -> "square_planar"
    "Capped Trigonal Prismatic (distorted)" -> "capped_trigonal_prismatic"
    """
    key = geometry_name.strip().lower()
    # Drop parenthetical descriptors so canonical names can be matched.
    key = re.sub(r"\s*\([^)]*\)", "", key)
    key = key.replace("-", " ").replace("/", " ")
    key = re.sub(r"\s+", "_", key)
    return key


def _get_geometry_os_preference_weight(
    metal_symbol: str,
    geometry_type: str | None,
    oxidation_state: int,
) -> int:
    """Return optional geometry-specific OS preference weight for one state."""
    if geometry_type is None:
        return 0

    geometry_props = METALS[metal_symbol].geometry_properties
    if not geometry_props:
        return 0

    target_key = _normalize_geometry_key(geometry_type)
    matched_props: dict[str, Any] | None = None
    for key, props in geometry_props.items():
        if _normalize_geometry_key(key) == target_key:
            matched_props = props
            break
    if matched_props is None:
        return 0

    os_weights = matched_props.get("oxidation_state_penalties", {})
    if not isinstance(os_weights, dict):
        return 0
    return int(os_weights.get(int(oxidation_state), 0))


def _json_compatible(value: Any) -> Any:
    """Convert values into JSON-compatible Python primitives."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    return value


def _serialize_rdmol_graph(
    mol: Chem.rdchem.Mol | None,
    coordinate_units: str = "angstrom",
) -> dict[str, Any] | None:
    """Serialize an RDKit molecule to atom/bond/position graph data."""
    if mol is None:
        return None

    atoms: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        atoms.append(
            {
                "index": int(atom.GetIdx()),
                "symbol": atom.GetSymbol(),
                "atomic_number": int(atom.GetAtomicNum()),
                "formal_charge": int(atom.GetFormalCharge()),
                "is_aromatic": bool(atom.GetIsAromatic()),
            }
        )

    bonds: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        bonds.append(
            {
                "begin": int(bond.GetBeginAtomIdx()),
                "end": int(bond.GetEndAtomIdx()),
                "order": float(bond.GetBondTypeAsDouble()),
                "bond_type": str(bond.GetBondType()),
                "is_aromatic": bool(bond.GetIsAromatic()),
            }
        )

    positions: dict[str, Any] | None = None
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        coordinates: list[list[float]] = []
        for idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(idx)
            coordinates.append([float(pos.x), float(pos.y), float(pos.z)])
        positions = {"units": coordinate_units, "coordinates": coordinates}

    return {"atoms": atoms, "bonds": bonds, "positions": positions}


@dataclass
class MetalInfo:
    """Properties of the transition-metal center.

    Produced by :func:`_score_and_flatten_states` via :func:`get_tm_attributes`
    and stored under ``ComplexState.metal``.

    Attributes
    ----------
    symbol : str
        Atomic symbol of the metal (e.g. ``"Fe"``).
    oxidation_state : int
        Predicted formal oxidation state.
    charge : int
        Formal charge applied to the metal atom in the assembled complex.
        Related to ``oxidation_state`` by the ligand field contribution.
    electron_count : int
        Total d-electron count at the metal center under the predicted
        oxidation state and ligand combination.
    """

    symbol: str
    oxidation_state: int
    charge: int
    electron_count: int

    @property
    def summary(self) -> str:
        charge_str = f"+{self.charge}" if self.charge > 0 else str(self.charge)
        ox_str = (
            f"+{self.oxidation_state}"
            if self.oxidation_state > 0
            else str(self.oxidation_state)
        )
        return f"{self.symbol}({ox_str}), charge={charge_str}, {self.electron_count} electron(s)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize metal-center properties."""
        return {
            "symbol": self.symbol,
            "oxidation_state": int(self.oxidation_state),
            "charge": int(self.charge),
            "electron_count": int(self.electron_count),
        }


@dataclass
class LigandSummary:
    """Aggregate ligand-field information for a single :class:`ComplexState`.

    Produced by :func:`_score_and_flatten_states` from a ligand combination
    enumerated by :func:`_enumerate_ligand_combinations` and stored under
    ``ComplexState.ligands``.

    Attributes
    ----------
    ligand_info : list[LigandInfo]
        One :class:`LigandInfo` per ligand in the complex, in the order they
        appear in the input molecule.
    candidate_ids : list[str]
        Ordered ``candidate_id`` values matching ``ligand_info``; used for
        deduplication keying.
    number_Ltype_connectors : int
        Effective L-type donor count across all ligands (CBC-aware: uses
        ``effective_l_count`` per ligand, so haptic η_n groups contribute η//2).
    number_Xtype_connectors : int
        Effective X-type donor count across all ligands (CBC-aware: uses
        ``effective_x_count`` per ligand, so odd-η haptic groups contribute η%2).
    total_charge : int
        Sum of ``total_charge`` from every :class:`LigandInfo` entry.
    haptic_group_counts : dict[int, int]
        Mapping of hapticity η to the number of haptic groups with that η,
        aggregated across all ligands.  For example, ``{4: 1, 6: 1}`` means
        one η4 group and one η6 group.  Empty dict when no haptic groups are
        present.
    """

    ligand_info: list[LigandInfo]
    candidate_ids: list[str]
    number_Ltype_connectors: int
    number_Xtype_connectors: int
    total_charge: int
    haptic_group_counts: dict[int, int] = None

    def __post_init__(self):
        if self.haptic_group_counts is None:
            self.haptic_group_counts = dict(
                Counter(
                    len(g)
                    for li in self.ligand_info
                    for g in (li.haptic_groups or [])
                    if len(g) >= 2
                )
            )

    @property
    def summary(self) -> str:
        charge_str = (
            f"+{self.total_charge}" if self.total_charge > 0 else str(self.total_charge)
        )
        haptic_str = (
            ", haptic: "
            + ", ".join(
                f"{v}×η{k}" for k, v in sorted(self.haptic_group_counts.items())
            )
            if self.haptic_group_counts
            else ""
        )
        return f"{len(self.ligand_info)} ligand(s), {self.number_Ltype_connectors}L/{self.number_Xtype_connectors}X donors, total charge={charge_str}{haptic_str}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize ligand summary data and per-ligand descriptors."""
        return {
            "candidate_ids": list(self.candidate_ids),
            "number_Ltype_connectors": int(self.number_Ltype_connectors),
            "number_Xtype_connectors": int(self.number_Xtype_connectors),
            "total_charge": int(self.total_charge),
            "haptic_group_counts": {
                str(k): int(v)
                for k, v in sorted((self.haptic_group_counts or {}).items())
            },
            "ligand_info": [li.to_dict() for li in self.ligand_info],
        }


@dataclass
class ComplexInfo:
    """Assembled-complex properties for a single :class:`ComplexState`.

    Attributes
    ----------
    rdmol : object
        Fully assembled RDKit molecule returned by :func:`reform_metal_complex`.
    smiles : str
        Canonical non-isomeric SMILES of the assembled complex.
    formula : str
        Molecular formula of the assembled complex.
    charge : int
        Net formal charge of the assembled complex (sum of all atom formal
        charges).
    number_metal_connections : int
        Effective coordination number using CBC-aligned haptic counting: each
        haptic group of η_n atoms contributes η//2 centroids (one per adjacent
        atom pair in bond-graph order) plus 1 for an odd η (the X-type donor
        position).  Non-haptic σ-bonds contribute 1 each.
    geometry_type : str
        Geometry label predicted from CBC-corrected coordination vectors.
        For haptic ligands, each η_n group is replaced by η//2 pair centroids
        (+ 1 for odd η) before geometry classification, so that an η4-diene
        contributes 2 sites and an η6-arene contributes 3 sites, matching the
        CBC L-count and giving chemically meaningful labels (e.g.
        Fe(CO)₃(η4-diene) → "Square Pyramidal", CpMn(CO)₃ → "Octahedral").
    """

    rdmol: object | None = None
    smiles: str | None = None
    formula: str | None = None
    charge: int | None = None
    number_metal_connections: int | None = None
    geometry_type: str | None = None

    @property
    def summary(self) -> str:
        formula_str = self.formula if self.formula is not None else "?"
        charge = self.charge if self.charge is not None else 0
        charge_str = f"+{charge}" if charge > 0 else str(charge)
        geom = self.geometry_type if self.geometry_type is not None else "?"
        n_bonds = (
            self.number_metal_connections
            if self.number_metal_connections is not None
            else 0
        )
        return f"{formula_str}, charge={charge_str}, {geom} ({n_bonds} bond(s))"

    def to_dict(
        self,
        include_graph: bool = True,
        coordinate_units: str = "angstrom",
    ) -> dict[str, Any]:
        """Serialize assembled-complex metadata and optional graph payload."""
        payload: dict[str, Any] = {
            "smiles": self.smiles,
            "formula": self.formula,
            "charge": self.charge,
            "number_metal_connections": self.number_metal_connections,
            "geometry_type": self.geometry_type,
        }
        if include_graph:
            payload["graph"] = _serialize_rdmol_graph(
                self.rdmol,
                coordinate_units=coordinate_units,
            )
        return payload


@dataclass
class ComplexState:
    """A single scored candidate state for a transition metal complex.

    The top-level return type of :func:`sanitize_complex`. Each entry
    represents one (ligand assignment, oxidation state) pair. States are sorted
    by ``score`` ascending (lower is better). The ``complex`` field is absent
    until after filtering inside :func:`sanitize_complex`.

    Attributes
    ----------
    score : int
        Weighted sum of all penalties in :class:`ScoreComponents`::

            1000 * oxidation_membership_penalty
            + 1000 * negative_charge_with_xtype_penalty
            +  100 * charge_consistency_penalty
            +   10 * electron_count_penalty
            +   10 * oxidation_state_preference_penalty
            +   20 * geometry_oxidation_preference_penalty
            +    1 * residual_valence_penalty

    score_components : ScoreComponents
        Decomposed scoring inputs and per-component penalties. See
        :class:`ScoreComponents`.
    predicted_complex_charge : int
        ``metal.charge + ligands.total_charge`` for this state.
    metal : MetalInfo
        Metal-center properties (symbol, oxidation state, charge, electron
        count). See :class:`MetalInfo`.
    ligands : LigandSummary
        Aggregated ligand-field information (connector counts, total charge,
        per-ligand details). See :class:`LigandSummary`.
    complex : ComplexInfo
        Assembled-complex properties (rdmol, SMILES, formula, geometry).
        Populated after filtering in :func:`sanitize_complex`. See
        :class:`ComplexInfo`.
    """

    score: int | None = None
    score_components: ScoreComponents | None = None
    predicted_complex_charge: int | None = None
    metal: MetalInfo | None = None
    ligands: LigandSummary | None = None
    complex: ComplexInfo | None = None

    @property
    def summary(self) -> str:
        score_str = str(self.score) if self.score is not None else "?"
        lines = [f"Score: {score_str}"]
        if self.metal is not None:
            lines.append(f"  Metal: {self.metal.summary}")
        if self.ligands is not None:
            lines.append(f"  Ligands: {self.ligands.summary}")
        if self.complex is not None:
            lines.append(f"  Complex: {self.complex.summary}")
        if self.score_components is not None:
            sc = self.score_components
            lines.append(
                f"  Penalties: oxid={sc.oxidation_membership_penalty}×1000, "
                f"charge={sc.charge_consistency_penalty}×100, "
                f"elec={sc.electron_count_penalty}×10, "
                f"valence={sc.residual_valence_penalty}×1"
            )
        return "\n".join(lines)

    def to_dict(
        self,
        include_graph: bool = True,
        coordinate_units: str = "angstrom",
        schema_version: int = 1,
    ) -> dict[str, Any]:
        """Serialize a scored state with a multi-metal-compatible schema.

        Notes
        -----
        The current processing pipeline predicts a single metal center.
        The serialized payload includes a ``metals_summary`` object for
        forward-compatible multi-metal schemas.
        """
        metals = [self.metal.to_dict()] if self.metal is not None else []
        metals_summary = {
            "metal_info": metals,
            "total_charge": int(sum(m.get("charge", 0) for m in metals)),
            "total_electron_count": int(
                sum(m.get("electron_count", 0) for m in metals)
            ),
        }
        payload: dict[str, Any] = {
            "schema_version": int(schema_version),
            "score": self.score,
            "predicted_complex_charge": self.predicted_complex_charge,
            "metals_summary": metals_summary,
            "metals": metals,
            "ligands": self.ligands.to_dict() if self.ligands is not None else None,
            "complex": (
                self.complex.to_dict(
                    include_graph=include_graph,
                    coordinate_units=coordinate_units,
                )
                if self.complex is not None
                else None
            ),
            "score_components": (
                self.score_components.to_dict()
                if self.score_components is not None
                else None
            ),
        }
        return _json_compatible(payload)

    def to_json(
        self,
        indent: int = 2,
        include_graph: bool = True,
        coordinate_units: str = "angstrom",
        schema_version: int = 1,
    ) -> str:
        """Serialize this state to a JSON string."""
        return json.dumps(
            self.to_dict(
                include_graph=include_graph,
                coordinate_units=coordinate_units,
                schema_version=schema_version,
            ),
            indent=indent,
            sort_keys=True,
        )


# Initialize logger with INFO level by default
configure_logger("INFO")

RDLogger.DisableLog("rdApp.*")
pt = GetPeriodicTable()


def sanitize_molecule(
    mol: Chem.rdchem.Mol,
    sanitize_aromaticity: bool = False,
    sanitize_kekulize: bool = False,
) -> None:
    """Sanitize a transition-metal-complex molecule in place.

    The function optionally updates formal charges using connectivity-derived
    heuristics, then applies RDKit sanitization with configurable aromaticity
    and kekulization stages.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule to sanitize
    sanitize_aromaticity : bool, default=False
        If ``False``, disable ``SANITIZE_SETAROMATICITY``.
    sanitize_kekulize : bool, default=False
        If ``False``, disable ``SANITIZE_KEKULIZE``.

    Returns
    -------
    None

    Examples
    --------
    >>> mol = mol_from_smiles("C[N+](=O)[O-]", sanitize=False)
    >>> sanitize_molecule(mol, sanitize_kekulize=True)
    """

    sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL
    if not sanitize_aromaticity:
        sanitize_ops ^= Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    if not sanitize_kekulize:
        sanitize_ops ^= Chem.SanitizeFlags.SANITIZE_KEKULIZE

    with suppress_stdout_stderr():
        Chem.SanitizeMol(
            mol,
            sanitizeOps=sanitize_ops,
        )


def compare_fingerprint(mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> bool:
    """Compare Morgan fingerprints for two molecules.

    Parameters
    ----------
    mol1 : rdkit.Chem.rdchem.Mol
        RDKit molecule
    mol2 : rdkit.Chem.rdchem.Mol
        RDKit molecule

    Returns
    -------
    bool
        Whether canonical tautomer-aware fingerprints are identical
    """
    fp1 = rdMolDescriptors.GetMorganFingerprint(mol1, 2)
    fp2 = rdMolDescriptors.GetMorganFingerprint(mol2, 2)

    return fp1 == fp2


def mol_from_smiles(
    smiles: str,
    sanitize: bool = True,
    sanitize_kwargs: dict[str, bool] | None = None,
) -> Chem.rdchem.Mol:
    """Convert a SMILES string into an RDKit molecule.

    Parameters
    ----------
    smiles : str
        SMILES string
    sanitize : bool, default=True
        Perform sanitization with :func:`sanitize_molecule`.
    sanitize_kwargs : dict of str to bool or None, default=None
        Keywords for :func:`sanitize_molecule`.


    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Parsed RDKit molecule.

    Examples
    --------
    >>> mol = mol_from_smiles("CCO")
    >>> mol.GetNumAtoms()
    3
    """
    sanitize_kwargs = {} if sanitize_kwargs is None else sanitize_kwargs
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if sanitize:
        sanitize_molecule(mol, **sanitize_kwargs)
    return mol


def mol_to_smiles(mol: Chem.rdchem.Mol) -> str:
    """Generate canonical non-isomeric SMILES from an RDKit molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule

    Returns
    -------
    str
        Canonical non-isomeric SMILES string with explicit hydrogens.
    """
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)


def wipe_molecule(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """Wipe all bond order and aromatic information from a molecule so that only single
    bonds remain between uncharged atoms.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Resulting molecule

    """
    mol = copy.deepcopy(mol)
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetFormalCharge(int(0))
    mol.UpdatePropertyCache(strict=False)
    return mol


def check_ligand_exception(
    mol: Chem.rdchem.Mol,
    metal_coordinating_indices: list[int],
) -> tuple[Chem.rdchem.Mol | None, list[int]]:
    """Apply hard-coded ligand exception corrections.

    Exceptions cover known motifs that are difficult to sanitize from raw
    connectivity alone (for example CO-like and azide-like edge cases).

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Ligand molecule.
    metal_coordinating_indices : list[int]
        Ligand molecule index of atom that was connected to the metal

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol or None
        Corrected molecule, or ``None`` if no exception pattern matches.
    metal_connected_orig_indices : list[int]
        Original atom indices connected to the metal.

    """
    mol = Chem.RWMol(copy.deepcopy(mol))
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    # Assume there is a metal
    metal_connected_orig_indices = [
        atm.GetIntProp("__original_index")
        for atm in mol.GetAtoms()
        if atm.GetIdx() in metal_coordinating_indices
    ]

    dummy_atoms = [
        atm for atm in mol.GetAtoms() if atm.GetIntProp("__original_index") == -1
    ]
    if dummy_atoms:
        mol.UpdatePropertyCache(strict=False)
        for atm in sorted(dummy_atoms, key=lambda x: -x.GetIdx()):
            mol.RemoveAtom(atm.GetIdx())

    formula: str = get_molecular_formula(mol)
    smiles: str | None = {  # Exceptions
        "C1O1": "[C-]#[O+]",
        # "H1N3": "[H][N]=[N+]=[N-]", # shouldn't be neutral
        "C1N1S1": "[N]#[C][S-]",
    }.get(formula, None)

    if smiles is None:
        return None, metal_connected_orig_indices

    tmp_mol = Chem.MolFromSmiles(smiles, sanitize=True)
    tmp_mol = Chem.AddHs(tmp_mol, explicitOnly=True)
    mol = brd.update_atom_bond_props(mol, tmp_mol)

    return mol, metal_connected_orig_indices


def is_coordinate_ring(
    mol: Chem.rdchem.Mol, metal_coordinating_indices: list[int]
) -> bool:
    """Determine whether coordinating atoms form a 5- or 6-member ring.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule to analyze.
    metal_coordinating_indices : list of int
        A list of atom indices that are coordinating a metal.

    Returns
    -------
    bool
        True if the metal-coordinating indices form a ring in the molecule, False otherwise.

    """

    if len(metal_coordinating_indices) in [5, 6]:
        Chem.FastFindRings(mol)
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) != len(metal_coordinating_indices):
                continue
            if set(ring) == set(metal_coordinating_indices):
                return True
    return False


def sanitize_ligand(
    mol: Chem.rdchem.Mol,
    delete_list: Sequence[Chem.rdchem.Atom] | None = None,
    wipe: bool = True,
    method: str = "openbabel",
    charge: int = None,
    sanitize: bool = True,
) -> Chem.rdchem.Mol | None:
    """Delete atoms from a molecule and then redetermine bond orders.

    Note:
     - An empty list can be provided to just redetermine bond orders for a molecule

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule
    delete_list : list[rdkit.Chem.rdchem.Atom], optional, default=[]
        List of RDKit atom objects to delete.
    wipe : bool, optional, default=True
        Whether to wipe bond information from the molecule
    method : str, default="openbabel"
        Choose the tool used to determine bond borders.

        - rdkit: ``rdDetermineBonds.DetermineBondOrders``
        - openbabel: ``PerceiveBondOrders``

    charge : int, default=None
        If using RDKit for bond orders, optionally set the charge.
    sanitize : bool, default=True
        If True, the resulting molecule will be sanitized

    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        Sanitized ligand molecule, or ``None`` if sanitization fails.
    """

    delete_list = [] if delete_list is None else delete_list
    mol_after = copy.deepcopy(mol)
    mol_after.UpdatePropertyCache(strict=False)
    if any(atm.GetNumImplicitHs() > 0 for atm in mol_after.GetAtoms()):
        raise ValueError("Provided molecule should not have implicit hydrogen atoms.")
    mol_after = Chem.RWMol(mol_after)
    Chem.SetAromaticity(mol_after, Chem.AromaticityModel.AROMATICITY_MDL)

    if wipe:
        mol_after = wipe_molecule(mol_after)

    if delete_list:
        delete_list = list(delete_list)
        delete_list.sort(key=lambda x: -x.GetIdx())
    for atm in delete_list:
        mol_after.RemoveAtom(atm.GetIdx())
    mol_after.UpdatePropertyCache(strict=False)
    mol_after = brd.determine_bonds(mol_after, method=method, charge=charge)
    if mol_after is not None and sanitize:
        try:
            Chem.SanitizeMol(mol_after)
        except Exception:
            mol_after = None

    return mol_after


def _normalize_charged_atoms(charged_atoms: ChargedAtoms) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for atom_idx in sorted(charged_atoms):
        atom_info = charged_atoms[atom_idx]
        normalized.append(
            {
                "atom_index": int(atom_idx),
                "info": {key: atom_info[key] for key in sorted(atom_info)},
            }
        )
    return normalized


def _build_candidate_id(candidate: LigandInfo) -> str:
    assert candidate.l_type_connectors is not None
    assert candidate.x_type_connectors is not None
    assert candidate.smiles is not None
    assert candidate.total_charge is not None
    assert candidate.hanging_bonds is not None
    assert candidate.charged_atoms is not None
    l_connectors = candidate.l_type_connectors
    x_connectors = candidate.x_type_connectors
    payload = {
        "smiles": candidate.smiles,
        "total_charge": int(candidate.total_charge),
        "hanging_bonds": int(candidate.hanging_bonds),
        "l_type_connectors": sorted(l_connectors),
        "x_type_connectors": sorted(x_connectors),
        "charged_atoms": _normalize_charged_atoms(candidate.charged_atoms),
        "haptic_groups": sorted([sorted(g) for g in candidate.haptic_groups])
        if candidate.haptic_groups is not None
        else [],
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"ligcand-{digest}"


def get_ligand_attributes(
    ligand_mol: Chem.rdchem.Mol,
    metal_coordinating_indices: list[int],
    add_hydrogens: bool = False,
) -> list[LigandInfo]:
    """Analyze ligand valence/bonding to determine connector attributes.

    Parameters
    ----------
    ligand_mol : rdkit.Chem.rdchem.Mol
        Ligand molecule with dummy atoms replacing ligand-metal bonds,
        denoted by ``atom.GetIntProp("__original_index") == -1``.
    metal_coordinating_indices : list[int]
        List of atom indices in ligand_mol that were connected to the metal. It is expected that they
        have a Internal Property, __original_index as well.
    add_hydrogens : bool, default=False
        If True, add explicit hydrogens to the ligand.

    Returns
    -------
    list of :class:`LigandInfo`
        Returns all sanitized candidate ligands.

    Notes
    -----
    ``l_type_connectors`` are neutral donor sites and ``x_type_connectors`` are
    anionic donor sites relative to the metal center.

    """

    ligand_mol = Chem.RWMol(copy.deepcopy(ligand_mol))
    ligand_mol = Chem.DeleteSubstructs(ligand_mol, Chem.MolFromSmarts("[#0]"))
    ligand_mol.UpdatePropertyCache(strict=False)
    if add_hydrogens:
        ligand_mol = Chem.AddHs(ligand_mol, addCoords=True, explicitOnly=True)
    ligand_mol.UpdatePropertyCache(strict=False)

    tmp_mol, metal_connected_orig_indices = check_ligand_exception(
        ligand_mol, metal_coordinating_indices
    )
    ligand_candidates: list[LigandInfo] = []
    if tmp_mol is not None:
        logger.debug("Ligand exception found.")
        _, hanging_bonds_after, charged_atoms_after = brd.assess_atoms(tmp_mol)
        total_charge_after = Chem.GetFormalCharge(tmp_mol)
        # Exception ligands are single-atom or small known motifs; no haptic groups.
        if total_charge_after < 0:
            l_conn, x_conn = [], metal_connected_orig_indices
            eff_l, eff_x = 0, len(metal_connected_orig_indices)
        else:
            l_conn, x_conn = metal_connected_orig_indices, []
            eff_l, eff_x = len(metal_connected_orig_indices), 0
        ligand_candidates.append(
            LigandInfo(
                index=0,
                rdmol=tmp_mol,
                total_charge=int(total_charge_after),
                hanging_bonds=hanging_bonds_after,
                charged_atoms=charged_atoms_after,
                l_type_connectors=l_conn,
                x_type_connectors=x_conn,
                haptic_groups=[],
                effective_l_count=eff_l,
                effective_x_count=eff_x,
            )
        )
    else:
        # Get prospective ligands, each with difference L-type and X-type connections
        dummy_atoms = [
            atm
            for atm in ligand_mol.GetAtoms()
            if atm.GetIntProp("__original_index") == -1
        ]
        dummy_atom_indices = [a.GetIdx() for a in dummy_atoms]
        metal_connected_atm_indices = {
            a1.GetIdx(): a2.GetIntProp("__original_index")
            for bond in ligand_mol.GetBonds()
            for a1, a2 in [
                (bond.GetBeginAtom(), bond.GetEndAtom()),
                (bond.GetEndAtom(), bond.GetBeginAtom()),
            ]
            if a1.GetIdx() in dummy_atom_indices
        }
        logger.debug(f"There are {len(dummy_atoms)} dummy atoms")

        # Detect haptic groups: connected components of coordinating atoms bonded to each other.
        haptic_groups = _find_haptic_groups(ligand_mol, metal_coordinating_indices)
        # _find_haptic_groups works on ligand-fragment indices. Convert each group
        # to original-complex indices (stored as __original_index on each atom) so
        # that haptic_coord_indices and the effective-count loop use the same index
        # space as metal_connected_atm_indices and l_type_connectors.
        haptic_groups = [
            [
                ligand_mol.GetAtomWithIdx(idx).GetIntProp("__original_index")
                for idx in group
            ]
            for group in haptic_groups
        ]
        haptic_coord_indices: set[int] = set()
        for group in haptic_groups:
            if len(group) >= 2:
                haptic_coord_indices.update(group)
        haptic_dummy_atoms = [
            atm
            for atm in dummy_atoms
            if metal_connected_atm_indices.get(atm.GetIdx()) in haptic_coord_indices
        ]
        non_haptic_dummy_atoms = [
            atm for atm in dummy_atoms if atm not in haptic_dummy_atoms
        ]

        # For haptic groups, determine which dummy atoms to delete based on ring/degree
        # heuristics.
        # All haptic atoms are classified as L-type (DATIVE bonds when reformed).
        haptic_delete: list[Chem.rdchem.Atom] = []
        for group in haptic_groups:
            if len(group) < 2:
                continue
            eta = len(group)
            group_dummy = [
                atm
                for atm in haptic_dummy_atoms
                if metal_connected_atm_indices.get(atm.GetIdx()) in group
            ]
            # Delete all dummy atoms for every haptic group regardless of η.
            # All haptic atoms are L-type; effective_l_count/effective_x_count
            # carry the CBC contribution (η//2 L, η%2 X), so all C–M bonds
            # should be DATIVE when reformed.
            haptic_delete.extend(group_dummy)

        # Non-haptic dummy atoms are enumerated as before (combinations for L/X classification)
        dummy_atom_combinations = []
        for k in range(len(non_haptic_dummy_atoms), -1, -1):
            dummy_atom_combinations.extend([*combinations(non_haptic_dummy_atoms, k)])
        # Prepend the haptic dummy atoms to each combination so they are always deleted
        dummy_atom_combinations = [
            haptic_delete + list(combo) for combo in dummy_atom_combinations
        ]

        ligand_prospects = {}
        for j, delete_list in enumerate(dummy_atom_combinations):
            new_ligand = sanitize_ligand(ligand_mol, delete_list=delete_list)
            if new_ligand is not None:
                _, hanging_bonds_after, charged_atoms_after = brd.assess_atoms(
                    new_ligand
                )
                # sanitize_ligand calls Chem.SanitizeMol, so RDKit has already
                # assigned authoritative formal charges (including on aromatic
                # anions such as Cp⁻ where valence-balance gives 0 but the
                # formal charge is -1). Use GetFormalCharge as the canonical
                # source; assess_atoms is kept only for hanging_bonds / charged_atoms.
                total_charge_after = Chem.GetFormalCharge(new_ligand)
                ligand_prospects[j] = LigandInfo(
                    index=j,
                    rdmol=new_ligand,
                    total_charge=int(total_charge_after),
                    hanging_bonds=hanging_bonds_after,
                    charged_atoms=charged_atoms_after,
                )
                if len(charged_atoms_after) < 6:
                    logger.debug("___________________________________________________")
                    logger.debug(f"{j}:", total_charge_after, hanging_bonds_after)
                    for ind, tmp in charged_atoms_after.items():
                        logger.debug("    ", ind, tmp)
            else:
                logger.debug("Sanitize failed")

        # Filter prospective ligands
        if not ligand_prospects:
            raise ValueError("Ligand could not be sanitized.")

        for ligand_prospect in ligand_prospects.values():
            ligand_prospect.l_type_connectors = [
                metal_connected_atm_indices[x.GetIdx()]
                for x in dummy_atom_combinations[ligand_prospect.index]
            ]
            ligand_prospect.x_type_connectors = [
                metal_connected_atm_indices[x.GetIdx()]
                for x in list(
                    set(dummy_atoms)
                    - set(dummy_atom_combinations[ligand_prospect.index])
                )
            ]

            # Compute effective L/X counts for CBC electron counting.
            # Haptic groups contribute η//2 L and η%2 X regardless of their
            # position in l_type_connectors (all haptic atoms are L-type).
            l_set = set(ligand_prospect.l_type_connectors)
            x_set = set(ligand_prospect.x_type_connectors)
            eff_l = 0
            eff_x = 0
            assigned_haptic: set[int] = set()
            for group in haptic_groups:
                if len(group) < 2:
                    continue
                # Only count this group if at least one member was assigned as L
                if any(orig_idx in l_set for orig_idx in group):
                    eta = len(group)
                    eff_l += eta // 2
                    eff_x += eta % 2
                    assigned_haptic.update(group)
            # Non-haptic contributors
            eff_l += sum(1 for idx in l_set if idx not in assigned_haptic)
            eff_x += sum(1 for idx in x_set if idx not in assigned_haptic)
            ligand_prospect.haptic_groups = [
                group for group in haptic_groups if len(group) >= 2
            ]
            ligand_prospect.effective_l_count = eff_l
            ligand_prospect.effective_x_count = eff_x

            ligand_candidates.append(ligand_prospect)

    if not ligand_candidates:
        raise ValueError("Ligand could not be sanitized.")

    for ligand_candidate in ligand_candidates:
        ligand_candidate.smiles = mol_to_smiles(ligand_candidate.rdmol)
        ligand_candidate.chemical_formula = get_molecular_formula(
            ligand_candidate.rdmol
        )
        ligand_candidate.candidate_id = _build_candidate_id(ligand_candidate)

    # Reorder prospects by sorting with the desired priorities.
    ligand_candidates.sort(
        key=lambda x: (
            len(x.charged_atoms or {}),
            abs(x.total_charge or 0),
            x.hanging_bonds or 0,
        )
    )

    return ligand_candidates


def assert_same_ring(
    mol: Chem.rdchem.Mol,
    ind1: int,
    ind2: int,
    max_ring_size: int = 6,
) -> bool:
    """Check whether two atoms belong to the same ring.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule to assess
    ind1 : int
        Index of first atom of interest
    ind2 : int
        Index of second atom of interest
    max_ring_size : int, default=6
        Maximum ring size to consider

    Returns
    -------
    bool
        True if the two indices are in the same ring.

    """
    ring_info = mol.GetRingInfo()

    indices = []
    for ring in ring_info.AtomRings():
        if ind1 in ring and len(ring) <= max_ring_size:
            indices.extend(list(set(ring)))
    if not indices:
        return False
    else:
        return ind2 in indices


def _haptic_group_to_cbc_positions(
    mol: Chem.rdchem.Mol,
    group: list[int],
    conf: "Chem.Conformer",
) -> list[np.ndarray]:
    """Return η//2 centroid positions for a haptic group, CBC-aligned.

    Atoms are ordered by bond-graph traversal (DFS from a chain endpoint, or
    from the lowest-index atom for a ring), then paired consecutively.  Each
    adjacent pair contributes one centroid, matching the η//2 L-type count
    used in CBC electron counting.  If η is odd (η3, η5), the unpaired atom
    (the X-type donor site) contributes its own position.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The parent molecule containing bond connectivity.
    group : list of int
        Atom indices of the haptic group (in the parent molecule).
    conf : rdkit.Chem.Conformer
        Conformer supplying 3D coordinates.

    Returns
    -------
    list of numpy.ndarray
        Between 1 and η//2 + η%2 position vectors in Å.

    Examples
    --------
    η4 diene [a,b,c,d] bonded a-b-c-d → centroids [(a+b)/2, (c+d)/2]
    η3 allyl [a,b,c] bonded a-b-c    → centroids [(a+b)/2], plus c position
    η6 benzene [a..f] ring           → centroids of 3 adjacent pairs
    """
    if len(group) == 1:
        return [np.array(conf.GetAtomPosition(group[0]))]

    # Build local adjacency within the group.
    group_set = set(group)
    adj: dict[int, list[int]] = {idx: [] for idx in group}
    for idx in group:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() in group_set:
                adj[idx].append(nb.GetIdx())

    # Start DFS from a degree-1 node (chain endpoint) or lowest-index node (ring).
    start = next((idx for idx in sorted(group) if len(adj[idx]) == 1), group[0])

    # DFS to get bond-graph order.
    visited: list[int] = []
    seen: set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        for nb in adj[node]:
            if nb not in seen:
                stack.append(nb)

    # Pair consecutive atoms; leftover (odd η) contributes its own position.
    positions: list[np.ndarray] = []
    i = 0
    while i + 1 < len(visited):
        pos_a = np.array(conf.GetAtomPosition(visited[i]))
        pos_b = np.array(conf.GetAtomPosition(visited[i + 1]))
        positions.append((pos_a + pos_b) / 2.0)
        i += 2
    if i < len(visited):
        positions.append(np.array(conf.GetAtomPosition(visited[i])))

    return positions


def _find_haptic_groups(
    mol: Chem.rdchem.Mol,
    coordinating_indices: list[int],
) -> list[list[int]]:
    """Identify groups of mutually-bonded coordinating atoms (haptic interactions).

    Builds a subgraph from *coordinating_indices* using bonds already present
    in *mol*, then returns its connected components.  Components of size ≥ 2
    represent haptic (η ≥ 2) groups; singletons represent ordinary σ-donors.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Ligand fragment molecule (post-cleavage from the metal center).
    coordinating_indices : list of int
        Atom indices in *mol* that were bonded to the metal.

    Returns
    -------
    list of list of int
        One inner list per connected component.  Each element is an atom index
        from *coordinating_indices*.  Components are returned in ascending order
        of their smallest member.
    """
    coord_set = set(coordinating_indices)
    # Union-Find
    parent = {i: i for i in coordinating_indices}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for bond in mol.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i1 in coord_set and i2 in coord_set:
            _union(i1, i2)

    groups: dict[int, list[int]] = {}
    for idx in coordinating_indices:
        root = _find(idx)
        groups.setdefault(root, []).append(idx)

    return sorted([sorted(g) for g in groups.values()])


def detect_additional_bonds(
    mol: Chem.rdchem.Mol,
    index: int | None = None,
    distance_tolerance: float = 0.3,
) -> Chem.rdchem.Mol:
    """Use the coordinates to check if any other bonds could be defined.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule
    index : int or None, default=None
        Index of target atom to look for bonds. If None, all bonds are added.
    distance_tolerance : float, default=0.2
        Additional distance tolerance used by coordinate-based bond detection.
    Returns
    -------
    rdkit.Chem.rdchem.Mol
        RDKit molecule
    """
    mol = Chem.RWMol(copy.deepcopy(mol))
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    if mol.GetNumConformers() == 0:
        logger.warning("Provided molecule does not have any coordinates")
        return mol

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    conformer = mol.GetConformer()
    positions = np.array(conformer.GetPositions())
    new_mol = brd.xyz_to_rdkit(
        symbols,
        positions,
        ignore_scale=True,
        distance_tolerance=distance_tolerance,
    )  # atom ids will be equivalent
    for bond in new_mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetBondBetweenAtoms(idx1, idx2) is None and (
            index is None or index in [idx1, idx2]
        ):
            atom1, atom2 = mol.GetAtomWithIdx(idx1), mol.GetAtomWithIdx(idx2)
            if (atom1.GetSymbol() == "H" and len(atom1.GetBonds()) >= 1) or (
                atom2.GetSymbol() == "H" and len(atom2.GetBonds()) >= 1
            ):
                continue
            mol.AddBond(idx1, idx2, Chem.rdchem.BondType.SINGLE)
    return mol


def correct_ferrocene(mol: Chem.rdchem.Mol, index: int) -> tuple[Chem.rdchem.Mol, int]:
    """Normalize bond annotations for ferrocene-like motifs.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule of interest
    index : int
        Index of the metal center of a ferrocene group

    Returns
    -------
    new_mol : rdkit.Chem.rdchem.Mol
        Output molecule with corrected ferrocene motif.
    new_index : int
        The atomic index of the ferrocene metal center, if changed
    """

    metal = mol.GetAtoms()[index]
    symbol = metal.GetSymbol()
    c_atoms = []
    for b in metal.GetBonds():
        carbon = (
            b.GetBeginAtom()
            if b.GetBeginAtom().GetSymbol() != symbol
            else b.GetEndAtom()
        )
        c_atoms.append(carbon.GetIdx())
        for bc in carbon.GetBonds():
            tmp_atm = (
                bc.GetBeginAtom()
                if bc.GetBeginAtomIdx() != carbon.GetIdx()
                else bc.GetEndAtom()
            )
            if assert_same_ring(mol, carbon.GetIdx(), tmp_atm.GetIdx()):
                bc.SetBondType(Chem.BondType.AROMATIC)
            else:
                bc.SetBondType(Chem.BondType.SINGLE)
        b.SetBondType(Chem.BondType.DATIVE)
        carbon.SetNoImplicit(True)
        carbon.SetNumExplicitHs(0)
        if carbon.GetDegree() < 4:
            carbon.SetNumExplicitHs(1)
        carbon.UpdatePropertyCache(strict=False)
    mol = Chem.AddHs(mol, addCoords=True, explicitOnly=True, onlyOnAtoms=c_atoms)

    new_index = index
    for a in mol.GetAtoms():
        a.SetIntProp("__original_index", a.GetIdx())
        if a.GetAtomicNum() in _METALS_ATOMIC_NUMS:
            new_index = a.GetIdx()

    return mol, new_index


def compute_centroid_excluding(
    conformer: Chem.rdchem.Conformer,
    exclude_atoms: list[int],
) -> Point3D:
    """Compute the centroid of a molecule while excluding specified atom indices.

    Parameters
    ----------
    conformer : rdkit.Chem.rdchem.Conformer
        RDKit conformer with 3D coordinates
    exclude_atoms : list[int]
        List of atom indices to exclude from centroid calculation

    Returns
    -------
    Point3D
        Centroid of the remaining atoms
    """
    positions = conformer.GetPositions()
    for i in range(len(positions)):
        if i in exclude_atoms:
            positions[i] = [np.nan, np.nan, np.nan]

    centroid = np.nanmean(positions, axis=0)
    return Point3D(*centroid)


def find_missing_coords(mol: Chem.rdchem.Mol, value: float = 0) -> bool:
    """Determine if an RDKit molecule has a relevant geometry

    In PDB CCD if the coordinates are missing, denoted by question marks in the cif, then the coordinate will be (0,0,0)

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule to assess
    value : float, default=0
        Value used to compare to coordinates.
        If all three coordinates for a non-metal atom equal this value, then a
        coordinate is considered missing.  Metal centres are excluded because
        Architector (and other builders) routinely place the metal at the
        coordinate origin, which is a valid geometry, not a missing datum.

    Returns
    -------
    bool
        Whether missing coordinates were detected.
    """

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in _METALS_ATOMIC_NUMS:
            continue
        pos = positions[atom.GetIdx()]
        if np.all(pos == value):
            return True
    return False


def fix_missing_coords(
    mol: Chem.rdchem.Mol,
    tmc_idx: int,
    missing_coord_indices: list[int],
) -> None:
    """Add coordinates to RDKit molecule with missing coordinates

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit to be repaired
    tmc_idx : int
        Atom index of the complex metal
    missing_coord_indices : list[int]
        Atom indices for which to find coordinates

    """

    # Move bad atoms closer
    conformer = mol.GetConformer()
    center = compute_centroid_excluding(conformer, missing_coord_indices)
    for i, atm_idx in enumerate(missing_coord_indices):
        radius = 1
        tmp_coord = Point3D(
            *tuple(
                np.array([center.x, center.y, center.z])
                + np.random.rand(3) * 2 * radius
                - radius
            )
        )
        conformer.SetAtomPosition(atm_idx, tmp_coord)

    # Optimize
    ff = Chem.AllChem.UFFGetMoleculeForceField(mol)
    metal_atoms = list(
        set(
            [
                x
                for b in mol.GetAtoms()[tmc_idx].GetBonds()
                for x in [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
            ]
        )
    )
    overlap = list(set(metal_atoms) & set(missing_coord_indices))
    for atm_idx in metal_atoms:
        if atm_idx not in overlap:
            ff.AddFixedPoint(atm_idx)
    ff.Minimize(maxIts=200000)


def find_metal_index(mol: Chem.rdchem.Mol) -> int:
    """Return the unique transition-metal atom index.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Transition-metal complex.

    Raises
    ------
    ValueError
        No transition metal found

    Returns
    -------
    int
        Index of transition metal
    """
    tmc_idx = None
    for a in mol.GetAtoms():
        a.SetNoImplicit(True)
        if a.GetAtomicNum() in _METALS_ATOMIC_NUMS:
            if tmc_idx is not None:
                raise ValueError(
                    "More than one metal detected! Multi-metal structures are not yet supported."
                )
            tmc_idx = a.GetIdx()
    if tmc_idx is None:
        raise ValueError(
            f"No transition metal found, molecule contains {set(a.GetAtomicNum() for a in mol.GetAtoms())}"
        )
    return tmc_idx


def get_tm_attributes(
    tm_mol: Chem.rdchem.Mol,
    n_ltype: int,
    n_xtype: int,
    n_electrons: int = 18,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Compute oxidation-state, charge, and electron-count possibilities.

    Parameters
    ----------
    tm_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule containing only the transition metal atom.
    n_ltype : int
        Number of L-type connectors (neutral ligands).
    n_xtype : int
        Number of X-type connectors (anionic ligands).
    n_electrons : int, default=18
        Target electron count.

    Returns
    -------
    oxidation_states : list of int
        List of possible oxidation states for the metal.
    charges : numpy.ndarray
        Array of formal charges corresponding to each oxidation state.
    electron_counts : numpy.ndarray
        Array of electron counts corresponding to each oxidation state.

    """

    atom = tm_mol.GetAtomWithIdx(0)
    n_group: int = METALS[atom.GetSymbol()].group
    charge = n_group + n_xtype + 2 * n_ltype - n_electrons
    oxidation_state = n_xtype + charge

    # Shift values based on realistic oxidation states
    oxidation_states: list[int] = METALS[atom.GetSymbol()].expected_oxidation_states
    offsets = np.array(oxidation_states) - oxidation_state
    charges = charge + offsets
    electron_counts = n_electrons - offsets

    return oxidation_states, charges, electron_counts


def cleave_mol_from_index(
    mol: Chem.rdchem.Mol,
    index: int,
    add_atom: str | None = None,
) -> tuple[list[Chem.rdchem.Mol], list[int]]:
    """Given an atomic index of an RDKit molecule, cleave the attaching bonds and return the resulting molecules

    The original atom index that corresponds to the output, `coordinating_atoms`, can be accessed with the atom
    int property, "__original_index".

    If an atom has a negative charge greater than one after cleavage, a dummy atom is added for each charge.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule
    index : int
        Index of atom to cleave from neighbors
    add_atom : str or None, default=None
        If not None, add an atom of this type in place of the metal center

    Returns
    -------
    fragments : list of rdkit.Chem.rdchem.Mol
        List of RDKit molecules resulting from cleaved bonds
    coordinating_atoms : list of int
        List of atom indices that were connected to the central atom

    """

    params = Chem.MolStandardize.rdMolStandardize.MetalDisconnectorOptions()
    params.splitAromaticC = True
    params.splitGrignards = True
    params.adjustCharges = False
    params.splitHydrides = True  # This should ensure hydrides are split

    # Metals of interest SMARTS, including all transition metals and main group metals
    metal_smarts = ",".join(f"#{m.atomic_number}" for m in METALS.values())
    MetalsOfInterest = (
        f"[{metal_smarts}]" "~[#1,#5,#6,#14,#15,#33,#51,#16,#34,#52,#17,#35,#53,#85]"
    )

    # Find atoms directly bonded to the metal center
    coordinating_atoms: list[int] = [
        int(x) for x in np.nonzero(Chem.rdmolops.GetAdjacencyMatrix(mol)[index, :])[0]
    ]
    for a in mol.GetAtoms():
        a.SetIntProp("__original_index", a.GetIdx())

    mdis = rdMolStandardize.MetalDisconnector(params)
    mdis.SetMetalNon(Chem.MolFromSmarts(MetalsOfInterest))
    frags = mdis.Disconnect(mol)

    frag_mols = list(rdmolops.GetMolFrags(frags, asMols=True, sanitizeFrags=False))
    logger.debug(f"Along with the metal, there are {len(frag_mols)-1} ligands")

    ind_metal: int = [
        ii
        for ii, f in enumerate(frag_mols)
        if sum([a.GetAtomicNum() in _METALS_ATOMIC_NUMS for a in f.GetAtoms()])
    ][0]
    if add_atom is not None:
        pos_metal = frag_mols[ind_metal].GetConformer().GetAtomPosition(0)
        for i, frag in enumerate(frag_mols):
            if i == ind_metal:
                continue
            add_atom_indices = []
            for atom in frag.GetAtoms():
                if atom.GetIntProp("__original_index") in coordinating_atoms:
                    add_atom_indices.append(atom.GetIdx())

            frag = Chem.RWMol(frag)
            new_atom_indices = []
            for idx in add_atom_indices:
                new_atom_idx = frag.AddAtom(Chem.Atom(add_atom))
                frag.GetAtomWithIdx(new_atom_idx).SetIntProp("__original_index", -1)
                new_atom_indices.append(new_atom_idx)
                frag.AddBond(idx, new_atom_idx, Chem.BondType.SINGLE)

            frag = frag.GetMol()
            conf = frag.GetConformer()
            for idx in new_atom_indices:
                conf.SetAtomPosition(idx, pos_metal)
            frag_mols[i] = frag

    return frag_mols, coordinating_atoms


def prepare_complex(
    mol: Chem.rdchem.Mol,
    value_missing_coord: float = 0,
    add_hydrogens: bool = False,
    distance_tolerance: float = 0.4,
) -> Chem.rdchem.Mol:
    """Prepare complex removing anomalous substructs, adding additional metal connections,
    checking for missing coordinates, and possible addition of hydrogens.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule representing the transition metal complex.
    value_missing_coord : float, default=0
        Value used to detect missing coordinates (e.g., 0 for (0,0,0)).
    add_hydrogens : bool, default=False
        If True, add explicit hydrogens to the structure if needed.
    distance_tolerance : float, default=0.4
        Tolerance used to detect additional bonds around metal

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule representing the transition metal complex.
    """

    mol = Chem.DeleteSubstructs(copy.deepcopy(mol), Chem.MolFromSmarts("[#0]"))
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    mol.UpdatePropertyCache(strict=False)
    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=True, explicitOnly=True)
        mol.UpdatePropertyCache(strict=False)

    tmc_idx = find_metal_index(mol)
    mol = detect_additional_bonds(
        mol,
        distance_tolerance=distance_tolerance,
    )

    # Detect and correct special cases
    if mol.GetAtoms()[tmc_idx].GetDegree() == 10:  # Detect ferrocene
        logger.debug("Detect ferrocene!")
        mol, tmc_idx = correct_ferrocene(mol, tmc_idx)

    missing_coord_indices: bool = find_missing_coords(mol, value=value_missing_coord)
    if missing_coord_indices:
        raise ValueError("Molecule missing coordinates")
    #    mol = fix_missing_coords(mol, tmc_idx, missing_coord_indices)

    return mol


def _enumerate_ligand_combinations(
    ligand_candidate_lists: list[list[LigandInfo]],
) -> list[dict]:
    """Build exhaustive ligand assignment enumeration across per-ligand candidates.

    Parameters
    ----------
    ligand_candidate_lists : list of list of :class:`LigandInfo`
        Candidate ligand prospects grouped by ligand position.

    Returns
    -------
    list of dict
        One entry per unique ligand assignment with:
            - ``ligand_info``: list of :class:`LigandInfo`, one per ligand position.
            - ``candidate_ids``: ordered per-ligand candidate ids.
            - ``number_Ltype_connectors``: total L-type connectors across ligands.
            - ``number_Xtype_connectors``: total X-type connectors across ligands
              (includes haptic η%2 contributions; used for CBC electron counting).
            - ``total_ligand_charge``: total ligand charge.
    """

    if not ligand_candidate_lists:
        return []

    combinations_out: list[dict] = []
    seen_keys: set[tuple[str, ...]] = set()
    for candidate_tuple in product(*ligand_candidate_lists):
        ligand_info = list(candidate_tuple)
        candidate_ids = [x.candidate_id for x in ligand_info]
        dedup_key = tuple(sorted(candidate_ids))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        combinations_out.append(
            {
                "ligand_info": ligand_info,
                "candidate_ids": candidate_ids,
                # Use effective counts when available (haptic-aware CBC electron counting);
                # fall back to raw connector-list lengths for non-haptic ligands.
                "number_Ltype_connectors": sum(
                    x.effective_l_count
                    if x.effective_l_count is not None
                    else len(x.l_type_connectors)
                    for x in ligand_info
                ),
                "number_Xtype_connectors": sum(
                    x.effective_x_count
                    if x.effective_x_count is not None
                    else len(x.x_type_connectors)
                    for x in ligand_info
                ),
                "total_ligand_charge": sum(int(x.total_charge) for x in ligand_info),
            }
        )

    return combinations_out


def _score_and_flatten_states(
    tm_mol: Chem.rdchem.Mol,
    ligand_combinations: list[dict],
    geometry_type: str | None = None,
    target_complex_charge: int = 0,
    target_electron_count: int = 18,
) -> list[ComplexState]:
    """Score ligand assignments against metal-centered chemical plausibility.

    Parameters
    ----------
    tm_mol : rdkit.Chem.rdchem.Mol
        Molecule containing the transition metal center.
    ligand_combinations : list of dict
        Candidate ligand assignments from :func:`_enumerate_ligand_combinations`.
    geometry_type : str or None, default=None
        Predicted coordination geometry label for the complex, used only for
        optional geometry-dependent oxidation-state preference weighting.
    target_complex_charge : int, default=0
        Desired total complex charge used for :attr:`ScoreComponents.charge_consistency_penalty`.
    target_electron_count : int, default=18
        Target electron count used for :attr:`ScoreComponents.electron_count_penalty`.

    Returns
    -------
    list of :class:`ComplexState`
        One entry per (ligand assignment, oxidation state) pair, sorted by
        ``score`` ascending (lower is better). The ``complex`` field is absent
        until :func:`sanitize_complex` assembles the molecule.
    """

    metal_symbol = tm_mol.GetAtomWithIdx(0).GetSymbol()
    expected_oxs = set(METALS[metal_symbol].expected_oxidation_states)
    os_weights = METALS[metal_symbol].oxidation_state_penalties

    scored_states: list[ComplexState] = []
    for combo in ligand_combinations:
        n_ltype = int(combo["number_Ltype_connectors"])
        n_xtype = int(combo["number_Xtype_connectors"])
        total_ligand_charge = int(combo["total_ligand_charge"])
        ligand_info = combo["ligand_info"]
        candidate_ids = combo["candidate_ids"]
        residual_valence_penalty = sum(int(x.hanging_bonds) for x in ligand_info)

        # Sigma-X count: only genuine covalent σ-donors. Haptic atoms are always
        # placed in l_type_connectors (DATIVE bonds), so x_type_connectors contains
        # only non-haptic donors. The η%2 == 1 bookkeeping X from odd-η haptic groups
        # never appears in x_type_connectors, so this count is naturally sigma-only.
        n_sigma_xtype = sum(len(x.x_type_connectors or []) for x in ligand_info)
        tm_oxs, tm_chgs, tm_nels = get_tm_attributes(tm_mol, n_ltype, n_xtype)
        for tm_ox, tm_chg, tm_nel in zip(tm_oxs, tm_chgs, tm_nels):
            oxidation_penalty = 0 if int(tm_ox) in expected_oxs else 1
            os_preference_penalty = os_weights.get(int(tm_ox), 0)
            geom_os_preference_penalty = _get_geometry_os_preference_weight(
                metal_symbol,
                geometry_type,
                int(tm_ox),
            )
            predicted_complex_charge = int(tm_chg) + total_ligand_charge
            charge_penalty = abs(predicted_complex_charge - target_complex_charge)
            electron_penalty = abs(int(tm_nel) - target_electron_count)
            # Genuine covalent σ-X bonds (Cl, H, alkyl, …) withdraw one electron
            # from the ligand toward the metal; a negative metal formal charge
            # alongside such donors is physically contradictory.
            # Haptic-group X contributions (η%2 == 1, e.g. η3-allyl, η5-Cp) are a
            # CBC bookkeeping remainder from pairing π-electrons and do NOT carry
            # this directional constraint — anionic allyl complexes like
            # [Fe(CO)₃(η3-allyl)]⁻ are real and valid at Fe(0).
            # Therefore only σ-X donors (n_sigma_xtype) trigger this penalty.
            negative_xtype_penalty = 1 if int(tm_chg) < 0 and n_sigma_xtype > 0 else 0

            score = (
                1000 * oxidation_penalty
                + 1000 * negative_xtype_penalty
                + 100 * charge_penalty
                + 10 * electron_penalty
                + 10 * os_preference_penalty
                + 20 * geom_os_preference_penalty
                + residual_valence_penalty
            )
            scored_states.append(
                ComplexState(
                    score=int(score),
                    score_components=ScoreComponents(
                        target_complex_charge=int(target_complex_charge),
                        target_electron_count=int(target_electron_count),
                        oxidation_membership_penalty=int(oxidation_penalty),
                        charge_consistency_penalty=int(charge_penalty),
                        electron_count_penalty=int(electron_penalty),
                        residual_valence_penalty=int(residual_valence_penalty),
                        negative_charge_with_xtype_penalty=int(negative_xtype_penalty),
                        oxidation_state_preference_penalty=int(os_preference_penalty),
                        geometry_oxidation_preference_penalty=int(
                            geom_os_preference_penalty
                        ),
                    ),
                    predicted_complex_charge=int(predicted_complex_charge),
                    metal=MetalInfo(
                        symbol=metal_symbol,
                        oxidation_state=int(tm_ox),
                        charge=int(tm_chg),
                        electron_count=int(tm_nel),
                    ),
                    ligands=LigandSummary(
                        ligand_info=ligand_info,
                        candidate_ids=candidate_ids,
                        number_Ltype_connectors=n_ltype,
                        number_Xtype_connectors=n_xtype,
                        total_charge=total_ligand_charge,
                    ),
                )
            )

    scored_states.sort(key=lambda x: x.score)
    return scored_states


def sanitize_complex(
    mol: Chem.rdchem.Mol,
    value_missing_coord: float = 0,
    add_hydrogens: bool = False,
    add_atom: str = "I",
    sanitize: bool = True,
    geometry_method: str = "angles",
    target_charge: int = 0,
    target_electron_count: int = 18,
    n_results: int | None = 5,
    score_cutoff: int | None = 1000,
    n_per_combination: int | None = None,
) -> list[ComplexState]:
    """Sanitize ligands, determining X-type and L-type, returning scored candidate states.

    Note that if coordinates are present in a conformer, bonds are detected to find all
    metal interaction points.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule representing the transition metal complex.
    value_missing_coord : float, default=0
        Value used to detect missing coordinates (e.g., 0 for (0,0,0)).
    add_hydrogens : bool, default=False
        If True, add explicit hydrogens to the structure if needed.
    add_atom : str, default="I"
        Element symbol of the "dummy atom" used in :func:`cleave_mol_from_index`
    sanitize : bool, default=True
        If True, the final complex will be sanitized with :func:`sanitize_molecule`
    geometry_method : str, default="angles"
        Method passed as ``mode`` to :func:`~tmos.geometry.get_geometry_from_mol`
        to predict the coordination geometry (e.g. ``"angles"``, ``"posym"``,
        ``"pymatgen"``, ``"rylm"``).
    target_charge : int, default=0
        Desired net charge of the complex, passed to
        :func:`_score_and_flatten_states` as ``target_complex_charge``.
    target_electron_count : int, default=18
        Target electron count for the metal centre, passed to
        :func:`_score_and_flatten_states`.  Useful for electron-deficient
        complexes such as porphyrins (typically 14–16e) that do not obey the
        18-electron rule.
    n_results : int or None, default=5
        Maximum number of states to return. ``None`` returns all.
    score_cutoff : int or None, default=1000
        Discard states with score >= this value. ``None`` disables the cutoff.
        Score >= 1000 indicates an unexpected oxidation state.
    n_per_combination : int or None, default=None
        Maximum number of states to retain per unique ligand combination
        (identified by its sorted ``candidate_ids`` tuple). ``None`` keeps all.

    Raises
    ------
    ValueError
        If the molecule does not contain a transition metal.

    Returns
    -------
    list of :class:`ComplexState`
        Candidate states sorted by ``score`` ascending (lower is better). Each
        entry contains ``score``, ``score_components`` (:class:`ScoreComponents`),
        ``predicted_complex_charge``, ``metal`` (:class:`MetalInfo`),
        ``ligands`` (:class:`LigandSummary`), and ``complex``
        (:class:`ComplexInfo`).

    Examples
    --------
    >>> # Typically called after loading a full TMC structure with coordinates.
    >>> # results = sanitize_complex(mol)
    >>> # isinstance(results, list)
    >>> # True
    """
    mol = prepare_complex(
        copy.deepcopy(mol),
        value_missing_coord=value_missing_coord,
        add_hydrogens=add_hydrogens,
    )
    tmc_idx = find_metal_index(mol)
    # Split the ligands from the metal center, note that we are adding a single bond at to each atom that
    # was connected to the metal center.
    frag_mols, coordinating_atoms = cleave_mol_from_index(
        mol, tmc_idx, add_atom=add_atom
    )
    geometry_type, n_bonds, _ = get_geometry_from_mol(
        mol, tmc_idx, mode=geometry_method
    )
    flag_tm = False
    ligand_candidate_lists: list[list[LigandInfo]] = []
    for i, f in enumerate(frag_mols):
        m = Chem.Mol(f)
        m.UpdatePropertyCache(strict=False)
        atoms = m.GetAtoms()
        for atom in atoms:  # Check that metal is found
            if atom.GetAtomicNum() in _METALS_ATOMIC_NUMS:
                if len(atoms) > 1:
                    raise ValueError("Not all ligands were separated.")
                flag_tm = True
                tm_mol = Chem.RWMol(frag_mols[i])
                break
        else:  # If the fragment is not the metal center
            logger.debug(f"Ligand {i+1} of {len(frag_mols)-1}")
            metal_coordinating_indices = [
                atm.GetIdx()
                for atm in m.GetAtoms()
                if atm.GetIntProp("__original_index") in coordinating_atoms
            ]
            all_ligand_candidates = get_ligand_attributes(
                m,
                metal_coordinating_indices,
            )
            ligand_candidate_lists.append(all_ligand_candidates)

    if not flag_tm:
        raise ValueError("No transition metal found")

    ligand_combinations = _enumerate_ligand_combinations(ligand_candidate_lists)
    logger.debug(f"Enumerated {len(ligand_combinations)} ligand combinations")

    scored_states = _score_and_flatten_states(
        tm_mol,
        ligand_combinations,
        geometry_type=geometry_type,
        target_complex_charge=target_charge,
        target_electron_count=target_electron_count,
    )
    logger.debug(
        f"Scoring complete: {len(scored_states)} (ligand combination, oxidation state) pairs"
    )

    # Filter: score_cutoff
    if score_cutoff is not None:
        scored_states = [s for s in scored_states if s.score < score_cutoff]

    # Filter: n_per_combination
    if n_per_combination is not None:
        per_combo_counts: dict[tuple[str, ...], int] = {}
        filtered: list[ComplexState] = []
        for state in scored_states:
            key = tuple(sorted(state.ligands.candidate_ids))
            count = per_combo_counts.get(key, 0)
            if count < n_per_combination:
                filtered.append(state)
                per_combo_counts[key] = count + 1
        scored_states = filtered

    # Filter: n_results
    if n_results is not None:
        scored_states = scored_states[:n_results]

    # Build rdmols only for retained states
    # Haptic-corrected geometry requires 3D coordinates; pre-fetch if available.
    _has_conformer = mol.GetNumConformers() > 0
    if _has_conformer:
        conf = mol.GetConformer()
        metal_pos = np.array(conf.GetAtomPosition(tmc_idx))
        metal_atom_ref = mol.GetAtomWithIdx(tmc_idx)

    for state in scored_states:
        tmp_tm_mol = copy.deepcopy(tm_mol)
        tmc_mol = reform_metal_complex(
            tmp_tm_mol,
            state.ligands.ligand_info,
            coordinating_atoms,
            tm_charge=state.metal.charge,
            sanitize=sanitize,
        )

        if _has_conformer:
            # Haptic-corrected geometry: replace each haptic group with its centroid.
            # Non-haptic coordinating atoms contribute their actual position.
            all_haptic_groups: list[list[int]] = [
                g
                for lig in state.ligands.ligand_info
                for g in (lig.haptic_groups or [])
            ]
            haptic_flat: set[int] = {idx for g in all_haptic_groups for idx in g}
            eff_positions: list[np.ndarray] = []
            for neighbor in metal_atom_ref.GetNeighbors():
                if neighbor.GetIdx() not in haptic_flat:
                    eff_positions.append(
                        np.array(conf.GetAtomPosition(neighbor.GetIdx()))
                    )
            for group in all_haptic_groups:
                for pos in _haptic_group_to_cbc_positions(mol, group, conf):
                    eff_positions.append(pos)

            if eff_positions:
                haptic_geom, haptic_n = get_geometry_from_positions(
                    metal_pos, eff_positions, tol=0.5
                )
            else:
                haptic_geom, haptic_n = geometry_type, n_bonds
        else:
            haptic_geom, haptic_n = geometry_type, n_bonds

        state.complex = ComplexInfo(
            rdmol=tmc_mol,
            smiles=mol_to_smiles(tmc_mol),
            formula=get_molecular_formula(tmc_mol),
            charge=int(sum(a.GetFormalCharge() for a in tmc_mol.GetAtoms())),
            number_metal_connections=haptic_n,
            geometry_type=haptic_geom,
        )

    return scored_states


def reform_metal_complex(
    tm_mol: Chem.rdchem.Mol,
    lig_info: list[LigandInfo],
    coordinating_atoms: list[int],
    tm_charge: int = 0,
    sanitize: bool = True,
) -> Chem.rdchem.RWMol:
    """Reconnects ligands to a transition metal center to reform a metal complex.

    This function takes a transition metal molecule and a list of ligand molecules,
    then combines them into a single complex. It reconnects the ligands to the metal
    center at specified coordinating atom indices, adjusting bond orders as needed.

    Parameters
    ----------
    tm_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule of the transition metal center.
    lig_info : list of :class:`LigandInfo`
        Ligand dictionaries returned by :func:`get_ligand_attributes`.
    coordinating_atoms : list[int]
        List of atom indices (from the original complex) that should be reconnected to the metal center.
    tm_charge : int, default=0
        Formal charge of the transition metal center.
    sanitize : bool, default=True
        If True, will sanitize the final complex with :func:`sanitize_molecule`

    Returns
    -------
    rdkit.Chem.rdchem.RWMol
        The reformed metal complex as an RDKit RWMol object with ligands reconnected.

    Raises
    ------
    UserWarning:
        If the bond order between the metal and a coordinating atom is changed during reconnection.

    Notes
    -----
        - The function assumes that the transition metal atom is the first atom in `tm_mol`.
        - Atom indices in `coordinating_atoms` refer to the original ligand atoms before combination.
        - When ``sanitize=True`` (the default), the resulting molecule is sanitized via
          :func:`sanitize_molecule`.
    """

    tm_symbol = tm_mol.GetAtoms()[0].GetSymbol()
    ltype_atoms, xtype_atoms = [], []
    for lig_dict in lig_info:
        ltype_atoms.extend(lig_dict.l_type_connectors)
        xtype_atoms.extend(lig_dict.x_type_connectors)
        tmp_mol = Chem.RWMol(copy.deepcopy(lig_dict.rdmol))
        remove_atoms = []
        for atm in tmp_mol.GetAtoms():
            if atm.GetIntProp("__original_index") == -1:
                remove_atoms.append(atm.GetIdx())

        for ind in sorted(remove_atoms, reverse=True):
            tmp_mol.RemoveAtom(ind)
        tm_mol = Chem.CombineMols(tm_mol, tmp_mol)

    # Add bonds
    tmc_mol = Chem.RWMol(tm_mol)
    coordinating_atoms_idx = [
        a.GetIdx()
        for a in tmc_mol.GetAtoms()
        if a.GetIntProp("__original_index") in coordinating_atoms
    ]
    tm_idx = [a.GetIdx() for a in tmc_mol.GetAtoms() if a.GetSymbol() == tm_symbol][0]
    tmc_mol.GetAtoms()[tm_idx].SetFormalCharge(int(tm_charge))

    for i in coordinating_atoms_idx:
        bond = tmc_mol.GetBondBetweenAtoms(i, tm_idx)
        if bond is not None:
            raise ValueError(
                f"There should not be a bond between {bond.GetBeginAtom().GetSymbol()}: {bond.GetBeginAtomIdx()}"
                f" and {bond.GetEndAtom().GetSymbol()}: {bond.GetEndAtomIdx()}"
            )
        atm = tmc_mol.GetAtoms()[i]
        # L-type atoms (including all haptic/η-coordinated atoms, which are
        # deliberately classified as L-type) receive DATIVE bonds — the
        # standard bond type for η-coordinated ligands in RDKit.
        if atm.GetIntProp("__original_index") in ltype_atoms:
            bond_type = bond_type_dict[0]
        elif atm.GetIntProp("__original_index") in xtype_atoms:
            bond_type = bond_type_dict[1]
        else:
            raise ValueError(
                f"Original index of {atm.GetSymbol()}: {atm.GetIdx()} is "
                f"{atm.GetIntProp('__original_index')} and cannot be found in metal connecting "
                f"atom lists: l-type {ltype_atoms} or x-type {xtype_atoms}"
            )
        tmc_mol.AddBond(i, tm_idx, bond_type)

    if sanitize:
        sanitize_molecule(tmc_mol)

    return tmc_mol
