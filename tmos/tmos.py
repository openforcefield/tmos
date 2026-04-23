"""Sanitize and generate oxidation state and other properties for transition metal complexes.

This module uses function architectures originally produced in [xyz2mol_tm](https://github.com/jensengroup/xyz2mol_tm/). However rather than using the Huckel method
an arrow pushing script is produced here with custom checks for ferrocene structures.
"""

import copy
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations, product
from typing import TypeAlias
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
    METALS_NUM,
    expected_oxidation_states,
    group_numbers,
)
from .geometry import get_geometry_from_mol

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
        lists, and charged-atom information. See :func:`_build_candidate_id`.
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

    @property
    def summary(self) -> str:
        smiles_str = self.smiles if self.smiles is not None else "?"
        charge = self.total_charge if self.total_charge is not None else 0
        charge_str = f"+{charge}" if charge > 0 else str(charge)
        n_l = len(self.l_type_connectors) if self.l_type_connectors is not None else 0
        n_x = len(self.x_type_connectors) if self.x_type_connectors is not None else 0
        hb = self.hanging_bonds if self.hanging_bonds is not None else 0
        l_list = (
            str(self.l_type_connectors) if self.l_type_connectors is not None else "[]"
        )
        x_list = (
            str(self.x_type_connectors) if self.x_type_connectors is not None else "[]"
        )
        n_charged = len(self.charged_atoms) if self.charged_atoms is not None else 0
        return "\n".join(
            [
                f"{smiles_str}\ncharge={charge_str}, {n_l}L/{n_x}X connectors, {hb} hanging bond(s)",
                f"  L-type: {l_list}",
                f"  X-type: {x_list}",
                f"  Charged atoms: {n_charged} atom(s)",
            ]
        )


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
    """

    target_complex_charge: int
    target_electron_count: int
    oxidation_membership_penalty: int
    charge_consistency_penalty: int
    electron_count_penalty: int
    residual_valence_penalty: int

    @property
    def summary(self) -> str:
        return "\n".join(
            [
                f"Score components (target charge={self.target_complex_charge}, target electrons={self.target_electron_count}):",
                f"  oxidation membership: {self.oxidation_membership_penalty} × 1000 = {1000 * self.oxidation_membership_penalty}",
                f"  charge consistency:   {self.charge_consistency_penalty} × 100 = {100 * self.charge_consistency_penalty}",
                f"  electron count:       {self.electron_count_penalty} × 10 = {10 * self.electron_count_penalty}",
                f"  residual valence:     {self.residual_valence_penalty} × 1 = {self.residual_valence_penalty}",
            ]
        )


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
        Total count of neutral (L-type) donor sites across all ligands.
    number_Xtype_connectors : int
        Total count of anionic (X-type) donor sites across all ligands.
    total_charge : int
        Sum of ``total_charge`` from every :class:`LigandInfo` entry.
    """

    ligand_info: list[LigandInfo]
    candidate_ids: list[str]
    number_Ltype_connectors: int
    number_Xtype_connectors: int
    total_charge: int

    @property
    def summary(self) -> str:
        charge_str = (
            f"+{self.total_charge}" if self.total_charge > 0 else str(self.total_charge)
        )
        return f"{len(self.ligand_info)} ligand(s), {self.number_Ltype_connectors}L/{self.number_Xtype_connectors}X donors, total charge={charge_str}"


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
        Number of bonds to the metal center as determined by
        :func:`~tmos.geometry.get_geometry_from_mol`.
    geometry_type : str
        Geometry label predicted by :func:`~tmos.geometry.get_geometry_from_mol`
        using the method selected by the ``geometry_method`` argument of
        :func:`sanitize_complex`.
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
            + 100 * charge_consistency_penalty
            +  10 * electron_count_penalty
            +   1 * residual_valence_penalty

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
        "H1N3": "[H][N]=[N+]=[N-]",
        "O1": "[O]([H])[H]",  # Instances of oxo tend to be unphysical
        "H1O2": "[O]([H])[H]",  # Instances of peroxide tend to be unphysical
        "C1N1S1": "[N]#[C][S-]",
    }.get(formula, None)

    if smiles is None:
        return None, metal_connected_orig_indices

    tmp_mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if smiles != "[O]([H])[H]":
        tmp_mol = Chem.AddHs(tmp_mol, explicitOnly=True)
        mol = brd.update_atom_bond_props(mol, tmp_mol)
    else:
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        if len(conf_ids) > 1:
            raise ValueError("Ligand molecule has multiple conformers")
        if tmp_mol.GetAtoms()[0].GetSymbol() != "O":
            raise ValueError("This should be an oxygen!")

        if metal_connected_orig_indices:
            tmp_mol.GetAtoms()[0].SetIntProp(
                "__original_index", metal_connected_orig_indices[0]
            )
        tmp_mol.AddConformer(
            Chem.rdchem.Conformer(tmp_mol.GetNumAtoms()), assignId=True
        )
        if metal_connected_orig_indices:
            brd.copy_atom_coords(
                tmp_mol, 0, mol, metal_coordinating_indices[0], confId2=conf_ids[0]
            )
        mol = Chem.AddHs(tmp_mol, explicitOnly=True, addCoords=True)
        for a in mol.GetAtoms():
            if a.HasProp("__original_index"):
                continue
            a.SetIntProp(
                "__original_index", -2
            )  # Not an atom of consequence and not in the orig mol

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
    method : str, default="hybrid"
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
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"ligcand-{digest}"


def get_ligand_attributes(
    ligand_mol: Chem.rdchem.Mol,
    metal_coordinating_indices: list[int],
    add_hydrogens: bool = False,
) -> LigandInfo | list[LigandInfo]:
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
    :class:`LigandInfo` or list of :class:`LigandInfo`
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
        total_charge_after, hanging_bonds_after, charged_atoms_after = brd.assess_atoms(
            tmp_mol
        )
        ligand_candidates.append(
            LigandInfo(
                index=0,
                rdmol=tmp_mol,
                total_charge=int(total_charge_after),
                hanging_bonds=hanging_bonds_after,
                charged_atoms=charged_atoms_after,
                l_type_connectors=metal_connected_orig_indices,
                x_type_connectors=[],
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

        # Assumes carbon rings
        dummy_atom_combinations = []
        if is_coordinate_ring(ligand_mol, metal_coordinating_indices):
            if len(metal_coordinating_indices) == 6:
                dummy_atom_combinations = [
                    dummy_atoms
                ]  # All dummy atoms should be deleted for a coordinated ring
            elif len(metal_coordinating_indices) == 5:
                # Check that all coordinating atoms have exactly 4 bonds (typical for 5-membered aromatic rings)
                if all(
                    ligand_mol.GetAtomWithIdx(idx).GetDegree() == 4
                    for idx in metal_coordinating_indices
                ):
                    dummy_atom_combinations = [dummy_atoms[1:]]
                elif (
                    sum(
                        ligand_mol.GetAtomWithIdx(idx).GetDegree() == 4
                        for idx in metal_coordinating_indices
                    )
                    == 4
                ):
                    dummy_atom_combinations = [
                        dummy_atoms
                    ]  # All dummy atoms should be deleted for a coordinated ring where one atom isn't saturated
                else:
                    dummy_atom_combinations = []
                    for k in range(len(dummy_atoms), -1, -1):
                        dummy_atom_combinations.extend([*combinations(dummy_atoms, k)])
        else:
            dummy_atom_combinations = []
            for k in range(len(dummy_atoms), -1, -1):
                dummy_atom_combinations.extend([*combinations(dummy_atoms, k)])

        ligand_prospects = {}
        for j, delete_list in enumerate(dummy_atom_combinations):
            new_ligand = sanitize_ligand(ligand_mol, delete_list=delete_list)
            if new_ligand is not None:
                total_charge_after, hanging_bonds_after, charged_atoms_after = (
                    brd.assess_atoms(new_ligand)
                )
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


def detect_additional_bonds(
    mol: Chem.rdchem.Mol,
    index: int | None = None,
    distance_tolerance: float = 0.2,
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
        if a.GetAtomicNum() in METALS_NUM:
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
        If the sum across all dimensions for one atom is equal to this value, then a coordinate is missing.

    Returns
    -------
    bool
        Whether missing coordinates were detected.
    """

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    pos_sum = np.sum(positions, axis=-1)

    return any(pos_sum == value)


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
        if a.GetAtomicNum() in METALS_NUM:
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
    n_group: int = group_numbers[atom.GetSymbol()]
    charge = n_group + n_xtype + 2 * n_ltype - n_electrons
    oxidation_state = n_xtype + charge

    # Shift values based on realistic oxidation states
    oxidation_states: list[int] = expected_oxidation_states[atom.GetSymbol()]
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
    metal_smarts = ",".join(f"#{atomic_num}" for atomic_num in METALS_NUM)
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
        if sum([a.GetAtomicNum() in METALS_NUM for a in f.GetAtoms()])
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
    mol = detect_additional_bonds(mol)

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
            - ``number_Xtype_connectors``: total X-type connectors across ligands.
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
                "number_Ltype_connectors": sum(
                    len(x.l_type_connectors) for x in ligand_info
                ),
                "number_Xtype_connectors": sum(
                    len(x.x_type_connectors) for x in ligand_info
                ),
                "total_ligand_charge": sum(int(x.total_charge) for x in ligand_info),
            }
        )

    return combinations_out


def _score_and_flatten_states(
    tm_mol: Chem.rdchem.Mol,
    ligand_combinations: list[dict],
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
    expected_oxs = set(expected_oxidation_states[metal_symbol])

    scored_states: list[ComplexState] = []
    for combo in ligand_combinations:
        n_ltype = int(combo["number_Ltype_connectors"])
        n_xtype = int(combo["number_Xtype_connectors"])
        total_ligand_charge = int(combo["total_ligand_charge"])
        ligand_info = combo["ligand_info"]
        candidate_ids = combo["candidate_ids"]
        residual_valence_penalty = sum(int(x.hanging_bonds) for x in ligand_info)

        tm_oxs, tm_chgs, tm_nels = get_tm_attributes(tm_mol, n_ltype, n_xtype)
        for tm_ox, tm_chg, tm_nel in zip(tm_oxs, tm_chgs, tm_nels):
            oxidation_penalty = 0 if int(tm_ox) in expected_oxs else 1
            predicted_complex_charge = int(tm_chg) + total_ligand_charge
            charge_penalty = abs(predicted_complex_charge - target_complex_charge)
            electron_penalty = abs(int(tm_nel) - target_electron_count)

            score = (
                1000 * oxidation_penalty
                + 100 * charge_penalty
                + 10 * electron_penalty
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
            if atom.GetAtomicNum() in METALS_NUM:
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
        tm_mol, ligand_combinations, target_complex_charge=target_charge
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
    for state in scored_states:
        tmp_tm_mol = copy.deepcopy(tm_mol)
        tmc_mol = reform_metal_complex(
            tmp_tm_mol,
            state.ligands.ligand_info,
            coordinating_atoms,
            tm_charge=state.metal.charge,
            sanitize=sanitize,
        )
        state.complex = ComplexInfo(
            rdmol=tmc_mol,
            smiles=mol_to_smiles(tmc_mol),
            formula=get_molecular_formula(tmc_mol),
            charge=int(sum(a.GetFormalCharge() for a in tmc_mol.GetAtoms())),
            number_metal_connections=n_bonds,
            geometry_type=geometry_type,
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
        - The function does not sanitize the resulting molecule, as this may break certain structures.
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
