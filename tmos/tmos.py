"""Sanitize and generate oxidation state and other properties for transition metal complexes.

This module uses function architectures originally produced in [xyz2mol_tm](https://github.com/jensengroup/xyz2mol_tm/). However rather than using the Huckel method
an arrow pushing script is produced here with custom checks for ferrocene structures.
"""

import copy
import warnings
from itertools import combinations

import numpy as np

from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Geometry import Point3D
from rdkit import RDLogger

from .utils import get_molecular_formula
from . import build_rdmol as brd
from .reference_values import bond_type_dict, METALS_NUM, expected_oxidation_states
from .geometry import get_geometry_from_mol

RDLogger.DisableLog("rdApp.*")
pt = GetPeriodicTable()


def sanitize_molecule(
    mol, update_charges=True, sanitize_aromaticity=False, sanitize_kekulize=False
):
    """Sanitize TMC molecule by updating formal charges to apparent value based on connectivity and use
    RDKit sanitization without SANITIZE_SETAROMATICITY or SANITIZE_KEKULIZE.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule to sanitize
    update_charges : bool, optional, default=True
        If True, the charges are updated to apparent values based on connectivity
    sanitize_aromaticity : bool, options, default=False
        If False, removes the flag ``Chem.SanitizeFlags.SANITIZE_SETAROMATICITY``
    sanitize_kekulize : bool, options, default=False
        If False, removes the flag ``Chem.SanitizeFlags.SANITIZE_KEKULIZE``
    """

    if update_charges:
        brd.update_formal_charges(mol)
    sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL
    if not sanitize_aromaticity:
        sanitize_ops ^= (
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        )  # Needed for porphyrins
    if not sanitize_kekulize:  # Should be True for porphyrins
        sanitize_ops ^= Chem.SanitizeFlags.SANITIZE_KEKULIZE
    Chem.SanitizeMol(
        mol,
        sanitizeOps=sanitize_ops,
    )


def mol_from_smiles(smiles, sanitize=False, sanitize_kwargs={}):
    """Convert a SMILES string into a RDKit Molecule

    Parameters
    ----------
    smiles : str
        SMILES string
    sanitize : bool, optional
        Perform sanitization with :func:`sanitize_molecule`, by default False
    sanitize_kwargs : dict, optional, default={}
        Keywords for :func:`sanitize_molecule`.


    Returns
    -------
    rdkit.Chem.rdchem.Mol
        RDKit molecule that was produced
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if sanitize:
        sanitize_molecule(mol, **sanitize_kwargs)
    return mol


def mol_to_smiles(mol):
    """Generate SMILES without isomeric information from a RDKit molecule

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule

    Returns
    -------
    str
        Nonisomeric SMILES string
    """
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def wipe_molecule(mol):
    """Wipe all bond order and aromatic information from a molecule so that only single
    bonds remain.

    Parameters
    ----------
    mol :rdkit.Chem.rdchem.Mol)
        RDKit molecule

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
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


def check_ligand_exception(mol):
    """Corrects formal charges and bonds for molecular exceptions.

    Exceptions include carbon monoxide and hydrazoic acid molecules.
    These molecules cannot be sanitized in RDKit as the connectivity does not align with
    RDKit's expectations.

    All exceptions returned by this function are L-type ligands

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        Resulting molecule, or None if the molecule is not an exception.
    metal_connected_orig_indices : list[int]
        Original index of atom that was connected to the metal
    """
    mol = Chem.RWMol(copy.deepcopy(mol))

    # Assume there is one
    dummy_atoms = [
        atm for atm in mol.GetAtoms() if atm.GetIntProp("__original_index") == -1
    ]
    if dummy_atoms:
        bond = dummy_atoms[0].GetBonds()[0]
        other_atom = (
            bond.GetBeginAtom()
            if bond.GetEndAtom().GetIdx() == dummy_atoms[0].GetIdx()
            else bond.GetEndAtom()
        )
        other_idx = other_atom.GetIdx()
        metal_connected_orig_indices = [other_atom.GetIntProp("__original_index")]
        mol.UpdatePropertyCache(strict=False)
        for atm in sorted(dummy_atoms, key=lambda x: -x.GetIdx()):
            mol.RemoveAtom(atm.GetIdx())
    else:
        metal_connected_orig_indices = []

    formula = get_molecular_formula(mol)
    smiles = {  # Exceptions
        "C1O1": "[C-]#[O+]",
        "H1N3": "[H][N]=[N+]=[N-]",
        "O1": "[O]([H])[H]",  # Instances of oxo tend to be unphysical
        "H1O2": "[O]([H])[H]",  # Instances of peroxide tend to be unphysical
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
        tmp_mol.GetAtoms()[0].SetIntProp(
            "__original_index", metal_connected_orig_indices[0]
        )
        tmp_mol.AddConformer(
            Chem.rdchem.Conformer(tmp_mol.GetNumAtoms()), assignId=True
        )
        brd.copy_atom_coords(tmp_mol, 0, mol, other_idx, confId2=conf_ids[0])
        mol = Chem.AddHs(tmp_mol, explicitOnly=True, addCoords=True)
        for a in mol.GetAtoms():
            if a.GetIdx() == 0:
                continue
            a.SetIntProp(
                "__original_index", -2
            )  # Not an atom of consequence and not in the orig mol

    return mol, metal_connected_orig_indices


def determine_bonds(mol):
    """Determine bond orders with OpenBabel, with MDA fallback.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        Resulting molecule, or None if the bonds could not be determined for the molecule.
    """
    mol = copy.deepcopy(mol)

    new_mol = brd.determine_bonds_openbabel(mol, return_implicit_Hs=True)
    if new_mol is None:
        new_mol = brd.determine_bonds_mda(mol)
    else:
        new_mol = brd.update_atom_bond_props(copy.deepcopy(mol), new_mol)
        _, _, charged_atoms = brd.assess_atoms(new_mol)
        if len(charged_atoms) > 0:
            new_mol_2 = brd.determine_bonds_mda(mol)
            if new_mol_2 is not None:
                _, _, charged_atoms_2 = brd.assess_atoms(new_mol_2)
                new_mol = (
                    new_mol if len(charged_atoms) < len(charged_atoms_2) else new_mol_2
                )

    return new_mol


def sanitize_ligand(
    mol,
    delete_list=[],
    wipe=True,
    tool="hybrid",
    charge=0,
    sanitize=True,
    verbose=False,
):
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
    tool : str, optional, default="hybrid"
        Choose the tool used to determine bond borders.

        - mdanalysis: ``_infer_bo_and_charges``
        - rdkit: ``rdDetermineBonds.DetermineBondOrders``
        - openbabel: ``PerceiveBondOrders``
        - hybrid: Run openbabel, and if None, run mdanalysis

    charge : int, optional, default=0
        If using RDKit for bond orders, optionally set the charge. If set to 0, some atoms may be defined as radicals.
    sanitize : bool, optional, default=True
        If True, the resulting molecule will be sanitized
    verbose : bool, optional, default=False
        Whether RDKit failure is returned

    Returns:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule
    """
    mol_after = copy.deepcopy(mol)
    mol_after.UpdatePropertyCache(strict=False)
    if any(atm.GetNumImplicitHs() > 0 for atm in mol_after.GetAtoms()):
        raise ValueError("Provided molecule has implicit hydrogen atoms.")
    mol_after = Chem.RWMol(mol_after)

    if wipe:
        mol_after = wipe_molecule(mol_after)

    if delete_list:
        delete_list = list(delete_list)
        delete_list.sort(key=lambda x: -x.GetIdx())
    for atm in delete_list:
        mol_after.RemoveAtom(atm.GetIdx())
    mol_after.UpdatePropertyCache(strict=False)

    if tool.lower() == "mdanalysis":
        mol_after = brd.determine_bonds_mda(mol_after)
    elif tool.lower() == "rdkit":
        mol_after = brd.determine_bonds_rdkit(mol_after, charge=charge, verbose=verbose)
    elif tool.lower() == "openbabel":
        mol_tmp = brd.determine_bonds_openbabel(mol_after, return_implicit_Hs=True)
        mol_after = (
            brd.update_atom_bond_props(mol_after, mol_tmp)
            if mol_tmp is not None
            else None
        )
    elif tool.lower() == "hybrid":
        mol_after = determine_bonds(mol_after)
    else:
        raise ValueError(
            f"Bond determination tool, {tool}, is not recognized. Must be hybrid (preferred), openbabel, mdanalysis, or rdkit."
        )

    if mol_after is not None:
        mol_after.UpdatePropertyCache(strict=False)
        brd.add_obvious_bonds(mol_after)
        if sanitize:
            try:
                sanitize_molecule(
                    mol_after, sanitize_kekulize=True
                )  # Settings are set for porphyrins and passing TM Benchmark subset of CCD
            except Chem.rdchem.AtomValenceException:
                warnings.warn("RDKit sanitize failed, passing molecule as is.")

    return mol_after


def get_ligand_attributes(
    ligand_mol, verbose=False, add_atom=None, add_hydrogens=False
):
    """
    Analyze default valence and bonds to determine ligand attributes.

    Parameters
    ----------
    ligand_mol : rdkit.Chem.rdchem.Mol)
        Ligand molecule with dummy atoms replacing ligand-metal bonds,
        denoted by ``atom.GetIntProp("__original_index") == -1``.
    verbose : bool, optional, default=False
        If True, print updates.
    add_atom : str, optional, default=None
        If an element symbol, the number of hanging bonds will be the number of this atom type present.
    add_hydrogens : bool, optional, default=False
        If True, add explicit hydrogens to the ligand.

    Returns
    -------
    dict: ligand_best, with keys:

        - "index" (int): Index of the ligand prospect.
        - "rdmol" (rdkit.Chem.rdchem.Mol): Resolved ligand molecule.
        - "smiles": Canonical explicit hydrogen smiles string generated from the RDKit molecule
                    of the complex. Note that the dummy atom type is present to denote where the metal
                    attaches; commonly I.
        - "total_charge" (int): Total charge of the ligand.
        - "hanging_bonds" (int): Number of unused valencies.
        - "charged_atoms" (dict): Atom information by index as defined in :func:`tmos.build_rdmol.assess_atoms`.
        - "L-type connectors" (list[int]): Original atom indices for L-type connectors.
        - "X-type connectors" (list[int]): Original atom indices for X-type connectors.

    """

    ligand_mol = Chem.RWMol(copy.deepcopy(ligand_mol))
    ligand_mol = Chem.DeleteSubstructs(ligand_mol, Chem.MolFromSmarts("[#0]"))
    ligand_mol.UpdatePropertyCache(strict=False)
    if add_hydrogens:
        ligand_mol = Chem.AddHs(ligand_mol, addCoords=True, explicitOnly=True)
    ligand_mol.UpdatePropertyCache(strict=False)

    tmp_mol, metal_connected_orig_indices = check_ligand_exception(ligand_mol)
    if tmp_mol is not None:
        if verbose:
            print("Ligand exception found.")
        total_charge_after, hanging_bonds_after, charged_atoms_after = brd.assess_atoms(
            tmp_mol
        )
        ligand_best = {
            "index": 0,
            "rdmol": tmp_mol,
            "total_charge": total_charge_after,
            "hanging_bonds": hanging_bonds_after,
            "charged_atoms": charged_atoms_after,
            "L-type connectors": metal_connected_orig_indices,
            "X-type connectors": [],
        }
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
        if verbose:
            print(f"There are {len(dummy_atoms)} dummy atoms")

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
                ligand_prospects[j] = {
                    "index": j,
                    "rdmol": new_ligand,
                    "total_charge": total_charge_after,
                    "hanging_bonds": hanging_bonds_after,
                    "charged_atoms": charged_atoms_after,
                }
                if verbose and len(charged_atoms_after) < 6:
                    print("___________________________________________________")
                    print(f"{j}:", total_charge_after, hanging_bonds_after)
                    for ind, tmp in charged_atoms_after.items():
                        print("    ", ind, tmp)
            else:
                if verbose:
                    print("Sanitize failed")

        # Filter prospective ligands
        if not ligand_prospects:
            raise ValueError("Ligand could not be sanitized.")

        # Find the best ligand prospect by sorting with the desired priorities
        ligand_best = min(
            reversed(
                ligand_prospects.values()
            ),  # All else being equal, choose the greatest number of X-type
            key=lambda x: (
                x["hanging_bonds"],
                abs(x["total_charge"]),
                len(x["charged_atoms"]),
            ),
        )
        ligand_best["L-type connectors"] = [
            metal_connected_atm_indices[x.GetIdx()]
            for x in dummy_atom_combinations[ligand_best["index"]]
        ]
        ligand_best["X-type connectors"] = [
            metal_connected_atm_indices[x.GetIdx()]
            for x in list(
                set(dummy_atoms) - set(dummy_atom_combinations[ligand_best["index"]])
            )
        ]

    if verbose:
        print(
            f"Total ligand charge: {ligand_best['total_charge']}, N_Hanging Bonds {ligand_best['hanging_bonds']}"
        )
        print("Charged atom info:")
        for x, y in ligand_best["charged_atoms"].items():
            print(f"    {x}: {y}")

    sanitize_molecule(ligand_best["rdmol"])
    ligand_best["smiles"] = mol_to_smiles(ligand_best["rdmol"])

    return ligand_best


def assert_same_ring(mol, ind1, ind2, max_ring_size=6):
    """Determine whether two atoms are in the same chemical ring

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule to assess
    ind1 : int
        Index of first atom of interest
    ind2 : int
        Index of second atom of interest
    max_ring_size : int, optional, default=6
        Maximum ring size to consider

    Returns
    -------
    bool
        True if the two indices are in the same ring
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


def correct_ferrocene(mol, index):
    """Correct a ferrocene containing molecule to ensure that all hydrogens are included

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule of interest
    index : int
        Index of the metal center of a ferrocene group

    Returns
    -------
    new_mol : rdkit.Chem.rdchem.Mol
        Output molecule with corrected ferrocene group
    new_index : int
        The atomic index of the ferrocene metal center, if changed
    tm_ox : int
        Oxidation state of the metal, ``tm_ox = 2`` for ferrocene
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
        carbon.SetNoImplicit(False)
        if carbon.GetDegree() < 4:
            carbon.SetNumExplicitHs(1)
        carbon.UpdatePropertyCache(strict=False)
    mol = Chem.AddHs(mol, addCoords=True, explicitOnly=True, onlyOnAtoms=c_atoms)

    for a in mol.GetAtoms():
        a.SetIntProp("__original_index", a.GetIdx())
        if a.GetAtomicNum() in METALS_NUM:
            # tm_atom = a.GetSymbol()
            new_index = a.GetIdx()

    return mol, new_index, 2


def compute_centroid_excluding(conformer, exclude_atoms):
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


def find_missing_coords(mol, value=0):
    """Determine if an RDKit molecule has a relevant geometry

    In PDB CCD if the coordinates are missing, denoted by question marks in the cif, then the coordinate will be (0,0,0)

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule to assess
    value : float, optional, default=0
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


def fix_missing_coords(mol, tmc_idx, missing_coord_indices):
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


def find_metal_index(mol):
    """Find the molecule index for the metal

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Transition state complex

    Raises:
    ValueError
        No transition metal found

    Returns:
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


def get_tm_attributes(tm_mol, n_ltype, n_xtype, n_electrons=18):
    """
    Compute possible oxidation states, formal charges, and electron counts for a transition metal center.

    Parameters
    ----------
    tm_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule containing only the transition metal atom.
    n_ltype : int
        Number of L-type connectors (neutral ligands).
    n_xtype : int
        Number of X-type connectors (anionic ligands).
    n_electrons : int, optional, default=18
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
    n_group = pt.GetNOuterElecs(atom.GetAtomicNum())
    charge = n_group + n_xtype + 2 * n_ltype - n_electrons
    oxidation_state = n_xtype + charge

    # Shift values based on realistic oxidation states
    oxidation_states = expected_oxidation_states[atom.GetSymbol()]
    offsets = np.array(oxidation_states) - oxidation_state
    charges = charge + offsets
    electron_counts = n_electrons - offsets

    return oxidation_states, charges, electron_counts


def cleave_mol_from_index(mol, index, verbose=False, add_atom=None):
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
    verbose : bool, optional, default=False
        If True, provide the number of resulting rdkit molecules
    add_atom : str, optional, default=None
        If not None, add an atom of this type in place of the metal center

    Returns
    -------
    fragments : list[rdkit.Chem.rdchem.Mol])
        List of RDKit molecules resulting from cleaved bonds
    coordinating_atoms : list[int])
        List of atom indices that were connected to the central atom

    """

    params = Chem.MolStandardize.rdMolStandardize.MetalDisconnectorOptions()
    params.splitAromaticC = True
    params.splitGrignards = True
    params.adjustCharges = False

    MetalsOfInterest = "[#3,#11,#12,#19,#13,#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#57,#72,#73,#74,#75,#76,#77,#78,#79,#80]~[B,#6,#14,#15,#33,#51,#16,#34,#52,Cl,Br,I,#85]"

    coordinating_atoms = [
        int(x) for x in np.nonzero(Chem.rdmolops.GetAdjacencyMatrix(mol)[index, :])[0]
    ]
    for a in mol.GetAtoms():
        a.SetIntProp("__original_index", a.GetIdx())

    mdis = rdMolStandardize.MetalDisconnector(params)
    mdis.SetMetalNon(Chem.MolFromSmarts(MetalsOfInterest))
    frags = mdis.Disconnect(mol)
    frag_mols = list(rdmolops.GetMolFrags(frags, asMols=True, sanitizeFrags=False))
    if verbose:
        print(f"Along with the metal, there are {len(frag_mols)-1} ligands")

    ind_metal = [
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
                chg = brd.get_atom_charge(frag.GetAtoms()[idx])
                n_new_atms = 1 if chg >= 0 else -chg
                for _ in range(n_new_atms):
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


def sanitize_complex(mol, verbose=False, value_missing_coord=0, add_hydrogens=False):
    """
    Separate ligands from a transition metal complex and determine appropriate charges.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule representing the transition metal complex.
    verbose : bool, optional, default=False
        If True, print updates during processing.
    value_missing_coord : float, optional, default=0
        Value used to detect missing coordinates (e.g., 0 for (0,0,0)).
    add_hydrogens : bool, optional, default=False
        If True, add explicit hydrogens to the structure if needed.

    Raises
    ------
    ValueError
        If the molecule does not contain a transition metal.

    Returns
    -------
    dict
        Dictionary containing:
            - "metal_info": dict with keys:
                - "rdmol": RDKit molecule of the transition metal center.
                - "oxidation_state": Oxidation state of the metal center.
                - "total_charge": Formal charge of the metal center.
                - "number_electrons": Electron count for the metal center.
            - "ligand_info": list of dict
                List of ligand information dictionaries, each with keys:
                    - "smiles": Canonical explicit hydrogen smiles string generated from the RDKit molecule
                                of the complex. Note that the dummy atom type is present to denote where the
                                metal attaches; commonly I.
                    - "rdmol": RDKit molecule of the ligand.
                    - "total_charge": Total charge of the ligand.
                    - "hanging_bonds": Number of unused valencies.
                    - "charged_atoms": Atom charge information (see :func:`tmos.build_rdmol.assess_atoms`).
                    - "L-type connectors": List of original atom indices for L-type connectors.
                    - "X-type connectors": List of original atom indices for X-type connectors.
            - "complex_info": dict with keys:
                - "smiles": Canonical explicit hydrogen smiles string generated from the RDKit molecule of the complex
                - "rdmol": RDKit molecule of the reformed transition metal complex.
                - "oxidation_state": Oxidation state of the metal center.
                - "total_charge": Overall charge of the complex.
                - "geometry": Geometry information of the complex.
    """
    tm_ox = None
    mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#0]"))
    mol.UpdatePropertyCache(strict=False)
    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=True, explicitOnly=True)
        mol.UpdatePropertyCache(strict=False)

    tmc_idx = find_metal_index(mol)

    # Detect and correct special cases
    if mol.GetAtoms()[tmc_idx].GetDegree() == 10:  # Detect ferrocene
        if verbose:
            print("Detect ferrocene!")
        mol, tmc_idx, tm_ox = correct_ferrocene(mol, tmc_idx)

    missing_coord_indices = find_missing_coords(mol, value=value_missing_coord)
    if missing_coord_indices:
        raise ValueError("Molecule missing coordinates")
    #    mol = fix_missing_coords(mol, tmc_idx, missing_coord_indices)

    # Split the ligands from the metal center, note that we are adding a single bonded At to each atom that
    # was connected to the metal center.
    add_atom = "I"
    frag_mols, coordinating_atoms = cleave_mol_from_index(
        mol, tmc_idx, verbose=verbose, add_atom=add_atom
    )
    geometry = get_geometry_from_mol(mol, tmc_idx)

    total_xtype = 0
    total_ltype = 0
    total_lig_charge = 0
    flag_tm = False
    lig_info = []
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
            if verbose:
                print(f"Ligand {i+1} of {len(frag_mols)-1}")
            best_ligand_info = get_ligand_attributes(
                m, verbose=verbose, add_atom=add_atom
            )

            total_xtype += len(best_ligand_info["X-type connectors"])
            total_ltype += len(best_ligand_info["L-type connectors"])
            total_lig_charge += best_ligand_info["total_charge"]
            lig_info.append(best_ligand_info)

    if not flag_tm:
        raise ValueError("No transition metal found")

    tm_oxs, tm_chgs, tm_nels = get_tm_attributes(tm_mol, total_ltype, total_xtype)

    # Assemble possible complexes
    outputs = {}
    for i, (tm_ox, tm_chg, tm_nel) in enumerate(zip(tm_oxs, tm_chgs, tm_nels)):
        tmp_tm_mol = copy.deepcopy(tm_mol)
        tmc_mol = reform_metal_complex(
            tmp_tm_mol,
            lig_info,
            coordinating_atoms,
            tm_charge=tm_chg,
        )
        charge = sum([a.GetFormalCharge() for a in tmc_mol.GetAtoms()])
        outputs[f"OS: {tm_ox}; q: {charge}; Nel: {tm_nel}"] = {
            "metal_info": {
                "rdmol": tmp_tm_mol,
                "oxidation_state": tm_ox,
                "total_charge": tm_chg,
                "number_electrons": tm_nel,
            },
            "ligand_info": lig_info,
            "complex_info": {
                "smiles": mol_to_smiles(tmc_mol),
                "rdmol": tmc_mol,
                "oxidation_state": tm_ox,
                "total_charge": charge,
                "geometry": geometry,
                "number_Ltype_connectors": sum(
                    [len(x["L-type connectors"]) for x in lig_info]
                ),
                "number_Xtype_connectors": sum(
                    [len(x["X-type connectors"]) for x in lig_info]
                ),
            },
        }

    return outputs


def reform_metal_complex(tm_mol, lig_info, coordinating_atoms, tm_charge=0):
    """Reconnects ligands to a transition metal center to reform a metal complex.

    This function takes a transition metal molecule and a list of ligand molecules,
    then combines them into a single complex. It reconnects the ligands to the metal
    center at specified coordinating atom indices, adjusting bond orders as needed.

    Parameters
    ----------
    tm_mol : tuple[rdkit.Chem.rdchem.Mol, int]
        A tuple containing the RDKit molecule of the transition metal center and its formal charge.
    lig_info : list[tuple[rdkit.Chem.rdchem.Mol, int, int, Any]]
        A list of dictionaries defined in :func:`get_ligand_assessment`
    coordinating_atoms : list[int]
        List of atom indices (from the original complex) that should be reconnected to the metal center.
    tm_charge : int, optional, default=0
        Formal charge of the transition metal center.

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
        ltype_atoms.extend(lig_dict["L-type connectors"])
        xtype_atoms.extend(lig_dict["X-type connectors"])
        tmp_mol = Chem.RWMol(copy.deepcopy(lig_dict["rdmol"]))
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

    sanitize_molecule(tmc_mol)

    return tmc_mol
