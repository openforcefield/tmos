"""Functions for building RDKit molecules.

Such functions include:
 - Utilities to import molecules from one package to RDKit
 - Determine the connectivity of the molecule from XYZ coordinates using RDKit or OpenBabel
 - Determining the bond orders of a molecule using RDKit, OpenBabel, or MDAnalysis

"""

import copy
from loguru import logger
import itertools
import os
from collections import defaultdict

import numpy as np
from deepdiff import DeepDiff

from qcelemental.physical_constants import constants
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.rdDetermineBonds import DetermineBondOrders
from rdkit import RDLogger

import periodictable
from MDAnalysis.converters.RDKitInferring import MDAnalysisInferrer
from openbabel import openbabel as ob

from .utils import first_traceback, get_molecular_formula
from .graph_mapping import (
    find_atom_mapping,
    implicit_hydrogen_atom_mapping,
    find_molecular_rings,
    mol_to_graph,
)
from .reference_values import (
    bond_order_dict,
    bond_type_dict,
    transition_metal_covalent_radii,
    METALS_NUM,
)

# Suppress all OpenBabel output including stereochemistry errors
ob.obErrorLog.SetOutputLevel(0)
ob.obErrorLog.StopLogging()
os.environ["BABEL_SILENCE"] = "1"

# Disable all RDKit logging
RDLogger.DisableLog("rdApp.*")
pt = GetPeriodicTable()

#############################################################################
############################## Atom Assessment ##############################
#############################################################################


def get_atom_charge(atom, ignore_multiple_charges=True, use_formal_charge=False):
    """Get the effective charge of an atom based on its default valence and total bond orders.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom to assess.
    ignore_multiple_charges : bool, optional, default=True
        If False, raises an error when multiple possible charges are encountered;
        otherwise, selects the lowest charge. This is relevant for atoms like sulfur
        that can have multiple charge states.
    use_formal_charge : bool, optional, default=False
        Whether to use the atomic formal charge. If True, the output is the number
        of hanging bonds (negative for hanging, positive for excess). Either may
        indicate that the formal charge of the atom should be updated.

    Returns
    -------
    float
        The formal charge of the atom. May be fractional if aromatics are involved.
        If `use_formal_charge` is True, returns the number of excess bonds.

    Raises
    ------
    ValueError
        If the atom could have multiple charge states given the bonding orders and
        `ignore_multiple_charges` is False.

    Notes
    -----
    If multiple possible charges are found and `ignore_multiple_charges` is True,
    the lowest charge is selected.
    """
    Ntotalbonds = np.array(pt.GetValenceList(atom.GetAtomicNum()))
    if Ntotalbonds[0] == -1:
        Ntotalbonds = np.array([0])
    bond_orders = [bond_order_dict[b.GetBondType().name] for b in atom.GetBonds()]
    charge = sum(bond_orders) - Ntotalbonds
    if use_formal_charge:
        charge -= atom.GetFormalCharge()
    charge = charge[np.where(np.min(np.abs(charge)) == np.abs(charge))[0]]
    if len(charge) > 1 and not ignore_multiple_charges:
        if atom.HasProp("__original_index"):
            raise ValueError(
                f"Atom {atom.GetSymbol()}, Ligand Index: {atom.GetIdx()}, Complex Index: "
                f"{atom.GetIntProp('__original_index')}, can have multiple charge states {charge}"
            )
        else:
            raise ValueError(
                f"Atom {atom.GetSymbol()}, Ligand Index: {atom.GetIdx()} "
                f"can have multiple charge states {charge}"
            )

    min_charge = min(charge)
    if abs(min_charge - round(min_charge)) < np.finfo(float).eps:
        return int(round(min_charge))
    return min_charge


def assess_atoms(mol, add_atom=None, use_formal_charge=False):
    """Assess an RDKit molecule's atoms to determine which are charged or not fully satisfied.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule.
    add_atom : str, optional
        If provided, the number of hanging bonds will be the number of this atom type present.
    use_formal_charge : bool, optional
        Whether to use the atomic formal charge in :func:`get_atom_charge`.

    Returns
    -------
    charge : float
        Total charge of the molecule.
    hanging_bonds : int
        Number of hanging bonds to contribute toward the oxidation state of the metal.
    charged_atoms : dict
        Detailed information on atoms showing a charge or not satisfied with full bonding.

        Each entry contains:
            symbol : str
                Atomic symbol.
            charge : int
                Formal charge of the atom as defined by :func:`get_atom_charge`.
            bond_orders : list of tuple
                List of bonds to this atom. Each bond is represented by a tuple:
                    bond_type : str
                        rdkit.Chem.BondType.name
                    bond_order : int
                        Custom integer value representing the bond type (DATIVE=0).
                    BeginAtom_Symbol : str
                        Atomic symbol of the "BeginAtom".
                    BeginAtomIdx : int
                        Molecule index of the "BeginAtom".
                    EndAtom_Symbol : str
                        Atomic symbol of the "EndAtom".
                    EndAtomIdx : int
                        Molecule index of the "EndAtom".
    """
    if mol is None:
        raise ValueError("Molecule is None")

    mol = copy.deepcopy(mol)

    charged_atoms = {}
    total_charge = 0
    hanging_bonds = 0

    if add_atom is not None:
        atoms_to_remove = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() == add_atom and not a.HasProp("__original_index")
        ]
        for idx in sorted(atoms_to_remove, reverse=True):
            mol.RemoveAtom(idx)

    for a in mol.GetAtoms():
        charge = get_atom_charge(a)
        if charge != 0:
            total_charge += charge
            charged_atoms[a.GetIdx()] = {
                "symbol": a.GetSymbol(),
                "charge": charge,
                "bond_orders": [
                    (
                        b.GetBondType().name,
                        bond_order_dict[b.GetBondType().name],
                        b.GetBeginAtom().GetSymbol(),
                        b.GetBeginAtomIdx(),
                        b.GetEndAtom().GetSymbol(),
                        b.GetEndAtomIdx(),
                    )
                    for b in a.GetBonds()
                ],
            }
            tmp_hanging_bonds = -get_atom_charge(a)  # , use_formal_charge=True)
            if tmp_hanging_bonds > 0:
                hanging_bonds += tmp_hanging_bonds
            elif tmp_hanging_bonds < 0:  # Excess of bonds
                logger.debug(
                    f"Atom {a.GetIdx()}, {a.GetSymbol()}, should have a +{-tmp_hanging_bonds} charge."
                )
            elif tmp_hanging_bonds % 1 > np.finfo(float).eps:
                logger.debug("Hanging bond is not an integer value")

    return total_charge, int(hanging_bonds), charged_atoms


def update_formal_charges(mol):
    """Update the formal charges of a molecule to align with their implied charge from connectivity.

    Note that formal charges of metals are not updated.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to update in place
    """
    for atm in mol.GetAtoms():
        if atm.GetAtomicNum() in METALS_NUM:
            continue
        charge = get_atom_charge(atm)
        if np.finfo(float).eps < abs(charge) < 1 - np.finfo(float).eps:
            logger.debug(
                f"Atom has a fractional charge {charge} (aromatic), setting to 0"
            )
            atm.SetFormalCharge(0)
        else:
            atm.SetFormalCharge(int(get_atom_charge(atm)))


def copy_atom_coords(mol1, index1, mol2, index2, confId1=0, confId2=0):
    """Copy the coordinates of one atom to another

    Parameters
    ----------
    mol1 : rdkit.Chem.Mol
        RDKit target molecule
    index1 : int
        Index of target atom
    mol2 : rdkit.Chem.Mol
        RDKit reference molecule
    index2 : int
        Index of reference atom
    confId : int, optional, default=0
        Conformer index
    """
    conf1 = mol1.GetConformer(confId1)
    conf2 = mol2.GetConformer(confId2)

    pos = conf2.GetAtomPosition(index2)  # returns Point3D
    conf1.SetAtomPosition(index1, Point3D(pos.x, pos.y, pos.z))


#############################################################################
########################### Conversion Functions ############################
#############################################################################


def qcelemental_to_rdkit(qcel_molecule, use_connectivity=True, distance_tolerance=0.2):
    """
    Convert a QCElemental molecule to an RDKit molecule.

    Parameters:
    -----------
    qcel_molecule : qcelemental.models.Molecule
        The QCElemental molecule object
    use_connectivity : bool
        Whether to use existing connectivity information if available
    distance_tolerance : float, optional, default=0.1
        Additional tolerance for bond distance cutoffs (Angstroms) in :func:`determine_connectivity`

    Returns:
    --------
    rdkit.Chem.Mol
        RDKit molecule object
    """

    mol = Chem.RWMol()
    for i, symbol in enumerate(qcel_molecule.symbols):
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    mol.UpdatePropertyCache(strict=False)

    # Set 3D coordinates
    if qcel_molecule.geometry is not None:
        coords = qcel_molecule.geometry.reshape(-1, 3) * constants.bohr2angstroms
        conformer = Chem.Conformer(len(qcel_molecule.symbols))
        for i, (x, y, z) in enumerate(coords):
            conformer.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conformer)

    # Add bonds if connectivity is provided and requested
    if (
        use_connectivity
        and hasattr(qcel_molecule, "connectivity")
        and qcel_molecule.connectivity is not None
    ):
        for bond in qcel_molecule.connectivity:
            atom1_idx, atom2_idx, bond_order = bond
            if bond_order not in bond_type_dict:
                bond_order = 1

            mol.AddBond(atom1_idx, atom2_idx, bond_type_dict[bond_order])
    else:
        mol = determine_connectivity(mol, distance_tolerance=distance_tolerance)

    mol.UpdatePropertyCache(strict=False)

    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
        )
    except Exception:  # Doesn't always work for metal complexes
        pass

    return mol


def xyz_to_rdkit(
    symbols, positions, distance_tolerance=0.2, method="hybrid", ignore_scale=False
):
    """
    Convert a QCElemental molecule to an RDKit molecule.

    Parameters:
    -----------
    symbols : list[str]
        List of element symbols
    positions : numpy.ndarray
        Matrix of the same length as ``symbols`` with x, y, and z coordinates in Angstroms
    distance_tolerance : float, optional, default=0.1
        Additional tolerance for bond distance cutoffs (Angstroms) in :func:`determine_connectivity`
    ignore_scale : bool, optional, default=False
        If True avoid an error when "H" is present and the minimum atomic distance is not between
        0.8 Å and 1.5 Å.
    method : str, optional, default="hybrid"
        Method to determine connectivity in :func:`determine_connectivity`

    Returns:
    --------
    rdkit.Chem.Mol
        RDKit molecule object
    """

    if len(symbols) != len(positions):
        raise ValueError(
            "Number of provided elements does not match the number of provided positions."
        )
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    distances[np.where(distances == 0)] = np.nan
    if np.sum(np.isnan(distances)) != len(symbols):
        raise ValueError("Atoms are overlapping")
    if (
        "H" in symbols
        and not ignore_scale
        and (0.8 >= np.nanmin(distances) or np.nanmin(distances) >= 1.5)
    ):
        raise ValueError(
            "Minimum distance is not in the range of a bonded hydrogen; a unit conversion may be required. If this is intentional, consider setting `ignore_scale=True`"
        )

    mol = Chem.RWMol()
    for i, symbol in enumerate(symbols):
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    # Set 3D coordinates
    conformer = Chem.Conformer(len(symbols))
    for i, pos in enumerate(positions):
        conformer.SetAtomPosition(i, Point3D(*pos))
    mol.AddConformer(conformer)

    mol.UpdatePropertyCache(strict=False)
    mol = mol.GetMol()
    mol = determine_connectivity(
        mol, distance_tolerance=distance_tolerance, method=method
    )
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
        )
    except Exception:  # Doesn't always work for metal complexes
        pass

    return mol


###############################################################################
########################### Connectivity Functions ############################
###############################################################################


def determine_connectivity(rdkit_mol, method="hybrid", distance_tolerance=0.2):
    """
    Determine connectivity for molecules, particularly transition metal organometallic complexes.

    Parameters:
    -----------
    rdkit_mol : rdkit.Chem.Mol
        RDKit molecule without bonds
    method : str, optional, default='hybrid'
        Method to use: 'rdkit', 'openbabel', "None", or 'hybrid' where openbabel is attempted and rdkit
        is the fall back.
    distance_tolerance : float, optional, default=0.1
        Additional tolerance for bond distance cutoffs (Angstroms)

    Returns:
    --------
    rdkit.Chem.Mol
        RDKit molecule with bonds added
    """
    if method == "openbabel":
        return _determine_connectivity_openbabel(rdkit_mol)
    elif method == "rdkit":
        return _determine_connectivity_rdkit(rdkit_mol, distance_tolerance)
    elif method == "None":
        return rdkit_mol
    else:
        return _determine_connectivity_hybrid(rdkit_mol, distance_tolerance)


def _determine_connectivity_openbabel(rdkit_mol):
    """Use OpenBabel to determine connectivity."""

    ob_mol = ob.OBMol()
    ob_conv = ob.OBConversion()
    ob_conv.SetInAndOutFormats("mol", "mol")
    mol_block = Chem.MolToMolBlock(rdkit_mol, forceV3000=True)
    ob_conv.ReadString(ob_mol, mol_block)

    # Determine connectivity
    ob_mol.ConnectTheDots()
    ob_mol.PerceiveBondOrders()

    # Convert back to RDKit
    mol_block_with_bonds = ob_conv.WriteString(ob_mol)
    mol_with_bonds = Chem.MolFromMolBlock(mol_block_with_bonds, sanitize=False)

    return mol_with_bonds


def _get_covalent_radius(symbol, fallback_radius=1.5):
    """Get covalent radius for an element symbol using periodictable."""
    try:
        element = getattr(periodictable, symbol)
        # periodictable stores covalent radius in pm, convert to Angstroms
        if hasattr(element, "covalent_radius") and element.covalent_radius is not None:
            return element.covalent_radius
        elif _is_transition_metal(symbol):
            return transition_metal_covalent_radii.get(symbol, fallback_radius)
        else:
            return fallback_radius
    except AttributeError:
        return fallback_radius


def _is_transition_metal(symbol):
    """Check if an element is a transition metal using periodictable."""

    return symbol in transition_metal_covalent_radii


def _correct_sulfonate_phosphate_interaction(mol, distances, r_cut=3.0, dist_frac=0.05):
    """Identify atoms that should not bond to metal centers.

    Finds indices of atoms that should not be bonded to the metal center either because:
    1. They are Hydrogen atoms already bonded to something
    2. They are atoms in central_sym (S, P, C, H) connected to electronegative atoms (O, N)
       that are significantly closer to the metal (more than dist_frac closer)

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule
    distances : numpy.ndarray
        Distance matrix between all atoms
    r_cut : float, optional, default=3.0
        Distance cutoff (in Å) for considering atoms near metals
    dist_frac : float, optional, default=0.05
        Fractional distance threshold. If connected electronegative atoms are closer to the metal
        by more than this fraction, the central atom should not bond to the metal.

    Returns
    -------
    block_bond_idx : list
        Atom indices that should not bond to metals
    """

    metal_indices = [
        a.GetIdx() for a in mol.GetAtoms() if _is_transition_metal(a.GetSymbol())
    ]
    electronegative_sym = ["O", "N", "C"]
    central_sym = [
        "S",
        "P",
        "C",
        "H",
        "N",
    ]  # Ensure bonds aren't mistakenly made with these

    possible_central_atoms = set()
    possible_elec_neg_idx = set()
    for metal_idx in metal_indices:
        for i, atm in enumerate(mol.GetAtoms()):
            if i != metal_idx and distances[metal_idx, i] <= r_cut:
                if atm.GetSymbol() in electronegative_sym:
                    possible_elec_neg_idx.add(i)
                if atm.GetSymbol() in central_sym:
                    possible_central_atoms.add(atm)

    block_bond_idx = []
    for atm in possible_central_atoms:
        # Block hydrogens that are already bonded to something
        if atm.GetSymbol() == "H" and atm.GetDegree() > 0:
            block_bond_idx.append(atm.GetIdx())
            continue

        # Find closest metal to this atom
        metal_idx = min(metal_indices, key=lambda m: distances[m, atm.GetIdx()])
        d_center = distances[metal_idx, atm.GetIdx()]

        # Get distances from metal to electronegative neighbors
        d_connections = np.array(
            [
                distances[metal_idx, a.GetIdx()]
                for b in atm.GetBonds()
                for a in [b.GetEndAtom(), b.GetBeginAtom()]
                if a.GetIdx() != atm.GetIdx() and a.GetIdx() in possible_elec_neg_idx
            ]
        )

        # If connections are significantly closer to metal than the central atom, block it
        # print(
        #    atm.GetIdx(),
        #    d_center,
        #    [distances[metal_idx, a.GetIdx()] for b in atm.GetBonds() for a in [b.GetEndAtom(), b.GetBeginAtom()]], # if a.GetIdx() != atm.GetIdx()],
        #    d_connections,
        #    (d_center - d_connections) / d_center
        # ) # NoteHere
        if (
            len(d_connections) > 0
            and np.max((d_center - d_connections) / d_center) > dist_frac
        ):
            block_bond_idx.append(atm.GetIdx())

    return block_bond_idx


def _determine_connectivity_rdkit(
    rdkit_mol, max_distance_tolerance=0.2, min_distance_tolerance=0.4
):
    """
    Determine molecular connectivity using RDKit with custom logic for metal atoms.

    This function analyzes the 3D coordinates of atoms in an RDKit molecule and adds bonds between atoms
    based on covalent radii and distance thresholds. It applies special rules for transition metals by
    increasing the bonding threshold to better account for metal complexes.

    Parameters
    ----------
    rdkit_mol : rdkit.Chem.Mol
        The input RDKit molecule, which must have at least one conformer with 3D coordinates.
    max_distance_tolerance : float, optional
        The maximum additional distance (in angstroms) allowed beyond the sum of covalent radii for bond formation.
        Default is 0.2.
    min_distance_tolerance : float, optional
        The minimum distance (in angstroms) below which atoms are considered too close to form a bond. Default is 0.4.

    Returns
    -------
    rdkit.Chem.Mol
        A new RDKit molecule object with bonds added according to the connectivity rules.

    Raises
    ------
    ValueError
        If the input molecule does not have any conformers.

    Notes
    -----
    - Transition metals are handled with a larger bonding threshold (scaled by 1.3).
    - Atoms are considered bonded if their distance is between the minimum and maximum thresholds and no bond already exists.
    """

    mol = Chem.RWMol(rdkit_mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("No conformers are available for this RDKit molecule.")
    else:
        conformer = mol.GetConformer()

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    radii = np.array([_get_covalent_radius(sym) for sym in symbols])
    positions = np.array(
        [conformer.GetAtomPosition(i) for i in range(len(radii))], dtype=float
    )
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    metal_idx = [
        a.GetIdx() for a in mol.GetAtoms() if _is_transition_metal(a.GetSymbol())
    ][0]

    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            one_is_tm = _is_transition_metal(symbols[i]) or _is_transition_metal(
                symbols[j]
            )
            factor = 2 if one_is_tm else 1
            max_bond_threshold = radii[i] + radii[j] + max_distance_tolerance * factor
            min_bond_threshold = radii[i] + radii[j] - min_distance_tolerance * factor

            if (
                distances[i, j] > min_bond_threshold
                and distances[i, j] < max_bond_threshold
                and mol.GetBondBetweenAtoms(i, j) is None
            ):
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    for idx in _correct_sulfonate_phosphate_interaction(mol, distances):
        if mol.GetBondBetweenAtoms(metal_idx, idx) is not None:
            mol.RemoveBond(metal_idx, idx)

    return mol.GetMol()


def _determine_connectivity_hybrid(rdkit_mol, distance_tolerance=0.2):
    """Use both RDKit and OpenBabel for best results."""

    try:
        mol_rdkit = _determine_connectivity_rdkit(rdkit_mol, distance_tolerance)
        if mol_rdkit is not None and mol_rdkit.GetNumBonds() > 0:
            return mol_rdkit
    except Exception:
        pass

    return _determine_connectivity_openbabel(rdkit_mol)


#############################################################################
########################### Bond Order Functions ############################
#############################################################################


def get_connections(mol, indices=None):
    """Get atom connectivity descriptors grouped by element symbol and degree.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    indices : list, optional
        Atom indices to analyze. If None, analyzes all atoms.

    Returns
    -------
    dict
        Keys are (element_symbol, degree) tuples, values are lists of atom indices.
    """
    atoms = (
        mol.GetAtoms()
        if indices is None
        else [a for a in mol.GetAtoms() if a.GetIdx() in indices]
    )
    descriptors = defaultdict(list)
    for a in atoms:
        descriptors[(a.GetSymbol(), a.GetDegree())].append(a.GetIdx())

    return descriptors


def case_carbon_nitrogen_rings(mol):
    aromatic_bonds = []
    positive_charge = []
    rings = find_molecular_rings(mol_to_graph(mol), min_ring_size=6, max_ring_size=6)
    logger.debug(f"There are {len(rings)} 6-member rings")
    for ring in rings:
        ac = get_connections(mol, ring)
        if (
            len(ac[("C", 3)]) == 6
            or (len(ac[("C", 3)]) == 5 and (len(ac[("N", 3)]) + len(ac[("N", 2)]) == 1))
            or (len(ac[("C", 3)]) == 4 and (len(ac[("N", 3)]) + len(ac[("N", 2)]) == 2))
        ):
            aromatic_bonds.extend(
                [sorted(ring[x - 1 : x + 1]) for x in range(1, len(ring))]
                + [sorted([ring[-1], ring[0]])]
            )
            if len(ac[("N", 3)]) == 1:
                positive_charge.append(ac[("N", 3)][0])

    logger.debug(f"There are {len(aromatic_bonds)} aromatic bonds")
    _, _, charged_atoms_before = assess_atoms(mol)
    for i, bond in enumerate(mol.GetBonds()):
        atm1, atm2 = bond.GetBeginAtom(), bond.GetEndAtom()
        pair = sorted([atm1.GetIdx(), atm2.GetIdx()])
        if (
            pair in aromatic_bonds
            and atm1.GetIdx() in charged_atoms_before
            and atm2.GetIdx() in charged_atoms_before
        ):
            adjacent_bonds = [
                b for b in aromatic_bonds if len(set(pair).intersection(b)) > 0
            ]
            if any(
                mol.GetBondBetweenAtoms(*b).GetBondTypeAsDouble() == 2.0
                for b in adjacent_bonds
            ):
                bond_order = 1
            else:
                bond_order = 2
            bond.SetBondType(bond_type_dict[bond_order])

    logger.debug(f"There are {len(positive_charge)} charged atoms")
    for idx in positive_charge:
        mol.GetAtoms()[idx].SetFormalCharge(1)

    mol.UpdatePropertyCache(strict=False)


def case_nitro(mol):
    """
    Identify nitro groups (*~N(~O)~O) in the molecule and set them to *-[N+](=O)-[O-],
    excluding cases where the nitro group is part of *~N(~O)~OH.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to modify in place.

    Returns
    -------
    None
        The molecule is modified in place.
    """
    nitro_pattern = Chem.MolFromSmarts("[N;X3](~[O;X1])~[O;X1]")
    matches = mol.GetSubstructMatches(nitro_pattern)
    for match in matches:
        central_nitrogen, oxygen1, oxygen2 = match

        nitrogen_atom = mol.GetAtomWithIdx(central_nitrogen)
        nitrogen_atom.SetFormalCharge(1)
        nitrogen_atom.SetNoImplicit(True)

        bond1 = mol.GetBondBetweenAtoms(central_nitrogen, oxygen1)
        bond2 = mol.GetBondBetweenAtoms(central_nitrogen, oxygen2)
        if bond2.GetBondType() == Chem.BondType.DOUBLE:
            bond1, bond2 = bond2, bond1
            oxygen1, oxygen2 = oxygen2, oxygen1

        oxygen_atom1 = mol.GetAtomWithIdx(oxygen1)
        oxygen_atom1.SetFormalCharge(0)
        bond1.SetBondType(Chem.BondType.DOUBLE)

        oxygen_atom2 = mol.GetAtomWithIdx(oxygen2)
        oxygen_atom2.SetFormalCharge(-1)
        bond2.SetBondType(Chem.BondType.SINGLE)

    mol.UpdatePropertyCache(strict=False)


def set_special_cases(mol):
    """Set aromatic bonds and formal charges for special molecular cases.

    This is to be run before bond determination

    Identifies:
    1. 6-membered carbon rings: Sets aromatic bond types
    2. Pyridine-like rings: Sets aromatic bond types
    3. Quaternary nitrogens: Assigns positive charges

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to modify in place.
    """
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    if any(atm.GetNumImplicitHs() > 0 for atm in mol.GetAtoms()):
        raise ValueError("Provided molecule has implicit hydrogen atoms.")

    case_nitro(mol)
    case_carbon_nitrogen_rings(mol)

    # Set positive charge: Nitrogen with 4 connections
    positive_charges = get_connections(mol)[("N", 4)]
    logger.debug(f"There are {len(positive_charges)} charged atoms")
    for idx in positive_charges:
        mol.GetAtoms()[idx].SetFormalCharge(1)

    mol.UpdatePropertyCache(strict=False)

    return mol


def correct_special_cases(mol):
    # Resolve the following CCD structures: 'KHK', 'N7H', 'NTE', 'NXC', 'SXC', 'T8K', 'U0J', 'VL2'
    case_nitro(mol)


def determine_bonds_mda(mol):
    """Determine bond orders with MDAnalysis, or None if failed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule that needs to be updated

    Returns
    -------
    mol : rdkit.Chem.Mol
        New RDKit molecule from mdanalysis determination of bond orders. Note that atom
        properties may have been lost. Returns None if bond determination failed.
    """
    mol = set_special_cases(mol)
    totalcharge, _, charged_atoms_before = assess_atoms(mol)

    try:
        mol = Chem.RWMol(mol)
        MolBondInferrer = MDAnalysisInferrer(max_iter=10000)
        mol = MolBondInferrer(mol)
        _, _, charged_atoms_after = assess_atoms(mol)

        if (
            len(DeepDiff(charged_atoms_before, charged_atoms_after)) == 0
            and totalcharge != 0
        ):
            logger.debug("MDAnalysis failed to determine molecular bond orders.")
            mol = None
    except Exception:
        logger.trace(first_traceback())
        logger.debug("MDAnalysis failed to determine molecular bond orders.")
        mol = None

    return mol


def determine_bonds_rdkit(mol, charge=0):
    """Determine bond orders with RDKit, or None if failed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule that needs to be updated
    charge : int
        Set the charge of the molecule when determining the bond orders

    Returns
    -------
    mol : rdkit.Chem.Mol
        New RDKit molecule from rdkit determination of bond orders. Note that atom
        properties may have been lost. Returns None if bond determination failed.
    """
    mol = set_special_cases(mol)
    mol = Chem.RWMol(mol)
    try:
        DetermineBondOrders(
            mol, charge=charge, maxIterations=1000, allowChargedFragments=False
        )
    except Exception:
        logger.debug(first_traceback())
        logger.debug("RDKit failed to determine molecular bond orders.")
        mol = None

    return mol


def determine_bonds_openbabel(mol):
    """Determine bond orders with Open Babel, or None if failed.

    Note that atom properties may have been lost and the default from openbabel is to
    process the molecule with implicit hydrogens only, so all hydrogens and their positions
    are deleted and then optionally restores with approximate coordinates.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule that needs to be updated

    Returns
    -------
    mol : rdkit.Chem.Mol
        New RDKit molecule from openbabel determination of bond orders. Note that atom
        properties may have been lost. Returns None if bond determination failed.
    """
    mol = set_special_cases(mol)
    mol = Chem.RWMol(mol)
    mol.UpdatePropertyCache(strict=False)

    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("mol", "mol")
    obMol = ob.OBMol()
    obConversion.ReadString(obMol, Chem.MolToMolBlock(mol, forceV3000=True))
    for _ in range(5):
        obMol.PerceiveBondOrders()
    aromatyper = ob.OBAromaticTyper()
    aromatyper.AssignAromaticFlags(obMol)
    obMol.PerceiveBondOrders()
    mol_block_processed = obConversion.WriteString(obMol)
    # sanitize=False needed for 5 member rings in porphyrins
    mol_openbabel = Chem.MolFromMolBlock(mol_block_processed, sanitize=False)

    if mol_openbabel is not None and any(
        b.GetBondType().name == "UNSPECIFIED" for b in mol_openbabel.GetBonds()
    ):
        mol_openbabel = None

    if mol_openbabel is None:
        logger.debug("Openbabel failed to determine molecular bond orders.")
    else:
        mol_openbabel.UpdatePropertyCache(strict=False)

    return mol_openbabel


def update_atom_bond_props(mol_to_change, mol_reference):
    """Update atom and bond properties of one molecule to match another.

    Molecules must either be identical in their atomic composition and connectivity, or the reference may be
    the same as the other but with implicit hydrogens, in which case it's assumed that the indices are simply
    shifted.
    Atom properties include the formal charge, whether the atom is aromatic, and setting NoImplicit = True.
    Bond properties include the bond order.

    Parameters
    ----------
    mol_to_change : rdkit.Chem.Mol
        RDKit molecule that needs to be updated
    mol_reference : rdkit.Chem.Mol
        RDKit molecule for reference to update the target molecule

    Raises
    ------
    ValueError
        If the chemical formulas of the provided molecules indicate that they cannot be mapped.
    """

    formula_to_change = get_molecular_formula(mol_to_change)
    formula_to_change_no_H = get_molecular_formula(
        mol_to_change, make_hydrogens_implicit=True
    )
    formula_reference = get_molecular_formula(mol_reference)
    if formula_to_change == formula_reference:
        atom_mapping = find_atom_mapping(mol_to_change, mol_reference)
    elif formula_to_change_no_H == formula_reference:
        atom_mapping = implicit_hydrogen_atom_mapping(mol_to_change)
    else:
        raise ValueError("The provided molecules cannot be mapped.")

    # Correct formal charges
    for orig_idx, correct_idx in atom_mapping.items():
        orig_atom = mol_to_change.GetAtomWithIdx(orig_idx)
        correct_atom = mol_reference.GetAtomWithIdx(correct_idx)
        orig_atom.SetFormalCharge(correct_atom.GetFormalCharge())
        orig_atom.SetNoImplicit(True)
        orig_atom.SetIsAromatic(correct_atom.GetIsAromatic())

    # Correct bonds
    reverse_mapping = {v: k for k, v in atom_mapping.items()}
    target_bonds = {}
    for bond in mol_reference.GetBonds():
        bond_key = tuple(
            sorted(
                [
                    reverse_mapping[bond.GetBeginAtomIdx()],
                    reverse_mapping[bond.GetEndAtomIdx()],
                ]
            )
        )
        target_bonds[bond_key] = bond.GetBondType()

    # Update existing bonds to correct types
    for bond in mol_to_change.GetBonds():
        bond_key = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
        if bond_key in target_bonds:
            target_bond_type = target_bonds[bond_key]
            if bond.GetBondType() != target_bond_type:
                bond.SetBondType(target_bond_type)

    mol_to_change.UpdatePropertyCache(strict=False)
    return mol_to_change


def update_bond_order(mol, idx1, idx2):
    """Update the bond order between two atoms and reset their formal charges.

    This function modifies a bond between two atoms by adjusting the bond type
    based on the current bond order and atom charges, then resets both atoms'
    formal charges to zero.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The RDKit molecule object containing the atoms and bond to modify.
    idx1 : int
        Index of the first atom in the bond.
    idx2 : int
        Index of the second atom in the bond.

    Returns
    -------
    None
        This function modifies the molecule object in-place.

    Raises
    ------
    ValueError
        If the bond order change cannot be rectified due to incompatible
        bond order and atom charge combination.
    Notes
    -----
    - If no bond exists between the specified atoms, the function returns early.
    - Aromatic bonds (bond order 1.5) are left unchanged.
    - After modification, the molecule's property cache is updated.
    """

    bond = mol.GetBondBetweenAtoms(idx1, idx2)
    if bond is None:
        return
    bo = bond_order_dict[bond.GetBondType().name]
    atom1 = mol.GetAtomWithIdx(idx1)
    chg1 = get_atom_charge(atom1)
    atom2 = mol.GetAtomWithIdx(idx2)
    if bo == 1.5:
        return
    logger.debug(
        f"Atm1 {atom1.GetSymbol()} Idx: {idx1} Chg: {chg1}; Atm2 {atom2.GetSymbol()} Idx: {idx2} Chg: {get_atom_charge(atom2)}; Bond {bond.GetBondType().name}={bo}"
    )
    if bo - chg1 not in bond_type_dict:
        raise ValueError(
            f"Bond order change cannot be rectified. Orig Bond Order: {bo}, Atom Charge {chg1}"
        )
    elif bo - chg1 == 0:
        logger.debug(f"No dative bonds! Skipping update between {idx1} and {idx2}")
        return
    bond.SetBondType(bond_type_dict[bo - chg1])
    atom1.SetFormalCharge(0)
    atom2.SetFormalCharge(0)
    mol.UpdatePropertyCache(strict=False)


def get_bo(mol, idx_a, idx_b):
    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    return bond_order_dict[bond.GetBondType().name] if bond is not None else None


def add_obvious_bonds(mol, degree_of_separation=10):
    """Correct bond order for adjacent atoms with hanging bonds.

    Sometimes after the determining bond order analysis (particularly for openbabel)
    there are adjacent atoms, each with a hanging bond. The success of this package
    is improved significantly by increasing those bond orders by one.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to update
    degree_of_separation : int, optional, default=10
        Degree of separation between hanging bonds
    """

    # Check adjacent bonds
    mol.UpdatePropertyCache(strict=False)
    _, _, charged_atoms = assess_atoms(mol)
    if charged_atoms:
        atoms_hanging_bonds = [index for index in charged_atoms.keys()]
        pairs = list(itertools.combinations(atoms_hanging_bonds, 2))
        for idx1, idx2 in pairs:
            if get_atom_charge(mol.GetAtomWithIdx(idx1)) == get_atom_charge(
                mol.GetAtomWithIdx(idx2)
            ):
                update_bond_order(mol, idx1, idx2)

    # Find conjugated rings
    tmp_rings = find_molecular_rings(
        mol_to_graph(mol), min_ring_size=5, max_ring_size=6
    )
    rings = []
    for ring in tmp_rings:
        ac = get_connections(mol, ring)
        if (
            len(ac[("C", 3)]) == 6
            or (len(ac[("C", 3)]) == 5 and (len(ac[("N", 3)]) + len(ac[("N", 2)]) == 1))
            or (len(ac[("C", 3)]) == 4 and (len(ac[("N", 3)]) + len(ac[("N", 2)]) == 2))
            or (len(ac[("C", 3)]) == 5 and sum(len(v) for v in ac.values()) == 5)
            or (
                len(ac[("C", 3)]) == 4
                and sum(len(v) for v in ac.values()) == 5
                and (len(ac[("N", 3)]) == 1 or len(ac[("N", 2)]) == 1)
            )
        ):
            rings.append(ring)
    rings_idxs = [atom for ring in rings for atom in ring]

    if degree_of_separation == 1:
        return
    _, _, charged_atoms = assess_atoms(mol)
    if charged_atoms:
        atoms_hanging_bonds = [
            index for index, tmp in charged_atoms.items() if abs(tmp["charge"]) == 1
        ]
        all_pairs = list(itertools.combinations(atoms_hanging_bonds, 2))
        pair_distances = []
        for idx1, idx2 in all_pairs:
            path = Chem.rdmolops.GetShortestPath(mol, idx1, idx2)
            # If there are hanging bonds, the first bond in the path will increase in order
            # To compensate, the second bond in the path must decrease in order, so it must
            # be greater than a single bond. Check that all bonds expected to increase in order
            # will do so.
            skip_path = False
            chg1 = get_atom_charge(mol.GetAtomWithIdx(idx1))
            for i, (idx_a, idx_b) in enumerate(zip(path[:-1], path[1:])):
                bo = get_bo(mol, idx_a, idx_b)
                if (
                    ((chg1 < 0 and i % 2 != 0) or (chg1 > 0 and i % 2 == 0))
                    and (bo is not None and bo <= 1)
                    and idx_a not in rings_idxs
                    and idx_b not in rings_idxs
                ) or len(path) - 1 > degree_of_separation:
                    skip_path = True
                    break
            if skip_path:
                continue
            else:
                pair_distances.append((len(path) - 1, idx1, idx2, path))
        pair_distances.sort(key=lambda x: x[0])  # shortest first

        # Select pairs ensuring each index appears only once
        used_indices = set()
        updated_bonds = set()
        for _, idx1, idx2, path in pair_distances:
            if idx1 not in used_indices and idx2 not in used_indices:
                used_indices.add(idx1)
                used_indices.add(idx2)
                if idx1 in rings_idxs:
                    path = list(path)[::-1]
                for i in range(len(path) - 1):
                    bond_key = tuple(sorted([path[i], path[i + 1]]))
                    if bond_key in updated_bonds:
                        continue
                    update_bond_order(mol, path[i], path[i + 1])
                    updated_bonds.add(bond_key)

                    # Check if either atom is in a ring and update all ring bonds
                    for ring in rings:
                        ring = list(ring)
                        if path[i] in ring or path[i + 1] in ring:
                            # Rotate the ring to start with either path[i] or path[i+1]
                            if path[i + 1] in ring:
                                start_idx = ring.index(path[i + 1])
                            elif path[i] in ring:
                                start_idx = ring.index(path[i])
                            ring = ring[start_idx:] + ring[:start_idx]
                            chg = get_atom_charge(mol.GetAtomWithIdx(ring[0]))
                            bo = get_bo(mol, ring[0], ring[-1])
                            if (chg < 0 and bo == 1) or (
                                chg > 0 and (bo is not None and bo > 1)
                            ):
                                ring = [ring[0]] + ring[1:][::-1]

                            ring_bonds = [
                                [ring[j], ring[(j + 1) % len(ring)]]
                                for j in range(len(ring))
                                if tuple(sorted([ring[j], ring[(j + 1) % len(ring)]]))
                                not in updated_bonds
                            ]
                            for j, (ring_idx1, ring_idx2) in enumerate(ring_bonds):
                                ring_bond_key = tuple(sorted([ring_idx1, ring_idx2]))
                                chg1 = get_atom_charge(mol.GetAtomWithIdx(ring_idx1))
                                chg2 = get_atom_charge(mol.GetAtomWithIdx(ring_idx2))
                                if (j == len(ring_bonds) - 1 and chg2 == 0) or (
                                    chg1 == 0 and chg2 == 0
                                ):
                                    continue

                                update_bond_order(mol, ring_idx1, ring_idx2)
                                updated_bonds.add(ring_bond_key)
