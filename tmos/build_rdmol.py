"""Functions for building RDKit molecules.

Such functions include:
 - Utilities to import molecules from one package to RDKit
 - Determine the connectivity of the molecule from XYZ coordinates using RDKit or OpenBabel
 - Determining the bond orders of a molecule using RDKit, OpenBabel, or MDAnalysis

"""

import copy
import warnings
import itertools

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

try:
    from openbabel import openbabel as ob

    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    warnings.warn(
        "OpenBabel not available. Will use RDKit-only connectivity detection."
    )

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

    return min(charge)


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
            tmp_hanging_bonds = -get_atom_charge(a, use_formal_charge=True)
            if tmp_hanging_bonds > 0:
                hanging_bonds += tmp_hanging_bonds
            elif tmp_hanging_bonds < 0:  # Excess of bonds
                warnings.warn(
                    f"Atom {a.GetIdx()}, {a.GetSymbol()}, should have a +{-tmp_hanging_bonds} charge."
                )
            elif tmp_hanging_bonds % 1 > np.finfo(float).eps:
                warnings.warn("Hanging bond is not an integer value")

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
            warnings.warn(
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


def qcelemental_to_rdkit(qcel_molecule, use_connectivity=True, distance_tolerance=0.1):
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
        Chem.SanitizeMol(mol)
    except Exception:  # Doesn't always work for metal complexes
        pass

    return mol


def xyz_to_rdkit(symbols, positions, distance_tolerance=0.1):
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

    Returns:
    --------
    rdkit.Chem.Mol
        RDKit molecule object
    """

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
    mol = determine_connectivity(mol, distance_tolerance=distance_tolerance)
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,  # Needed for porphyrins
        )
    except Exception:  # Doesn't always work for metal complexes
        pass

    return mol


###############################################################################
########################### Connectivity Functions ############################
###############################################################################


def determine_connectivity(rdkit_mol, method="hybrid", distance_tolerance=0.1):
    """
    Determine connectivity for molecules, particularly transition metal organometallic complexes.

    Parameters:
    -----------
    rdkit_mol : rdkit.Chem.Mol
        RDKit molecule without bonds
    method : str, optional, default='hybrid'
        Method to use: 'rdkit', 'openbabel', or 'hybrid' where openbabel is attempted and rdkit
        is the fall back.
    distance_tolerance : float, optional, default=0.1
        Additional tolerance for bond distance cutoffs (Angstroms)

    Returns:
    --------
    rdkit.Chem.Mol
        RDKit molecule with bonds added
    """

    if method == "openbabel" and OPENBABEL_AVAILABLE:
        return _determine_connectivity_openbabel(rdkit_mol)
    elif method == "rdkit" or not OPENBABEL_AVAILABLE:
        return _determine_connectivity_rdkit(rdkit_mol, distance_tolerance)
    else:
        return _determine_connectivity_hybrid(rdkit_mol, distance_tolerance)


def _determine_connectivity_openbabel(rdkit_mol):
    """Use OpenBabel to determine connectivity."""
    if not OPENBABEL_AVAILABLE:
        raise ImportError("OpenBabel not available")

    ob_mol = ob.OBMol()
    ob_conv = ob.OBConversion()
    ob_conv.SetInFormat("sdf")
    ob_conv.ReadString(ob_mol, Chem.MolToMolBlock(rdkit_mol))

    # Determine connectivity
    ob_mol.ConnectTheDots()
    ob_mol.PerceiveBondOrders()

    # Convert back to RDKit
    ob_conv.SetOutFormat("sdf")
    mol_block_with_bonds = ob_conv.WriteString(ob_mol)
    mol_with_bonds = Chem.MolFromMolBlock(mol_block_with_bonds, sanitize=False)

    return mol_with_bonds


def _get_covalent_radius(symbol, fallback_radius=1.5):
    """Get covalent radius for an element symbol using periodictable."""
    try:
        element = getattr(periodictable, symbol)
        # periodictable stores covalent radius in pm, convert to Angstroms
        if hasattr(element, "covalent_radius") and element.covalent_radius is not None:
            return element.covalent_radius / 100.0  # pm to Angstroms
        elif _is_transition_metal(symbol):
            return transition_metal_covalent_radii.get(symbol)
        else:
            return fallback_radius
    except AttributeError:
        return fallback_radius


def _is_transition_metal(symbol):
    """Check if an element is a transition metal using periodictable."""
    try:
        element = getattr(periodictable, symbol)
        # Transition metals have d-block elements (groups 3-12)
        # Plus lanthanides and actinides (f-block)
        return (
            (
                hasattr(element, "group")
                and element.group is not None
                and 3 <= element.group <= 12
            )
            or (57 <= element.number <= 71)
            or (89 <= element.number <= 103)
        )
    except AttributeError:
        # Fallback to hardcoded list for elements not in periodictable
        return symbol in transition_metal_covalent_radii


def _determine_connectivity_rdkit(rdkit_mol, distance_tolerance=0.1):
    """Use RDKit to determine connectivity with custom metal-aware logic."""

    mol = Chem.RWMol(rdkit_mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("No conformers are available for this RDKit molecule.")
    else:
        conformer = mol.GetConformer()

    # Calculate distances and add bonds
    for i in range(mol.GetNumAtoms()):
        symbol_i = mol.GetAtomWithIdx(i).GetSymbol()
        pos_i = conformer.GetAtomPosition(i)

        for j in range(i + 1, mol.GetNumAtoms()):
            symbol_j = mol.GetAtomWithIdx(j).GetSymbol()

            radius_i = _get_covalent_radius(symbol_i)
            radius_j = _get_covalent_radius(symbol_j)
            bond_threshold = radius_i + radius_j + distance_tolerance
            if _is_transition_metal(symbol_i) or _is_transition_metal(symbol_j):
                bond_threshold *= 1.3  # Increase threshold for metal complexes

            if pos_i.Distance(conformer.GetAtomPosition(j)) < bond_threshold:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    return mol.GetMol()


def _determine_connectivity_hybrid(rdkit_mol, distance_tolerance=0.1):
    """Use both RDKit and OpenBabel for best results."""

    if OPENBABEL_AVAILABLE:
        try:
            mol_ob = _determine_connectivity_openbabel(rdkit_mol)
            if mol_ob is not None and mol_ob.GetNumBonds() > 0:
                return mol_ob
        except Exception:
            pass

    # Fall back to RDKit method
    return _determine_connectivity_rdkit(rdkit_mol, distance_tolerance)


#############################################################################
########################### Bond Order Functions ############################
#############################################################################


def determine_bonds_mda(mol, verbose=False):
    """Determine bond orders with MDAnalysis, or None if failed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule that needs to be updated
    verbose : bool, optional, default=False
        If True and an error in bond determination occurs, the primary traceback is printed.

    Returns
    -------
    mol : rdkit.Chem.Mol
        New RDKit molecule from mdanalysis determination of bond orders. Note that atom
        properties may have been lost. Returns None if bond determination failed.
    """
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    if any(atm.GetNumImplicitHs() > 0 for atm in mol.GetAtoms()):
        raise ValueError("Provided molecule has implicit hydrogen atoms.")
    totalcharge, _, charged_atoms_before = assess_atoms(mol)
    rings = find_molecular_rings(mol_to_graph(mol), min_ring_size=6, max_ring_size=6)

    # Find 6-member rings and set aromatic
    aromatic_bonds = [
        sorted(ring[x - 1 : x + 1]) for ring in rings for x in range(1, len(ring))
    ]
    aromatic_bonds += [sorted([ring[-1], ring[0]]) for ring in rings if len(ring) > 2]
    for bond in mol.GetBonds():
        atm1, atm2 = bond.GetBeginAtom(), bond.GetEndAtom()
        pair = sorted([atm1.GetIdx(), atm2.GetIdx()])
        if (
            pair in aromatic_bonds
            and atm1.GetIdx() in charged_atoms_before
            and atm2.GetIdx() in charged_atoms_before
        ):
            bond.SetBondType(bond_type_dict[1.5])

    try:
        mol = Chem.RWMol(mol)
        MolBondInferrer = MDAnalysisInferrer(max_iter=2000)
        mol = MolBondInferrer(mol)
        _, _, charged_atoms_after = assess_atoms(mol)

        if (
            len(DeepDiff(charged_atoms_before, charged_atoms_after)) == 0
            and totalcharge != 0
        ):
            warnings.warn("MDAnalysis failed to determine molecular bond orders.")
            mol = None
    except Exception:
        if verbose:
            warnings.warn(first_traceback())
        warnings.warn("MDAnalysis failed to determine molecular bond orders.")
        mol = None

    return mol


def determine_bonds_rdkit(mol, charge=0, verbose=False):
    """Determine bond orders with RDKit, or None if failed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule that needs to be updated
    charge : int
        Set the charge of the molecule when determining the bond orders
    verbose : bool, optional, default=False
        If True and an error in bond determination occurs, the primary traceback is printed.

    Returns
    -------
    mol : rdkit.Chem.Mol
        New RDKit molecule from rdkit determination of bond orders. Note that atom
        properties may have been lost. Returns None if bond determination failed.
    """
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    if any(atm.GetNumImplicitHs() > 0 for atm in mol.GetAtoms()):
        raise ValueError("Provided molecule has implicit hydrogen atoms.")
    mol = Chem.RWMol(mol)
    try:
        DetermineBondOrders(
            mol, charge=charge, maxIterations=1000, allowChargedFragments=False
        )
    except Exception:
        if verbose:
            warnings.warn(first_traceback())
        warnings.warn("RDKit failed to determine molecular bond orders.")
        mol = None

    return mol


def determine_bonds_openbabel(mol, return_implicit_Hs=False, verbose=False):
    """Determine bond orders with Open Babel, or None if failed.

    Note that atom properties may have been lost and the default from openbabel is to
    process the molecule with implicit hydrogens only, so all hydrogens and their positions
    are deleted and then optionally restores with approximate coordinates.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule that needs to be updated
    return_implicit_Hs : bool, optional, default=False
        If False, the inherent implicit hydrogens from the openbabel process are added back with
        approximated coordinates.
    verbose : bool, optional, default=False
        If True and an error in bond determination occurs, the primary traceback is printed.

    Returns
    -------
    mol : rdkit.Chem.Mol
        New RDKit molecule from openbabel determination of bond orders. Note that atom
        properties may have been lost. Returns None if bond determination failed.
    """
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    if any(atm.GetNumImplicitHs() > 0 for atm in mol.GetAtoms()):
        raise ValueError("Provided molecule has implicit hydrogen atoms.")
    mol = Chem.RWMol(mol)
    mol.UpdatePropertyCache(strict=False)
    try:
        mol_no_H = Chem.RemoveHs(mol)  # PerceiveBondOrders requires implicit Hs
    except Chem.rdchem.AtomValenceException:
        if verbose:
            warnings.warn(first_traceback())
        warnings.warn(
            "RDKit sanitize failed to remove hydrogens in preparing for OpenBabel."
        )
        mol_openbabel = None
    else:
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("mol", "mol")
        obMol = ob.OBMol()
        obConversion.ReadString(obMol, Chem.MolToMolBlock(mol_no_H, forceV3000=True))

        obMol.PerceiveBondOrders()
        aromatyper = ob.OBAromaticTyper()
        aromatyper.AssignAromaticFlags(obMol)
        obMol.PerceiveBondOrders()
        mol_block_processed = obConversion.WriteString(obMol)
        # sanitize=False needed for 5 member rings in porphyrins
        mol_openbabel = Chem.MolFromMolBlock(mol_block_processed, sanitize=False)

    if mol_openbabel is None:
        warnings.warn("Openbabel failed to determine molecular bond orders.")
    else:
        if not return_implicit_Hs:
            mol_openbabel = Chem.AddHs(
                mol_openbabel, addCoords=True, explicitOnly=False
            )
        mol_openbabel.UpdatePropertyCache(strict=False)

    return mol_openbabel


def update_atom_bond_props(mol_to_change, mol_reference, sanitize=True):
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

    if sanitize:
        try:
            Chem.SanitizeMol(
                mol_to_change,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
            )
        except Chem.rdchem.AtomValenceException:
            warnings.warn("RDKit sanitize failed, passing molecule as is.")

    return mol_to_change


def add_obvious_bonds(mol):
    """Correct bond order for adjacent atoms with hanging bonds.

    Sometimes after the determining bond order analysis (particularly for openbabel)
    there are adjacent atoms, each with a hanging bond. The success of this package
    is improved significantly by increasing those bond orders by one.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to update
    """
    mol.UpdatePropertyCache(strict=False)
    _, _, charged_atoms = assess_atoms(mol)
    if charged_atoms:
        atoms_hanging_bonds = [
            index for index, tmp in charged_atoms.items() if tmp["charge"] < 0
        ]
        pairs = list(itertools.combinations(atoms_hanging_bonds, 2))
        for idx1, idx2 in pairs:
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond is not None:
                bo = bond_order_dict[bond.GetBondType().name]
                atom1 = mol.GetAtomWithIdx(idx1)
                chg1 = get_atom_charge(atom1)
                atom2 = mol.GetAtomWithIdx(idx2)
                chg2 = get_atom_charge(atom2)
                if bo == 1.5 or chg1 != chg2:
                    continue
                if bo - chg1 not in bond_type_dict:
                    raise ValueError(
                        f"Bond order change cannot be rectified. Orig Bond Order: {bo}, Atom Charge {chg1}"
                    )
                bond.SetBondType(bond_type_dict[bo - chg1])
                atom1.SetFormalCharge(0)
                atom2.SetFormalCharge(0)

        mol.UpdatePropertyCache(strict=False)
