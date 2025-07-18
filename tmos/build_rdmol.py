"""Functions for building RDKit molecules.

Such functions include:
 - Utilities to import molecules from one package to RDKit
 - Determine the connectivity of the molecule from XYZ coordinates using RDKit or OpenBabel
 - Determining the bond orders of a molecule using RDKit, OpenBabel, or MDAnalysis

"""

import copy
import warnings

from qcelemental.physical_constants import constants
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdDetermineBonds import DetermineBondOrders

import periodictable
from MDAnalysis.converters.RDKit import _infer_bo_and_charges

try:
    from openbabel import openbabel as ob

    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    warnings.warn(
        "OpenBabel not available. Will use RDKit-only connectivity detection."
    )

from .utils import first_traceback, get_molecular_formula
from .graph_mapping import find_atom_mapping, implicit_hydrogen_atom_mapping

transition_metal_covalent_radii = {
    # Transition metals (first row)
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    # Transition metals (second row)
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    # Transition metals (third row)
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
}


#############################################################################
########################### Conversion Functions ############################
#############################################################################


def qcelemental_to_rdkit(qcel_molecule, use_connectivity=True):
    """
    Convert a QCElemental molecule to an RDKit molecule.

    Parameters:
    -----------
    qcel_molecule : qcelemental.models.Molecule
        The QCElemental molecule object
    use_connectivity : bool
        Whether to use existing connectivity information if available

    Returns:
    --------
    rdkit.Chem.Mol
        RDKit molecule object
    """

    mol = Chem.RWMol()
    for i, symbol in enumerate(qcel_molecule.symbols):
        atom = Chem.Atom(symbol)
        mol.AddAtom(atom)

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
            # Convert bond order to RDKit bond type
            if bond_order == 1:
                bond_type = Chem.BondType.SINGLE
            elif bond_order == 2:
                bond_type = Chem.BondType.DOUBLE
            elif bond_order == 3:
                bond_type = Chem.BondType.TRIPLE
            elif bond_order == 1.5:
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = Chem.BondType.SINGLE

            mol.AddBond(atom1_idx, atom2_idx, bond_type)

    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
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
        atom_i = mol.GetAtomWithIdx(i)
        symbol_i = atom_i.GetSymbol()
        pos_i = conformer.GetAtomPosition(i)

        for j in range(i + 1, mol.GetNumAtoms()):
            atom_j = mol.GetAtomWithIdx(j)
            symbol_j = atom_j.GetSymbol()
            pos_j = conformer.GetAtomPosition(j)

            # Calculate distance
            distance = pos_i.Distance(pos_j)

            # Get covalent radii using periodictable
            radius_i = _get_covalent_radius(symbol_i)
            radius_j = _get_covalent_radius(symbol_j)

            # Calculate bond distance threshold
            bond_threshold = radius_i + radius_j + distance_tolerance

            # Special handling for metal-ligand bonds (more generous distance)
            if _is_transition_metal(symbol_i) or _is_transition_metal(symbol_j):
                bond_threshold *= 1.3  # Increase threshold for metal complexes

            # Add bond if atoms are close enough
            if distance < bond_threshold:
                try:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                except Exception:
                    # Skip if bond already exists or causes issues
                    pass

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


def determine_bonds_hybrid(mol, charge=0, verbose=False):
    """Determine bond orders with OpenBabel, with MDA fallback, and RDKit as another fallback."""
    # TODO: Interleave assessment between the methods
    new_mol = determine_bonds_openbabel(mol)
    if new_mol is None:
        new_mol = determine_bonds_mda(mol)
        if new_mol is None:
            new_mol = determine_bonds_rdkit(mol, charge=charge, verbose=verbose)
    return new_mol


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
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    mol = Chem.RWMol(Chem.AddHs(mol, addCoords=True, explicitOnly=True))
    _infer_bo_and_charges(mol)
    return mol


def determine_bonds_rdkit(mol, charge=0, verbose=False):
    """Determine bond orders with RDKit, or None if failed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule that needs to be updated

    Returns
    -------
    mol : rdkit.Chem.Mol
        New RDKit molecule from rdkit determination of bond orders. Note that atom
        properties may have been lost. Returns None if bond determination failed.
    """
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    mol = Chem.RWMol(Chem.AddHs(mol, addCoords=True, explicitOnly=True))
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


def determine_bonds_openbabel(mol, return_implicit_Hs=False):
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
    mol = copy.deepcopy(mol)
    mol.UpdatePropertyCache(strict=False)
    mol = Chem.RWMol(Chem.AddHs(mol, addCoords=True, explicitOnly=True))
    try:
        mol_no_H = Chem.RemoveHs(mol)  # PerceiveBondOrders requires implicit Hs
    except Chem.rdchem.AtomValenceException:
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
        mol_openbabel = Chem.MolFromMolBlock(mol_block_processed, sanitize=True)

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
                mol_to_change
            )  # , sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        except Chem.rdchem.AtomValenceException:
            warnings.warn("RDKit sanitize failed, passing molecule as is.")

    return mol_to_change
