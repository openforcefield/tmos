"""Functions for building RDKit molecules.

Such functions include:
 - Utilities to import molecules from one package to RDKit
 - Determine the connectivity of the molecule from XYZ coordinates using RDKit or OpenBabel
 - Determining the bond orders of a molecule using RDKit, OpenBabel, or MDAnalysis

"""

import copy
from loguru import logger
import os
from typing import Protocol, TypeAlias, TypedDict

import numpy as np

from qcelemental.physical_constants import constants
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import GetPeriodicTable
from rdkit import RDLogger
from rdkit.Chem.rdchem import Atom, Mol

import periodictable
from openbabel import openbabel as ob

from .utils import get_molecular_formula
from .graph_mapping import (
    find_atom_mapping,
    implicit_hydrogen_atom_mapping,
)
from .reference_values import (
    bond_order_dict,
    bond_type_dict,
    transition_metal_covalent_radii,
    METALS_NUM,
)
from tmos._rdkit_bond_typing import determine_bonds, molecule_charge_penalty

__all__: list[str] = [
    "determine_bonds",
    "molecule_charge_penalty",
    "assess_atoms",
    "get_atom_charge",
    "update_formal_charges",
    "copy_atom_coords",
    "qcelemental_to_rdkit",
    "xyz_to_rdkit",
    "determine_connectivity",
    "update_atom_bond_props",
]

# Suppress all OpenBabel output including stereochemistry errors
ob.obErrorLog.SetOutputLevel(0)
ob.obErrorLog.StopLogging()
os.environ["BABEL_SILENCE"] = "1"

# Disable all RDKit logging
RDLogger.DisableLog("rdApp.*")
pt = GetPeriodicTable()

BondOrderTuple: TypeAlias = tuple[str, int, str, int, str, int]


class ChargedAtomInfo(TypedDict):
    """Per-atom charge diagnostics produced by `assess_atoms`."""

    symbol: str
    charge: int | float
    bond_orders: list[BondOrderTuple]


class QCEleMoleculeLike(Protocol):
    """Structural typing interface for QCElemental-like molecules."""

    symbols: list[str]
    geometry: np.ndarray | None
    connectivity: list[tuple[int, int, int | float]] | None


#############################################################################
############################## Atom Assessment ##############################
#############################################################################


def get_atom_charge(
    atom: Atom,
    ignore_multiple_charges: bool = True,
    use_formal_charge: bool = False,
) -> int | float:
    """Estimate atom charge from valence and bond-order balance.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom to assess.
    ignore_multiple_charges : bool, default=True
        If False, raises an error when multiple possible charges are encountered;
        otherwise, selects the lowest charge. This is relevant for atoms like sulfur
        that can have multiple charge states.
    use_formal_charge : bool, default=False
        Whether to use the atomic formal charge. If True, the output is the number
        of hanging bonds (negative for hanging, positive for excess). Either may
        indicate that the formal charge of the atom should be updated.

    Returns
    -------
    int or float
        Charge-like balance value. May be fractional for aromatic contexts.

    Raises
    ------
    ValueError
        If the atom could have multiple charge states given the bonding orders and
        `ignore_multiple_charges` is False.

    Notes
    -----
    If multiple possible charges are found and ``ignore_multiple_charges`` is ``True``,
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


def assess_atoms(
    mol: Mol,
    add_atom: str | None = None,
    use_formal_charge: bool = False,
) -> tuple[int | float, int, dict[int, ChargedAtomInfo]]:
    """Assess atom-wise charge state and unsatisfied valence indicators.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule.
    add_atom : str or None, default=None
        If provided, the number of hanging bonds will be the number of this atom type present.
    use_formal_charge : bool, default=False
        Whether to use the atomic formal charge in :func:`get_atom_charge`.

    Returns
    -------
    charge : int or float
        Total charge of the molecule.
    hanging_bonds : int
        Number of hanging bonds to contribute toward the oxidation state of the metal.
    charged_atoms : dict of int to ChargedAtomInfo
        Per-atom diagnostics for non-zero charge sites.

        Each value includes atom symbol, assigned charge, and local bond-order
        tuples for debugging/reporting.
    """
    if mol is None:
        raise ValueError("Molecule is None")

    mol = copy.deepcopy(mol)

    charged_atoms: dict[int, ChargedAtomInfo] = {}
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
        charge = get_atom_charge(a, use_formal_charge=use_formal_charge)
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


def update_formal_charges(mol: Mol) -> None:
    """Update the formal charges of a molecule to align with their implied charge from connectivity.

    Note that formal charges of metals are not updated.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to update in place.

    Returns
    -------
    None
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


def copy_atom_coords(
    mol1: Mol,
    index1: int,
    mol2: Mol,
    index2: int,
    confId1: int = 0,
    confId2: int = 0,
) -> None:
    """Copy atom coordinates between molecules.

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
    confId1 : int, default=0
        Target conformer index in ``mol1``.
    confId2 : int, default=0
        Reference conformer index in ``mol2``.

    Returns
    -------
    None
    """
    conf1 = mol1.GetConformer(confId1)
    conf2 = mol2.GetConformer(confId2)

    pos = conf2.GetAtomPosition(index2)  # returns Point3D
    conf1.SetAtomPosition(index1, Point3D(pos.x, pos.y, pos.z))


#############################################################################
########################### Conversion Functions ############################
#############################################################################


def qcelemental_to_rdkit(
    qcel_molecule: QCEleMoleculeLike,
    use_connectivity: bool = True,
    distance_tolerance: float = 0.2,
) -> Mol:
    """Convert a QCElemental molecule to an RDKit molecule.

    Parameters
    ----------
    qcel_molecule : qcelemental.models.Molecule
        QCElemental molecule object.
    use_connectivity : bool, default=True
        Whether to use provided connectivity when available.
    distance_tolerance : float, default=0.2
        Additional tolerance (Å) for bond distance cutoffs in
        :func:`determine_connectivity`.

    Returns
    -------
    rdkit.Chem.Mol
        RDKit molecule object.

    Examples
    --------
    >>> # mol = qcelemental_to_rdkit(qcel_molecule)
    >>> # isinstance(mol, Mol)
    >>> # True
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
    symbols: list[str],
    positions: np.ndarray,
    distance_tolerance: float = 0.20,
    method: str = "custom",
    ignore_scale: bool = False,
) -> Mol:
    """Build an RDKit molecule from symbols and Cartesian coordinates.

    If the resulting molecule cannot be sanitized, all bond orders will be single. For more
    in depth sanitization of bond order assignment, run the molecule through ``determine_bonds``.

    Parameters
    ----------
    symbols : list[str]
        Element symbols.
    positions : numpy.ndarray
        Matrix matching ``symbols`` with x, y, z coordinates in Å.
    distance_tolerance : float, default=0.20
        Additional tolerance (Å) above the covalent-radius sum for bond
        detection in :func:`determine_connectivity`.
    method : str, default="custom"
        Connectivity method passed to :func:`determine_connectivity`.
        ``"custom"`` uses the covalent-radius heuristic directly.
        ``"rdkit"`` and ``"openbabel"`` are also accepted.
    ignore_scale : bool, default=False
        If True avoid an error when "H" is present and the minimum atomic distance is not between
        0.8 Å and 1.5 Å.

    Returns
    -------
    rdkit.Chem.Mol
        RDKit molecule object.

    Examples
    --------
    >>> symbols = ["O", "H", "H"]
    >>> positions = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.3, 0.9, 0.0]])
    >>> mol = xyz_to_rdkit(symbols, positions, ignore_scale=True)
    >>> mol.GetNumAtoms()
    3
    """
    positions = np.array(positions)
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
        atom.SetNumExplicitHs(0)
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


def determine_connectivity(
    rdkit_mol: Mol,
    method: str = "custom",
    distance_tolerance: float = 0.2,
) -> Mol:
    """Determine molecular connectivity.

    Parameters
    ----------
    rdkit_mol : rdkit.Chem.Mol
        RDKit molecule without bonds.
    method : str, default="custom"
        Method to use:

        - ``"custom"`` — standalone covalent-radius heuristic with valence
          guards (see :func:`_determine_connectivity_custom`).  Default.
        - ``"rdkit"`` — RDKit's built-in coordinate-based perception.
        - ``"openbabel"`` — OpenBabel's ``ConnectTheDots`` perception.

    distance_tolerance : float, default=0.2
        Additional tolerance for bond distance cutoffs (Å).

    Returns
    -------
    rdkit.Chem.Mol
        RDKit molecule with bonds added.

    Raises
    ------
    ValueError
        If ``method`` is unsupported.

    Examples
    --------
    >>> # connected = determine_connectivity(mol, method="rdkit")
    >>> # connected.GetNumBonds() >= 0
    >>> # True
    """
    if method == "openbabel":
        return _determine_connectivity_openbabel(rdkit_mol)
    elif method == "rdkit":
        return _determine_connectivity_rdkit(rdkit_mol, distance_tolerance)
    elif method == "custom":
        return _determine_connectivity_custom(
            rdkit_mol, max_distance_tolerance=distance_tolerance
        )
    else:
        raise ValueError(
            f"Connectivity method, {method}, is not supported. Must be: "
            "'rdkit', 'openbabel', or 'custom'"
        )


def _determine_connectivity_openbabel(rdkit_mol: Mol) -> Mol:
    """Use OpenBabel to assign atom connectivity.

    Parameters
    ----------
    rdkit_mol : rdkit.Chem.Mol
        Molecule without bonds.

    Returns
    -------
    rdkit.Chem.Mol
        Molecule with inferred connectivity.
    """

    ob_mol = ob.OBMol()
    ob_conv = ob.OBConversion()
    ob_conv.SetInAndOutFormats("mol", "mol")
    mol_block = Chem.MolToMolBlock(rdkit_mol)
    ob_conv.ReadString(ob_mol, mol_block)

    ob_mol.ConnectTheDots()
    mol_block_with_bonds = ob_conv.WriteString(ob_mol)
    mol_with_bonds = Chem.MolFromMolBlock(mol_block_with_bonds, sanitize=False)

    return mol_with_bonds


def _get_covalent_radius(symbol: str, fallback_radius: float = 1.5) -> float:
    """Return covalent radius for an element symbol.

    Parameters
    ----------
    symbol : str
        Element symbol.
    fallback_radius : float, default=1.5
        Radius used when lookup fails.

    Returns
    -------
    float
        Covalent radius in Å.
    """
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


def _is_transition_metal(symbol: str) -> bool:
    """Check whether an element is treated as a transition metal.

    Parameters
    ----------
    symbol : str
        Element symbol.

    Returns
    -------
    bool
        ``True`` when symbol is in transition-metal radius table.
    """

    return symbol in transition_metal_covalent_radii


# Atomic numbers that commonly carry +1 formal charge in organic molecules.
# These elements are allowed one bond beyond their neutral valence maximum
# during connectivity detection (which runs before charge assignment).
_CATIONIC_ATOMIC_NUMS: frozenset[int] = frozenset(
    {
        7,  # N  — ammonium, iminium  (neutral max 3 → allow 4)
        15,  # P  — phosphonium         (neutral max 5 → allow 6)
        16,  # S  — sulfonium           (neutral max 6 → allow 7)
    }
)


def _correct_sulfonate_phosphate_interaction(
    mol: Mol,
    distances: np.ndarray,
    r_cut: float = 3.0,
    dist_frac: float = 0.05,
) -> list[int]:
    """Return candidate atom indices to block from metal bonding.

    The heuristic blocks atoms when neighboring electronegative atoms are
    substantially closer to the metal center.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule to evaluate.
    distances : numpy.ndarray
        Pairwise distance matrix.
    r_cut : float, default=3.0
        Distance cutoff (Å) for local metal-neighbor consideration.
    dist_frac : float, default=0.05
        Relative closeness threshold for blocking a candidate atom.

    Returns
    -------
    list of int
        Atom indices that should not bond to the metal center.
    """

    metal_indices = [
        a.GetIdx() for a in mol.GetAtoms() if _is_transition_metal(a.GetSymbol())
    ]
    electronegative_sym: list[str] = ["O", "N", "C"]
    central_sym: list[str] = [
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

        if (
            len(d_connections) > 0
            and np.max((d_center - d_connections) / d_center) > dist_frac
        ):
            block_bond_idx.append(atm.GetIdx())

    return block_bond_idx


def _max_valence_for_connectivity(
    atomic_num: int,
    symbol: str,
) -> int:
    """Return the maximum allowed bond count for an atom during connectivity assignment.

    Transition metals are exempt (returns a large sentinel) so that their
    coordination number is unrestricted by this heuristic.

    Parameters
    ----------
    atomic_num : int
        RDKit atomic number.
    symbol : str
        Element symbol.

    Returns
    -------
    int
        Maximum number of bonds the atom may form during connectivity detection.
    """
    if _is_transition_metal(symbol):
        return 14  # effectively uncapped for TMs
    vlist = [v for v in pt.GetValenceList(atomic_num) if v != -1]
    base = max(vlist) if vlist else 12

    # Allow one extra bond for elements that commonly carry +1 formal charge.
    # Connectivity is assigned before bond typing / charge assignment, so we
    # must be permissive enough to admit bonds that are valid under a +1 fc.
    return base + 1 if atomic_num in _CATIONIC_ATOMIC_NUMS else base


def _is_valence_satisfied(degree: int, atomic_num: int, symbol: str) -> bool:
    """Return True when *degree* matches a valid standard valence for the atom.

    Used as a soft connectivity guard: if both endpoints of a candidate bond
    are already at a satisfied standard valence, the bond is skipped (except
    for transition metals, which are always considered unsatisfied).

    Parameters
    ----------
    degree : int
        Current number of bonds on the atom.
    atomic_num : int
        RDKit atomic number.
    symbol : str
        Element symbol.

    Returns
    -------
    bool
        ``True`` when degree is already a valid closed-shell valence count.
    """
    if _is_transition_metal(symbol):
        return False
    vlist = [v for v in pt.GetValenceList(atomic_num) if v != -1]
    return degree in vlist


def _determine_connectivity_rdkit(mol: Mol, distance_tolerance: float = 0.2) -> Mol:
    """Assign connectivity using RDKit native coordinate perception.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input RDKit molecule.
    distance_tolerance : float, default=0.2
        Present for interface consistency; currently unused by RDKit backend.

    Returns
    -------
    rdkit.Chem.Mol
        Molecule with inferred connectivity.
    """
    Chem.rdDetermineBonds.DetermineConnectivity(mol)
    return mol


def _determine_connectivity_custom(
    rdkit_mol: Mol,
    max_distance_tolerance: float = 0.2,
    min_distance_tolerance: float = 0.4,
) -> Mol:
    """Assign or refine connectivity using covalent-radius distance heuristics.

    Can be used as a standalone connectivity method (when the input molecule
    has no bonds) or as a post-processing step after RDKit/OpenBabel detection.
    When used standalone, the full bond graph is built from scratch using
    covalent radii and the valence rules below.  When used as post-processing,
    existing bonds seed the degree counters so the method augments/prunes the
    prior graph rather than replacing it.

    It applies special rules for transition metals by increasing the bonding
    threshold to better account for metal complexes.

    Parameters
    ----------
    rdkit_mol : rdkit.Chem.Mol
        The input RDKit molecule, which must have at least one conformer with 3D coordinates.
    max_distance_tolerance : float, default=0.2
        Maximum extra distance (Å) above radius sum for bond formation.
    min_distance_tolerance : float, default=0.4
        Minimum distance (Å) below which atoms are considered too close.

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
        - Existing bonds from upstream methods (RDKit/OpenBabel) are preserved as
            the starting graph and count toward valence limits.
    - Transition metals are handled with a larger bonding threshold (scaled by 1.3)
      and are exempt from the valence-based pruning rules below.
    - **Rule 1 — hard cap**: a bond is rejected if it would bring either endpoint's
      degree above its context-aware maximum valence.  For sulfur the cap is
      dynamic: without any O neighbors S is limited to valence 2 (thioether/thiol);
      each O neighbor unlocks the next hypervalent shell (1 O → 4, ≥2 O → 6).
      Other elements use the static maximum from their valence list (e.g. O → 2,
      H → 1).
    - **Rule 2 — soft satisfied skip**: a bond is rejected when *both* endpoints
      already sit at a valid standard valence (i.e. their running degree is already
      in ``GetValenceList``).  This prevents spurious homoatomic bonds such as P-P
      in phosphate cages where each P has three O-bonds (degree 3 ∈ {3, 5}) and
      would otherwise accept further bonds up to its maximum of 5.
    """

    mol = Chem.RWMol(rdkit_mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("No conformers are available for this RDKit molecule.")
    else:
        conformer = mol.GetConformer()

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    radii = np.array([_get_covalent_radius(sym) for sym in symbols])
    positions = np.array(
        [conformer.GetAtomPosition(i) for i in range(len(radii))], dtype=float
    )
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    metal_indices = [
        a.GetIdx() for a in mol.GetAtoms() if _is_transition_metal(a.GetSymbol())
    ]
    metal_idx = metal_indices[0] if metal_indices else None

    max_valences = [
        _max_valence_for_connectivity(an, sym) for an, sym in zip(atomic_nums, symbols)
    ]
    degrees: list[int] = [0] * mol.GetNumAtoms()

    # Collect all candidate (i, j) pairs that fall within bonding distance,
    # then process them in ascending distance order.  This ensures that when
    # an atom's valence cap is reached by Rule 1 the *closest* (most reliably
    # real) bonds are retained while longer spurious connections are dropped —
    # directly implementing the valence-based "fewer or equal connections"
    # principle without relying on atom-index traversal order.
    n = mol.GetNumAtoms()
    candidate_pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            one_is_tm: bool = _is_transition_metal(symbols[i]) or _is_transition_metal(
                symbols[j]
            )
            factor: int = 2 if one_is_tm else 1
            max_bond_threshold = radii[i] + radii[j] + max_distance_tolerance * factor
            min_bond_threshold = radii[i] + radii[j] - min_distance_tolerance * factor
            d = distances[i, j]
            if min_bond_threshold < d < max_bond_threshold:
                candidate_pairs.append((d, i, j))
    # Short bonds first: when an atom's degree cap is hit, closest neighbours
    # (which are almost always the genuine covalent bonds) are kept.
    candidate_pairs.sort()

    for d, i, j in candidate_pairs:
        if mol.GetBondBetweenAtoms(i, j) is not None:
            continue
        # Rule 1 — hard cap: neither atom may exceed its maximum valence.
        if degrees[i] >= max_valences[i] or degrees[j] >= max_valences[j]:
            continue

        # Rule 2 — soft satisfied skip: if *both* endpoints already sit at
        # a standard closed-shell valence, the bond is almost certainly
        # spurious (e.g. P-P in a phosphate cage where each P already has
        # three O bonds and degree=3 is valid for P).  Transition-metal
        # atoms are never considered satisfied so TM bonds are unaffected.
        if _is_valence_satisfied(
            degrees[i], atomic_nums[i], symbols[i]
        ) and _is_valence_satisfied(degrees[j], atomic_nums[j], symbols[j]):
            continue

        mol.AddBond(i, j, Chem.BondType.SINGLE)
        degrees[i] += 1
        degrees[j] += 1

    if metal_idx is not None:
        for idx in _correct_sulfonate_phosphate_interaction(mol, distances):
            if mol.GetBondBetweenAtoms(metal_idx, idx) is not None:
                mol.RemoveBond(metal_idx, idx)

    return mol.GetMol()


def update_atom_bond_props(mol_to_change: Mol, mol_reference: Mol) -> Mol:
    """Update atom and bond properties of one molecule to match another.

    Molecules must either be identical in their atomic composition and connectivity, or the reference may be
    the same as the other but with implicit hydrogens, in which case it's assumed that the indices are simply
    shifted.
    Atom properties include the formal charge, whether the atom is aromatic, and setting NoImplicit = True.
    Bond properties include the bond order.

    Parameters
    ----------
    mol_to_change : rdkit.Chem.Mol
        RDKit molecule that needs to be updated.
    mol_reference : rdkit.Chem.Mol
        RDKit molecule used as property reference.

    Returns
    -------
    rdkit.Chem.Mol
        Updated target molecule.

    Raises
    ------
    ValueError
        If the chemical formulas of the provided molecules indicate that they cannot be mapped.
    """

    formula_to_change: str = get_molecular_formula(mol_to_change)
    formula_to_change_no_H: str = get_molecular_formula(
        mol_to_change, make_hydrogens_implicit=True
    )
    formula_reference: str = get_molecular_formula(mol_reference)
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
