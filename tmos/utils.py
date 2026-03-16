"""General utilities"""

import os
import sys
import traceback
from collections import defaultdict
import json
import re
from contextlib import contextmanager

import numpy as np
import py3Dmol
from loguru import logger

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole

IPythonConsole.molSize = 500, 500


def configure_logger(level="INFO"):
    """Configure the loguru logger with a specified level.

    This function should be called once at the start of the application to set up
    logging configuration. It removes the default handler and adds a new one with
    the specified level. Can also be called later to change the logging level.

    Parameters
    ----------
    level : str, optional
        Logging level to use. Valid options are: "TRACE", "DEBUG", "INFO",
        "SUCCESS", "WARNING", "ERROR", "CRITICAL". Default is "INFO".

    Examples
    --------
    # Set to INFO level (default) - typically called automatically on import
    configure_logger()

    # Change to DEBUG level for more verbose output
    configure_logger("DEBUG")

    # Change to WARNING level for less verbose output
    configure_logger("WARNING")
    """
    # Remove default handler
    logger.remove()

    # Add new handler with specified level
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def save_to_json(result, filename, indent=4):
    """Save result dictionary to a json file, with RDKit Molecules removed

    Parameters
    ----------
    result : dict
        Dictionary output of functions such as :func:`sanitize_complex`
    filename : str
        filename of output file
    indent : int, optional, default=4
        Indentation of json file
    """

    def remove_rdmol(obj):
        if isinstance(obj, dict):
            return {k: remove_rdmol(v) for k, v in obj.items() if k != "rdmol"}
        elif isinstance(obj, list):
            return [remove_rdmol(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        else:
            return obj

    with open(filename, "w") as f:
        json.dump(remove_rdmol(result), f, indent=indent)


def first_traceback(keyword="During handling of the above exception"):
    """Isolate the first error in the traceback that caused the issue.

    Parameters
    ----------
    keyword : str, optional
        Keyword string used to identify the cutoff point of the traceback,
        by default "During handling of the above exception"

    Returns
    -------
    str
        Isolated traceback string
    """
    error_msg = traceback.format_exc().split("\n")
    ind = [i for i, x in enumerate(error_msg) if keyword in x]
    ind = ind[0] if len(ind) > 0 else len(error_msg)
    return "\n".join(error_msg[:ind])


def get_molecular_formula(mol, make_hydrogens_implicit=False):
    """Get molecular formula with elements sorted in alphabetical order.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule
    make_hydrogens_implicit : bool, optional, default=True
        If True, hydrogens will be removed from the molecular formula

    Returns
    -------
    str
        Molecular formula
    """
    comp = defaultdict(lambda: 0)
    for atom in mol.GetAtoms():
        element_symbol = atom.GetSymbol()
        comp[element_symbol] += 1
    comp = dict(sorted(comp.items()))
    if make_hydrogens_implicit and "H" in comp:
        del comp["H"]
    formula = "".join([str(x) for k, val in comp.items() for x in [k, val]])
    return formula


def molecular_formula_to_dict(formula):
    """Get molecular formula dictionary with elements sorted in alphabetical order.

    Parameters
    ----------
    formula : str
        Molecular formula, e.g., C1H4

    Returns
    -------
    dict
        Dictionary with keys as elements and values as the number of elements in the molecule.
    """

    pattern = r"([A-Z][a-z]*)(\d+)"
    matches = re.findall(pattern, formula)
    formula_dict = {element: int(count) for element, count in matches}

    return formula_dict


def view3D(molecule, label_idx=False, label_symbol=False, kekulize=True, indices=[]):
    """
    Format 3D view of an RDKit molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        RDKit molecule to be viewed.
    label_idx : bool, optional
        If True, atom indices will be displayed as labels. Default is False.
    label_symbol : bool, optional
        If True, atom symbol will be displayed as labels. Default is False.
    kekulize : bool, optional
        If True, kekulize the molecule before rendering. Default is True.
    indices : list, optional
        If indices are provided, only those atoms are labeled

    Returns
    -------
    py3Dmol.view
        3Dmol.js view object for visualization.
    """
    if molecule is None:
        return None
    mol = Chem.Mol(molecule)
    view = py3Dmol.view(
        data=Chem.MolToMolBlock(
            mol, kekulize=kekulize
        ),  # Convert the RDKit molecule for py3Dmol
        style={"stick": {}, "sphere": {"scale": 0.3}},
    )

    def label_func(i, atom):
        if label_symbol and label_idx:
            return f"{i}: {atom.GetSymbol()}"
        elif label_idx:
            return str(i)
        elif label_symbol:
            return atom.GetSymbol()
        else:
            return None

    if label_func:
        for i, atom in enumerate(mol.GetAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            label = (
                label_func(i, atom)
                if (indices and i in indices) or not indices
                else None
            )
            if label:
                view.addLabel(
                    label,
                    {
                        "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                        "backgroundColor": "white",
                        "backgroundOpacity": 0.2,
                        "fontColor": "black",
                    },
                )

    return view


def diagnose_atom_valence(atom, show=True):
    """
    Inspect an RDKit Atom for valence/charge/bonding inconsistencies useful for debugging.
    Args:
        atom: rdkit.Chem.rdchem.Atom
        show: if True, print a short human-readable summary (function still returns the dict)
    Returns:
        dict with detailed diagnostics
    """
    # basic properties
    idx = atom.GetIdx()
    Z = atom.GetAtomicNum()
    symbol = atom.GetSymbol()
    formal_charge = atom.GetFormalCharge()
    degree = atom.GetDegree()  # number of directly bonded neighbors
    num_explicit_h = atom.GetNumExplicitHs()
    num_implicit_h = atom.GetNumImplicitHs()
    total_h = atom.GetTotalNumHs()
    num_radical_e = atom.GetNumRadicalElectrons()
    hybrid = (
        atom.GetHybridization().name
        if hasattr(atom.GetHybridization(), "name")
        else str(atom.GetHybridization())
    )
    is_aromatic = atom.GetIsAromatic()
    is_in_ring = atom.IsInRing()
    mass = atom.GetMass()
    rdkit_total_valence = atom.GetTotalValence()

    # neighbor / bond info
    bonds = list(atom.GetBonds())
    neighbor_info = []
    sum_bond_orders = 0.0
    for b in bonds:
        nb_idx = b.GetOtherAtomIdx(idx)
        nb = b.GetOtherAtom(atom)
        btype = str(b.GetBondType())
        try:
            bord = float(b.GetBondTypeAsDouble())
        except Exception:
            # fallback for aromatic etc.
            bord = 1.5 if b.GetIsAromatic() else 1.0
        sum_bond_orders += bord
        neighbor_info.append(
            {
                "nbr_idx": nb_idx,
                "nbr_symbol": nb.GetSymbol(),
                "bond_type": btype,
                "bond_order": bord,
                "is_aromatic": b.GetIsAromatic(),
                "is_in_ring": b.IsInRing(),
            }
        )

    # computed valence from explicit bonding + Hs + radical electrons
    computed_valence = sum_bond_orders + total_h + num_radical_e

    # naive typical-valence lookup for common elements (fallback=None)
    typical_valence_map = {
        "H": 1,
        "C": 4,
        "N": 3,
        "O": 2,
        "F": 1,
        "P": 3,
        "S": 2,
        "Cl": 1,
        "Br": 1,
        "I": 1,
        "B": 3,
    }
    typical_valence = typical_valence_map.get(symbol)

    # diagnostics / heuristics
    messages = []
    if abs(computed_valence - rdkit_total_valence) > 1e-6:
        messages.append(
            f"RDKit reported total_valence={rdkit_total_valence:.2f} but sum(bond_orders)+Hs+radicals={computed_valence:.2f}."
        )
    if formal_charge != 0 and typical_valence is not None:
        # allow for common charged states by shifting typical valence expectation by charge
        if abs((typical_valence - formal_charge) - computed_valence) > 0.5:
            messages.append(
                f"Formal charge {formal_charge:+d} shifts expected valence; computed ({computed_valence:.2f}) differs "
                f"from typical({typical_valence})±charge."
            )
    if typical_valence is not None and computed_valence > typical_valence + 0.5:
        messages.append(
            f"Possible hypervalence: computed valence {computed_valence:.2f} > typical valence {typical_valence}."
        )
    if total_h == 0 and symbol == "H":
        messages.append(
            "Hydrogen with zero H count (impossible) — likely indexing/explicit-H mismatch."
        )
    if (
        is_aromatic
        and any(bd["bond_type"].upper() == "DOUBLE" for bd in neighbor_info)
        and symbol in ("C", "N", "O")
    ):
        messages.append(
            "Aromatic atom but contains explicit double bonds in neighbors — check kekulization/state."
        )
    if not messages:
        messages.append("No obvious valence problems detected by heuristics.")

    result = {
        "idx": idx,
        "atomic_num": Z,
        "symbol": symbol,
        "formal_charge": formal_charge,
        "degree": degree,
        "num_explicit_h": num_explicit_h,
        "num_implicit_h": num_implicit_h,
        "total_h": total_h,
        "num_radical_electrons": num_radical_e,
        "hybridization": hybrid,
        "is_aromatic": is_aromatic,
        "is_in_ring": is_in_ring,
        "mass": mass,
        "rdkit_total_valence": rdkit_total_valence,
        "sum_bond_orders": sum_bond_orders,
        "computed_valence": computed_valence,
        "typical_valence": typical_valence,
        "neighbors": neighbor_info,
        "messages": messages,
    }

    if show:
        summary = (
            f"{symbol}{idx} Z={Z} charge={formal_charge:+d} deg={degree} "
            f"bonds={len(bonds)} bond_order_sum={sum_bond_orders:.2f} Hs={total_h} "
            f"rdkit_valence={rdkit_total_valence:.2f} computed_valence={computed_valence:.2f}"
        )
        print(summary)
        for m in messages:
            print(" -", m)
        # compact neighbor table
        print(
            " neighbors:",
            ", ".join(
                f"{n['nbr_symbol']}[{n['nbr_idx']}]:{n['bond_type']}({n['bond_order']})"
                for n in neighbor_info
            )
            or " none",
        )

    return result
