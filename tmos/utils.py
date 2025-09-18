"""General utilities"""

import traceback
from collections import defaultdict
import json
import re

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


def view3D(molecule, label_idx=False, label_symbol=False, kekulize=True):
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
            label = label_func(i, atom)
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
