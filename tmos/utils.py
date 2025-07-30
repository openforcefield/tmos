"""General utilities"""

import traceback
from collections import defaultdict
import json

import numpy as np
import py3Dmol

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole

IPythonConsole.molSize = 500, 500


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


def view3D(molecule, labels=False, kekulize=True):
    """
    Format 3D view of an RDKit molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        RDKit molecule to be viewed.
    labels : bool, optional
        If True, atom indices will be displayed as labels. Default is False.
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

    # Add atom labels (indices)
    if labels:
        if mol.GetNumConformers() == 0:
            raise ValueError(
                "Cannot add 3D model labels without conformer coordinates."
            )
        for i, atom in enumerate(mol.GetAtoms()):
            pos = mol.GetConformer().GetAtomPosition(i)
            view.addLabel(
                str(i),
                {
                    "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.2,
                    "fontColor": "black",
                },
            )

    return view
