"""General utilities"""

import traceback
from collections import defaultdict


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
