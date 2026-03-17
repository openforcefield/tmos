"""General utilities"""

import os
import sys
import traceback
from collections import defaultdict
from collections.abc import Generator
import json
import re
from contextlib import contextmanager
from typing import TypeAlias, TypedDict

import numpy as np
import py3Dmol
from loguru import logger

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdchem import Atom, Mol

IPythonConsole.molSize = 500, 500

DiagnosticValue: TypeAlias = (
    int | float | str | bool | None | list[str] | list["NeighborInfo"]
)


class NeighborInfo(TypedDict):
    """Schema for per-neighbor bond diagnostics."""

    nbr_idx: int
    nbr_symbol: str
    bond_type: str
    bond_order: float
    is_aromatic: bool
    is_in_ring: bool


def configure_logger(level: str = "INFO") -> None:
    """Configure the module logger output format and verbosity.

    Parameters
    ----------
    level : str, default="INFO"
        Logging level to use. Valid options are: "TRACE", "DEBUG", "INFO",
        "SUCCESS", "WARNING", "ERROR", "CRITICAL".

    Returns
    -------
    None

    Examples
    --------
    >>> configure_logger()
    >>> configure_logger("DEBUG")
    >>> configure_logger("WARNING")
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
def suppress_stdout_stderr() -> Generator[None, None, None]:
    """Temporarily suppress ``stdout`` and ``stderr``.

    Yields
    ------
    None
        Control to the wrapped context with output streams redirected.
    """
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


def save_to_json(result: dict[str, object], filename: str, indent: int = 4) -> None:
    """Save a result dictionary to JSON after removing RDKit molecules.

    Parameters
    ----------
    result : dict of str to object
        Dictionary output from processing functions.
    filename : str
        Output JSON path.
    indent : int, default=4
        JSON indentation level.

    Returns
    -------
    None
    """

    def remove_rdmol(obj: object) -> object:
        """Recursively remove ``rdmol`` keys and cast NumPy integers.

        Parameters
        ----------
        obj : object
            Nested object to sanitize.

        Returns
        -------
        object
            Sanitized object suitable for JSON encoding.
        """
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


def first_traceback(keyword: str = "During handling of the above exception") -> str:
    """Return the leading traceback segment for the current exception.

    Parameters
    ----------
    keyword : str, default="During handling of the above exception"
        Marker line used to truncate chained traceback output.

    Returns
    -------
    str
        Traceback text up to (but excluding) the first matching marker line.
    """
    error_msg: list[str] = traceback.format_exc().split("\n")
    indices: list[int] = [i for i, x in enumerate(error_msg) if keyword in x]
    cutoff = indices[0] if len(indices) > 0 else len(error_msg)
    return "\n".join(error_msg[:cutoff])


def get_molecular_formula(mol: Mol, make_hydrogens_implicit: bool = False) -> str:
    """Build an alphabetically ordered molecular formula string.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule.
    make_hydrogens_implicit : bool, default=False
        If ``True``, omit hydrogen counts from the output formula.

    Returns
    -------
    str
        Molecular formula as concatenated ``ElementCount`` tokens.
    """
    comp: defaultdict[str, int] = defaultdict(lambda: 0)
    for atom in mol.GetAtoms():
        element_symbol = atom.GetSymbol()
        comp[element_symbol] += 1
    comp_dict: dict[str, int] = dict(sorted(comp.items()))
    if make_hydrogens_implicit and "H" in comp_dict:
        del comp_dict["H"]
    formula: str = "".join([str(x) for k, val in comp_dict.items() for x in [k, val]])
    return formula


def molecular_formula_to_dict(formula: str) -> dict[str, int]:
    """Parse a compact molecular formula string into a dictionary.

    Parameters
    ----------
    formula : str
        Molecular formula string, for example ``"C1H4"``.

    Returns
    -------
    dict of str to int
        Mapping from element symbol to atom count.
    """

    pattern = r"([A-Z][a-z]*)(\d+)"
    matches: list[tuple[str, str]] = re.findall(pattern, formula)
    formula_dict: dict[str, int] = {element: int(count) for element, count in matches}

    return formula_dict


def view3D(
    molecule: Mol | None,
    label_idx: bool = False,
    label_symbol: bool = False,
    kekulize: bool = True,
    indices: list[int] | None = None,
) -> None | py3Dmol.view:
    """Create a `py3Dmol` view for an RDKit molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol or None
        Molecule to render. Returns ``None`` when this is ``None``.
    label_idx : bool, default=False
        If ``True``, display atom indices as labels.
    label_symbol : bool, default=False
        If ``True``, display atom symbols as labels.
    kekulize : bool, default=True
        If ``True``, kekulize the molecule before rendering.
    indices : list of int or None, default=None
        If provided, only these atoms are labeled.

    Returns
    -------
    py3Dmol.view or None
        View object for interactive display.

    Examples
    --------
    >>> # viewer = view3D(mol, label_idx=True)
    >>> # viewer is not None
    >>> # True
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

    def label_func(i: int, atom: Atom) -> str | None:
        """Build an optional atom label string for one atom.

        Parameters
        ----------
        i : int
            Atom index.
        atom : rdkit.Chem.rdchem.Atom
            Atom object.

        Returns
        -------
        str or None
            Label text for rendering.
        """
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


def diagnose_atom_valence(atom: Atom, show: bool = True) -> dict[str, DiagnosticValue]:
    """Summarize valence and charge diagnostics for one atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to inspect.
    show : bool, default=True
        If ``True``, print a compact human-readable summary.

    Returns
    -------
    dict of str to DiagnosticValue
        Structured diagnostics including atom properties, neighbor bond data,
        computed valence metrics, and heuristic warning messages.

    Examples
    --------
    >>> # report = diagnose_atom_valence(atom, show=False)
    >>> # "messages" in report
    >>> # True
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
    neighbor_info: list[NeighborInfo] = []
    sum_bond_orders = 0.0
    for b in bonds:
        nb_idx = b.GetOtherAtomIdx(idx)
        nb = b.GetOtherAtom(atom)
        btype = str(b.GetBondType())
        try:
            bord = float(b.GetBondTypeAsDouble())
        except Exception:
            # fallback for aromatic etc.
            bord: float = 1.5 if b.GetIsAromatic() else 1.0
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
    typical_valence_map: dict[str, int] = {
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
    typical_valence: int | None = typical_valence_map.get(symbol)

    # diagnostics / heuristics
    messages: list[str] = []
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
        summary: str = (
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
