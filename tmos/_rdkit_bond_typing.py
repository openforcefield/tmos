"""Utilities for building and sanitizing RDKit molecules from 3D geometries (e.g., XYZ).

This module provides routines to infer molecular connectivity, bond orders, and formal charges
from atom positions using a combination of RDKit and OpenBabel perception. It includes
custom cleanup logic to resolve common over-valence/charge assignment artifacts.

Public API
----------
- ``determine_bonds``: assign connectivity, bond orders, and formal charges.
- ``molecule_charge_penalty``: score formal-charge placement for optimization.
"""

import contextlib
import os
import time
from collections.abc import Generator, Iterable
from typing import TypeAlias

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBondOrders
from rdkit.Chem.rdchem import Atom, Bond, Mol
from openbabel import openbabel as ob

import tmos

# ---------------------------------------------------------------------------
# Stage-timing profiling state (module-level accumulators)
# ---------------------------------------------------------------------------
#: Cumulative wall-clock seconds per stage key across all determine_bonds calls.
_STAGE_TIMINGS: dict[str, float] = {}
#: Cumulative call counts per stage key.
_STAGE_COUNTS: dict[str, int] = {}

Path: TypeAlias = list[int]
PathAtomDeltas: TypeAlias = dict[int, int]
Objective: TypeAlias = tuple[int, int, int, int, int, int, int]


def reset_stage_timings() -> None:
    """Clear accumulated stage timing and call-count data."""
    _STAGE_TIMINGS.clear()
    _STAGE_COUNTS.clear()


def get_stage_timings() -> dict[str, dict[str, float | int]]:
    """Return a snapshot of cumulative stage timings and call counts.

    Returns
    -------
    dict
        ``{stage_name: {"time": float, "calls": int}}`` for every recorded stage.
    """
    return {
        k: {"time": _STAGE_TIMINGS[k], "calls": _STAGE_COUNTS.get(k, 0)}
        for k in _STAGE_TIMINGS
    }


@contextlib.contextmanager
def _time_stage(name: str) -> Generator[None, None, None]:
    """Accumulate wall-clock timing for one named stage.

    Parameters
    ----------
    name : str
        Stage key used to aggregate timing and call counts.

    Yields
    ------
    None
        Control to the wrapped stage body.
    """
    t0: float = time.perf_counter()
    try:
        yield
    finally:
        elapsed: float = time.perf_counter() - t0
        _STAGE_TIMINGS[name] = _STAGE_TIMINGS.get(name, 0.0) + elapsed
        _STAGE_COUNTS[name] = _STAGE_COUNTS.get(name, 0) + 1


# Suppress OpenBabel logging
ob.obErrorLog.SetOutputLevel(0)
ob.obErrorLog.StopLogging()
os.environ["BABEL_SILENCE"] = "1"

pt = Chem.GetPeriodicTable()

# Maximum additional positive formal charge tolerated in over-valence checks.
_MAX_BOND_FC: dict[int, int] = {
    1: 0,
    6: 0,
    9: 0,
    15: 0,
    16: 0,
    17: 0,
    35: 0,
    53: 0,
}  # H, C, F, P, S, Cl, Br, I → 0
_DEFAULT_MAX_BOND_FC = 1  # N, O, P, … → 1

# Elements that prefer lower-shell tie breaks (often cationic in common motifs).
_PREFER_LOWER_SHELL: frozenset[int] = frozenset({7, 15})  # N, P

# Priority for selecting negatively charged neighbors for bond promotion.
_NEG_NBOR_PRIORITY: dict[int, int] = {8: 0, 16: 1, 7: 2, 15: 3}  # O, S, N, P; rest = 99

# Element-wise charge preference penalties (smaller = preferred charge placement).
# Goal: minimise the number of charged atoms while preferring negative charge on
# O/S and positive charge on N/P, and strongly disfavoring [N-], [C+], [C-], O+.
_NEG_CHARGE_PENALTY: dict[int, int] = {
    8: 1,  # O-
    16: 7,  # S-
    15: 10,  # P-
    7: 40,  # N-
    6: 30,  # C-
    9: 12,
    17: 12,
    35: 12,
    53: 12,  # halides as covalent anions are disfavored
}
_POS_CHARGE_PENALTY: dict[int, int] = {
    7: 2,  # N+  (> NEG[O]=1 so N-[O-] is preferred over [N+]=O)
    15: 3,  # P+
    16: 5,  # S+
    8: 30,  # O+
    6: 14,  # C+
    # Strongly discourage positive halogens (move charge off Cl/Br/I/F if possible).
    9: 40,
    17: 40,
    35: 40,
    53: 40,
}

# Element-wise formal-charge bounds used by charge balancing passes.
_FORMAL_CHARGE_BOUNDS: dict[int, tuple[int, int]] = {
    6: (-1, 1),
    7: (-1, 1),
    8: (-1, 1),
    9: (-1, 0),
    15: (-1, 1),
    16: (-1, 1),
    17: (-1, 0),
    35: (-1, 0),
    53: (-1, 0),
}


def _formal_charge_in_bounds(atomic_num: int, formal_charge: int) -> bool:
    """Return ``True`` if formal charge is within element-specific bounds.

    Parameters
    ----------
    atomic_num : int
        Atomic number.
    formal_charge : int
        Formal charge to validate.

    Returns
    -------
    bool
        ``True`` when formal charge is allowed for this element.
    """
    lo, hi = _FORMAL_CHARGE_BOUNDS.get(atomic_num, (-2, 2))
    return lo <= formal_charge <= hi


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _best_shell(atom: Atom, bv: float) -> int | None:
    """Return the nearest valid valence shell for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.Atom
        Atom to evaluate.
    bv : float
        Current bond valence.

    Returns
    -------
    int | None
        Chosen shell, or ``None`` when no valid shell exists.
    """
    valences = _valid_valences(atom)
    if not valences:
        return None
    if atom.GetAtomicNum() in _PREFER_LOWER_SHELL:
        return min(valences, key=lambda v: (abs(v - bv), +v))  # lower shell wins ties
    return min(valences, key=lambda v: (abs(v - bv), -v))  # higher shell wins ties


def _valid_valences(atom: Atom) -> list[int]:
    """Return sorted neutral valence shells for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to evaluate.

    Returns
    -------
    list of int
        Non-negative valence values from the periodic table.
    """

    return sorted(v for v in pt.GetValenceList(atom.GetAtomicNum()) if v >= 0)


def _best_formal_charge(atom: Atom, bv: float) -> int | None:
    """Return preferred bounded formal charge for one atom at bond valence ``bv``.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to evaluate.
    bv : float
        Bond valence estimate.

    Returns
    -------
    int or None
        Preferred formal charge, or ``None`` when no valence shells exist.
    """
    valences = _valid_valences(atom)
    if not valences:
        return None

    atomic_num = atom.GetAtomicNum()
    prefer_lower = atomic_num in _PREFER_LOWER_SHELL
    candidates: list[tuple[float, int, float, int]] = []
    for shell in valences:
        fc = int(round(bv - shell))
        if not _formal_charge_in_bounds(atomic_num, fc):
            continue
        shell_err = abs(bv - shell)
        tie_shell = shell if prefer_lower else -shell
        candidates.append(
            (shell_err, _atom_charge_penalty(atomic_num, fc), tie_shell, fc)
        )

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return candidates[0][3]

    # Fallback when all bounded states are excluded: preserve old behavior and
    # clamp into allowed bounds so downstream charge balancing stays feasible.
    best = _best_shell(atom, bv)
    if best is None:
        return None
    fc = int(round(bv - best))
    lo, hi = _FORMAL_CHARGE_BOUNDS.get(atomic_num, (-2, 2))
    return max(lo, min(hi, fc))


def _bond_valence(atom: Atom, integer: bool = False) -> float | int:
    """Return bond-valence sum for one atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to evaluate.
    integer : bool, default=False
        If ``True``, truncate each bond order sum to an integer-compatible
        value used by over-valence checks.

    Returns
    -------
    float or int
        Bond-order sum including explicit hydrogens.

    ``integer=False`` (default): aromatic bonds contribute 1.5 — the float
    bond valence RDKit uses internally.
    ``integer=True``: each bond is truncated to its integer order, matching
    the over-valence check that SanitizeMol performs.
    """
    bv = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + atom.GetNumExplicitHs()
    return int(bv) if integer else bv


def _charge_adjusted_valences(atom: Atom) -> list[int]:
    """Return valence shells after element/charge-specific adjustments.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to evaluate.

    Returns
    -------
    list of int
        Candidate valence shells, including C+ restriction to valence 3.
    """
    valences = _valid_valences(atom)
    if atom.GetAtomicNum() == 6 and atom.GetFormalCharge() > 0:
        valences = [v for v in valences if v <= 3]
        if 3 not in valences:
            valences.append(3)
    return sorted(set(valences))


def _effective_valence(atom: Atom) -> float | int:
    """Return effective valence used in deficit calculations.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to evaluate.

    Returns
    -------
    float or int
        ``bond_valence - effective_formal_charge`` with C+ treated as neutral.
    """
    eff_fc = atom.GetFormalCharge()
    if atom.GetAtomicNum() == 6 and eff_fc > 0:
        eff_fc = 0
    return _bond_valence(atom) - eff_fc


def _valence_deficit(atom: Atom) -> float:
    """Return positive deficit to the nearest reachable valence shell.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to evaluate.

    Returns
    -------
    float
        Missing valence units. Returns ``0.0`` when no deficit exists.
    """
    valences = _charge_adjusted_valences(atom)
    if not valences:
        return 0.0
    ev = _effective_valence(atom)
    target = next((v for v in valences if v >= ev), None)
    if target is None:
        return 0.0
    return max(0.0, target - ev)


def _max_allowed_int_bv(atom: Atom) -> int:
    """Return integer bond-valence cap at the atom's current formal charge.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to evaluate.

    Returns
    -------
    int
        Maximum allowed integer bond valence.
    """
    return _max_allowed_int_bv_at_fc(atom, atom.GetFormalCharge())


def _max_allowed_int_bv_at_fc(atom: Atom, proposed_fc: int) -> int:
    """Maximum integer bond valence allowed for *atom* at a *proposed* formal charge.

    Unlike :func:`_max_allowed_int_bv`, this does not read the atom's current
    formal charge — it evaluates the cap hypothetically at ``proposed_fc``.
    Used by :meth:`_GraphMoveEngine._force_charge_balance` to prevent assigning a
    formal charge that would leave the atom over-valenced (e.g. C⁺ with bv=4).

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to evaluate.
    proposed_fc : int
        Hypothetical formal charge.

    Returns
    -------
    int
        Maximum allowed integer bond valence at ``proposed_fc``.
    """
    atomic_num = atom.GetAtomicNum()
    if atomic_num == 1:
        return 1
    valences = _valid_valences(atom)
    # Mirror _charge_adjusted_valences: C cations may only reach valence 3.
    if atomic_num == 6 and proposed_fc > 0:
        valences = [v for v in valences if v <= 3]
        if not valences:
            valences: list[int] = [3]
    if not valences:
        return 99
    max_fc_tol: int = _MAX_BOND_FC.get(atomic_num, _DEFAULT_MAX_BOND_FC)
    return max(valences) + max_fc_tol


def _has_overvalenced_atoms(mol: Mol) -> bool:
    """Return ``True`` when any atom exceeds integer bond-valence allowance.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to inspect.

    Returns
    -------
    bool
        ``True`` if at least one atom is currently over-valenced.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        if _bond_valence(atom, integer=True) > _max_allowed_int_bv(atom):
            return True
    return False


def molecule_charge_penalty(mol: Mol) -> int:
    """Return element-aware formal-charge penalty for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to score.

    Returns
    -------
    int
        Lower values indicate preferred charge localization.

    Lower values indicate more preferred charge localization.
    This score only reflects formal-charge placement (not over-valence).
    """
    score = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        fc = atom.GetFormalCharge()
        if fc == 0:
            continue
        score += 2 + abs(fc)
        if fc < 0:
            score += _NEG_CHARGE_PENALTY.get(atom.GetAtomicNum(), 6)
        else:
            score += _POS_CHARGE_PENALTY.get(atom.GetAtomicNum(), 6)
    return score


def _atom_charge_penalty(atomic_num: int, formal_charge: int) -> int:
    """Return element-aware penalty for one atom formal charge.

    Parameters
    ----------
    atomic_num : int
        Atomic number.
    formal_charge : int
        Formal charge to score.

    Returns
    -------
    int
        Penalty value where smaller is preferred.
    """
    if formal_charge == 0:
        return 0
    score = 2 + abs(formal_charge)
    if formal_charge < 0:
        score += _NEG_CHARGE_PENALTY.get(atomic_num, 6)
    else:
        score += _POS_CHARGE_PENALTY.get(atomic_num, 6)
    return score


def _target_charge_or_none(mol: Mol) -> int | None:
    """Return target charge stored in ``_target_charge``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule carrying optional target-charge metadata.

    Returns
    -------
    int or None
        Parsed target charge or ``None`` when missing/invalid.
    """
    if not mol.HasProp("_target_charge"):
        return None
    try:
        return int(mol.GetProp("_target_charge"))
    except Exception:
        return None


def _target_charge_delta(mol: Mol) -> int:
    """Return current-total-charge minus target charge.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to evaluate.

    Returns
    -------
    int
        Signed delta. Returns ``0`` when no target is present.
    """
    target: None | int = _target_charge_or_none(mol)
    if target is None:
        return 0
    current: int = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return current - target


def _bond_distance(conf, idx_a: int, idx_b: int) -> float:
    """Return Euclidean distance between two atoms from one conformer."""
    p0 = conf.GetAtomPosition(idx_a)
    p1 = conf.GetAtomPosition(idx_b)
    return float(np.linalg.norm(np.array([p1.x - p0.x, p1.y - p0.y, p1.z - p0.z])))


def _trial_after_bond_removal(mol: Mol, idx_a: int, idx_b: int) -> Mol:
    """Return a trial molecule after removing one bond and refreshing cache."""
    trial_rw = Chem.RWMol(mol)
    trial_rw.RemoveBond(idx_a, idx_b)
    trial = trial_rw.GetMol()
    trial.UpdatePropertyCache(strict=False)
    return trial


def _try_sanitize_or_none(mol: Mol) -> Mol | None:
    """Return molecule if sanitization succeeds, else ``None``."""
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _assign_and_sanitize_or_none(mol: Mol) -> Mol | None:
    """Assign formal charges and return sanitizable candidate, if any."""
    return _try_sanitize_or_none(_assign_formal_charges(Chem.Mol(mol)))


def _repair_overvalence_and_reassign(mol: Mol) -> Mol:
    """Fix over-valence when present, then refresh formal charges."""
    repaired = Chem.Mol(mol)
    if _has_overvalenced_atoms(repaired):
        repaired = _fix_overvalenced(repaired)
        repaired = _assign_formal_charges(repaired)
    return repaired


def _enforce_target_charge(mol: Mol) -> Mol:
    """Adjust formal charges to match molecule property ``_target_charge``.

    Uses one-charge-unit moves only when the destination charge is compatible
    with one of the atom's valid valence shells at current bond valence.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to adjust.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Molecule with formal charges nudged toward target total.
    """
    target_charge: None | int = _target_charge_or_none(mol)
    if target_charge is None:
        return mol

    for _ in range(128):
        current_charge: int = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        delta: int = target_charge - current_charge
        if delta == 0:
            break

        step: int = 1 if delta > 0 else -1
        best_atom: Atom | None = None
        best_new_fc: int | None = None
        best_score = None
        best_move = 0

        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num in (0, 1):
                continue

            old_fc = atom.GetFormalCharge()
            bv = _bond_valence(atom)
            valences = _valid_valences(atom)
            if not valences:
                continue

            allowed_fc: list[int] = sorted({int(round(bv - v)) for v in valences})
            for new_fc in allowed_fc:
                if not _formal_charge_in_bounds(atomic_num, new_fc):
                    continue
                move = new_fc - old_fc
                if move == 0 or move * step <= 0:
                    continue
                if abs(move) > abs(delta):
                    continue

                penalty_delta = _atom_charge_penalty(
                    atomic_num, new_fc
                ) - _atom_charge_penalty(atomic_num, old_fc)
                score = (penalty_delta / abs(move), penalty_delta, abs(new_fc))

                if (
                    best_atom is None
                    or best_score is None
                    or score < best_score
                    or (score == best_score and abs(move) > abs(best_move))
                ):
                    best_atom = atom
                    best_new_fc = new_fc
                    best_score = score
                    best_move = move

        if best_atom is None:
            break

        if best_new_fc is None:
            break
        best_atom.SetFormalCharge(best_new_fc)

    mol.UpdatePropertyCache(strict=False)
    return mol


def _charge_score(mol: Mol) -> int:
    """Element-aware charge score to guide global bond-order moves.

    Lower is better.  Strongly penalizes impossible over-valence, then penalizes
    charged atoms with element-dependent preferences:
    - prefer O-/S- over N-/C-
    - prefer N+/P+ over O+/C+

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to score.

    Returns
    -------
    int
        Composite integer score. Lower is better.
    """
    score = molecule_charge_penalty(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        radicals = atom.GetNumRadicalElectrons()
        if radicals > 0:
            score += 20000 * radicals
        int_bv = int(_bond_valence(atom, integer=True))
        over = int_bv - _max_allowed_int_bv(atom)
        if over > 0:
            score += 1000 * over
    return score


def _charge_objective(mol: Mol) -> Objective:
    """Lexicographic objective for trial bond-order moves.

    Primary key is ``_charge_score``; the target-charge gap is a secondary
    key to gently steer toward the requested net charge without overriding
    local chemistry. Tie-breakers favor fewer charged atoms, then fewer
    negative nitrogens (common over-separation artefact), then lower total
    absolute formal charge.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to score.

    Returns
    -------
    tuple of int
        Lexicographic objective tuple for move comparison.
    """
    charged_atoms = 0
    neg_nitrogen = 0
    protonated_ring_n_cations = 0
    radical_atoms = 0
    abs_charge = 0
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if atom.GetNumRadicalElectrons() > 0:
            radical_atoms += 1
        if fc != 0:
            charged_atoms += 1
            abs_charge += abs(fc)
            if atom.GetAtomicNum() == 7 and fc < 0:
                neg_nitrogen += 1
            if (
                atom.GetAtomicNum() == 7
                and fc > 0
                and atom.IsInRing()
                and atom.GetTotalNumHs() > 0
            ):
                protonated_ring_n_cations += 1
    return (
        abs(_target_charge_delta(mol)),
        _charge_score(mol),
        radical_atoms,
        charged_atoms,
        neg_nitrogen,
        protonated_ring_n_cations,
        abs_charge,
    )


def _apply_alternating_path_shift(
    mol: Mol,
    path: Path,
    first_delta: int = 1,
) -> Mol | None:
    """Try one arrow-pushing move along *path*.

    Bonds along the path are alternately promoted/demoted:
    ``first_delta, -first_delta, first_delta, ...`` on successive edges.
    Returns a new molecule if the move is valence-safe and non-aromatic, else None.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to modify.
    path : list of int
        Ordered atom indices along the path.
    first_delta : int, default=1
        Initial bond-order delta (+1 or -1).

    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        Updated molecule when feasible; otherwise ``None``.
    """
    if len(path) < 2:
        return None

    bond_updates = []
    atom_delta = {}
    for i, (idx_a, idx_b) in enumerate(zip(path[:-1], path[1:])):
        atom_a = mol.GetAtomWithIdx(idx_a)
        atom_b = mol.GetAtomWithIdx(idx_b)
        if atom_a.GetAtomicNum() == 1 or atom_b.GetAtomicNum() == 1:
            return None
        bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
        if bond is None:
            return None
        delta: int = first_delta if i % 2 == 0 else -first_delta
        order = int(bond.GetBondTypeAsDouble())
        new_order: int = order + delta
        if new_order < 1 or new_order > 3:
            return None
        bond_updates.append((bond.GetIdx(), new_order))
        atom_delta[idx_a] = atom_delta.get(idx_a, 0) + delta
        atom_delta[idx_b] = atom_delta.get(idx_b, 0) + delta

    for idx, delta in atom_delta.items():
        atom = mol.GetAtomWithIdx(idx)
        new_bv = _bond_valence(atom, integer=True) + delta
        if new_bv < 0 or new_bv > _max_allowed_int_bv(atom):
            return None

    rw = Chem.RWMol(mol)
    for bidx, new_order in bond_updates:
        bond = rw.GetBondWithIdx(bidx)
        bond.SetBondType(
            Chem.BondType.SINGLE
            if new_order == 1
            else (Chem.BondType.DOUBLE if new_order == 2 else Chem.BondType.TRIPLE)
        )
    rw.UpdatePropertyCache(strict=False)
    return rw.GetMol()


def _apply_alternating_path_shift_via_kekulize(
    mol: Mol,
    path: Path,
    first_delta: int = 1,
) -> Mol | None:
    """Attempt path shift on a temporary kekulized copy of *mol*.

    This enables resonance-style updates in aromatic systems where the direct
    path shift rejects aromatic bonds. If kekulization fails, returns ``None``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to modify.
    path : list of int
        Ordered atom indices along the path.
    first_delta : int, default=1
        Initial bond-order delta (+1 or -1).

    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        Shifted molecule or ``None``.
    """
    try:
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None

    shifted = _apply_alternating_path_shift(kek, path, first_delta=first_delta)
    if shifted is None:
        return None
    shifted.UpdatePropertyCache(strict=False)
    return shifted


def _compute_path_deltas(
    mol: Mol,
    path: Path,
    first_delta: int = 1,
) -> PathAtomDeltas | None:
    """Check path-move feasibility and return per-atom BV deltas without copying *mol*.

    Performs the same phase-1 checks as :func:`_apply_alternating_path_shift`
    (H atoms, bond existence, order bounds 1–3, atom BV caps) purely through
    arithmetic on the existing molecule — no ``RWMol`` is ever constructed.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
    path : list of int
        Ordered atom indices along the path.
    first_delta : int
        +1 or −1 for the first bond on the path.

    Returns
    -------
    dict of int to int or None
        Map of atom index → cumulative BV change for every atom on the path.
        ``None`` if the move is infeasible for any reason.
    """
    if len(path) < 2:
        return None

    atom_delta = {}
    for i, (idx_a, idx_b) in enumerate(zip(path[:-1], path[1:])):
        atom_a = mol.GetAtomWithIdx(idx_a)
        atom_b = mol.GetAtomWithIdx(idx_b)
        if atom_a.GetAtomicNum() == 1 or atom_b.GetAtomicNum() == 1:
            return None
        bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
        if bond is None:
            return None
        delta: int = first_delta if i % 2 == 0 else -first_delta
        order = int(bond.GetBondTypeAsDouble())
        new_order: int = order + delta
        if new_order < 1 or new_order > 3:
            return None
        atom_delta[idx_a] = atom_delta.get(idx_a, 0) + delta
        atom_delta[idx_b] = atom_delta.get(idx_b, 0) + delta

    for idx, delta in atom_delta.items():
        atom = mol.GetAtomWithIdx(idx)
        new_bv = _bond_valence(atom, integer=True) + delta
        if new_bv < 0 or new_bv > _max_allowed_int_bv(atom):
            return None

    return atom_delta


def _generate_paths_between_atoms(
    mol: Mol,
    idx_a: int,
    idx_b: int,
    graph: nx.Graph | None = None,
) -> Generator[Path, None, None]:
    """Yield candidate paths between two atoms lazily.

    Paths are generated one-by-one (without materializing the full set) so
    callers can evaluate each path and stop as soon as they find an improving
    move. Path generation is exhaustive over simple paths in the heavy-atom
    graph.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule used to build a heavy-atom graph when ``graph`` is ``None``.
    idx_a : int
        Source atom index.
    idx_b : int
        Destination atom index.
    graph : networkx.Graph or None, default=None
        Optional precomputed heavy-atom graph.

    Yields
    ------
    list of int
        Candidate simple paths between ``idx_a`` and ``idx_b``.
    """
    if graph is None:
        graph = tmos.graph_mapping.mol_to_graph(mol, remove_hydrogens=True)

    try:
        for path in nx.all_simple_paths(graph, idx_a, idx_b):
            if len(path) < 2:
                continue
            yield list(path)
    except Exception:
        return


def _set_bond_order_int(bond: Bond, order: int) -> None:
    """Set a bond type from integer order.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        Bond to mutate.
    order : int
        Target order in ``{1, 2, 3}``.
    """
    bond.SetBondType(
        Chem.BondType.SINGLE
        if order == 1
        else (Chem.BondType.DOUBLE if order == 2 else Chem.BondType.TRIPLE)
    )


def _neutral_sink_priority(atom: Atom) -> int:
    """Priority for neutral sink atoms (higher = preferred sink).

    Encodes preference for oxygen on S/N over oxygen on C.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        Atom to score as a sink.

    Returns
    -------
    int
        Priority score.
    """
    if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() > 0:
        return 0

    atomic_num = atom.GetAtomicNum()
    if atomic_num == 8:
        neighbors = [
            n.GetAtomicNum() for n in atom.GetNeighbors() if n.GetAtomicNum() != 1
        ]
        if any(z in (16, 7) for z in neighbors):
            return 140
        if any(z == 15 for z in neighbors):
            return 120
        if any(z == 6 for z in neighbors):
            return 60
        return 80

    if atomic_num == 7:
        return 80
    if atomic_num == 16:
        return 55
    return 0


def _strip_radicals_and_reassign(mol: Mol) -> Mol:
    """Clear radicals and recompute formal charges.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to normalize.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Molecule with zero radical electrons where possible.
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumRadicalElectrons(0)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    out = _assign_formal_charges(out)
    return out


def _convert_radical_carbocations(mol: Mol) -> Mol:
    """Convert tricoordinate C radicals to C⁺ cations when target charge demands it.

    OBabel occasionally assigns bv=3, rad=1, fc=0 to a benzylic CH₂ that is
    chemically a carbocation [CH₂⁺].  :func:`_assign_formal_charges` leaves
    fc=0 because bv=3.0 maps to a neutral carbon via the radical (non-bonding)
    electron path rather than to C⁺ (valid shell 3, fc=+1).

    This pass fires only when the molecule has a positive ``_target_charge``
    and is currently under-charged (current total charge < target).  For each
    tricoordinate C radical (bv==3, rad≥1, fc==0) it sets fc=+1 and removes
    one radical electron, stopping once the target charge is reached.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to adjust.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Molecule after targeted radical-to-cation conversion.
    """
    target: None | int = _target_charge_or_none(mol)
    if target is None:
        return mol
    current: int = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if current >= target:
        return mol
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if current >= target:
            break
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetNumRadicalElectrons() < 1:
            continue
        if _bond_valence(atom, integer=True) != 3:
            continue
        atom.SetFormalCharge(1)
        atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1)
        current += 1
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


class _GraphMoveEngine:
    """Unified graph-based bond-order move optimizer.

    Uses one chemistry objective (``_charge_objective``) across both local
    relay moves and path-based arrow-pushing moves.
    """

    def __init__(
        self,
        max_iters: int = 24,
        use_relay: bool = True,
        use_paths: bool = True,
        path_mode: str = "charged_sinks",
    ) -> None:
        """Configure one graph-move optimizer instance.

        Parameters
        ----------
        max_iters : int, default=24
            Maximum optimization iterations.
        use_relay : bool, default=True
            Enable local relay moves.
        use_paths : bool, default=True
            Enable path-based alternating shifts.
        path_mode : str, default="charged_sinks"
            Path endpoint strategy.
        """
        self.max_iters: int = max_iters
        self.use_relay: bool = use_relay
        self.use_paths: bool = use_paths
        self.path_mode: str = path_mode

    @staticmethod
    def _prep_candidate(candidate: Mol) -> Mol:
        """Normalize one candidate after a bond-order move.

        Parameters
        ----------
        candidate : rdkit.Chem.rdchem.Mol
            Candidate molecule.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            Candidate after cleanup primitives.
        """
        candidate = _prepare_for_cleanup(candidate)
        candidate = _fix_overvalenced(candidate)
        candidate = _assign_formal_charges(candidate)
        return candidate

    @staticmethod
    def _is_active_site(atom: Atom) -> bool:
        """Check whether an atom is charge/radical-active.

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            Atom to inspect.

        Returns
        -------
        bool
            ``True`` when formal charge is non-zero or radicals are present.
        """
        return atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() > 0

    @staticmethod
    def _atom_charge_cost(atom: Atom) -> int:
        """Return local penalty for one atom state.

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            Atom to score.

        Returns
        -------
        int
            Local atom penalty used for prioritization.
        """
        radicals = atom.GetNumRadicalElectrons()
        if radicals > 0:
            return 10000 + 1000 * radicals
        return _atom_charge_penalty(atom.GetAtomicNum(), atom.GetFormalCharge())

    @classmethod
    def _atom_site_penalty(cls, mol: Mol, atom_idx: int) -> int:
        """Return optimization priority penalty for one atom.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to inspect.
        atom_idx : int
            Atom index.

        Returns
        -------
        int
            Site penalty used for ranking.
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        charge_cost = cls._atom_charge_cost(atom)
        if charge_cost > 0:
            return charge_cost

        if atom.GetAtomicNum() in (0, 1):
            return 0
        valences = _charge_adjusted_valences(atom)
        if not valences:
            return 0
        deficit: float = _valence_deficit(atom)
        if deficit <= 0:
            return 0
        return 10 + int(np.ceil(deficit * 10.0))

    @classmethod
    def _priority_active_atoms(cls, mol: Mol) -> list[int]:
        """Return active heavy atoms sorted by descending site penalty.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to inspect.

        Returns
        -------
        list of int
            Active heavy-atom indices.
        """
        active = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() != 1 and cls._is_active_site(atom)
        ]
        active.sort(key=lambda idx: cls._atom_site_penalty(mol, idx), reverse=True)
        return active

    @classmethod
    def _priority_sink_atoms(
        cls,
        mol: Mol,
        exclude: Iterable[int] | None = None,
    ) -> list[int]:
        """Return preferred neutral sink atoms.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to inspect.
        exclude : iterable of int or None, default=None
            Atom indices to omit.

        Returns
        -------
        list of int
            Atom indices ordered by sink priority.
        """
        exclude = set() if exclude is None else set(exclude)
        sinks = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx in exclude or atom.GetAtomicNum() == 1:
                continue
            pri: int = _neutral_sink_priority(atom)
            if pri > 0:
                sinks.append((idx, pri))
        sinks.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sinks]

    @classmethod
    def _priority_partner_atoms(
        cls,
        mol: Mol,
        src_idx: int,
        include_zero: bool = False,
        max_partners: int = 24,
    ) -> list[int]:
        """Return candidate partner atoms for one source atom.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to inspect.
        src_idx : int
            Source atom index.
        include_zero : bool, default=False
            Include zero-penalty atoms when ``True``.
        max_partners : int, default=24
            Maximum number of returned atoms.

        Returns
        -------
        list of int
            Partner atom indices.
        """
        partners = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx == src_idx or atom.GetAtomicNum() == 1:
                continue
            score = cls._atom_site_penalty(mol, idx)
            if score <= 0 and not include_zero:
                continue
            partners.append((idx, score))
        partners.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in partners[:max_partners]]

    @classmethod
    def _best_path_move_between_pair(
        cls,
        engine: "_GraphMoveEngine",
        base: Mol,
        src: int,
        dst: int,
        require_dst_improve: bool,
        graph: nx.Graph | None = None,
    ) -> Mol | None:
        """Return best improving path move for one source/destination pair.

        Parameters
        ----------
        engine : _GraphMoveEngine
            Engine used to apply path moves.
        base : rdkit.Chem.rdchem.Mol
            Baseline molecule.
        src : int
            Source atom index.
        dst : int
            Destination atom index.
        require_dst_improve : bool
            Require destination penalty reduction when ``True``.
        graph : networkx.Graph or None, default=None
            Optional precomputed heavy-atom graph.

        Returns
        -------
        rdkit.Chem.rdchem.Mol or None
            First improving candidate, otherwise ``None``.
        """
        base_obj = _charge_objective(base)
        base_score = _charge_score(base)
        src_penalty_base = cls._atom_site_penalty(base, src)
        dst_penalty_base = cls._atom_site_penalty(base, dst)

        with _time_stage("path_enum"):
            path_iter = _generate_paths_between_atoms(
                base,
                src,
                dst,
                graph=graph,
            )

        src_atom = base.GetAtomWithIdx(src)
        src_fc_base = src_atom.GetFormalCharge()
        src_bv_base = _bond_valence(src_atom)  # float, used for shell estimate
        src_cur_penalty = _atom_charge_penalty(src_atom.GetAtomicNum(), src_fc_base)
        dst_atom = base.GetAtomWithIdx(dst)
        dst_fc_base = dst_atom.GetFormalCharge()
        dst_bv_base = _bond_valence(dst_atom)
        dst_cur_penalty = _atom_charge_penalty(dst_atom.GetAtomicNum(), dst_fc_base)

        with _time_stage("path_eval"):
            for path in path_iter:
                path_atom_set = set(path)
                for first_delta in (1, -1):
                    atom_delta = _compute_path_deltas(base, path, first_delta)
                    if atom_delta is None:
                        continue

                    src_delta = atom_delta[src]
                    src_bv_est = src_bv_base + src_delta
                    src_shell_est = _best_shell(src_atom, src_bv_est)
                    if src_shell_est is not None:
                        src_fc_est = int(round(src_bv_est - src_shell_est))
                        if (
                            _atom_charge_penalty(src_atom.GetAtomicNum(), src_fc_est)
                            >= src_cur_penalty
                        ):
                            continue

                    if require_dst_improve:
                        dst_delta = atom_delta[dst]
                        dst_bv_est = dst_bv_base + dst_delta
                        dst_shell_est = _best_shell(dst_atom, dst_bv_est)
                        if dst_shell_est is not None:
                            dst_fc_est = int(round(dst_bv_est - dst_shell_est))
                            if (
                                _atom_charge_penalty(
                                    dst_atom.GetAtomicNum(), dst_fc_est
                                )
                                >= dst_cur_penalty
                            ):
                                continue

                    candidate = engine._apply_path_move(base, path, first_delta)
                    if candidate is None:
                        continue
                    candidate = _assign_formal_charges_local(candidate, path_atom_set)

                    cand_score = _charge_score(candidate)
                    cand_obj = _charge_objective(candidate)
                    if cand_score > base_score:
                        continue
                    if cand_score == base_score and cand_obj >= base_obj:
                        continue

                    src_penalty_cand = cls._atom_site_penalty(candidate, src)
                    dst_penalty_cand = cls._atom_site_penalty(candidate, dst)
                    if src_penalty_cand >= src_penalty_base:
                        continue
                    if require_dst_improve and dst_penalty_cand >= dst_penalty_base:
                        continue

                    promoted = _reduce_charge_by_bond_promotion(Chem.Mol(candidate))
                    promoted_score = _charge_score(promoted)
                    promoted_obj = _charge_objective(promoted)
                    if promoted_score > base_score:
                        return candidate
                    if promoted_score == base_score and promoted_obj >= base_obj:
                        return candidate

                    promoted_src_penalty = cls._atom_site_penalty(promoted, src)
                    promoted_dst_penalty = cls._atom_site_penalty(promoted, dst)
                    if promoted_src_penalty >= src_penalty_base:
                        return candidate
                    if require_dst_improve and promoted_dst_penalty >= dst_penalty_base:
                        return candidate

                    return promoted

        return None

    @staticmethod
    def _apply_adjacent_move(mol: Mol, bond_idx: int, delta: int) -> Mol | None:
        """Apply one local adjacent bond-order move if valence-safe.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to modify.
        bond_idx : int
            Bond index to update.
        delta : int
            Bond-order increment or decrement.

        Returns
        -------
        rdkit.Chem.rdchem.Mol or None
            Updated molecule when feasible; otherwise ``None``.
        """
        trial = Chem.RWMol(mol)
        bond = trial.GetBondWithIdx(bond_idx)
        if bond is None:
            return None

        old_order = int(round(bond.GetBondTypeAsDouble()))
        new_order = old_order + delta
        if not (1 <= new_order <= 3):
            return None

        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        begin_atom = mol.GetAtomWithIdx(begin_idx)
        end_atom = mol.GetAtomWithIdx(end_idx)

        for atom in (begin_atom, end_atom):
            if atom.GetAtomicNum() == 1:
                return None
            new_bv = _bond_valence(atom, integer=True) + delta
            if new_bv < 0 or new_bv > _max_allowed_int_bv(atom):
                return None

        _set_bond_order_int(bond, new_order)
        trial.UpdatePropertyCache(strict=False)
        return trial.GetMol()

    @classmethod
    def _promote_atom_neighbors_greedy(cls, mol: Mol) -> Mol:
        """Greedy local neutralization: promote eligible neighbor bonds per atom.

        Atoms are processed in descending local charge cost (e.g., C- first).
        For each active atom, apply the best legal promotion that does not worsen
        global charge penalty and does not worsen that atom's |formal charge|,
        then re-evaluate the same atom before moving on.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to optimize.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            Molecule after local greedy promotions.
        """
        work = _assign_formal_charges(Chem.Mol(mol))

        changed_any = True
        while changed_any:
            changed_any = False

            atom_order = sorted(
                [a.GetIdx() for a in work.GetAtoms() if a.GetAtomicNum() != 1],
                key=lambda idx: cls._atom_charge_cost(work.GetAtomWithIdx(idx)),
                reverse=True,
            )

            for atom_idx in atom_order:
                improved_this_atom = True
                while improved_this_atom:
                    improved_this_atom = False
                    atom = work.GetAtomWithIdx(atom_idx)
                    if not cls._is_active_site(atom):
                        break

                    base_score = _charge_score(work)
                    base_obj = _charge_objective(work)
                    base_abs_fc = abs(atom.GetFormalCharge())

                    best_candidate = None
                    best_penalty = None
                    best_obj = None

                    for bond in atom.GetBonds():
                        other = bond.GetOtherAtom(atom)
                        if other.GetAtomicNum() == 1:
                            continue

                        order = int(round(bond.GetBondTypeAsDouble()))
                        if order >= 3:
                            continue

                        candidate = cls._apply_adjacent_move(
                            work, bond.GetIdx(), delta=1
                        )
                        if candidate is None:
                            continue
                        candidate = cls._prep_candidate(candidate)

                        cand_atom = candidate.GetAtomWithIdx(atom_idx)
                        cand_abs_fc = abs(cand_atom.GetFormalCharge())
                        if atom.GetFormalCharge() != 0 and cand_abs_fc > base_abs_fc:
                            continue

                        cand_score = _charge_score(candidate)
                        cand_obj = _charge_objective(candidate)
                        if cand_score > base_score:
                            continue
                        if cand_score == base_score and cand_obj >= base_obj:
                            continue

                        if (
                            best_candidate is None
                            or best_penalty is None
                            or cand_score < best_penalty
                            or (
                                cand_score == best_penalty
                                and (best_obj is None or cand_obj < best_obj)
                            )
                        ):
                            best_candidate = candidate
                            best_penalty = cand_score
                            best_obj = cand_obj

                    if best_candidate is None:
                        break

                    work = best_candidate
                    changed_any = True
                    improved_this_atom = True

        return work

    def _iter_relay_specs(
        self, mol: Mol
    ) -> Generator[tuple[int, int, int], None, None]:
        """Yield legal relay-move bond-index triplets.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to inspect.

        Yields
        ------
        tuple of (int, int, int)
            ``(ab_idx, bc_idx, cd_idx)`` relay specifications.
        """
        seen_moves = set()
        for center in mol.GetBonds():
            b = center.GetBeginAtom()
            c = center.GetEndAtom()

            if b.GetAtomicNum() == 1 or c.GetAtomicNum() == 1:
                continue
            center_order = int(center.GetBondTypeAsDouble())
            if center_order < 2:
                continue
            if b.GetFormalCharge() == 0 and c.GetFormalCharge() == 0:
                continue

            b_neighbors = []
            for bb in b.GetBonds():
                a = bb.GetOtherAtom(b)
                if a.GetIdx() == c.GetIdx() or a.GetAtomicNum() == 1:
                    continue
                if int(bb.GetBondTypeAsDouble()) >= 3:
                    continue
                if (not bb.IsInRing()) and b.IsInRing() and a.IsInRing():
                    continue
                b_neighbors.append(bb.GetIdx())

            c_neighbors = []
            for cb in c.GetBonds():
                d = cb.GetOtherAtom(c)
                if d.GetIdx() == b.GetIdx() or d.GetAtomicNum() == 1:
                    continue
                if int(cb.GetBondTypeAsDouble()) >= 3:
                    continue
                if (not cb.IsInRing()) and c.IsInRing() and d.IsInRing():
                    continue
                c_neighbors.append(cb.GetIdx())

            if not b_neighbors or not c_neighbors:
                continue

            for ab_idx in b_neighbors:
                for cd_idx in c_neighbors:
                    if ab_idx == cd_idx:
                        continue
                    move_key = tuple(sorted((ab_idx, center.GetIdx(), cd_idx)))
                    if move_key in seen_moves:
                        continue
                    seen_moves.add(move_key)
                    yield ab_idx, center.GetIdx(), cd_idx

    @staticmethod
    def _apply_relay_move(
        mol: Mol,
        ab_idx: int,
        bc_idx: int,
        cd_idx: int,
    ) -> Mol | None:
        """Apply a three-bond relay move when all target orders are valid.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to modify.
        ab_idx : int
            First bond index.
        bc_idx : int
            Center bond index (demoted).
        cd_idx : int
            Third bond index.

        Returns
        -------
        rdkit.Chem.rdchem.Mol or None
            Updated molecule or ``None``.
        """
        trial = Chem.RWMol(mol)
        t_ab = trial.GetBondWithIdx(ab_idx)
        t_bc = trial.GetBondWithIdx(bc_idx)
        t_cd = trial.GetBondWithIdx(cd_idx)
        if t_ab is None or t_bc is None or t_cd is None:
            return None

        o_ab = int(t_ab.GetBondTypeAsDouble())
        o_bc = int(t_bc.GetBondTypeAsDouble())
        o_cd = int(t_cd.GetBondTypeAsDouble())
        n_ab: int = o_ab + 1
        n_bc: int = o_bc - 1
        n_cd: int = o_cd + 1
        if n_ab < 1 or n_ab > 3 or n_bc < 1 or n_bc > 3 or n_cd < 1 or n_cd > 3:
            return None

        _set_bond_order_int(t_ab, n_ab)
        _set_bond_order_int(t_bc, n_bc)
        _set_bond_order_int(t_cd, n_cd)
        trial.UpdatePropertyCache(strict=False)
        return trial.GetMol()

    def _path_source_target_atoms(self, mol: Mol) -> tuple[list[int], list[int]]:
        """Select source/target atom lists for path exploration.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to inspect.

        Returns
        -------
        tuple of (list of int, list of int)
            ``(sources, targets)`` ordered by priority.
        """
        active_atoms = self._priority_active_atoms(mol)

        if self.path_mode == "neg_pos":
            src = [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() != 1 and atom.GetFormalCharge() < 0
            ]
            dst = [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() != 1 and atom.GetFormalCharge() > 0
            ]
            src.sort(key=lambda idx: self._atom_site_penalty(mol, idx), reverse=True)
            dst.sort(key=lambda idx: self._atom_site_penalty(mol, idx), reverse=True)
            if src and dst:
                return src, dst
            return active_atoms, active_atoms

        source = list(active_atoms)
        sinks = self._priority_sink_atoms(mol, exclude=active_atoms)
        target = list(active_atoms) + sinks
        return source, target

    def _iter_path_moves(
        self,
        mol: Mol,
        graph: nx.Graph | None = None,
    ) -> Generator[tuple[Path, int], None, None]:
        """Yield prioritized path moves as ``(path, first_delta)`` pairs.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to inspect.
        graph : networkx.Graph or None, default=None
            Optional precomputed heavy-atom graph.

        Yields
        ------
        tuple of (list of int, int)
            Candidate path and initial bond-order delta.
        """
        source_atoms, target_atoms = self._path_source_target_atoms(mol)

        endpoint_scores_cache = {}

        def _endpoint_delta_scores(atom_idx: int) -> dict[int, int]:
            """Estimate endpoint penalty reduction for ±1 bond-valence shifts.

            Parameters
            ----------
            atom_idx : int
                Atom index at a path endpoint.

            Returns
            -------
            dict of int to int
                Improvement scores keyed by delta ``{1, -1}``.
            """
            cached = endpoint_scores_cache.get(atom_idx)
            if cached is not None:
                return cached

            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() in (0, 1):
                scores: dict[int, int] = {1: 0, -1: 0}
                endpoint_scores_cache[atom_idx] = scores
                return scores

            # Radical sites are hard constraints; do not over-fit direction here.
            if atom.GetNumRadicalElectrons() > 0:
                scores: dict[int, int] = {1: 0, -1: 0}
                endpoint_scores_cache[atom_idx] = scores
                return scores

            current_fc = atom.GetFormalCharge()
            current_penalty = _atom_charge_penalty(atom.GetAtomicNum(), current_fc)
            current_bv = _bond_valence(atom)

            scores: dict[int, int] = {1: 0, -1: 0}
            for delta in (1, -1):
                new_bv = current_bv + delta
                if new_bv < 0 or new_bv > _max_allowed_int_bv(atom):
                    continue

                shell = _best_shell(atom, new_bv)
                if shell is None:
                    continue

                new_fc = int(round(new_bv - shell))
                new_penalty: int = _atom_charge_penalty(atom.GetAtomicNum(), new_fc)
                scores[delta] = max(0, current_penalty - new_penalty)

            endpoint_scores_cache[atom_idx] = scores
            return scores

        seen_paths = set()
        for src in source_atoms:
            src_scores = _endpoint_delta_scores(src)
            for dst in target_atoms:
                if src == dst:
                    continue

                dst_scores = _endpoint_delta_scores(dst)
                # Pair-level pre-screen: skip if endpoint states suggest no
                # local penalty reduction in either direction.
                if max(src_scores.values()) <= 0 and max(dst_scores.values()) <= 0:
                    continue

                for path in _generate_paths_between_atoms(
                    mol,
                    src,
                    dst,
                    graph=graph,
                ):
                    key = tuple(path)
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)

                    odd_edges: bool = (len(path) - 1) % 2 == 1
                    dst_delta_if_pos: int = 1 if odd_edges else -1
                    dst_delta_if_neg: int = -1 if odd_edges else 1

                    pos_score = 2 * src_scores[1] + dst_scores[dst_delta_if_pos]
                    neg_score = 2 * src_scores[-1] + dst_scores[dst_delta_if_neg]

                    if pos_score <= 0 and neg_score <= 0:
                        continue

                    first_delta: int = 1 if pos_score >= neg_score else -1
                    yield path, first_delta

    @staticmethod
    def _apply_path_move(mol: Mol, path: Path, first_delta: int) -> Mol | None:
        """Apply one alternating path move with kekulized fallback.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to modify.
        path : list of int
            Ordered path atom indices.
        first_delta : int
            Initial bond-order delta.

        Returns
        -------
        rdkit.Chem.rdchem.Mol or None
            Updated molecule when feasible; otherwise ``None``.
        """
        cand = _apply_alternating_path_shift(mol, path, first_delta=first_delta)
        if cand is None:
            cand = _apply_alternating_path_shift_via_kekulize(
                mol,
                path,
                first_delta=first_delta,
            )
        return cand

    def optimize(self, mol: Mol) -> Mol:
        """Optimize one molecule with configured relay/path move passes.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to optimize.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            Optimized molecule.
        """
        mol = _assign_formal_charges(mol)
        current_obj = _charge_objective(mol)

        for _ in range(self.max_iters):
            work = self._kekulized_or_none(Chem.Mol(mol))
            if work is None:
                work = Chem.Mol(mol)
            work = _assign_formal_charges(work)

            # Build H-free graph once per iteration; shared across all path moves.
            iter_graph = (
                tmos.graph_mapping.mol_to_graph(work, remove_hydrogens=True)
                if self.use_paths
                else None
            )

            best_mol = None
            best_obj = current_obj

            if self.use_relay:
                for ab_idx, bc_idx, cd_idx in self._iter_relay_specs(work):
                    candidate = self._apply_relay_move(work, ab_idx, bc_idx, cd_idx)
                    if candidate is None:
                        continue
                    candidate = self._prep_candidate(candidate)
                    cand_obj = _charge_objective(candidate)
                    if cand_obj < best_obj:
                        best_obj = cand_obj
                        best_mol = candidate

            if self.use_paths:
                for path, first_delta in self._iter_path_moves(
                    work,
                    graph=iter_graph,
                ):
                    candidate = self._apply_path_move(work, path, first_delta)
                    if candidate is None:
                        continue
                    candidate = self._prep_candidate(candidate)
                    cand_obj = _charge_objective(candidate)
                    if cand_obj < best_obj:
                        best_obj = cand_obj
                        best_mol = candidate
                        break

                if best_mol is not None and best_obj < current_obj:
                    mol = best_mol
                    current_obj = best_obj
                    continue

            if best_mol is None:
                break

            mol = best_mol
            current_obj = best_obj

        return mol

    @staticmethod
    def _sanitize_candidate_or_none(mol: Mol) -> Mol | None:
        """Return sanitizable candidate, or ``None``.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Candidate molecule.

        Returns
        -------
        rdkit.Chem.rdchem.Mol or None
            Sanitized molecule when possible.
        """

        # Fast path: no structural edits, only formal-charge assignment.
        sanitized = _assign_and_sanitize_or_none(mol)
        if sanitized is not None:
            return sanitized

        # Repair path: demote over-valence first, then reassign charges.
        repaired = _repair_overvalence_and_reassign(mol)
        sanitized = _assign_and_sanitize_or_none(repaired)
        if sanitized is not None:
            return sanitized

        # Last resort: remove radicals, then repeat valence repair.
        cleaned = _strip_radicals_and_reassign(Chem.Mol(mol))
        cleaned = _repair_overvalence_and_reassign(cleaned)
        return _assign_and_sanitize_or_none(cleaned)

    @staticmethod
    def _kekulized_or_none(mol: Mol) -> Mol | None:
        """Return kekulized copy of a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to kekulize.

        Returns
        -------
        rdkit.Chem.rdchem.Mol or None
            Kekulized molecule, or ``None`` on failure.
        """
        try:
            candidate = Chem.Mol(mol)
            Chem.Kekulize(candidate, clearAromaticFlags=True)
            return candidate
        except Exception:
            return None

    @staticmethod
    def _prefer_lower_penalty(current: Mol, candidate: Mol | None) -> Mol:
        """Return the lower-penalty molecule between two candidates.

        Parameters
        ----------
        current : rdkit.Chem.rdchem.Mol
            Current best molecule.
        candidate : rdkit.Chem.rdchem.Mol or None
            New candidate to compare.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            Preferred molecule according to objective/tolerance rules.
        """
        if candidate is None:
            return current
        curr_obj = _charge_objective(current)
        cand_obj = _charge_objective(candidate)

        # Softly prefer candidates that close the target-charge gap, even if
        # the chemistry objective is slightly worse, provided charge_score does
        # not degrade beyond a small tolerance. This helps converge to the
        # requested net charge without overturning local chemistry preferences.
        curr_gap: int = abs(_target_charge_delta(current))
        cand_gap: int = abs(_target_charge_delta(candidate))
        if cand_gap < curr_gap:
            curr_score = _charge_score(current)
            cand_score = _charge_score(candidate)
            if cand_score <= curr_score + 4:
                return candidate

        if cand_obj < curr_obj:
            return candidate
        return current

    @staticmethod
    def _charge_cost_with_fc(atom: Atom, new_fc: int) -> int:
        """Return atom cost for a hypothetical formal charge.

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            Atom to score.
        new_fc : int
            Proposed formal charge.

        Returns
        -------
        int
            Local charge/radical penalty.
        """
        radicals = atom.GetNumRadicalElectrons()
        if radicals > 0:
            return 10000 + 1000 * radicals
        return _atom_charge_penalty(atom.GetAtomicNum(), new_fc)

    @classmethod
    def _force_charge_balance(cls, mol: Mol) -> Mol:
        """Nudge formal charges toward the target total without changing bonds.

        Only moves that leave the atom within its allowed integer bond valence
        at the proposed new formal charge are considered.  This prevents, for
        example, assigning fc=+1 to a tetravalent carbon (bv=4) which would
        create an over-valenced C⁺ that fails RDKit sanitization.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to adjust.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            Molecule with charges nudged toward target total.
        """
        target: None | int = _target_charge_or_none(mol)
        if target is None:
            return mol

        rw = Chem.RWMol(mol)
        for _ in range(64):
            current: int = sum(a.GetFormalCharge() for a in rw.GetAtoms())
            if current == target:
                break
            step: int = 1 if target > current else -1

            best_idx = None
            best_score = None

            for atom in rw.GetAtoms():
                if atom.GetAtomicNum() in (0, 1):
                    continue

                new_fc = atom.GetFormalCharge() + step
                if not _formal_charge_in_bounds(atom.GetAtomicNum(), new_fc):
                    continue
                # Guard: skip if the atom would become over-valenced at new_fc.
                int_bv = _bond_valence(atom, integer=True)
                if int_bv > _max_allowed_int_bv_at_fc(atom, new_fc):
                    continue

                score = cls._charge_cost_with_fc(atom, new_fc)
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = atom.GetIdx()

            if best_idx is None:
                break

            rw.GetAtomWithIdx(best_idx).SetFormalCharge(
                rw.GetAtomWithIdx(best_idx).GetFormalCharge() + step
            )

        out = rw.GetMol()
        out.UpdatePropertyCache(strict=False)
        return out

    @classmethod
    def finalize_sanitized(cls, mol: Mol) -> Mol:
        """Return best sanitizable candidate.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Input molecule.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            Best sanitizable molecule found by fallback strategy.
        """
        best_candidate = None
        best_objective = None

        candidate_builders = (
            lambda m: Chem.Mol(m),
            lambda m: cls._kekulized_or_none(Chem.Mol(m)),
        )

        seen_states = set()
        for builder in candidate_builders:
            try:
                candidate = builder(mol)
            except Exception:
                continue
            if candidate is None:
                continue
            candidate = _prepare_for_cleanup(candidate)
            state = _molecule_state_fingerprint(candidate)
            if state in seen_states:
                continue
            seen_states.add(state)
            sanitized = cls._sanitize_candidate_or_none(candidate)
            if sanitized is None:
                sanitized = _try_sanitize_or_none(
                    _strip_radicals_and_reassign(candidate)
                )
            if sanitized is None:
                continue
            objective = _charge_objective(sanitized)
            if (
                best_candidate is None
                or best_objective is None
                or objective < best_objective
            ):
                best_candidate = sanitized
                best_objective = objective

        if best_candidate is not None:
            forced = cls._force_charge_balance(best_candidate)
            if _target_charge_delta(forced) == 0:
                sanitized_forced = _try_sanitize_or_none(Chem.Mol(forced))
                if sanitized_forced is not None:
                    return sanitized_forced

                # Over-valence artefact from charge adjustment — demote and retry.
                forced = _repair_overvalence_and_reassign(forced)
                sanitized_forced = _try_sanitize_or_none(Chem.Mol(forced))
                if sanitized_forced is not None:
                    return sanitized_forced

            sanitized_forced = _try_sanitize_or_none(Chem.Mol(forced))
            if sanitized_forced is not None:
                return sanitized_forced

            fixed = _repair_overvalence_and_reassign(Chem.Mol(best_candidate))
            sanitized_fixed = _try_sanitize_or_none(fixed)
            if sanitized_fixed is not None:
                return sanitized_fixed
            return best_candidate

        partial = _prepare_for_cleanup(Chem.Mol(mol))
        partial = _strip_radicals_and_reassign(partial)
        partial = _repair_overvalence_and_reassign(partial)
        sanitized_partial = _try_sanitize_or_none(Chem.Mol(partial))
        if sanitized_partial is not None:
            return sanitized_partial

        fallback = _repair_overvalence_and_reassign(Chem.Mol(partial))
        sanitized_fallback = _try_sanitize_or_none(Chem.Mol(fallback))
        if sanitized_fallback is not None:
            return sanitized_fallback
        return fallback

    @classmethod
    def _path_lookahead_first_improvement(
        cls,
        mol: Mol,
        path_mode: str = "charged_sinks",
        max_active_sources: int = 8,
        max_partners: int = 16,
    ) -> Mol:
        """Try lazy path moves and accept the first improving candidate.

        Source/target pairs are prioritized by active-site penalties and each
        pair consumes generated paths one-by-one, returning immediately when an
        improving move is found.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to inspect.
        path_mode : str, default="charged_sinks"
            Source/target pairing mode.
        max_active_sources : int, default=8
            Maximum source active sites to inspect.
        max_partners : int, default=16
            Maximum partner atoms per source.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            First improving candidate, or baseline molecule.
        """
        engine = cls(
            max_iters=1,
            use_relay=False,
            use_paths=True,
            path_mode=path_mode,
        )

        base = _assign_formal_charges(Chem.Mol(mol))
        kek = cls._kekulized_or_none(base)
        if kek is not None:
            base = _assign_formal_charges(kek)
        active_atoms = cls._priority_active_atoms(base)
        if not active_atoms:
            return base

        # Bound pair exploration to highest-penalty sites first.
        active_atoms = active_atoms[:max_active_sources]

        # Build H-free graph once for all (src, dst) pair lookups in this call.
        graph = tmos.graph_mapping.mol_to_graph(base, remove_hydrogens=True)

        # 1) High-penalty atom pairs first: require both atoms to improve.
        for src in active_atoms:
            for dst in cls._priority_partner_atoms(
                base,
                src,
                include_zero=True,
                max_partners=max_partners,
            ):
                dst_penalty = cls._atom_site_penalty(base, dst)
                candidate = cls._best_path_move_between_pair(
                    engine,
                    base,
                    src,
                    dst,
                    require_dst_improve=dst_penalty > 0,
                    graph=graph,
                )
                if candidate is not None:
                    return candidate

        # 2) If active-active cannot resolve, route into preferred neutral sinks.
        sinks = cls._priority_sink_atoms(base, exclude=active_atoms)
        for src in active_atoms:
            for sink in sinks:
                if src == sink:
                    continue
                candidate = cls._best_path_move_between_pair(
                    engine,
                    base,
                    src,
                    sink,
                    require_dst_improve=False,
                    graph=graph,
                )
                if candidate is not None:
                    return candidate

        return base

    @classmethod
    def _stage_once(
        cls,
        mol: Mol,
        local_engine: "_GraphMoveEngine",
        global_engine: "_GraphMoveEngine",
    ) -> Mol:
        """Run one deterministic cleanup stage sequence.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to process.
        local_engine : _GraphMoveEngine
            Engine for local optimization.
        global_engine : _GraphMoveEngine
            Engine for global optimization.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            Molecule after one cleanup stage.
        """
        with _time_stage("fix_overvalenced"):
            mol = _fix_overvalenced(mol)
        with _time_stage("promote_underbonded"):
            mol = _promote_underbonded(mol)
        with _time_stage("assign_formal_charges"):
            mol = _assign_formal_charges(mol)
        with _time_stage("convert_radical_carbocations"):
            mol = _convert_radical_carbocations(mol)
        with _time_stage("reduce_charge_by_bond_promotion"):
            mol = _reduce_charge_by_bond_promotion(mol)
        with _time_stage("promote_atom_neighbors_greedy"):
            mol = cls._promote_atom_neighbors_greedy(mol)

        move_base = cls._kekulized_or_none(Chem.Mol(mol))
        if move_base is None:
            move_base = Chem.Mol(mol)
        move_base = _assign_formal_charges(move_base)
        has_active_sites = bool(cls._priority_active_atoms(move_base))

        if has_active_sites:
            with _time_stage("path_lookahead_charged_sinks"):
                path_candidate = cls._path_lookahead_first_improvement(
                    Chem.Mol(move_base),
                    path_mode="charged_sinks",
                    max_active_sources=1,
                    max_partners=3,
                )
            mol = cls._prefer_lower_penalty(mol, path_candidate)

        has_active_sites = bool(cls._priority_active_atoms(mol))
        with _time_stage("local_engine_optimize"):
            local_candidate = (
                local_engine.optimize(Chem.Mol(mol))
                if has_active_sites
                else Chem.Mol(mol)
            )
        mol = cls._prefer_lower_penalty(mol, local_candidate)

        has_active_sites = bool(cls._priority_active_atoms(mol))
        if has_active_sites:
            with _time_stage("path_lookahead_neg_pos"):
                negpos_path_candidate = cls._path_lookahead_first_improvement(
                    Chem.Mol(mol),
                    path_mode="neg_pos",
                    max_active_sources=1,
                    max_partners=3,
                )
            mol = cls._prefer_lower_penalty(mol, negpos_path_candidate)

        has_active_sites = bool(cls._priority_active_atoms(mol))
        with _time_stage("global_engine_optimize"):
            global_candidate = (
                global_engine.optimize(Chem.Mol(mol))
                if has_active_sites
                else Chem.Mol(mol)
            )
        mol = cls._prefer_lower_penalty(mol, global_candidate)

        with _time_stage("stage_finalize"):
            mol = _prepare_for_cleanup(mol)
            mol = _fix_overvalenced(mol)
            mol = _assign_formal_charges(mol)
        return mol

    @classmethod
    def cleanup_best(cls, mol: Mol, max_rounds: int = 8) -> Mol:
        """Run deterministic cleanup loop and return the best sanitizable molecule.

        This unifies convergence control and sanitize fallback selection so callers
        execute one cleanup entrypoint with bounded rounds and cycle checks.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            Molecule to clean.
        max_rounds : int, default=8
            Maximum cleanup rounds.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            Best sanitizable molecule found.
        """

        def _candidate_with_target_charge_or_none(candidate: Mol) -> Mol | None:
            """Return candidate only if it satisfies target charge, or can be repaired.

            Parameters
            ----------
            candidate : rdkit.Chem.rdchem.Mol
                Candidate molecule.

            Returns
            -------
            rdkit.Chem.rdchem.Mol or None
                Charge-consistent candidate when available.
            """
            target: None | int = _target_charge_or_none(candidate)
            if target is None:
                return candidate

            current: int = sum(a.GetFormalCharge() for a in candidate.GetAtoms())
            if current == target:
                try:
                    sanitized = Chem.Mol(candidate)
                    Chem.SanitizeMol(sanitized)
                    return sanitized
                except Exception:
                    pass

            repaired = cls._force_charge_balance(Chem.Mol(candidate))
            if _has_overvalenced_atoms(repaired):
                repaired = _fix_overvalenced(repaired)

            if sum(a.GetFormalCharge() for a in repaired.GetAtoms()) == target:
                try:
                    Chem.SanitizeMol(repaired)
                    return repaired
                except Exception:
                    pass

            repaired = _assign_formal_charges(repaired)

            if sum(a.GetFormalCharge() for a in repaired.GetAtoms()) == target:
                try:
                    Chem.SanitizeMol(repaired)
                    return repaired
                except Exception:
                    pass

            return _retarget_bond_orders_rdkit(candidate, target)

        def _prefer_candidate(current: Mol | None, candidate: Mol | None) -> Mol | None:
            """Return preferred candidate by objective then charge penalty.

            Parameters
            ----------
            current : rdkit.Chem.rdchem.Mol or None
                Current best candidate.
            candidate : rdkit.Chem.rdchem.Mol or None
                New candidate to compare.

            Returns
            -------
            rdkit.Chem.rdchem.Mol or None
                Preferred candidate.
            """
            if candidate is None:
                return current
            if current is None:
                return candidate

            cand_key = (
                _charge_objective(candidate),
                molecule_charge_penalty(candidate),
            )
            curr_key = (_charge_objective(current), molecule_charge_penalty(current))
            return candidate if cand_key < curr_key else current

        work = _prepare_for_cleanup(Chem.Mol(mol))

        local_engine = cls(
            max_iters=3,
            use_relay=True,
            use_paths=True,
            path_mode="charged_sinks",
        )
        global_engine = cls(
            max_iters=4,
            use_relay=False,
            use_paths=True,
            path_mode="neg_pos",
        )

        last_work_objective = _charge_objective(work)
        stagnant_rounds = 0

        seen_states = set()
        for _ in range(max_rounds):
            before = _molecule_state_fingerprint(work)
            if before in seen_states:
                break
            seen_states.add(before)

            work = cls._stage_once(work, local_engine, global_engine)
            work_objective = _charge_objective(work)

            quick = cls._sanitize_candidate_or_none(work)
            if (
                quick is not None
                and _target_charge_delta(quick) == 0
                and molecule_charge_penalty(quick) == 0
            ):
                return quick

            if (
                work_objective[0] == 0
                and work_objective[2] == 0
                and work_objective[3] == 0
            ):
                break
            if work_objective >= last_work_objective:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            last_work_objective = work_objective

            after = _molecule_state_fingerprint(work)
            if after == before or stagnant_rounds >= 1:
                break

        for _ in range(2):
            sanitized = cls._sanitize_candidate_or_none(work)
            if sanitized is not None:
                best_accept: Mol | None = None

                forced = cls._force_charge_balance(sanitized)
                if _target_charge_delta(forced) == 0:
                    try:
                        Chem.SanitizeMol(forced)
                    except Exception:
                        forced = _fix_overvalenced(forced)
                        forced = _assign_formal_charges(forced)
                        try:
                            Chem.SanitizeMol(forced)
                        except Exception:
                            pass
                    accepted = _candidate_with_target_charge_or_none(forced)
                    best_accept = _prefer_candidate(best_accept, accepted)

                try:
                    Chem.SanitizeMol(forced)
                    accepted = _candidate_with_target_charge_or_none(forced)
                    best_accept = _prefer_candidate(best_accept, accepted)
                except Exception:
                    fixed = _fix_overvalenced(Chem.Mol(sanitized))
                    fixed = _assign_formal_charges(fixed)
                    try:
                        Chem.SanitizeMol(fixed)
                        accepted = _candidate_with_target_charge_or_none(fixed)
                        best_accept = _prefer_candidate(best_accept, accepted)
                    except Exception:
                        accepted = _candidate_with_target_charge_or_none(sanitized)
                        best_accept = _prefer_candidate(best_accept, accepted)

                if best_accept is not None:
                    return best_accept

            next_work = cls._stage_once(Chem.Mol(work), local_engine, global_engine)
            if _molecule_state_fingerprint(next_work) == _molecule_state_fingerprint(
                work
            ):
                break

            if _charge_objective(next_work) <= _charge_objective(work):
                work = next_work
            else:
                break

        final = cls.finalize_sanitized(work)
        accepted = _candidate_with_target_charge_or_none(final)
        if accepted is not None:
            return accepted

        target = _target_charge_or_none(final)
        if target is not None:
            for seed in (work, mol):
                retry = _retarget_bond_orders_rdkit(Chem.Mol(seed), target)
                if retry is not None:
                    return retry

                forced = cls._force_charge_balance(Chem.Mol(seed))
                forced = _assign_formal_charges(forced)
                if sum(a.GetFormalCharge() for a in forced.GetAtoms()) != target:
                    continue
                try:
                    Chem.SanitizeMol(forced)
                    return forced
                except Exception:
                    continue

        return final


def _score_structure_for_cleanup(candidate_struct: Mol) -> Objective:
    """Return cleanup objective after lightweight retyping on a trial structure."""
    probe = Chem.Mol(candidate_struct)
    probe.UpdatePropertyCache(strict=False)
    probe = _assign_formal_charges(probe)
    probe = _reduce_charge_by_bond_promotion(probe)
    return _charge_objective(probe)


def _secondary_cleanup_features(
    mol: Mol,
) -> tuple[dict[int, int], dict[int, int], dict[int, bool]]:
    """Return per-atom features used by secondary overvalence cleanup.

    Returns
    -------
    tuple
        ``(heavy_degree, oxygen_neighbor_total, has_hetero_neighbor)`` maps.
    """
    heavy_degree: dict[int, int] = {}
    oxygen_total: dict[int, int] = {}
    has_hetero_neighbor: dict[int, bool] = {}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        nbrs = list(atom.GetNeighbors())
        heavy_degree[idx] = sum(1 for nbr in nbrs if nbr.GetAtomicNum() != 1)
        oxygen_total[idx] = sum(1 for nbr in nbrs if nbr.GetAtomicNum() == 8)
        has_hetero_neighbor[idx] = any(nbr.GetAtomicNum() not in (1, 6) for nbr in nbrs)

    return heavy_degree, oxygen_total, has_hetero_neighbor


def _is_secondary_cleanup_candidate(ai: Atom, aj: Atom, bond: Bond) -> bool:
    """Return ``True`` if this heavy-atom single bond is eligible for pruning."""
    if ai.GetAtomicNum() == 1 or aj.GetAtomicNum() == 1:
        return False
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False

    z_i = ai.GetAtomicNum()
    z_j = aj.GetAtomicNum()

    # Preserve most ring bonds unless chemistry suggests a likely closure artefact.
    if bond.IsInRing() and not ({z_i, z_j} & {15, 16}) and not (z_i == 6 and z_j == 6):
        return False

    return True


def _secondary_cleanup_stretch_cutoff(
    z_i: int,
    z_j: int,
    bond_in_ring: bool,
    hetero_nbr_i: bool,
    hetero_nbr_j: bool,
    oxygen_i_excl: int,
    oxygen_j_excl: int,
    heavy_i_excl: int,
    heavy_j_excl: int,
) -> tuple[float, bool, bool]:
    """Return adaptive stretch cutoff and bridge flags for one bond candidate."""
    stretch_cutoff = 0.10
    oxygen_rich_ps_bridge = False
    hypercoord_pp_bridge = False

    if z_i == 6 and z_j == 6:
        stretch_cutoff = 0.16
        if bond_in_ring and hetero_nbr_i and hetero_nbr_j:
            stretch_cutoff = 0.12

    if {z_i, z_j}.issubset({15, 16}) and oxygen_i_excl >= 2 and oxygen_j_excl >= 2:
        oxygen_rich_ps_bridge = True
        stretch_cutoff = 0.00

    if z_i == 15 and z_j == 15 and heavy_i_excl >= 3 and heavy_j_excl >= 3:
        hypercoord_pp_bridge = True
        stretch_cutoff = min(stretch_cutoff, 0.03)

    return stretch_cutoff, oxygen_rich_ps_bridge, hypercoord_pp_bridge


def _collect_secondary_stretched_candidates(
    mol: Mol,
    heavy_degree: dict[int, int],
    oxygen_total: dict[int, int],
    has_hetero_nbr: dict[int, bool],
    conf,
) -> list[tuple[float, int, int, float, bool, bool]]:
    """Collect stretched heavy-atom single-bond candidates for secondary cleanup."""
    stretched: list[tuple[float, int, int, float, bool, bool]] = []
    if conf is None:
        return stretched

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ai = mol.GetAtomWithIdx(i)
        aj = mol.GetAtomWithIdx(j)
        if not _is_secondary_cleanup_candidate(ai, aj, bond):
            continue

        if heavy_degree[i] < 2 or heavy_degree[j] < 2:
            continue

        z_i = ai.GetAtomicNum()
        z_j = aj.GetAtomicNum()

        dist = _bond_distance(conf, i, j)
        rsum = float(pt.GetRcovalent(z_i) + pt.GetRcovalent(z_j))
        stretch = dist - rsum

        oxygen_i_excl = oxygen_total[i] - (1 if z_j == 8 else 0)
        oxygen_j_excl = oxygen_total[j] - (1 if z_i == 8 else 0)
        heavy_i_excl = heavy_degree[i] - (1 if z_j != 1 else 0)
        heavy_j_excl = heavy_degree[j] - (1 if z_i != 1 else 0)

        stretch_cutoff, oxygen_rich_ps_bridge, hypercoord_pp_bridge = (
            _secondary_cleanup_stretch_cutoff(
                z_i,
                z_j,
                bond.IsInRing(),
                has_hetero_nbr[i],
                has_hetero_nbr[j],
                oxygen_i_excl,
                oxygen_j_excl,
                heavy_i_excl,
                heavy_j_excl,
            )
        )
        if stretch <= stretch_cutoff:
            continue

        stretched.append(
            (stretch, i, j, dist, oxygen_rich_ps_bridge, hypercoord_pp_bridge)
        )

    return stretched


def _evaluate_secondary_bond_removal(
    mol: Mol,
    idx_a: int,
    idx_b: int,
    dist: float,
    best_obj: Objective,
    oxygen_rich_ps_bridge: bool,
    hypercoord_pp_bridge: bool,
) -> tuple[Objective, float, int, int] | None:
    """Return accepted trial update for one secondary bond-removal candidate."""
    trial = _trial_after_bond_removal(mol, idx_a, idx_b)

    # Skip removals that immediately create an obviously over-valenced
    # structure under temporary charge assignment.
    probe = _assign_formal_charges(Chem.Mol(trial))
    if _has_overvalenced_atoms(probe):
        return None

    trial_obj = _score_structure_for_cleanup(trial)
    keep_if_non_worse = (
        oxygen_rich_ps_bridge or hypercoord_pp_bridge
    ) and trial_obj <= best_obj
    if trial_obj < best_obj or keep_if_non_worse:
        return (trial_obj, dist, idx_a, idx_b)

    return None


def _fix_overvalenced(mol: Mol) -> Mol:
    """Demote or remove bonds so each atom stays within allowed charge/valence.

    Cleanup works on explicit bond orders (single/double/triple), with aromatic
    bonds removed upstream by kekulization.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to repair.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Molecule after over-valence corrections.
    """
    mol = Chem.RWMol(mol)
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None

    def _geom_longest(bonds: list[Bond], atom_idx: int) -> Bond:
        """Return furthest bond endpoint from the center atom.

        Parameters
        ----------
        bonds : list of rdkit.Chem.rdchem.Bond
            Candidate bonds attached to ``atom_idx``.
        atom_idx : int
            Index of the central atom.

        Returns
        -------
        rdkit.Chem.rdchem.Bond
            Selected bond. If no conformer exists, returns the first bond.
        """
        if conf is None:
            return bonds[0]
        p0 = conf.GetAtomPosition(atom_idx)
        p0 = np.array([p0.x, p0.y, p0.z])
        return max(
            bonds,
            key=lambda b: np.linalg.norm(
                np.array(
                    [
                        conf.GetAtomPosition(b.GetOtherAtomIdx(atom_idx)).x,
                        conf.GetAtomPosition(b.GetOtherAtomIdx(atom_idx)).y,
                        conf.GetAtomPosition(b.GetOtherAtomIdx(atom_idx)).z,
                    ]
                )
                - p0
            ),
        )

    changed = True
    while changed:
        changed = False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                continue
            valences = _charge_adjusted_valences(atom)
            if not valences:
                continue
            int_bv = _bond_valence(atom, integer=True)
            max_fc: int = _MAX_BOND_FC.get(atom.GetAtomicNum(), _DEFAULT_MAX_BOND_FC)
            if int_bv - max(valences) <= max_fc:
                continue  # within acceptable formal-charge range

            idx = atom.GetIdx()

            # --- Priority 1: higher-order bonds ---
            # Demote the longest double/triple bond by one order.
            # Physical basis: longer bonds are weaker and more likely to be an
            # OBabel over-assignment (e.g. second N=O in a nitro group).
            hi_order = [b for b in atom.GetBonds() if int(b.GetBondTypeAsDouble()) >= 2]
            if hi_order:
                longest = _geom_longest(hi_order, idx)
                new_order: int = int(longest.GetBondTypeAsDouble()) - 1
                longest.SetBondType(
                    Chem.BondType.SINGLE if new_order == 1 else Chem.BondType.DOUBLE
                )
                mol.UpdatePropertyCache(strict=False)
                changed = True
                break

            # --- Priority 2: remove longest single bond (connectivity error) ---
            # All bonds are single; the extra bond
            # is a pure OBabel ConnectTheDots artefact.  Remove the longest one
            # (furthest neighbour = least likely to be a real covalent bond).
            single = [b for b in atom.GetBonds()]
            if single and conf is not None:
                filtered = single
                if atom.GetAtomicNum() in (15, 16):
                    filtered = [
                        b
                        for b in single
                        if b.GetOtherAtom(atom).GetAtomicNum() not in (7, 8, 15, 16)
                    ]
                if not filtered:
                    filtered = single
                longest = _geom_longest(filtered, idx)
                mol.RemoveBond(longest.GetBeginAtomIdx(), longest.GetEndAtomIdx())
                mol.UpdatePropertyCache(strict=False)
                changed = True
                break

            # No bonds to modify — give up on atom
            continue
        # end for-atom
    # end while-changed

    mol.UpdatePropertyCache(strict=False)

    # Secondary bounded pass: remove only stretched heavy-atom single bonds when
    # doing so improves the downstream chemistry objective after charge/bond
    # reassignment.
    best_struct = mol.GetMol()
    best_struct.UpdatePropertyCache(strict=False)

    best_obj = _score_structure_for_cleanup(best_struct)

    for _ in range(2):
        heavy_degree, oxygen_total, has_hetero_nbr = _secondary_cleanup_features(
            best_struct
        )
        conf_best = (
            best_struct.GetConformer() if best_struct.GetNumConformers() > 0 else None
        )
        candidate_updates: list[tuple[Objective, float, int, int]] = []

        # Gather candidate stretched single bonds and evaluate only the most
        # suspicious ones for speed.
        stretched = _collect_secondary_stretched_candidates(
            best_struct,
            heavy_degree,
            oxygen_total,
            has_hetero_nbr,
            conf_best,
        )

        if not stretched:
            break

        stretched.sort(reverse=True)
        for _, i, j, dist, oxygen_rich_ps_bridge, hypercoord_pp_bridge in stretched[:8]:
            evaluated = _evaluate_secondary_bond_removal(
                best_struct,
                i,
                j,
                dist,
                best_obj,
                oxygen_rich_ps_bridge,
                hypercoord_pp_bridge,
            )
            if evaluated is not None:
                candidate_updates.append(evaluated)

        if not candidate_updates:
            break

        # Prefer best objective first, then longer bond as weakest-link tie-breaker.
        candidate_updates.sort(key=lambda x: (x[0], -x[1]))
        chosen_obj, _, idx_a, idx_b = candidate_updates[0]
        best_struct = _trial_after_bond_removal(best_struct, idx_a, idx_b)
        best_obj = chosen_obj

    best_struct.UpdatePropertyCache(strict=False)
    return best_struct


def _promote_underbonded(mol: Mol) -> Mol:
    """Promote non-aromatic bonds between atom pairs that are both under-bonded.

    An atom is under-bonded when its effective valence (float bv − fc) is below
    the lowest valid shell it can reach — SanitizeMol would assign radical
    electrons.  Promoting the shared bond simultaneously reduces the deficit for
    both atoms.

    Must be called before ``_assign_formal_charges`` so that atoms not yet
    carrying a formal charge are not mistakenly flagged as satisfied.  For
    example, a ring N with only 2 explicit bonds has bv=2, fc=0, ev=2 < 3
    (lowest valid) → correctly flagged as under-bonded, and its bond to the
    adjacent under-bonded ring C is promoted to a double bond.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with explicit bond orders.

    Returns
    -------
    rdkit.Chem.Mol
        Molecule after local bond-order promotions.
    """

    def _unused(atom: Atom) -> float | int:
        """Return valence deficit used for local promotion eligibility.

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            Atom to evaluate.

        Returns
        -------
        float or int
            Deficit magnitude.
        """
        if atom.GetAtomicNum() == 0:
            return 0
        return _valence_deficit(atom)

    mol = Chem.RWMol(mol)
    changed = True
    while changed:
        changed = False
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
                continue
            if int(bond.GetBondTypeAsDouble()) == 1 and (
                (
                    a1.GetAtomicNum() in (15, 16)
                    and a2.GetAtomicNum() == 8
                    and a2.GetFormalCharge() == 0
                )
                or (
                    a2.GetAtomicNum() in (15, 16)
                    and a1.GetAtomicNum() == 8
                    and a1.GetFormalCharge() == 0
                )
            ):
                # Leave neutral S/P–O single bonds to the targeted charge-promotion pass.
                continue
            unused1: float | int = _unused(a1)
            unused2: float | int = _unused(a2)
            promote_ok: bool = (unused1 >= 1 and unused2 >= 1) or (
                (unused1 >= 1 and unused2 >= 0.5) or (unused2 >= 1 and unused1 >= 0.5)
            )
            if promote_ok:
                new_order: int = int(bond.GetBondTypeAsDouble()) + 1
                if new_order not in (2, 3):
                    continue
                # Avoid over-promoting beyond double when only one side is weakly under-bonded.
                if new_order == 3 and (unused1 < 1 or unused2 < 1):
                    continue
                bond.SetBondType(
                    Chem.BondType.DOUBLE if new_order == 2 else Chem.BondType.TRIPLE
                )
                changed = True
                break  # restart so _unused is recomputed from the updated graph
    mol.UpdatePropertyCache(strict=False)
    return mol.GetMol()


def _assign_formal_charges(mol: Mol) -> Mol:
    """Assign formal charges from float bond valence.

    For each heavy atom the valid valence shell closest to the float bond
    valence is chosen via :func:`_best_shell`, which breaks ties by
    electronegativity / proton-affinity:

    - N, P  → prefer the *lower* shell on a tie → fc = +1  (ammonium, nitro N⁺)
    - O, S, halogens, C … → prefer the *higher* shell → fc = −1  (sulfonate O⁻)

    Uses explicit bond orders from the cleanup graph (single/double/triple).

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with explicit bond orders.

    Returns
    -------
    rdkit.Chem.Mol
        Molecule with formal charges assigned.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        if atom.GetAtomicNum() == 1:
            atom.SetFormalCharge(0)
            continue
        bv = _bond_valence(atom)
        best_fc = _best_formal_charge(atom, bv)
        if best_fc is None:
            continue
        atom.SetFormalCharge(best_fc)

        # Special-case tricoordinate carbocations (e.g., CCl3+):
        # if carbon is 3-connected, all bonds are single, and the default
        # assignment produced fc=0, shift to fc=+1 to avoid artificial
        # under-valence forcing extra double bonds.
        if (
            atom.GetAtomicNum() == 6
            and atom.GetFormalCharge() == 0
            and abs(bv - 3.0) < 0.2
            and all(int(b.GetBondTypeAsDouble()) == 1 for b in atom.GetBonds())
            and atom.GetTotalNumHs() == 0
        ):
            atom.SetFormalCharge(1)

        if atom.GetNumRadicalElectrons() > 0:
            implied_shell = int(round(bv)) - atom.GetFormalCharge()
            if implied_shell in _valid_valences(atom):
                atom.SetNumRadicalElectrons(0)

    mol = _enforce_target_charge(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol


def _assign_formal_charges_local(mol: Mol, atom_indices: Iterable[int]) -> Mol:
    """Reassign formal charges only for *atom_indices*.

    Equivalent to :func:`_assign_formal_charges` restricted to the given set
    of atom indices.  For a path move only the path atoms have changed bond
    orders, so there is no need to re-derive charges for the rest of the
    molecule.  ``_enforce_target_charge`` is still called globally at the end
    because net-charge balancing may touch any atom.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with explicit bond orders (modified in-place).
    atom_indices : iterable of int
        Atom indices to update.

    Returns
    -------
    rdkit.Chem.Mol
    """
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 0:
            continue
        if atom.GetAtomicNum() == 1:
            atom.SetFormalCharge(0)
            continue
        bv = _bond_valence(atom)
        best_fc = _best_formal_charge(atom, bv)
        if best_fc is None:
            continue
        atom.SetFormalCharge(best_fc)
        # Tricoordinate carbocation special case (mirrors _assign_formal_charges).
        if (
            atom.GetAtomicNum() == 6
            and atom.GetFormalCharge() == 0
            and abs(bv - 3.0) < 0.2
            and all(int(b.GetBondTypeAsDouble()) == 1 for b in atom.GetBonds())
            and atom.GetTotalNumHs() == 0
        ):
            atom.SetFormalCharge(1)

    mol = _enforce_target_charge(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol


def _reduce_charge_by_bond_promotion(mol: Mol) -> Mol:
    """Absorb spurious negative charges into higher-order bonds.

    After ``_assign_formal_charges``, atoms such as S and P may sit at a lower
    valid valence (e.g. S at valence 4) while adjacent O or N atoms carry
    negative formal charges that arose because OpenBabel assigned single bonds
    where double bonds are chemically correct.  Classic examples:

    - Sulfonate:  S([O-])([O-])[O-]  →  S(=O)(=O)[O-]   (S4 → S6)
    - Sulfonyl:   S([O-])([O-])      →  S(=O)(=O)        (S4 → S6)
    - Sulfinyl:   S([O-])(...)       →  S(=O)(...)        (S2 → S4)
    - Phosphonate: P([O-])([O-])[O-] →  P(=O)(...)

    For each atom A whose current integer bond valence (bv) is below one of
    its valid neutral valence shells v', the function:

    1. Collects all non-aromatic single bonds from A to negatively charged
       neighbours B (fc_B < 0).
    2. Finds the smallest v' > bv reachable by promoting exactly (v' − bv)
       of those bonds.
    3. Promotes those bonds from single to double, which simultaneously
       raises B's bond valence (neutralising the negative charge) and moves
       A to the valid shell v' with fc = 0.

    Only promotions that result in A having fc >= 0 are accepted (we never
    make A more negative).  The promotion that minimises total |formal charge|
    across the molecule is preferred; ties are broken by choosing the shortest
    bonds (best geometry).

    Formal charges are re-derived by ``_assign_formal_charges`` at the end.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with assigned formal charges.

    Returns
    -------
    rdkit.Chem.Mol
        Updated molecule after applying the best improving local moves.
    """
    rw = Chem.RWMol(mol)
    conf = rw.GetConformer() if rw.GetNumConformers() > 0 else None

    def _local_atom_penalty(m: Mol, atom_indices: set[int]) -> int:
        """Compute summed atom-charge penalties for a local atom subset.

        Parameters
        ----------
        m : rdkit.Chem.rdchem.Mol
            Molecule to score.
        atom_indices : set of int
            Atom indices in the local region.

        Returns
        -------
        int
            Sum of local atom penalties.
        """
        total = 0
        for atom_idx in atom_indices:
            atom = m.GetAtomWithIdx(atom_idx)
            total += _atom_charge_penalty(atom.GetAtomicNum(), atom.GetFormalCharge())
        return total

    def _sort_key_for_neighbor(atom: Atom, center_idx: int) -> tuple[int, int, float]:
        """Return ranking key for selecting promotion neighbors.

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            Neighbor atom candidate.
        center_idx : int
            Center atom index.

        Returns
        -------
        tuple of (int, int, float)
            Sorting key ``(is_non_negative, element_priority, distance)``.
        """
        ep: int = _NEG_NBOR_PRIORITY.get(atom.GetAtomicNum(), 99)
        is_negative: int = 0 if atom.GetFormalCharge() < 0 else 1
        if conf is None:
            return (is_negative, ep, 0.0)
        dist = _bond_distance(conf, center_idx, atom.GetIdx())
        return (is_negative, ep, dist)

    changed = True
    while changed:
        changed = False
        base = _assign_formal_charges(rw.GetMol())
        base_obj = _charge_objective(base)
        base_penalty = molecule_charge_penalty(base)

        best_candidate = None
        best_score = None

        for atom in base.GetAtoms():
            if atom.GetAtomicNum() in (0, 1):
                continue

            atom_idx = atom.GetIdx()
            valences = _charge_adjusted_valences(atom)
            if not valences:
                continue

            bv_int = _bond_valence(atom, integer=True)
            max_fc_A: int = _MAX_BOND_FC.get(atom.GetAtomicNum(), _DEFAULT_MAX_BOND_FC)

            candidate_single_bonds = []
            for b in atom.GetBonds():
                if int(b.GetBondTypeAsDouble()) != 1:
                    continue
                other = b.GetOtherAtom(atom)
                if other.GetAtomicNum() == 1:
                    continue
                # Classic path: negative neighbours.
                if other.GetFormalCharge() < 0:
                    candidate_single_bonds.append(b)
                    continue
                # Allow neutral O as a sink when center is under-valent S/P.
                if (
                    atom.GetAtomicNum() in (15, 16)
                    and other.GetAtomicNum() == 8
                    and other.GetFormalCharge() == 0
                    and bv_int < min(valences)
                ):
                    candidate_single_bonds.append(b)

            candidate_single_bonds = sorted(
                candidate_single_bonds,
                key=lambda b: _sort_key_for_neighbor(b.GetOtherAtom(atom), atom_idx),
            )

            # Promotion candidates: A-B(1) -> A=B (prefer negatives, then O neighbors)
            for n_promote in range(1, len(candidate_single_bonds) + 1):
                if bv_int + n_promote > max(valences) + max_fc_A:
                    break
                bundle = candidate_single_bonds[:n_promote]

                trial_rw = Chem.RWMol(base)
                affected = {atom_idx}
                for b in bundle:
                    tb = trial_rw.GetBondBetweenAtoms(
                        b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    )
                    if tb is None:
                        continue
                    tb.SetBondType(Chem.BondType.DOUBLE)
                    affected.add(tb.GetBeginAtomIdx())
                    affected.add(tb.GetEndAtomIdx())
                trial_rw.UpdatePropertyCache(strict=False)
                candidate = _assign_formal_charges(trial_rw.GetMol())

                cand_obj = _charge_objective(candidate)
                cand_penalty = molecule_charge_penalty(candidate)
                base_local = _local_atom_penalty(base, affected)
                cand_local = _local_atom_penalty(candidate, affected)

                score = (cand_obj, cand_penalty, cand_local, len(bundle))
                base_score = (base_obj, base_penalty, base_local, len(bundle))
                if score >= base_score:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = candidate

            # Demotion candidates: A=B -> A-B, accepted only if atom-penalty improves.
            demotable = [
                b
                for b in atom.GetBonds()
                if int(b.GetBondTypeAsDouble()) == 2
                and b.GetOtherAtom(atom).GetAtomicNum() != 1
                and b.GetOtherAtom(atom).GetAtomicNum() in (7, 8, 15, 16)
            ]
            for bond in demotable:
                trial_rw = Chem.RWMol(base)
                tb = trial_rw.GetBondBetweenAtoms(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                )
                if tb is None:
                    continue
                tb.SetBondType(Chem.BondType.SINGLE)
                trial_rw.UpdatePropertyCache(strict=False)
                candidate = _assign_formal_charges(trial_rw.GetMol())

                affected = {
                    atom_idx,
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                }
                cand_obj = _charge_objective(candidate)
                cand_penalty = molecule_charge_penalty(candidate)
                base_local = _local_atom_penalty(base, affected)
                cand_local = _local_atom_penalty(candidate, affected)

                score = (cand_obj, cand_penalty, cand_local, 1)
                base_score = (base_obj, base_penalty, base_local, 1)
                if score >= base_score:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = candidate

        if best_candidate is None:
            break

        rw = Chem.RWMol(best_candidate)
        rw.UpdatePropertyCache(strict=False)
        changed = True

    mol = rw.GetMol()
    mol = _assign_formal_charges(mol)
    return mol


def _prepare_for_cleanup(mol: Mol) -> Mol:
    """Prepare explicit-bond-order graph for cleanup passes.

    Aromatic flags are cleared so optimization always works with single/double
    (and occasional triple) bonds only.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to prepare.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Kekulized-or-original molecule with updated cache.
    """
    candidate = Chem.Mol(mol)
    try:
        Chem.Kekulize(candidate, clearAromaticFlags=True)
    except Exception:
        pass
    candidate.UpdatePropertyCache(strict=False)
    return candidate


def _molecule_state_fingerprint(
    mol: Mol,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Return a hashable molecule state fingerprint.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to encode.

    Returns
    -------
    tuple
        ``(atom_state, bond_state)`` where each state is immutable.
    """
    atom_state = tuple(
        (atom.GetAtomicNum(), atom.GetFormalCharge()) for atom in mol.GetAtoms()
    )
    bond_state = tuple(
        sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                int(round(b.GetBondTypeAsDouble() * 2)),
            )
            for b in mol.GetBonds()
        )
    )
    return (atom_state, bond_state)


def _restore_explicit_hydrogen_flags(mol: Mol) -> Mol:
    """Restore explicit-hydrogen flags after MOL-block round trip.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to modify.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Molecule with ``NoImplicit=True`` and explicit-H count reset.
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    return rw.GetMol()


def _retarget_bond_orders_rdkit(mol: Mol, charge: int) -> Mol | None:
    """Attempt a target-charge bond-order re-perception with RDKit.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Input molecule with connectivity.
    charge : int
        Target net charge.

    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        Sanitizable, target-charge-consistent molecule when successful.
    """
    try:
        candidate = Chem.Mol(mol)
        DetermineBondOrders(
            candidate,
            charge=int(charge),
            maxIterations=800,
            allowChargedFragments=True,
        )
        candidate.UpdatePropertyCache(strict=False)
        candidate = _prepare_for_cleanup(candidate)
        candidate = _fix_overvalenced(candidate)
        candidate = _assign_formal_charges(candidate)
        if sum(a.GetFormalCharge() for a in candidate.GetAtoms()) != int(charge):
            return None
        Chem.SanitizeMol(candidate)
        return candidate
    except Exception:
        return None


def _initial_bonding_rdkit(mol: Mol, charge: int | None = None) -> Mol:
    """Assign connectivity and bond orders using RDKit native perception.

    If provided charge is None, a charge of 0 is assumed.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Input molecule.
    charge : int or None, default=None
        Optional target net charge.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Molecule with inferred connectivity and bond orders.
    """
    if charge is None:
        charge = 0
    DetermineBondOrders(
        mol,
        charge=charge,
        maxIterations=1000,
        allowChargedFragments=True,
    )
    return mol


def _initial_bonding_openbabel(mol: Mol, charge: int | None = None) -> Mol:
    """Assign connectivity and bond orders with OpenBabel.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Input molecule.
    charge : int or None, default=None
        Optional target net charge.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        Molecule with inferred bonds and target charge metadata.
    """
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("mol", "mol")
    ob_mol = ob.OBMol()
    ob_conversion.ReadString(ob_mol, Chem.MolToMolBlock(mol))
    if charge is not None:
        ob_mol.SetTotalCharge(charge)
    for _ in range(5):
        ob_mol.PerceiveBondOrders()
        ob_mol.FindRingAtomsAndBonds()
        ob.OBAromaticTyper().AssignAromaticFlags(ob_mol)
    ob_mol.PerceiveBondOrders()

    out = Chem.MolFromMolBlock(
        ob_conversion.WriteString(ob_mol),
        sanitize=False,
        removeHs=False,
    )
    if charge is not None:
        out.SetProp("_target_charge", str(int(charge)))
    return _restore_explicit_hydrogen_flags(out)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def determine_bonds(
    mol: Mol,
    charge: int | None = None,
    method: str = "openbabel",
    custom_cleanup: bool = True,
    cleanup_max_iters: int = 10,
) -> Mol:
    """Assign connectivity, bond orders, and formal charges.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with atoms and a conformer but no bonds.
    charge : int or None, default=None
        Net molecular charge. When provided, final formal charges are forced
        to match this total.
    method : str, optional
        Backend for initial bond/order detection. Supported values are
        ``"rdkit"`` and ``"openbabel"``.
    custom_cleanup : bool, optional
        Run charge/bond cleanup with deterministic move engines.
    cleanup_max_iters : int, optional
        Maximum cleanup rounds.

    Returns
    -------
    rdkit.Chem.Mol
        Sanitized molecule.

    Raises
    ------
    ValueError
        If ``method`` is not one of {"rdkit", "openbabel"}, or if the calculated
        formal charge does not match ``charge`` after cleanup.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from tmos._rdkit_bond_typing import determine_bonds
    >>> xyz = "3\\n\\nO 0 0 0\\nH 0 0 1\\nH 0 1 0\\n"
    >>> mol = Chem.MolFromXYZBlock(xyz)
    >>> out = determine_bonds(mol, charge=0, method="openbabel")
    >>> Chem.GetFormalCharge(out)
    0
    """
    mol.UpdatePropertyCache(strict=False)
    if charge is not None:
        mol.SetProp("_target_charge", str(int(charge)))

    backend = {
        "rdkit": _initial_bonding_rdkit,
        "openbabel": _initial_bonding_openbabel,
    }.get(method)

    if backend is None:
        raise ValueError(
            f"method={method!r} is not supported; choose 'rdkit' or 'openbabel'."
        )

    with _time_stage("initial_bonding"):
        mol = backend(mol, charge)
    mol.UpdatePropertyCache(strict=False)
    initial_bonded = Chem.Mol(mol)

    if custom_cleanup:
        with _time_stage("cleanup_best"):
            mol = _GraphMoveEngine.cleanup_best(mol, max_rounds=cleanup_max_iters)
        # If cleanup converges to a charged/charge-mismatched local minimum,
        # try a bounded RDKit re-perception from both current and initial seeds
        # and keep only objective-improving candidates.
        target_for_retarget = (
            int(charge)
            if charge is not None
            else sum(a.GetFormalCharge() for a in mol.GetAtoms())
        )
        needs_retarget = molecule_charge_penalty(mol) > 0 or (
            charge is not None
            and sum(a.GetFormalCharge() for a in mol.GetAtoms()) != int(charge)
        )
        if needs_retarget:
            with _time_stage("retarget_after_cleanup"):
                best = Chem.Mol(mol)
                best_obj = _charge_objective(best)
                for seed in (Chem.Mol(mol), Chem.Mol(initial_bonded)):
                    trial = _retarget_bond_orders_rdkit(seed, target_for_retarget)
                    if trial is None:
                        continue
                    if charge is not None and (
                        sum(a.GetFormalCharge() for a in trial.GetAtoms())
                        != int(charge)
                    ):
                        continue
                    trial_obj = _charge_objective(trial)
                    if trial_obj < best_obj:
                        best = trial
                        best_obj = trial_obj
                mol = best
        if (
            charge is not None
            and sum([a.GetFormalCharge() for a in mol.GetAtoms()]) != charge
        ):
            raise ValueError("Inconsistent charge with target!")

    Chem.SanitizeMol(mol)

    return mol
