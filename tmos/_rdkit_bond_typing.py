"""Utilities for building and sanitizing RDKit molecules from 3D geometries (e.g., XYZ).

This module provides routines to infer molecular connectivity, bond orders, and formal charges
from atom positions using a combination of RDKit and OpenBabel perception. It includes
custom cleanup logic to resolve common over-valence/charge assignment artifacts.

Public API
----------
- ``determine_bonds``: assign connectivity, bond orders, and formal charges.
- ``molecule_charge_penalty``: score formal-charge placement for optimization.
"""

import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdDetermineBonds import DetermineBondOrders
from openbabel import openbabel as ob

import tmos

# Suppress OpenBabel logging
ob.obErrorLog.SetOutputLevel(0)
ob.obErrorLog.StopLogging()
os.environ["BABEL_SILENCE"] = "1"

pt = Chem.GetPeriodicTable()

# Maximum additional positive formal charge tolerated in over-valence checks.
_MAX_BOND_FC = {
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
_PREFER_LOWER_SHELL = frozenset({7, 15})  # N, P

# Priority for selecting negatively charged neighbors for bond promotion.
_NEG_NBOR_PRIORITY = {8: 0, 16: 1, 7: 2, 15: 3}  # O, S, N, P; rest = 99

# Element-wise charge preference penalties (smaller = preferred charge placement).
# Goal: minimise the number of charged atoms while preferring negative charge on
# O/S and positive charge on N/P, and strongly disfavoring [N-], [C+], [C-], O+.
_NEG_CHARGE_PENALTY = {
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
_POS_CHARGE_PENALTY = {
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _best_shell(atom, bv):
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


def _valid_valences(atom):
    """Sorted list of valid neutral valence values for *atom* (periodic table)."""
    return sorted(v for v in pt.GetValenceList(atom.GetAtomicNum()) if v >= 0)


def _bond_valence(atom, integer=False):
    """Sum of bond orders for *atom*, including explicit Hs.

    ``integer=False`` (default): aromatic bonds contribute 1.5 — the float
    bond valence RDKit uses internally.
    ``integer=True``: each bond is truncated to its integer order, matching
    the over-valence check that SanitizeMol performs.
    """
    bv = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + atom.GetNumExplicitHs()
    return int(bv) if integer else bv


def _charge_adjusted_valences(atom):
    """Valid valence shells with carbon cations treated as 3-coordinate."""
    valences = _valid_valences(atom)
    if atom.GetAtomicNum() == 6 and atom.GetFormalCharge() > 0:
        valences = [v for v in valences if v <= 3]
        if 3 not in valences:
            valences.append(3)
    return sorted(set(valences))


def _effective_valence(atom):
    """Effective valence for deficit checks (ignores positive charge on C)."""
    eff_fc = atom.GetFormalCharge()
    if atom.GetAtomicNum() == 6 and eff_fc > 0:
        eff_fc = 0
    return _bond_valence(atom) - eff_fc


def _valence_deficit(atom):
    """Positive deficit to the nearest valid shell for deficit accounting."""
    valences = _charge_adjusted_valences(atom)
    if not valences:
        return 0.0
    ev = _effective_valence(atom)
    target = next((v for v in valences if v >= ev), None)
    if target is None:
        return 0.0
    return max(0.0, target - ev)


def _max_allowed_int_bv(atom):
    """Maximum integer bond valence allowed for *atom* at its current formal charge."""
    return _max_allowed_int_bv_at_fc(atom, atom.GetFormalCharge())


def _max_allowed_int_bv_at_fc(atom, proposed_fc):
    """Maximum integer bond valence allowed for *atom* at a *proposed* formal charge.

    Unlike :func:`_max_allowed_int_bv`, this does not read the atom's current
    formal charge — it evaluates the cap hypothetically at ``proposed_fc``.
    Used by :meth:`_GraphMoveEngine._force_charge_balance` to prevent assigning a
    formal charge that would leave the atom over-valenced (e.g. C⁺ with bv=4).
    """
    atomic_num = atom.GetAtomicNum()
    if atomic_num == 1:
        return 1
    valences = _valid_valences(atom)
    # Mirror _charge_adjusted_valences: C cations may only reach valence 3.
    if atomic_num == 6 and proposed_fc > 0:
        valences = [v for v in valences if v <= 3]
        if not valences:
            valences = [3]
    if not valences:
        return 99
    max_fc_tol = _MAX_BOND_FC.get(atomic_num, _DEFAULT_MAX_BOND_FC)
    return max(valences) + max_fc_tol


def molecule_charge_penalty(mol):
    """Return element-aware penalty for charged atom states in *mol*.

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


def _atom_charge_penalty(atomic_num, formal_charge):
    """Element-aware penalty for a single atom formal charge assignment."""
    if formal_charge == 0:
        return 0
    score = 2 + abs(formal_charge)
    if formal_charge < 0:
        score += _NEG_CHARGE_PENALTY.get(atomic_num, 6)
    else:
        score += _POS_CHARGE_PENALTY.get(atomic_num, 6)
    return score


def _target_charge_or_none(mol):
    """Return integer target charge from ``_target_charge`` if present."""
    if not mol.HasProp("_target_charge"):
        return None
    try:
        return int(mol.GetProp("_target_charge"))
    except Exception:
        return None


def _target_charge_delta(mol):
    """Return signed delta: current formal charge minus target charge."""
    target = _target_charge_or_none(mol)
    if target is None:
        return 0
    current = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return current - target


def _enforce_target_charge(mol):
    """Adjust formal charges to match molecule property ``_target_charge``.

    Uses one-charge-unit moves only when the destination charge is compatible
    with one of the atom's valid valence shells at current bond valence.
    """
    target_charge = _target_charge_or_none(mol)
    if target_charge is None:
        return mol

    for _ in range(128):
        current_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        delta = target_charge - current_charge
        if delta == 0:
            break

        step = 1 if delta > 0 else -1
        best_atom = None
        best_new_fc = None
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

            allowed_fc = sorted({int(round(bv - v)) for v in valences})
            for new_fc in allowed_fc:
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

        best_atom.SetFormalCharge(best_new_fc)

    mol.UpdatePropertyCache(strict=False)
    return mol


def _charge_score(mol):
    """Element-aware charge score to guide global bond-order moves.

    Lower is better.  Strongly penalizes impossible over-valence, then penalizes
    charged atoms with element-dependent preferences:
    - prefer O-/S- over N-/C-
    - prefer N+/P+ over O+/C+
    """
    score = molecule_charge_penalty(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        radicals = atom.GetNumRadicalElectrons()
        if radicals > 0:
            score += 20000 * radicals
        int_bv = _bond_valence(atom, integer=True)
        over = int_bv - _max_allowed_int_bv(atom)
        if over > 0:
            score += 1000 * over
    return score


def _charge_objective(mol):
    """Lexicographic objective for trial bond-order moves.

    Primary key is ``_charge_score``; the target-charge gap is a secondary
    key to gently steer toward the requested net charge without overriding
    local chemistry. Tie-breakers favor fewer charged atoms, then fewer
    negative nitrogens (common over-separation artefact), then lower total
    absolute formal charge.
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


def _apply_alternating_path_shift(mol, path, first_delta=1):
    """Try one arrow-pushing move along *path*.

    Bonds along the path are alternately promoted/demoted:
    ``first_delta, -first_delta, first_delta, ...`` on successive edges.
    Returns a new molecule if the move is valence-safe and non-aromatic, else None.
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
        delta = first_delta if i % 2 == 0 else -first_delta
        order = int(bond.GetBondTypeAsDouble())
        new_order = order + delta
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


def _apply_alternating_path_shift_via_kekulize(mol, path, first_delta=1):
    """Attempt path shift on a temporary kekulized copy of *mol*.

    This enables resonance-style updates in aromatic systems where the direct
    path shift rejects aromatic bonds. If kekulization fails, returns ``None``.
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


def _ring_paths_between_atoms(mol, idx_a, idx_b, max_path=8):
    """Return atom-index paths between idx_a and idx_b along shared rings.

    For each ring containing both atoms, return both directed paths around the
    ring (clockwise/counterclockwise). This enables evaluating "arrow pushing
    the other way around the ring" when one direction overcharges an atom.
    """
    paths = []
    seen = set()
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if idx_a not in ring or idx_b not in ring:
            continue
        ring_list = list(ring)
        ia = ring_list.index(idx_a)
        ib = ring_list.index(idx_b)

        # Forward path ia -> ib
        if ia <= ib:
            path_f = ring_list[ia : ib + 1]
        else:
            path_f = ring_list[ia:] + ring_list[: ib + 1]

        # Backward path ia -> ib (the opposite ring direction)
        ring_rev = [ring_list[ia]] + list(
            reversed(ring_list[:ia] + ring_list[ia + 1 :])
        )
        ib_rev = ring_rev.index(idx_b)
        path_b = ring_rev[: ib_rev + 1]

        for path in (path_f, path_b):
            n_bonds = len(path) - 1
            if n_bonds < 1 or n_bonds > max_path:
                continue
            tup = tuple(path)
            if tup in seen:
                continue
            seen.add(tup)
            paths.append(path)

    return paths


def _all_paths_between_atoms(mol, idx_a, idx_b, max_path=8, max_paths_per_pair=64):
    """Enumerate candidate paths between two atoms.

    Prefers ``tmos.graph_mapping.find_all_paths`` when available to capture
    non-shortest alternatives (critical for ring-direction resonance moves).
    Falls back to shortest-path + ring-path enumeration.
    """
    seen = set()
    paths = []

    # Fallback shortest path
    shortest = Chem.rdmolops.GetShortestPath(mol, idx_a, idx_b)
    if shortest:
        n_bonds = len(shortest) - 1
        if 1 <= n_bonds <= max_path:
            tup = tuple(shortest)
            if tup not in seen:
                seen.add(tup)
                paths.append(list(shortest))

    # Fallback ring alternatives
    for ring_path in _ring_paths_between_atoms(mol, idx_a, idx_b, max_path=max_path):
        tup = tuple(ring_path)
        if tup in seen:
            continue
        seen.add(tup)
        paths.append(list(ring_path))

    # Preferred: all simple paths via tmos graph mapping
    if paths:
        try:
            graph = tmos.graph_mapping.mol_to_graph(mol)
            all_paths = tmos.graph_mapping.find_all_paths(
                graph, idx_a, idx_b, cutoff=max_path
            )
            for path in all_paths:
                n_bonds = len(path) - 1
                if n_bonds < 1 or n_bonds > max_path:
                    continue
                tup = tuple(path)
                if tup in seen:
                    continue
                seen.add(tup)
                paths.append(list(path))
                if len(paths) >= max_paths_per_pair:
                    break
        except Exception:
            pass

    paths.sort(key=lambda p: len(p))
    return paths[:max_paths_per_pair]


def _set_bond_order_int(bond, order):
    """Set RDKit bond type from integer order 1/2/3."""
    bond.SetBondType(
        Chem.BondType.SINGLE
        if order == 1
        else (Chem.BondType.DOUBLE if order == 2 else Chem.BondType.TRIPLE)
    )


def _neutral_sink_priority(atom):
    """Priority for neutral sink atoms (higher = preferred sink).

    Encodes preference for oxygen on S/N over oxygen on C.
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


def _strip_radicals_and_reassign(mol):
    """Zero radical electrons, then reassign formal charges."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumRadicalElectrons(0)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    out = _assign_formal_charges(out)
    return out


def _convert_radical_carbocations(mol):
    """Convert tricoordinate C radicals to C⁺ cations when target charge demands it.

    OBabel occasionally assigns bv=3, rad=1, fc=0 to a benzylic CH₂ that is
    chemically a carbocation [CH₂⁺].  :func:`_assign_formal_charges` leaves
    fc=0 because bv=3.0 maps to a neutral carbon via the radical (non-bonding)
    electron path rather than to C⁺ (valid shell 3, fc=+1).

    This pass fires only when the molecule has a positive ``_target_charge``
    and is currently under-charged (current total charge < target).  For each
    tricoordinate C radical (bv==3, rad≥1, fc==0) it sets fc=+1 and removes
    one radical electron, stopping once the target charge is reached.
    """
    target = _target_charge_or_none(mol)
    if target is None:
        return mol
    current = sum(a.GetFormalCharge() for a in mol.GetAtoms())
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
        max_iters=24,
        max_path=12,
        max_paths_per_pair=128,
        use_relay=True,
        use_paths=True,
        path_mode="charged_sinks",
    ):
        self.max_iters = max_iters
        self.max_path = max_path
        self.max_paths_per_pair = max_paths_per_pair
        self.use_relay = use_relay
        self.use_paths = use_paths
        self.path_mode = path_mode

    @staticmethod
    def _prep_candidate(candidate):
        candidate = _prepare_for_cleanup(candidate)
        candidate = _fix_overvalenced(candidate)
        candidate = _assign_formal_charges(candidate)
        return candidate

    @staticmethod
    def _is_active_site(atom):
        return atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() > 0

    @staticmethod
    def _atom_charge_cost(atom):
        radicals = atom.GetNumRadicalElectrons()
        if radicals > 0:
            return 10000 + 1000 * radicals
        return _atom_charge_penalty(atom.GetAtomicNum(), atom.GetFormalCharge())

    @classmethod
    def _atom_site_penalty(cls, mol, atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        charge_cost = cls._atom_charge_cost(atom)
        if charge_cost > 0:
            return charge_cost

        if atom.GetAtomicNum() in (0, 1):
            return 0
        valences = _charge_adjusted_valences(atom)
        if not valences:
            return 0
        deficit = _valence_deficit(atom)
        if deficit <= 0:
            return 0
        return 10 + int(np.ceil(deficit * 10.0))

    @classmethod
    def _priority_active_atoms(cls, mol):
        active = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() != 1 and cls._is_active_site(atom)
        ]
        active.sort(key=lambda idx: cls._atom_site_penalty(mol, idx), reverse=True)
        return active

    @classmethod
    def _priority_sink_atoms(cls, mol, exclude=None):
        exclude = set() if exclude is None else set(exclude)
        sinks = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx in exclude or atom.GetAtomicNum() == 1:
                continue
            pri = _neutral_sink_priority(atom)
            if pri > 0:
                sinks.append((idx, pri))
        sinks.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sinks]

    @classmethod
    def _priority_partner_atoms(cls, mol, src_idx, include_zero=False, max_partners=24):
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
        engine,
        base,
        src,
        dst,
        max_path,
        max_paths_per_pair,
        require_dst_improve,
    ):
        base_obj = _charge_objective(base)
        base_score = _charge_score(base)
        src_penalty_base = cls._atom_site_penalty(base, src)
        dst_penalty_base = cls._atom_site_penalty(base, dst)

        paths = _all_paths_between_atoms(
            base,
            src,
            dst,
            max_path=max_path,
            max_paths_per_pair=max_paths_per_pair,
        )
        paths_by_length = {}
        for path in paths:
            length = len(path) - 1
            paths_by_length.setdefault(length, []).append(path)

        for length in sorted(paths_by_length):
            tier_best = None
            tier_best_score = None
            tier_best_obj = None
            for path in paths_by_length[length]:
                for first_delta in (1, -1):
                    candidate = engine._apply_path_move(base, path, first_delta)
                    if candidate is None:
                        continue
                    candidate = engine._prep_candidate(candidate)
                    candidate = _reduce_charge_by_bond_promotion(candidate)

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

                    if (
                        tier_best is None
                        or tier_best_score is None
                        or cand_score < tier_best_score
                        or (
                            cand_score == tier_best_score
                            and (tier_best_obj is None or cand_obj < tier_best_obj)
                        )
                    ):
                        tier_best = candidate
                        tier_best_score = cand_score
                        tier_best_obj = cand_obj
            if tier_best is not None:
                return tier_best

        return None

    @staticmethod
    def _apply_adjacent_move(mol, bond_idx, delta):
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
    def _promote_atom_neighbors_greedy(cls, mol):
        """Greedy local neutralization: promote eligible neighbor bonds per atom.

        Atoms are processed in descending local charge cost (e.g., C- first).
        For each active atom, apply the best legal promotion that does not worsen
        global charge penalty and does not worsen that atom's |formal charge|,
        then re-evaluate the same atom before moving on.
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

    def _iter_relay_specs(self, mol):
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
    def _apply_relay_move(mol, ab_idx, bc_idx, cd_idx):
        trial = Chem.RWMol(mol)
        t_ab = trial.GetBondWithIdx(ab_idx)
        t_bc = trial.GetBondWithIdx(bc_idx)
        t_cd = trial.GetBondWithIdx(cd_idx)
        if t_ab is None or t_bc is None or t_cd is None:
            return None

        o_ab = int(t_ab.GetBondTypeAsDouble())
        o_bc = int(t_bc.GetBondTypeAsDouble())
        o_cd = int(t_cd.GetBondTypeAsDouble())
        n_ab = o_ab + 1
        n_bc = o_bc - 1
        n_cd = o_cd + 1
        if n_ab < 1 or n_ab > 3 or n_bc < 1 or n_bc > 3 or n_cd < 1 or n_cd > 3:
            return None

        _set_bond_order_int(t_ab, n_ab)
        _set_bond_order_int(t_bc, n_bc)
        _set_bond_order_int(t_cd, n_cd)
        trial.UpdatePropertyCache(strict=False)
        return trial.GetMol()

    def _path_source_target_atoms(self, mol):
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

    def _iter_path_moves(self, mol):
        source_atoms, target_atoms = self._path_source_target_atoms(mol)
        seen_paths = set()
        for src in source_atoms:
            for dst in target_atoms:
                if src == dst:
                    continue
                for path in _all_paths_between_atoms(
                    mol,
                    src,
                    dst,
                    max_path=self.max_path,
                    max_paths_per_pair=self.max_paths_per_pair,
                ):
                    key = tuple(path)
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    yield path, 1
                    yield path, -1

    @staticmethod
    def _apply_path_move(mol, path, first_delta):
        cand = _apply_alternating_path_shift(mol, path, first_delta=first_delta)
        if cand is None:
            cand = _apply_alternating_path_shift_via_kekulize(
                mol,
                path,
                first_delta=first_delta,
            )
        return cand

    def optimize(self, mol):
        mol = _assign_formal_charges(mol)
        current_obj = _charge_objective(mol)

        for _ in range(self.max_iters):
            work = self._kekulized_or_none(Chem.Mol(mol))
            if work is None:
                work = Chem.Mol(mol)
            work = _assign_formal_charges(work)

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
                for path, first_delta in self._iter_path_moves(work):
                    candidate = self._apply_path_move(work, path, first_delta)
                    if candidate is None:
                        continue
                    candidate = self._prep_candidate(candidate)
                    cand_obj = _charge_objective(candidate)
                    if cand_obj < best_obj:
                        best_obj = cand_obj
                        best_mol = candidate

            if best_mol is None:
                break

            mol = best_mol
            current_obj = best_obj

        return mol

    @staticmethod
    def _sanitize_candidate_or_none(mol):
        try:
            candidate = _assign_formal_charges(Chem.Mol(mol))
            Chem.SanitizeMol(candidate)
            return candidate
        except Exception:
            try:
                cleaned = _strip_radicals_and_reassign(mol)
                Chem.SanitizeMol(cleaned)
                return cleaned
            except Exception:
                return None

    @staticmethod
    def _kekulized_or_none(mol):
        try:
            candidate = Chem.Mol(mol)
            Chem.Kekulize(candidate, clearAromaticFlags=True)
            return candidate
        except Exception:
            return None

    @staticmethod
    def _prefer_lower_penalty(current, candidate):
        """Return the lower-penalty molecule between current and candidate."""
        if candidate is None:
            return current
        curr_obj = _charge_objective(current)
        cand_obj = _charge_objective(candidate)

        # Softly prefer candidates that close the target-charge gap, even if
        # the chemistry objective is slightly worse, provided charge_score does
        # not degrade beyond a small tolerance. This helps converge to the
        # requested net charge without overturning local chemistry preferences.
        curr_gap = abs(_target_charge_delta(current))
        cand_gap = abs(_target_charge_delta(candidate))
        if cand_gap < curr_gap:
            curr_score = _charge_score(current)
            cand_score = _charge_score(candidate)
            if cand_score <= curr_score + 4:
                return candidate

        if cand_obj < curr_obj:
            return candidate
        return current

    @staticmethod
    def _charge_cost_with_fc(atom, new_fc):
        """Cost of assigning *new_fc* to *atom* without mutating the input."""
        radicals = atom.GetNumRadicalElectrons()
        if radicals > 0:
            return 10000 + 1000 * radicals
        return _atom_charge_penalty(atom.GetAtomicNum(), new_fc)

    @classmethod
    def _force_charge_balance(cls, mol):
        """Nudge formal charges toward the target total without changing bonds.

        Only moves that leave the atom within its allowed integer bond valence
        at the proposed new formal charge are considered.  This prevents, for
        example, assigning fc=+1 to a tetravalent carbon (bv=4) which would
        create an over-valenced C⁺ that fails RDKit sanitization.
        """
        target = _target_charge_or_none(mol)
        if target is None:
            return mol

        rw = Chem.RWMol(mol)
        for _ in range(64):
            current = sum(a.GetFormalCharge() for a in rw.GetAtoms())
            if current == target:
                break
            step = 1 if target > current else -1

            best_idx = None
            best_score = None

            for atom in rw.GetAtoms():
                if atom.GetAtomicNum() in (0, 1):
                    continue

                new_fc = atom.GetFormalCharge() + step
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
    def finalize_sanitized(cls, mol):
        """Return best sanitizable candidate; avoid hard KekulizeException failure."""
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
                sanitized = _strip_radicals_and_reassign(candidate)
                try:
                    Chem.SanitizeMol(sanitized)
                except Exception:
                    sanitized = None
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
                try:
                    Chem.SanitizeMol(forced)
                    return forced
                except Exception:
                    # Over-valence artefact from charge adjustment — demote and retry.
                    forced = _fix_overvalenced(forced)
                    forced = _assign_formal_charges(forced)
                    try:
                        Chem.SanitizeMol(forced)
                        return forced
                    except Exception:
                        pass
            try:
                Chem.SanitizeMol(forced)
                return forced
            except Exception:
                fixed = _fix_overvalenced(Chem.Mol(best_candidate))
                fixed = _assign_formal_charges(fixed)
                try:
                    Chem.SanitizeMol(fixed)
                    return fixed
                except Exception:
                    return best_candidate

        partial = _prepare_for_cleanup(Chem.Mol(mol))
        partial = _strip_radicals_and_reassign(partial)
        Chem.SanitizeMol(partial)
        return partial

    @classmethod
    def _apply_tmos_obvious_bonds(cls, mol, degree_of_separation=8):
        """Apply tmos global bond-order correction when available."""

        rw = Chem.RWMol(mol)
        try:
            tmos.build_rdmol.add_obvious_bonds(
                rw, degree_of_separation=degree_of_separation
            )
        except Exception:
            return mol

        out = rw.GetMol()
        out = _fix_overvalenced(out)
        out = _assign_formal_charges(out)
        return out

    @classmethod
    def _path_lookahead_first_improvement(
        cls,
        mol,
        max_path=12,
        max_paths_per_pair=128,
        path_mode="charged_sinks",
    ):
        """Try path moves shortest→longest and accept first improving length tier.

        For each path length L (starting from shortest), evaluate all candidate
        path moves of that length and pick the best objective at length L.
        Return immediately when any improving candidate exists at L; otherwise
        continue to the next longer length.
        """
        engine = cls(
            max_iters=1,
            max_path=max_path,
            max_paths_per_pair=max_paths_per_pair,
            use_relay=False,
            use_paths=True,
            path_mode=path_mode,
        )

        base = _assign_formal_charges(Chem.Mol(mol))
        kek = cls._kekulized_or_none(base)
        if kek is not None:
            base = _assign_formal_charges(kek)
        active_atoms = cls._priority_active_atoms(base)

        # 1) High-penalty atom pairs first: require both atoms to improve.
        for src in active_atoms:
            for dst in cls._priority_partner_atoms(base, src, include_zero=True):
                dst_penalty = cls._atom_site_penalty(base, dst)
                candidate = cls._best_path_move_between_pair(
                    engine,
                    base,
                    src,
                    dst,
                    max_path=max_path,
                    max_paths_per_pair=max_paths_per_pair,
                    require_dst_improve=dst_penalty > 0,
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
                    max_path=max_path,
                    max_paths_per_pair=max_paths_per_pair,
                    require_dst_improve=False,
                )
                if candidate is not None:
                    return candidate

        return base

    @classmethod
    def _stage_once(cls, mol, local_engine, global_engine):
        """Run one deterministic cleanup stage sequence."""
        mol = _fix_overvalenced(mol)
        mol = _promote_underbonded(mol)
        mol = _assign_formal_charges(mol)
        mol = _convert_radical_carbocations(mol)
        mol = _reduce_charge_by_bond_promotion(mol)
        mol = cls._promote_atom_neighbors_greedy(mol)

        move_base = cls._kekulized_or_none(Chem.Mol(mol))
        if move_base is None:
            move_base = Chem.Mol(mol)
        move_base = _assign_formal_charges(move_base)

        path_candidate = cls._path_lookahead_first_improvement(
            Chem.Mol(move_base),
            max_path=15,
            max_paths_per_pair=256,
            path_mode="charged_sinks",
        )
        mol = cls._prefer_lower_penalty(mol, path_candidate)

        local_candidate = local_engine.optimize(Chem.Mol(mol))
        mol = cls._prefer_lower_penalty(mol, local_candidate)

        negpos_path_candidate = cls._path_lookahead_first_improvement(
            Chem.Mol(mol),
            max_path=10,
            max_paths_per_pair=128,
            path_mode="neg_pos",
        )
        mol = cls._prefer_lower_penalty(mol, negpos_path_candidate)

        tmos_candidate = cls._apply_tmos_obvious_bonds(Chem.Mol(mol))
        mol = cls._prefer_lower_penalty(mol, tmos_candidate)

        global_candidate = global_engine.optimize(Chem.Mol(mol))
        mol = cls._prefer_lower_penalty(mol, global_candidate)

        mol = _prepare_for_cleanup(mol)
        mol = _fix_overvalenced(mol)
        mol = _assign_formal_charges(mol)
        return mol

    @classmethod
    def cleanup_best(cls, mol, max_rounds=8):
        """Run deterministic cleanup loop and return the best sanitizable molecule.

        This unifies convergence control and sanitize fallback selection so callers
        execute one cleanup entrypoint with bounded rounds and cycle checks.
        """
        work = _prepare_for_cleanup(Chem.Mol(mol))

        local_engine = cls(
            max_iters=10,
            max_path=15,
            max_paths_per_pair=256,
            use_relay=True,
            use_paths=True,
            path_mode="charged_sinks",
        )
        global_engine = cls(
            max_iters=18,
            max_path=10,
            max_paths_per_pair=128,
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
            if work_objective >= last_work_objective:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            last_work_objective = work_objective

            after = _molecule_state_fingerprint(work)
            if after == before or stagnant_rounds >= 2:
                break

        for _ in range(3):
            sanitized = cls._sanitize_candidate_or_none(work)
            if sanitized is not None:
                forced = cls._force_charge_balance(sanitized)
                if _target_charge_delta(forced) == 0:
                    try:
                        Chem.SanitizeMol(forced)
                        return forced
                    except Exception:
                        forced = _fix_overvalenced(forced)
                        forced = _assign_formal_charges(forced)
                        try:
                            Chem.SanitizeMol(forced)
                            return forced
                        except Exception:
                            pass
                try:
                    Chem.SanitizeMol(forced)
                    return forced
                except Exception:
                    fixed = _fix_overvalenced(Chem.Mol(sanitized))
                    fixed = _assign_formal_charges(fixed)
                    try:
                        Chem.SanitizeMol(fixed)
                        return fixed
                    except Exception:
                        return sanitized

            next_work = cls._stage_once(Chem.Mol(work), local_engine, global_engine)
            if _molecule_state_fingerprint(next_work) == _molecule_state_fingerprint(
                work
            ):
                break

            if _charge_objective(next_work) <= _charge_objective(work):
                work = next_work
            else:
                break

        return cls.finalize_sanitized(work)


def _fix_overvalenced(mol):
    """Demote or remove bonds so each atom stays within allowed charge/valence.

    Cleanup works on explicit bond orders (single/double/triple), with aromatic
    bonds removed upstream by kekulization.
    """
    mol = Chem.RWMol(mol)
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None

    def _geom_longest(bonds, atom_idx):
        """Return the bond in *bonds* whose other endpoint is furthest from
        atom *atom_idx* in the conformer (or the first bond if no conformer)."""
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
            max_fc = _MAX_BOND_FC.get(atom.GetAtomicNum(), _DEFAULT_MAX_BOND_FC)
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
                new_order = int(longest.GetBondTypeAsDouble()) - 1
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
                    continue
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
    return mol.GetMol()


def _promote_underbonded(mol):
    """Promote non-aromatic bonds between atom pairs that are both under-bonded.

    An atom is under-bonded when its effective valence (float bv − fc) is below
    the lowest valid shell it can reach — SanitizeMol would assign radical
    electrons.  Promoting the shared bond simultaneously reduces the deficit for
    both atoms.

    Must be called *before* _assign_formal_charges so that atoms not yet
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

    def _unused(atom):
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
            unused1 = _unused(a1)
            unused2 = _unused(a2)
            promote_ok = (unused1 >= 1 and unused2 >= 1) or (
                (unused1 >= 1 and unused2 >= 0.5) or (unused2 >= 1 and unused1 >= 0.5)
            )
            if promote_ok:
                new_order = int(bond.GetBondTypeAsDouble()) + 1
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


def _assign_formal_charges(mol):
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
        best = _best_shell(atom, bv)
        if best is None:
            continue
        atom.SetFormalCharge(int(round(bv - best)))

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

    mol = _enforce_target_charge(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol


def _reduce_charge_by_bond_promotion(mol):
    """Absorb spurious negative charges into higher-order bonds.

    After _assign_formal_charges, atoms such as S and P may sit at a lower
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

    Formal charges are re-derived by _assign_formal_charges at the end.

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

    def _local_atom_penalty(m, atom_indices):
        total = 0
        for atom_idx in atom_indices:
            atom = m.GetAtomWithIdx(atom_idx)
            total += _atom_charge_penalty(atom.GetAtomicNum(), atom.GetFormalCharge())
        return total

    def _sort_key_for_neighbor(atom, center_idx):
        ep = _NEG_NBOR_PRIORITY.get(atom.GetAtomicNum(), 99)
        is_negative = 0 if atom.GetFormalCharge() < 0 else 1
        if conf is None:
            return (is_negative, ep, 0.0)
        p0 = conf.GetAtomPosition(center_idx)
        p1 = conf.GetAtomPosition(atom.GetIdx())
        dist = np.linalg.norm(np.array([p1.x - p0.x, p1.y - p0.y, p1.z - p0.z]))
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
            max_fc_A = _MAX_BOND_FC.get(atom.GetAtomicNum(), _DEFAULT_MAX_BOND_FC)

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


def _prepare_for_cleanup(mol):
    """Prepare explicit-bond-order graph for cleanup passes.

    Aromatic flags are cleared so optimization always works with single/double
    (and occasional triple) bonds only.
    """
    candidate = Chem.Mol(mol)
    try:
        Chem.Kekulize(candidate, clearAromaticFlags=True)
    except Exception:
        pass
    candidate.UpdatePropertyCache(strict=False)
    return candidate


def _molecule_state_fingerprint(mol):
    """Hashable state describing atom charges and bond orders/connectivity."""
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


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def _restore_explicit_hydrogen_flags(mol):
    """Re-apply explicit-hydrogen atom flags after MOL-block round-trip."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    return rw.GetMol()


def _initial_bonding_rdkit(mol, charge=None):
    """Assign connectivity and bond orders using RDKit native perception.

    If provided charge is None, a charge of 0 is assumed.
    """
    if charge is None:
        charge = 0
    rdDetermineBonds.DetermineConnectivity(mol)
    DetermineBondOrders(
        mol,
        charge=charge,
        maxIterations=1000,
        allowChargedFragments=True,
    )
    return mol


def _initial_bonding_obabel(mol, charge=None):
    """Assign connectivity and bond orders using OpenBabel perception."""
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("mol", "mol")
    ob_mol = ob.OBMol()
    ob_conversion.ReadString(ob_mol, Chem.MolToMolBlock(mol))
    if charge is not None:
        ob_mol.SetTotalCharge(charge)
    ob_mol.ConnectTheDots()

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


def determine_bonds(
    mol,
    charge=None,
    method="obabel",
    custom_cleanup=True,
    cleanup_max_iters=10,
):
    """Assign connectivity, bond orders, and formal charges.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with atoms and a conformer but no bonds.
    charge : int
        Net molecular charge (used by the rdkit method).
    method : str, optional
        Backend for initial bond/order detection. Options include
        {"rdkit", "obabel"}. Defaults to "rdkit".
    custom_cleanup : bool, optional
        Run charge/bond cleanup with deterministic move engines.
    cleanup_max_iters : int, optional
        Maximum cleanup rounds. Defaults to 8.

    Returns
    -------
    rdkit.Chem.Mol
        Sanitized molecule.

    Raises
    ------
    ValueError
        If ``method`` is not one of {"rdkit", "obabel"}, or if the calculated
        formal charge does not match ``charge`` after cleanup.
    """
    mol.UpdatePropertyCache(strict=False)
    if charge is not None:
        mol.SetProp("_target_charge", str(int(charge)))

    backend = {
        "rdkit": _initial_bonding_rdkit,
        "obabel": _initial_bonding_obabel,
    }.get(method)

    if backend is None:
        raise ValueError(
            f"method={method!r} is not supported; choose 'rdkit' or 'obabel'."
        )

    mol = backend(mol, charge)
    mol.UpdatePropertyCache(strict=False)

    if custom_cleanup:
        mol = _GraphMoveEngine.cleanup_best(mol, max_rounds=cleanup_max_iters)
        if (
            charge is not None
            and sum([a.GetFormalCharge() for a in mol.GetAtoms()]) != charge
        ):
            raise ValueError("Inconsistent charge with target!")

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol = _GraphMoveEngine.finalize_sanitized(mol)

    Chem.SanitizeMol(mol)

    return mol
