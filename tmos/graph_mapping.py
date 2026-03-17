"""Turn RDKit molecules into graphical representations, then perform comparisons and analyses."""

from collections import defaultdict

import networkx as nx
from loguru import logger
from rdkit.Chem.rdchem import Mol

EnvironmentKey = tuple[str, int, tuple[str, ...]]


def mol_to_graph(mol: Mol, remove_hydrogens: bool = False) -> nx.Graph:
    """Convert an RDKit molecule to a `networkx.Graph`.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    remove_hydrogens : bool, default=False
        If ``True``, hydrogen atoms are excluded from the graph.  Nodes retain
        their original RDKit atom indices so paths returned by NetworkX map
        directly back to atom indices in *mol* without re-indexing.
        Defaults to ``False``.

    Returns
    -------
    networkx.Graph
        Graph where node IDs are RDKit atom indices and edges are bonds.
        Node attributes:

        - ``symbol`` : str
            Atomic symbol.
        - ``degree`` : int
            Degree in the generated graph.
        - ``atom_idx`` : int
            Original RDKit atom index (same as node ID).

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> graph = mol_to_graph(mol)
    >>> sorted(graph.nodes)
    [0, 1, 2]
    """
    G = nx.Graph()

    atom_degree = defaultdict(lambda: 0)
    for b in mol.GetBonds():
        begin_idx = b.GetBeginAtomIdx()
        end_idx = b.GetEndAtomIdx()
        if remove_hydrogens and (
            mol.GetAtomWithIdx(begin_idx).GetAtomicNum() == 1
            or mol.GetAtomWithIdx(end_idx).GetAtomicNum() == 1
        ):
            continue
        atom_degree[begin_idx] += 1
        atom_degree[end_idx] += 1

    for atom in mol.GetAtoms():
        if remove_hydrogens and atom.GetAtomicNum() == 1:
            continue
        idx = atom.GetIdx()
        G.add_node(
            idx,
            symbol=atom.GetSymbol(),
            degree=atom_degree[idx],
            atom_idx=idx,
        )

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if remove_hydrogens and (
            mol.GetAtomWithIdx(begin_idx).GetAtomicNum() == 1
            or mol.GetAtomWithIdx(end_idx).GetAtomicNum() == 1
        ):
            continue
        G.add_edge(begin_idx, end_idx)

    return G


def get_atom_environment(graph: nx.Graph, atom_idx: int) -> EnvironmentKey:
    """Return local environment key for one atom.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph of the molecule.
    atom_idx : int
        Index of the atom (node index in the graph).

    Returns
    -------
    tuple of (str, int, tuple[str, ...])
        ``(symbol, degree, sorted_neighbor_symbols)``.
    """
    node_data = graph.nodes[atom_idx]
    symbol = node_data["symbol"]
    degree = node_data["degree"]

    # Get neighbor symbols (ignore bond orders)
    neighbor_symbols = []
    for neighbor_idx in graph.neighbors(atom_idx):
        neighbor_symbol = graph.nodes[neighbor_idx]["symbol"]
        neighbor_symbols.append(neighbor_symbol)

    neighbor_symbols.sort()

    return (symbol, degree, tuple(neighbor_symbols))


def implicit_hydrogen_atom_mapping(mol: Mol) -> dict[int, int]:
    """Map heavy-atom indices from explicit-H to implicit-H ordering.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule with explicit hydrogens.

    Returns
    -------
    dict of int to int
        Mapping from original atom index to heavy-atom index after removing H atoms.
    """
    atom_mapping: dict[int, int] = {}
    i = 0
    for atm in mol.GetAtoms():
        if atm.GetSymbol() != "H":
            atom_mapping[atm.GetIdx()] = i
            i += 1
    return atom_mapping


def find_atom_mapping(mol1: Mol, mol2: Mol) -> dict[int, int]:
    """Map atom indices between two topologically equivalent molecules.

    Parameters
    ----------
    mol1 : rdkit.Chem.rdchem.Mol
        Source molecule.
    mol2 : rdkit.Chem.rdchem.Mol
        Target molecule.

    Returns
    -------
    dict of int to int
        Mapping from atom index in ``mol1`` to atom index in ``mol2``.

    Raises
    ------
    ValueError
        If node counts, environment signatures, or connectivity checks fail.

    Notes
    -----
    When multiple atoms share the same first-order environment, mapping is
    assigned by encounter order within each environment bucket.

    Examples
    --------
    >>> from rdkit import Chem
    >>> m1 = Chem.MolFromSmiles("CCO")
    >>> m2 = Chem.MolFromSmiles("OCC")
    >>> mapping = find_atom_mapping(m1, m2)
    >>> sorted(mapping.keys()) == list(range(m1.GetNumAtoms()))
    True
    """

    orig_graph = mol_to_graph(mol1)
    correct_graph = mol_to_graph(mol2)
    if orig_graph.number_of_nodes() != correct_graph.number_of_nodes():
        raise ValueError("Number of atoms do not match in provided molecules.")

    orig_environments: dict[int, EnvironmentKey] = {}
    for atom_idx in orig_graph.nodes():
        orig_environments[atom_idx] = get_atom_environment(orig_graph, atom_idx)

    correct_environments: dict[int, EnvironmentKey] = {}
    for atom_idx in correct_graph.nodes():
        correct_environments[atom_idx] = get_atom_environment(correct_graph, atom_idx)

    orig_by_env = defaultdict(list)
    for atom_idx, env in orig_environments.items():
        orig_by_env[env].append(atom_idx)

    correct_by_env = defaultdict(list)
    for atom_idx, env in correct_environments.items():
        correct_by_env[env].append(atom_idx)

    if set(orig_by_env.keys()) != set(correct_by_env.keys()):
        raise ValueError("Atom environments do not match. Mapping failed.")

    mapping: dict[int, int] = {}
    for env in orig_by_env.keys():
        orig_atoms = orig_by_env[env]
        correct_atoms = correct_by_env[env]
        if len(orig_atoms) != len(correct_atoms):
            raise ValueError(
                f"For the atom environment {env}, there are {len(orig_atoms)} atoms for the original and {len(correct_atoms)} for the new molecule. Mapping failed."
            )

        if len(orig_atoms) == 1:
            mapping[orig_atoms[0]] = correct_atoms[0]
        else:
            # For multiple atoms with same environment, assign in order. TODO: extend to 2nd order connectivity
            for i, orig_atom in enumerate(orig_atoms):
                mapping[orig_atom] = correct_atoms[i]

    validate_mapping(orig_graph, correct_graph, mapping)

    return mapping


def validate_mapping(
    orig_graph: nx.Graph,
    correct_graph: nx.Graph,
    mapping: dict[int, int],
) -> None:
    """Validate that a node mapping preserves graph connectivity.

    Parameters
    ----------
    orig_graph : networkx.Graph
        The graph representation of the original molecule.
    correct_graph : networkx.Graph
        The graph representation of the target molecule.
    mapping : dict of int to int
        A dictionary mapping node indices from `orig_graph` to `correct_graph`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If mapping cardinality, uniqueness, or edge consistency checks fail.

    """
    if len(mapping) != orig_graph.number_of_nodes():
        raise ValueError(
            "Mapping validation failed: The number of mapping entries does not match the number of nodes."
        )

    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            "Mapping validation failed: Replicate mapping entries detected."
        )

    # Check that connectivity is preserved
    for orig_u, orig_v in orig_graph.edges():
        if orig_u in mapping and orig_v in mapping:
            correct_u = mapping[orig_u]
            correct_v = mapping[orig_v]

            if not correct_graph.has_edge(correct_u, correct_v):
                raise ValueError(
                    f"Mapping validation failed: Bond between atoms {correct_u} and {correct_v} is not found."
                )

    # Check that no extra edges are created
    for correct_u, correct_v in correct_graph.edges():
        orig_u, orig_v = None, None
        for orig_atom, correct_atom in mapping.items():
            if correct_atom == correct_u:
                orig_u = orig_atom
            if correct_atom == correct_v:
                orig_v = orig_atom

        if orig_u is not None and orig_v is not None:
            if not orig_graph.has_edge(orig_u, orig_v):
                raise ValueError(
                    f"Mapping validation failed: Bond between atoms {orig_u} and {orig_v} is not found."
                )


##############################################
################### Rings ####################
##############################################


def find_molecular_rings(
    mol_graph: nx.Graph,
    min_ring_size: int = 3,
    max_ring_size: int = 12,
) -> list[tuple[int, ...]]:
    """Find unique rings from a graph cycle basis.

    Parameters
    ----------
    mol_graph : networkx.Graph
        Graph where nodes are atom indices and edges are bonds.
    min_ring_size : int, default=3
        Minimum cycle size to keep.
    max_ring_size : int, default=12
        Maximum cycle size to keep.

    Returns
    -------
    list of tuple[int, ...]
        Unique rings as tuples of atom indices, sorted by ring size then index.

    Examples
    --------
    >>> import networkx as nx
    >>> graph = nx.Graph()
    >>> graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    >>> find_molecular_rings(graph)
    [(0, 1, 2)]
    """
    if not isinstance(mol_graph, nx.Graph):
        raise ValueError("Input must be a NetworkX Graph")

    if mol_graph.number_of_nodes() < min_ring_size:
        return []

    try:
        cycles = []
        cycle_basis = nx.minimum_cycle_basis(mol_graph)
        for cycle in cycle_basis:
            if len(cycle) >= min_ring_size and len(cycle) <= max_ring_size:
                min_idx: int = cycle.index(min(cycle))
                normalized_cycle = tuple(cycle[min_idx:] + cycle[:min_idx])
                cycles.append(normalized_cycle)

    except Exception as e:
        logger.info(f"Error finding cycles: {e}")
        return []

    # Remove duplicates and sort by ring size
    unique_cycles = []
    seen_cycles = set()
    for cycle in cycles:
        canonical = tuple(sorted(cycle))
        if canonical not in seen_cycles:
            seen_cycles.add(canonical)
            unique_cycles.append(cycle)

    # Sort by ring size, then by first atom index
    unique_cycles.sort(key=lambda x: (len(x), min(x)))

    return unique_cycles


def find_all_paths(
    graph: nx.Graph,
    source: int,
    target: int,
    cutoff: int | None = None,
) -> list[list[int]]:
    """Find all simple paths between two nodes.

    A simple path visits each node at most once. For dense graphs or large
    cutoff values the number of paths can grow exponentially; use ``cutoff``
    to limit search depth when necessary.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph to search. Nodes are typically atom indices.
    source : int
        Index of the starting node.
    target : int
        Index of the ending node.
    cutoff : int or None, default=None
        Maximum number of edges in any returned path. If ``None`` (default),
        no depth limit is applied.

    Returns
    -------
    paths : list[list[int]]
        List of paths, each path being an ordered list of node indices from
        ``source`` to ``target``. Returns an empty list if no path exists or
        if ``source`` and ``target`` are not present in the graph.

    Raises
    ------
    ValueError
        If ``source`` or ``target`` is not a node in the graph.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])
    >>> paths = find_all_paths(G, 0, 3)
    >>> print(paths)  # [[0, 1, 2, 3], [0, 2, 3]]
    """
    if source not in graph:
        raise ValueError(f"Source node {source} is not present in the graph.")
    if target not in graph:
        raise ValueError(f"Target node {target} is not present in the graph.")

    if source == target:
        return [[source]]

    try:
        paths = list(
            nx.all_simple_paths(graph, source=source, target=target, cutoff=cutoff)
        )
    except Exception:
        return []

    return paths
