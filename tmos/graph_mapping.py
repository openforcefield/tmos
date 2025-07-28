"""Turn RDKit molecules into graphical representations, then perform comparisons and analyses."""

from collections import defaultdict

import networkx as nx


def mol_to_graph(mol):
    """Convert an RDKit molecule to a NetworkX graph with atom features.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.

    Returns
    -------
    G : networkx.Graph
        NetworkX graph where nodes correspond to atom indices and edges correspond to bonds.
        Each node contains the following attributes:
            - symbol (str): Atomic symbol (e.g., 'C', 'O', 'N').
            - degree (int): Number of bonds (degree) for the atom.
    """
    G = nx.Graph()

    atom_degree = defaultdict(lambda: 0)
    for b in mol.GetBonds():
        atom_degree[b.GetBeginAtomIdx()] += 1
        atom_degree[b.GetEndAtomIdx()] += 1

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        G.add_node(
            idx,
            symbol=atom.GetSymbol(),
            degree=atom_degree[idx],
        )

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        G.add_edge(begin_idx, end_idx)

    return G


def get_atom_environment(graph, atom_idx):
    """Get the chemical environment of an atom, including its neighbors.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph of the molecule.
    atom_idx : int
        Index of the atom (node index in the graph).

    Returns
    -------
    environment : tuple
        Tuple describing the atom's environment:
            (symbol, degree, tuple of sorted neighbor symbols)
        where:
            - symbol (str): Atomic symbol of the atom.
            - degree (int): Number of bonds (degree) for the atom.
            - tuple of neighbor symbols (tuple of str): Sorted atomic symbols of neighboring atoms.
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


def implicit_hydrogen_atom_mapping(mol):
    """Map atom indices to indices in implicit molecule case.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule with explicit hydrogens

    Returns
    -------
    dict
        A dictionary mapping atom indices from `mol` to atom indices when hydrogens are removed.
    """
    atom_mapping = defaultdict()
    i = 0
    for atm in mol.GetAtoms():
        if atm.GetSymbol() != "H":
            atom_mapping[atm.GetIdx()] = i
            i += 1
    return atom_mapping


def find_atom_mapping(mol1, mol2):
    """Find a mapping between atom indices of two molecules based on element symbols and connectivity.

    Parameters
    ----------
    mol1 : rdkit.Chem.rdchem.Mol
        The original RDKit molecule.
    mol2 : rdkit.Chem.rdchem.Mol
        The target RDKit molecule to map onto.

    Returns
    -------
    dict
        A dictionary mapping atom indices from `mol1` to atom indices in `mol2`.

    Raises
    ------
    ValueError
        If the number of atoms or atom environments do not match between the molecules.
    """

    orig_graph = mol_to_graph(mol1)
    correct_graph = mol_to_graph(mol2)
    if orig_graph.number_of_nodes() != correct_graph.number_of_nodes():
        raise ValueError("Number of atoms do not match in provided molecules.")

    orig_environments = {}
    for atom_idx in orig_graph.nodes():
        orig_environments[atom_idx] = get_atom_environment(orig_graph, atom_idx)

    correct_environments = {}
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

    mapping = {}
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


def validate_mapping(orig_graph, correct_graph, mapping):
    """Validate that a given atom mapping preserves the connectivity between two molecular graphs.

    Parameters
    ----------
    orig_graph : networkx.Graph
        The graph representation of the original molecule.
    correct_graph : networkx.Graph
        The graph representation of the target molecule.
    mapping : dict
        A dictionary mapping node indices from `orig_graph` to `correct_graph`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the mapping does not preserve the connectivity between the two graphs.

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


def find_molecular_rings(mol_graph, min_ring_size=3, max_ring_size=12):
    """
    Find all rings (cycles) in a molecular graph represented as a NetworkX graph.

    Parameters:
    -----------
    mol_graph : nx.Graph
        NetworkX graph where nodes represent atomic indices and edges represent bonds
    min_ring_size : int, default=3
        Minimum ring size to consider
    max_ring_size : int, default=12
        Maximum ring size to consider (helps avoid very large cycles)

    Returns:
    --------
    list[tuple[int, ...]]
        List of tuples, each containing atomic indices forming a ring,
        sorted by ring size (smallest first)

    Example:
    --------
    >>> import networkx as nx
    >>> # Create a simple 6-membered ring (benzene-like)
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)])
    >>> rings = find_molecular_rings(G)
    >>> print(rings)  # [(0, 1, 2, 3, 4, 5)]
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
                min_idx = cycle.index(min(cycle))
                normalized_cycle = tuple(cycle[min_idx:] + cycle[:min_idx])
                cycles.append(normalized_cycle)

    except Exception as e:
        print(f"Error finding cycles: {e}")
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
