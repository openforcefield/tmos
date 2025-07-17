from collections import defaultdict

import networkx as nx


def mol_to_graph(mol):
    """
    Convert RDKit molecule to NetworkX graph with atom features.
    Atom index = node index.

    Args:
        mol: RDKit molecule object

    Returns:
        NetworkX graph with atom indices as nodes and bonds as edges
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
    """
    Get the chemical environment of an atom including its neighbors.

    Args:
        graph: NetworkX graph of molecule
        atom_idx: Index of the atom (same as node index)

    Returns:
        Tuple describing the atom's environment
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


def find_atom_mapping(mol1, mol2):
    """
    Find mapping between atoms in original and correct molecule graphs using
    element symbols and connectivity patterns.

    Args:
        mol1 (rdkit.Chem.rdchem.Mol): RDKit molecule
        mol2 (rdkit.Chem.rdchem.Mol): RDKit molecule

    Returns:
        Dictionary mapping original atom indices to correct atom indices
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

    return validate_mapping(orig_graph, correct_graph, mapping)


def validate_mapping(orig_graph, correct_graph, mapping):
    """
    Validate that the atom mapping preserves graph connectivity.

    Args:
        orig_graph: NetworkX graph of original molecule
        correct_graph: NetworkX graph of correct molecule
        mapping: Dictionary mapping original atom indices to correct atom indices

    Returns:
        Atom mapping
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

    return mapping
