import numpy as np

ideal_angles = {
    3: {  # 3 angles
        "Trigonal Planar": np.array([120, 120, 120]),
        "Trigonal Pyramidal": np.array([107, 107, 107]),
        "T-Shaped": np.array([90, 90, 180]),
    },
    4: {  # 6 angles
        "Square Planar": np.array([90, 90, 90, 90, 180, 180]),
        "Tetrahedral": np.array([109.5, 109.5, 109.5, 109.5, 109.5, 109.5]),
        "Disphenoidal": np.array([90, 90, 90, 90, 120, 180]),
    },
    5: {  # 10 angles
        "Square Pyramidal": np.array([90, 90, 90, 90, 90, 90, 90, 90, 180, 180]),
        "Trigonal Bipyramidal": np.array([90, 90, 90, 90, 90, 90, 120, 120, 120, 180]),
        "Pentagonal Planar": np.array([72, 72, 72, 72, 72, 144, 144, 144, 144, 144]),
    },
    6: {  # 15 angles
        "Octahedral": np.array(
            [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 180, 180, 180]
        ),
        "Trigonal Prismatic": np.array(
            [60, 60, 60, 60, 60, 60, 90, 90, 90, 120, 120, 120, 150, 150, 150]
        ),
        #        "Pentagonal Pyramidal": np.array([72, 72, 72, 72, 72, 90, 90, 90, 90, 90, 144, 144, 144, 144, 144]),
    },
    7: {  # 21 angles
        "Pentagonal Bipyamidal": np.array(
            [
                72,
                72,
                72,
                72,
                72,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
                144,
                144,
                144,
                144,
                144,
                180,
            ]
        ),
        "Capped Trigonal Prism": np.array(
            [
                60,
                60,
                60,
                60,
                60,
                60,
                90,
                90,
                90,
                90,
                90,
                90,
                120,
                120,
                120,
                120,
                120,
                120,
                120,
                120,
                120,
            ]
        ),
    },
}


def get_coordinates(mol, index):
    """Get atom coordinates

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule
        index (int): Index of an atom

    Returns:
        numpy.ndarray: Atomic coordinates
    """
    atm = mol.GetConformer().GetAtomPosition(index)
    return np.array([atm.x, atm.y, atm.z])


def get_distance(mol, ind1, ind2):
    """Get the distance between two atoms

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule
        ind1 (int): Index of an atom
        ind2 (int): Index of an atom

    Returns:
        float: Distance between atoms
    """

    return np.linalg.norm(get_coordinates(mol, ind1) - get_coordinates(mol, ind2))


def get_geometry_from_xyz(positions, central_idx, r_cut=2.5, tol=15, verbose=False):
    """Determine the bonded geometry of a central atom based on atomic positions from xyz coordinates.

    Args:
        positions (numpy.ndarray): Array of atomic positions (shape: Nx3).
        central_idx (int): Index of the central atom in the positions array.
        r_cut (float, optional): Distance cutoff to filter neighboring atoms. Defaults to 2 Angstroms.
        tol (float, optional): Tolerance for angle comparison in degrees. Defaults to 15.
        verbose (bool, optional): If True, prints additional debugging information. Defaults to False.

    Returns:
        str: The determined geometry of the central atom, or an empty string if undetermined.
    """

    central_pos = positions[central_idx]
    dist = np.linalg.norm(positions - central_pos, axis=-1)
    positions = np.array(positions[np.where(np.logical_and(dist < r_cut, dist > 0))[0]])
    positions -= np.array(central_pos)

    n = len(positions)
    if n < 2:
        return ""

    # Compute all pairwise angles
    angles = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            v1, v2 = positions[i], positions[j]
            angle = np.degrees(
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            )
            angles.append(angle)
    angles = np.sort(np.array(angles))
    avg_angle = np.mean(angles)

    # Heuristic classification
    if n == 2:
        if np.allclose(avg_angle, 180, atol=tol):
            return "Linear"
        else:
            return "Bent"
    elif n == 10:
        return "Ferrocene"
    else:
        scores = {
            key: np.mean(np.abs(angles - value))
            for key, value in ideal_angles[n].items()
        }
        geometry = min(scores, key=scores.get)
        if verbose:
            print(scores)
        if scores[geometry] > tol:
            print(
                f"This {n}-coordinate center is closest to {geometry} but not within tolerance."
            )
            return "Undetermined"
        else:
            return geometry


def get_geometry_from_mol(mol, central_idx, tol=15, verbose=False):
    """Determine the bonded geometry of a central atom based on atomic positions from an RDKit molecule coordinates.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule
        central_idx (int): Index of the central atom in the positions array.
        tol (float, optional): Tolerance for angle comparison in degrees. Defaults to 15.
        verbose (bool, optional): If True, prints additional debugging information. Defaults to False.

    Returns:
        str: The determined geometry of the central atom, or an empty string if undetermined.
    """

    conf = mol.GetConformer()
    central_pos = np.array(conf.GetAtomPosition(central_idx))
    atom = mol.GetAtomWithIdx(central_idx)
    neighbors = atom.GetNeighbors()

    if len(neighbors) < 2:
        return ""

    positions = [
        np.array(conf.GetAtomPosition(n.GetIdx())) - central_pos for n in neighbors
    ]

    # Compute all pairwise angles
    angles = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            v1, v2 = positions[i], positions[j]
            angle = np.degrees(
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            )
            angles.append(angle)
    angles = np.sort(np.array(angles))
    avg_angle = np.mean(angles)
    n = len(neighbors)

    # Heuristic classification
    if n == 2:
        if np.allclose(avg_angle, 180, atol=tol):
            return "Linear"
        else:
            return "Bent"
    elif n == 10:
        return "Ferrocene"
    else:
        scores = {
            key: np.mean(np.abs(angles - value))
            for key, value in ideal_angles[n].items()
        }
        geometry = min(scores, key=scores.get)
        if verbose:
            print(scores)
        if scores[geometry] > tol:
            print(
                f"This {n}-coordinate center is closest to {geometry} but not within tolerance."
            )
            return "Undetermined"
        else:
            return geometry
