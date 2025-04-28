import numpy as np


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


def get_geometry(
    mol, central_idx, tol=10, tol_planarity=0.1, tol_count_in_plane=0.1, verbose=False
):
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
    angles = np.array(angles)
    avg_angle = np.mean(angles)
    n = len(neighbors)

    # Planarity test using PCA
    _, s, _ = np.linalg.svd(positions)
    planarity_score = s[-1]  # smallest singular value ~ deviation from plane

    # Heuristic classification
    def num_close_angles(angles, target):
        return sum(np.isclose(angles, target * np.ones(len(angles)), atol=tol))

    print(n, len(angles), np.sort(angles))
    if verbose:
        print(
            "Number of Bonds",
            n,
            "planarity score",
            planarity_score,
            "is planar",
            planarity_score < tol_planarity,
        )
    if n == 2:
        if np.allclose(avg_angle, 180, atol=tol):
            return "Linear"
        else:
            return "Bent"
    elif n == 3:
        if planarity_score < tol_planarity:
            if np.all(angles > 40):
                return "Trigonal Planar"
        elif num_close_angles(angles, 90) == 2:
            return "T-Shaped"
        elif np.all(angles > 40):
            return "Trigonal Pyramidal"
    elif n == 4:
        if planarity_score < tol_planarity and num_close_angles(angles, 90) == 4:
            return "Square Planar"
        elif num_close_angles(angles, 109.5) == len(angles):
            return "Tetrahedral"
        elif num_close_angles(angles, 90) >= 4 and num_close_angles(angles, 170) == 1:
            return "Disphenoidal"
    elif n == 5:
        if num_close_angles(angles, 90) == 8 and num_close_angles(angles, 180) == 2:
            return "Square Pyramidal"
        elif num_close_angles(angles, 90) == 6 and num_close_angles(angles, 180) == 1:
            return "Trigonal Bipyramidal"
    elif n == 6:
        if num_close_angles(angles, 90) == 12:
            return "Octahedral"
        elif sum(np.logical_and(angles > 125, angles < 155)) >= 4:
            return "Trigonal Prismatic"
        elif num_close_angles(angles, 72) >= 5 and num_close_angles(angles, 90) >= 5:
            return "Pentagonal Pyramidal"
    elif n == 7:
        return "Pentagonal Bipyamidal"
    elif n == 10:
        return "Ferrocene"
    else:
        return f"Geometry with {n} neighbors: avg angle {avg_angle:.2f}, planarity {planarity_score:.4f}"

    return "Undetermined"
