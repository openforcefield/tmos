import warnings

import numpy as np

from .reference_values import ideal_angles


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


def get_geometry_from_xyz(
    positions, central_idx, r_cut=2.5, tol=15, verbose=False, ignore_scale=False
):
    """Determine the bonded geometry of a central atom based on atomic positions from xyz coordinates.

    Args:
        positions (numpy.ndarray): Array of atomic positions (shape: Nx3).
        central_idx (int): Index of the central atom in the positions array.
        r_cut (float, optional): Distance cutoff to filter neighboring atoms. Defaults to 2 Angstroms.
        tol (float, optional): Tolerance for angle comparison in degrees. Defaults to 15.
        verbose (bool, optional): If True, prints additional debugging information. Defaults to False.
        ignore_scale (bool, optional): If True, the warning that the minimum bond is not between 0.8 and 1.5
        Å is ignored.

    Returns:
        str: The determined geometry of the central atom, or an empty string if undetermined.
    """

    central_pos = positions[central_idx]
    dist = np.linalg.norm(positions - central_pos, axis=-1)
    dist[dist == 0] = np.nan
    if (np.nanmin(dist) < 0.8 or np.nanmin(dist) > 1.5) and not ignore_scale:
        raise ValueError(
            f"Bond distances are peculiar for coordinates in Å. Min distance is {np.nanmin(dist)}. "
            "If the coordinate scale is intentional, consider setting `ignore_scale==True` and adjusting"
            " `r_cut`"
        )

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
                np.arccos(
                    np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                        -1.0,
                        1.0,
                    )
                )
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
                np.arccos(
                    np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                        -1.0,
                        1.0,
                    )
                )
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
            warnings.warn(
                f"This {n}-coordinate center is closest to {geometry} but not within tolerance."
            )
            return "Undetermined"
        else:
            return geometry
