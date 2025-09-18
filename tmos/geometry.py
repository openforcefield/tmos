import numpy as np
from loguru import logger

from .reference_values import ideal_angles


def get_coordinates(mol, index):
    """Get the 3D coordinates of an atom from an RDKit molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    index : int
        Index of the atom whose coordinates are to be retrieved.

    Returns
    -------
    numpy.ndarray
        A 1D array of shape (3,) containing the x, y, z coordinates of the atom.
    """

    atm = mol.GetConformer().GetAtomPosition(index)
    return np.array([atm.x, atm.y, atm.z])


def get_distance(mol, ind1, ind2):
    """
    Get the distance between two atoms.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule.
    ind1 : int
        Index of the first atom.
    ind2 : int
        Index of the second atom.

    Returns
    -------
    float
        Distance between atoms.
    """

    return np.linalg.norm(get_coordinates(mol, ind1) - get_coordinates(mol, ind2))


def get_geometry_from_xyz(
    positions, central_idx, r_cut=2.5, tol=15, ignore_scale=False
):
    """Determine the bonded geometry of a central atom based on atomic positions from xyz coordinates.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of atomic positions with shape (N, 3).
    central_idx : int
        Index of the central atom in the positions array.
    r_cut : float, optional
        Distance cutoff (in Å) to filter neighboring atoms. Defaults to 2.5.
    tol : float, optional
        Tolerance for angle comparison in degrees. Defaults to 15.
    ignore_scale : bool, optional
        If True, ignores the warning when the minimum bond is not between 0.8 and 1.5 Å.

    Returns
    -------
    str or tuple
        The determined geometry of the central atom as a string, or a tuple (geometry, n) where n is the number of neighbors.
        Returns an empty string if undetermined.

    Raises
    ------
    ValueError
        If the minimum bond distance is outside the expected range and `ignore_scale` is False.

    Notes
    -----
    This function uses heuristics based on pairwise angles between neighboring atoms to classify the geometry.
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
    if n == 1:
        return "Monocoordinate", n
    elif n == 0:
        return "Element", n

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
            return "Linear", n
        else:
            return "Bent", n
    elif n == 10:
        return "Ferrocene", n
    else:
        scores = {
            key: np.mean(np.abs(angles - value))
            for key, value in ideal_angles[n].items()
        }
        geometry = min(scores, key=scores.get)
        logger.debug(f"Geometry scores: {scores}")
        if scores[geometry] > tol:
            logger.info(
                f"This {n}-coordinate center is closest to {geometry} but not within tolerance."
            )
            return "Undetermined", n
        else:
            return geometry, n


def get_geometry_from_mol(mol, central_idx, tol=15):
    """
    Determine the bonded geometry of a central atom based on atomic positions from an RDKit molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    central_idx : int
        Index of the central atom in the molecule.
    tol : float, optional, default=15
        Tolerance for angle comparison in degrees.

    Returns
    -------
    geometry_name : str
        The determined geometry of the central atom, or "Undetermined" if not within tolerance.
    n : int
        Number of neighboring atoms.
    """

    conf = mol.GetConformer()
    central_pos = np.array(conf.GetAtomPosition(central_idx))
    atom = mol.GetAtomWithIdx(central_idx)
    neighbors = atom.GetNeighbors()
    n = len(neighbors)

    if n == 1:
        return "Monocoordinate", n
    elif n == 0:
        return "Element", n

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
    # Heuristic classification
    if n == 2:
        if np.allclose(avg_angle, 180, atol=tol):
            return "Linear", n
        else:
            return "Bent", n
    elif n == 10:
        return "Ferrocene", n
    else:
        scores = {
            key: np.mean(np.abs(angles - value))
            for key, value in ideal_angles[n].items()
        }
        geometry = min(scores, key=scores.get)
        logger.debug(scores)
        if scores[geometry] > tol:
            logger.warning(
                f"This {n}-coordinate center is closest to {geometry} but not within tolerance."
            )
            return "Undetermined", n
        else:
            return geometry, n
