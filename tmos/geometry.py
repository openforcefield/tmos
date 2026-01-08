import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from loguru import logger

from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core.structure import Molecule
from posym import SymmetryMolecule
from rylm.rylm import Rylm, Similarity, Fingerprint

from .reference_values import (
    ideal_angles,
    coordinate_eigenvalues,
    geometry_point_group,
    steinhardt_order_parameters,
)


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


def get_neighbor_angles(positions):
    """
    Calculate the angles between all pairs of vectors in a given list of positions.

    Parameters
    ----------
    positions : list of array-like
        A list of vectors (e.g., 2D or 3D coordinates) represented as array-like objects.

    Returns
    -------
    numpy.ndarray
        A sorted array of angles (in degrees) between all unique pairs of vectors in the input list.

    Notes
    -----
    The angles are calculated using the dot product and the arccosine function, ensuring that
    the result is within the range [0, 180] degrees. The resulting angles are sorted in ascending
    order before being returned.
    """
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
    return angles


def get_angle_similarity(angles1, angles2, metric="rmsd", max_angle=90):
    """Calculate the similarity between two sets of angles using a specified metric.

    Parameters
    ----------
    angles1 : array-like
        First array of angles (in degrees).
    angles2 : array-like
        Second array of angles (in degrees). Must have the same length as `angles1`.
    metric : str, optional
        Similarity metric to use. Options are:

        - "gaussian kernel": Computes similarity using a Gaussian kernel based on squared differences.
        - "rmsd": Computes similarity as 1 minus the root-mean-square deviation normalized by `max_angle`.

        Defaults to "gaussian kernel".
    max_angle : float, optional
        Maximum angle value (in degrees) used for normalization in the similarity calculation.
        Defaults to 90.

    Returns
    -------
    float
        Similarity score between the two angle sets. Higher values indicate greater similarity.
        For "gaussian kernel", the value ranges from 0 (completely dissimilar) to 1 (identical).
        For "rmsd", the value also ranges from 0 to 1.

    Raises
    ------
    ValueError
        If the specified metric is not recognized.

    Notes
    -----
    Both input angle arrays are sorted before comparison to ensure consistent similarity evaluation
    regardless of the order of angles in the input arrays.
    """
    angles1, angles2 = np.sort(angles1), np.sort(angles2)
    if metric == "gaussian kernel":
        similarity = np.exp(-np.sum(np.square(angles1 - angles2)) / max_angle**2)
    elif metric == "rmsd":
        similarity = root_mean_squared_similarity(
            angles1, angles2, scale_factor=max_angle
        )
    else:
        raise ValueError(f"Similarity method, {metric}, is not recognized.")

    return similarity


def root_mean_squared_similarity(vec1, vec2, scale_factor=1):
    """Calculate the root-mean-squared similarity between two vectors.

    Parameters
    ----------
    vec1 : array-like
        First vector or array of values.
    vec2 : array-like
        Second vector or array of values. Must have the same length as `vec1`.
    scale_factor : float, optional
        Scaling factor used to normalize the root-mean-square deviation.
        Defaults to 1.

    Returns
    -------
    float
        Similarity score between the two vectors, calculated as 1 minus the normalized
        root-mean-square deviation. Values range from 0 (completely dissimilar) to 1 (identical).
        When vectors are identical, the similarity is 1. As the RMS deviation increases,
        the similarity approaches 0.

    Notes
    -----
    The similarity is computed as:

    .. math::
        \\text{similarity} = 1 - \\frac{\\sqrt{\\text{mean}((\\text{vec1} - \\text{vec2})^2)}}{\\text{scale_factor}}

    The `scale_factor` parameter is used to normalize the RMS deviation to a [0, 1] range.
    Typically, `scale_factor` should be set to the maximum expected deviation between vectors.
    """
    return 1 - np.sqrt(np.mean(np.square(vec1 - vec2))) / scale_factor


def orientation_tensor_eigs(coords, central_idx):
    """
    Compute the eigenvalues and shape metrics of the orientation tensor
    for a set of neighbor atoms around a metal center.

    Parameters
    ----------
    coords : (N+1, 3) array-like
        Atomic coordinates. The first row is the metal center,
        and the remaining rows are the bonded neighbor atoms.
    central_idx : int
        The index of the center of geometry that should be used as a reference
        for forming vectors.

    Returns
    -------
    eigs : (3,) ndarray
        Eigenvalues of the orientation tensor, sorted descending (λ1 ≥ λ2 ≥ λ3).
    planarity : float
        λ1 - λ2, higher means more planar (two dominant axes).
    asphericity : float
        λ1 - 0.5*(λ2 + λ3), measures deviation from spherical symmetry.
    tensor : (3,3) ndarray
        The full orientation tensor (for optional diagnostics).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N+1, 3) with first row as metal center.")

    center = coords[central_idx]
    vecs = (
        np.concatenate([coords[:central_idx], coords[central_idx + 1 :]], axis=0)
        - center
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("At least one neighbor is at the same position as the center.")
    unitvecs = vecs / norms

    tensor = (unitvecs.T @ unitvecs) / len(unitvecs)
    eigs = np.linalg.eigvalsh(tensor)[::-1]
    planarity = eigs[0] - eigs[1]
    asphericity = eigs[0] - 0.5 * (eigs[1] + eigs[2])

    return eigs, planarity, asphericity, tensor


def get_geometry_from_posym(rdmol, tol=1e-6, match_tol=50):
    """
    Analyze the point group geometry of a given RDKit molecule using posymm

    Args:
        rdmol (rdkit.Chem.rdchem.Mol): The RDKit molecule to analyze.
        tol (float, optional): If the percent match of different geometries is within this tolerance, they will be
        considered equal. Defaults to 1e-6.
        match_tol (float, optional): A percent measure of a match must be below this value to be considered a match.

    Returns:
        point_group (str): The Schoenflies symbol of the point group.

    """
    coordinates = rdmol.GetConformers()[0].GetPositions()
    symbols = [a.GetSymbol() for a in rdmol.GetAtoms()]

    sym_groups = list(geometry_point_group[len(symbols) - 1].keys())
    measures = {
        sym: SymmetryMolecule(
            group=sym, coordinates=coordinates, symbols=symbols
        ).measure
        for sym in sym_groups
    }
    min_value = min(measures.values())

    logger.debug(f"Point Group Measures: {measures}")
    point_groups = [key for key, val in measures.items() if tol > val - min_value]
    if len(point_groups) == 1:
        point_group = point_groups[0]
    elif len(symbols) - 1 == 3:
        point_group = [x for x in ["C3v", "D3h", "C2v"] if x in point_groups][
            0
        ]  # provides preference
    else:
        point_group = point_groups[0]

    return point_group


def get_geometry_from_pymatgen(
    rdmol, tolerance=0.3, eigen_tolerance=0.2, matrix_tolerance=0.1
):
    """
    Analyze the point group geometry of a given RDKit molecule.

    Args:
        rdmol (rdkit.Chem.rdchem.Mol): The RDKit molecule to analyze.
        tolerance (float, optional): Tolerance for point group analysis. Defaults to 0.3.
        eigen_tolerance (float, optional): Controls the numerical tolerance used when checking the eigenvalues
        of rotation matrices associated with symmetry operations.
        A proper symmetry operation in 3D can be represented by a 3×3 rotation matrix R.
        The eigenvalues of R encode the rotation angle (e.g., 1 for a 0° rotation, complex roots of unity for
        2-, 3-, 4-, 6-fold rotations, etc.). For example:

        - 2-fold rotation → eigenvalues {1, -1, -1}
        - 3-fold rotation → eigenvalues {1, exp(±2πi/3)}

        Defaults to 0.2. PyMatGen default is 0.01.
        matrix_tolerance (float, optional): Controls the tolerance for comparing rotation matrices to their
        ideal integer forms. In perfect symmetry, rotation matrices should contain exact integers (like 0, ±1)
        for Cartesian or lattice directions. In practice, small floating-point deviations occur (e.g., 0.00002
        instead of 0, -0.99998 instead of -1). Defaults to 0.1.

    Returns:
        point_group (str): The Schoenflies symbol of the point group.
    """
    pymatgen_mol = Molecule(
        species=[atom.GetSymbol() for atom in rdmol.GetAtoms()],
        coords=[
            rdmol.GetConformer().GetAtomPosition(i) for i in range(rdmol.GetNumAtoms())
        ],
    )
    pga = PointGroupAnalyzer(
        pymatgen_mol,
        tolerance=tolerance,
        eigen_tolerance=eigen_tolerance,
        matrix_tolerance=matrix_tolerance,
    )
    point_group = pga.get_pointgroup().sch_symbol
    return point_group


def rdmol_to_unit_coordinates(rdmol, central_idx):
    """Convert RDKit molecule coordinates to unit vectors centered on a specific atom.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        RDKit molecule object with a conformer.
    central_idx : int
        Index of the central atom to use as the origin.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) where the first row is the central atom at the origin [0, 0, 0]
        and remaining rows are unit vectors pointing from the central atom to each neighbor.

    Notes
    -----
    This function reorders atoms so the central atom is first, translates coordinates to place
    the central atom at the origin, and normalizes all neighbor vectors to unit length.
    """
    coords = [
        rdmol.GetConformer().GetAtomPosition(i) for i in range(rdmol.GetNumAtoms())
    ]
    coords = np.array([[p.x, p.y, p.z] for p in coords], dtype=float)
    order = [central_idx] + [i for i in range(coords.shape[0]) if i != central_idx]
    coords = coords[order]
    coords -= coords[0]
    norms = np.linalg.norm(coords[1:], axis=1)[:, np.newaxis]
    norms[norms == 0] = 1.0
    coords[1:] /= norms

    return coords


def get_geometry_from_rylm(rdmol, central_idx, metric="cosine"):
    """
    Analyze the point group geometry of a given RDKit molecule.

    All fingerprints in tmos are generated with include_n_coord=True, include_w=True, frequencies=[4, 6, 8, 10, 12]

    Args:
        rdmol (rdkit.Chem.rdchem.Mol): The RDKit molecule to analyze. The first atom is expected
        to be the center.
        central_idx (int): Index of central atom
        metric (str): Distance metric accepted by :class:`rylm.rylm.Similarity`

    Returns:
        geometry (str): The Schoenflies symbol of the point group.
    """
    coords = rdmol_to_unit_coordinates(rdmol, central_idx)

    geometry_fingerprints_dict = steinhardt_order_parameters[len(coords) - 1]
    geometry_fingerprints = {
        geom: Fingerprint(**v) for geom, v in geometry_fingerprints_dict.items()
    }

    tmp_fp = next(iter(geometry_fingerprints.values()))
    rylm = Rylm(
        include_n_coord=tmp_fp.include_n_coord,
        include_w=tmp_fp.include_w,
        frequencies=tmp_fp.frequencies,
    )
    fingerprint = rylm.calculate(coords)

    similarity_metric = Similarity(
        metric=metric,
        normalize=True,
    )
    similarity = {
        geom: similarity_metric.calculate(fp, fingerprint)
        for geom, fp in geometry_fingerprints.items()
    }

    logger.debug("Fingerprint Similarity:")
    for geom, sim in similarity.items():
        logger.debug(f"    {geom}: {sim}")

    try:
        geometry = max(similarity, key=similarity.get)
    except Exception as e:
        raise ValueError(f"A geometry returned None: {similarity}:\n{e}")

    return geometry


def get_geometry_from_angles(
    rdmol, central_idx=0, metric="rmsd", tol=0.75, kwargs_angles={}, kwargs_eig={}
):
    """
    Determine the bonded geometry of a central atom based on atomic positions from an RDKit molecule.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    central_idx : int
        Index of the central atom in the molecule.
    tol : float, optional, default=0.75
        Tolerance for angle comparison in degrees.
    kwargs_angles : dict, optional, default={}
        Keyword arguments for :func:`get_angle_similarity`.
    kwargs_eig : dict, optional, default={}
        Keyword arguments for :func:`root_mean_squared_similarity`.

    Returns
    -------
    geometry_name : str
        The determined geometry of the central atom, or "Undetermined" if not within tolerance.

    """
    coords = rdmol_to_unit_coordinates(rdmol, central_idx)
    n = len(coords) - 1
    angles = get_neighbor_angles(coords[1:])
    eigenvalues, _, _, _ = orientation_tensor_eigs(coords, central_idx=central_idx)

    scores = {
        key: [
            float(get_angle_similarity(angles, value, **kwargs_angles)),
            float(
                root_mean_squared_similarity(
                    eigenvalues, coordinate_eigenvalues[n][key], **kwargs_eig
                )
            ),
        ]
        for key, value in ideal_angles[n].items()
    }
    geometry = max(scores, key=lambda k: np.mean(scores[k]))

    logger.debug(angles)
    logger.debug(scores)
    if np.mean(scores[geometry]) < tol:
        logger.warning(
            f"This {n}-coordinate center is closest to {geometry} but not within tolerance {np.mean(scores[geometry])} < tol={tol}."
        )
        geometry = "Undetermined"

    return geometry


def isolate_geometry_atoms(rdmol, central_idx):
    """Extract central atom and its neighbors as a new molecule with normalized geometry.

    The central atom is placed at the origin [0, 0, 0] and neighbor positions are
    converted to unit vectors. All atomic properties (aromaticity, formal charge,
    chirality, hybridization) are reset to default values, and all bonds are set
    to single bonds.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        RDKit molecule object with a conformer.
    central_idx : int
        Index of the central atom in the molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        A new RDKit molecule containing only the central atom and its direct neighbors.

    """
    atoms_to_keep = [central_idx] + [
        atom.GetIdx() for atom in rdmol.GetAtomWithIdx(central_idx).GetNeighbors()
    ]
    editable_mol = Chem.EditableMol(Chem.Mol())

    for idx in atoms_to_keep:
        atom = rdmol.GetAtomWithIdx(idx)
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        atom.SetHybridization(Chem.rdchem.HybridizationType.UNSPECIFIED)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        new_idx = editable_mol.AddAtom(atom)
        if idx == central_idx:
            new_tmc = new_idx
        else:
            editable_mol.AddBond(new_tmc, new_idx, Chem.BondType.SINGLE)

    # Finalize the new molecule
    old_rdmol = rdmol
    rdmol = editable_mol.GetMol()
    conf = Chem.Conformer(rdmol.GetNumAtoms())
    for i, idx in enumerate(atoms_to_keep):
        pos = old_rdmol.GetConformer().GetAtomPosition(idx)
        if i != 0:
            vec0 = np.array([pos.x, pos.y, pos.z])
            vec = np.array([pos.x, pos.y, pos.z]) - vec0
            vec /= np.linalg.norm(vec)
            pos = Point3D(*vec)
        else:
            pos = Point3D(0.0, 0.0, 0.0)
        conf.SetAtomPosition(i, pos)

    rdmol.AddConformer(conf, assignId=True)
    return rdmol


def get_geometry_from_mol(rdmol, central_idx, mode="angles", kwargs_mode={}):
    """
    Determine the bonded geometry of a central atom based on atomic positions from an RDKit molecule.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    central_idx : int
        Index of the central atom in the molecule.
    mode : str, optional
        Mode used to calculate the point group. Default is 'angles'.

        - "posym": :func:`get_geometry_from_posym`
        - "pymatgen": :func:`get_geometry_from_pymatgen`
        - "rylm": :func:`get_geometry_from_rylm`
        - "angles": :func:`get_geometry_from_angles`

    kwargs_mode : dict, optional, default={}
        Keyword arguments for the chosen calculation mode.
    tol : float, optional, default=0.8
        Tolerance for angle comparison in degrees.

    Returns
    -------
    geometry_name : str
        The determined geometry of the central atom, or "Undetermined" if not within tolerance.
    n : int
        Number of neighboring atoms.
    rdmol : rdkit.Chem.rdchem.Mol
        The modified RDKit molecule with updated conformer.
    """
    atoms_to_keep = [central_idx] + [
        atom.GetIdx() for atom in rdmol.GetAtomWithIdx(central_idx).GetNeighbors()
    ]
    rdmol = isolate_geometry_atoms(rdmol, central_idx)
    n = rdmol.GetNumAtoms() - 1

    if n == 1:
        return "Monocoordinate", n, rdmol
    elif n == 0:
        return "Element", n, rdmol

    if mode == "pymatgen":
        keywords = ["tolerance", "eigen_tolerance", "matrix_tolerance"]
        if not all(x in keywords for x, _ in kwargs_mode.items()):
            raise ValueError(
                f"For calculation mode 'posym', the keywords are: {', '.join(keywords)}"
            )
        point_group = get_geometry_from_pymatgen(rdmol, **kwargs_mode)
        geometry = geometry_point_group[len(atoms_to_keep) - 1].get(point_group, None)
    elif mode == "posym":
        keywords = ["tol", "match_tol"]
        if not all(x in keywords for x, _ in kwargs_mode.items()):
            raise ValueError(
                f"For calculation mode 'posym', the keywords are: {', '.join(keywords)}"
            )
        point_group = get_geometry_from_posym(rdmol, **kwargs_mode)
        geometry = geometry_point_group[len(atoms_to_keep) - 1].get(point_group, None)
    elif mode == "rylm":
        keywords = ["metric"]
        if not all(x in keywords for x, _ in kwargs_mode.items()):
            raise ValueError(
                f"For calculation mode 'rylm', the keywords are: {', '.join(keywords)}"
            )
        geometry = get_geometry_from_rylm(rdmol, 0, **kwargs_mode)
    elif mode == "angles":
        geometry = get_geometry_from_angles(rdmol, central_idx=0, metric="rmsd")
    else:
        raise ValueError(f"Calculation mode, {mode}, is not supported.")

    return geometry, n, rdmol
