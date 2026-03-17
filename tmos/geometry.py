import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from loguru import logger
from typing import TypeAlias

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

GeometryResult: TypeAlias = tuple[str, int]
GeometryFromMolResult: TypeAlias = tuple[str, int, Chem.rdchem.Mol]


def _to_float(value: object, default: float) -> float:
    """Convert one value to ``float`` with fallback.

    Parameters
    ----------
    value : object
        Candidate value.
    default : float
        Fallback value when conversion fails.

    Returns
    -------
    float
        Converted value or ``default``.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _to_int(value: object, default: int) -> int:
    """Convert one value to ``int`` with fallback.

    Parameters
    ----------
    value : object
        Candidate value.
    default : int
        Fallback value when conversion fails.

    Returns
    -------
    int
        Converted value or ``default``.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _to_str(value: object, default: str) -> str:
    """Convert one value to ``str`` with fallback.

    Parameters
    ----------
    value : object
        Candidate value.
    default : str
        Fallback value when conversion fails.

    Returns
    -------
    str
        Converted value or ``default``.
    """
    if isinstance(value, str):
        return value
    return default


def get_coordinates(mol: Chem.rdchem.Mol, index: int) -> np.ndarray:
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


def get_distance(mol: Chem.rdchem.Mol, ind1: int, ind2: int) -> float:
    """Get the distance between two atoms.

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

    return float(
        np.linalg.norm(get_coordinates(mol, ind1) - get_coordinates(mol, ind2))
    )


def get_geometry_from_xyz(
    positions: np.ndarray,
    central_idx: int,
    r_cut: float = 2.5,
    tol: float = 15,
    ignore_scale: bool = False,
) -> GeometryResult:
    """Assign local coordination geometry from Cartesian coordinates.

    Parameters
    ----------
    positions : numpy.ndarray
        Atomic positions with shape ``(N, 3)`` in Å.
    central_idx : int
        Index of the central atom in the positions array.
    r_cut : float, default=2.5
        Distance cutoff (in Å) to filter neighboring atoms.
    tol : float, default=15
        Tolerance for angle comparison in degrees.
    ignore_scale : bool, default=False
        If True, ignores the warning when the minimum bond is not between 0.8 and 1.5 Å.

    Returns
    -------
    tuple of (str, int)
        ``(geometry_label, n_neighbors)``.

    Raises
    ------
    ValueError
        If the shortest bond-like distance is outside expected Å scale and
        ``ignore_scale`` is ``False``.

    Notes
    -----
    Uses pairwise-angle heuristics against ideal-angle templates.
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

    n: int = len(positions)
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
    avg_angle = float(np.mean(angles))

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
            key: float(np.mean(np.abs(angles - value)))
            for key, value in ideal_angles[n].items()
        }
        geometry = min(scores, key=lambda key: scores[key])
        logger.debug(f"Geometry scores: {scores}")
        if scores[geometry] > tol:
            logger.info(
                f"This {n}-coordinate center is closest to {geometry} but not within tolerance."
            )
            return "Undetermined", n
        else:
            return geometry, n


def get_neighbor_angles(positions: np.ndarray) -> np.ndarray:
    """Calculate all pairwise neighbor angles.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of neighbor vectors with shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Sorted pairwise angles (degrees) for all unique vector pairs.

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


def get_angle_similarity(
    angles1: np.ndarray,
    angles2: np.ndarray,
    metric: str = "rmsd",
    max_angle: float = 90,
) -> float:
    """Compute similarity between two angle sets.

    Parameters
    ----------
    angles1 : numpy.ndarray
        First angle vector in degrees.
    angles2 : numpy.ndarray
        Second angle vector in degrees. Must match ``angles1`` length.
    metric : str, default="rmsd"
        Similarity metric to use. Options are:

        - "gaussian kernel": Computes similarity using a Gaussian kernel based on squared differences.
        - "rmsd": Computes similarity as 1 minus the root-mean-square deviation normalized by `max_angle`.

    max_angle : float, default=90
        Maximum angle value (in degrees) used for normalization in the similarity calculation.

    Returns
    -------
    float
        Similarity score where larger values indicate closer agreement.

    Raises
    ------
    ValueError
        If the specified metric is not recognized.

    Notes
    -----
    Inputs are sorted before scoring, so permutation order does not affect the
    result.
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


def root_mean_squared_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    scale_factor: float = 1,
) -> float:
    """Return normalized RMS similarity between two vectors.

    Parameters
    ----------
    vec1 : numpy.ndarray
        First vector.
    vec2 : numpy.ndarray
        Second vector. Must have the same length as ``vec1``.
    scale_factor : float, default=1
        Scaling factor used to normalize the root-mean-square deviation.
        Defaults to 1.

    Returns
    -------
    float
        Similarity score ``1 - RMSD/scale_factor``.

    Notes
    -----
    Choose ``scale_factor`` to set the expected deviation scale.
    """
    return 1 - np.sqrt(np.mean(np.square(vec1 - vec2))) / scale_factor


def orientation_tensor_eigs(
    coords: np.ndarray,
    central_idx: int,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Compute orientation-tensor eigenmetrics for one coordination shell.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates with shape ``(N+1, 3)``.
    central_idx : int
        The index of the center of geometry that should be used as a reference
        for forming vectors.

    Returns
    -------
    eigs : numpy.ndarray
        Eigenvalues sorted descending.
    planarity : float
        λ1 - λ2, higher means more planar (two dominant axes).
    asphericity : float
        λ1 - 0.5*(λ2 + λ3), measures deviation from spherical symmetry.
    tensor : numpy.ndarray
        Full orientation tensor.

    Raises
    ------
    ValueError
        If coordinate shape is invalid or a neighbor overlaps the center.
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


def get_geometry_from_posym(
    rdmol: Chem.rdchem.Mol,
    tol: float = 1e-6,
    match_tol: float = 50,
) -> str:
    """Assign a Schoenflies point group using `posym` best-match scoring.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        Molecule with 3D coordinates.
    tol : float, default=1e-6
        Numerical tolerance used to collect equivalent best matches.
    match_tol : float, default=50
        Reserved argument for API compatibility.

    Returns
    -------
    str
        Selected Schoenflies point-group symbol.
    """
    coordinates = rdmol.GetConformers()[0].GetPositions()
    symbols = [a.GetSymbol() for a in rdmol.GetAtoms()]

    sym_groups: list[str] = list(geometry_point_group[len(symbols) - 1].keys())
    measures = {
        sym: SymmetryMolecule(
            group=sym, coordinates=coordinates, symbols=symbols
        ).measure
        for sym in sym_groups
    }
    min_value = min(measures.values())

    logger.debug(f"Point Group Measures: {measures}")
    point_groups: list[str] = [
        key for key, val in measures.items() if tol > val - min_value
    ]
    if len(point_groups) == 1:
        point_group: str = point_groups[0]
    elif len(symbols) - 1 == 3:
        point_group: str = [x for x in ["C3v", "D3h", "C2v"] if x in point_groups][
            0
        ]  # provides preference
    else:
        point_group: str = point_groups[0]

    return point_group


def get_geometry_from_pymatgen(
    rdmol: Chem.rdchem.Mol,
    tolerance: float = 0.3,
    eigen_tolerance: float = 0.2,
    matrix_tolerance: float = 0.1,
) -> str:
    """Assign a Schoenflies point group using `pymatgen` symmetry analysis.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        Molecule with 3D coordinates.
    tolerance : float, default=0.3
        Geometric matching tolerance.
    eigen_tolerance : float, default=0.2
        Tolerance for symmetry-operation eigenvalue checks.
    matrix_tolerance : float, default=0.1
        Tolerance for rotation-matrix matching to ideal forms.

    Returns
    -------
    str
        Schoenflies point-group symbol.
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
    point_group: str = pga.get_pointgroup().sch_symbol
    return point_group


def rdmol_to_unit_coordinates(rdmol: Chem.rdchem.Mol, central_idx: int) -> np.ndarray:
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


def get_geometry_from_rylm(
    rdmol: Chem.rdchem.Mol,
    central_idx: int,
    metric: str = "cosine",
) -> str:
    """Assign geometry label from rotational fingerprints (`rylm`).

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        Molecule with 3D coordinates.
    central_idx : int
        Index of the center atom.
    metric : str, default="cosine"
        Similarity metric accepted by :class:`rylm.rylm.Similarity`.

    Returns
    -------
    str
        Geometry label with highest fingerprint similarity score.

    Examples
    --------
    >>> # label = get_geometry_from_rylm(mol, central_idx=0)
    >>> # isinstance(label, str)
    >>> # True
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
        geometry = max(similarity, key=lambda key: similarity[key])
    except Exception as e:
        raise ValueError(f"A geometry returned None: {similarity}:\n{e}")

    return geometry


def get_geometry_from_angles(
    rdmol: Chem.rdchem.Mol,
    central_idx: int = 0,
    tol: float = 0.5,
    kwargs_angles: dict[str, object] | None = None,
    kwargs_eig: dict[str, object] | None = None,
) -> str:
    """Determine geometry from angle and orientation similarity scores.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    central_idx : int, default=0
        Index of the central atom in the molecule.
    tol : float, default=0.5
        Tolerance for angle comparison in degrees.
    kwargs_angles : dict of str to object or None, default=None
        Keyword arguments for :func:`get_angle_similarity`.
    kwargs_eig : dict of str to object or None, default=None
        Keyword arguments for :func:`root_mean_squared_similarity`.

    Returns
    -------
    str
        The determined geometry of the central atom, or "Undetermined" if not within tolerance.

    Notes
    -----
    Final scores average angle similarity and orientation-eigenvalue
    similarity for each candidate geometry.

    """
    kwargs_angles = {} if kwargs_angles is None else kwargs_angles
    kwargs_eig = {} if kwargs_eig is None else kwargs_eig

    metric = _to_str(kwargs_angles.get("metric", "rmsd"), "rmsd")
    max_angle = _to_float(kwargs_angles.get("max_angle", 90), 90.0)
    scale_factor = _to_float(kwargs_eig.get("scale_factor", 1), 1.0)

    coords = rdmol_to_unit_coordinates(rdmol, central_idx)
    n: int = len(coords) - 1
    angles = get_neighbor_angles(coords[1:])
    eigenvalues, _, _, _ = orientation_tensor_eigs(coords, central_idx=central_idx)

    scores: dict[str, list[float]] = {
        key: [
            float(
                get_angle_similarity(
                    angles,
                    value,
                    metric=metric,
                    max_angle=max_angle,
                )
            ),
            float(
                root_mean_squared_similarity(
                    eigenvalues,
                    coordinate_eigenvalues[n][key],
                    scale_factor=scale_factor,
                )
            ),
        ]
        for key, value in ideal_angles[n].items()
    }
    geometry: str = max(scores, key=lambda k: np.mean(scores[k]))

    logger.debug(angles)
    logger.debug(scores)
    if np.mean(scores[geometry]) < tol:
        logger.warning(
            f"This {n}-coordinate center is closest to {geometry} but not within tolerance {np.mean(scores[geometry])} < tol={tol}."
        )
        geometry = "Undetermined"

    return geometry


def isolate_geometry_atoms(
    rdmol: Chem.rdchem.Mol,
    central_idx: int,
    normalize: bool = True,
) -> Chem.rdchem.Mol:
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
    normalize : bool, default=True
        If ``True``, scale neighbor vectors to unit length.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        A new RDKit molecule containing only the central atom and its direct neighbors.

    """
    atoms_to_keep = [central_idx] + [
        atom.GetIdx() for atom in rdmol.GetAtomWithIdx(central_idx).GetNeighbors()
    ]

    editable_mol = Chem.EditableMol(Chem.Mol())
    new_tmc = 0
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
    pos = old_rdmol.GetConformer().GetAtomPosition(central_idx)
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    vec0 = np.array([pos.x, pos.y, pos.z])
    for i, idx in enumerate(atoms_to_keep[1:]):
        pos = old_rdmol.GetConformer().GetAtomPosition(idx)
        vec = np.array([pos.x, pos.y, pos.z]) - vec0
        if normalize:
            vec /= np.linalg.norm(vec)
        pos = Point3D(*vec)
        conf.SetAtomPosition(i + 1, pos)

    rdmol.AddConformer(conf, assignId=True)
    return rdmol


def get_geometry_from_mol(
    rdmol: Chem.rdchem.Mol,
    central_idx: int,
    mode: str = "angles",
    kwargs_mode: dict[str, object] | None = None,
) -> GeometryFromMolResult:
    """Determine bonded geometry of a central atom from an RDKit molecule.

    Parameters
    ----------
    rdmol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    central_idx : int
        Index of the central atom in the molecule.
    mode : str, default="angles"
        Mode used to calculate the geometry label.

        - "posym": :func:`get_geometry_from_posym`
        - "pymatgen": :func:`get_geometry_from_pymatgen`
        - "rylm": :func:`get_geometry_from_rylm`
        - "angles": :func:`get_geometry_from_angles`

    kwargs_mode : dict of str to object or None, default=None
        Keyword arguments for the chosen calculation mode.

    Returns
    -------
    geometry_name : str
        The determined geometry of the central atom, or "Undetermined" if not within tolerance.
    n : int
        Number of neighboring atoms.
    rdmol : rdkit.Chem.rdchem.Mol
        The modified RDKit molecule with updated conformer.

    Raises
    ------
    ValueError
        If ``mode`` is unsupported or invalid keywords are provided for the chosen mode.

    Examples
    --------
    >>> # geometry, n, local = get_geometry_from_mol(mol, central_idx=0, mode="angles")
    >>> # n >= 0
    >>> # True
    """
    kwargs_mode = {} if kwargs_mode is None else kwargs_mode

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
        keywords: list[str] = ["tolerance", "eigen_tolerance", "matrix_tolerance"]
        if not all(x in keywords for x, _ in kwargs_mode.items()):
            raise ValueError(
                f"For calculation mode 'posym', the keywords are: {', '.join(keywords)}"
            )
        tolerance = _to_float(kwargs_mode.get("tolerance", 0.3), 0.3)
        eigen_tolerance = _to_float(kwargs_mode.get("eigen_tolerance", 0.2), 0.2)
        matrix_tolerance = _to_float(kwargs_mode.get("matrix_tolerance", 0.1), 0.1)
        point_group: str = get_geometry_from_pymatgen(
            rdmol,
            tolerance=tolerance,
            eigen_tolerance=eigen_tolerance,
            matrix_tolerance=matrix_tolerance,
        )
        geometry = geometry_point_group[len(atoms_to_keep) - 1].get(point_group, None)
    elif mode == "posym":
        keywords: list[str] = ["tol", "match_tol"]
        if not all(x in keywords for x, _ in kwargs_mode.items()):
            raise ValueError(
                f"For calculation mode 'posym', the keywords are: {', '.join(keywords)}"
            )
        posym_tol = _to_float(kwargs_mode.get("tol", 1e-6), 1e-6)
        match_tol = _to_float(kwargs_mode.get("match_tol", 50), 50.0)
        point_group: str = get_geometry_from_posym(
            rdmol,
            tol=posym_tol,
            match_tol=match_tol,
        )
        geometry = geometry_point_group[len(atoms_to_keep) - 1].get(point_group, None)
    elif mode == "rylm":
        keywords: list[str] = ["metric"]
        if not all(x in keywords for x, _ in kwargs_mode.items()):
            raise ValueError(
                f"For calculation mode 'rylm', the keywords are: {', '.join(keywords)}"
            )
        metric = _to_str(kwargs_mode.get("metric", "cosine"), "cosine")
        geometry = get_geometry_from_rylm(rdmol, 0, metric=metric)
    elif mode == "angles":
        angle_central_idx = _to_int(kwargs_mode.get("central_idx", 0), 0)
        angle_tol = _to_float(kwargs_mode.get("tol", 0.5), 0.5)
        angle_kwargs = kwargs_mode.get("kwargs_angles", None)
        eig_kwargs = kwargs_mode.get("kwargs_eig", None)
        parsed_angle_kwargs = angle_kwargs if isinstance(angle_kwargs, dict) else None
        parsed_eig_kwargs = eig_kwargs if isinstance(eig_kwargs, dict) else None
        geometry = get_geometry_from_angles(
            rdmol,
            central_idx=angle_central_idx,
            tol=angle_tol,
            kwargs_angles=parsed_angle_kwargs,
            kwargs_eig=parsed_eig_kwargs,
        )
    else:
        raise ValueError(f"Calculation mode, {mode}, is not supported.")

    if geometry is None:
        geometry = "Undetermined"

    return geometry, n, rdmol
