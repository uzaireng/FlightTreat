from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from .math_skeleton import (
    DegeneratePanelError,
    InvalidMeshConfigError,
    InvertedNormalError,
    MeshSingularityError,
    OpenSurfaceError,
    NormalOrientation,
    PanelMetrics3D,
)


def _closed_from_edges(facets: np.ndarray, tol: float = 1e-10) -> bool:
    # Geometric edge closure without Python loops. Quantize vertices to a stable
    # tolerance, then count each undirected edge twice.
    scale = max(1.0, float(np.ptp(facets.reshape(-1, 3), axis=0).max()))
    q = np.rint(facets / (tol * scale)).astype(np.int64)
    edges = np.concatenate(
        (
            q[:, [0, 1]],
            q[:, [1, 2]],
            q[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _orientation_flip(
    n_faces: int,
    face_a: np.ndarray,
    face_b: np.ndarray,
    same_direction: np.ndarray,
) -> np.ndarray:
    """Relative panel flips from a spanning forest of the dual graph.

    One SciPy breadth-first tree roots every component. Path XOR from that
    root is then folded with pointer jumping, which is O(F log F) array
    work and does not walk panels in Python. A consistent surface has one
    solution once the lowest-index face of each component stays unflipped.
    """

    n = int(n_faces)
    if n <= 0:
        raise InvalidMeshConfigError("Orientation repair requires at least one facet.")

    face_a = np.asarray(face_a, dtype=np.int64)
    face_b = np.asarray(face_b, dtype=np.int64)
    disagree = np.asarray(same_direction, dtype=bool)
    rows = np.concatenate((face_a, face_b))
    cols = np.concatenate((face_b, face_a))
    adjacency = csr_matrix(
        (np.ones(rows.shape[0], dtype=np.float64), (rows, cols)),
        shape=(n, n),
    )
    _components, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    if n > 1:
        first[1:] = sorted_labels[1:] != sorted_labels[:-1]
    seeds = np.asarray(order[first], dtype=np.int64)

    super_id = n
    seed_count = int(seeds.shape[0])
    rooted = csr_matrix(
        (
            np.ones(rows.shape[0] + 2 * seed_count, dtype=np.float64),
            (
                np.concatenate(
                    (rows, np.full(seed_count, super_id, dtype=np.int64), seeds)
                ),
                np.concatenate(
                    (cols, seeds, np.full(seed_count, super_id, dtype=np.int64))
                ),
            ),
        ),
        shape=(n + 1, n + 1),
    )
    _visit, pred = breadth_first_order(
        rooted,
        i_start=super_id,
        directed=False,
        return_predecessors=True,
    )
    parent = np.asarray(pred[:n], dtype=np.int64)
    if np.any(parent < 0):
        raise InvalidMeshConfigError(
            "Surface orientation graph has an unreachable panel."
        )

    lo = np.minimum(face_a, face_b)
    hi = np.maximum(face_a, face_b)
    key = lo * np.int64(n) + hi
    key_order = np.argsort(key, kind="mergesort")
    key_sorted = key[key_order]
    disagree_sorted = disagree[key_order]

    is_root = parent == super_id
    nodes = np.arange(n, dtype=np.int64)
    query = np.minimum(parent, nodes) * np.int64(n) + np.maximum(parent, nodes)
    pos = np.searchsorted(key_sorted, query)
    clipped = np.clip(pos, 0, max(int(key_sorted.size) - 1, 0))
    found = (pos < key_sorted.size) & (key_sorted[clipped] == query)
    if np.any(~is_root & ~found):
        raise InvalidMeshConfigError(
            "Orientation tree referenced a panel edge that is not shared."
        )

    edge_xor = np.zeros(n, dtype=bool)
    edge_xor[~is_root] = disagree_sorted[pos[~is_root]]
    jump_parent = parent.copy()
    jump_parent[is_root] = nodes[is_root]
    xor_jump = edge_xor.copy()
    steps = int(np.ceil(np.log2(n))) if n > 1 else 1
    for _ in range(steps):
        xor_jump = np.logical_xor(xor_jump, xor_jump[jump_parent])
        jump_parent = jump_parent[jump_parent]

    expected = np.logical_xor(xor_jump[face_a], disagree)
    if np.any(xor_jump[face_b] != expected):
        raise InvertedNormalError(
            "Surface winding cannot be made consistent across shared edges."
        )
    return xor_jump


def _consistent_winding(facets: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Flip triangles so every shared edge is traversed in opposite directions.

    A single signed-volume flip cannot repair a closed body whose caps and
    sides were wound independently. Neighbor agreement comes first; the
    caller then applies one global flip.
    """
    scale = max(1.0, float(np.ptp(facets.reshape(-1, 3), axis=0).max()))
    quantized = np.rint(facets / (tol * scale)).astype(np.int64)
    _unique, inverse = np.unique(quantized.reshape(-1, 3), axis=0, return_inverse=True)
    corner_ids = inverse.reshape(quantized.shape[0], 3)
    directed = np.stack(
        (corner_ids[:, [0, 1]], corner_ids[:, [1, 2]], corner_ids[:, [2, 0]]),
        axis=1,
    ).reshape(-1, 2)
    undirected = np.sort(directed, axis=1)
    face_ids = np.repeat(np.arange(quantized.shape[0], dtype=np.int64), 3)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    undirected = undirected[order]
    directed = directed[order]
    face_ids = face_ids[order]

    if undirected.shape[0] < 2:
        return facets
    change = np.any(undirected[1:] != undirected[:-1], axis=1)
    starts = np.concatenate((np.array([0], dtype=np.int64), np.flatnonzero(change) + 1))
    ends = np.concatenate((starts[1:], np.array([undirected.shape[0]], dtype=np.int64)))
    paired = np.flatnonzero((ends - starts) == 2)
    if paired.size == 0:
        return facets

    left = starts[paired]
    right = left + 1
    same_direction = np.all(directed[left] == directed[right], axis=1)
    face_a = face_ids[left]
    face_b = face_ids[right]
    cross = face_a != face_b
    if np.any((~cross) & same_direction):
        raise InvertedNormalError(
            "Surface winding cannot be made consistent across shared edges."
        )
    if not np.any(cross):
        return facets

    flip = _orientation_flip(
        int(quantized.shape[0]),
        face_a[cross],
        face_b[cross],
        same_direction[cross],
    )
    if not np.any(flip):
        return facets
    oriented = np.array(facets, copy=True, order="C")
    oriented[flip] = oriented[flip][:, [0, 2, 1], :]
    return oriented


def _triangle_geometry(
    facets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e1 = facets[:, 1] - facets[:, 0]
    e2 = facets[:, 2] - facets[:, 0]
    cross = np.cross(e1, e2)
    twice_area = np.linalg.norm(cross, axis=1)
    return e1, cross, twice_area, 0.5 * twice_area


def _subdivide_triangles(facets: np.ndarray) -> np.ndarray:
    """Split every triangle into four by edge midpoints. Vectorized, same winding."""

    corner_a = facets[:, 0]
    corner_b = facets[:, 1]
    corner_c = facets[:, 2]
    mid_ab = 0.5 * (corner_a + corner_b)
    mid_bc = 0.5 * (corner_b + corner_c)
    mid_ca = 0.5 * (corner_c + corner_a)
    child = np.stack(
        (
            np.stack((corner_a, mid_ab, mid_ca), axis=1),
            np.stack((corner_b, mid_bc, mid_ab), axis=1),
            np.stack((corner_c, mid_ca, mid_bc), axis=1),
            np.stack((mid_ab, mid_bc, mid_ca), axis=1),
        ),
        axis=0,
    )
    return np.ascontiguousarray(child.reshape(-1, 3, 3), dtype=np.float64)


def refine_surface_triangles(facets: np.ndarray, density: int) -> np.ndarray:
    """Return a copy of ``facets``, split when ``density`` is 2 or 4.

    ``1`` leaves the triangulation unchanged. ``2`` halves every edge.
    ``4`` halves them again. The caller owns the returned array.
    """

    level = int(density)
    if level not in (1, 2, 4):
        raise InvalidMeshConfigError("surface_density must be 1, 2, or 4.")
    out = np.array(facets, dtype=np.float64, copy=True, order="C")
    if level == 1:
        return out
    out = _subdivide_triangles(out)
    if level == 4:
        out = _subdivide_triangles(out)
    return out


def build_panel_metrics_3d(
    facets: np.ndarray,
    *,
    normal_orientation: NormalOrientation = "outward",
    min_panel_area: float = 1e-14,
    require_closed: bool = False,
    is_closed_hint: bool = False,
) -> PanelMetrics3D:
    f = np.asarray(facets, dtype=np.float64)
    if f.ndim != 3 or f.shape[1:] != (3, 3) or f.shape[0] < 1:
        raise InvalidMeshConfigError("facets must have shape (F,3,3), F>=1.")
    if not np.all(np.isfinite(f)):
        raise InvalidMeshConfigError("facets contain non-finite values.")
    if normal_orientation not in ("outward", "inward"):
        raise InvalidMeshConfigError(f"Unsupported normal orientation: {normal_orientation!r}")
    f = _consistent_winding(np.ascontiguousarray(f, dtype=np.float64).copy())

    # Edge topology is the closure test. A True hint only retries a looser
    # weld so a noisy watertight body still flips; it must not mark an open
    # wing skin as closed.
    closed = _closed_from_edges(f)
    if not closed and is_closed_hint:
        closed = _closed_from_edges(f, tol=1e-7)
    if require_closed and not closed:
        raise OpenSurfaceError("Closed 3D surface required, but the triangle skin is open.")

    e1, cross, twice_area, area = _triangle_geometry(f)
    if np.any(~np.isfinite(area)) or np.any(area <= float(min_panel_area)):
        raise DegeneratePanelError("One or more 3D triangles have insufficient area.")

    # Flip vertices, not just the normal vector, so n̂ = (v1 × v2) / ||v1 × v2||
    # stays true of the arrays Module 3 will read.
    if closed:
        signed_volume6 = float(np.sum(np.einsum("ij,ij->i", f[:, 0], np.cross(f[:, 1], f[:, 2]))))
        if abs(signed_volume6) <= np.finfo(np.float64).eps * max(1.0, float(np.sum(area))):
            raise InvertedNormalError("Closed surface has indeterminate signed volume.")
        is_outward = signed_volume6 > 0.0
        if is_outward != (normal_orientation == "outward"):
            f = np.ascontiguousarray(f[:, [0, 2, 1], :])
            e1, cross, twice_area, area = _triangle_geometry(f)
    elif normal_orientation == "inward":
        # Open skins have no volume. Supplied winding is treated as outward;
        # "inward" reverses that winding.
        f = np.ascontiguousarray(f[:, [0, 2, 1], :])
        e1, cross, twice_area, area = _triangle_geometry(f)

    normals = cross / twice_area[:, None]
    centroids = np.mean(f, axis=1)
    edge_len = np.linalg.norm(e1, axis=1)
    if np.any(edge_len <= np.finfo(np.float64).eps):
        raise DegeneratePanelError("Zero-length triangle edge encountered.")
    tangent_1 = e1 / edge_len[:, None]
    tangent_2 = np.cross(normals, tangent_1)
    t2len = np.linalg.norm(tangent_2, axis=1)
    if np.any(t2len <= np.finfo(np.float64).eps):
        raise MeshSingularityError("Unable to construct non-singular tangent basis.")
    tangent_2 /= t2len[:, None]

    transforms = np.stack((tangent_1, tangent_2, normals), axis=1)
    inverse_transforms = np.transpose(transforms, (0, 2, 1))

    gram = np.einsum("fij,fkj->fik", transforms, transforms)
    if np.max(np.abs(gram - np.eye(3))) > 5e-12:
        raise MeshSingularityError("3D local panel basis is not orthonormal.")

    return {
        "facets": f,
        "centroids": np.ascontiguousarray(centroids),
        "normals": np.ascontiguousarray(normals),
        "tangent_1": np.ascontiguousarray(tangent_1),
        "tangent_2": np.ascontiguousarray(tangent_2),
        "areas": np.ascontiguousarray(area),
        "transforms": np.ascontiguousarray(transforms),
        "inverse_transforms": np.ascontiguousarray(inverse_transforms),
        "is_closed": closed,
        "normal_orientation": normal_orientation,
    }
