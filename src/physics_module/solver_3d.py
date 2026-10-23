"""
3D constant-strength source/doublet panel solver.

Closed bodies use a Morino Dirichlet condition (interior potential zero) with
``σ = n · V∞``. Open wing skins add a wake sheet whose doublet is the
trailing-edge jump, so the wake is not an extra unknown. Influence products
are batched over collocation rows.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import gmres

from math_module.math_skeleton import PanelMetrics3D

from .physics_skeleton import InviscidSolution, InvalidFlowConditionError, SingularInfluenceMatrixError

_FOUR_PI = 4.0 * np.pi
_COND_LIMIT = 1.0e12


def _solid_angle(targets: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Signed solid angle of every triangle at every target. Shape ``(B, F)``."""

    r1 = tris[None, :, 0, :] - targets[:, None, :]
    r2 = tris[None, :, 1, :] - targets[:, None, :]
    r3 = tris[None, :, 2, :] - targets[:, None, :]
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    n3 = np.linalg.norm(r3, axis=-1)
    triple = np.einsum("bfi,bfi->bf", r1, np.cross(r2, r3))
    denom = (
        n1 * n2 * n3
        + n3 * np.einsum("bfi,bfi->bf", r1, r2)
        + n2 * np.einsum("bfi,bfi->bf", r1, r3)
        + n1 * np.einsum("bfi,bfi->bf", r2, r3)
    )
    return 2.0 * np.arctan2(triple, denom)


def _panel_influence(targets: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solid angle and source potential from one pass over the same edges.

    The formulas match ``_solid_angle`` and the previous source integral.
    Building them together avoids a second all-pairs solid angle.
    """

    r1 = tris[None, :, 0, :] - targets[:, None, :]
    r2 = tris[None, :, 1, :] - targets[:, None, :]
    r3 = tris[None, :, 2, :] - targets[:, None, :]
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    n3 = np.linalg.norm(r3, axis=-1)
    triple = np.einsum("bfi,bfi->bf", r1, np.cross(r2, r3))
    denom = (
        n1 * n2 * n3
        + n3 * np.einsum("bfi,bfi->bf", r1, r2)
        + n2 * np.einsum("bfi,bfi->bf", r1, r3)
        + n1 * np.einsum("bfi,bfi->bf", r2, r3)
    )
    omega = 2.0 * np.arctan2(triple, denom)
    vertex_a = tris[:, 0, :]
    vertex_b = tris[:, 1, :]
    vertex_c = tris[:, 2, :]
    normal = np.cross(vertex_b - vertex_a, vertex_c - vertex_a)
    nhat = normal / np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
    height = np.einsum("bfi,fi->bf", r1, nhat)

    def edge_term(start: np.ndarray, stop: np.ndarray) -> np.ndarray:
        edge = stop - start
        length = np.linalg.norm(edge, axis=-1)
        direction = edge / np.maximum(length[..., None], 1e-30)
        outward = np.cross(nhat[None, :, :], direction)
        distance = np.einsum("bfi,bfi->bf", start, outward)
        radius_start = np.linalg.norm(start, axis=-1)
        radius_stop = np.linalg.norm(stop, axis=-1)
        numer = radius_start + radius_stop + length
        denom_edge = np.maximum(radius_start + radius_stop - length, 1e-30)
        return distance * np.log(numer / denom_edge)

    edges = edge_term(r1, r2) + edge_term(r2, r3) + edge_term(r3, r1)
    # ∫ dS/r = -Σ d_i ln((R+R+L)/(R+R-L)) - h Ω.
    integral = -edges - height * omega
    return omega, -integral / _FOUR_PI


def _source_potential(targets: np.ndarray, tris: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Unit-source potential ``-1/(4π) ∫ dS/r`` for every triangle.

    The integral is the exact edge formula. ``areas`` selects the disk self
    term applied by the caller; the in-plane self value is singular here.
    Shape ``(B, F)``.
    """

    del areas
    _omega, source = _panel_influence(targets, tris)
    return source


def _source_self(areas: np.ndarray) -> np.ndarray:
    """In-plane self potential of a unit source, disk-equivalent of the panel area.

    A 7-point rule evaluated a panel-scale distance off the centroid is singular.
    The center potential of a disk with the same area is ``-R/2``.
    """

    return -0.5 * np.sqrt(np.maximum(areas, 0.0) / np.pi)


def _replace_source_diagonal(potential: np.ndarray, row_start: int, areas: np.ndarray) -> None:
    rows = np.arange(potential.shape[0])
    potential[rows, row_start + rows] = _source_self(areas[row_start : row_start + rows.shape[0]])


def _batched(targets: np.ndarray, tris: np.ndarray, areas: np.ndarray, batch: int):
    """Yield ``(start, solid_angle, source_potential)`` row blocks."""

    for start in range(0, targets.shape[0], batch):
        block = targets[start : start + batch]
        yield start, *_panel_influence(block, tris)


def _row_batch(n_panels: int) -> int:
    # Three vertex vectors of shape (B, F, 3) should stay near 128 MB.
    # Larger blocks cut Python launch cost. The integrals are unchanged.
    bytes_per_row = max(n_panels, 1) * 3 * 8 * 6
    return max(8, min(512, int(128 * 1024 * 1024 / bytes_per_row)))


def _trailing_panels(panel: PanelMetrics3D) -> tuple[np.ndarray, np.ndarray]:
    """Aft surface panels on the upper and lower side of an open skin.

    Returns two index arrays of equal length (possibly empty).
    """

    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    normals = np.asarray(panel["normals"], dtype=np.float64)
    if panel["is_closed"] or centroids.shape[0] < 4:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    y = centroids[:, 1]
    span = float(np.ptp(y))
    dy = max(span / 40.0, 1e-6)
    ybin = np.round((y - float(np.min(y))) / dy).astype(np.int64)
    upper = normals[:, 2] >= 0.0
    lower = ~upper
    upper_idx = []
    lower_idx = []
    for bucket in np.unique(ybin):
        members = np.flatnonzero(ybin == bucket)
        up = members[upper[members]]
        lo = members[lower[members]]
        if up.size == 0 or lo.size == 0:
            continue
        upper_idx.append(int(up[np.argmax(centroids[up, 0])]))
        lower_idx.append(int(lo[np.argmax(centroids[lo, 0])]))
    if not upper_idx:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.asarray(upper_idx, dtype=np.int64), np.asarray(lower_idx, dtype=np.int64)


def _wake_triangles(
    panel: PanelMetrics3D,
    upper_idx: np.ndarray,
    lower_idx: np.ndarray,
    freestream: np.ndarray,
    wake_length: float,
) -> np.ndarray:
    """Wake triangles shed from each upper/lower trailing pair. Shape ``(W, 3, 3)``."""

    if upper_idx.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    direction = freestream / max(float(np.linalg.norm(freestream)), 1e-12)
    origin = centroids[upper_idx]
    end = origin + wake_length * direction
    # A zero-thickness wake uses the upper trailing centroid and a spanwise neighbour.
    if upper_idx.size == 1:
        span = np.array([0.0, wake_length * 0.05, 0.0], dtype=np.float64)
        a = origin
        b = origin + span
    else:
        order = np.argsort(centroids[upper_idx, 1])
        ordered = upper_idx[order]
        a = centroids[ordered[:-1]]
        b = centroids[ordered[1:]]
        end = a + wake_length * direction
        end_b = b + wake_length * direction
        # Winding puts the sheet normal toward +z when the spanwise edge
        # runs toward +y. The body self-term is +1/2 on the interior, so
        # the wake must use that same side for μ_upper - μ_lower to add
        # circulation instead of cancelling it. Triangles are stored as
        # every strip's first triangle, then every strip's second.
        tri_1 = np.stack((a, end_b, b), axis=1)
        tri_2 = np.stack((a, end, end_b), axis=1)
        return np.ascontiguousarray(np.concatenate((tri_1, tri_2), axis=0), dtype=np.float64)
    end_b = b + wake_length * direction
    tri = np.stack((a, end_b, b), axis=1)
    return np.ascontiguousarray(tri, dtype=np.float64)


def _panel_potential(
    targets: np.ndarray,
    tris: np.ndarray,
    areas: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    drop_source_self: bool = False,
) -> np.ndarray:
    """Perturbation potential of the source/doublet surface at off-body points.

    ``drop_source_self`` zeros the panel's own source column. That term is
    nearly singular a short distance off the centroid and its tangential
    difference swamps the freestream gradient. The self doublet is kept: at
    constant height its tangential derivative is the sheet jump.
    """

    out = np.zeros(targets.shape[0], dtype=np.float64)
    batch = _row_batch(int(areas.shape[0]))
    aligned = drop_source_self and targets.shape[0] == areas.shape[0]
    for start, omega, source in _batched(targets, tris, areas, batch):
        stop = start + omega.shape[0]
        if aligned:
            rows = np.arange(stop - start)
            source[rows, start + rows] = 0.0
        out[start:stop] = omega @ mu / _FOUR_PI + source @ sigma
    return out


def _sampled_surface_velocity(
    panel: PanelMetrics3D,
    mu: np.ndarray,
    sigma: np.ndarray,
    freestream: np.ndarray,
) -> np.ndarray:
    """Tangential velocity from the exterior potential, for a closed body.

    Samples sit just outside the surface. Differencing μ between centroids
    mixes the cap and the side wall on a coarse triangulation and reports a
    stagnation panel as a high-speed one.
    """

    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    normals = np.asarray(panel["normals"], dtype=np.float64)
    t1 = np.asarray(panel["tangent_1"], dtype=np.float64)
    t2 = np.asarray(panel["tangent_2"], dtype=np.float64)
    areas = np.asarray(panel["areas"], dtype=np.float64)
    tris = np.asarray(panel["facets"], dtype=np.float64)
    scale = np.sqrt(np.maximum(areas, 1e-16))
    height = np.maximum(1e-4, 0.08 * scale)
    step = np.maximum(1e-4, 0.25 * scale)
    base = centroids + height[:, None] * normals
    phi_0 = _panel_potential(base, tris, areas, mu, sigma, drop_source_self=True)
    phi_1 = _panel_potential(base + step[:, None] * t1, tris, areas, mu, sigma, drop_source_self=True)
    phi_2 = _panel_potential(base + step[:, None] * t2, tris, areas, mu, sigma, drop_source_self=True)
    g1 = (phi_1 - phi_0) / step
    g2 = (phi_2 - phi_0) / step
    velocity = freestream[None, :] + g1[:, None] * t1 + g2[:, None] * t2
    velocity = velocity - normals * np.sum(velocity * normals, axis=1, keepdims=True)
    return velocity


def _doublet_gradient_velocity(
    panel: PanelMetrics3D,
    mu: np.ndarray,
    freestream: np.ndarray,
) -> np.ndarray:
    """Exterior tangential speed on an open wing from the doublet sheet.

    The interior condition absorbs the onset, so on a non-lifting surface
    μ tracks ``-V∞·r``. The tangential perturbation is the chordwise
    derivative of ``μ + V∞·r``. An off-body sample of one constant-strength
    face never sees the trailing-edge jump, and a neighbor difference
    across a cosine-clustered edge divides a smooth doublet by a vanishing
    step. Each spanwise strip is therefore fit as one polynomial in x.
    """

    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    normals = np.asarray(panel["normals"], dtype=np.float64)
    count = int(mu.shape[0])
    perturbation = mu + centroids @ freestream
    slope = np.zeros(count, dtype=np.float64)
    span = float(np.ptp(centroids[:, 1]))
    dy = max(span / 40.0, 1e-6)
    ybin = np.round((centroids[:, 1] - float(np.min(centroids[:, 1]))) / dy).astype(np.int64)
    side = (normals[:, 2] >= 0.0).astype(np.int64)
    # One fit per strip, not a walk over panels. The strips are independent.
    for bucket in np.unique(ybin):
        for which in (0, 1):
            idx = np.flatnonzero((ybin == bucket) & (side == which))
            if idx.size < 6:
                continue
            order = np.argsort(centroids[idx, 0], kind="mergesort")
            idx = idx[order]
            chord = centroids[idx, 0]
            chord_span = float(chord[-1] - chord[0])
            if chord_span < 1e-8:
                continue
            coordinate = (chord - float(chord[0])) / chord_span
            degree = min(4, int(idx.size) - 1)
            basis = np.vander(coordinate, N=degree + 1, increasing=False)
            coeff, *_rest = np.linalg.lstsq(basis, perturbation[idx], rcond=1e-12)
            slope[idx] = np.polyval(np.polyder(coeff), coordinate) / chord_span
    velocity = np.broadcast_to(freestream, (count, 3)).copy()
    velocity[:, 0] += slope
    velocity -= normals * np.sum(velocity * normals, axis=1, keepdims=True)
    return velocity


def _surface_velocity(
    panel: PanelMetrics3D,
    mu: np.ndarray,
    sigma: np.ndarray,
    freestream: np.ndarray,
) -> np.ndarray:
    """Tangential velocity used for the surface pressure."""

    if panel["is_closed"]:
        return _sampled_surface_velocity(panel, mu, sigma, freestream)
    return _doublet_gradient_velocity(panel, mu, freestream)


def solve_3d(
    panel: PanelMetrics3D,
    alpha_rad: float,
    *,
    kutta: bool,
    wake_length_chords: float,
    dense_panel_limit: int,
    model: str,
) -> tuple[InviscidSolution, np.ndarray]:
    """Solve the source/doublet system and return centroid velocity."""

    if model not in ("constant_source_doublet", "constant_doublet"):
        raise InvalidFlowConditionError(f"Unknown 3D panel model {model!r}.")
    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    normals = np.asarray(panel["normals"], dtype=np.float64)
    areas = np.asarray(panel["areas"], dtype=np.float64)
    tris = np.asarray(panel["facets"], dtype=np.float64)
    count = int(areas.shape[0])
    freestream = np.array([np.cos(alpha_rad), 0.0, np.sin(alpha_rad)], dtype=np.float64)
    if model == "constant_doublet":
        sigma = np.zeros(count, dtype=np.float64)
    else:
        sigma = normals @ freestream

    upper_idx, lower_idx = _trailing_panels(panel) if kutta and not panel["is_closed"] else (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )
    root = centroids[np.abs(centroids[:, 1] - np.median(centroids[:, 1])) <= max(float(np.ptp(centroids[:, 1])) / 20.0, 1e-6)]
    root_chord = float(np.ptp(root[:, 0])) if root.size else float(np.ptp(centroids[:, 0]))
    wake_length = max(float(wake_length_chords), 0.0) * max(root_chord, 1e-6)
    wake = _wake_triangles(panel, upper_idx, lower_idx, freestream, wake_length) if kutta else np.zeros((0, 3, 3))

    batch = _row_batch(count + int(wake.shape[0]))
    # Collocation is offset slightly inside so the self solid angle is finite.
    inside = centroids - 1e-4 * np.sqrt(areas)[:, None] * normals

    source_rhs = np.zeros(count, dtype=np.float64)
    all_sigma = sigma
    matrix = np.zeros((count, count), dtype=np.float64)
    # One pass fills both the source right-hand side and the doublet matrix.
    # Wake carries doublet only, so it is added after this body pass.
    for start, omega, source_phi in _batched(inside, tris, areas, batch):
        stop = start + source_phi.shape[0]
        _replace_source_diagonal(source_phi, start, areas)
        # Interior Dirichlet: φ_μ + φ_σ + V∞·r = 0 at the interior point.
        # Omitting the onset leaves a right-hand side of panel scale, and the
        # trailing-edge jump then collapses as the chordwise mesh is refined.
        source_rhs[start:stop] = -(source_phi @ all_sigma) - inside[start:stop] @ freestream
        matrix[start:stop] = omega / _FOUR_PI
    # Body self solid-angle is 1/2 on the interior. Set it before the wake
    # is added: a later overwrite would delete the wake influence on the
    # trailing-edge rows, which is the near-field Kutta coupling.
    np.fill_diagonal(matrix, 0.5)
    # Wake columns: μ_wake = μ_upper - μ_lower, not an extra unknown.
    if wake.shape[0] and upper_idx.size > 1:
        order = np.argsort(centroids[upper_idx, 1])
        ordered_u = upper_idx[order]
        ordered_l = lower_idx[order]
        station_count = int(ordered_u.size - 1)
        # _wake_triangles concatenates every strip's first triangle,
        # then every strip's second. Both belong to that strip.
        jump_index = np.concatenate(
            (np.arange(station_count, dtype=np.int64), np.arange(station_count, dtype=np.int64))
        )
        if jump_index.shape[0] == wake.shape[0] and station_count >= 1:
            selector = np.zeros((wake.shape[0], station_count), dtype=np.float64)
            selector[np.arange(wake.shape[0]), jump_index] = 1.0
        else:
            selector = None
        if selector is not None:
            for start in range(0, inside.shape[0], batch):
                block_targets = inside[start : start + batch]
                omega = _solid_angle(block_targets, wake)
                block = omega / _FOUR_PI
                summed = block @ selector
                row = slice(start, start + summed.shape[0])
                matrix[row, ordered_u[:-1]] += summed
                matrix[row, ordered_l[:-1]] -= summed

    if count <= int(dense_panel_limit):
        try:
            condition = float(np.linalg.cond(matrix, 1))
        except np.linalg.LinAlgError as exc:
            raise SingularInfluenceMatrixError("3D influence matrix condition estimate failed.") from exc
        if not np.isfinite(condition) or condition > _COND_LIMIT:
            raise SingularInfluenceMatrixError(
                f"3D influence matrix condition number {condition:.3e} exceeds {_COND_LIMIT:.0e}."
            )
        try:
            mu = np.linalg.solve(matrix, source_rhs)
        except np.linalg.LinAlgError as exc:
            raise SingularInfluenceMatrixError("3D influence matrix is singular.") from exc
    else:
        # The matrix is already formed. GMRES only changes the factorization,
        # not the kernels, so each matvec is one product.
        condition = float("nan")
        mu, info = gmres(matrix, source_rhs, rtol=1e-6, atol=0.0, restart=40, maxiter=80)
        if info != 0 or not np.all(np.isfinite(mu)):
            raise SingularInfluenceMatrixError("GMRES failed to converge for the 3D influence system.")
        mu = np.asarray(mu, dtype=np.float64)

    velocity = _surface_velocity(panel, mu, sigma, freestream)
    speed2 = np.sum(velocity * velocity, axis=1)
    cp = 1.0 - speed2
    solution = InviscidSolution(
        gamma=None,
        mu=np.ascontiguousarray(mu, dtype=np.float64),
        sigma=np.ascontiguousarray(sigma, dtype=np.float64),
        cp_inviscid=np.ascontiguousarray(cp, dtype=np.float64),
        v_tangential=np.ascontiguousarray(velocity, dtype=np.float64),
        influence_condition_number=float(condition),
    )
    return solution, np.ascontiguousarray(velocity, dtype=np.float64)
