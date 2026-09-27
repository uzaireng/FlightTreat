from __future__ import annotations

import numpy as np

from .math_skeleton import BoundaryRegion, CartesianGrid2D, DomainExtents2D, InvalidMeshConfigError


def _blocked_point_count(n_points: int, n_edges: int, budget: int) -> int:
    """Largest point block whose (block × edges) kernel stays inside ``budget``."""

    if n_points <= 0:
        return 0
    if n_edges <= 0:
        return n_points
    return max(512, min(n_points, budget // n_edges))


def _points_inside_polygon(
    px: np.ndarray,
    py: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Even-odd ray cast. Each iteration is one NumPy kernel over a point block."""

    a = polygon[:-1]
    b = polygon[1:]
    ax = np.ascontiguousarray(a[:, 0])
    ay = np.ascontiguousarray(a[:, 1])
    bx = np.ascontiguousarray(b[:, 0])
    by = np.ascontiguousarray(b[:, 1])
    edge_dx = bx - ax
    safe_dy = by - ay
    safe_dy = np.where(np.abs(safe_dy) > 0.0, safe_dy, 1.0)

    n_points = int(px.shape[0])
    inside = np.empty(n_points, dtype=bool)
    chunk = _blocked_point_count(n_points, int(ax.shape[0]), 2_000_000)
    if chunk == 0:
        return inside
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        xp = px[start:stop, None]
        yp = py[start:stop, None]
        crossing = (ay > yp) != (by > yp)
        x_hit = ax + (yp - ay) * edge_dx / safe_dy
        hit = crossing & (xp < x_hit)
        inside[start:stop] = np.bitwise_and(np.sum(hit, axis=1), 1).astype(bool)
    return inside


def _points_near_polyline(
    px: np.ndarray,
    py: np.ndarray,
    polygon: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """True where a point lies within ``tolerance`` of a closed polyline."""

    a = polygon[:-1]
    b = polygon[1:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    ax = np.ascontiguousarray(a[:, 0])
    ay = np.ascontiguousarray(a[:, 1])
    abx = np.ascontiguousarray(ab[:, 0])
    aby = np.ascontiguousarray(ab[:, 1])
    tol2 = float(tolerance) * float(tolerance)

    n_points = int(px.shape[0])
    near = np.empty(n_points, dtype=bool)
    chunk = _blocked_point_count(n_points, int(ax.shape[0]), 1_500_000)
    if chunk == 0:
        return near
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        qx = px[start:stop, None]
        qy = py[start:stop, None]
        u = (qx - ax) * abx + (qy - ay) * aby
        u = np.clip(u / np.maximum(denom, np.finfo(np.float64).tiny), 0.0, 1.0)
        cx = ax + u * abx
        cy = ay + u * aby
        d2 = (qx - cx) ** 2 + (qy - cy) ** 2
        near[start:stop] = np.min(d2, axis=1) <= tol2
    return near


def build_cartesian_grid_2d(
    points_2d: np.ndarray,
    *,
    n_x: int,
    n_y: int,
    domain: DomainExtents2D,
) -> CartesianGrid2D:
    p = np.asarray(points_2d, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 4:
        raise InvalidMeshConfigError("points_2d must have shape (N,2).")
    if not np.all(np.isfinite(p)):
        raise InvalidMeshConfigError("points_2d contains non-finite values.")
    counts_are_ints = isinstance(n_x, (int, np.integer)) and isinstance(n_y, (int, np.integer))
    if isinstance(n_x, bool) or isinstance(n_y, bool) or not counts_are_ints:
        raise InvalidMeshConfigError("n_x and n_y must be integers.")
    n_x = int(n_x)
    n_y = int(n_y)
    close_tol = 1e-10 * max(1.0, float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])))
    if float(np.linalg.norm(p[0] - p[-1])) > close_tol:
        p = np.vstack((p, p[:1]))
    xmin, xmax = float(np.min(p[:, 0])), float(np.max(p[:, 0]))
    ymin, ymax = float(np.min(p[:, 1])), float(np.max(p[:, 1]))
    if not (domain["x_min"] < xmin and domain["x_max"] > xmax and
            domain["y_min"] < ymin and domain["y_max"] > ymax):
        raise InvalidMeshConfigError("Domain must be strictly larger than geometry AABB.")
    if n_x < 8 or n_y < 8:
        raise InvalidMeshConfigError("Cartesian grid requires n_x,n_y >= 8.")

    x_nodes = np.linspace(domain["x_min"], domain["x_max"], int(n_x), dtype=np.float64)
    y_nodes = np.linspace(domain["y_min"], domain["y_max"], int(n_y), dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_nodes, y_nodes, indexing="xy")
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()

    # Only the geometry AABB can contain the solid. The farfield stays a no-op.
    pad = 1e-9 * max(1.0, xmax - xmin, ymax - ymin)
    in_box = (
        (flat_x >= xmin - pad)
        & (flat_x <= xmax + pad)
        & (flat_y >= ymin - pad)
        & (flat_y <= ymax + pad)
    )
    inside_flat = np.zeros(flat_x.shape[0], dtype=bool)
    box_idx = np.flatnonzero(in_box)
    if box_idx.size:
        inside_flat[box_idx] = _points_inside_polygon(
            flat_x[box_idx], flat_y[box_idx], p
        )
    inside = inside_flat.reshape(grid_x.shape)

    mask = np.full((n_y, n_x), int(BoundaryRegion.INTERIOR_FLUID), dtype=np.int32)
    mask[:, 0] = int(BoundaryRegion.INLET)
    mask[:, -1] = int(BoundaryRegion.OUTLET)
    mask[0, :] = int(BoundaryRegion.BOTTOM_FARFIELD)
    mask[-1, :] = int(BoundaryRegion.TOP_FARFIELD)

    # Solid takes precedence over external boundary tags, except a pathological
    # geometry touching the domain is already rejected above.
    mask[inside] = int(BoundaryRegion.INTERNAL_SOLID)

    # Mark grid points close to the contour as wall; this is visualization tagging,
    # not a volume mesh boundary reconstruction.
    dx = float(x_nodes[1] - x_nodes[0])
    dy = float(y_nodes[1] - y_nodes[0])
    tol = 0.75 * np.hypot(dx, dy)
    near_box = (
        (flat_x >= xmin - tol)
        & (flat_x <= xmax + tol)
        & (flat_y >= ymin - tol)
        & (flat_y <= ymax + tol)
        & ~inside_flat
    )
    wall_flat = np.zeros(flat_x.shape[0], dtype=bool)
    near_idx = np.flatnonzero(near_box)
    if near_idx.size:
        wall_flat[near_idx] = _points_near_polyline(
            flat_x[near_idx], flat_y[near_idx], p, tol
        )
    mask[wall_flat.reshape(grid_x.shape)] = int(BoundaryRegion.AIRFOIL_WALL)

    return {
        "x_nodes": np.ascontiguousarray(x_nodes),
        "y_nodes": np.ascontiguousarray(y_nodes),
        "grid_x": np.ascontiguousarray(grid_x),
        "grid_y": np.ascontiguousarray(grid_y),
        "boundary_mask": np.ascontiguousarray(mask, dtype=np.int32),
        "n_x": int(n_x),
        "n_y": int(n_y),
    }
