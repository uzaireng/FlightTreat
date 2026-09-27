from __future__ import annotations

import numpy as np

from .math_skeleton import (
    ClusteringScheme,
    DegeneratePanelError,
    InvalidMeshConfigError,
    PanelMetrics2D,
    IBLStationMesh1D,
)


_EPS = np.finfo(np.float64).eps


def _validate_contour(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 4:
        raise InvalidMeshConfigError("points_2d must have shape (N,2), N>=4.")
    if not np.all(np.isfinite(p)):
        raise InvalidMeshConfigError("points_2d contains non-finite values.")
    q = np.ascontiguousarray(p, dtype=np.float64).copy()
    if np.linalg.norm(q[0] - q[-1]) > 1e-10 * max(1.0, np.ptp(q[:, 0]), np.ptp(q[:, 1])):
        raise InvalidMeshConfigError("2D contour must be closed: first point must equal last point.")
    return q


def _distribution(
    n_nodes: int,
    scheme: str,
    *,
    le_at_end: bool,
    le_factor: float,
    te_factor: float,
) -> np.ndarray:
    """Monotone map u→q on [0, 1] with LE/TE clustering."""
    u = np.linspace(0.0, 1.0, int(n_nodes), dtype=np.float64)
    if scheme == "uniform":
        q = u.copy()
    elif scheme == "half_cosine":
        # Pack only at the leading edge. Orient the zero-derivative end to LE.
        if le_at_end:
            q = np.sin(0.5 * np.pi * u)
        else:
            q = 1.0 - np.cos(0.5 * np.pi * u)
    else:
        q = 0.5 * (1.0 - np.cos(np.pi * u))

    # Extra weights use a cubic that is flat only at the requested end.
    # Derivative 1 at the other end keeps that end's spacing intact.
    start_factor = float(te_factor if le_at_end else le_factor)
    end_factor = float(le_factor if le_at_end else te_factor)
    q = _pack_start(q, start_factor)
    q = _pack_end(q, end_factor)
    q[0] = 0.0
    q[-1] = 1.0
    return q


def _pack_end(q: np.ndarray, factor: float) -> np.ndarray:
    """Pack samples toward q=1. factor 0 is the identity."""
    if factor <= 0.0:
        return q
    slope = 1.0 / (1.0 + float(factor))
    return q + (1.0 - slope) * q**2 + (slope - 1.0) * q**3


def _pack_start(q: np.ndarray, factor: float) -> np.ndarray:
    """Pack samples toward q=0. factor 0 is the identity."""
    if factor <= 0.0:
        return q
    return 1.0 - _pack_end(1.0 - q, factor)


def _branch_resample(
    branch: np.ndarray,
    n_nodes: int,
    scheme: str,
    *,
    le_at_end: bool,
    le_factor: float,
    te_factor: float,
) -> np.ndarray:
    # Arc-length interpolation. No assumption that y is monotone.
    seg = np.linalg.norm(np.diff(branch, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if total <= 0.0:
        raise DegeneratePanelError("Contour branch has zero arc length.")

    q = _distribution(
        n_nodes,
        scheme,
        le_at_end=le_at_end,
        le_factor=le_factor,
        te_factor=te_factor,
    )
    idx = np.searchsorted(s, q * total, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    ds = s[idx + 1] - s[idx]
    alpha = np.divide(q * total - s[idx], ds, out=np.zeros_like(q), where=ds > 0.0)
    return branch[idx] + alpha[:, None] * (branch[idx + 1] - branch[idx])


def cosine_cluster_airfoil_nodes(
    points_2d: np.ndarray,
    n_panels: int,
    *,
    scheme: ClusteringScheme = "cosine",
    le_cluster_factor: float = 1.0,
    te_cluster_factor: float = 1.0,
) -> np.ndarray:
    """Return (n_panels+1,2) closed nodes, TE -> LE -> TE."""
    p = _validate_contour(points_2d)
    if int(n_panels) < 4:
        raise InvalidMeshConfigError("n_panels must be >= 4.")
    if scheme not in ("cosine", "half_cosine", "uniform"):
        raise InvalidMeshConfigError(f"Unknown clustering scheme: {scheme!r}.")
    if (
        not np.isfinite(le_cluster_factor)
        or not np.isfinite(te_cluster_factor)
        or le_cluster_factor < 0.0
        or te_cluster_factor < 0.0
    ):
        raise InvalidMeshConfigError("Cluster factors must be finite and >= 0.")

    # Remove duplicate closing point for branch analysis.
    q = p[:-1]
    xmin = np.min(q[:, 0])
    xmax = np.max(q[:, 0])
    le_candidates = np.flatnonzero(np.isclose(q[:, 0], xmin, rtol=0.0, atol=1e-12 * max(1.0, abs(xmax - xmin))))
    te_candidates = np.flatnonzero(np.isclose(q[:, 0], xmax, rtol=0.0, atol=1e-12 * max(1.0, abs(xmax - xmin))))
    if le_candidates.size == 0 or te_candidates.size == 0:
        raise InvalidMeshConfigError("Could not identify LE/TE from x extrema.")
    le0 = int(le_candidates[np.argmin(q[le_candidates, 1] ** 2)])
    te0 = int(te_candidates[np.argmin(q[te_candidates, 1] ** 2)])

    # Roll the contour to TE, then choose the direction whose first half
    # reaches LE. Both cyclic branches are retained.
    rolled = np.roll(q, -te0, axis=0)
    le = int(np.argmin(np.linalg.norm(rolled - q[le0], axis=1)))
    if le < 2 or le > len(rolled) - 2:
        raise InvalidMeshConfigError("Contour must contain distinct TE and LE regions.")

    branch_a = np.vstack((rolled[: le + 1],))
    branch_b = np.vstack((rolled[le:], rolled[:1]))
    # branch_a is TE -> LE. branch_b is LE -> TE.
    # Both factors apply on every branch, at the end that actually is LE or TE.
    p_per_branch = n_panels // 2
    p_other = n_panels - p_per_branch
    na = _branch_resample(
        branch_a,
        p_per_branch + 1,
        scheme,
        le_at_end=True,
        le_factor=float(le_cluster_factor),
        te_factor=float(te_cluster_factor),
    )
    nb = _branch_resample(
        branch_b,
        p_other + 1,
        scheme,
        le_at_end=False,
        le_factor=float(le_cluster_factor),
        te_factor=float(te_cluster_factor),
    )
    # nb starts at LE and ends at TE; omit duplicated LE when joining.
    nodes = np.vstack((na, nb[1:]))
    # Exact closure and endpoint preservation.
    nodes[-1] = nodes[0]
    if nodes.shape[0] != n_panels + 1:
        raise InvalidMeshConfigError("Internal clustering produced incorrect panel count.")
    return np.ascontiguousarray(nodes, dtype=np.float64)


def build_panel_metrics_2d(
    nodes: np.ndarray,
    *,
    min_panel_length: float = 1e-12,
    clustering: ClusteringScheme = "cosine",
) -> PanelMetrics2D:
    n = np.asarray(nodes, dtype=np.float64)
    if n.ndim != 2 or n.shape[1] != 2 or n.shape[0] < 5:
        raise InvalidMeshConfigError("nodes must have shape (P+1,2), P>=4.")
    if clustering not in ("cosine", "half_cosine", "uniform"):
        raise InvalidMeshConfigError(f"Unknown clustering scheme: {clustering!r}.")
    n = np.ascontiguousarray(n, dtype=np.float64).copy()
    if np.linalg.norm(n[0] - n[-1]) > 1e-10 * max(1.0, np.ptp(n[:, 0]), np.ptp(n[:, 1])):
        raise InvalidMeshConfigError("Panel nodes must be closed.")
    d = np.diff(n, axis=0)
    length = np.linalg.norm(d, axis=1)
    if np.any(~np.isfinite(length)) or np.any(length <= min_panel_length):
        raise DegeneratePanelError("2D panel length is zero or below min_panel_length.")

    tangent = d / length[:, None]
    collocation = 0.5 * (n[:-1] + n[1:])

    # Determine contour orientation from signed polygon area.
    signed_area2 = float(np.sum(n[:-1, 0] * n[1:, 1] - n[1:, 0] * n[:-1, 1]))
    if abs(signed_area2) <= _EPS:
        raise InvalidMeshConfigError("2D contour has near-zero signed area.")
    if signed_area2 > 0.0:  # CCW -> right normal is outward
        normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    else:  # CW -> left normal is outward
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    beta = np.arctan2(tangent[:, 1], tangent[:, 0])

    # Vectorized discrete curvature using turning angle between adjacent panels.
    t_prev = tangent[:-1]
    t_next = tangent[1:]
    cross_z = t_prev[:, 0] * t_next[:, 1] - t_prev[:, 1] * t_next[:, 0]
    dot_t = np.clip(np.sum(t_prev * t_next, axis=1), -1.0, 1.0)
    turn = np.arctan2(cross_z, dot_t)
    ds_mid = 0.5 * (length[:-1] + length[1:])
    k_mid = np.divide(turn, ds_mid, out=np.zeros_like(turn), where=ds_mid > 0.0)
    curvature = np.empty_like(length)
    curvature[1:-1] = 0.5 * (k_mid[:-1] + k_mid[1:])
    curvature[0] = k_mid[0]
    curvature[-1] = k_mid[-1]

    x_min, x_max = float(np.min(n[:, 0])), float(np.max(n[:, 0]))
    chord = x_max - x_min
    if chord <= 0.0:
        raise InvalidMeshConfigError("2D contour chord must be positive.")
    x_over_c = (collocation[:, 0] - x_min) / chord

    arc_nodes = np.concatenate(([0.0], np.cumsum(length)))
    arc_length = 0.5 * (arc_nodes[:-1] + arc_nodes[1:])

    le_index = int(np.argmin(n[:, 0]))
    te_index = int(np.argmax(n[:, 0]))
    return {
        "nodes": n,
        "collocation": np.ascontiguousarray(collocation),
        "tangent": np.ascontiguousarray(tangent),
        "normal": np.ascontiguousarray(normal),
        "length": np.ascontiguousarray(length),
        "angle_beta": np.ascontiguousarray(beta),
        "curvature": np.ascontiguousarray(curvature),
        "x_over_c": np.ascontiguousarray(x_over_c),
        "arc_length": np.ascontiguousarray(arc_length),
        "chord": float(chord),
        "le_index": le_index,
        "te_index": te_index,
        "is_closed": True,
        "clustering": clustering,
        "te_first": bool(te_index == 0),
    }


def _closed_arc(unique: np.ndarray, start: int, end: int, step: int) -> np.ndarray:
    """Walk the closed unique-node ring from start to end, inclusive."""
    n = int(unique.shape[0])
    if step > 0:
        if start <= end:
            idx = np.arange(start, end + 1)
        else:
            idx = np.concatenate((np.arange(start, n), np.arange(0, end + 1)))
    elif start >= end:
        idx = np.arange(start, end - 1, -1)
    else:
        idx = np.concatenate((np.arange(start, -1, -1), np.arange(n - 1, end - 1, -1)))
    return np.ascontiguousarray(unique[idx], dtype=np.float64)


def build_ibl_stations_1d(panel: PanelMetrics2D) -> IBLStationMesh1D:
    nodes = np.asarray(panel["nodes"], dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[0] < 5 or nodes.shape[1] != 2:
        raise InvalidMeshConfigError("IBL stations require a closed (P+1, 2) node array.")
    if np.linalg.norm(nodes[0] - nodes[-1]) > 1e-8 * max(1.0, float(np.ptp(nodes))):
        raise InvalidMeshConfigError("IBL stations require a closed contour.")

    unique = nodes[:-1]
    n_unique = int(unique.shape[0])
    le = int(panel["le_index"])
    te = int(panel["te_index"])
    if le == nodes.shape[0] - 1:
        le = 0
    if te == nodes.shape[0] - 1:
        te = 0
    if not (0 <= le < n_unique and 0 <= te < n_unique) or le == te:
        raise InvalidMeshConfigError("LE and TE must be distinct nodes on the closed contour.")

    forward = _closed_arc(unique, le, te, 1)
    backward = _closed_arc(unique, le, te, -1)
    if forward.shape[0] < 2 or backward.shape[0] < 2:
        raise InvalidMeshConfigError("IBL split produced an empty surface branch.")

    # Geometric upper/lower. Contour winding must not swap the two sides.
    forward_y = float(np.mean(forward[:, 1]))
    backward_y = float(np.mean(backward[:, 1]))
    if forward_y > backward_y or (
        np.isclose(forward_y, backward_y) and float(np.max(forward[:, 1])) >= float(np.max(backward[:, 1]))
    ):
        upper_xy = forward
        lower_xy = backward
    else:
        upper_xy = backward
        lower_xy = forward

    upper_d = np.diff(upper_xy, axis=0)
    lower_d = np.diff(lower_xy, axis=0)
    upper_ds = np.linalg.norm(upper_d, axis=1)
    lower_ds = np.linalg.norm(lower_d, axis=1)
    if np.any(upper_ds <= 0.0) or np.any(lower_ds <= 0.0):
        raise DegeneratePanelError("IBL branch contains a zero-length segment.")

    # Node tangent = normalized average of neighboring branch directions.
    def node_tangent(xy: np.ndarray) -> np.ndarray:
        d = np.diff(xy, axis=0)
        seg = np.linalg.norm(d, axis=1)
        out = np.empty_like(xy)
        out[0] = d[0] / seg[0]
        out[-1] = d[-1] / seg[-1]
        if xy.shape[0] > 2:
            mid = d[:-1] / seg[:-1, None] + d[1:] / seg[1:, None]
            norm = np.linalg.norm(mid, axis=1)
            out[1:-1] = mid / np.maximum(norm[:, None], np.finfo(np.float64).tiny)
        return out

    ut = node_tangent(upper_xy)
    lt = node_tangent(lower_xy)
    return {
        "s_upper": np.concatenate(([0.0], np.cumsum(upper_ds))),
        "s_lower": np.concatenate(([0.0], np.cumsum(lower_ds))),
        "xy_upper": upper_xy,
        "xy_lower": lower_xy,
        "tangent_upper": ut,
        "tangent_lower": lt,
        "ds_upper": upper_ds,
        "ds_lower": lower_ds,
        "stagnation_xy": nodes[le].copy(),
    }
